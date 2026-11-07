"""Presets are base experiments stored as JSON files: discovery, editing, errors, the CLI, and the
description field, which documents an experiment without ever being part of what runs."""

import json
import os
import shutil
from pathlib import Path

import pytest
import torch

import fct.exp
from fct.cli import main
from fct.data import make_dataset
from fct.exp import (
    ExpConfig,
    PresetError,
    exp_hash,
    exp_text,
    expand,
    load_exp,
    packaged_presets,
    preset,
    preset_names,
    run_setting,
    same_experiment,
    save_exp,
    setting_config,
    setting_dir,
    setting_statuses,
    with_axes,
)
from fct.exp_migrate import migrate_exp
from fct.spatial import build_spaces, parameter_factors
from fct.train import (
    compute_loss,
    largest_hessian_eigenvalue,
    loss_coefficients,
    make_net_from_config,
    make_weight_penalty,
    prior_coefficient,
    prior_stiffness,
)
from tests.report_helpers import QUIET

PACKAGED = [
    "full",
    "lambda_exp",
    "linear_ring",
    "linear_ring_equivalence",
    "linear_ring_tiny",
    "modularity",
    "phase_transition",
    "spatial_exp",
    "spatial_n_scan",
    "spatial_tiny",
    "tiny",
]


@pytest.fixture
def folder(tmp_path) -> Path:
    """A copy of the packaged presets folder, to edit freely."""
    copy = tmp_path / "presets"
    shutil.copytree(str(packaged_presets()), copy)
    return copy


@pytest.fixture
def use_folder(monkeypatch):
    """Make `folder` the presets the library and the CLI see by default."""

    def use(directory: Path) -> None:
        monkeypatch.setattr(fct.exp, "packaged_presets", lambda: directory)

    return use


def _cli(capsys, *args) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(["exp", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


def _edit(path: Path, **changes) -> None:
    tree = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**tree, **changes}, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- the packaged files
def test_the_packaged_presets_are_the_eleven_base_experiments():
    assert preset_names() == PACKAGED


@pytest.mark.parametrize("name", PACKAGED)
def test_every_packaged_preset_validates_and_describes_itself(name):
    exp = preset(name)
    assert exp.description


def test_full_and_modularity_are_two_files_of_the_same_experiment():
    assert same_experiment(preset("full"), preset("modularity"))
    assert preset("full").description != preset("modularity").description


def test_lambda_exp_is_four_axes_of_which_two_hold_seven_log_spaced_values():
    exp = preset("lambda_exp")
    assert len(expand(exp)) == 2 * 2 * 7 * 7
    assert [len(v) for v in exp.axes.values()] == [2, 2, 7, 7]


LINEAR_RING = ["linear_ring", "linear_ring_equivalence", "linear_ring_tiny"]
PARAMETRISATION = "spatial.parametrisation"


@pytest.mark.parametrize("name", LINEAR_RING)
def test_the_linear_ring_presets_are_the_linear_theory_at_sigma_zero(name):
    """The dictionary of the experiment plan at sigma = 0: no input noise field, deterministic
    grids, latents on [0, 1], an orthonormal teacher, the spectrum monitor, every path started
    from the prior, B a shared field of at least 16."""
    exp = preset(name)
    assert "noise" not in json.dumps(exp.model_dump())
    assert (exp.task, exp.phi, exp.L, exp.g, exp.dtype) == (
        "linear_teacher",
        "id",
        1,
        1.0,
        "float64",
    )
    assert (exp.optimizer, exp.monitor_kind, exp.init) == ("sgld", "spectrum", "prior")
    assert (exp.weight_lambda, exp.rate_lambda) == (1.0, 0.0)
    teacher = exp.linear_teacher
    assert (teacher.latent_min, teacher.latent_max, teacher.teacher) == (0.0, 1.0, "orthonormal")
    assert exp.B >= 16 and "net.B" not in exp.axes
    assert exp.spatial.box == exp.N  # length scales in sites
    for population in (exp.spatial.input, exp.spatial.hidden, exp.spatial.output):
        assert population.embedding.kind == "grid"
    assert exp.spatial.input.kernel.length_scale == exp.spatial.output.kernel.length_scale == 0


def test_linear_ring_and_its_penalty_twin_differ_only_in_the_path_lr_and_run_length():
    """Two ROOTs for the equivalence of the parametrisations at the base point: the same setting
    (one beta axis, same data, net and loss) on the whitened and the penalty path."""
    whitened, penalty = preset("linear_ring"), preset("linear_ring_equivalence")
    assert whitened.spatial.parametrisation == "whitened"
    assert penalty.spatial.parametrisation == "penalty"
    assert whitened.axes == penalty.axes == {"train.beta": [10.0]}
    path_fields = {"spatial", "lr", "n_epochs", "log_int", "save_int", "monitor_int"}
    shared = {"description", *path_fields}
    assert whitened.model_dump(exclude=shared) == penalty.model_dump(exclude=shared)
    assert whitened.spatial.model_copy(update={"parametrisation": "penalty"}) == penalty.spatial


def test_the_pipeline_preset_pairs_lr_with_the_parametrisation():
    exp = preset("linear_ring_tiny")
    settings = expand(exp)
    assert list(exp.axes) == [PARAMETRISATION, "train.lr"]
    assert settings[0].axes == {PARAMETRISATION: "penalty", "train.lr": exp.axes["train.lr"][0]}
    assert settings[3].axes == {PARAMETRISATION: "whitened", "train.lr": exp.axes["train.lr"][1]}


# Each path's run-start check value as a fraction of the value it refuses (1): penalty lr *
# weight_lambda / (2 beta) * stiffness, whitened lr * lambda_max(loss Hessian) at the prior
# initial state (the only state it is checked at; at stationarity linear_ring's is 1.14, the
# description says so). The real runs take 0.9 of each bound; the pipeline preset keeps both
# far inside. Keys: (preset, setting index).
PENALTY_CHECK = {("linear_ring_equivalence", 0): (0.89, 0.9), ("linear_ring_tiny", 0): (0.2, 0.25)}
WHITENED_CHECK = {("linear_ring", 0): (0.85, 0.9), ("linear_ring_tiny", 3): (0.0, 0.05)}


def _whitened_check_value(exp: ExpConfig, index: int) -> float:
    """lr * lambda_max(loss Hessian) of a setting at its initial net, the quantity `fct.train`
    checks when the run starts (the ensemble's largest member value)."""
    cfg = setting_config(exp, expand(exp)[index])
    dataset = make_dataset(cfg.data, torch.device("cpu"), torch.float64)
    x, y = dataset.x, dataset.y
    net = make_net_from_config(cfg, x.shape[1], y.shape[1], torch.device("cpu"), torch.float64)
    penalty = make_weight_penalty(cfg, net)
    coefficients = loss_coefficients(cfg.train)
    lam = largest_hessian_eigenvalue(
        lambda: compute_loss(net, x, y, coefficients, penalty)[0], net.parameters()
    )
    return cfg.train.lr * lam


@pytest.mark.parametrize(("name", "index"), sorted(PENALTY_CHECK))
def test_the_penalty_path_runs_at_its_stated_fraction_of_its_check(name, index):
    cfg = setting_config(preset(name), expand(preset(name))[index])
    assert cfg.spatial.parametrisation == "penalty"
    factors = parameter_factors(
        build_spaces(cfg.spatial, m=cfg.data.D_in, N=cfg.net.N, L=1, n_out=cfg.data.D_in)
    )
    value = cfg.train.lr * prior_coefficient(cfg.train) * prior_stiffness(factors)
    low, high = PENALTY_CHECK[(name, index)]
    assert low <= value <= high


@pytest.mark.parametrize(("name", "index"), sorted(WHITENED_CHECK))
def test_the_whitened_path_runs_at_its_stated_fraction_of_its_check(name, index):
    exp = preset(name)
    assert setting_config(exp, expand(exp)[index]).spatial.parametrisation == "whitened"
    low, high = WHITENED_CHECK[(name, index)]
    assert low <= _whitened_check_value(exp, index) <= high


def test_off_the_diagonal_the_penalty_at_the_whitened_lr_is_refused(tmp_path):
    with pytest.raises(ValueError, match="diverges on the spatial prior"):
        run_setting(preset("linear_ring_tiny"), 1, tmp_path, **QUIET)


# ---------------------------------------------------------------------------- editing a folder
def test_editing_a_file_edits_the_preset(folder):
    assert preset("tiny", folder).N == 8
    _edit(folder / "tiny.json", N=16)
    assert preset("tiny", folder).N == 16
    assert preset("tiny").N == 8  # the packaged file is untouched


def test_a_new_file_is_a_new_preset(folder):
    shutil.copy(folder / "tiny.json", folder / "mine_2.json")
    assert "mine_2" in preset_names(folder)
    assert preset("mine_2", folder) == preset("tiny", folder)


def test_only_regular_files_with_identifier_stems_are_presets(folder):
    (folder / "x.json").mkdir()  # a folder, even though it ends in .json
    (folder / ".hidden.json").write_text("{}")
    (folder / ".json").write_text("{}")
    (folder / "README.md").write_text("notes")
    (folder / "tiny.json.bak").write_text("{}")
    assert preset_names(folder) == PACKAGED


def test_non_identifier_json_files_are_named_in_the_unknown_preset_error(folder):
    shutil.copy(folder / "tiny.json", folder / "my-preset.json")
    shutil.copy(folder / "tiny.json", folder / "2fast.json")
    (folder / ".hidden.json").write_text("{}")
    assert preset_names(folder) == PACKAGED
    with pytest.raises(PresetError) as caught:
        preset("my-preset", folder)
    message = str(caught.value)
    assert "unknown preset 'my-preset'" in message
    assert all(name in message for name in PACKAGED)
    assert "my-preset.json" in message and "2fast.json" in message
    assert "[A-Za-z_][A-Za-z0-9_]*" in message
    assert ".hidden.json" not in message


def test_an_unknown_preset_error_without_ignored_files_does_not_mention_ignoring():
    with pytest.raises(PresetError, match="unknown preset 'huge'") as caught:
        preset("huge")
    assert "Ignored" not in str(caught.value)


def test_a_path_is_not_a_preset_name(folder):
    outside = folder.parent / "outside.json"
    shutil.copy(folder / "tiny.json", outside)
    with pytest.raises(PresetError, match="unknown preset"):
        preset("../outside", folder)


def test_a_missing_presets_folder_raises_a_preset_error_naming_it(tmp_path):
    with pytest.raises(PresetError, match="nowhere"):
        preset_names(tmp_path / "nowhere")


# ---------------------------------------------------------------------------- bad files
@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"{not json", "not a valid experiment"),
        (b'{"N": 8}', "not a valid experiment"),
        (b"[1, 2]", "not a valid experiment"),
        (b"\xff\xfe\x00", "cannot read"),
    ],
    ids=["broken-json", "missing-fields", "not-an-object", "not-utf8"],
)
def test_an_invalid_preset_file_raises_a_preset_error_naming_its_full_path(folder, content, reason):
    (folder / "tiny.json").write_bytes(content)
    with pytest.raises(PresetError, match=reason) as caught:
        preset("tiny", folder)
    assert str(folder / "tiny.json") in str(caught.value)


def test_an_unreadable_preset_file_raises_a_preset_error_naming_its_full_path(folder):
    path = folder / "tiny.json"
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("permissions are not enforced for this user")
        with pytest.raises(PresetError, match="cannot read") as caught:
            preset("tiny", folder)
        assert str(path) in str(caught.value)
    finally:
        path.chmod(0o644)


def test_a_preset_file_that_is_a_valid_json_but_a_bad_experiment_names_the_field(folder):
    _edit(folder / "tiny.json", N=-3)
    with pytest.raises(PresetError, match="N") as caught:
        preset("tiny", folder)
    assert str(folder / "tiny.json") in str(caught.value)


# ---------------------------------------------------------------------------- the CLI
def test_exp_init_writes_the_preset_of_an_edited_folder(folder, use_folder, tmp_path, capsys):
    use_folder(folder)
    _edit(folder / "tiny.json", N=16)
    root = tmp_path / "root"
    assert _cli(capsys, "init", root, "--preset", "tiny")[0] == 0
    assert load_exp(root).N == 16


def test_exp_init_accepts_a_preset_added_to_the_folder(folder, use_folder, tmp_path, capsys):
    use_folder(folder)
    shutil.copy(folder / "tiny.json", folder / "extra.json")
    _edit(folder / "extra.json", N=12, description="mine")
    root = tmp_path / "root"
    assert _cli(capsys, "init", root, "--preset", "extra")[0] == 0
    assert (load_exp(root).N, load_exp(root).description) == (12, "mine")


def test_an_orthonormal_teacher_preset_round_trips_through_exp_init(
    folder, use_folder, tmp_path, capsys
):
    use_folder(folder)
    shutil.copy(folder / "tiny.json", folder / "ortho.json")
    teacher = json.loads((folder / "tiny.json").read_text(encoding="utf-8"))["linear_teacher"]
    _edit(folder / "ortho.json", linear_teacher={**teacher, "teacher": "orthonormal"})
    root = tmp_path / "root"
    assert _cli(capsys, "init", root, "--preset", "ortho")[0] == 0
    exp = load_exp(root)
    assert exp.linear_teacher.teacher == "orthonormal"
    assert '"teacher": "orthonormal"' in (root / "exp.json").read_text(encoding="utf-8")
    teachers = {
        setting_config(exp, s).data.teacher for s in expand(exp) if s.task == "linear_teacher"
    }
    assert teachers == {"orthonormal"}


def test_a_gaussian_teacher_is_left_out_of_exp_json(tmp_path):
    save_exp(preset("tiny"), tmp_path)
    assert '"teacher":' not in (tmp_path / "exp.json").read_text(encoding="utf-8")
    assert load_exp(tmp_path).linear_teacher.teacher == "gaussian"


def test_help_lists_the_presets_of_the_folder(folder, use_folder, capsys):
    use_folder(folder)
    shutil.copy(folder / "tiny.json", folder / "extra.json")
    with pytest.raises(SystemExit) as stop:
        main(["exp", "init", "--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "extra" in out and "phase_transition" in out and "--n-lambda" not in out


def test_an_unknown_preset_is_an_error_line_that_names_ignored_files(
    folder, use_folder, tmp_path, capsys
):
    use_folder(folder)
    shutil.copy(folder / "tiny.json", folder / "bad-name.json")
    code, _, err = _cli(capsys, "init", tmp_path / "r", "--preset", "bad-name")
    assert code == 1 and err.startswith("error:") and "bad-name.json" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("content", [b"{broken", b'{"N": 1}', b"\xff\xfe"])
def test_an_invalid_preset_file_is_an_error_line_not_a_traceback(
    folder, use_folder, tmp_path, capsys, content
):
    use_folder(folder)
    (folder / "tiny.json").write_bytes(content)
    code, _, err = _cli(capsys, "init", tmp_path / "r", "--preset", "tiny")
    assert code == 1 and err.startswith("error:") and str(folder / "tiny.json") in err
    assert "Traceback" not in err
    assert not (tmp_path / "r" / "exp.json").exists()


def test_a_broken_presets_folder_fails_only_exp_init(tmp_path, use_folder, capsys):
    root = tmp_path / "root"
    save_exp(with_axes(preset("tiny"), {"net.L": [1, 2]}), root)
    use_folder(tmp_path / "no_such_folder")
    for args in (["count", root], ["list", root], ["status", root]):
        assert _cli(capsys, *args)[0] == 0
    code, _, err = _cli(capsys, "init", tmp_path / "other", "--preset", "tiny")
    assert code == 1 and err.startswith("error:") and "no_such_folder" in err
    with pytest.raises(SystemExit) as stop:  # even --help does not crash
        main(["exp", "init", "--help"])
    assert stop.value.code == 0


# ---------------------------------------------------------------------------- overrides
def test_without_overrides_init_keeps_the_presets_description(tmp_path, capsys):
    assert _cli(capsys, "init", tmp_path, "--preset", "spatial_exp")[0] == 0
    assert load_exp(tmp_path).description == preset("spatial_exp").description


def test_overrides_replace_the_description_with_a_true_one(tmp_path, capsys):
    args = ["--set", "n_epochs=4", "--set", "linear_teacher.n_features=6"]
    args += ["--axis", "net.N=[4,8]", "--axis", 'net.phi=["tanh"]']
    assert _cli(capsys, "init", tmp_path, "--preset", "spatial_tiny", *args)[0] == 0
    assert load_exp(tmp_path).description == (
        "Preset 'spatial_tiny' with overrides: --set n_epochs=4; "
        '--set linear_teacher.n_features=6; --axis net.N=[4,8]; --axis net.phi=["tanh"]'
    )


def test_one_override_is_enough_for_a_derived_description(tmp_path, capsys):
    assert _cli(capsys, "init", tmp_path, "--preset", "tiny", "--set", "N=16")[0] == 0
    assert load_exp(tmp_path).description == "Preset 'tiny' with overrides: --set N=16"


def test_overrides_change_nothing_but_what_they_name(tmp_path, capsys):
    assert _cli(capsys, "init", tmp_path, "--preset", "tiny", "--set", "N=16")[0] == 0
    expected = preset("tiny").model_copy(update={"N": 16})
    assert same_experiment(load_exp(tmp_path), expected)


# ---------------------------------------------------------------------------- the description
def test_a_description_is_omitted_from_exp_json_when_there_is_none():
    exp = preset("tiny").model_copy(update={"description": None})
    assert "description" not in json.loads(exp_text(exp))
    assert json.loads(exp_text(preset("tiny")))["description"] == preset("tiny").description


def test_an_exp_json_without_a_description_loads_with_none(tmp_path):
    (tmp_path / "exp.json").write_text(
        exp_text(preset("tiny").model_copy(update={"description": None})), encoding="utf-8"
    )
    assert load_exp(tmp_path).description is None


def test_B_is_omitted_from_exp_json_at_the_default_and_written_otherwise():
    assert "B" not in json.loads(exp_text(preset("tiny")))
    exp = ExpConfig.model_validate({**preset("tiny").model_dump(), "B": 8})
    assert json.loads(exp_text(exp))["B"] == 8


def test_a_shared_B_survives_the_save_load_round_trip(tmp_path):
    exp = ExpConfig.model_validate({**preset("tiny").model_dump(), "B": 8})
    save_exp(exp, tmp_path)
    loaded = load_exp(tmp_path)
    assert loaded.B == 8
    assert same_experiment(loaded, exp)


def test_save_exp_writes_utf_8(tmp_path):
    exp = preset("tiny").model_copy(update={"description": "lambda λ, ≥"})
    save_exp(exp, tmp_path)
    raw = (tmp_path / "exp.json").read_bytes()
    assert "λ".encode() in raw
    assert load_exp(tmp_path).description == exp.description
    assert not list(tmp_path.glob("*.tmp"))


def test_same_experiment_ignores_only_the_description():
    a = preset("tiny")
    assert same_experiment(a, a.model_copy(update={"description": "other"}))
    assert same_experiment(a, a.model_copy(update={"description": None}))
    assert not same_experiment(a, a.model_copy(update={"N": a.N + 1}))
    assert not same_experiment(a, a.model_copy(update={"spatial": None, "lr": a.lr * 2}))


def test_the_description_is_in_no_hash_directory_or_setting_config(tmp_path):
    a = with_axes(preset("tiny"), {"net.L": [1, 2]}, description="one")
    b = with_axes(preset("tiny"), {"net.L": [1, 2]}, description="two")
    assert exp_hash(a) == exp_hash(b)
    for sa, sb in zip(expand(a), expand(b), strict=True):
        assert setting_dir(tmp_path, sa) == setting_dir(tmp_path, sb)
        ca, cb = setting_config(a, sa), setting_config(b, sb)
        assert ca == cb
        assert ca.section_hash("data", "net", "train") == cb.section_hash("data", "net", "train")


def _setting_files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_description_only_change_makes_nothing_stale_or_refused(tmp_path, capsys):
    exp = with_axes(preset("tiny"), {"net.L": [1, 2]}, description="before")
    save_exp(exp, tmp_path)
    run_setting(exp, 0, tmp_path, **QUIET)
    artefacts = {k: v for k, v in _setting_files(tmp_path).items() if k != "exp.json"}

    edited = exp.model_copy(update={"description": "after"})
    save_exp(edited, tmp_path)
    assert load_exp(tmp_path).description == "after"

    states = [s for *_, s in setting_statuses(tmp_path, load_exp(tmp_path))]
    assert states == ["done", "missing"]
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)  # done: not refused, not redone
    run_setting(load_exp(tmp_path), 1, tmp_path, **QUIET)  # a new setting beside the old one
    code, out, _ = _cli(capsys, "status", tmp_path)
    assert code == 0 and "2 done, 0 partial, 0 missing, 0 stale" in out
    kept = {k: v for k, v in _setting_files(tmp_path).items() if k in artefacts}
    assert kept == artefacts  # nothing that was written for setting 0 changed


def test_cli_status_and_run_accept_a_description_only_edit(tmp_path, capsys):
    exp = with_axes(preset("tiny"), {"net.L": [1]}, description="before")
    save_exp(exp, tmp_path)
    assert _cli(capsys, "run", tmp_path, "--index", 0)[0] == 0
    _edit(tmp_path / "exp.json", description="after")
    code, out, _ = _cli(capsys, "status", tmp_path)
    assert code == 0 and "1 done, 0 partial, 0 missing, 0 stale" in out
    code, out, err = _cli(capsys, "run", tmp_path, "--index", 0)
    assert code == 0 and "already done" in out and err == ""


def test_a_resumed_migration_ignores_a_description_edited_into_exp_json(tmp_path):
    exp = with_axes(preset("tiny"), {"net.L": [1, 2]})
    tree = json.loads(exp_text(exp))
    legacy = {("grid" if key == "axes" else key): value for key, value in tree.items()}
    (tmp_path / "sweep.json").write_text(json.dumps(legacy, indent=2) + "\n", encoding="utf-8")
    # an earlier run wrote exp.json and was interrupted before removing sweep.json; meanwhile
    # the description was edited
    edited = json.dumps({"description": "edited", **tree}, indent=2) + "\n"
    (tmp_path / "exp.json").write_text(edited, encoding="utf-8")
    lines = migrate_exp(tmp_path)
    assert any("resuming" in line for line in lines)
    assert not (tmp_path / "sweep.json").exists()
    assert (tmp_path / "exp.json").read_text(encoding="utf-8") == edited
    assert load_exp(tmp_path).description == "edited"


def test_migration_still_writes_the_exp_json_without_a_description(tmp_path):
    exp = with_axes(preset("tiny"), {"net.L": [1, 2]})
    text = exp_text(exp)
    tree = json.loads(text)
    legacy = {("grid" if key == "axes" else key): value for key, value in tree.items()}
    (tmp_path / "sweep.json").write_text(json.dumps(legacy, indent=2) + "\n", encoding="utf-8")
    migrate_exp(tmp_path)
    assert "description" not in json.loads((tmp_path / "exp.json").read_text(encoding="utf-8"))
    assert load_exp(tmp_path) == ExpConfig.model_validate(tree)


def test_set_description_is_kept_over_the_derived_one(tmp_path, capsys):
    args = ["--set", "N=16", "--set", "description=my own words"]
    assert _cli(capsys, "init", tmp_path, "--preset", "tiny", *args)[0] == 0
    assert load_exp(tmp_path).description == "my own words"


def test_set_description_null_gives_no_description(tmp_path, capsys):
    assert _cli(capsys, "init", tmp_path, "--preset", "tiny", "--set", "description=null")[0] == 0
    assert load_exp(tmp_path).description is None
    assert "description" not in json.loads((tmp_path / "exp.json").read_text(encoding="utf-8"))


def test_the_frozen_regulariser_and_the_tiny_solve_size_are_in_the_descriptions():
    assert "0.0002" in preset("tiny").description
    assert "0.0002" in preset("full").description
    assert "0.0002" in preset("modularity").description
    assert "at most P" in preset("spatial_tiny").description
    assert "decades" not in preset("lambda_exp").description
