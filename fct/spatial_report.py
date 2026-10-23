"""What the spatial-prior notebook shows: lists of markdown strings and figures for `report.show`,
one function per cell. Each only assembles library results (`fct.spatial`, `fct.spatial_view`,
`fct.grid_view.tuning_view`, `fct.draw`); the notebook cells call them and nothing else.
"""

from collections.abc import Mapping, Sequence

from fct.analyse import example_outputs, pick_examples
from fct.config import Config
from fct.draw import (
    fig_empirical_kernels,
    fig_example_outputs,
    fig_kernel,
    fig_loss_curves,
    fig_tuning_by_position,
    fig_weights_over_embedding,
)
from fct.grid_view import tuning_view
from fct.model import MLP
from fct.report import Item
from fct.spatial import LayerSpace, Population
from fct.spatial_view import TrainRun, ordered_kernel_comparison

EXAMPLE_SEED = 0


def _populations(spaces: Sequence[LayerSpace]) -> list[Population]:
    """Every population once, input first (consecutive layers share one)."""
    return [spaces[0].pre, *(s.post for s in spaces)]


def kernel_items(spaces: Sequence[LayerSpace]) -> list[Item]:
    """How much of each population's kernel the eigenvalue floor changed, then `fig_kernel` of
    every layer."""
    lines = [
        f"- {p.name}: {p.report.n_floored} of {len(p.K)} eigenvalues raised to the floor "
        f"({p.report.fraction_floored:.0%}); condition number {p.report.cond_before:.3g} "
        f"before, {p.report.cond_after:.3g} after"
        for p in _populations(spaces)
    ]
    text = "**Eigenvalue floor** (`min_rel_eig`) of each population's kernel:\n\n" + "\n".join(
        lines
    )
    return [text, *(fig_kernel(space) for space in spaces)]


def final_losses_text(history: Mapping[str, Sequence[float]], cfg: Config) -> str:
    """The last logged loss terms, and what was trained."""
    tc = cfg.train
    sampler = f"SGLD, beta = {tc.beta:g}" if tc.optimizer == "sgld" else "Adam"
    last = {key: history[key][-1] for key in ("epoch", "loss", "mse", "weight_l2", "rate_l2")}
    return (
        f"**Trained with {sampler}**, lr = {tc.lr:g}, {tc.n_epochs} epochs. At the last logged "
        f"epoch ({last['epoch']}): loss {last['loss']:.4g}, mse {last['mse']:.4g}, spatial weight "
        f"penalty {last['weight_l2']:.4g}, rate penalty {last['rate_l2']:.4g}."
    )


def training_items(run: TrainRun, cfg: Config, n_examples: int = 4) -> list[Item]:
    """The final losses, the loss curves and example outputs of the trained net."""
    picks = pick_examples(len(run.dataset.x), min(n_examples, len(run.dataset.x)), EXAMPLE_SEED)
    outputs = example_outputs(run.net, run.dataset, picks)
    return [
        final_losses_text(run.history, cfg),
        fig_loss_curves(run.history, title=f"{cfg.train.optimizer}: training"),
        fig_example_outputs(cfg.data.task, outputs, cfg.data, title="trained network"),
    ]


def weight_items(net: MLP, spaces: Sequence[LayerSpace]) -> list[Item]:
    """The first layer's weights U (hidden, input): the kernels they show beside the prior's
    and U as an image over the two rings."""
    space, W = spaces[0], net.U.detach()
    d = space.post.embedding.positions.shape[1]
    if d != 1:
        # Sorting units around a ring is what makes the weights readable; the tuning panels
        # (the main result) still cover d = 2, so this view is skipped rather than failing.
        return [f"**First-layer weights**: arranged over the embedding for d = 1 only (d = {d})."]
    text = (
        "**First-layer weights.** The empirical kernels of the trained `U` (one sample "
        "covariance each, so noisy) beside the prior's `K_post` and `K_pre`; then `U` itself "
        "with hidden units and input units sorted around their rings. A prior with a long "
        "length scale shows as smooth bands."
    )
    return [
        text,
        fig_empirical_kernels(ordered_kernel_comparison(space, W), space),
        fig_weights_over_embedding(W, space),
    ]


def tuning_by_position_items(
    run: TrainRun,
    spaces: Sequence[LayerSpace],
    n_bins: int,
    kind: str,
    n_shuffles: int,
    seed: int,
    n_cols: int,
) -> list[Item]:
    """For each hidden layer, the tuning of every neuron to every latent, one panel per neuron
    placed by its position (`fig_tuning_by_position`)."""
    items: list[Item] = []
    for layer in range(run.net.L):
        view = tuning_view(run.net, run.dataset, layer, n_bins, kind, n_shuffles, seed)
        positions = spaces[layer].post.embedding.positions
        fig = fig_tuning_by_position(
            view.curves, view.neuron_mir, view.latent_names, positions, n_cols
        )
        fig.suptitle(f"hidden layer {layer + 1}: tuning of every neuron, by position ({kind} MIR)")
        items.append(fig)
    return items
