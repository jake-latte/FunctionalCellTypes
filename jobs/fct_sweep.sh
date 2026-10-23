#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FunctionalCellTypes — grid sweep (any number of settings) as a SLURM array
#
#  One array task trains one setting: `python -m fct sweep run ROOT --index $TASK --stage train`.
#  jobs/fct_solve.sh runs this same script with FCT_STAGE=solve, which solves the mean-field
#  saddle point of each setting instead (see that file).
#  The grid itself lives in ROOT/sweep.json (write it first with
#  `python -m fct sweep init ROOT --preset full`, or `--preset phase_transition --n-lambda 8`);
#  this script only needs its size (`sweep count`).
#
#  Usage (ROOT = sweep root, as the first argument or in $FCT_SWEEP_ROOT):
#    bash jobs/fct_sweep.sh ROOT --submit              # submit the full array
#    bash jobs/fct_sweep.sh ROOT --submit --index N    # submit ONLY setting N
#    bash jobs/fct_sweep.sh ROOT --dry-run [--index N] # print the sbatch command, run nothing
#    bash jobs/fct_sweep.sh ROOT --list                # index -> setting directory
#    bash jobs/fct_sweep.sh ROOT --status              # done / partial / missing / stale per setting
#
#  Environment overrides:
#    FCT_MAX_CONCURRENT  cap on simultaneously-running array tasks (default 4)
#    FCT_PYTHON          interpreter (default <project>/.venv/bin/python)
#    FCT_STAGE           train (default) or solve; set by jobs/fct_solve.sh, not normally by hand
#    FCT_ENTRY, FCT_LOG_NAME   the script sbatch submits and the log-file prefix; set by the
#                        wrapper (default: this script, fct_sweep)
#
#  SLURM runs this script directly per array task (ROOT as its only argument).
# ═══════════════════════════════════════════════════════════════════════════════

# ── SLURM directives ──────────────────────────────────────────────────────────
# --array, --output and --error are NOT set here: --submit passes them on the sbatch
# command line, because #SBATCH lines cannot expand variables and the log directory
# must follow the project location.
#SBATCH -J fct-sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
# Exclude the Quadro P5000s (sm_61); this torch build supports sm_75+ only.
#SBATCH --exclude=gpu-380-10,gpu-380-12,gpu-380-14
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-08:00

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MAX_CONCURRENT="${FCT_MAX_CONCURRENT:-4}"

# ═══════════════════════════════════════════════════════════════════════════════
#  Everything below this line is machinery — rarely needs editing.
# ═══════════════════════════════════════════════════════════════════════════════

STAGE="${FCT_STAGE:-train}"
LOG_NAME="${FCT_LOG_NAME:-fct_sweep}"

USAGE="Use: bash jobs/${LOG_NAME}.sh [ROOT] (--submit | --dry-run | --list | --status) [--index N]"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ "${STAGE}" == "train" || "${STAGE}" == "solve" ]] || die "FCT_STAGE must be train or solve (got '${STAGE}')."

# SLURM executes a copy of this script from its spool directory, so the project
# location is passed in through FCT_PROJECT by --submit; interactively it is derived
# from where this file lives.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FCT_PROJECT:-$(dirname "${SELF_DIR}")}"
PYTHON="${FCT_PYTHON:-${PROJECT}/.venv/bin/python}"
# The python is run after a cd into the project, so a relative FCT_PYTHON must be
# resolved against the caller's directory first.
[[ "${PYTHON}" == /* || "${PYTHON}" != */* ]] || PYTHON="${PWD}/${PYTHON}"
LOG_DIR="${PROJECT}/tmp/logs"
# The script sbatch submits: this one, or the wrapper that selected the stage.
ENTRY="${FCT_ENTRY:-${SELF_DIR}/$(basename "${BASH_SOURCE[0]}")}"

# ── Parse CLI: ROOT, one action, optional --index N ───────────────────────────
ROOT="${FCT_SWEEP_ROOT:-}"
ROOT_GIVEN=""
ACTION=""
INDEX=""
INDEX_GIVEN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit|--dry-run|--list|--status)
            [[ -z "${ACTION}" ]] || die "only one action allowed (got ${ACTION} and $1). ${USAGE}"
            ACTION="$1"; shift ;;
        --index)
            [[ $# -ge 2 ]] || die "--index requires a value. ${USAGE}"
            INDEX="$2"; INDEX_GIVEN=1; shift 2 ;;
        --index=*) INDEX="${1#*=}"; INDEX_GIVEN=1; shift ;;
        -*) die "Unrecognised argument: '$1'. ${USAGE}" ;;
        *)  [[ -z "${ROOT_GIVEN}" ]] || die "more than one ROOT given ('${ROOT}' and '$1'). ${USAGE}"
            ROOT="$1"; ROOT_GIVEN=1; shift ;;
    esac
done

if [[ -n "${INDEX_GIVEN}" ]]; then
    [[ "${INDEX}" =~ ^[0-9]+$ ]] || die "--index needs a non-negative integer (got '${INDEX}')."
    INDEX=$((10#${INDEX}))   # base 10: '010' is setting 10, not octal 8; '08' is valid
    [[ "${ACTION}" == "--submit" || "${ACTION}" == "--dry-run" ]] \
        || die "--index only applies to --submit and --dry-run."
fi

[[ -n "${ROOT}" ]] || die "no sweep ROOT given (pass it as an argument or set FCT_SWEEP_ROOT). ${USAGE}"
[[ -d "${ROOT}" ]] || die "sweep ROOT '${ROOT}' does not exist."
ROOT="$(cd "${ROOT}" && pwd)"   # absolute: array tasks start in a different cwd

fct() { (cd "${PROJECT}" && "${PYTHON}" -m fct "$@"); }

# ── Job execution (runs inside SLURM: no action, array task id set) ───────────
if [[ -z "${ACTION}" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "no action given. ${USAGE}"
    # Without FCT_PROJECT the project would be guessed from SLURM's spool copy of this script.
    [[ -n "${FCT_PROJECT:-}" ]] || die "array task started without FCT_PROJECT; submit with --submit."
    [[ -f "${PYTHON}" && -x "${PYTHON}" ]] || die "python '${PYTHON}' is not an executable file."
    echo "========================================================"
    echo "Job: ${SLURM_JOB_ID:-?}, array task: ${SLURM_ARRAY_TASK_ID}"
    echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
    echo "root=${ROOT}"
    echo "========================================================"
    echo "stage=${STAGE}"
    fct sweep run "${ROOT}" --index "${SLURM_ARRAY_TASK_ID}" --stage "${STAGE}"
    EXIT_CODE=$?
    echo "Task ${SLURM_ARRAY_TASK_ID} finished with exit code ${EXIT_CODE}."
    exit ${EXIT_CODE}
fi

# ── Read-only actions ─────────────────────────────────────────────────────────
case "${ACTION}" in
    --list)   fct sweep list "${ROOT}";   exit $? ;;
    --status) fct sweep status "${ROOT}" --stage "${STAGE}"; exit $? ;;
esac

# ── --submit / --dry-run: build one sbatch command and either print or run it ─
N_TOTAL=$(fct sweep count "${ROOT}") || die "could not read the sweep size from ${ROOT}/sweep.json."
[[ "${N_TOTAL}" =~ ^[0-9]+$ ]] || die "could not determine sweep size (got '${N_TOTAL}')."

# --export is comma-delimited, so a comma in either path would split it.
[[ "${PROJECT}${PYTHON}" != *,* ]] || die "project or python path contains a comma; sbatch --export cannot carry it."

if [[ -n "${INDEX_GIVEN}" ]]; then
    if (( INDEX >= N_TOTAL )); then
        die "--index '${INDEX}' out of range [0, ${N_TOTAL})"
    fi
    ARRAY_SPEC="${INDEX}"
else
    ARRAY_SPEC="0-$((N_TOTAL-1))%${MAX_CONCURRENT}"
fi

SBATCH_CMD=(
    sbatch
    --array="${ARRAY_SPEC}"
    --output="${LOG_DIR}/${LOG_NAME}_%A_%a.out"
    --error="${LOG_DIR}/${LOG_NAME}_%A_%a.err"
    --export="ALL,FCT_PROJECT=${PROJECT},FCT_PYTHON=${PYTHON}"
    "${ENTRY}" "${ROOT}"
)

if [[ "${ACTION}" == "--dry-run" ]]; then
    echo "Dry run — nothing submitted. --submit would run:"
    printf '  %q' "${SBATCH_CMD[@]}"; echo
    echo "  per task: ${PYTHON} -m fct sweep run ${ROOT} --index \$SLURM_ARRAY_TASK_ID --stage ${STAGE}"
    exit 0
fi

echo "Settings: ${N_TOTAL}  (array ${ARRAY_SPEC})  root=${ROOT}"
mkdir -p "${LOG_DIR}"
"${SBATCH_CMD[@]}"
