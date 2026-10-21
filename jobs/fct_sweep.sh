#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FunctionalCellTypes — 36-setting sweep as a SLURM array
#
#  One array task trains one setting: `python -m fct sweep run ROOT --index $TASK`.
#  The grid itself lives in ROOT/sweep.json (write it first with
#  `python -m fct sweep init ROOT --preset full`); this script only needs its size.
#
#  Usage (ROOT = sweep root, as the first argument or in $FCT_SWEEP_ROOT):
#    bash jobs/fct_sweep.sh ROOT --submit              # submit the full array
#    bash jobs/fct_sweep.sh ROOT --submit --index N    # submit ONLY setting N
#    bash jobs/fct_sweep.sh ROOT --dry-run [--index N] # print the sbatch command, run nothing
#    bash jobs/fct_sweep.sh ROOT --list                # index -> setting directory
#    bash jobs/fct_sweep.sh ROOT --status              # done / partial / missing per setting
#
#  Environment overrides:
#    FCT_MAX_CONCURRENT  cap on simultaneously-running array tasks (default 4)
#    FCT_PYTHON          interpreter (default <project>/.venv/bin/python)
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

USAGE="Use: bash jobs/fct_sweep.sh [ROOT] (--submit | --dry-run | --list | --status) [--index N]"

die() { echo "ERROR: $*" >&2; exit 1; }

# SLURM executes a copy of this script from its spool directory, so the project
# location is passed in through FCT_PROJECT by --submit; interactively it is derived
# from where this file lives.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${FCT_PROJECT:-$(dirname "${SELF_DIR}")}"
PYTHON="${FCT_PYTHON:-${PROJECT}/.venv/bin/python}"
LOG_DIR="${PROJECT}/tmp/logs"

# ── Parse CLI: ROOT, one action, optional --index N ───────────────────────────
ROOT="${FCT_SWEEP_ROOT:-}"
ACTION=""
INDEX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit|--dry-run|--list|--status)
            [[ -z "${ACTION}" ]] || die "only one action allowed (got ${ACTION} and $1). ${USAGE}"
            ACTION="$1"; shift ;;
        --index)   INDEX="${2:?--index requires a value}"; shift 2 ;;
        --index=*) INDEX="${1#*=}"; shift ;;
        -*) die "Unrecognised argument: '$1'. ${USAGE}" ;;
        *)  ROOT="$1"; shift ;;
    esac
done

[[ -n "${ROOT}" ]] || die "no sweep ROOT given (pass it as an argument or set FCT_SWEEP_ROOT). ${USAGE}"
[[ -d "${ROOT}" ]] || die "sweep ROOT '${ROOT}' does not exist."
ROOT="$(cd "${ROOT}" && pwd)"   # absolute: array tasks start in a different cwd

fct() { (cd "${PROJECT}" && "${PYTHON}" -m fct "$@"); }

# ── Job execution (runs inside SLURM: no action, array task id set) ───────────
if [[ -z "${ACTION}" ]]; then
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || die "no action given. ${USAGE}"
    echo "========================================================"
    echo "Job: ${SLURM_JOB_ID:-?}, array task: ${SLURM_ARRAY_TASK_ID}"
    echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
    echo "root=${ROOT}"
    echo "========================================================"
    fct sweep run "${ROOT}" --index "${SLURM_ARRAY_TASK_ID}"
    EXIT_CODE=$?
    echo "Task ${SLURM_ARRAY_TASK_ID} finished with exit code ${EXIT_CODE}."
    exit ${EXIT_CODE}
fi

# ── Read-only actions ─────────────────────────────────────────────────────────
case "${ACTION}" in
    --list)   fct sweep list "${ROOT}";   exit $? ;;
    --status) fct sweep status "${ROOT}"; exit $? ;;
esac

# ── --submit / --dry-run: build one sbatch command and either print or run it ─
N_TOTAL=$(fct sweep count "${ROOT}") || die "could not read the sweep size from ${ROOT}/sweep.json."
[[ "${N_TOTAL}" =~ ^[0-9]+$ ]] || die "could not determine sweep size (got '${N_TOTAL}')."

if [[ -n "${INDEX}" ]]; then
    if ! [[ "${INDEX}" =~ ^[0-9]+$ ]] || (( INDEX >= N_TOTAL )); then
        die "--index '${INDEX}' out of range [0, ${N_TOTAL})"
    fi
    ARRAY_SPEC="${INDEX}"
else
    ARRAY_SPEC="0-$((N_TOTAL-1))%${MAX_CONCURRENT}"
fi

SBATCH_CMD=(
    sbatch
    --array="${ARRAY_SPEC}"
    --output="${LOG_DIR}/fct_sweep_%A_%a.out"
    --error="${LOG_DIR}/fct_sweep_%A_%a.err"
    --export="ALL,FCT_PROJECT=${PROJECT},FCT_PYTHON=${PYTHON}"
    "${SELF_DIR}/$(basename "${BASH_SOURCE[0]}")" "${ROOT}"
)

if [[ "${ACTION}" == "--dry-run" ]]; then
    echo "Dry run — nothing submitted. --submit would run:"
    printf '  %q' "${SBATCH_CMD[@]}"; echo
    echo "  per task: ${PYTHON} -m fct sweep run ${ROOT} --index \$SLURM_ARRAY_TASK_ID"
    exit 0
fi

echo "Settings: ${N_TOTAL}  (array ${ARRAY_SPEC})  root=${ROOT}"
mkdir -p "${LOG_DIR}"
"${SBATCH_CMD[@]}"
