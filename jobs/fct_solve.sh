#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FunctionalCellTypes — solve the mean-field saddle point of every setting of an experiment
#
#  The same array job as jobs/fct_exp.sh, with the stage set to "solve": one array task
#  solves one setting (`python -m fct exp run ROOT --index $TASK --stage solve`), using the
#  trained net of that setting if there is one and solving on its own otherwise. The solver
#  settings are the `meanfield` section of ROOT/exp.json. A solve resumes from its latest
#  checkpoint and a finished one is skipped, so resubmitting an array is safe.
#
#  Usage and environment overrides are those of jobs/fct_exp.sh:
#    bash jobs/fct_solve.sh ROOT --submit [--index X]   # X = N (one setting) or A:B (A .. B-1)
#    bash jobs/fct_solve.sh ROOT --dry-run [--index X]
#    bash jobs/fct_solve.sh ROOT --list | --status     # --status shows the solve stage
#
#  This file fixes the stage, its own job name and the log prefix, and hands everything else to
#  fct_exp.sh. The #SBATCH header below repeats fct_exp.sh's (apart from -J) because sbatch
#  reads it from the file that is submitted, which is this one; tests/test_slurm_script.py
#  checks that the two agree. SLURM runs it directly per array task (ROOT as its only argument).
# ═══════════════════════════════════════════════════════════════════════════════

# --array, --output and --error are passed by --submit (see jobs/fct_exp.sh). Keep the rest
# equal to the header of jobs/fct_exp.sh.
#SBATCH -J fct-solve
#SBATCH -p gpu
#SBATCH --gres=gpu:1
# Exclude the Quadro P5000s (sm_61); this torch build supports sm_75+ only.
#SBATCH --exclude=gpu-380-10,gpu-380-12,gpu-380-14
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-08:00

# SLURM executes a copy of this file from its spool directory, so inside a task the project
# is the one --submit passed in FCT_PROJECT; interactively it is derived from this file.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT="${FCT_PROJECT:-$(dirname "$(dirname "${SELF}")")}"

export FCT_STAGE=solve
export FCT_LOG_NAME=fct_solve
export FCT_ENTRY="${PROJECT}/jobs/fct_solve.sh"
exec bash "${PROJECT}/jobs/fct_exp.sh" "$@"
