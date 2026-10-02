#!/usr/bin/env bash
# T7.3 -- submit the photograph-failure diagnostic (or its smoke, or the dataset preparation) on
# Picasso.  Run on the login node from the code copy at ~/execs/ihdm/wt/T7.3 (rsynced from the
# worktree, GIT_SHA beside it).
#
# Usage:
#   bash slurm/diag_eval/submit_photo_eval.sh prepare [--dry-run|--test-only]
#   bash slurm/diag_eval/submit_photo_eval.sh smoke   [--dry-run|--test-only]
#   ARRAY_SPEC=0 bash slurm/diag_eval/submit_photo_eval.sh eval [--dry-run|--test-only]
#
# prepare: slurm/eval/prepare_eval.sbatch for lsun_church_r128 and lsun_church_n32k ONLY (seed
#          lists + reference Inception features, checked against expected_seed_lists.csv), QOS
#          short, --time 00:45:00, record to ~/execs/ihdm/diag_eval/photo_diagnostic/.
# smoke:   task 0 (baseline) at 60k only, 32 LSD seeds and 4 x 5 held-out chains, --time
#          00:30:00, results to ~/execs/ihdm/diag_eval/photo_smoke/ (never resumed by eval).
# eval:    the array of photo_eval.sbatch, tasks 0 baseline, 1 r128, 2 n32k (ARRAY_SPEC, default
#          0-2, so a task can be given as soon as its run has finished), 45k/50k/55k/60k,
#          --time 02:30:00, results to ~/execs/ihdm/diag_eval/photo_diagnostic/.
set -euo pipefail

MODE="${1:-}"
FLAG="${2:-}"
USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/execs/ihdm/wt/T7.3}"
LOGS_DIR="${USER_ROOT}/execs/ihdm/logs"
RESULTS="${USER_ROOT}/execs/ihdm/diag_eval/photo_diagnostic"
RUN_DIRS="${USER_ROOT}/fscratch/runs/ihdm/lsun_church_A0_s1:${USER_ROOT}/fscratch/runs/ihdm_diag/lsun_church_r128_A0_s1:${USER_ROOT}/fscratch/runs/ihdm_diag/lsun_church_n32k_A0_s1"
QOS="${QOS:-medium_uma}"

case "${MODE}" in
    prepare)
        WORKER="${REPO_DIR}/slurm/eval/prepare_eval.sbatch"
        ARGS=(--qos=short --time=00:45:00
              --output="${LOGS_DIR}/diag_photo_prepare_%j.out" --error="${LOGS_DIR}/diag_photo_prepare_%j.err"
              --export="ALL,IHDM_REPO_DIR=${REPO_DIR},IHDM_DATASETS=lsun_church_r128:lsun_church_n32k,IHDM_EVAL_HOME=${RESULTS}"
              "${WORKER}")
        OUT_DIR="${RESULTS}"
        ;;
    smoke|eval)
        WORKER="${REPO_DIR}/slurm/diag_eval/photo_eval.sbatch"
        if [[ "${MODE}" == smoke ]]; then
            ARRAY="0"; TIME="00:30:00"; STEPS="60000"
            OUT_DIR="${USER_ROOT}/execs/ihdm/diag_eval/photo_smoke"
            EXTRA="--n-lsd 32 --n-seeds 4"
        else
            ARRAY="${ARRAY_SPEC:-0-2}"; TIME="02:30:00"; STEPS="45000:50000:55000:60000"
            OUT_DIR="${RESULTS}"
            EXTRA=""
        fi
        # Lists are colon-separated and EVAL_EXTRA_ARGS holds spaces and no comma, so all survive
        # --export.
        ARGS=(--array="${ARRAY}" --time="${TIME}" --qos="${QOS}"
              --output="${LOGS_DIR}/diag_photo_%A_%a.out" --error="${LOGS_DIR}/diag_photo_%A_%a.err"
              --export="ALL,IHDM_REPO_DIR=${REPO_DIR},RUN_DIRS=${RUN_DIRS},STEPS=${STEPS},OUT_DIR=${OUT_DIR},EVAL_EXTRA_ARGS=${EXTRA}"
              "${WORKER}")
        ;;
    *)
        echo "usage: $0 prepare|smoke|eval [--dry-run|--test-only]" >&2
        exit 2
        ;;
esac

[[ -f "${WORKER}" ]] || { echo "FATAL: no worker at ${WORKER}" >&2; exit 1; }
[[ -s "${REPO_DIR}/GIT_SHA" ]] || { echo "FATAL: ${REPO_DIR}/GIT_SHA is missing" >&2; exit 1; }
mkdir -p "${LOGS_DIR}" "${OUT_DIR}"

# Picasso's Lua wrapper prepends ANSI + a warning banner to --parsable output: take the last line.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

echo "mode:     ${MODE}   git: $(cat "${REPO_DIR}/GIT_SHA")"
echo "command:  sbatch --parsable ${ARGS[*]}"
case "${FLAG}" in
    --dry-run) exit 0 ;;
    --test-only) sbatch --test-only "${ARGS[@]}"; exit $? ;;
    "") ;;
    *) echo "unknown flag ${FLAG}" >&2; exit 2 ;;
esac

RAW=$(sbatch --parsable "${ARGS[@]}")
JOB_ID=$(_clean_job_id "${RAW}")
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${RAW@Q}; check squeue" >&2; exit 1; }
echo "submitted ${JOB_ID}"
echo "monitor:  squeue -j ${JOB_ID}"
echo "logs:     ${LOGS_DIR}/diag_photo_*${JOB_ID}*"
echo "results:  ${OUT_DIR}"
