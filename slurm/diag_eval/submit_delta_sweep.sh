#!/usr/bin/env bash
# T7.2 -- submit the delta sweep (or its smoke) on Picasso.  Run on the login node from the code
# copy at ~/execs/ihdm/wt/T7.2 (rsynced from the worktree, GIT_SHA beside it).
#
# Usage:
#   bash slurm/diag_eval/submit_delta_sweep.sh sweep [--dry-run|--test-only]
#   bash slurm/diag_eval/submit_delta_sweep.sh smoke [--dry-run|--test-only]
#
# sweep: array 0-3 (ixi_A0_s1, ixi_A3_s1, lsun_church_A0_s1, lsun_church_A3_s1), delta in
#        {0.0125, 0.02, 0.03}, 2,100 chains per task, --time 02:30:00, results to
#        ~/execs/ihdm/diag_eval/delta_sweep/.
# smoke: task 0 only (ixi_A0_s1), delta 0.0125, 32 LSD seeds and 4 x 5 held-out chains,
#        --time 00:30:00, results to ~/execs/ihdm/diag_eval/smoke/ (never resumed by the sweep).
set -euo pipefail

MODE="${1:-}"
FLAG="${2:-}"
USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/execs/ihdm/wt/T7.2}"
LOGS_DIR="${USER_ROOT}/execs/ihdm/logs"
WORKER="${REPO_DIR}/slurm/diag_eval/delta_sweep.sbatch"
RUNS="ixi_A0_s1:ixi_A3_s1:lsun_church_A0_s1:lsun_church_A3_s1"

case "${MODE}" in
    sweep)
        ARRAY="0-3"
        TIME="02:30:00"
        DELTAS="0.0125:0.02:0.03"
        OUT_DIR="${USER_ROOT}/execs/ihdm/diag_eval/delta_sweep"
        EXTRA=""
        ;;
    smoke)
        ARRAY="0"
        TIME="00:30:00"
        DELTAS="0.0125"
        OUT_DIR="${USER_ROOT}/execs/ihdm/diag_eval/smoke"
        EXTRA="--n-lsd 32 --n-seeds 4"
        ;;
    *)
        echo "usage: $0 sweep|smoke [--dry-run|--test-only]" >&2
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

# EVAL_EXTRA_ARGS holds spaces and no comma, so it survives --export.
ARGS=(--array="${ARRAY}" --time="${TIME}"
      --output="${LOGS_DIR}/diag_eval_%A_%a.out" --error="${LOGS_DIR}/diag_eval_%A_%a.err"
      --export="ALL,IHDM_REPO_DIR=${REPO_DIR},RUNS=${RUNS},DELTAS=${DELTAS},OUT_DIR=${OUT_DIR},EVAL_EXTRA_ARGS=${EXTRA}"
      "${WORKER}")

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
echo "logs:     ${LOGS_DIR}/diag_eval_${JOB_ID}_<task>.out"
echo "results:  ${OUT_DIR}"
