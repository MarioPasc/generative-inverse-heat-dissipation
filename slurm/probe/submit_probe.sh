#!/usr/bin/env bash
# Submit the T3.2 Picasso probe (one A100 job, six sequential steps).
#
#   bash slurm/probe/submit_probe.sh --dry-run   # print the sbatch command, submit nothing
#   bash slurm/probe/submit_probe.sh --test-only # sbatch --test-only, submit nothing
#   bash slurm/probe/submit_probe.sh             # --test-only, then the real submission
#
# Run from the login node, inside the Picasso checkout.  The worker carries its own `#SBATCH`
# header (resources, QOS, A100 constraint, log paths); this launcher only creates the log and run
# directories, runs the mandatory `--test-only` probe, and captures the job id safely.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/probe.sbatch"

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
LOGS_DIR="${USER_ROOT}/execs/ihdm/logs"
export IHDM_REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
export IHDM_ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm/probe}"

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          ;;
    *) echo "usage: $0 [--dry-run|--test-only]" >&2; exit 2 ;;
esac

[[ -f "${WORKER}" ]] || { echo "FATAL: no worker at ${WORKER}" >&2; exit 1; }
mkdir -p "${LOGS_DIR}" "${IHDM_RUN_ROOT}"

EXPORTS="ALL,IHDM_REPO_DIR=${IHDM_REPO_DIR},IHDM_ENV_PREFIX=${IHDM_ENV_PREFIX},IHDM_DATA_ROOT=${IHDM_DATA_ROOT},IHDM_RUN_ROOT=${IHDM_RUN_ROOT}"

echo "worker:    ${WORKER}"
echo "repo:      ${IHDM_REPO_DIR}  ($(git -C "${IHDM_REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a) on $(git -C "${IHDM_REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a))"
echo "run root:  ${IHDM_RUN_ROOT}"
echo "logs:      ${LOGS_DIR}/probe_<jobid>.{out,err}"
echo "sbatch:    sbatch --export=${EXPORTS} ${WORKER}"

if [[ "${MODE}" == "dry" ]]; then
    echo "[DRY-RUN] nothing submitted"
    exit 0
fi

echo ""
echo "---- sbatch --test-only ----"
sbatch --test-only --export="${EXPORTS}" "${WORKER}"
rc=$?
if (( rc != 0 )); then
    echo "FATAL: --test-only rejected the request (exit ${rc}); nothing submitted" >&2
    exit 1
fi
[[ "${MODE}" == "test" ]] && { echo "[TEST-ONLY] nothing submitted"; exit 0; }

echo ""
echo "---- sbatch ----"
# Picasso's Lua sbatch wrapper prints ANSI codes and a warning banner on stdout, so the job id is
# the LAST line; a line-by-line `sed 's/[^0-9]//g'` would return a multi-line "id".
RAW=$(sbatch --export="${EXPORTS}" "${WORKER}" 2>&1)
echo "${RAW}"
JOB_ID=$(printf '%s\n' "${RAW}" | grep -oE '[0-9]+' | tail -1)
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unparsable job id; run 'squeue' NOW -- the job may already be queued" >&2
    exit 1
fi

echo ""
echo "Submitted job ${JOB_ID}"
squeue -j "${JOB_ID}" 2>&1 || true
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Log:      ${LOGS_DIR}/probe_${JOB_ID}.out"
echo "Accounting: sacct -j ${JOB_ID} -n -P -o JobID,State,Elapsed,MaxRSS,NodeList"
