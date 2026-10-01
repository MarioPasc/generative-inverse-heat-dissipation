#!/usr/bin/env bash
# T7.1 (M7 diagnostic) -- submit the two one-factor Churches runs of `slurm/diag_train/cells.csv`
# (lsun_church_r128_A0_s1, lsun_church_n32k_A0_s1) through the production worker
# `slurm/array/train_array.sbatch`, UNCHANGED: this launcher only points the worker at another
# cell table, code copy and run root through the env overrides the worker already honours
# (IHDM_CELLS, IHDM_REPO_DIR, IHDM_RUN_ROOT, N_ITERS). Same recipe, same resume rule, same exit
# codes as the 30 production runs.
#
#   bash slurm/diag_train/submit_diag.sh --dry-run                     # print, submit nothing
#   TIME_LIMIT=hh:mm:ss ARRAY_SPEC=0 bash slurm/diag_train/submit_diag.sh --test-only
#   TIME_LIMIT=hh:mm:ss ARRAY_SPEC=0 bash slurm/diag_train/submit_diag.sh   # one cell, 60k
#
# TIME_LIMIT has no default on purpose: the two cells differ in image size (128² against 192²),
# so each gets its own measured limit (it/s of the smoke x 60,000 x 1.3, rounded up; see
# README.md and docs/RESULTS/diagnostic_training.md). Submit one cell per call.
#
# Smoke (a short job, its own run root, never the production one):
#   N_ITERS=500 QOS=short TIME_LIMIT=00:45:00 ARRAY_SPEC=0-1 \
#   IHDM_RUN_ROOT=/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm_diag_smoke \
#       bash slurm/diag_train/submit_diag.sh
#
# Run on the login node from the code copy (~/execs/ihdm/wt/T7.1). Every resource flag is on the
# sbatch command line (they win over the worker's #SBATCH header), so the submission record in
# docs/RESULTS/diagnostic_training.md is the complete request.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/../array/train_array.sbatch"
CELLS="${SCRIPT_DIR}/cells.csv"

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
PRODUCTION_RUN_ROOT="${USER_ROOT}/fscratch/runs/ihdm"
LOGS_DIR="${IHDM_LOGS_DIR:-${USER_ROOT}/execs/ihdm/logs}"
export IHDM_REPO_DIR="${IHDM_REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export IHDM_ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm_diag}"
export IHDM_CELLS="${IHDM_REPO_DIR}/slurm/diag_train/cells.csv"

N_ITERS="${N_ITERS:-60000}"
QOS="${QOS:-medium_uma}"
TIME_LIMIT="${TIME_LIMIT:-}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
JOB_NAME="${JOB_NAME:-ihdm-diag}"

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          ;;
    *) echo "usage: $0 [--dry-run|--test-only]" >&2; exit 2 ;;
esac

[[ -f "${WORKER}" ]] || { echo "FATAL: no worker at ${WORKER}" >&2; exit 1; }
[[ -f "${CELLS}" ]] || { echo "FATAL: no cell table at ${CELLS}" >&2; exit 1; }
[[ -n "${TIME_LIMIT}" ]] || { echo "FATAL: set TIME_LIMIT (per cell, see README.md)" >&2; exit 1; }
# The 30 production runs live under PRODUCTION_RUN_ROOT; a diagnostic run must never share it.
[[ "${IHDM_RUN_ROOT%/}" != "${PRODUCTION_RUN_ROOT}" ]] \
    || { echo "FATAL: IHDM_RUN_ROOT is the production run root ${PRODUCTION_RUN_ROOT}" >&2; exit 1; }
# The worker reads IHDM_CELLS; this launcher validates CELLS. They must be the same table.
cmp -s "${CELLS}" "${IHDM_CELLS}" \
    || { echo "FATAL: ${IHDM_CELLS} is not ${CELLS}; sync the code copy first" >&2; exit 1; }

N_CELLS=$(awk -F, 'NR > 1 && NF > 1 {n++} END {print n + 0}' "${CELLS}")
(( N_CELLS > 0 )) || { echo "FATAL: ${CELLS} has no data rows" >&2; exit 1; }
DENSE=$(awk -F, 'NR > 1 && NF > 1 {if ($1 != NR - 2) bad = 1} END {print bad + 0}' "${CELLS}")
(( DENSE == 0 )) || { echo "FATAL: ${CELLS} index column is not dense and 0-based" >&2; exit 1; }
ARRAY_SPEC="${ARRAY_SPEC:-0-$(( N_CELLS - 1 ))}"

# Expand ARRAY_SPEC (a,b / a-b / trailing %K) and check every index against the table and its
# dataset against the data root, so a typo or a missing transfer fails here, not on a GPU.
INDICES=$(
    awk -v spec="${ARRAY_SPEC}" 'BEGIN {
        sub(/%.*$/, "", spec); n = split(spec, parts, ",")
        for (i = 1; i <= n; i++) {
            if (parts[i] ~ /^[0-9]+-[0-9]+$/) { split(parts[i], r, "-"); for (k = r[1]; k <= r[2]; k++) print k }
            else if (parts[i] ~ /^[0-9]+$/) print parts[i]
            else print "BAD:" parts[i]
        }
    }'
)
for index in ${INDICES}; do
    ROW=$(awk -F, -v want="${index}" 'NR > 1 && $1 == want' "${CELLS}")
    [[ -n "${ROW}" ]] || { echo "FATAL: --array=${ARRAY_SPEC}: no row ${index} in ${CELLS}" >&2; exit 1; }
    DATASET_ID=$(cut -d, -f3 <<<"${ROW}")
    for f in images.npy index.csv splits.json meta.json; do
        [[ -f "${IHDM_DATA_ROOT}/${DATASET_ID}/${f}" ]] \
            || { echo "FATAL: ${IHDM_DATA_ROOT}/${DATASET_ID}/${f} missing" >&2; exit 1; }
    done
    echo "cell ${ROW}"
done

EXPORTS="ALL,IHDM_REPO_DIR=${IHDM_REPO_DIR},IHDM_ENV_PREFIX=${IHDM_ENV_PREFIX},IHDM_DATA_ROOT=${IHDM_DATA_ROOT},IHDM_RUN_ROOT=${IHDM_RUN_ROOT},IHDM_CELLS=${IHDM_CELLS},N_ITERS=${N_ITERS}"
for kv in IHDM_REPO_DIR IHDM_ENV_PREFIX IHDM_DATA_ROOT IHDM_RUN_ROOT IHDM_CELLS; do
    [[ "${!kv}" == *","* ]] && { echo "FATAL: ${kv} contains a comma; --export would truncate it" >&2; exit 1; }
done

SBATCH_ARGS=(
    --array="${ARRAY_SPEC}"
    --job-name="${JOB_NAME}"
    --time="${TIME_LIMIT}"
    --qos="${QOS}"
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --constraint=a100
    --gres=gpu:1
    --account=tic_163_uma
    --output="${LOGS_DIR}/diag_train_%A_%a.out"
    --error="${LOGS_DIR}/diag_train_%A_%a.err"
    --export="${EXPORTS}"
)

echo "worker:      ${WORKER}"
echo "cells:       ${CELLS} (${N_CELLS} rows)"
echo "array:       ${ARRAY_SPEC}"
echo "n_iters:     ${N_ITERS}"
echo "time limit:  ${TIME_LIMIT}   qos: ${QOS}"
echo "code:        ${IHDM_REPO_DIR}  (GIT_SHA $(cat "${IHDM_REPO_DIR}/GIT_SHA" 2>/dev/null || echo n/a))"
echo "data root:   ${IHDM_DATA_ROOT}"
echo "run root:    ${IHDM_RUN_ROOT}"
echo "logs:        ${LOGS_DIR}/diag_train_<arrayjobid>_<task>.{out,err}"
echo "sbatch:      sbatch ${SBATCH_ARGS[*]} ${WORKER}"
echo ""
echo "---- quota (read live) ----"
quota 2>&1 || echo "[warn] \`quota\` not available here"

if [[ "${MODE}" == "dry" ]]; then
    echo ""; echo "[DRY-RUN] nothing submitted"; exit 0
fi

mkdir -p "${LOGS_DIR}" "${IHDM_RUN_ROOT}"

echo ""; echo "---- sbatch --test-only ----"
sbatch --test-only "${SBATCH_ARGS[@]}" "${WORKER}"
rc=$?
(( rc == 0 )) || { echo "FATAL: --test-only rejected the request (exit ${rc}); nothing submitted" >&2; exit 1; }
[[ "${MODE}" == "test" ]] && { echo "[TEST-ONLY] nothing submitted"; exit 0; }

echo ""; echo "---- sbatch ----"
# Picasso's Lua wrapper prints ANSI codes and a banner on stdout: the job id is the LAST number
# on the LAST line (slurm/array/submit_array.sh, T3.3).
RAW=$(sbatch "${SBATCH_ARGS[@]}" "${WORKER}" 2>&1)
echo "${RAW}"
JOB_ID=$(printf '%s\n' "${RAW}" | tail -n 1 | grep -oE '[0-9]+' | tail -1)
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unparsable job id; run 'squeue' NOW -- the job may already be queued" >&2
    exit 1
fi

echo ""
echo "Submitted array ${JOB_ID} (--array=${ARRAY_SPEC}, N_ITERS=${N_ITERS}, --time=${TIME_LIMIT})"
squeue -j "${JOB_ID}" 2>&1 || true
cat <<EOF

---- monitoring ----
squeue -j ${JOB_ID} -o "%.14i %.12j %.8T %.10M %.6D %R"
sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS,NodeList
grep -m1 '^CELL' ${LOGS_DIR}/diag_train_${JOB_ID}_*.out
tail -n 3 ${IHDM_RUN_ROOT}/<run_id>/metrics.jsonl
EOF
