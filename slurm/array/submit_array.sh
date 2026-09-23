#!/usr/bin/env bash
# Submit the T3.3 training array: 30 runs of `slurm/array/cells.csv`, one A100 task each.
#
#   bash slurm/array/submit_array.sh --dry-run          # print the sbatch line, submit nothing
#   bash slurm/array/submit_array.sh --test-only        # sbatch --test-only, submit nothing
#   bash slurm/array/submit_array.sh                    # --test-only, then the real submission
#
#   ARRAY_SPEC='6' bash slurm/array/submit_array.sh     # resubmit one index (resumes that run)
#   N_ITERS=60000 bash slurm/array/submit_array.sh      # extend every run to 60k (D10)
#
# Run from the login node, inside the Picasso checkout.  The worker carries its own `#SBATCH`
# header; every resource flag is repeated on the `sbatch` command line so the submission is
# self-documenting in `docs/RESULTS/submissions.md` and a one-off override needs no edit to the
# worker.  Command-line flags win over the worker's directives.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/train_array.sbatch"
CELLS="${SCRIPT_DIR}/cells.csv"

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
LOGS_DIR="${IHDM_LOGS_DIR:-${USER_ROOT}/execs/ihdm/logs}"
export IHDM_REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
export IHDM_ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm}"
export IHDM_CELLS="${IHDM_CELLS:-${IHDM_REPO_DIR}/slurm/array/cells.csv}"

# D16 (T3.2, job 2405546): 1.71 it/s at batch 16 -> 6.84 h per 40k run; 1.5x rounded up = 11 h.
# QOS `medium` (3-day wall); `short` caps at 2 h and would TIMEOUT every task.
N_ITERS="${N_ITERS:-40000}"
QOS="${QOS:-medium}"
TIME_LIMIT="${TIME_LIMIT:-11:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
JOB_NAME="${JOB_NAME:-ihdm-train}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          ;;
    *) echo "usage: $0 [--dry-run|--test-only]" >&2; exit 2 ;;
esac

[[ -f "${WORKER}" ]] || { echo "FATAL: no worker at ${WORKER}" >&2; exit 1; }
[[ -f "${CELLS}" ]] || { echo "FATAL: no cell table at ${CELLS}" >&2; exit 1; }

# ----------------------------------------------------------------------------------------------
# The array range is DERIVED from the cell table, never hard-coded: a hard-coded count cannot
# disagree with the decode, so it cannot catch a decode that changed.
# ----------------------------------------------------------------------------------------------
N_CELLS=$(awk -F, 'NR > 1 && NF > 1 {n++} END {print n + 0}' "${CELLS}")
(( N_CELLS > 0 )) || { echo "FATAL: ${CELLS} has no data rows" >&2; exit 1; }
MAX_INDEX=$(( N_CELLS - 1 ))

# The index column must be dense and 0-based, or `--array=0-${MAX_INDEX}` addresses rows that the
# worker's index lookup will not find.
DENSE=$(awk -F, -v n="${N_CELLS}" 'NR > 1 && NF > 1 {if ($1 != NR - 2) bad = 1} END {print bad + 0}' "${CELLS}")
(( DENSE == 0 )) || { echo "FATAL: ${CELLS} index column is not dense and 0-based" >&2; exit 1; }

ARRAY_SPEC="${ARRAY_SPEC:-0-${MAX_INDEX}%${MAX_CONCURRENT}}"

# Every index named in ARRAY_SPEC must exist in the table.  `--array` accepts `a,b`, `a-b` and a
# trailing `%K`; expand those forms and check each index, so a typo fails here rather than after
# a task has already burned a GPU hour.
BAD=$(
    awk -F, 'NR > 1 && NF > 1 {seen[$1] = 1}
         END {
             spec = SPEC; sub(/%.*$/, "", spec)
             n = split(spec, parts, ",")
             for (i = 1; i <= n; i++) {
                 if (parts[i] ~ /^[0-9]+-[0-9]+$/) {
                     split(parts[i], r, "-")
                     for (k = r[1]; k <= r[2]; k++) if (!(k in seen)) bad = bad " " k
                 } else if (parts[i] ~ /^[0-9]+$/) {
                     if (!(parts[i] in seen)) bad = bad " " parts[i]
                 } else {
                     bad = bad " <unparsable:" parts[i] ">"
                 }
             }
             print bad
         }' SPEC="${ARRAY_SPEC}" "${CELLS}"
)
[[ -z "${BAD// /}" ]] || { echo "FATAL: --array=${ARRAY_SPEC} names indices absent from ${CELLS}:${BAD}" >&2; exit 1; }

mkdir -p "${LOGS_DIR}" "${IHDM_RUN_ROOT}"

# No comma may appear inside an --export VALUE: SLURM splits on commas and would truncate it.
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
    --output="${LOGS_DIR}/train_%A_%a.out"
    --error="${LOGS_DIR}/train_%A_%a.err"
    --export="${EXPORTS}"
)

echo "worker:      ${WORKER}"
echo "cells:       ${CELLS} (${N_CELLS} rows, indices 0-${MAX_INDEX})"
echo "array:       ${ARRAY_SPEC}"
echo "n_iters:     ${N_ITERS}"
echo "repo:        ${IHDM_REPO_DIR}  ($(git -C "${IHDM_REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a) on $(git -C "${IHDM_REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a))"
echo "run root:    ${IHDM_RUN_ROOT}"
echo "logs:        ${LOGS_DIR}/train_<arrayjobid>_<task>.{out,err}"
echo "sbatch:      sbatch ${SBATCH_ARGS[*]} ${WORKER}"

echo ""
echo "---- quota (read live; the array writes ~40 files per run) ----"
quota 2>&1 || echo "[warn] \`quota\` not available here"

if [[ "${MODE}" == "dry" ]]; then
    echo ""
    echo "[DRY-RUN] nothing submitted"
    exit 0
fi

echo ""
echo "---- sbatch --test-only ----"
sbatch --test-only "${SBATCH_ARGS[@]}" "${WORKER}"
rc=$?
if (( rc != 0 )); then
    echo "FATAL: --test-only rejected the request (exit ${rc}); nothing submitted" >&2
    exit 1
fi
[[ "${MODE}" == "test" ]] && { echo "[TEST-ONLY] nothing submitted"; exit 0; }

echo ""
echo "---- sbatch ----"
# Picasso's Lua sbatch wrapper prints ANSI codes and a warning banner on stdout, so the job id is
# the LAST number on the LAST line; a line-by-line `sed 's/[^0-9]//g'` would return a multi-line
# "id" and the guard below would fire AFTER the array was already queued.
RAW=$(sbatch "${SBATCH_ARGS[@]}" "${WORKER}" 2>&1)
echo "${RAW}"
JOB_ID=$(printf '%s\n' "${RAW}" | grep -oE '[0-9]+' | tail -1)
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unparsable job id; run 'squeue' NOW -- the array may already be queued" >&2
    exit 1
fi

echo ""
echo "Submitted array ${JOB_ID} (--array=${ARRAY_SPEC}, N_ITERS=${N_ITERS})"
# Picasso's squeue wrapper rejects `-u`; it already scopes output to the calling user.
squeue -j "${JOB_ID}" 2>&1 || true

cat <<EOF

---- monitoring ----
squeue                                                   # your queue (Picasso's wrapper rejects -u)
squeue -j ${JOB_ID} -o "%.14i %.12j %.8T %.10M %.6D %R"   # this array, task by task
squeue --start -j ${JOB_ID}                              # estimated start of the pending tasks
sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS,NodeList | head -40
sacct -j ${JOB_ID} -X -n -P -o State | sort | uniq -c     # state histogram
tail -f ${LOGS_DIR}/train_${JOB_ID}_0.out

# which runs have finished (H-PICASSO §5):
for r in \$(cut -d, -f2 ${IHDM_CELLS} | tail -n +2); do
    test -f ${IHDM_RUN_ROOT}/\$r/DONE && echo "DONE \$r" || echo "---- \$r"
done

---- resubmission ----
ARRAY_SPEC='6' bash slurm/array/submit_array.sh          # one index; the run resumes
ARRAY_SPEC="\$(sacct -j ${JOB_ID} -n -X -o JobID,State | awk '\$2 != "COMPLETED" {split(\$1, a, "_"); print a[2]}' | paste -sd, -)%${MAX_CONCURRENT}" \\
    bash slurm/array/submit_array.sh                     # every task that did not COMPLETE
N_ITERS=60000 bash slurm/array/submit_array.sh           # extend all 30 runs (D10)
scancel ${JOB_ID}                                        # cancel the whole array
scancel ${JOB_ID}_22                                     # cancel one task
EOF
