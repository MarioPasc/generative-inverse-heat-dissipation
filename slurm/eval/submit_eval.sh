#!/usr/bin/env bash
# Submit the T5.1 evaluation jobs.  One launcher, four workers:
#
#   bash slurm/eval/submit_eval.sh prepare [--dry-run|--test-only]
#   bash slurm/eval/submit_eval.sh timing  [--dry-run|--test-only]
#   PREPARE_JOB=<id> TRAIN_ARRAY=<id> bash slurm/eval/submit_eval.sh gate  [--dry-run|--test-only]
#   PREPARE_JOB=<id> [TRAIN_ARRAY=<id>] bash slurm/eval/submit_eval.sh array [--dry-run|--test-only]
#
#   ARRAY_SPEC='6,17' PREPARE_JOB=<id> bash slurm/eval/submit_eval.sh array   # resubmit indices
#   AMP=fp16 ... submit_eval.sh array                                         # only if main decides
#
# Order (slurm/eval/README.md): prepare -> gate -> main's extension decision -> array.
#
# Dependencies.  PREPARE_JOB adds `afterok:<prepare>` to gate and array (omit it once the prepare
# job has COMPLETED and its 24 files exist: the workers check them anyway).  TRAIN_ARRAY adds
# `aftercorr:<train>` to the array (eval task i starts when training task i COMPLETES) and
# `afterok:<train>_0:<train>_3` to the gate.  `--test-only` drops every dependency flag, because
# sbatch --test-only rejects a dependency on a job it cannot see.
#
# Run from the login node.  Every resource flag is repeated on the sbatch line so the submission
# is self-documenting; command-line flags win over the workers' #SBATCH directives.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
LOGS_DIR="${IHDM_LOGS_DIR:-${USER_ROOT}/execs/ihdm/logs}"
export IHDM_REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
export IHDM_ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm}"
export IHDM_EVAL_HOME="${IHDM_EVAL_HOME:-${USER_ROOT}/execs/ihdm/eval}"
export IHDM_CELLS="${IHDM_CELLS:-${IHDM_REPO_DIR}/slurm/array/cells.csv}"
export AMP="${AMP:-off}"
export N_ITERS="${N_ITERS:-40000}"

QOS="${QOS:-medium_uma}"
CPUS="${CPUS:-8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
PREPARE_JOB="${PREPARE_JOB:-}"
TRAIN_ARRAY="${TRAIN_ARRAY:-}"

WHAT="${1:-}"
MODE="submit"
case "${2:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          ;;
    *) echo "usage: $0 {prepare|timing|gate|array} [--dry-run|--test-only]" >&2; exit 2 ;;
esac
case "${AMP}" in off|fp16|bf16) ;; *) echo "FATAL: AMP must be off, fp16 or bf16" >&2; exit 2 ;; esac
for kv in IHDM_REPO_DIR IHDM_ENV_PREFIX IHDM_DATA_ROOT IHDM_RUN_ROOT IHDM_EVAL_HOME IHDM_CELLS; do
    [[ "${!kv}" == *","* ]] && { echo "FATAL: ${kv} contains a comma; --export would truncate it" >&2; exit 1; }
done
EXPORTS="ALL,IHDM_REPO_DIR=${IHDM_REPO_DIR},IHDM_ENV_PREFIX=${IHDM_ENV_PREFIX},IHDM_DATA_ROOT=${IHDM_DATA_ROOT},IHDM_RUN_ROOT=${IHDM_RUN_ROOT},IHDM_EVAL_HOME=${IHDM_EVAL_HOME},IHDM_CELLS=${IHDM_CELLS},AMP=${AMP},N_ITERS=${N_ITERS}"

COMMON=(--qos="${QOS}" --ntasks=1 --cpus-per-task="${CPUS}" --constraint=a100 --gres=gpu:1
        --account=tic_163_uma --export="${EXPORTS}")
DEPS=()

case "${WHAT}" in
    prepare)
        WORKER="${SCRIPT_DIR}/prepare_eval.sbatch"
        ARGS=(--job-name=ihdm-eval-prep --time="${TIME_LIMIT:-01:00:00}" --mem="${MEM:-32G}"
              --output="${LOGS_DIR}/eval_prepare_%j.out" --error="${LOGS_DIR}/eval_prepare_%j.err")
        ;;
    timing)
        WORKER="${SCRIPT_DIR}/timing.sbatch"
        ARGS=(--job-name=ihdm-eval-timing --time="${TIME_LIMIT:-02:00:00}" --mem="${MEM:-32G}"
              --output="${LOGS_DIR}/eval_timing_%j.out" --error="${LOGS_DIR}/eval_timing_%j.err")
        ;;
    gate)
        WORKER="${SCRIPT_DIR}/gate.sbatch"
        ARGS=(--job-name=ihdm-eval-gate --time="${TIME_LIMIT:-03:00:00}" --mem="${MEM:-32G}"
              --output="${LOGS_DIR}/eval_gate_%j.out" --error="${LOGS_DIR}/eval_gate_%j.err")
        # The gate cells are the index column of cells.csv (0 = ixi_A0_s1, 3 = lsun_church_A0_s1).
        GATE_CELLS="${GATE_CELLS:-0:3}"
        EXPORTS="${EXPORTS},GATE_CELLS=${GATE_CELLS}"
        COMMON=(--qos="${QOS}" --ntasks=1 --cpus-per-task="${CPUS}" --constraint=a100
                --gres=gpu:1 --account=tic_163_uma --export="${EXPORTS}")
        UPSTREAM=()
        if [[ -n "${TRAIN_ARRAY}" ]]; then
            for idx in ${GATE_CELLS//:/ }; do UPSTREAM+=("${TRAIN_ARRAY}_${idx}"); done
        fi
        [[ -n "${PREPARE_JOB}" ]] && UPSTREAM+=("${PREPARE_JOB}")
        (( ${#UPSTREAM[@]} > 0 )) && DEPS=(--dependency="afterok:$(IFS=:; echo "${UPSTREAM[*]}")")
        ;;
    array)
        WORKER="${SCRIPT_DIR}/eval_array.sbatch"
        CELLS="${IHDM_CELLS}"
        [[ -f "${CELLS}" ]] || { echo "FATAL: no cell table at ${CELLS}" >&2; exit 1; }
        # The range is derived from the cell table, never hard-coded (slurm/array/submit_array.sh).
        N_CELLS=$(awk -F, 'NR > 1 && NF > 1 {n++} END {print n + 0}' "${CELLS}")
        (( N_CELLS > 0 )) || { echo "FATAL: ${CELLS} has no data rows" >&2; exit 1; }
        ARRAY_SPEC="${ARRAY_SPEC:-0-$(( N_CELLS - 1 ))%${MAX_CONCURRENT}}"
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
        # --time: measured per-run cost x 1.3, docs/RESULTS/evaluation_plan.md §3.
        ARGS=(--array="${ARRAY_SPEC}" --job-name=ihdm-eval --time="${TIME_LIMIT:-10:00:00}"
              --mem="${MEM:-48G}"
              --output="${LOGS_DIR}/eval_%A_%a.out" --error="${LOGS_DIR}/eval_%A_%a.err")
        ARRAY_DEP=""
        [[ -n "${PREPARE_JOB}" ]] && ARRAY_DEP="afterok:${PREPARE_JOB}"
        [[ -n "${TRAIN_ARRAY}" ]] && ARRAY_DEP="${ARRAY_DEP:+${ARRAY_DEP},}aftercorr:${TRAIN_ARRAY}"
        [[ -n "${ARRAY_DEP}" ]] && DEPS=(--dependency="${ARRAY_DEP}")
        ;;
    *)
        echo "usage: $0 {prepare|timing|gate|array} [--dry-run|--test-only]" >&2
        exit 2
        ;;
esac

[[ -f "${WORKER}" ]] || { echo "FATAL: no worker at ${WORKER}" >&2; exit 1; }
mkdir -p "${LOGS_DIR}" "${IHDM_EVAL_HOME}"

SBATCH_ARGS=("${ARGS[@]}" "${COMMON[@]}")
echo "what:        ${WHAT}"
echo "worker:      ${WORKER}"
echo "repo:        ${IHDM_REPO_DIR}  ($(git -C "${IHDM_REPO_DIR}" rev-parse --short HEAD 2>/dev/null || cat "${IHDM_REPO_DIR}/.git_sha" 2>/dev/null || echo n/a))"
echo "amp:         ${AMP}"
echo "eval home:   ${IHDM_EVAL_HOME}"
echo "dependency:  ${DEPS[*]:-none}"
echo "sbatch:      sbatch ${SBATCH_ARGS[*]} ${DEPS[*]:-} ${WORKER}"
echo ""
echo "---- quota (read live) ----"
quota 2>&1 | tail -n 3 || echo "[warn] \`quota\` not available here"

if [[ "${MODE}" == "dry" ]]; then
    echo ""
    echo "[DRY-RUN] nothing submitted"
    exit 0
fi

echo ""
echo "---- sbatch --test-only (dependency flags dropped) ----"
sbatch --test-only "${SBATCH_ARGS[@]}" "${WORKER}"
rc=$?
if (( rc != 0 )); then
    echo "FATAL: --test-only rejected the request (exit ${rc}); nothing submitted" >&2
    exit 1
fi
[[ "${MODE}" == "test" ]] && { echo "[TEST-ONLY] nothing submitted"; exit 0; }

echo ""
echo "---- sbatch ----"
# Picasso's Lua sbatch wrapper prints ANSI codes and a banner on stdout: the id is the LAST number
# on the LAST line (slurm/array/submit_array.sh).
RAW=$(sbatch "${SBATCH_ARGS[@]}" "${DEPS[@]}" "${WORKER}" 2>&1)
echo "${RAW}"
JOB_ID=$(printf '%s\n' "${RAW}" | grep -oE '[0-9]+' | tail -1)
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unparsable job id; run '/usr/bin/squeue -u ${USER}' NOW -- the job may be queued" >&2
    exit 1
fi
if (( ${#DEPS[@]} > 0 )) && scontrol show job "${JOB_ID}" 2>/dev/null | grep -q 'Dependency=(null)'; then
    echo "FATAL: the dependency was dropped on ${JOB_ID}; cancelling it" >&2
    scancel "${JOB_ID}"
    exit 1
fi
echo ""
echo "Submitted ${WHAT} ${JOB_ID}"
scontrol show job "${JOB_ID}" 2>/dev/null | grep -oE '(JobState|Dependency|TimeLimit)=[^ ]+' | paste -sd' ' -

cat <<EOF

---- monitoring ----
/usr/bin/squeue -u ${USER} -j ${JOB_ID}      # Picasso's squeue wrapper rejects -u; call the binary
sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS,NodeList | head -40
ls ${LOGS_DIR}/eval_*${JOB_ID}*
EOF
