#!/usr/bin/env bash
# T7.2 -- delta sweep: one array task = one 60k run evaluated at every noise level of DELTAS.
# Submitted by `slurm/diag_eval/submit_delta_sweep.sh`, which supplies `--array`, `--time` and
# the log paths on the command line.  Built on `slurm/eval/eval_array.sbatch` (same shadow run,
# same helpers from `slurm/eval/common.sh`, same copy-back discipline).
#
# Per task:
#   1. decode the run (RUNS, colon-separated), refuse a run without DONE or whose largest EMA
#      checkpoint is not STEP, and a dataset without the six one-writer files of prepare_eval;
#   2. stage clean-fid's weights to /tmp;
#   3. build a shadow run on $LOCALSCRATCH (links into the real run), unpacking this task's earlier
#      tar if one exists so a resubmission reuses every sample set already drawn;
#   4. for every delta: `evaluate_run --ckpts STEP --amp AMP --delta <d> --final-from-lsd
#      --n-seeds 40 --n-per-seed 5` (the frozen 500-seed LSD list, rng 2026, batch 32);
#   5. write <run_id>_delta_sweep.json (slurm/diag_eval/collect.py task) and return ONE tar of the
#      shadow tree plus that JSON to OUT_DIR, both atomically.  The tar is also written on
#      TIMEOUT/SIGTERM (trap).  Nothing is written on FSCRATCH or into the run directory.
#
# Sizing: 3 deltas x (500 + 40 x 5) = 2,100 chains x 2.42 s/chain (measured at 60k on A100 fp16,
# T5.1) = 1.41 h of sampling, plus three Inception passes on 500 samples; --time comes from the
# launcher.
#
#SBATCH -J ihdm-delta
#SBATCH --time=02:30:00
#SBATCH --qos=medium_uma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/diag_eval_%A_%a.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/diag_eval_%A_%a.err

set -uo pipefail
# Before ANY python: the env on FSCRATCH has no .pyc, and the first import would write ~900 there.
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/ihdm_pyc_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/execs/ihdm/wt/T7.2}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm}"
OUT_DIR="${OUT_DIR:-${USER_ROOT}/execs/ihdm/diag_eval/delta_sweep}"
RUNS="${RUNS:-ixi_A0_s1:ixi_A3_s1:lsun_church_A0_s1:lsun_church_A3_s1}"
DELTAS="${DELTAS:-0.0125:0.02:0.03}"
STEP="${STEP:-60000}"
AMP="${AMP:-fp16}"
SAMPLE_BATCH="${SAMPLE_BATCH:-32}"
N_SEEDS="${N_SEEDS:-40}"
N_PER_SEED="${N_PER_SEED:-5}"
# Extra evaluate_run flags for the SMOKE only (e.g. "--n-lsd 32"); the sweep never sets this.
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
PY="${ENV_PREFIX}/bin/python"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${IHDM_TASK_ID:-}}"
MYLOCALSCRATCH="${LOCALSCRATCH:-/tmp}/${USER}/delta_${SLURM_JOB_ID:-local}_${TASK_ID:-x}"

echo "=========================================="
echo "BEGIN TASK"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Array job:   ${SLURM_ARRAY_JOB_ID:-n/a}  task: ${TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Git commit:  $(cat "${REPO_DIR}/GIT_SHA" 2>/dev/null || echo n/a)"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Data root:   ${IHDM_DATA_ROOT}"
echo "Run root:    ${IHDM_RUN_ROOT}  (read only)"
echo "Out dir:     ${OUT_DIR}"
echo "Runs:        ${RUNS}"
echo "Deltas:      ${DELTAS}   step: ${STEP}   amp: ${AMP}   batch: ${SAMPLE_BATCH}"
echo "Held-out:    ${N_SEEDS} seeds x ${N_PER_SEED}"
echo "Local:       ${MYLOCALSCRATCH}"
echo "=========================================="

[[ -n "${TASK_ID}" ]] || { echo "[FATAL] no SLURM_ARRAY_TASK_ID (and no IHDM_TASK_ID)" >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}" >&2; exit 1; }
# shellcheck source=slurm/eval/common.sh
source "${REPO_DIR}/slurm/eval/common.sh" || { echo "[FATAL] cannot source common.sh" >&2; exit 1; }

IFS=':' read -r -a RUN_LIST <<<"${RUNS}"
IFS=':' read -r -a DELTA_LIST <<<"${DELTAS}"
(( TASK_ID >= 0 && TASK_ID < ${#RUN_LIST[@]} )) || { echo "[FATAL] task ${TASK_ID} outside the ${#RUN_LIST[@]} runs" >&2; exit 1; }
RUN_ID="${RUN_LIST[${TASK_ID}]}"
RUN_DIR="${IHDM_RUN_ROOT}/${RUN_ID}"
SUFFIX=$(ihdm_amp_suffix "${AMP}") || exit 1
SHADOW="${MYLOCALSCRATCH}/${RUN_ID}"
TAR_DEST="${OUT_DIR}/${RUN_ID}_delta_sweep.tar"
JSON_DEST="${OUT_DIR}/${RUN_ID}_delta_sweep.json"
echo "Task ${TASK_ID} -> run_id=${RUN_ID} deltas=${DELTA_LIST[*]}"

[[ -f "${RUN_DIR}/DONE" ]] || { echo "[FATAL] ${RUN_DIR}/DONE is missing" >&2; exit 1; }
LAST_STEP=$(ihdm_last_ema_step "${RUN_DIR}")
(( LAST_STEP == STEP )) || { echo "[FATAL] largest EMA checkpoint is ${LAST_STEP}, not STEP=${STEP}" >&2; exit 1; }
DATASET_ID=$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['data']['dataset'])" "${RUN_DIR}/config.json") \
    || { echo "[FATAL] cannot read data.dataset from ${RUN_DIR}/config.json" >&2; exit 1; }
SIGMA=$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['model']['sigma'])" "${RUN_DIR}/config.json") \
    || { echo "[FATAL] cannot read model.sigma" >&2; exit 1; }
echo "Dataset:     ${DATASET_ID}   sigma: ${SIGMA}"
ihdm_check_eval_caches "${IHDM_DATA_ROOT}/${DATASET_ID}" || exit 1

mkdir -p "${MYLOCALSCRATCH}" "${OUT_DIR}"

FINALIZED=0
finalize() {
    [[ "${FINALIZED}" == 1 ]] && return 0
    FINALIZED=1
    # Mirror the WHOLE shadow tree (links are stored as links); partial results come back too.
    if [[ -d "${SHADOW}" ]] && tar -cf "${MYLOCALSCRATCH}/${RUN_ID}.tar" -C "${MYLOCALSCRATCH}" "${RUN_ID}"; then
        if ihdm_copy_atomic "${MYLOCALSCRATCH}/${RUN_ID}.tar" "${TAR_DEST}"; then
            echo "copy-back:   ${TAR_DEST} ($(du -h "${TAR_DEST}" | cut -f1))"
            [[ -n "${MYLOCALSCRATCH}" && -d "${MYLOCALSCRATCH}" ]] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
            return 0
        fi
    fi
    echo "[FATAL] copy-back failed; results remain at ${MYLOCALSCRATCH} on $(hostname)" >&2
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ihdm_env_setup || exit 1
ihdm_stage_inception || exit 1

if [[ -f "${TAR_DEST}" ]]; then
    tar -xf "${TAR_DEST}" -C "${MYLOCALSCRATCH}" && echo "resume:      unpacked ${TAR_DEST}"
fi
ihdm_shadow_run "${RUN_DIR}" "${SHADOW}" || exit 1

RC=0
for DELTA in "${DELTA_LIST[@]}"; do
    CMD=("${PY}" -m ihdm.cli.evaluate_run
         --run "${SHADOW}"
         --ckpts "${STEP}"
         --device cuda
         --amp "${AMP}"
         --delta "${DELTA}"
         --final-from-lsd
         --sample-batch "${SAMPLE_BATCH}"
         --n-seeds "${N_SEEDS}"
         --n-per-seed "${N_PER_SEED}")
    if [[ -n "${EVAL_EXTRA_ARGS}" ]]; then
        echo "[warn] EVAL_EXTRA_ARGS='${EVAL_EXTRA_ARGS}': NOT the sweep counts (smoke only)"
        read -r -a EXTRA <<<"${EVAL_EXTRA_ARGS}"
        CMD+=("${EXTRA[@]}")
    fi
    echo ""
    echo "[cmd] ${CMD[*]}"
    T0=$(date +%s)
    "${CMD[@]}"
    rc=$?
    echo "DELTA ${DELTA}: exit=${rc} in $(( $(date +%s) - T0 )) s"
    (( rc == 0 )) || RC=${rc}
done

if (( RC == 0 )); then
    META=$(printf '{"job_id": "%s", "array_job_id": "%s", "task_id": "%s", "node": "%s", "git_sha": "%s", "dataset": "%s", "eval_extra_args": "%s"}' \
        "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_JOB_ID:-}" "${TASK_ID}" "$(hostname)" \
        "$(cat "${REPO_DIR}/GIT_SHA" 2>/dev/null || echo unknown)" "${DATASET_ID}" "${EVAL_EXTRA_ARGS}")
    "${PY}" "${REPO_DIR}/slurm/diag_eval/collect.py" task \
        --shadow "${SHADOW}" --run-id "${RUN_ID}" --amp "${AMP}" --deltas "${DELTAS}" \
        --step "${STEP}" --sigma "${SIGMA}" --meta "${META}" \
        --out "${MYLOCALSCRATCH}/${RUN_ID}_delta_sweep.json" \
        && ihdm_copy_atomic "${MYLOCALSCRATCH}/${RUN_ID}_delta_sweep.json" "${JSON_DEST}" \
        && echo "result:      ${JSON_DEST}" \
        || RC=3
fi

echo ""
echo "========== artefacts =========="
find "${SHADOW}" -name request.json 2>/dev/null | sort | while read -r f; do
    echo "  ${f#${SHADOW}/}: $("${PY}" -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['shape'], r.get('delta'), round(r.get('s_per_chain') or 0, 4), 's/chain')" "${f}")"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "END TASK     task=${TASK_ID} run_id=${RUN_ID} status=$( ((RC == 0)) && echo ok || echo FAILED) exit=${RC}"
echo "sacct:"
sacct -j "${SLURM_JOB_ID:-0}" -n -P -o JobID,State,Elapsed,MaxRSS,NodeList 2>&1 | head -n 5
echo "=========================================="

exit ${RC}
