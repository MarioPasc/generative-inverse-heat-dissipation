#!/usr/bin/env bash
# T5.1 -- evaluation of one training run of the 30-cell array.  One array task = one row of
# `slurm/array/cells.csv`, read by `$SLURM_ARRAY_TASK_ID` (the same index as the training task).
# Submitted by `slurm/eval/submit_eval.sh array`, which supplies `--array`, `--time`,
# `--dependency` and the log paths on the command line.
#
# Per task:
#   1. decode the cell; refuse a run without `DONE`, or whose largest EMA checkpoint is not
#      N_ITERS (T3.5: the evaluated steps follow the run length, 5k..N_ITERS every 5k);
#   2. refuse a dataset without the six one-writer files of `prepare_eval` (the task never writes
#      a seed list or a reference feature matrix);
#   3. stage clean-fid's weights to /tmp;
#   4. build a shadow run on $LOCALSCRATCH (links to config.json, manifest.json, checkpoints/,
#      grids/, metrics.jsonl of the real run), so `samples*/` and `metrics*/` land on local disk;
#      unpack every gate tar of this run (the legacy 35k/40k name and every pair-named one, T3.5)
#      and any earlier partial tar, so a resubmission after a TIMEOUT reuses every sample set
#      already drawn (the cache is signature-checked);
#   5. `evaluate_run --ckpts all` with the D16 counts (the CLI defaults) and `--amp ${AMP}`;
#   6. return ONE tar of the shadow run to ~/execs/ihdm/eval/<run_id><amp>.tar plus a plain copy
#      of its summary.json at ~/execs/ihdm/eval/<run_id><amp>_summary.json, both atomically.
# The tar is also written on TIMEOUT/SIGTERM (trap), so the next submission resumes.  Nothing per
# run is written on FSCRATCH.
#
# Sizing (docs/RESULTS/evaluation_plan.md §3, §9): (N_ITERS / 5000) x 500 + 2,000 + 40 x 50 chains
# per run (8,000 at 40k, 10,000 at 60k) plus the Inception pass; --time is computed from those
# counts by submit_eval.sh (measured s/chain, + 0.25 h, x 1.3, rounded up to 15 min).
#
#SBATCH -J ihdm-eval
#SBATCH --time=10:00:00
#SBATCH --qos=medium_uma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_%A_%a.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_%A_%a.err

set -uo pipefail
# Before ANY python: the env on FSCRATCH has no .pyc, and the first import would write ~900 there.
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/ihdm_pycache_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm}"
EVAL_HOME="${IHDM_EVAL_HOME:-${USER_ROOT}/execs/ihdm/eval}"
CELLS="${IHDM_CELLS:-${REPO_DIR}/slurm/array/cells.csv}"
N_ITERS="${N_ITERS:-40000}"
AMP="${AMP:-off}"
FORCE_EVAL="${FORCE_EVAL:-0}"
# Extra evaluate_run flags, for a LOCAL dry run with tiny counts only; the array uses the D16
# defaults of the CLI and submit_eval.sh never sets this.
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
PY="${ENV_PREFIX}/bin/python"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${IHDM_TASK_ID:-}}"
MYLOCALSCRATCH="${LOCALSCRATCH:-/tmp}/${USER}/eval_${SLURM_JOB_ID:-local}_${TASK_ID:-x}"

echo "=========================================="
echo "BEGIN TASK"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Array job:   ${SLURM_ARRAY_JOB_ID:-n/a}  task: ${TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || cat "${REPO_DIR}/.git_sha" 2>/dev/null || echo n/a)"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Data root:   ${IHDM_DATA_ROOT}"
echo "Run root:    ${IHDM_RUN_ROOT}  (read only)"
echo "Eval home:   ${EVAL_HOME}"
echo "Cells:       ${CELLS}"
echo "N_ITERS:     ${N_ITERS}   AMP: ${AMP}"
echo "Local:       ${MYLOCALSCRATCH}"
echo "=========================================="

[[ -n "${TASK_ID}" ]] || { echo "[FATAL] no SLURM_ARRAY_TASK_ID (and no IHDM_TASK_ID)" >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}" >&2; exit 1; }
[[ -f "${CELLS}" ]] || { echo "[FATAL] no cell table at ${CELLS}" >&2; exit 1; }
# shellcheck source=slurm/eval/common.sh
source "${REPO_DIR}/slurm/eval/common.sh" || { echo "[FATAL] cannot source common.sh" >&2; exit 1; }

ihdm_decode_cell "${TASK_ID}" "${CELLS}" || exit 1
SUFFIX=$(ihdm_amp_suffix "${AMP}") || exit 1
RUN_DIR="${IHDM_RUN_ROOT}/${RUN_ID}"
SHADOW="${MYLOCALSCRATCH}/${RUN_ID}"
TAR_DEST="${EVAL_HOME}/${RUN_ID}${SUFFIX}.tar"
SUMMARY_DEST="${EVAL_HOME}/${RUN_ID}${SUFFIX}_summary.json"
GATE_DIR="${EVAL_HOME}/gate"
echo "Run dir:     ${RUN_DIR}"
echo "Returns:     ${TAR_DEST} + ${SUMMARY_DEST}"

if [[ -f "${SUMMARY_DEST}" && "${FORCE_EVAL}" != "1" ]]; then
    echo "SKIP         ${SUMMARY_DEST} exists; set FORCE_EVAL=1 to re-evaluate"
    echo "END TASK     index=${CELL_INDEX} run_id=${RUN_ID} status=skipped exit=0"
    exit 0
fi

# The run must be finished at N_ITERS: DONE, the EMA checkpoint of N_ITERS present, and no later
# one (an extended run evaluated with a stale N_ITERS would silently score a different range).
ihdm_check_n_iters "${N_ITERS}" || exit 1
LAST_CKPT="${RUN_DIR}/checkpoints/ema_iter_$(printf '%06d' "${N_ITERS}").pt"
[[ -f "${RUN_DIR}/DONE" ]] || { echo "[FATAL] ${RUN_DIR}/DONE is missing; the run has not finished" >&2; exit 1; }
[[ -f "${LAST_CKPT}" ]] || { echo "[FATAL] ${LAST_CKPT} is missing" >&2; exit 1; }
LAST_STEP=$(ihdm_last_ema_step "${RUN_DIR}")
(( LAST_STEP == N_ITERS )) || { echo "[FATAL] largest EMA checkpoint is ${LAST_STEP}, not N_ITERS=${N_ITERS}" >&2; exit 1; }
ihdm_check_eval_caches "${IHDM_DATA_ROOT}/${DATASET_ID}" || exit 1

mkdir -p "${MYLOCALSCRATCH}" "${EVAL_HOME}"

FINALIZED=0
finalize() {
    [[ "${FINALIZED}" == 1 ]] && return 0
    FINALIZED=1
    # Mirror the WHOLE shadow tree (links are stored as links); partial results come back too,
    # so a resubmission resumes from them.
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

# ---- shadow run, resumed from earlier partial results ----------------------------------------
# Gate tars first (legacy name, then pair-named in step order), this run's partial tar last.
mapfile -t GATE_TARS < <(ihdm_gate_tars "${GATE_DIR}" "${RUN_ID}" "${SUFFIX}")
for previous in "${GATE_TARS[@]}" "${TAR_DEST}"; do
    if [[ -f "${previous}" ]]; then
        tar -xf "${previous}" -C "${MYLOCALSCRATCH}" && echo "resume:      unpacked ${previous}"
    fi
done
ihdm_shadow_run "${RUN_DIR}" "${SHADOW}" || exit 1

CMD=("${PY}" -m ihdm.cli.evaluate_run
     --run "${SHADOW}"
     --ckpts all
     --device cuda
     --amp "${AMP}")
if [[ -n "${EVAL_EXTRA_ARGS}" ]]; then
    echo "[warn] EVAL_EXTRA_ARGS='${EVAL_EXTRA_ARGS}': NOT the D16 counts (local dry run only)"
    read -r -a EXTRA <<<"${EVAL_EXTRA_ARGS}"
    CMD+=("${EXTRA[@]}")
fi
echo ""
echo "[cmd] ${CMD[*]}"
echo ""
T0=$(date +%s)
"${CMD[@]}"
RC=$?
T1=$(date +%s)

METRICS="${SHADOW}/metrics${SUFFIX}"
if (( RC == 0 )) && [[ -f "${METRICS}/summary.json" ]]; then
    ihdm_copy_atomic "${METRICS}/summary.json" "${SUMMARY_DEST}" && echo "summary:     ${SUMMARY_DEST}"
fi

echo ""
echo "========== artefacts =========="
echo "metrics files: $(ls -1 "${METRICS}" 2>/dev/null | paste -sd' ' -)"
echo "sample sets:   $(find "${SHADOW}/samples${SUFFIX}" -name samples.npy 2>/dev/null | wc -l)"
find "${SHADOW}/samples${SUFFIX}" -name request.json 2>/dev/null | sort | while read -r f; do
    echo "  ${f#${SHADOW}/}: $("${PY}" -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['shape'], round(r.get('s_per_chain') or 0, 4), 's/chain')" "${f}")"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Eval time:   $((T1 - T0)) s"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "END TASK     index=${CELL_INDEX} run_id=${RUN_ID} status=$( ((RC == 0)) && echo ok || echo FAILED) exit=${RC}"
echo "sacct:"
sacct -j "${SLURM_JOB_ID:-0}" -n -P -o JobID,State,Elapsed,MaxRSS,NodeList 2>&1 | head -n 5
echo "=========================================="

exit ${RC}
