#!/usr/bin/env bash
# T3.3 -- one training run of the 30-cell array.  One array task = one (dataset, arm, seed) cell of
# `slurm/array/cells.csv`, read by `$SLURM_ARRAY_TASK_ID`.  Submitted by
# `slurm/array/submit_array.sh`, which runs `sbatch --test-only` first and supplies `--array`,
# `--qos` and the log paths on the command line.
#
# Sizing comes from the T3.2 probe (D16, job 2405546, A100-SXM4-40GB): batch 16 peaks at 28.5 GB
# of VRAM and runs at 1.71 it/s, so 40,000 iterations take 6.84 h.  `--time=11:00:00` is 1.5x that;
# `--mem=32G` is host RAM, against the probe's 18.4 GB MaxRSS.  QOS `medium_uma` (3-day wall) is
# required: the probe's `short` caps at 2 h and would TIMEOUT every task.
#
# A100 selection is `--constraint=a100` plus an UNTYPED `--gres=gpu:1`: exa[01-04] advertise
# `gpu:8` with the feature `a100`, so `--gres=gpu:A100:1` matches no node, and a bare
# `--constraint=dgx` would also match the B200 nodes.
#
# Idempotence (D10).  The trainer resumes from `checkpoints-meta/checkpoint.pth` and its loop is
# `range(initial_step, n_iters + 1)`, so:
#   * resubmitting an index after a TIMEOUT continues that run where it stopped;
#   * resubmitting with N_ITERS=60000 extends every run without retraining anything;
#   * resubmitting a run that already reached N_ITERS is skipped here, before python starts, so a
#     blanket `--array=0-29` resubmission costs seconds per finished cell instead of a GPU hour.
#
# Output goes straight to FSCRATCH rather than `$LOCALSCRATCH`: a run writes ~40 files in 6.8 h
# (16 EMA checkpoints, `full_final.pt`, the rolling checkpoint, 16 grids, `metrics.jsonl`,
# `manifest.json`, `config.json`, tensorboard), which is not the thousands-of-small-files pattern
# the manual's §4.9 rule is about, and staging would break the resume rule above -- the trainer
# reads the previous submission's checkpoint out of the run directory.  T5.1's evaluation, which
# does write thousands of samples, uses `$LOCALSCRATCH`.
#
#SBATCH -J ihdm-train
#SBATCH --time=11:00:00
#SBATCH --qos=medium_uma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/train_%A_%a.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/train_%A_%a.err

set -uo pipefail

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm}"

CELLS="${IHDM_CELLS:-${REPO_DIR}/slurm/array/cells.csv}"
N_ITERS="${N_ITERS:-40000}"
PY="${ENV_PREFIX}/bin/python"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${IHDM_TASK_ID:-}}"

echo "=========================================="
echo "BEGIN TASK"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Array job:   ${SLURM_ARRAY_JOB_ID:-n/a}  task: ${TASK_ID:-n/a}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Git branch:  $(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Data root:   ${IHDM_DATA_ROOT}"
echo "Run root:    ${IHDM_RUN_ROOT}"
echo "Cells:       ${CELLS}"
echo "N_ITERS:     ${N_ITERS}"
echo "=========================================="

[[ -n "${TASK_ID}" ]] || { echo "[FATAL] no SLURM_ARRAY_TASK_ID (and no IHDM_TASK_ID)" >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}; run slurm/setup/create_env.sbatch" >&2; exit 1; }
[[ -f "${CELLS}" ]] || { echo "[FATAL] no cell table at ${CELLS}" >&2; exit 1; }

# ----------------------------------------------------------------------------------------------
# Decode the row.  Matched on the INDEX COLUMN, never on the line number, so a row that is ever
# added or removed cannot silently shift every cell by one.  The decoded tuple is echoed: a wrong
# decode produces a complete, plausible, wrong result set, which is the most expensive failure
# mode available here.
# ----------------------------------------------------------------------------------------------
ROW=$(awk -F, -v want="${TASK_ID}" 'NR > 1 && $1 == want {print; found = 1} END {exit !found}' "${CELLS}")
if [[ -z "${ROW}" ]]; then
    echo "[FATAL] no row with index ${TASK_ID} in ${CELLS}" >&2
    exit 1
fi
if [[ $(printf '%s\n' "${ROW}" | wc -l) -ne 1 ]]; then
    echo "[FATAL] index ${TASK_ID} matches more than one row in ${CELLS}" >&2
    exit 1
fi

IFS=, read -r CELL_INDEX RUN_ID DATASET_ID ARM SEED TIER <<<"${ROW}"
for field in CELL_INDEX RUN_ID DATASET_ID ARM SEED TIER; do
    [[ -n "${!field}" ]] || { echo "[FATAL] empty ${field} in row: ${ROW}" >&2; exit 1; }
done
[[ "${CELL_INDEX}" == "${TASK_ID}" ]] || { echo "[FATAL] decode mismatch: ${CELL_INDEX} != ${TASK_ID}" >&2; exit 1; }
[[ "${RUN_ID}" == "${DATASET_ID}_${ARM}_s${SEED}" ]] \
    || { echo "[FATAL] run_id ${RUN_ID} disagrees with ${DATASET_ID}/${ARM}/s${SEED}" >&2; exit 1; }

WORKDIR="${IHDM_RUN_ROOT}/${RUN_ID}"

echo "CELL         index=${CELL_INDEX} run_id=${RUN_ID} dataset=${DATASET_ID} arm=${ARM} seed=${SEED} tier=${TIER}"
echo "Workdir:     ${WORKDIR}"

# ----------------------------------------------------------------------------------------------
# Idempotence: skip a cell that has already reached N_ITERS.  `DONE` is an empty marker, so the
# step it was written at is read from the highest `ema_iter_XXXXXX.pt`, which `_finalise` writes
# at `final_step` before touching `DONE`.
# ----------------------------------------------------------------------------------------------
last_ema_step() {
    local d="$1/checkpoints" best=0 step name
    [[ -d "${d}" ]] || { echo 0; return; }
    for f in "${d}"/ema_iter_*.pt; do
        [[ -e "${f}" ]] || continue
        name="$(basename "${f}")"; name="${name#ema_iter_}"; name="${name%.pt}"
        step=$((10#${name}))
        (( step > best )) && best=${step}
    done
    echo "${best}"
}

if [[ -f "${WORKDIR}/DONE" ]]; then
    REACHED=$(last_ema_step "${WORKDIR}")
    if (( REACHED >= N_ITERS )); then
        echo "SKIP         ${RUN_ID} is DONE at step ${REACHED} >= N_ITERS=${N_ITERS}; nothing to do"
        echo "END TASK     index=${CELL_INDEX} run_id=${RUN_ID} status=skipped exit=0"
        echo "Finished:    $(date --iso-8601=seconds)"
        exit 0
    fi
    echo "EXTEND       ${RUN_ID} is DONE at step ${REACHED} < N_ITERS=${N_ITERS}; resuming to extend"
fi

# ----------------------------------------------------------------------------------------------
# Environment.  Same lines as the T3.2 probe worker, which is the configuration that has actually
# run on these nodes; `PYTHONPATH` goes ahead of the editable install's .pth entry so the library
# and the entry points always come from this checkout.
# ----------------------------------------------------------------------------------------------
cd "${REPO_DIR}" || exit 1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# The env ships no .pyc for sympy, triton, PIL, pip, ...: without this, the first task writes
# ~900 .pyc files and ~120 __pycache__ dirs into the env on FSCRATCH, which sits at its file
# quota (T3.4, measured 2026-09-25). Node-local, per job, never on FSCRATCH or $HOME.
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/ihdm_pycache_${SLURM_JOB_ID:-$$}"
echo "Pycache:     ${PYTHONPYCACHEPREFIX}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi -L || echo "[warn] nvidia-smi not available"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "[warn] nvidia-smi query not available"

mkdir -p "${IHDM_RUN_ROOT}"

# ----------------------------------------------------------------------------------------------
# Train.  Batch 16, lr 1e-4 (recipe v2, D19) and the cadences are frozen in
# `configs/spectral/arms.py` (D4", D4"', D5", D19) and are deliberately NOT overridden here; the
# only run-varying flags are the seed, the iteration count and the work directory.
#
# Exit codes of train.py (D19): 0 finished (or already past N_ITERS); 3 the non-finite-loss
# guard aborted (10 consecutive / >100 total skipped steps; the rolling checkpoint holds the
# weights that produced the last non-finite loss); 4 the recipe check refused to resume a run
# directory trained with another recipe (nothing was written); 1 any Python error; 137/143 the
# job was killed (TIMEOUT, scancel). See slurm/array/README.md.
# ----------------------------------------------------------------------------------------------
CMD=("${PY}" train.py
     --config "configs/spectral/arms.py:${DATASET_ID},${ARM}"
     --config.seed="${SEED}"
     --config.training.n_iters="${N_ITERS}"
     --workdir "${WORKDIR}")

echo ""
echo "[cmd] ${CMD[*]}"
echo ""

T0=$(date +%s)
"${CMD[@]}"
RC=$?
T1=$(date +%s)

# ----------------------------------------------------------------------------------------------
# Evidence and exit.  The exit code is propagated so `sacct` distinguishes COMPLETED from FAILED
# and the resubmission recipe in `slurm/array/README.md` can select on it.
# ----------------------------------------------------------------------------------------------
echo ""
echo "========== artefacts =========="
echo "DONE marker: $(test -f "${WORKDIR}/DONE" && echo present || echo MISSING)"
echo "EMA checkpoints: $(ls -1 "${WORKDIR}/checkpoints"/ema_iter_*.pt 2>/dev/null | wc -l) (last step $(last_ema_step "${WORKDIR}"))"
echo "last metrics line: $(tail -n 1 "${WORKDIR}/metrics.jsonl" 2>/dev/null || echo 'no metrics.jsonl')"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Train time:  $((T1 - T0)) s"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
case "${RC}" in
    0) STATUS=ok ;;
    3) STATUS=FAILED_nonfinite_guard ;;
    4) STATUS=FAILED_recipe_mismatch ;;
    *) STATUS=FAILED ;;
esac
echo "END TASK     index=${CELL_INDEX} run_id=${RUN_ID} status=${STATUS} exit=${RC}"
echo "sacct:"
sacct -j "${SLURM_JOB_ID:-0}" -n -P -o JobID,State,Elapsed,MaxRSS,NodeList 2>&1 | head -n 5
echo "=========================================="

exit ${RC}
