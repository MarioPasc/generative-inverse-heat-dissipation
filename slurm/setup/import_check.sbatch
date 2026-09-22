#!/usr/bin/env bash
# Prove the experiment runs on an A100 node of Picasso (T3.1 step 4).
#
#   sbatch --test-only slurm/setup/import_check.sbatch    # always first
#   sbatch            slurm/setup/import_check.sbatch
#
# Five checks, each reported with its own exit status so one failure still leaves the others'
# evidence in the log:
#   1. nvidia-smi                      -- the node really has a GPU
#   2. torch version / CUDA / device   -- the PyPI wheel's bundled CUDA works with this driver
#   3. pytest -m "not integration"     -- the code base passes its own suite on the cluster
#   4. ihdm.cli.validate_dataset       -- the copied data still satisfies the format contract
#   5. train.py, 20 iterations         -- the trainer reaches `DONE` end to end
#
# A100 selection is `--constraint=a100` plus an UNTYPED `--gres=gpu:1`: exa[01-04] advertise
# `gpu:8` with the feature `a100`, so `--gres=gpu:A100:1` matches no node, and a bare
# `--constraint=dgx` would also match the B200 nodes.
#
# Step 5 needs `schedules/` (ticket T1.3) in the checkout: run this only after that merge.
#
#SBATCH -J ihdm-import-check
#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/import_check_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/import_check_%j.err

set -uo pipefail

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm}"
WORKDIR="${IHDM_RUN_ROOT}/import_check/ixi_A0_s1"

PY="${ENV_PREFIX}/bin/python"

echo "=========================================="
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Git branch:  $(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Data root:   ${IHDM_DATA_ROOT}"
echo "Workdir:     ${WORKDIR}"
echo "=========================================="

[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}; run create_env.sbatch first" >&2; exit 1; }

cd "${REPO_DIR}"
# Ahead of the editable install's .pth entry, so the library and the entry points always come
# from this checkout even if the env was installed from another one.
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# tests/train/conftest.py pins torch to few threads in the subprocesses it spawns; match it here
# so the suite does not oversubscribe the 8 cores this job reserved.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

FAILURES=()
run_step() {
    local name="$1"; shift
    echo ""
    echo "---------- ${name} ----------"
    if "$@"; then
        echo "[ok] ${name}"
    else
        local rc=$?
        echo "[FAIL] ${name} (exit ${rc})"
        FAILURES+=("${name}")
    fi
}

run_step "1. nvidia-smi" nvidia-smi

run_step "2. torch / CUDA / device" "${PY}" - <<'PYEOF'
import torch

print("torch          ", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available ", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch does not see a GPU on this node")
print("device name    ", torch.cuda.get_device_name(0))
print("capability     ", torch.cuda.get_device_capability(0))
print("total memory GB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok      ", bool(torch.isfinite(x @ x).all().item()))
PYEOF

run_step "3. pytest (not integration)" "${PY}" -m pytest -q -m "not integration"

run_step "4. validate_dataset ixi lsun_church" "${PY}" -m ihdm.cli.validate_dataset ixi lsun_church

mkdir -p "$(dirname "${WORKDIR}")"
run_step "5. train.py 20 iterations (ixi,A0)" "${PY}" train.py \
    --config configs/spectral/arms.py:ixi,A0 \
    --config.seed=1 \
    --config.training.n_iters=20 \
    --config.training.ckpt_every=10 \
    --config.training.log_every=5 \
    --config.training.eval_every=10 \
    --config.training.grid_every=10 \
    --config.training.batch_size=4 \
    --config.eval.batch_size=4 \
    --workdir "${WORKDIR}"

echo ""
echo "---------- run artefacts ----------"
ls -la "${WORKDIR}" 2>&1 || true
echo "--- DONE marker ---"
test -f "${WORKDIR}/DONE" && echo "DONE present" || echo "DONE MISSING"
echo "--- metrics.jsonl (last 5) ---"
tail -n 5 "${WORKDIR}/metrics.jsonl" 2>&1 || echo "no metrics.jsonl"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Duration:    $((ELAPSED / 60))m $((ELAPSED % 60))s"
if (( ${#FAILURES[@]} > 0 )); then
    echo "IMPORT CHECK FAILED: ${FAILURES[*]}"
    exit 1
fi
echo "IMPORT CHECK PASSED"
