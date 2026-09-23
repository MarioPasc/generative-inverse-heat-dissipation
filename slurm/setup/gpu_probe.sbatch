#!/usr/bin/env bash
# Read the A100 driver and decide whether the installed torch wheel runs on it (T3.1, extra).
#
#   sbatch --test-only slurm/setup/gpu_probe.sbatch
#   sbatch            slurm/setup/gpu_probe.sbatch
#
# `environment.yml` pins only `torch>=2.4`, so pip resolves whatever is newest at build time.
# A wheel built against a CUDA major version the exa nodes' driver predates leaves
# `torch.cuda.is_available()` False and every training run dead. That question cannot be answered
# from the login node (no `nvidia-smi` there), and the import check that would answer it is
# blocked on ticket T1.3, so this one-minute job answers it on its own and lets a cu12x rebuild
# overlap with T1.3 rather than queue behind it.
#
# A100 selection: `--constraint=a100` plus an UNTYPED `--gres=gpu:1`.
#
#SBATCH -J ihdm-gpu-probe
#SBATCH --time=00:05:00
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/gpu_probe_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/gpu_probe_%j.err

set -uo pipefail

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
PY="${ENV_PREFIX}/bin/python"

echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date --iso-8601=seconds)"
echo ""

echo "---------- nvidia-smi ----------"
nvidia-smi
echo ""
echo "driver/gpu (csv):"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "---------- torch ----------"
"${PY}" - <<'PYEOF'
import torch

print("torch            ", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available   ", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch does not see the GPU -- the driver rejects this wheel's CUDA")
print("device name      ", torch.cuda.get_device_name(0))
print("capability       ", torch.cuda.get_device_capability(0))
print("total memory GB  ", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
a = torch.randn(2048, 2048, device="cuda:0")
print("matmul finite    ", bool(torch.isfinite(a @ a).all().item()))
print("PROBE PASSED")
PYEOF
rc=$?

echo ""
echo "Finished: $(date --iso-8601=seconds)"
exit "${rc}"
