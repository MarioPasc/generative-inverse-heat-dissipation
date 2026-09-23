#!/usr/bin/env bash
# Build the `ihdm` conda environment on Picasso (T3.1 step 2).
#
#   sbatch --test-only slurm/setup/create_env.sbatch     # always first
#   sbatch            slurm/setup/create_env.sbatch
#
# CPU job, not an interactive login-node build: the login node kills anything above 10 min of
# CPU and this takes tens of minutes. `--constraint=download` pins the one node that has
# outbound internet (`picasso4`, the only node advertising the `download` feature); a generic
# `cpu` node cannot be assumed to reach conda-forge or PyPI.
#
# Re-running with IHDM_ENV_RECREATE=1 deletes and rebuilds the prefix; otherwise an existing
# prefix is left alone and only `pip install -e .` is repeated.
#
#SBATCH -J ihdm-create-env
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=download
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/create_env_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/create_env_%j.err

set -euo pipefail

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
CONDA_SH="/mnt/home/soft/miniconda/programs/x86_64/miniconda3_py310_23.1.0/etc/profile.d/conda.sh"
RECREATE="${IHDM_ENV_RECREATE:-0}"

echo "=========================================="
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "Git branch:  $(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

[[ -f "${REPO_DIR}/environment.yml" ]] || {
    echo "[FATAL] no environment.yml at ${REPO_DIR}; clone the repo first (see slurm/setup/README.md)" >&2
    exit 1
}

# Package caches must not land in $HOME: its file quota is the tightest on the cluster and a
# torch environment unpacks tens of thousands of files. Keep every cache on FSCRATCH and clean
# it at the end, so only the prefix itself is durable.
export CONDA_PKGS_DIRS="${USER_ROOT}/fscratch/conda_pkgs"
export PIP_CACHE_DIR="${USER_ROOT}/fscratch/pip_cache"
mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"

# shellcheck source=/dev/null
source "${CONDA_SH}"

if [[ "${RECREATE}" == "1" && -d "${ENV_PREFIX}" ]]; then
    echo "[env] IHDM_ENV_RECREATE=1: removing ${ENV_PREFIX}"
    conda env remove -y -p "${ENV_PREFIX}"
fi

if [[ -d "${ENV_PREFIX}" ]]; then
    echo "[env] prefix already exists; skipping 'conda env create' (set IHDM_ENV_RECREATE=1 to rebuild)"
else
    echo "[env] conda env create -f ${REPO_DIR}/environment.yml -p ${ENV_PREFIX}"
    conda env create -f "${REPO_DIR}/environment.yml" -p "${ENV_PREFIX}"
fi

# The environment's own interpreter, never `conda activate` + bare `pip`: activation in a
# non-interactive batch shell is the classic way of installing into the wrong environment.
PY="${ENV_PREFIX}/bin/python"
[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}" >&2; exit 1; }

echo "[pip] editable install of the repository"
"${PY}" -m pip install --no-input --no-cache-dir -e "${REPO_DIR}"

echo "[clean] conda clean --all"
conda clean --all --yes >/dev/null

echo "=========================================="
echo "python:      $("${PY}" --version 2>&1)"
"${PY}" - <<'PYEOF'
import importlib.metadata as md

import torch

print(f"torch:       {torch.__version__} (bundled CUDA {torch.version.cuda})")
print(f"cuda avail:  {torch.cuda.is_available()}  (False on a CPU node is expected)")
for pkg in ("numpy", "SimpleITK", "nibabel", "ml_collections", "absl-py", "opencv-python-headless"):
    try:
        print(f"{pkg:<12} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:<12} MISSING")
PYEOF

echo "[check] the repository imports from the new environment"
"${PY}" -c "import ihdm, ihdm.data.format, ihdm.cli.validate_dataset; print('ihdm at', ihdm.__file__)"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "Env prefix:  ${ENV_PREFIX}"
