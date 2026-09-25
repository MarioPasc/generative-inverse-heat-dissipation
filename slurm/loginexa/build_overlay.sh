#!/usr/bin/env bash
# T3.4 -- build the V100-capable overlay of the cluster `ihdm` env, under $HOME (never FSCRATCH).
#
# Why: the cluster env ships torch 2.14.0+cu130, whose wheels carry no sm_70 kernels (CUDA 13
# dropped Volta), so nothing CUDA runs on loginexa's Tesla V100-DGXS-32GB. The overlay is a venv
# created with `--system-site-packages` on top of that env, holding only torch/torchvision
# 2.14.0+cu126 (the same torch version, built for sm_50..sm_90) and the CUDA 12 runtime wheels
# they pull in. Every other package (numpy, ml_collections, absl, the editable `ihdm`, ...) is
# read from the base env, so the overlay tests the same code against the same libraries; only the
# CUDA build of torch differs from what the A100 array runs.
#
# File budget: FSCRATCH is at its file quota, so the overlay lives in $HOME, whose soft limit is
# 35k files. `--no-compile` skips ~10k .pyc files, and the C++ headers of torch and of the CUDA
# wheels (never read at run time: no torch.compile, no cpp_extension here) are pruned.
#
# Run ON loginexa (it has internet; no GPU needed), detached, under the 25-minute rule:
#   ssh picasso 'ssh loginexa "nohup timeout 25m bash <repo>/slurm/loginexa/build_overlay.sh \
#       > ~/execs/ihdm/logs/loginexa/overlay_$(date +%Y%m%d_%H%M%S).log 2>&1 &"'
# Idempotent: re-running on a complete overlay only re-verifies it.
set -euo pipefail

# The base env holds no .pyc for pip, sympy, triton, PIL, yaml, absl, ...; without this line the
# first import writes ~900 of them (and ~120 __pycache__ dirs) INTO the base env on FSCRATCH,
# which is at its file quota (measured on the first build, 2026-09-25, and reverted).
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/ihdm_pycache_${USER}}"

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
BASE_ENV="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
OVERLAY="${IHDM_OVERLAY:-${USER_ROOT}/execs/ihdm/overlay/ihdm-v100}"
TORCH_SPEC="torch==2.14.0+cu126"
TORCHVISION_SPEC="torchvision==0.29.0+cu126"
INDEX_URL="https://download.pytorch.org/whl/cu126"

case "${OVERLAY}" in
    */fscratch/*) echo "[FATAL] the overlay must not live on FSCRATCH: ${OVERLAY}" >&2; exit 1 ;;
esac
[[ -x "${BASE_ENV}/bin/python" ]] || { echo "[FATAL] no base interpreter at ${BASE_ENV}/bin/python" >&2; exit 1; }

count_files() { find "$1" \( -type f -o -type l \) 2>/dev/null | wc -l; }

echo "== build_overlay.sh on $(hostname) at $(date --iso-8601=seconds)"
echo "base env: ${BASE_ENV}"
echo "overlay:  ${OVERLAY}"
echo "-- quota before"
quota 2>&1 || echo "[warn] quota not available on $(hostname)"

if [[ ! -x "${OVERLAY}/bin/python" ]]; then
    mkdir -p "$(dirname "${OVERLAY}")"
    # --without-pip: the base env's pip is visible through the system site-packages and installs
    # into the venv (sys.prefix), so a second copy of pip (~1k files) is not needed.
    "${BASE_ENV}/bin/python" -m venv --system-site-packages --without-pip "${OVERLAY}"
fi
PY="${OVERLAY}/bin/python"
SITE="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "venv site-packages: ${SITE}"

if ! "${PY}" -c 'import torch, sys; sys.exit(0 if torch.__version__ == "2.14.0+cu126" else 1)' 2>/dev/null; then
    echo "-- pip install ${TORCH_SPEC} ${TORCHVISION_SPEC}"
    PIP_NO_CACHE_DIR=1 "${PY}" -m pip install --no-compile --no-cache-dir --progress-bar off \
        "${TORCH_SPEC}" "${TORCHVISION_SPEC}" --index-url "${INDEX_URL}"
fi

echo "-- files in the overlay before pruning: $(count_files "${OVERLAY}")"
for d in "${SITE}/torch/include" "${SITE}/torch/share/cmake"; do
    [[ -d "${d}" ]] && { echo "prune ${d} ($(count_files "${d}") files)"; rm -rf "${d}"; }
done
if [[ -d "${SITE}/nvidia" ]]; then
    while IFS= read -r d; do
        echo "prune ${d} ($(count_files "${d}") files)"; rm -rf "${d}"
    done < <(find "${SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name include)
fi
echo "-- files in the overlay after pruning: $(count_files "${OVERLAY}")"
du -sh "${OVERLAY}"

echo "-- verify (CPU only; the GPU check is H1 of slurm/loginexa/harness.sh)"
"${PY}" - <<'EOF'
import sys
import torch, torchvision
print("python", sys.version.split()[0], "prefix", sys.prefix)
print("torch", torch.__version__, torch.__file__)
print("torchvision", torchvision.__version__, torchvision.__file__)
print("cuda build", torch.version.cuda, "arch list", torch.cuda.get_arch_list())
assert torch.__version__ == "2.14.0+cu126", torch.__version__
assert "/execs/ihdm/overlay/" in torch.__file__, torch.__file__
assert "sm_70" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
import numpy, ml_collections, absl  # noqa: F401  (from the base env)
print("numpy", numpy.__version__, numpy.__file__)
print("OVERLAY OK")
EOF
{
    echo "built $(date --iso-8601=seconds) on $(hostname) by slurm/loginexa/build_overlay.sh"
    echo "base ${BASE_ENV}"
    echo "${TORCH_SPEC} ${TORCHVISION_SPEC} from ${INDEX_URL}"
} > "${OVERLAY}/OVERLAY_INFO.txt"
echo "-- quota after"
quota 2>&1 || echo "[warn] quota not available on $(hostname)"
echo "== done at $(date --iso-8601=seconds)"
