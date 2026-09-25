# shellcheck shell=bash
# T3.4 -- environment shared by every loginexa script (sourced, never run).
#
# loginexa: 4x Tesla V100-DGXS-32GB (sm_70), no SLURM, a 30-minute limit per session, shared with
# other users. The rules, from the ticket and the `test-picasso-loginexa` skill:
#   * the python is the V100 overlay (slurm/loginexa/build_overlay.sh), by absolute path;
#   * every process runs under `timeout 25m`;
#   * a GPU is used only when `nvidia-smi` shows < 1000 MiB on it, pinned with
#     CUDA_VISIBLE_DEVICES, at most two GPUs at once, never one another user holds;
#   * nothing is written on FSCRATCH: data are read from it, .pyc files go to local /tmp
#     (PYTHONPYCACHEPREFIX; the base env holds none for sympy, triton, pip, ...), run directories
#     go to loginexa's local /tmp or to $HOME.

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
LX_REPO="${LX_REPO:-${USER_ROOT}/execs/ihdm/wt/T3.4}"
LX_OVERLAY="${LX_OVERLAY:-${USER_ROOT}/execs/ihdm/overlay/ihdm-v100}"
LX_PY="${LX_OVERLAY}/bin/python"
LX_LOGS="${LX_LOGS:-${USER_ROOT}/execs/ihdm/logs/loginexa}"
LX_SCRATCH="${LX_SCRATCH:-/tmp/ihdm_T3.4}"          # loginexa's local disk
LX_S_ROOT="${LX_S_ROOT:-${USER_ROOT}/execs/ihdm/loginexa_runs}"   # multi-session S runs

export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export PYTHONPATH="${LX_REPO}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${LX_SCRATCH}/pycache"
export MPLCONFIGDIR="${LX_SCRATCH}/mplconfig"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# V100 accommodation, not a recipe change: batch 16 peaks at 27.9 GiB allocated on a 31.7 GiB
# card (H1), and the first H1 run logged a caching-allocator OOM-and-retry at the eval step.
# Expandable segments cut fragmentation; the numerics and max_memory_allocated are unchanged.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${LX_LOGS}" "${LX_SCRATCH}" "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

# Refuse any output path on FSCRATCH (the v2 array's run root and main's clone live there).
lx_assert_not_fscratch() {
    local p
    for p in "$@"; do
        case "$(readlink -m "${p}")" in
            */fscratch/*|/mnt2/fscratch/*)
                echo "[FATAL] refusing to write on FSCRATCH: ${p}" >&2; return 1 ;;
        esac
    done
}

# lx_gpu_free <index>: 0 if the GPU shows < 1000 MiB used (nobody on it), 1 otherwise.
lx_gpu_free() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ')
    [[ -n "${used}" ]] && (( used < 1000 ))
}

# lx_pin_gpu <index>: export CUDA_VISIBLE_DEVICES=<index> if that GPU is free, else fail.
lx_pin_gpu() {
    if ! lx_gpu_free "$1"; then
        echo "[FATAL] GPU $1 is not free:" >&2
        nvidia-smi --query-gpu=index,memory.used --format=csv >&2
        return 1
    fi
    export CUDA_VISIBLE_DEVICES="$1"
    echo "GPU:         $1 ($(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader -i "$1"))"
}

lx_banner() {
    echo "=========================================="
    echo "$1 on $(hostname) at $(date --iso-8601=seconds)"
    echo "repo:        ${LX_REPO} ($(cat "${LX_REPO}/.t34_sha" 2>/dev/null || echo 'no .t34_sha'))"
    echo "python:      ${LX_PY}"
    echo "data root:   ${IHDM_DATA_ROOT}"
    echo "pycache:     ${PYTHONPYCACHEPREFIX}"
    echo "=========================================="
}
