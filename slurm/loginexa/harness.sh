#!/usr/bin/env bash
# T3.4 -- the loginexa harness of the training/logging path (ticket §5, docs/RESULTS/loginexa_harness.md).
# Runs ON loginexa, one item per invocation, as one detached process under `timeout 25m`:
#
#   ssh picasso 'ssh loginexa "nohup timeout 25m bash <repo>/slurm/loginexa/harness.sh <item> <gpu> [args] \
#        > ~/execs/ihdm/logs/loginexa/<item>_<stamp>.log 2>&1 < /dev/null &"'
#
# (`slurm/loginexa/launch.sh` builds that line from the workstation.) Every python step inside has
# its own shorter timeout so the item can still print its verdict. Items:
#
#   h1 <gpu>                          environment: V100/sm_70, 5 production train steps at batch 16, pytest tests/train
#   h2 <gpu> <first> <last>           the real worker slurm/array/train_array.sbatch on cells first..last, N_ITERS=2
#   h3 <gpu> <dataset,arm>            300 iterations, shortened cadence, full artefact + schema check
#   h4 <gpu> <dataset,arm>            SIGTERM mid-run and resume; extension 300 -> 400; recipe-mismatch refusal
#   h5 <gpu>                          skip path and abort path on the real CUDA GradScaler (injected NaN)
#   h6 <gpu> <run_dir>                load_ema_model + ihdm.cli.evaluate_run on the H3 checkpoints
#   s <gpu> <dataset,arm> <seed> [lr] one 24-minute session of stability check S (resumes; n_iters 4000)
#   s-status <gpu> <dataset,arm> <seed> [lr]   DONE | RUNNING | FAILED | BUSY | IDLE (for chain_s.sh)
#   diag <gpu> <fixture_dir>          ihdm.cli.diagnose_nan on an array-1 fixture
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm/loginexa/common.sh
source "${HERE}/common.sh"

ITEM="${1:?item}"
GPU="${2:?gpu index}"
shift 2
cd "${LX_REPO}" || exit 1

s_paths() {  # sets S_SPEC S_SEED S_LR S_TAG S_RUN_ID S_WORKDIR S_SESSIONS
    S_SPEC="$1"; S_SEED="$2"; S_LR="${3:-}"
    S_TAG="lr${S_LR:-1e-4}"
    S_RUN_ID="${S_SPEC/,/_}_s${S_SEED}"
    S_WORKDIR="${LX_S_ROOT}/S_${S_TAG}/${S_RUN_ID}"
    S_SESSIONS="${LX_LOGS}/S_${S_TAG}_${S_RUN_ID}.sessions"
}

case "${ITEM}" in
h1)
    lx_banner "H1 environment"
    lx_pin_gpu "${GPU}" || exit 75
    nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,memory.used --format=csv
    timeout 10m "${LX_PY}" slurm/loginexa/h1_env.py
    echo "H1 h1_env.py exit=$?"
    echo "---- pytest tests/train (the overlay python) ----"
    timeout 12m "${LX_PY}" -m pytest -q -p no:cacheprovider tests/train 2>&1 | tail -n 15
    echo "H1 pytest exit=${PIPESTATUS[0]}"
    ;;
s)
    s_paths "$@"
    lx_banner "S session ${S_RUN_ID} (${S_TAG})"
    lx_assert_not_fscratch "${S_WORKDIR}" || exit 1
    if [[ -f "${S_WORKDIR}/DONE" ]]; then echo "S ${S_RUN_ID} already DONE"; exit 0; fi
    lx_pin_gpu "${GPU}" || exit 75
    mkdir -p "$(dirname "${S_WORKDIR}")"
    # Production recipe (configs/spectral/arms.py); only n_iters (stop at 4000) and resume_every
    # (lose <= 100 steps per 25-minute session) differ, both outside the recipe check (D19).
    CMD=("${LX_PY}" train.py --config "configs/spectral/arms.py:${S_SPEC}"
         --config.seed="${S_SEED}" --config.training.n_iters=4000
         --config.training.resume_every=100 --workdir "${S_WORKDIR}")
    [[ -n "${S_LR}" ]] && CMD+=(--config.optim.lr="${S_LR}")
    echo "[cmd] ${CMD[*]}"
    T0=$(date +%s)
    timeout 24m "${CMD[@]}"
    RC=$?
    echo "S_SESSION_END run=${S_RUN_ID} rc=${RC} seconds=$(( $(date +%s) - T0 ))"
    echo "$(date --iso-8601=seconds) rc=${RC} last=$(tail -n 1 "${S_WORKDIR}/metrics.jsonl" 2>/dev/null | cut -c1-120)" >> "${S_SESSIONS}"
    ;;
s-status)
    s_paths "$@"
    if [[ -f "${S_WORKDIR}/DONE" ]]; then echo "DONE ${S_RUN_ID}"; exit 0; fi
    if pgrep -f -- "--workdir ${S_WORKDIR}" > /dev/null; then echo "RUNNING ${S_RUN_ID}"; exit 0; fi
    LAST_RC=$(tail -n 1 "${S_SESSIONS}" 2>/dev/null | sed -n 's/.* rc=\([0-9]*\) .*/\1/p')
    if [[ -n "${LAST_RC}" && "${LAST_RC}" != 0 && "${LAST_RC}" != 124 && "${LAST_RC}" != 75 ]]; then
        echo "FAILED ${S_RUN_ID} rc=${LAST_RC}"; exit 0
    fi
    if ! lx_gpu_free "${GPU}"; then echo "BUSY ${S_RUN_ID} gpu=${GPU}"; exit 0; fi
    echo "IDLE ${S_RUN_ID}"
    ;;
*)
    echo "unknown item ${ITEM}" >&2; exit 2 ;;
esac
