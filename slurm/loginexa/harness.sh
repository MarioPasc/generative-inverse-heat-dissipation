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
h2)
    FIRST="${1:?first cell}"; LAST="${2:?last cell}"
    lx_banner "H2 real worker, cells ${FIRST}..${LAST}, N_ITERS=2"
    lx_pin_gpu "${GPU}" || exit 75
    H2_ROOT="${LX_SCRATCH}/h2_gpu${GPU}"
    lx_assert_not_fscratch "${H2_ROOT}" || exit 1
    for ((i = FIRST; i <= LAST; i++)); do
        echo "======== H2 cell ${i} ========"
        rm -rf "${H2_ROOT}"; mkdir -p "${H2_ROOT}"
        T0=$(date +%s)
        # The production worker, run with bash and the SLURM variables set by hand; only the
        # paths (repo, env, run root) point at the harness's tree, overlay and local disk.
        IHDM_REPO_DIR="${LX_REPO}" IHDM_ENV_PREFIX="${LX_OVERLAY}" IHDM_RUN_ROOT="${H2_ROOT}" \
            N_ITERS=2 SLURM_ARRAY_TASK_ID="${i}" SLURM_ARRAY_JOB_ID=h2 SLURM_JOB_ID="h2_${i}" \
            SLURM_CPUS_PER_TASK=4 timeout 6m bash slurm/array/train_array.sbatch \
            > "${H2_ROOT}.worker.log" 2>&1
        RC=$?
        grep -E '^(CELL|Workdir:|Pycache:|N_ITERS:|END TASK|DONE marker|EMA checkpoints|\[FATAL\])' "${H2_ROOT}.worker.log"
        grep -E 'Traceback|Error|error:' "${H2_ROOT}.worker.log" | grep -v FutureWarning | head -5
        RUN="$(find "${H2_ROOT}" -mindepth 1 -maxdepth 1 -type d | head -1)"
        if [[ -n "${RUN}" ]]; then
            "${LX_PY}" slurm/loginexa/check_run.py "${RUN}" --data-root "${IHDM_DATA_ROOT}" \
                --lr 1e-4 --label "H2[${i}]" \
                | grep -E '^(manifest|tail|PROBLEM|H2)|check_run' | cut -c1-400
        fi
        echo "H2 cell=${i} worker_rc=${RC} seconds=$(( $(date +%s) - T0 ))"
        rm -rf "${H2_ROOT}" "${H2_ROOT}.worker.log"
    done
    ;;
h3)
    SPEC="${1:?dataset,arm}"; RUN="${LX_SCRATCH}/h3/${SPEC/,/_}_s1"
    lx_banner "H3 ${RUN}"
    lx_pin_gpu "${GPU}" || exit 75
    lx_assert_not_fscratch "${RUN}" || exit 1
    rm -rf "${RUN}"; mkdir -p "$(dirname "${RUN}")"
    CMD=("${LX_PY}" train.py --config "configs/spectral/arms.py:${SPEC}" --config.seed=1
         --config.training.n_iters=300 --config.training.log_every=10
         --config.training.eval_every=50 --config.training.ckpt_every=100
         --config.training.resume_every=50 --config.training.grid_every=100 --workdir "${RUN}")
    echo "[cmd] ${CMD[*]}"
    T0=$(date +%s)
    timeout 21m "${CMD[@]}" > "${RUN}.train.log" 2>&1
    echo "H3 train rc=$? seconds=$(( $(date +%s) - T0 ))"
    grep -E 'Traceback|Error|OOM|out of memory' "${RUN}.train.log" | grep -v FutureWarning | head -5
    "${LX_PY}" slurm/loginexa/check_run.py "${RUN}" --data-root "${IHDM_DATA_ROOT}" --lr 1e-4 \
        --label H3 | cut -c1-600
    ;;
h4)
    SPEC="${1:?dataset,arm}"; RUN="${LX_SCRATCH}/h4/${SPEC/,/_}_s1"
    lx_banner "H4 resume / extension / recipe check on ${RUN}"
    lx_pin_gpu "${GPU}" || exit 75
    lx_assert_not_fscratch "${RUN}" || exit 1
    rm -rf "${RUN}"; mkdir -p "$(dirname "${RUN}")"
    BASE=("${LX_PY}" train.py --config "configs/spectral/arms.py:${SPEC}" --config.seed=1
          --config.training.log_every=10 --config.training.eval_every=50
          --config.training.ckpt_every=100 --config.training.resume_every=50
          --config.training.grid_every=100 --workdir "${RUN}")
    last_train() { awk -F'[:,]' '/"kind": "train"/ {s=$2} END {print s+0}' "${RUN}/metrics.jsonl" 2>/dev/null || echo 0; }
    saved_step() { "${LX_PY}" -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step']))" "${RUN}/checkpoints-meta/checkpoint.pth"; }
    snapshot() { "${LX_PY}" - "${RUN}" <<'EOF'
import hashlib, sys
from pathlib import Path
run = Path(sys.argv[1])
for p in sorted(run.rglob("*")):
    if p.is_file():
        print(p.relative_to(run), p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()[:16])
EOF
    }
    echo "---- H4a: start n_iters=300, SIGTERM once a train line past step 170 is written"
    "${BASE[@]}" --config.training.n_iters=300 > "${RUN}.a.log" 2>&1 &
    PID=$!
    until (( $(last_train) >= 170 )) || ! kill -0 "${PID}" 2>/dev/null; do sleep 5; done
    kill -TERM "${PID}"; wait "${PID}"; RCA=$?
    echo "H4a SIGTERM after train step $(last_train): rc=${RCA}"
    echo "H4a rolling checkpoint step: $(saved_step)"
    echo "---- H4b: the same command resumes"
    BEFORE=$(wc -l < "${RUN}/metrics.jsonl")
    timeout 9m "${BASE[@]}" --config.training.n_iters=300 > "${RUN}.b.log" 2>&1
    echo "H4b rc=$?"
    tail -n "+$((BEFORE + 1))" "${RUN}/metrics.jsonl" | head -n 2 | cut -c1-160
    "${LX_PY}" slurm/loginexa/check_run.py "${RUN}" --data-root "${IHDM_DATA_ROOT}" --lr 1e-4 \
        --label H4b | grep -E '^(PROBLEM|-- H-TRAIN §3)|check_run' | cut -c1-600
    echo "---- H4c: extension 300 -> 400"
    BEFORE=$(wc -l < "${RUN}/metrics.jsonl")
    timeout 6m "${BASE[@]}" --config.training.n_iters=400 > "${RUN}.c.log" 2>&1
    echo "H4c rc=$?"
    tail -n "+$((BEFORE + 1))" "${RUN}/metrics.jsonl" | head -n 2 | cut -c1-160
    "${LX_PY}" slurm/loginexa/check_run.py "${RUN}" --data-root "${IHDM_DATA_ROOT}" --lr 1e-4 \
        --label H4c | grep -E '^(PROBLEM|-- H-TRAIN §3)|check_run' | cut -c1-600
    for BAD in --config.optim.lr=5e-5 --config.training.batch_size=8 --config.seed=2; do
        echo "---- H4d: recipe mismatch ${BAD}"
        snapshot > "${RUN}.snap_before"
        timeout 4m "${BASE[@]}" --config.training.n_iters=500 "${BAD}" > "${RUN}.d.log" 2>&1
        RC=$?
        snapshot > "${RUN}.snap_after"
        grep -E 'Refusing to resume' "${RUN}.d.log" | cut -c1-400
        if cmp -s "${RUN}.snap_before" "${RUN}.snap_after"; then SAME=identical; else SAME=CHANGED; fi
        echo "H4d ${BAD} rc=${RC} run_dir=${SAME} files=$(wc -l < "${RUN}.snap_after")"
    done
    ;;
h5)
    R="${LX_SCRATCH}/h5"
    lx_banner "H5 skip and abort paths on the CUDA GradScaler"
    lx_pin_gpu "${GPU}" || exit 75
    lx_assert_not_fscratch "${R}" || exit 1
    rm -rf "${R}"; mkdir -p "${R}"
    INJ=("${LX_PY}" slurm/loginexa/train_inject.py --arm-spec ixi,A0
         --set training.log_every=5 --set training.eval_every=20 --set training.ckpt_every=20
         --set training.grid_every=20 --set training.resume_every=10)
    events() { "${LX_PY}" - "$1" <<'EOF'
import json, sys
for line in open(sys.argv[1] + "/metrics.jsonl"):
    r = json.loads(line)
    if r["kind"] in ("skip", "abort", "done", "resume") or (r["kind"] == "train" and r["step"] >= 15):
        keep = {k: r[k] for k in ("step", "kind", "loss", "grad_norm", "amp_scale", "n_skipped",
                                   "consecutive", "reason", "resume_saved") if k in r}
        print("  ", json.dumps(keep))
EOF
    }
    echo "---- H5a: two isolated non-finite losses (steps 25, 33) under the enabled scaler"
    timeout 6m "${INJ[@]}" --set training.n_iters=40 --inject 25,33 --workdir "${R}/skip" > "${R}/a.log" 2>&1
    echo "H5a rc=$? (expect 0)"; events "${R}/skip"
    "${LX_PY}" slurm/loginexa/check_run.py "${R}/skip" --lr 1e-4 --allow-skips --label H5a \
        | grep -E '^PROBLEM|check_run'
    echo "---- H5b: ten consecutive non-finite losses (steps 12-21)"
    timeout 6m "${INJ[@]}" --set training.n_iters=40 --inject 12-21 --workdir "${R}/abort" > "${R}/b.log" 2>&1
    echo "H5b rc=$? (expect 3)"; events "${R}/abort"
    "${LX_PY}" -c "import torch,sys; print('H5b rolling checkpoint step', int(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step']))" "${R}/abort/checkpoints-meta/checkpoint.pth"
    echo "---- H5c: resume H5b without injection to n_iters=30 (the 10 skips carry over)"
    timeout 6m "${INJ[@]}" --set training.n_iters=30 --workdir "${R}/abort" > "${R}/c.log" 2>&1
    echo "H5c rc=$? (expect 0)"; events "${R}/abort" | tail -n 3
    echo "---- H5d: disabled scaler (optim.automatic_mp=false, fp32, batch 4): abort at once"
    timeout 6m "${INJ[@]}" --set optim.automatic_mp=false --set training.batch_size=4 \
        --set eval.batch_size=4 --set training.n_iters=15 --inject 12 --workdir "${R}/noamp" > "${R}/d.log" 2>&1
    echo "H5d rc=$? (expect 3)"; events "${R}/noamp"
    "${LX_PY}" -c "import torch,sys; s=torch.load(sys.argv[1], map_location='cpu', weights_only=True); print('H5d rolling checkpoint step', int(s['step']), 'finite', all(bool(torch.isfinite(t).all()) for t in s['model'].values() if t.is_floating_point()))" "${R}/noamp/checkpoints-meta/checkpoint.pth"
    grep -hE 'Traceback|Error' "${R}"/*.log | grep -v FutureWarning | head -5
    ;;
h6)
    RUN="${1:?H3 run dir}"
    lx_banner "H6 downstream on ${RUN}"
    lx_pin_gpu "${GPU}" || exit 75
    lx_assert_not_fscratch "${RUN}" || exit 1
    timeout 3m "${LX_PY}" - "${RUN}" <<'EOF'
import sys
from pathlib import Path
import torch
from ihdm.sampling.loader import load_ema_model, load_run_config
run = Path(sys.argv[1])
config = load_run_config(run)
for step in (200, 300):
    model = load_ema_model(run / "checkpoints" / f"ema_iter_{step:06d}.pt", config, "cuda")
    x = torch.rand(2, 1, 192, 192, device="cuda")
    with torch.no_grad():
        out = model(x, torch.tensor([1, 200], device="cuda"))
    print(f"H6 load_ema_model step={step} training={model.training} "
          f"n_params={sum(p.numel() for p in model.parameters())} out_finite={bool(torch.isfinite(out).all())}")
EOF
    EVAL=("${LX_PY}" -m ihdm.cli.evaluate_run --run "${RUN}" --ckpts 200,300 --n-lsd 16
          --n-final 16 --n-seeds 2 --n-per-seed 3 --skip-inception)
    T0=$(date +%s)
    timeout 12m "${EVAL[@]}" > "${RUN}.eval.log" 2>&1
    echo "H6 evaluate_run rc=$? seconds=$(( $(date +%s) - T0 ))"; tail -n 3 "${RUN}.eval.log" | cut -c1-300
    T0=$(date +%s)
    timeout 6m "${EVAL[@]}" --gate 200,300 > "${RUN}.gate.log" 2>&1
    echo "H6 evaluate_run --gate rc=$? seconds=$(( $(date +%s) - T0 ))"; tail -n 3 "${RUN}.gate.log" | cut -c1-300
    grep -hE 'Traceback|Error' "${RUN}.eval.log" "${RUN}.gate.log" | head -5
    "${LX_PY}" - "${RUN}" <<'EOF'
import json, math, sys
from pathlib import Path
def walk(node, path, bad):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}.{k}", bad)
    elif isinstance(node, list):
        for i, v in enumerate(node[:1000]):
            walk(v, f"{path}[{i}]", bad)
    elif node is None or (isinstance(node, float) and not math.isfinite(node)):
        bad.append(path)
def reject(token):
    raise ValueError(f"non-strict token {token}")
for f in sorted((Path(sys.argv[1]) / "metrics").glob("*.json")):
    bad = []
    try:
        data = json.loads(f.read_text(), parse_constant=reject)
    except ValueError as error:
        print(f"H6 {f.name}: INVALID {error}")
        continue
    walk(data, f.stem, bad)
    keys = sorted(data) if isinstance(data, dict) else type(data).__name__
    print(f"H6 {f.name}: {len(json.dumps(data))} bytes, top keys {keys}, null/nan leaves {len(bad)} {bad[:8]}")
EOF
    ;;
diag)
    FIX="${1:?fixture dir}"; shift
    lx_banner "diagnose_nan ${FIX}"
    lx_pin_gpu "${GPU}" || exit 75
    OUT="${LX_LOGS}/diag_$(basename "${FIX}").json"
    timeout 23m "${LX_PY}" -m ihdm.cli.diagnose_nan --fixture "${FIX}" --out "${OUT}" "$@"
    echo "diag rc=$?"
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
