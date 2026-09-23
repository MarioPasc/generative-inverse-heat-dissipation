#!/usr/bin/env bash
# T3.2 -- Picasso A100 probe: three epochs on two datasets, memory at batch 24, resume, sampler
# timing.  Submitted by `slurm/probe/submit_probe.sh` (which runs `sbatch --test-only` first).
#
# Six sequential steps in one job (SCBI asks for jobs of >= 2 h; this one is sized at 1h50 and is
# a single pre-array probe, not a campaign):
#   1. lsun_church,A0 seed 1, batch 16, 600 iterations  (3 epochs of 3200 images)
#   2. ixi,A0         seed 1, batch 16, 600 iterations
#   3. memory probe:  ixi,A0 at batch 24, 40 iterations -- T2.3's affine fit predicts 42.2 GB,
#      i.e. an OOM on a 40 GB A100.  The step is wrapped so the OOM does NOT abort the worker.
#   4. resume/extension: step 2's command again with n_iters=800 -> `resume` event at 600 and
#      `ema_iter_000800.pt`.
#   5. offline sampler on `ema_iter_000800.pt` at sampling batch 32, 64 and 128.
#   6. artefact listing, tail of every metrics.jsonl, GPU memory trace, sacct.
#
# A100 selection is `--constraint=a100` plus an UNTYPED `--gres=gpu:1`: exa[01-04] advertise
# `gpu:8` with the feature `a100`, so `--gres=gpu:A100:1` matches no node, and a bare
# `--constraint=dgx` would also match the B200 nodes.
#
# Every step keeps going after a failure and its exit status is reported at the end, so one OOM
# still leaves the other five steps' evidence in the log.
#
#SBATCH -J ihdm-probe
#SBATCH --time=01:50:00
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/probe_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/probe_%j.err

set -uo pipefail

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm/probe}"

CHURCH_DIR="${IHDM_RUN_ROOT}/lsun_church_A0_s1"
IXI_DIR="${IHDM_RUN_ROOT}/ixi_A0_s1"
MEM_DIR="${IHDM_RUN_ROOT}/mem_b24"

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
echo "Run root:    ${IHDM_RUN_ROOT}"
echo "=========================================="

[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}; run create_env.sbatch first" >&2; exit 1; }

cd "${REPO_DIR}" || exit 1
# Ahead of the editable install's .pth entry, so the library and the entry points always come
# from this checkout even if the env was installed from another one.
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "[warn] nvidia-smi not available"

mkdir -p "${IHDM_RUN_ROOT}"

# --- GPU memory trace -------------------------------------------------------------------------
# `gpu_mem_peak_gb` in metrics.jsonl is torch's process-local allocator peak; this trace is the
# device-level number (allocator cache + context included), which is what has to fit in 40 GB.
GPU_TRACE="${IHDM_RUN_ROOT}/gpu_mem_trace_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 5 > "${GPU_TRACE}" 2>/dev/null &
TRACE_PID=$!
trace_stop() { kill "${TRACE_PID}" 2>/dev/null; wait "${TRACE_PID}" 2>/dev/null; }
trap trace_stop EXIT

STEPS=()
STATUS=()
run_step() {
    local name="$1"; shift
    local t0 t1 rc
    echo ""
    echo "========== ${name} =========="
    echo "[cmd] $*"
    t0=$(date +%s)
    "$@"
    rc=$?
    t1=$(date +%s)
    if (( rc == 0 )); then
        echo "[ok] ${name} ($((t1 - t0)) s)"
        STATUS+=("ok")
    else
        echo "[FAIL] ${name} (exit ${rc}, $((t1 - t0)) s)"
        STATUS+=("FAILED(exit ${rc})")
    fi
    STEPS+=("${name}")
    return 0
}

TRAIN_CADENCE=(
    --config.training.ckpt_every=200
    --config.training.grid_every=200
    --config.training.eval_every=200
    --config.training.log_every=20
)

# ==============================================================================================
# 1. lsun_church, A0, seed 1, batch 16 (the config default), 600 iterations = 3 epochs
# ==============================================================================================
run_step "1. train lsun_church,A0 s1 batch 16, 600 iters" "${PY}" train.py \
    --config configs/spectral/arms.py:lsun_church,A0 \
    --config.seed=1 \
    --config.training.n_iters=600 \
    "${TRAIN_CADENCE[@]}" \
    --workdir "${CHURCH_DIR}"

# ==============================================================================================
# 2. ixi, A0, seed 1, batch 16, 600 iterations
# ==============================================================================================
run_step "2. train ixi,A0 s1 batch 16, 600 iters" "${PY}" train.py \
    --config configs/spectral/arms.py:ixi,A0 \
    --config.seed=1 \
    --config.training.n_iters=600 \
    "${TRAIN_CADENCE[@]}" \
    --workdir "${IXI_DIR}"

# ==============================================================================================
# 3. memory probe at batch 24 (40 iterations).  `--config.eval.batch_size` must be passed
#    explicitly: ml_collections applies flags after the factory, so the factory's
#    "eval follows training" rule does not see the CLI override (T2.1 log §6).
#    T2.3's fit peak(B) = 1.7046 B + 1.2997 GB predicts 42.2 GB -> an OOM is the expected result.
# ==============================================================================================
run_step "3. memory probe ixi,A0 batch 24, 40 iters" "${PY}" train.py \
    --config configs/spectral/arms.py:ixi,A0 \
    --config.seed=1 \
    --config.training.batch_size=24 \
    --config.eval.batch_size=24 \
    --config.training.n_iters=40 \
    --config.training.log_every=10 \
    --config.training.ckpt_every=40 \
    --config.training.eval_every=40 \
    --config.training.grid_every=1000000 \
    --workdir "${MEM_DIR}"

# ==============================================================================================
# 4. resume / extension: step 2's command with n_iters=800 on the same workdir.
#    train.py writes the rolling checkpoint in its finaliser, so initial_step is 600 and the
#    first metrics line of this invocation must be {"kind": "resume", "step": 600}.
# ==============================================================================================
run_step "4. resume ixi,A0 to 800 iters" "${PY}" train.py \
    --config configs/spectral/arms.py:ixi,A0 \
    --config.seed=1 \
    --config.training.n_iters=800 \
    "${TRAIN_CADENCE[@]}" \
    --workdir "${IXI_DIR}"

# ==============================================================================================
# 5. sampler timing on ema_iter_000800.pt.  200-step chains; `request.json` carries
#    timing.s_per_chain and timing.samples_per_s.  Sampling batch 32 / 64 / 128; the last two
#    draw the SAME 128 training seeds (same --rng-seed default), so they are directly comparable.
# ==============================================================================================
#    A deadline guard protects step 6: the 128-chain steps are only STARTED if their projected
#    cost (128 x the s/chain measured by 5a) still fits before ${DEADLINE_S}.  Never trim a
#    measurement to make it fit -- skip it and say so, so the number that is reported is real.
WALL_S=6600            # --time=01:50:00
DEADLINE_S=$(( WALL_S - 420 ))   # leave 7 min for step 6 and the epilogue

elapsed_s() { echo $(( $(date +%s) - START_TIME )); }

s_per_chain_of() {   # $1 = sample dir; echoes the measured s/chain, or nothing
    "${PY}" -c "import json,sys;print('%.4f' % json.load(open(sys.argv[1]))['timing']['s_per_chain'])" \
        "$1/request.json" 2>/dev/null
}

run_step "5a. sample batch 32 (4 held-out seeds x 8)" "${PY}" -m ihdm.cli.sample_ckpt \
    --run "${IXI_DIR}" --ckpt ema_iter_000800.pt \
    --source seed --n-seeds 4 --n-per-seed 8 --batch 32 \
    --out "${IXI_DIR}/probe_samples_b32"

SPC=$(s_per_chain_of "${IXI_DIR}/probe_samples_b32")
echo "[budget] measured s/chain at batch 32: ${SPC:-unknown}; elapsed $(elapsed_s) s of ${WALL_S}"

run_sampler_128() {   # $1 = batch, $2 = out dir name, $3 = step label
    local projected
    if [[ -n "${SPC}" ]]; then
        projected=$(awk -v s="${SPC}" 'BEGIN{printf "%d", 128*s + 120}')
        if (( $(elapsed_s) + projected > DEADLINE_S )); then
            echo ""
            echo "========== $3 =========="
            echo "[SKIPPED] projected ${projected} s does not fit before the ${DEADLINE_S} s deadline"
            STEPS+=("$3"); STATUS+=("SKIPPED (no wall time)")
            return 0
        fi
        echo "[budget] $3 projected ${projected} s"
    fi
    run_step "$3" "${PY}" -m ihdm.cli.sample_ckpt \
        --run "${IXI_DIR}" --ckpt ema_iter_000800.pt \
        --source train --n-seeds 128 --n-per-seed 1 --batch "$1" \
        --out "${IXI_DIR}/$2"
}

run_sampler_128 64  probe_samples_b64  "5b. sample batch 64 (128 train seeds x 1)"
run_sampler_128 128 probe_samples_b128 "5c. sample batch 128 (same 128 train seeds)"

# ==============================================================================================
# 6. evidence
# ==============================================================================================
trace_stop
trap - EXIT

echo ""
echo "========== 6a. artefact listing =========="
for d in "${CHURCH_DIR}" "${IXI_DIR}" "${MEM_DIR}"; do
    echo "--- ${d} ---"
    ls -R "${d}" 2>&1 | head -n 60
    echo "--- DONE marker: $(test -f "${d}/DONE" && echo present || echo MISSING) ---"
done

echo ""
echo "========== 6b. metrics.jsonl (kinds and cadence) =========="
for d in "${CHURCH_DIR}" "${IXI_DIR}" "${MEM_DIR}"; do
    echo "--- ${d} ---"
    "${PY}" - "${d}" <<'PYEOF'
import json, sys
from pathlib import Path

path = Path(sys.argv[1]) / "metrics.jsonl"
if not path.exists():
    print("no metrics.jsonl")
    raise SystemExit(0)
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
kinds: dict[str, list[int]] = {}
for row in rows:
    kinds.setdefault(row["kind"], []).append(row.get("step"))
for kind, steps in sorted(kinds.items()):
    print(f"  {kind:8s} n={len(steps):3d} steps={steps[:4]}...{steps[-1] if steps else None}")
train = [r for r in rows if r["kind"] == "train"]
warm = [r for r in train if (r.get("step") or 0) >= 100] or train[1:] or train
if warm:
    import statistics
    its = sorted(r["it_per_s"] for r in warm if r.get("it_per_s") is not None)
    imgs = sorted(r["img_per_s"] for r in warm if r.get("img_per_s") is not None)
    peaks = [r["gpu_mem_peak_gb"] for r in warm if r.get("gpu_mem_peak_gb") is not None]
    if its:
        print(f"  median it/s (step>=100) = {statistics.median(its):.4f}")
    if imgs:
        print(f"  median img/s(step>=100) = {statistics.median(imgs):.3f}")
    if peaks:
        print(f"  max gpu_mem_peak_gb     = {max(peaks):.3f}")
    print(f"  last train line: {json.dumps(train[-1], sort_keys=True)}")
evals = [r for r in rows if r["kind"] == "eval"]
if evals:
    print(f"  last eval  line: {json.dumps(evals[-1], sort_keys=True)}")
resumes = [r for r in rows if r["kind"] == "resume"]
print(f"  resume events: {[json.dumps(r, sort_keys=True) for r in resumes]}")
aborts = [r for r in rows if r["kind"] == "abort"]
print(f"  abort events : {len(aborts)}")
PYEOF
done

echo ""
echo "========== 6c. sampler request.json timing =========="
for d in "${IXI_DIR}"/probe_samples_b*; do
    echo "--- ${d} ---"
    "${PY}" -c "import json,sys;r=json.load(open(sys.argv[1]));print(' batch',r['request']['batch_size'],'n',r['shapes']['n_samples'],'wall_s %.2f'%r['timing']['wall_s'],'s_per_chain %.3f'%r['timing']['s_per_chain'],'samples_per_s %.4f'%r['timing']['samples_per_s'])" \
        "${d}/request.json" 2>&1 || echo "  no request.json"
done

echo ""
echo "========== 6d. device GPU memory trace =========="
if [[ -s "${GPU_TRACE}" ]]; then
    awk -F', ' '{if ($2+0 > m) m = $2+0} END {printf "  samples=%d  max memory.used = %d MiB (%.2f GB)\n", NR, m, m/1024}' "${GPU_TRACE}"
    echo "  trace file: ${GPU_TRACE}"
else
    echo "  no trace recorded"
fi

echo ""
echo "========== 6e. checkpoints =========="
for d in "${CHURCH_DIR}" "${IXI_DIR}"; do
    echo "--- ${d}/checkpoints ---"
    ls -la "${d}/checkpoints" 2>&1 | head -n 20
done

echo ""
echo "========== step summary =========="
for i in "${!STEPS[@]}"; do
    printf "  %-52s %s\n" "${STEPS[$i]}" "${STATUS[$i]}"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "sacct:"
sacct -j "${SLURM_JOB_ID:-0}" -n -P -o JobID,State,Elapsed,MaxRSS,NodeList 2>&1 | head -n 5
echo "PROBE FINISHED"
