#!/usr/bin/env bash
# T5.1 -- A100 timing job: sizes the evaluation array and settles the AMP question of T4.3 §6.
# Submitted by `slurm/eval/submit_eval.sh timing`.
#
# Input: a finished array-1 run kept read-only in $HOME (default
# ~/execs/ihdm/fixtures/array_2408239/ixi_A0_s2: EMA checkpoints 5,000 and 7,500, lr 2e-4).
# Weight quality does not matter for timing.  Nothing is written inside the fixture or on FSCRATCH:
#   * a shadow DATASET on $LOCALSCRATCH links the four inputs of `ixi`, so the seed lists the
#     probe needs are drawn there (a second, independent check of `prepare_eval`'s ixi hashes);
#   * a shadow RUN on $LOCALSCRATCH links the fixture's checkpoints/grids/manifest, and carries a
#     copy of `config.json` whose `data.root` points at the shadow dataset -- the only departure
#     from the real worker, which links `config.json` as is.
#
# Steps (each guarded by the 2 h wall):
#   1. timing_probe.py --phases AB : the 500-seed LSD set at 7,500 under --amp off/fp16/bf16 at
#      batch 32 (s/chain, LSD, paired AMP-minus-off difference), Inception time for 2k samples;
#   2. evaluate_run --ckpts 5000 --gate 5000,7500 --n-lsd 500 --skip-inception, in `off` when it
#      fits the remaining wall, else in the fastest AMP mode (it reuses that mode's 7,500 set);
#   3. timing_probe.py --phases C  : s/chain at batch 64 on 128 of the same seeds, per mode.
# The records (JSON only, plus one tar of the sample trees) go to ~/execs/ihdm/eval/timing_<job>/.
#
#SBATCH -J ihdm-eval-timing
#SBATCH --time=02:00:00
#SBATCH --qos=medium_uma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_timing_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_timing_%j.err

set -uo pipefail
# Before ANY python: the env on FSCRATCH has no .pyc, and the first import would write ~900 there.
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/ihdm_pycache_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"

START_TIME=$(date +%s)
WALL_S="${WALL_S:-7200}"

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
FIXTURE="${IHDM_TIMING_FIXTURE:-${USER_ROOT}/execs/ihdm/fixtures/array_2408239/ixi_A0_s2}"
EVAL_HOME="${IHDM_EVAL_HOME:-${USER_ROOT}/execs/ihdm/eval}"
PY="${ENV_PREFIX}/bin/python"
# Counts and steps: the D16 defaults on the fixture; overridable only for a local dry run.
STEP_EARLY="${TIMING_STEP_EARLY:-5000}"
STEP="${TIMING_STEP:-7500}"
N_LSD="${TIMING_N:-500}"
N_B64="${TIMING_N_B64:-128}"
N_INCEPTION="${TIMING_N_INCEPTION:-2000}"

MYLOCALSCRATCH="${LOCALSCRATCH:-/tmp}/${USER}/eval_timing_${SLURM_JOB_ID:-local}"
SHADOW_DATA="${MYLOCALSCRATCH}/data"
SHADOW_RUN="${MYLOCALSCRATCH}/run/$(basename "${FIXTURE}")"
OUT="${MYLOCALSCRATCH}/out"
DEST="${EVAL_HOME}/timing_${SLURM_JOB_ID:-local}"

echo "=========================================="
echo "BEGIN TIMING"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || cat "${REPO_DIR}/.git_sha" 2>/dev/null || echo n/a)"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Fixture:     ${FIXTURE}"
echo "Data root:   ${IHDM_DATA_ROOT}"
echo "Local:       ${MYLOCALSCRATCH}"
echo "Results:     ${DEST}"
echo "=========================================="

[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}" >&2; exit 1; }
# shellcheck source=slurm/eval/common.sh
source "${REPO_DIR}/slurm/eval/common.sh" || { echo "[FATAL] cannot source common.sh" >&2; exit 1; }

mkdir -p "${OUT}" "${DEST}"

FINALIZED=0
finalize() {
    [[ "${FINALIZED}" == 1 ]] && return 0
    FINALIZED=1
    # JSON records and logs come back as plain files; the sample trees as ONE tar (file quota).
    if [[ -d "${SHADOW_RUN}" ]]; then
        (cd "${SHADOW_RUN}" && find . -name '*.json' -not -path './config.json' -not -path './manifest.json' \
            | tar -cf "${OUT}/records.tar" -T -) 2>/dev/null || true
        (cd "${SHADOW_RUN}" && tar -cf "${OUT}/samples.tar" samples* 2>/dev/null) || true
    fi
    if cp -a "${OUT}/." "${DEST}/"; then
        echo "copy-back:   ${OUT} -> ${DEST} ($(ls -1 "${DEST}" | wc -l) files)"
        [[ -n "${MYLOCALSCRATCH}" && -d "${MYLOCALSCRATCH}" ]] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
    else
        echo "[FATAL] copy-back failed; results remain at ${MYLOCALSCRATCH} on $(hostname)" >&2
    fi
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ihdm_env_setup || exit 1
ihdm_stage_inception || exit 1

# ---- shadow dataset and shadow run -------------------------------------------------------------
DATASET_ID=$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['data']['dataset'])" "${FIXTURE}/config.json") \
    || { echo "[FATAL] cannot read ${FIXTURE}/config.json" >&2; exit 1; }
mkdir -p "${SHADOW_DATA}/${DATASET_ID}"
for f in images.npy index.csv splits.json meta.json; do
    ln -sfn "${IHDM_DATA_ROOT}/${DATASET_ID}/${f}" "${SHADOW_DATA}/${DATASET_ID}/${f}"
done
ihdm_shadow_run "${FIXTURE}" "${SHADOW_RUN}" || exit 1
rm -f "${SHADOW_RUN}/config.json"
"${PY}" -c "
import json, sys
cfg = json.load(open(sys.argv[1])); cfg['data']['root'] = sys.argv[2]
json.dump(cfg, open(sys.argv[3], 'w'), indent=2)
" "${FIXTURE}/config.json" "${SHADOW_DATA}" "${SHADOW_RUN}/config.json" || exit 1
echo "dataset:     ${DATASET_ID} (shadow at ${SHADOW_DATA}/${DATASET_ID})"
# main's condition: the shadow inputs must hash-equal the FSCRATCH originals.
for f in images.npy splits.json; do
    H_ORIG=$(sha256sum "${IHDM_DATA_ROOT}/${DATASET_ID}/${f}" | cut -d' ' -f1)
    H_SHADOW=$(sha256sum "${SHADOW_DATA}/${DATASET_ID}/${f}" | cut -d' ' -f1)
    echo "shadow hash: ${f} original=${H_ORIG} shadow=${H_SHADOW} ($(readlink -f "${SHADOW_DATA}/${DATASET_ID}/${f}"))"
    [[ "${H_ORIG}" == "${H_SHADOW}" ]] || { echo "[FATAL] shadow ${f} differs from the original" >&2; exit 1; }
done
ls -la "${SHADOW_RUN}" "${SHADOW_RUN}/checkpoints/"

# ---- 1. probe phases A and B -------------------------------------------------------------------
DEADLINE=$(( START_TIME + WALL_S - 600 ))
echo ""
echo "[cmd] timing_probe.py --phases AB (deadline $(date -d @${DEADLINE} --iso-8601=seconds))"
T0=$(date +%s)
"${PY}" slurm/eval/timing_probe.py --run "${SHADOW_RUN}" --step "${STEP}" --n "${N_LSD}" --batch 32 \
    --n-inception "${N_INCEPTION}" \
    --modes off,fp16,bf16 --phases AB --expected slurm/eval/expected_seed_lists.csv --deadline "${DEADLINE}" --out "${OUT}/timing_ab.json"
RC_AB=$?
echo "probe AB:    exit ${RC_AB} in $(( $(date +%s) - T0 )) s"
cp -f "${OUT}/timing_ab.json" "${DEST}/timing_ab.json" 2>/dev/null || true

# ---- 2. the plateau gate on 5000,7500 ----------------------------------------------------------
GATE_MODE=$("${PY}" - "${OUT}/timing_ab.json" "$(( START_TIME + WALL_S - 900 - $(date +%s) ))" "${N_LSD}" <<'EOF'
import json, sys
try:
    phases = json.load(open(sys.argv[1]))["phases"]
except (OSError, ValueError, KeyError):
    print("skip"); raise SystemExit
remaining = float(sys.argv[2])
cost = {m: int(sys.argv[3]) * phases[f"A_{m}"]["s_per_chain"] * 1.1 + 180 for m in ("off", "fp16", "bf16") if f"A_{m}" in phases}
if "off" in cost and cost["off"] <= remaining:
    print("off")
else:
    fitting = sorted((c, m) for m, c in cost.items() if c <= remaining)
    print(fitting[0][1] if fitting else "skip")
EOF
)
RC_GATE=0
echo ""
if [[ "${GATE_MODE}" == "skip" ]]; then
    echo "gate:        SKIPPED (does not fit the remaining wall, or phase A produced nothing)"
else
    echo "[cmd] evaluate_run --ckpts ${STEP_EARLY} --gate ${STEP_EARLY},${STEP} --n-lsd ${N_LSD} --skip-inception --amp ${GATE_MODE}"
    T0=$(date +%s)
    "${PY}" -m ihdm.cli.evaluate_run --run "${SHADOW_RUN}" --ckpts "${STEP_EARLY}" \
        --gate "${STEP_EARLY},${STEP}" --n-lsd "${N_LSD}" --skip-inception --device cuda \
        --amp "${GATE_MODE}"
    RC_GATE=$?
    echo "gate:        mode ${GATE_MODE}, exit ${RC_GATE} in $(( $(date +%s) - T0 )) s"
fi

# ---- 3. probe phase C --------------------------------------------------------------------------
DEADLINE=$(( START_TIME + WALL_S - 300 ))
echo ""
echo "[cmd] timing_probe.py --phases C (deadline $(date -d @${DEADLINE} --iso-8601=seconds))"
T0=$(date +%s)
"${PY}" slurm/eval/timing_probe.py --run "${SHADOW_RUN}" --step "${STEP}" --n-b64 "${N_B64}" \
    --modes off,fp16,bf16 --phases C --deadline "${DEADLINE}" --reserve 120 --out "${OUT}/timing_c.json"
RC_C=$?
echo "probe C:     exit ${RC_C} in $(( $(date +%s) - T0 )) s"

RC=0
(( RC_AB != 0 )) && RC=${RC_AB}
(( RC == 0 && RC_GATE != 0 )) && RC=${RC_GATE}
(( RC == 0 && RC_C != 0 )) && RC=${RC_C}

echo ""
echo "========== artefacts =========="
ls -la "${OUT}"
find "${SHADOW_RUN}" -name request.json -path '*lsd*' | sort | while read -r f; do
    echo "${f#${SHADOW_RUN}/}: $("${PY}" -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['signature']['amp'], r['shape'], round(r['s_per_chain'], 4))" "${f}")"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "END TIMING   status=$( ((RC == 0)) && echo ok || echo FAILED) exit=${RC} (probe AB ${RC_AB}, gate ${RC_GATE} [${GATE_MODE}], probe C ${RC_C})"
echo "sacct:"
sacct -j "${SLURM_JOB_ID:-0}" -n -P -o JobID,State,Elapsed,MaxRSS,NodeList 2>&1 | head -n 5
echo "=========================================="

exit ${RC}
