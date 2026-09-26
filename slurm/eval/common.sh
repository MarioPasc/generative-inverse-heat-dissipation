#!/usr/bin/env bash
# T5.1 -- shell helpers shared by the evaluation workers (`prepare_eval.sbatch`, `eval_array.sbatch`,
# `gate.sbatch`, `timing.sbatch`).  Sourced from `${REPO_DIR}/slurm/eval/common.sh`, never from
# `BASH_SOURCE`: SLURM runs a copy of the batch script out of its spool directory.
#
# Nothing here writes to FSCRATCH.  The only durable writes are the ones the callers make to
# `$HOME/execs/ihdm/eval/` and, for `prepare_eval`, the 24 dataset-cache files.

# Where clean-fid looks for the weights (hard-coded `/tmp` in `cleanfid.features.feature_extractor`,
# which also passes download=True; compute nodes are offline, so the file must be staged).
IHDM_INCEPTION_DEST="/tmp/inception-2015-12-05.pt"
IHDM_INCEPTION_SRC="${IHDM_INCEPTION_SRC:-/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/cache/inception-2015-12-05.pt}"
# sha256 of the StyleGAN2-ADA torchscript Inception that clean-fid downloads (checked 2026-09-25
# on the copy fetched from nvlabs-fi-cdn.nvidia.com into IHDM_INCEPTION_SRC).
IHDM_INCEPTION_SHA256="f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4"

# The six per-dataset files `prepare_eval` writes and every other job only reads.
IHDM_EVAL_CACHE_FILES=(
    eval_seeds_500.npy eval_seeds_500.json
    eval_seeds_final_2000.npy eval_seeds_final_2000.json
    _features_inception_ref.npy _features_inception_ref.json
)

ihdm_pycache_guard() {
    # The cluster env ships no .pyc for many packages; without a prefix the first import writes
    # ~900 .pyc files INTO the env on FSCRATCH (T3.4, relayed by main, 2026-09-25).  Refuse to run
    # python without a prefix outside FSCRATCH.
    case "${PYTHONPYCACHEPREFIX:-}" in
        ""|*fscratch*) echo "[FATAL] PYTHONPYCACHEPREFIX='${PYTHONPYCACHEPREFIX:-}' is unset or on FSCRATCH" >&2; return 1 ;;
    esac
    echo "pycache:     ${PYTHONPYCACHEPREFIX}"
}

ihdm_env_setup() {
    # Same lines as slurm/array/train_array.sbatch; PYTHONPATH goes ahead of the editable
    # install so the library comes from REPO_DIR.  PYTHONPYCACHEPREFIX is exported by every
    # worker on its first line (see ihdm_pycache_guard), before any python can start.
    cd "${REPO_DIR}" || { echo "[FATAL] cannot cd to ${REPO_DIR}" >&2; return 1; }
    export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    ihdm_pycache_guard || return 1
    nvidia-smi -L || echo "[warn] nvidia-smi not available"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        || echo "[warn] nvidia-smi query not available"
}

ihdm_stage_inception() {
    # Copy the weights to /tmp atomically (tasks sharing a node may stage concurrently): copy
    # to a per-job temporary name in /tmp, check the digest, then rename.
    local have tmp
    if [[ -f "${IHDM_INCEPTION_DEST}" ]]; then
        have=$(sha256sum "${IHDM_INCEPTION_DEST}" | cut -d' ' -f1)
        if [[ "${have}" == "${IHDM_INCEPTION_SHA256}" ]]; then
            echo "inception:   ${IHDM_INCEPTION_DEST} already staged (sha256 ok)"
            return 0
        fi
        echo "[warn] ${IHDM_INCEPTION_DEST} has sha256 ${have}; restaging"
    fi
    [[ -f "${IHDM_INCEPTION_SRC}" ]] || { echo "[FATAL] no Inception weights at ${IHDM_INCEPTION_SRC}" >&2; return 1; }
    tmp="/tmp/.inception-2015-12-05.pt.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"
    cp "${IHDM_INCEPTION_SRC}" "${tmp}" || { echo "[FATAL] cannot copy the weights to /tmp" >&2; return 1; }
    have=$(sha256sum "${tmp}" | cut -d' ' -f1)
    if [[ "${have}" != "${IHDM_INCEPTION_SHA256}" ]]; then
        rm -f "${tmp}"
        echo "[FATAL] ${IHDM_INCEPTION_SRC} has sha256 ${have}, expected ${IHDM_INCEPTION_SHA256}" >&2
        return 1
    fi
    mv -f "${tmp}" "${IHDM_INCEPTION_DEST}"
    echo "inception:   staged ${IHDM_INCEPTION_SRC} -> ${IHDM_INCEPTION_DEST} (sha256 ok)"
}

ihdm_check_eval_caches() {
    # Refuse to start an evaluation whose dataset lacks any of the one-writer files: without them
    # `evaluate_run` would write its own seed lists and reference features, which is exactly the
    # concurrent second writer the one-writer rule exists to prevent.
    local dataset_dir="$1" f missing=0
    for f in "${IHDM_EVAL_CACHE_FILES[@]}"; do
        if [[ ! -s "${dataset_dir}/${f}" ]]; then
            echo "[FATAL] ${dataset_dir}/${f} is missing; run prepare_eval first" >&2
            missing=1
        fi
    done
    return ${missing}
}

ihdm_decode_cell() {
    # Decode one row of cells.csv by its INDEX COLUMN (slurm/array/train_array.sbatch).  Sets
    # CELL_INDEX RUN_ID DATASET_ID ARM SEED TIER in the caller's scope.
    local want="$1" cells="$2" row
    row=$(awk -F, -v want="${want}" 'NR > 1 && $1 == want {print; found = 1} END {exit !found}' "${cells}")
    [[ -n "${row}" ]] || { echo "[FATAL] no row with index ${want} in ${cells}" >&2; return 1; }
    [[ $(printf '%s\n' "${row}" | wc -l) -eq 1 ]] || { echo "[FATAL] index ${want} matches more than one row" >&2; return 1; }
    IFS=, read -r CELL_INDEX RUN_ID DATASET_ID ARM SEED TIER <<<"${row}"
    local field
    for field in CELL_INDEX RUN_ID DATASET_ID ARM SEED TIER; do
        [[ -n "${!field}" ]] || { echo "[FATAL] empty ${field} in row: ${row}" >&2; return 1; }
    done
    [[ "${CELL_INDEX}" == "${want}" ]] || { echo "[FATAL] decode mismatch: ${CELL_INDEX} != ${want}" >&2; return 1; }
    [[ "${RUN_ID}" == "${DATASET_ID}_${ARM}_s${SEED}" ]] \
        || { echo "[FATAL] run_id ${RUN_ID} disagrees with ${DATASET_ID}/${ARM}/s${SEED}" >&2; return 1; }
    echo "CELL         index=${CELL_INDEX} run_id=${RUN_ID} dataset=${DATASET_ID} arm=${ARM} seed=${SEED} tier=${TIER}"
}

ihdm_shadow_run() {
    # Build a run directory on local disk whose inputs are links into the real run, so that
    # `samples*/` and `metrics*/` land on $LOCALSCRATCH and nothing is written into the run.
    local run_dir="$1" shadow="$2" item
    [[ -f "${run_dir}/config.json" ]] || { echo "[FATAL] no config.json in ${run_dir}" >&2; return 1; }
    [[ -d "${run_dir}/checkpoints" ]] || { echo "[FATAL] no checkpoints/ in ${run_dir}" >&2; return 1; }
    mkdir -p "${shadow}"
    for item in config.json manifest.json checkpoints grids metrics.jsonl; do
        if [[ -e "${run_dir}/${item}" ]]; then
            ln -sfn "${run_dir}/${item}" "${shadow}/${item}"
        fi
    done
    echo "shadow run:  ${shadow} -> links into ${run_dir}"
}

ihdm_copy_atomic() {
    # Copy SRC to DEST through a temporary name in DEST's directory, then rename: a reader sees
    # the old file or the new one, never a truncated one.
    local src="$1" dest="$2" tmp
    tmp="$(dirname "${dest}")/.$(basename "${dest}").tmp.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"
    cp "${src}" "${tmp}" && mv -f "${tmp}" "${dest}"
}

# Spacing of the evaluated checkpoints and of the gate pair (ihdm.metrics.run_eval.EVAL_STRIDE).
IHDM_EVAL_STRIDE=5000

ihdm_check_n_iters() {
    # N_ITERS must be a positive multiple of the stride: the final and held-out sets are drawn at
    # the largest checkpoint only when it is one of the evaluated steps (T3.5, D22).
    local n="$1"
    [[ "${n}" =~ ^[0-9]+$ ]] && (( n >= IHDM_EVAL_STRIDE && n % IHDM_EVAL_STRIDE == 0 )) \
        || { echo "[FATAL] N_ITERS='${n}' is not a positive multiple of ${IHDM_EVAL_STRIDE}" >&2; return 1; }
}

ihdm_gate_steps() {
    # Set GATE_EARLY and GATE_LATE in the caller's scope from N_ITERS (T3.5, D22): by default the
    # last two evaluated steps, N_ITERS - 5000 and N_ITERS.  Values already set in the environment
    # win.  Refuses a pair that is not two increasing multiples of the stride.
    local n="$1"
    ihdm_check_n_iters "${n}" || return 1
    GATE_LATE="${GATE_LATE:-${n}}"
    GATE_EARLY="${GATE_EARLY:-$(( n - IHDM_EVAL_STRIDE ))}"
    local s
    for s in "${GATE_EARLY}" "${GATE_LATE}"; do
        [[ "${s}" =~ ^[0-9]+$ ]] && (( s > 0 && s % IHDM_EVAL_STRIDE == 0 )) \
            || { echo "[FATAL] gate step '${s}' is not a positive multiple of ${IHDM_EVAL_STRIDE}" >&2; return 1; }
    done
    (( GATE_EARLY < GATE_LATE )) \
        || { echo "[FATAL] gate ${GATE_EARLY} vs ${GATE_LATE}: the earlier step must come first" >&2; return 1; }
}

ihdm_gate_stem() {
    # The file stem of one gate result: <run_id><amp>_gate_<early:06d>_<late:06d> (T3.5).  The
    # step pair is in the name so the 55k/60k gate never overwrites the 35k/40k one.
    local run_id="$1" suffix="$2" early="$3" late="$4"
    printf '%s%s_gate_%06d_%06d\n' "${run_id}" "${suffix}" "${early}" "${late}"
}

ihdm_gate_tars() {
    # Print, one per line, every gate tar of a run and precision in GATE_DIR, in unpack order:
    # the legacy `<run_id><amp>_gate.tar` first (written by gate job 2432703 before T3.5; it holds
    # the 35k/40k pair), then the pair-named tars in ascending step order.  Every one is
    # unpacked: the sample cache is signature-checked, so a set from any gate is reused safely.
    local gate_dir="$1" run_id="$2" suffix="$3" f
    [[ -f "${gate_dir}/${run_id}${suffix}_gate.tar" ]] && echo "${gate_dir}/${run_id}${suffix}_gate.tar"
    for f in "${gate_dir}/${run_id}${suffix}"_gate_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].tar; do
        [[ -f "${f}" ]] && echo "${f}"
    done
    return 0
}

ihdm_last_ema_step() {
    # The largest step among <run>/checkpoints/ema_iter_*.pt (0 when none); mirrors
    # slurm/array/train_array.sbatch.
    local d="$1/checkpoints" best=0 step name f
    for f in "${d}"/ema_iter_*.pt; do
        [[ -e "${f}" ]] || continue
        name="$(basename "${f}")"; name="${name#ema_iter_}"; name="${name%.pt}"
        [[ "${name}" =~ ^[0-9]+$ ]] || continue
        step=$((10#${name}))
        (( step > best )) && best=${step}
    done
    echo "${best}"
}

ihdm_amp_suffix() {
    # "" for off, "_amp-<mode>" otherwise; mirrors ihdm.metrics.run_eval.samples_dirname.
    case "$1" in
        off) echo "" ;;
        fp16|bf16) echo "_amp-$1" ;;
        *) echo "[FATAL] AMP must be off, fp16 or bf16, got '$1'" >&2; return 1 ;;
    esac
}
