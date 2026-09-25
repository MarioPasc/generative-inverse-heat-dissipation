#!/usr/bin/env bash
# T5.1 -- the ONE writer of the evaluation inputs.  Submitted by `slurm/eval/submit_eval.sh prepare`.
#
# For each of the four datasets under $IHDM_DATA_ROOT it writes the two D17 seed lists
# (`ensure_seed_lists`) and `_features_inception_ref.npy`, each with its JSON sidecar: 6 files per
# dataset, 24 in total, and nothing else on FSCRATCH.  The lists are first drawn into a shadow
# copy of each dataset on $LOCALSCRATCH and their sha256 compared with
# `slurm/eval/expected_seed_lists.csv` (computed locally with the same code); on any mismatch the
# job stops before touching FSCRATCH.  Idempotent: a rerun reads every file back and writes none.
#
# Why one writer: the torchscript Inception is not bitwise deterministic across calls (up to 2e-3,
# T4.3 §6), so every run of a dataset must be scored against ONE stored reference matrix.  The 30
# evaluation tasks refuse to start when a cache file is missing (`ihdm_check_eval_caches`).
#
# Resources: the reference pass is 4 x 800 images through Inception, minutes on an A100; the
# 1 h wall is headroom for the dataset loads.  This job is shorter than SCBI's 2 h floor, which is
# accepted for a one-off setup job (it runs once per campaign).
#
#SBATCH -J ihdm-eval-prep
#SBATCH --time=01:00:00
#SBATCH --qos=medium_uma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_prepare_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_prepare_%j.err

set -uo pipefail
# Before ANY python: the env on FSCRATCH has no .pyc, and the first import would write ~900 there.
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/ihdm_pycache_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
EVAL_HOME="${IHDM_EVAL_HOME:-${USER_ROOT}/execs/ihdm/eval}"
DATASETS="${IHDM_DATASETS:-ixi:lsun_bedroom:lsun_church:oasis1}"
PY="${ENV_PREFIX}/bin/python"

MYLOCALSCRATCH="${LOCALSCRATCH:-/tmp}/${USER}/eval_prepare_${SLURM_JOB_ID:-local}"

echo "=========================================="
echo "BEGIN PREPARE"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || cat "${REPO_DIR}/.git_sha" 2>/dev/null || echo n/a)"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Data root:   ${IHDM_DATA_ROOT}"
echo "Datasets:    ${DATASETS}"
echo "Eval home:   ${EVAL_HOME}"
echo "Local:       ${MYLOCALSCRATCH}"
echo "=========================================="

[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}" >&2; exit 1; }
# shellcheck source=slurm/eval/common.sh
source "${REPO_DIR}/slurm/eval/common.sh" || { echo "[FATAL] cannot source common.sh" >&2; exit 1; }

mkdir -p "${MYLOCALSCRATCH}" "${EVAL_HOME}"
cleanup() { [[ -n "${MYLOCALSCRATCH}" && -d "${MYLOCALSCRATCH}" ]] && rm -rf --one-file-system "${MYLOCALSCRATCH}"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ihdm_env_setup || exit 1
ihdm_stage_inception || exit 1

echo ""
echo "---- dataset folders before ----"
for d in ${DATASETS//:/ }; do
    echo "${d}: $(ls -1 "${IHDM_DATA_ROOT}/${d}" | wc -l) entries: $(ls -1 "${IHDM_DATA_ROOT}/${d}" | paste -sd' ' -)"
done

CMD=("${PY}" slurm/eval/prepare_eval.py
     --data-root "${IHDM_DATA_ROOT}"
     --datasets "${DATASETS//:/,}"
     --expected slurm/eval/expected_seed_lists.csv
     --shadow-root "${MYLOCALSCRATCH}/shadow"
     --device cuda
     --out "${MYLOCALSCRATCH}/prepare.json")
echo ""
echo "[cmd] ${CMD[*]}"
echo ""
T0=$(date +%s)
"${CMD[@]}"
RC=$?
T1=$(date +%s)

if [[ -f "${MYLOCALSCRATCH}/prepare.json" ]]; then
    ihdm_copy_atomic "${MYLOCALSCRATCH}/prepare.json" "${EVAL_HOME}/prepare_${SLURM_JOB_ID:-local}.json" \
        && echo "record:      ${EVAL_HOME}/prepare_${SLURM_JOB_ID:-local}.json"
fi

echo ""
echo "========== artefacts =========="
for d in ${DATASETS//:/ }; do
    for f in "${IHDM_EVAL_CACHE_FILES[@]}"; do
        p="${IHDM_DATA_ROOT}/${d}/${f}"
        if [[ -f "${p}" ]]; then echo "${d}/${f}  $(sha256sum "${p}" | cut -c1-16)  $(stat -c %s "${p}") B"; else echo "${d}/${f}  MISSING"; fi
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Prepare:     $((T1 - T0)) s"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "END PREPARE  status=$( ((RC == 0)) && echo ok || echo FAILED) exit=${RC}"
echo "sacct:"
sacct -j "${SLURM_JOB_ID:-0}" -n -P -o JobID,State,Elapsed,MaxRSS,NodeList 2>&1 | head -n 5
echo "=========================================="

exit ${RC}
