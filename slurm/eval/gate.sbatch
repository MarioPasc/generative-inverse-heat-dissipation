#!/usr/bin/env bash
# T5.1 -- the plateau gate of D10/D17 on the first finished A0 runs of `ixi` and `lsun_church`
# (cells 0 and 3 of `slurm/array/cells.csv`).  Submitted by `slurm/eval/submit_eval.sh gate`, with
# `afterok` on those two training tasks and on the prepare job.  The decision to extend the
# campaign stays with `main`; this job only writes the statistic.
#
# Per cell, sequentially:
#   evaluate_run --ckpts ${GATE_EARLY} --gate ${GATE_EARLY},${GATE_LATE} --n-lsd 500 --skip-inception
# The pair follows the run length (T3.5, D22): GATE_EARLY = N_ITERS - 5000 and GATE_LATE = N_ITERS
# by default (35k/40k at N_ITERS=40000, 55k/60k at 60000); GATE_EARLY/GATE_LATE set in the
# environment win.
# Naming ONLY the earlier step in --ckpts is deliberate and pinned by
# tests/metrics/test_run_eval.py::test_the_gate_command_draws_only_the_two_lsd_sets: the final and
# held-out sets belong to the largest available checkpoint and are drawn whenever that step is in
# --ckpts, so `--ckpts all` or `--ckpts 35000,40000` would cost 5,000 chains instead of 1,000.
#
# Returns per cell, in ~/execs/ihdm/eval/gate/, named by the step pair so no gate overwrites
# another: `<run_id><amp>_gate_<early:06d>_<late:06d>.json` (plain) and `.tar` (the shadow run:
# both 500-sample LSD sets, ckpt_<early>.json, gate.json), which the evaluation array unpacks so it
# does not redraw those two sets.  The files of gate job 2432703 keep their pre-T3.5 names
# `<run_id><amp>_gate.{json,tar}` (35k/40k); the array reads both forms (common.sh ihdm_gate_tars).
#
#SBATCH -J ihdm-eval-gate
#SBATCH --time=03:00:00
#SBATCH --qos=medium_uma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_gate_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/eval_gate_%j.err

set -uo pipefail
# Before ANY python: the env on FSCRATCH has no .pyc, and the first import would write ~900 there.
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/ihdm_pycache_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"

START_TIME=$(date +%s)

USER_ROOT="/mnt/home/users/tic_163_uma/mpascual"
REPO_DIR="${IHDM_REPO_DIR:-${USER_ROOT}/fscratch/repos/generative-inverse-heat-dissipation}"
ENV_PREFIX="${IHDM_ENV_PREFIX:-${USER_ROOT}/fscratch/conda_envs/ihdm}"
export IHDM_DATA_ROOT="${IHDM_DATA_ROOT:-${USER_ROOT}/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
export IHDM_RUN_ROOT="${IHDM_RUN_ROOT:-${USER_ROOT}/fscratch/runs/ihdm}"
EVAL_HOME="${IHDM_EVAL_HOME:-${USER_ROOT}/execs/ihdm/eval}"
CELLS="${IHDM_CELLS:-${REPO_DIR}/slurm/array/cells.csv}"
GATE_CELLS="${GATE_CELLS:-0:3}"
N_ITERS="${N_ITERS:-40000}"
GATE_EARLY="${GATE_EARLY:-}"
GATE_LATE="${GATE_LATE:-}"
GATE_N_LSD="${GATE_N_LSD:-500}"
AMP="${AMP:-off}"
PY="${ENV_PREFIX}/bin/python"

MYLOCALSCRATCH="${LOCALSCRATCH:-/tmp}/${USER}/eval_gate_${SLURM_JOB_ID:-local}"
GATE_HOME="${EVAL_HOME}/gate"

echo "=========================================="
echo "BEGIN GATE"
echo "Job:         ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start:       $(date --iso-8601=seconds)"
echo "Repo:        ${REPO_DIR}"
echo "Git commit:  $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || cat "${REPO_DIR}/.git_sha" 2>/dev/null || echo n/a)"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Run root:    ${IHDM_RUN_ROOT}  (read only)"
echo "Cells:       ${GATE_CELLS//:/,} of ${CELLS}"
echo "Returns:     ${GATE_HOME}"

[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}" >&2; exit 1; }
[[ -f "${CELLS}" ]] || { echo "[FATAL] no cell table at ${CELLS}" >&2; exit 1; }
# shellcheck source=slurm/eval/common.sh
source "${REPO_DIR}/slurm/eval/common.sh" || { echo "[FATAL] cannot source common.sh" >&2; exit 1; }
SUFFIX=$(ihdm_amp_suffix "${AMP}") || exit 1
ihdm_gate_steps "${N_ITERS}" || exit 1
echo "Gate:        ${GATE_EARLY} vs ${GATE_LATE} (N_ITERS=${N_ITERS}), n_lsd=${GATE_N_LSD}, amp=${AMP}"
echo "=========================================="

mkdir -p "${MYLOCALSCRATCH}" "${GATE_HOME}"
cleanup() { [[ -n "${MYLOCALSCRATCH}" && -d "${MYLOCALSCRATCH}" ]] && rm -rf --one-file-system "${MYLOCALSCRATCH}"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ihdm_env_setup || exit 1

RC=0
for idx in ${GATE_CELLS//:/ }; do
    echo ""
    echo "------------------------------------------"
    ihdm_decode_cell "${idx}" "${CELLS}" || { RC=1; continue; }
    RUN_DIR="${IHDM_RUN_ROOT}/${RUN_ID}"
    SHADOW="${MYLOCALSCRATCH}/${RUN_ID}"
    for step in "${GATE_EARLY}" "${GATE_LATE}"; do
        f="${RUN_DIR}/checkpoints/ema_iter_$(printf '%06d' "${step}").pt"
        [[ -f "${f}" ]] || { echo "[FATAL] ${f} is missing" >&2; RC=1; continue 2; }
    done
    ihdm_check_eval_caches "${IHDM_DATA_ROOT}/${DATASET_ID}" || { RC=1; continue; }
    ihdm_shadow_run "${RUN_DIR}" "${SHADOW}" || { RC=1; continue; }

    T0=$(date +%s)
    "${PY}" -m ihdm.cli.evaluate_run --run "${SHADOW}" --ckpts "${GATE_EARLY}" \
        --gate "${GATE_EARLY},${GATE_LATE}" --n-lsd "${GATE_N_LSD}" --skip-inception \
        --device cuda --amp "${AMP}"
    rc=$?
    echo "gate ${RUN_ID}: exit ${rc} in $(( $(date +%s) - T0 )) s"
    (( rc != 0 )) && RC=${rc}

    STEM=$(ihdm_gate_stem "${RUN_ID}" "${SUFFIX}" "${GATE_EARLY}" "${GATE_LATE}")
    GATE_JSON="${SHADOW}/metrics${SUFFIX}/gate.json"
    if [[ -f "${GATE_JSON}" ]]; then
        ihdm_copy_atomic "${GATE_JSON}" "${GATE_HOME}/${STEM}.json" \
            && echo "gate.json:   ${GATE_HOME}/${STEM}.json"
    fi
    if tar -cf "${MYLOCALSCRATCH}/${STEM}.tar" -C "${MYLOCALSCRATCH}" "${RUN_ID}"; then
        ihdm_copy_atomic "${MYLOCALSCRATCH}/${STEM}.tar" "${GATE_HOME}/${STEM}.tar" \
            && echo "gate tar:    ${GATE_HOME}/${STEM}.tar"
    fi
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Finished:    $(date --iso-8601=seconds)"
echo "Duration:    $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "END GATE     status=$( ((RC == 0)) && echo ok || echo FAILED) exit=${RC}"
echo "sacct:"
sacct -j "${SLURM_JOB_ID:-0}" -n -P -o JobID,State,Elapsed,MaxRSS,NodeList 2>&1 | head -n 5
echo "=========================================="

exit ${RC}
