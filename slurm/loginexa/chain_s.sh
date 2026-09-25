#!/usr/bin/env bash
# T3.4 -- chain 25-minute loginexa sessions of one run of stability check S, from the WORKSTATION.
#
#   bash slurm/loginexa/chain_s.sh <gpu> <dataset,arm> <seed> [lr]
#
# Every POLL seconds it asks loginexa for the run's state (`harness.sh s-status`) and launches the
# next session (`harness.sh s`, which resumes from checkpoints-meta/checkpoint.pth) when the
# previous one has ended. It stops on DONE (exit 0) or on a session that ended with anything but
# 0 / 124 (timeout) / 75 (GPU not free), e.g. 3 (non-finite guard) or 4 (recipe check) (exit 1).
# A GPU taken by another user is waited for, never shared.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO="${LX_REPO:-/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/wt/T3.4}"
GPU="${1:?gpu}"; SPEC="${2:?dataset,arm}"; SEED="${3:?seed}"; LR="${4:-}"
POLL="${POLL:-90}"

status() {
    ssh -o ConnectTimeout=30 picasso \
        "ssh loginexa 'bash ${REMOTE_REPO}/slurm/loginexa/harness.sh s-status ${GPU} ${SPEC} ${SEED} ${LR}'" 2>/dev/null \
        | tail -n 1
}

while true; do
    STATE="$(status)"
    echo "$(date +%H:%M:%S) ${STATE:-<no answer>}"
    case "${STATE}" in
        DONE*)    exit 0 ;;
        FAILED*)  exit 1 ;;
        IDLE*)    bash "${HERE}/launch.sh" s "${GPU}" "${SPEC}" "${SEED}" ${LR:+"${LR}"} ;;
        BUSY*)    sleep 240 ;;
        *)        ;;  # RUNNING, or a transient ssh failure
    esac
    sleep "${POLL}"
done
