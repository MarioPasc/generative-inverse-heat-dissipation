#!/usr/bin/env bash
# T3.4 -- run several loginexa harness items one after another on one GPU, from the WORKSTATION.
#
#   bash slurm/loginexa/queue.sh <gpu> <item>[:arg[:arg...]] ...
#   e.g. bash slurm/loginexa/queue.sh 2 diag:/mnt/.../fixtures/array_2408239/ixi_A0_s1 h3:ixi,A0 h2:0:7
#
# Before each item it waits until the GPU shows < 1000 MiB (never shares it); it then launches
# the item with launch.sh (detached, `timeout 25m` on loginexa) and waits until the item's
# process has ended. One loginexa session per item, so no session outlives the 30-minute limit.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO="${LX_REPO:-/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/wt/T3.4}"
GPU="${1:?gpu}"; shift
POLL="${POLL:-60}"

remote() { ssh -o ConnectTimeout=30 picasso "ssh loginexa '$1'" 2>/dev/null; }

for spec in "$@"; do
    IFS=: read -r -a PARTS <<<"${spec}"
    ITEM="${PARTS[0]}"; ARGS=("${PARTS[@]:1}")
    until [[ "$(remote "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU}" | tr -d ' ')" =~ ^[0-9]+$ ]] \
          && (( $(remote "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU}" | tr -d ' ') < 1000 )); do
        echo "$(date +%H:%M:%S) gpu ${GPU} busy; waiting before ${spec}"
        sleep 120
    done
    echo "$(date +%H:%M:%S) launch ${spec} on gpu ${GPU}"
    bash "${HERE}/launch.sh" "${ITEM}" "${GPU}" "${ARGS[@]}"
    sleep 20
    # "[h]arness": the regex matches the item's process but not the remote shell running pgrep.
    while remote "pgrep -f \"[h]arness.sh ${ITEM} ${GPU}\" > /dev/null && echo running" | grep -q running; do
        sleep "${POLL}"
    done
    echo "$(date +%H:%M:%S) finished ${spec}"
done
echo "$(date +%H:%M:%S) queue on gpu ${GPU} done"
