#!/usr/bin/env bash
# T3.4 -- launch one loginexa harness item from the WORKSTATION, detached, under `timeout 25m`.
#
#   bash slurm/loginexa/launch.sh <item> <gpu> [args...]     # prints LOG=<remote log path>
#
# loginexa is reachable only through picasso (`ssh picasso 'ssh loginexa ...'`). The item runs
# `slurm/loginexa/harness.sh` from the rsynced tree in $HOME; poll the printed log with
#   ssh picasso "tail -n 20 <log>"
set -euo pipefail

REMOTE_REPO="${LX_REPO:-/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/wt/T3.4}"
REMOTE_LOGS="${LX_LOGS:-/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/loginexa}"
ITEM="${1:?item}"; GPU="${2:?gpu}"; shift 2
TAG="$(printf '%s_' "${ITEM}" "$@" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-60)"
LOG="${REMOTE_LOGS}/${TAG}gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
ARGS="$(printf ' %q' "${ITEM}" "${GPU}" "$@")"

# `mkdir ...; nohup ... &` and not `mkdir ... && nohup ... &`: the latter backgrounds a subshell
# that keeps the ssh channel's stdout open until the item ends (the launch then blocks 25 min).
# shellcheck disable=SC2029  # expanded locally on purpose: every value is known here
ssh picasso "ssh loginexa 'mkdir -p ${REMOTE_LOGS}; nohup timeout 25m bash ${REMOTE_REPO}/slurm/loginexa/harness.sh${ARGS} > ${LOG} 2>&1 < /dev/null & sleep 3; test -s ${LOG} && echo started || echo \"log still empty after 3 s\"'"
echo "LOG=${LOG}"
