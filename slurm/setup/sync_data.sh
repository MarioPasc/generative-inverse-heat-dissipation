#!/usr/bin/env bash
# Copy the four validated datasets to Picasso and verify them by hash (T3.1 step 3).
#
# Runs on the WORKSTATION, not on the cluster:
#
#   bash slurm/setup/sync_data.sh --dry-run    # rsync --dry-run, no writes
#   bash slurm/setup/sync_data.sh              # copy, then verify
#   bash slurm/setup/sync_data.sh --verify     # verify only, copy nothing
#
# Each dataset folder is ~146 MB (`images.npy`, `index.csv`, `splits.json`, `meta.json`, `qc/`).
# The parent's `_cache/` is never copied: it is preprocessing scratch, not data. Verification
# compares the remote `sha256sum images.npy` against the LOCAL `meta.json.sha256_images`, so it
# checks the transfer against the value recorded when the dataset was written, not against a
# hash recomputed on either side.

set -euo pipefail

LOCAL_ROOT="${IHDM_DATA_ROOT:-/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project}"
REMOTE_HOST="${IHDM_PICASSO_HOST:-picasso}"
REMOTE_ROOT="${IHDM_PICASSO_DATA_ROOT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/spectral_allocation_heat_diffusion_project}"
DATASETS=(ixi oasis1 lsun_church lsun_bedroom)
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=60)

MODE="sync"
case "${1:-}" in
    --dry-run) MODE="dry-run" ;;
    --verify)  MODE="verify" ;;
    "")        ;;
    *) echo "usage: $0 [--dry-run|--verify]" >&2; exit 2 ;;
esac

echo "local  : ${LOCAL_ROOT}"
echo "remote : ${REMOTE_HOST}:${REMOTE_ROOT}"
echo "mode   : ${MODE}"
echo ""

for dataset in "${DATASETS[@]}"; do
    [[ -d "${LOCAL_ROOT}/${dataset}" ]] || { echo "[FATAL] missing local dataset ${dataset}" >&2; exit 1; }
done

if [[ "${MODE}" != "verify" ]]; then
    echo "=== quota before ==="
    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" 'quota'

    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}'"

    RSYNC_FLAGS=(-avh --progress --exclude '_cache/')
    [[ "${MODE}" == "dry-run" ]] && RSYNC_FLAGS+=(--dry-run)

    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "=== rsync ${dataset} ==="
        rsync "${RSYNC_FLAGS[@]}" -e "ssh ${SSH_OPTS[*]}" \
            "${LOCAL_ROOT}/${dataset}/" \
            "${REMOTE_HOST}:${REMOTE_ROOT}/${dataset}/"
    done

    if [[ "${MODE}" == "dry-run" ]]; then
        echo ""
        echo "dry run: nothing was written, nothing to verify"
        exit 0
    fi

    echo ""
    echo "=== quota after ==="
    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" 'quota'
fi

echo ""
echo "=== sha256 verification (remote images.npy vs local meta.json.sha256_images) ==="

REMOTE_HASHES=$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" \
    "cd '${REMOTE_ROOT}' && for d in ${DATASETS[*]}; do printf '%s ' \"\$d\"; sha256sum \"\$d/images.npy\" 2>/dev/null | cut -d' ' -f1 || echo MISSING; done")

status=0
for dataset in "${DATASETS[@]}"; do
    expected=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256_images'])" \
        "${LOCAL_ROOT}/${dataset}/meta.json")
    actual=$(awk -v d="${dataset}" '$1 == d {print $2}' <<<"${REMOTE_HASHES}")
    if [[ "${expected}" == "${actual}" && -n "${actual}" ]]; then
        echo "OK   ${dataset}  ${actual}"
    else
        echo "FAIL ${dataset}  expected ${expected}  remote ${actual:-<none>}"
        status=1
    fi
done

echo ""
if (( status == 0 )); then
    echo "all four datasets match their recorded hashes"
else
    echo "HASH MISMATCH: re-run the rsync for the datasets marked FAIL" >&2
fi
exit "${status}"
