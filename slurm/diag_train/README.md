# `slurm/diag_train/` — the two M7 diagnostic training runs (T7.1)

Post-hoc diagnostic of the photograph failure (`docs/SPECIFICATIONS/M7-diagnostics/`): arm A0,
seed 1, the unchanged 60k recipe, on two datasets that each change one factor of `lsun_church`.
Submission record: `docs/RESULTS/diagnostic_training.md`; log:
`docs/AGENT-LOGS/M7-diagnostics/T7.1-photo-diagnostic.md`.

| index | run_id | what changes against `lsun_church_A0_s1` |
|---|---|---|
| 0 | `lsun_church_r128_A0_s1` | resolution and framing: the same 4,000 photos, whole scene resized to 128² (σ_B,max = 64, schedule `log_W2_128`) |
| 1 | `lsun_church_n32k_A0_s1` | data size: 32,000 train images of 192² native crops, the same 800 `ref` / 40 `seed` |

## Files

| file | what |
|---|---|
| `cells.csv` | the two rows, same columns as `slurm/array/cells.csv`; pinned to `configs.spectral.arms.DIAGNOSTIC_CELLS` by `tests/slurm/test_diag_cells.py` |
| `submit_diag.sh` | launcher (login node): checks the cells and the data, `sbatch --test-only`, submits, parses the job id |

There is no worker here: the launcher runs the production worker `slurm/array/train_array.sbatch`
unchanged, through the overrides it already honours — `IHDM_CELLS` (this table), `IHDM_REPO_DIR`
(the code copy `~/execs/ihdm/wt/T7.1`), `IHDM_RUN_ROOT` (`fscratch/runs/ihdm_diag`) and `N_ITERS`
(60,000). The recipe, the resume rule (a resubmission continues from
`checkpoints-meta/checkpoint.pth`; a finished run is skipped) and the exit codes (0 ok, 3
non-finite guard, 4 recipe mismatch) are therefore those of the 30 production runs. Logs are
`~/execs/ihdm/logs/diag_train_<arrayjobid>_<index>.{out,err}`. The launcher refuses the production
run root `fscratch/runs/ihdm`.

## Commands (login node, from `~/execs/ihdm/wt/T7.1`)

```bash
bash slurm/diag_train/submit_diag.sh --dry-run                       # what would be submitted
TIME_LIMIT=<r128> ARRAY_SPEC=0 bash slurm/diag_train/submit_diag.sh  # lsun_church_r128_A0_s1
TIME_LIMIT=<n32k> ARRAY_SPEC=1 bash slurm/diag_train/submit_diag.sh  # lsun_church_n32k_A0_s1
```

`TIME_LIMIT` has no default: each cell gets its own limit, measured it/s on an A100 × 60,000 × 1.3,
rounded up (the values and the job ids are in `docs/RESULTS/diagnostic_training.md`). One cell
per call, so the 128² run does not carry the 192² run's limit into the queue.

Smoke (its own run root; `QOS=short` caps at 2 h):

```bash
N_ITERS=500 QOS=short TIME_LIMIT=00:45:00 ARRAY_SPEC=0-1 \
IHDM_RUN_ROOT=/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm_diag_smoke \
    bash slurm/diag_train/submit_diag.sh
```

## Provenance caveat

The code copy is an `rsync` of the T7.1 worktree without `.git`, so the worker prints
`Git commit: n/a` and `manifest.json` records `git_sha: "unknown"`. The commit is in
`~/execs/ihdm/wt/T7.1/GIT_SHA`, printed by the launcher and recorded in the submission record.
