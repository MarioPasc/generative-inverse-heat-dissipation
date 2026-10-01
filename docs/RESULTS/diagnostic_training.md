# Diagnostic training runs (M7, T7.1) — submission record

Post-hoc diagnostic of the photograph failure, `docs/SPECIFICATIONS/M7-diagnostics/README.md`
(the reading rule is pre-registered there). Two runs, arm A0, seed 1, the 60k recipe of the 30
production runs; each changes one factor of `lsun_church_A0_s1`. Launcher and cells:
`slurm/diag_train/`; log: `docs/AGENT-LOGS/M7-diagnostics/T7.1-photo-diagnostic.md`.

| run_id | dataset | N (train / ref / seed) | side | σ_B,max | schedule |
|---|---|---|---|---|---|
| `lsun_church_r128_A0_s1` | `lsun_church_r128` | 4,000 (3,200 / 800 / 40) | 128 | 64 | `log_W2_128` |
| `lsun_church_n32k_A0_s1` | `lsun_church_n32k` | 32,800 (32,000 / 800 / 40) | 192 | 96 | `log_W2` |

Everything else is A0's recipe, unchanged: batch 16, lr 1e-4, warm-up 1,000, clip 1.0, fp16 AMP,
EMA 0.999, K = 200, σ = 0.01, δ = 1.25σ, σ_B,min = 0.5, log spacing, cadences (ckpt and grid every
2,500, eval every 500, rolling checkpoint every 500), 60,000 iterations, seed 1.

## 1. Inputs on Picasso

| what | where | check |
|---|---|---|
| code | `~/execs/ihdm/wt/T7.1` (`git ls-files` rsync, no `.git`), `GIT_SHA` file | `b483fcf9ae4df2a0be9ed5e53781ff4eb203f9b5` at the smoke |
| `lsun_church_r128` | `fscratch/datasets/spectral_allocation_heat_diffusion_project/lsun_church_r128/` (4 files) | `images.npy` 65,536,128 bytes, sha256 `d386aeebfc0096e5ce939179770ab9c813cd3a2ef3df1d98263692953a812333`, identical to the local copy; `validate_dataset` OK |
| `lsun_church_n32k` | `…/lsun_church_n32k/` (4 files) | `images.npy` 1,209,139,328 bytes, sha256 `d21f24623dc16770d72c1f491f04525f1c4a4b114439f55d46f1d09d0b7bb915`, identical; `validate_dataset` OK |
| run root | `fscratch/runs/ihdm_diag/` | separate from the production `fscratch/runs/ihdm/` (the launcher refuses that one) |

## 2. Smoke (A100, job 2550583, `N_ITERS=500`, QOS `short`)

Command (login node, from `~/execs/ihdm/wt/T7.1`):

```bash
N_ITERS=500 QOS=short TIME_LIMIT=00:45:00 ARRAY_SPEC=0-1 \
IHDM_RUN_ROOT=/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm_diag_smoke \
    bash slurm/diag_train/submit_diag.sh
```

| | `lsun_church_r128_A0_s1` (task 0) | `lsun_church_n32k_A0_s1` (task 1) |
|---|---|---|
| node, GPU | exa04, A100-SXM4-40GB | exa02, A100-SXM4-40GB |
| `CELL` line | `index=0 run_id=lsun_church_r128_A0_s1 dataset=lsun_church_r128 arm=A0 seed=1 tier=M7` | `index=1 run_id=lsun_church_n32k_A0_s1 dataset=lsun_church_n32k arm=A0 seed=1 tier=M7` |
| state, elapsed, train time | COMPLETED, 00:03:02, 179 s | COMPLETED, 00:05:55, 354 s |
| `END TASK` | `status=ok exit=0` | `status=ok exit=0` |
| it/s (median of the 9 log windows from step 100) | **3.665** (3.612–3.673) | **1.712** (1.711–1.714) |
| VRAM peak | 13.43 GB | 28.54 GB |
| train loss, step 0 → 500 | 1.686 → 0.354 | 3.814 → 0.848 |
| eval loss, step 0 → 500 | 1.719 → 0.309 | 3.904 → 0.579 |
| skips | 0 | 0 |
| config | `data.image_size` 128, σ_B,max 64.0, `log_W2_128` | `data.image_size` 192, σ_B,max 96.0, `log_W2` |
| grid PNG | `iter_000500.png` 1042 × 262 (8 tiles of 128²) | `iter_000500.png` 1554 × 390 (8 tiles of 192²) |
| `check_run` (`ihdm.train.validate_run`, data hash included) | **PASS, 0 problems** | **PASS, 0 problems** |
| `config_sha256` | `92f5dd4a5d5c…` | `181365c9aa98…` |

n32k's 1.712 it/s reproduces the 192² production rate (1.69–1.71, D16/T3.5), as expected for the
same model, batch and side. `manifest.json.git_sha` is `"unknown"` (no `.git` in the copy); the
commit is the `GIT_SHA` file above.

## 3. Time limits (measured it/s × 60,000 × 1.3, rounded up)

| run | 60,000 / it/s | × 1.3 | `--time` |
|---|---|---|---|
| `lsun_church_r128_A0_s1` | 4.55 h | 5.91 h | **06:00:00** |
| `lsun_church_n32k_A0_s1` | 9.74 h | 12.66 h | **13:00:00** |

The it/s excludes the eval, checkpoint and grid steps (≈ 120 evals, 24 grids, 120 rolling
checkpoints over 60k: roughly 20 min at 128², 30 min at 192²), which the 1.3 covers. A TIMEOUT is
recoverable: resubmitting the same command resumes from `checkpoints-meta/checkpoint.pth`.

## 4. Production submission (GO from `main` on 2026-10-01; submitted 09:39Z)

Both requests passed `sbatch --test-only` on 2026-10-01 (test jobs 2550715 and 2550716; Slurm's
worst-case start bound was 2026-10-16, while the smoke, whose bound was 16 h out, started within 5
min). From the workstation:

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/execs/ihdm/wt/T7.1 && TIME_LIMIT=06:00:00 ARRAY_SPEC=0 bash slurm/diag_train/submit_diag.sh'
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/execs/ihdm/wt/T7.1 && TIME_LIMIT=13:00:00 ARRAY_SPEC=1 bash slurm/diag_train/submit_diag.sh'
```

Defaults the launcher applies: `N_ITERS=60000`, `QOS=medium_uma`,
`IHDM_RUN_ROOT=/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm_diag`, data root
`fscratch/datasets/spectral_allocation_heat_diffusion_project`, `--cpus-per-task=8 --mem=32G
--constraint=a100 --gres=gpu:1 --account=tic_163_uma`, logs
`~/execs/ihdm/logs/diag_train_<jobid>_<index>.{out,err}`. The full `sbatch` lines are printed by
the launcher and in §2 of the log.

| run | array job id (task) | submitted (UTC) | `--time` | first-task check |
|---|---|---|---|---|
| `lsun_church_r128_A0_s1` | **2550808** (`2550808_0`) | 2026-10-01T09:39:05Z | 06:00:00 | **pending**: PENDING (`Priority`) at 10:10Z |
| `lsun_church_n32k_A0_s1` | **2550811** (`2550811_1`) | 2026-10-01T09:39:11Z | 13:00:00 | **pending**: PENDING (`Priority`) at 10:10Z |

**First-task check, still to do** (neither task started within the 30 min watched): when a task
runs, its `.out` must show `CELL index=<0|1> run_id=lsun_church_<r128|n32k>_A0_s1 …` and
`N_ITERS: 60000`; its first `metrics.jsonl` rows must be finite `train` lines at steps 0, 50, …
with `lr` ramping over the 1,000-step warm-up, and an `eval` line at step 0. Expect ≈ 3.66 it/s
(r128) and ≈ 1.71 it/s (n32k), and `config.json` with `data.image_size` 128 / 192. At the end,
`check_run.py <run> --data-root <data root> --lr 1e-4` must PASS (as in §2).

Both launches printed `GIT_SHA b483fcf9ae4df2a0be9ed5e53781ff4eb203f9b5`, `N_ITERS 60000`, QOS
`medium_uma`, run root `fscratch/runs/ihdm_diag`; the `--test-only` probes of the real submission
(2550807, 2550810) gave worst-case start bounds of 2026-10-13. Logs:
`~/execs/ihdm/logs/diag_train_2550808_0.{out,err}` and `diag_train_2550811_1.{out,err}`.

Monitoring:

```bash
ssh picasso 'sacct -j 2550808,2550811 -X -o JobID,State,Elapsed,Start,NodeList'
ssh picasso 'grep -m1 "^CELL" ~/execs/ihdm/logs/diag_train_2550808_0.out ~/execs/ihdm/logs/diag_train_2550811_1.out'
ssh picasso 'tail -n 2 /mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm_diag/lsun_church_*_A0_s1/metrics.jsonl'
```

## 5. Clean-up for `main`

The smoke runs (20 files, 4.1 GB) are no longer needed once this record is merged:

```bash
ssh picasso 'rm -rf /mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm_diag_smoke'
```
