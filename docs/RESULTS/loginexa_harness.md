# loginexa harness of the training and logging path (T3.4, D19)

The production training path (`slurm/array/train_array.sbatch` → `train.py` → `scripts/losses.py`
→ `ihdm/train/*`, recipe v2 of `configs/spectral/arms.py`) exercised on Picasso's loginexa node
before the v2 array is resubmitted. Procedure: `docs/HARNESSES/training.md` §7; scripts:
`slurm/loginexa/`. Every item below quotes its log verbatim (trimmed to the verdict lines; the
full logs are in `~/execs/ihdm/logs/loginexa/` on Picasso).

## Environment

| item | value |
|---|---|
| node | loginexa, 4× Tesla V100-DGXS-32GB (compute capability 7.0, 32768 MiB), driver 580.159.04, 20 cores, 251 GB RAM, kernel 6.8.0-88; shared: GPUs 0 and 1 held 17.4 GB and 13.5 GB of another user's work all day and were never used; GPUs 2 and 3 only |
| python | `~/execs/ihdm/overlay/ihdm-v100/bin/python` (3.11.16): `--system-site-packages` venv over `fscratch/conda_envs/ihdm`, holding torch 2.14.0+cu126 and torchvision 0.29.0+cu126 (arch list `sm_50 … sm_90`), cuDNN 9.10.2, CUDA 12.6 runtime wheels; every other package from the cluster env (numpy 2.4.6, …) |
| overlay build | `slurm/loginexa/build_overlay.sh`, 2 min on loginexa; 13,091 files installed, 3,231 after pruning the C++ headers (torch/include 9,529); 5.2 GB; 5,246 inodes with directories |
| `$HOME` file count (`quota`) | **18.6k before** (09:05, before any write of this ticket) → **24.1k after** the overlay (+ the rsynced tree, 288 entries) — soft limit 35k |
| FSCRATCH file count | 248.8k before; 249.8k right after the first overlay build (892 `.pyc` + 118 `__pycache__` written into the base env by the build's imports), **248.8k after the revert**; nothing written there since (`PYTHONPYCACHEPREFIX` on loginexa's `/tmp`) |
| environment of every process | `slurm/loginexa/common.sh`: `IHDM_DATA_ROOT=<fscratch>/datasets/spectral_allocation_heat_diffusion_project` (read only), `PYTHONPATH=~/execs/ihdm/wt/T3.4`, `PYTHONPYCACHEPREFIX=/tmp/ihdm_T3.4/pycache`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `OMP_NUM_THREADS=4`; `timeout 25m` on every launch, a GPU only if `nvidia-smi` shows < 1000 MiB |
| run directories | H1–H7 on loginexa's local `/tmp/ihdm_T3.4/…` (removed at the end); the S runs in `~/execs/ihdm/loginexa_runs/` |

## Summary

| item | verdict | evidence (below) |
|---|---|---|
| H1 environment | (pending) | |
| H2 all 30 cells through the real worker | **PASS** (30/30) | `H2[i] check_run PASS`, `status=ok exit=0` ×30 |
| H3 artefacts and cadence, MRI and photograph cell | **PASS** (2/2) | `H3 check_run PASS` ×2 |
| H4 resume, extension, recipe check | (pending) | |
| H5 skip and abort paths on the CUDA scaler | (pending) | |
| H6 downstream (`load_ema_model`, `evaluate_run`) | (pending) | |
| H7 V100 memory and throughput at batch 16 | **PASS** | 0.87–0.88 it/s, 28.52 GiB peak |
| S (pre-registered stability check) | **PASS at lr 1e-4** | `docs/RESULTS/nan_diagnosis.md` §4 |

## H3 — full artefacts and cadence (300 iterations, shortened cadence)

`harness.sh h3 <gpu> <cell>`: `train.py --config configs/spectral/arms.py:<cell> --config.seed=1
--config.training.n_iters=300 --config.training.log_every=10 --config.training.eval_every=50
--config.training.ckpt_every=100 --config.training.resume_every=50 --config.training.grid_every=100`,
then `check_run.py <run> --data-root $IHDM_DATA_ROOT --lr 1e-4` (the line-by-line schema of every
kind, strict JSON, warm-up lr against 1e-4, pre-clip `grad_norm` varying, `amp_scale` a power of
two on every train line, cadence of every kind, manifest keys and schedule/data hashes against the
file, `schedules.json` and `meta.json`, EMA checkpoint keys and prefixes, `full_final.pt` keys and
step, grids, `DONE`). Code `677cec4`. Verbatim (`h3_ixi_A0_gpu2_20260925_111815.log`):

```
H3 train rc=0 seconds=721
-- expectation RunExpectation(n_iters=300, log_every=10, eval_every=50, ckpt_every=100, grid_every=100, batch_size=16, lr=0.0001, warmup=1000, amp=True, gpu=True, allow_skips=False, expect_done=True)
top: ['DONE', 'checkpoints', 'checkpoints-meta', 'config.json', 'grids', 'manifest.json', 'metrics.jsonl', 'tensorboard']
checkpoints: ['ema_iter_000100.pt (233.1 MiB)', 'ema_iter_000200.pt (233.1 MiB)', 'ema_iter_000300.pt (233.1 MiB)', 'full_final.pt (932.5 MiB)']
checkpoints-meta: ['checkpoint.pth (932.5 MiB)']
grids: ['iter_000100.png (0.6 MiB)', 'iter_000200.png (0.5 MiB)', 'iter_000300.png (0.6 MiB)', 'seeds.npy (0.3 MiB)']
tail: {"step": 300, "kind": "done", "n_skipped": 0}
manifest: {'run_id': 'ixi_A0_s1', 'git_sha': 'unknown', 'gpu': 'Tesla V100-DGXS-32GB', 'torch': '2.14.0+cu126', 'cuda': '12.6', 'n_params': 61056257, 'batch_size': 16, 'n_iters': 300, ...}
manifest schedule: log_W2 ad9c8c1163f5d7904587490fc107d4497039f1aafd1c8090a41ad11c2351bdb0
manifest data.images_sha256: b666e407e9afcc03fe2c51eb55ed99b0b72a34c80515e78874d64e155864bf4b
-- H-TRAIN §3 cadence: {'train': (31, [0, 10, 20], 300), 'eval': (7, [0, 50, 100], 300), 'ckpt': (3, [100, 200, 300], 300), 'grid': (3, [100, 200, 300], 300), 'done': (1, [300], 300)}
H3 check_run PASS /tmp/ihdm_T3.4/h3/ixi_A0_s1 (0 problems)
```

and (`h3_lsun_church_A3_gpu3_20260925_112242.log`): `H3 train rc=0 seconds=714`, the same listing,
cadence `{'train': (31, …, 300), 'eval': (7, …), 'ckpt': (3, [100, 200, 300], 300), 'grid': (3, …),
'done': (1, [300], 300)}`, **`H3 check_run PASS /tmp/ihdm_T3.4/h3/lsun_church_A3_s1 (0 problems)`**.

Selected train lines (`loss_per_octave` omitted; `git_sha` is `unknown` because the rsynced tree has
no `.git`, the SHA of each session is in its banner):

| cell | step | loss | it/s | peak GiB | pre-clip `grad_norm` | `amp_scale` |
|---|---|---|---|---|---|---|
| ixi A0 | 0 | 3.8487 | 0.60 | 27.83 | null (fp16 gradient overflow at 65,536) | 32768 |
| ixi A0 | 10 | 3.9612 | 0.90 | 28.52 | 409.7 | 512 |
| ixi A0 | 100 | 1.2192 | 0.88 | 28.52 | 147.5 | 256 |
| ixi A0 | 300 | 0.3779 | 0.88 | 28.52 | 657.1 | 256 |
| lsun A3 | 0 | 3.7345 | 0.60 | 27.83 | null | 32768 |
| lsun A3 | 10 | 4.2624 | 0.91 | 28.52 | 175.2 | 1024 |
| lsun A3 | 300 | 0.5072 | 0.88 | 28.52 | 617.3 | 256 |

30 distinct non-null pre-clip norms in 31 train lines per run (one null, step 0); eval loss
3.88 → 0.375 (ixi) and 3.73 → 0.713 (lsun) over 300 steps; `loss_per_octave` at 300 is finite in
every octave the schedule reaches and `null` above it (lsun A3 stops at 24 px: `32-64` and
`64-96` null). Grids: `grids/iter_000300.png` of ixi was looked at — the eight training seeds on
top, eight EMA samples below with head-shaped blobs and dark background after 300 iterations
(EMA 0.999), no NaN/black/saturated tile.

## H7 — V100 memory and throughput at batch 16

From every train line of the S runs (4,000 steps each, production recipe, fp16 autocast) and H3:
**0.868 it/s** (ixi, median of 50-step windows; 0.857–0.896) and **0.873 it/s** (lsun,
0.862–0.900), 13.9–14.1 img/s, against 1.71 it/s on the A100 (T3.2) — the V100 is 1.96× slower;
**peak allocated 28.516 GiB** in every window from step 10 on (27.83 GiB in the first window),
the A100's 28.544 GiB within 0.1 %; `nvidia-smi` showed 30,189 MiB used of 32,768 during S
(expandable segments on). A 40k-step run would take ≈ 12.8 h on a V100; the A100 array's
`--time=11:00:00` is sized from the A100's 6.84 h and is unaffected.

## H2 — all 30 cells through the real worker (`N_ITERS=2`)

`harness.sh h2 <gpu> <first> <last>` runs, for each index, the production worker exactly as SLURM
would, with the SLURM variables set by hand and only the paths pointed at the harness:

```bash
IHDM_REPO_DIR=~/execs/ihdm/wt/T3.4 IHDM_ENV_PREFIX=~/execs/ihdm/overlay/ihdm-v100 \
IHDM_RUN_ROOT=/tmp/ihdm_T3.4/h2_gpu<g> N_ITERS=2 SLURM_ARRAY_TASK_ID=<i> SLURM_ARRAY_JOB_ID=h2 \
SLURM_JOB_ID=h2_<i> SLURM_CPUS_PER_TASK=4 timeout 6m bash slurm/array/train_array.sbatch
```

then `check_run.py <run> --data-root $IHDM_DATA_ROOT --lr 1e-4` (with the production cadence at
`n_iters=2`: a train and an eval line at step 0, a checkpoint and a grid at step 2, `done` at 2,
`full_final.pt`, `DONE`), and deletes the run directory (≈ 2.1 GB) before the next cell. Four
sessions (cells 0–7 and 16–23 on GPU 2, 8–15 and 24–29 on GPU 3), 11:31–12:04, code `21adc56`–
`e284287` (the trainer is identical across them). Every cell's worker log shows the decoded row
(`CELL index=i run_id=…`, checked against `cells.csv` by the worker itself), `Pycache:
/tmp/ihdm_pycache_h2_<i>`, and `END TASK … status=ok exit=0`. **30 of 30 PASS**:

| index | run_id (decoded) | tier | worker | schedule (sha256) | data sha256 | step-0 loss | check_run | s |
|---|---|---|---|---|---|---|---|---|
| 0 | `ixi_A0_s1` | 1 | ok (exit 0) | `log_W2` ad9c8c1163f5… | b666e407e9af… | finite (value not printed by the first H2 session) | PASS | 119 |
| 1 | `ixi_A0_s2` | 1 | ok (exit 0) | `log_W2` ad9c8c1163f5… | b666e407e9af… | finite (value not printed by the first H2 session) | PASS | 115 |
| 2 | `ixi_A0_s3` | 1 | ok (exit 0) | `log_W2` ad9c8c1163f5… | b666e407e9af… | finite (value not printed by the first H2 session) | PASS | 116 |
| 3 | `lsun_church_A0_s1` | 1 | ok (exit 0) | `log_W2` ad9c8c1163f5… | 299c076853b0… | finite (value not printed by the first H2 session) | PASS | 116 |
| 4 | `lsun_church_A0_s2` | 1 | ok (exit 0) | `log_W2` ad9c8c1163f5… | 299c076853b0… | finite (value not printed by the first H2 session) | PASS | 115 |
| 5 | `lsun_church_A0_s3` | 1 | ok (exit 0) | `log_W2` ad9c8c1163f5… | 299c076853b0… | finite (value not printed by the first H2 session) | PASS | 115 |
| 6 | `ixi_A3_s1` | 1 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | b666e407e9af… | finite (value not printed by the first H2 session) | PASS | 116 |
| 7 | `ixi_A3_s2` | 1 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | b666e407e9af… | finite (value not printed by the first H2 session) | PASS | 115 |
| 8 | `ixi_A3_s3` | 1 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | b666e407e9af… | 3.7507593631744385 | PASS | 113 |
| 9 | `lsun_church_A3_s1` | 1 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | 299c076853b0… | 3.734490156173706 | PASS | 114 |
| 10 | `lsun_church_A3_s2` | 1 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | 299c076853b0… | 3.7439205646514893 | PASS | 114 |
| 11 | `lsun_church_A3_s3` | 1 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | 299c076853b0… | 3.7352845668792725 | PASS | 115 |
| 12 | `ixi_A1_s1` | 2 | ok (exit 0) | `log_W8` 840237df58f9… | b666e407e9af… | 3.751293182373047 | PASS | 114 |
| 13 | `ixi_A1_s2` | 2 | ok (exit 0) | `log_W8` 840237df58f9… | b666e407e9af… | 3.7673168182373047 | PASS | 113 |
| 14 | `lsun_church_A1_s1` | 2 | ok (exit 0) | `log_W8` 840237df58f9… | 299c076853b0… | 3.729823112487793 | PASS | 114 |
| 15 | `lsun_church_A1_s2` | 2 | ok (exit 0) | `log_W8` 840237df58f9… | 299c076853b0… | 3.741638422012329 | PASS | 114 |
| 16 | `ixi_A2_s1` | 2 | ok (exit 0) | `ixi_W2` 9c80d11c6141… | b666e407e9af… | 3.838407516479492 | PASS | 116 |
| 17 | `ixi_A2_s2` | 2 | ok (exit 0) | `ixi_W2` 9c80d11c6141… | b666e407e9af… | 3.9464807510375977 | PASS | 115 |
| 18 | `lsun_church_A2_s1` | 2 | ok (exit 0) | `ixi_W2` 9c80d11c6141… | 299c076853b0… | 3.7999520301818848 | PASS | 115 |
| 19 | `lsun_church_A2_s2` | 2 | ok (exit 0) | `ixi_W2` 9c80d11c6141… | 299c076853b0… | 3.791059970855713 | PASS | 115 |
| 20 | `lsun_church_A2p_s1` | 2 | ok (exit 0) | `lsun_church_W2` fa28a66e71c6… | 299c076853b0… | 3.8006129264831543 | PASS | 116 |
| 21 | `lsun_church_A2p_s2` | 2 | ok (exit 0) | `lsun_church_W2` fa28a66e71c6… | 299c076853b0… | 3.8053805828094482 | PASS | 115 |
| 22 | `oasis1_A0_s1` | 3 | ok (exit 0) | `log_W2` ad9c8c1163f5… | 8396e1c113ab… | 3.801084518432617 | PASS | 115 |
| 23 | `oasis1_A0_s2` | 3 | ok (exit 0) | `log_W2` ad9c8c1163f5… | 8396e1c113ab… | 3.833141326904297 | PASS | 116 |
| 24 | `lsun_bedroom_A0_s1` | 3 | ok (exit 0) | `log_W2` ad9c8c1163f5… | 7e4c98d15446… | 3.792151927947998 | PASS | 114 |
| 25 | `lsun_bedroom_A0_s2` | 3 | ok (exit 0) | `log_W2` ad9c8c1163f5… | 7e4c98d15446… | 3.8098320960998535 | PASS | 114 |
| 26 | `oasis1_A3_s1` | 3 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | 8396e1c113ab… | 3.7523374557495117 | PASS | 115 |
| 27 | `oasis1_A3_s2` | 3 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | 8396e1c113ab… | 3.7607791423797607 | PASS | 114 |
| 28 | `lsun_bedroom_A3_s1` | 3 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | 7e4c98d15446… | 3.723694086074829 | PASS | 114 |
| 29 | `lsun_bedroom_A3_s2` | 3 | ok (exit 0) | `ixi_W8` 8bfe56bd7592… | 7e4c98d15446… | 3.739612102508545 | PASS | 113 |


The schedule of every arm is the one `04` §2 assigns (A0 `log_W2`, A1 `log_W8`, A2 `ixi_W2`, A3
`ixi_W8`, A2′ `lsun_church_W2`), and `check_run` verified each manifest hash against the file
and `schedules/schedules.json`; the data hash of every dataset (`ixi` b666…, `lsun_church` 299c…,
`oasis1` 8396…, `lsun_bedroom` 7e4c…) against its `meta.json`. Peak memory at step 0 is 27.83 GiB
in every cell (batch 16 of 192² on the V100); every step-0 loss is finite (3.72–3.95). Verbatim
sample (cell 29):

```
CELL         index=29 run_id=lsun_bedroom_A3_s2 dataset=lsun_bedroom arm=A3 seed=2 tier=3
END TASK     index=29 run_id=lsun_bedroom_A3_s2 status=ok exit=0
first train line: {"step": 0, "kind": "train", "loss": 3.739612102508545, "lr": 0.0, "it_per_s": 0.6019646549227528, "im…
H2[29] check_run PASS /tmp/ihdm_T3.4/h2_gpu3/lsun_bedroom_A3_s2 (0 problems)
H2 cell=29 worker_rc=0 seconds=113
```
