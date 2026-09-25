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
| `$HOME` file count (`quota`) | **18.6k before** (09:05, before any write of this ticket) → 24.1k after the overlay → **24.7k at the end** (12:59) — soft limit 35k. New entries since 09:05: 5,624 = overlay 5,246 + rsynced tree 309 + S runs 42 + loginexa logs 37 + the CUDA JIT cache `~/.nv/ComputeCache` 182 (written by the driver on the V100; left in place, regenerable) |
| FSCRATCH file count | 248.8k before; 249.8k right after the first overlay build (892 `.pyc` + 118 `__pycache__` written into the base env by the build's imports), **248.8k after the revert**; nothing written there since (`PYTHONPYCACHEPREFIX` on loginexa's `/tmp`) |
| environment of every process | `slurm/loginexa/common.sh`: `IHDM_DATA_ROOT=<fscratch>/datasets/spectral_allocation_heat_diffusion_project` (read only), `PYTHONPATH=~/execs/ihdm/wt/T3.4`, `PYTHONPYCACHEPREFIX=/tmp/ihdm_T3.4/pycache`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `OMP_NUM_THREADS=4`; `timeout 25m` on every launch, a GPU only if `nvidia-smi` shows < 1000 MiB |
| run directories | H1–H7 on loginexa's local `/tmp/ihdm_T3.4/…` and the worker's `/tmp/ihdm_pycache_h2_*`: **removed at 12:59** (`ls -d /tmp/ihdm*` → 0); the S runs kept in `~/execs/ihdm/loginexa_runs/S_lr1e-4/` (4.6 GB, 42 entries: evidence and a v2 checkpoint at 4,000), the overlay kept for the next harness run (5.3 GB), logs in `~/execs/ihdm/logs/loginexa/` |

## Summary

| item | verdict | evidence (below) |
|---|---|---|
| H1 environment | **PASS** | `H1 overall PASS`, `185 passed` on loginexa |
| H2 all 30 cells through the real worker | **PASS** (30/30) | `H2[i] check_run PASS`, `status=ok exit=0` ×30 |
| H3 artefacts and cadence, MRI and photograph cell | **PASS** (2/2) | `H3 check_run PASS` ×2 |
| H4 resume, extension, recipe check | **PASS** | resume at 151, 301, 401; `rc=4 run_dir=identical` ×4 |
| H5 skip and abort paths on the CUDA scaler | **PASS** | skips 25, 33 then `done n_skipped 2`; abort at 10 consecutive (exit 3); carried 10; disabled scaler exit 3 |
| H6 downstream (`load_ema_model`, `evaluate_run`) | **PASS** | both loads finite; `evaluate_run` and `--gate` rc=0, strict JSON, no NaN (4 documented nulls) |
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

## H1 — environment

`harness.sh h1 3` (`slurm/loginexa/h1_env.py` then `pytest tests/train` with the overlay python),
code `e284287`, `h1_gpu3_20260925_121415.log`. The first H1 run (09:26, `bdd6e00`) had already
shown the device, the fp16 step and pytest green, and failed only its own pre-clip-norm check,
which looked at 5 steps while the scaler needs 7 to find its scale; the check was lengthened to 25
steps and rerun. Verbatim:

```
H1 device PASS Tesla V100-DGXS-32GB capability=(7, 0) torch=2.14.0+cu126 cuda=12.6 arch=['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90'] cudnn=91002
H1 step=1 loss=3.8487 grad_norm_preclip=inf amp_scale=32768 dt=1.65s
…
H1 step=7 loss=3.9628 grad_norm_preclip=inf amp_scale=512 dt=1.09s
H1 step=8 loss=3.9381 grad_norm_preclip=834.2 amp_scale=512 dt=1.18s
…
H1 step=25 loss=3.8310 grad_norm_preclip=292.9 amp_scale=512 dt=1.11s
H1 train_step_fp16_b16 PASS batch=(16, 1, 192, 192) dtype_autocast=fp16 losses=[3.8449, 3.8516, 3.8261, 3.8422, 3.831] (last 5) peak_alloc_gib=28.29 reserved_gib=28.86 step_s=1.110 (mean of steps 6-25, data loading included)
H1 grad_norm_hook PASS pre-clip norms=['inf', 'inf', 'inf', 'inf', 'inf', 'inf', 'inf', '834.2', '623.3', '386', '512.8', '463.1', '281.4', '183.5', '420.4', '249.5', '669', '375.6', '608.2', '288.9', '355.9', '239.4', '207.9', '253.9', '292.9'] (inf while the GradScaler is finding its scale, then finite, varying, not capped at grad_clip=1.0)
H1 amp_scale PASS scales=[32768.0, 16384.0, 8192.0, 4096.0, 2048.0, 1024.0, 512.0, 512.0, …, 512.0]
H1 eval_step PASS eval_loss=3.9368
H1 overall PASS
---- pytest tests/train (the overlay python) ----
185 passed in 101.72s (0:01:41)
H1 pytest exit=0
```

The five production steps through the released `create_model` (`DataParallel`) and
`scripts.losses.get_step_fn` run the real U-Net (61,056,257 parameters) in fp16 autocast at
batch 16 on real `ixi` batches; the step-1 loss 3.8487 equals the S run's and H3's step-0 loss
(same seed, same data order).

## H4 — resume after SIGTERM, extension, recipe check

`harness.sh h4 2 ixi,A0`, code `e284287`, `h4_ixi_A0_gpu2_20260925_120409.log`, H3's shortened
cadence (`resume_every=50`). Verbatim:

```
---- H4a: start n_iters=300, SIGTERM once a train line past step 170 is written
H4a SIGTERM after train step 170: rc=143
H4a rolling checkpoint step: 151
---- H4b: the same command resumes
H4b rc=0
{"step": 151, "kind": "resume", "from": "/tmp/ihdm_T3.4/h4/ixi_A0_s1/checkpoints-meta/checkpoint.pth"}
{"step": 160, "kind": "train", "loss": 0.5446709394454956, "lr": 1.6000000000000003e-05, …}
-- H-TRAIN §3 cadence: {'train': (33, [0, 10, 20], 300), 'eval': (7, [0, 50, 100], 300), 'ckpt': (3, [100, 200, 300], 300), 'grid': (3, [100, 200, 300], 300), 'resume': (1, [151], 151), 'done': (1, [300], 300)}
H4b check_run PASS /tmp/ihdm_T3.4/h4/ixi_A0_s1 (0 problems)
---- H4c: extension 300 -> 400
H4c rc=0
{"step": 301, "kind": "resume", "from": "/tmp/ihdm_T3.4/h4/ixi_A0_s1/checkpoints-meta/checkpoint.pth"}
-- H-TRAIN §3 cadence: {'train': (43, [0, 10, 20], 400), 'eval': (9, [0, 50, 100], 400), 'ckpt': (4, [100, 200, 300], 400), 'grid': (4, [100, 200, 300], 400), 'resume': (2, [151, 301], 301), 'done': (2, [300, 400], 400)}
H4c check_run PASS /tmp/ihdm_T3.4/h4/ixi_A0_s1 (0 problems)
---- H4e: the clone and the data root moved (same code, same images): 400 -> 410 resumes
H4e moved clone+data rc=0
{"step": 401, "kind": "resume", "from": "/tmp/ihdm_T3.4/h4/ixi_A0_s1/checkpoints-meta/checkpoint.pth"}
    "root": "/tmp/ihdm_T3.4/h4_moved/data",
    "file": "/tmp/ihdm_T3.4/h4_moved/clone/schedules/log_W2.npy",
---- H4f: a data root with other images (meta.json sha256_images changed) is refused
E0925 12:23:20.875733 … train.py:156] Refusing to resume: dataset content of /tmp/ihdm_T3.4/h4/ixi_A0_s1 differs from this invocation's: data images_sha256: b666e407e9afcc03... -> ffff0000... (/tmp/ihdm_T3.4/h4_moved/data/ixi/meta.json)
H4f other images rc=4 run_dir=identical
---- H4d: recipe mismatch --config.optim.lr=5e-5
E0925 12:23:50.722215 … train.py:156] Refusing to resume: recipe of /tmp/ihdm_T3.4/h4/ixi_A0_s1 (manifest recipe_sha256 9e3ec7add745) differs from this invocation's (694ac506b10b): optim.lr: 0.0001 -> 5e-05
H4d --config.optim.lr=5e-5 rc=4 run_dir=identical files=21
---- H4d: recipe mismatch --config.training.batch_size=8
E0925 12:24:20.182868 … train.py:156] Refusing to resume: recipe of … differs from this invocation's (8179bdefa976): training.batch_size: 16 -> 8
H4d --config.training.batch_size=8 rc=4 run_dir=identical files=21
---- H4d: recipe mismatch --config.seed=2
E0925 12:24:49.733294 … train.py:156] Refusing to resume: recipe of … differs from this invocation's (6fdc23a79a25): seed: 1 -> 2
H4d --config.seed=2 rc=4 run_dir=identical files=21
```

The SIGTERM (exit 143) left the rolling checkpoint of step 150 (`step` 151); the restart logged
exactly one `resume` at 151, replayed steps 151–170 (the duplicates the cadence check allows after
a resume) and finished; the extension continued the EMA numbering (`ema_iter_000400.pt`) with a
second `done`; a resume from a copied clone and a copied data root (same images) went through;
every refused resume exited 4 and left all 21 files of the run directory byte-identical and with
unchanged mtimes (`run_dir=identical`: size, mtime and sha256 of every file compared before and
after), with no `resume` line.

## H5 — skip path and abort path on the CUDA GradScaler

`harness.sh h5 3`, code `e284287`, `h5_gpu3_20260925_120415.log`; the non-finite losses are
injected by `slurm/loginexa/train_inject.py` (the loss times NaN inside the autocast region, at
chosen loop steps), `ixi,A0`, `log_every=5`. Verbatim:

```
---- H5a: two isolated non-finite losses (steps 25, 33) under the enabled scaler
H5a rc=0 (expect 0)
   {"step": 20, "kind": "train", "loss": 3.9509220123291016, "grad_norm": 236.93280029296875, "amp_scale": 512.0}
   {"step": 25, "kind": "skip", "loss": null, "n_skipped": 1, "consecutive": 1}
   {"step": 25, "kind": "train", "loss": 3.947147846221924, "grad_norm": null, "amp_scale": 256.0}
   {"step": 30, "kind": "train", "loss": 3.914700984954834, "grad_norm": 309.2106018066406, "amp_scale": 256.0}
   {"step": 33, "kind": "skip", "loss": null, "n_skipped": 2, "consecutive": 1}
   {"step": 35, "kind": "train", "loss": 3.754249095916748, "grad_norm": 306.876953125, "amp_scale": 128.0}
   {"step": 40, "kind": "done", "n_skipped": 2}
H5a check_run PASS /tmp/ihdm_T3.4/h5/skip (0 problems)
---- H5b: ten consecutive non-finite losses (steps 12-21)
H5b rc=3 (expect 3)
   {"step": 12, "kind": "skip", "loss": null, "n_skipped": 1, "consecutive": 1}
   …
   {"step": 20, "kind": "train", "loss": null, "grad_norm": null, "amp_scale": 1.0}
   {"step": 21, "kind": "skip", "loss": null, "n_skipped": 10, "consecutive": 10}
   {"step": 21, "kind": "abort", "loss": "nan", "n_skipped": 10, "consecutive": 10, "reason": "10 consecutive non-finite training losses (max_consecutive_skips=10)", "resume_saved": true}
H5b rolling checkpoint step 22
---- H5c: resume H5b without injection to n_iters=30 (the 10 skips carry over)
H5c rc=0 (expect 0)
   {"step": 30, "kind": "done", "n_skipped": 10}
---- H5d: disabled scaler (optim.automatic_mp=false, fp32, batch 4): abort at once
H5d rc=3 (expect 3)
   {"step": 12, "kind": "abort", "loss": "nan", "n_skipped": 0, "consecutive": 0, "reason": "non-finite training loss with the GradScaler disabled: the optimiser step applied the non-finite gradients", "resume_saved": false}
H5d rolling checkpoint step 11 finite True
```

Each injected non-finite loss halved the real CUDA scaler's scale (512 → 256 at step 25, → 128
after 33; ten consecutive: 512 → 1.0) — the step was a no-op; the train cadence kept its line at
every log step (`loss: null` for a window of skipped steps only); the tenth consecutive skip
aborted with exit 3 and kept the rolling checkpoint (step 22); the resume carried `n_skipped=10`
to the final `done`; with `optim.automatic_mp=false` the first non-finite loss aborted with exit 3
and left the previous rolling checkpoint (step 11) untouched and finite.

## H6 — downstream: `load_ema_model` and `evaluate_run` on the H3 checkpoints

`harness.sh h6 2 /tmp/ihdm_T3.4/h3/ixi_A0_s1`, code `e284287`,
`h6__tmp_ihdm_T3.4_h3_ixi_A0_s1_gpu2_20260925_122545.log`: `ihdm.sampling.loader.load_ema_model`
on `ema_iter_000200.pt` and `ema_iter_000300.pt` (with `load_run_config`), then the ticket's two
commands (`evaluate_run` is T5.1's, run unchanged). Verbatim:

```
H6 load_ema_model step=200 training=False n_params=61056257 out_finite=True
H6 load_ema_model step=300 training=False n_params=61056257 out_finite=True
H6 evaluate_run rc=0 seconds=476
2026-09-25 12:32:47,595 INFO ihdm.metrics.run_eval final set: 16 chains at step 300 in 132.4 s (8.273 s/chain) -> /tmp/ihdm_T3.4/h3/ixi_A0_s1/samples/000300/final
2026-09-25 12:33:43,846 INFO ihdm.metrics.run_eval heldout set: 6 chains at step 300 in 55.5 s (9.248 s/chain) -> /tmp/ihdm_T3.4/h3/ixi_A0_s1/samples/000300/heldout
OK ixi_A0_s1 dataset=ixi arm=A0 seed=1 steps=[200, 300] lsd={200: 0.3856, 300: 0.4497} t_tau=None final_lsd=0.44740589850027607 M=1.2177296119855652 D_pix=0.007858124063315087 kid=None fid=None n_ref=None -> /tmp/ihdm_T3.4/h3/ixi_A0_s1/metrics
H6 evaluate_run --gate rc=0 seconds=12
2026-09-25 12:34:06,630 INFO ihdm.metrics.run_eval gate 200 vs 300: LSD 0.3856 -> 0.4497, difference -0.0641 [-0.1131, -0.0181], extend=False
GATE [200, 300] difference=-0.06410 CI=[-0.11308, -0.01810] extend=False
H6 ckpt_000200.json: 781 bytes, top keys ['checkpoint', 'checkpoint_sha256', 'lsd', 'lsd_octaves', 'n_reference', 'n_samples', 'n_seeds', 'sample_batch', 'sample_rng_seed', 'samples_dir', 'seed_list', 'seed_list_sha256', 'step', 'variance_ratio'], null/nan leaves 0 []
H6 ckpt_000300.json: 778 bytes, top keys [...], null/nan leaves 0 []
H6 final.json: 3105 bytes, top keys ['M', 'M_lp', 'checkpoint', …], null/nan leaves 1 ['final.inception']
H6 gate.json: 589 bytes, top keys ['difference', 'extend', 'lsd_a', 'lsd_b', 'n_bins', 'n_reference', 'n_seeds', 'relative_change', 'rule', 'sample_batch', 'sample_rng_seed', 'seed_list_sha256', 'step_a', 'step_b'], null/nan leaves 0 []
H6 summary.json: 3576 bytes, top keys ['a0_final_lsd', 'checkpoint_selection', …, 't_tau', 't_tau_note', 'variance_ratio_by_step'], null/nan leaves 3 ['summary.a0_final_lsd', 'summary.final.inception', 'summary.t_tau']
```

Both commands exit 0 and write every file; every file is strict JSON (parsed with `NaN`/`Infinity`
rejected) and holds **no NaN**. The four `null` leaves are the documented "not computed" markers
of the options the ticket prescribes, each with its note in the same file: `final.inception`
(`"inception_note": "skipped (--skip-inception)"`) and `t_tau` / `a0_final_lsd` (`"t_tau_note":
"not computed: --a0-final-lsd was not given (T_tau needs the A0 run's final LSD)"`). No defect to
report to T5.1. The seed lists the command reads (`<dataset>/eval_seeds_*.npy`, created at 09:32 by
T5.1's work) already existed, so nothing was written on FSCRATCH (checked: no FSCRATCH entry newer
than 09:33 at 12:36). Sampling on the V100: 8.27 s per 200-step chain per image (A100: 3.21 s).

## What this harness does not cover

- The A100 itself: the V100 runs the same code and libraries except the CUDA build of torch
  (cu126 against the array's cu130) and cuDNN (9.10 against 9.24); fp16 range and the GradScaler
  semantics are the same, kernel choices and accumulation order are not.
- SLURM: the worker ran with bash and hand-set `SLURM_*` variables; `sbatch --test-only` of the
  launcher is `main`'s step (T3.3 validated it for array 1; the launcher is unchanged).
- The FSCRATCH run root and a continuous 40k-step run: every harness run wrote to loginexa's
  `/tmp` or `$HOME`, and the longest run was S (4,000 steps in four resumed sessions).
