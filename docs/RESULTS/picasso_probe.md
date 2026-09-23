# Picasso probe — measured result (T3.2)

One A100 job that trains A0 for three epochs on `lsun_church` and on `ixi`, measures peak memory
at batch 16 and 24, tests resume/extension, times the offline sampler at three sampling batches,
and verifies every artefact of `04-run-artifacts.md` §3. Every number below was read from the job's
own output on the cluster; nothing here is extrapolated except where the text says "derived", and
each derived figure states the measured quantities it is built from.

How to repeat it: `slurm/probe/README.md`. Trimmed job log:
`docs/RESULTS/picasso_probe/probe_2405546.log`.

## 1. The job

| field | value |
|---|---|
| job id | `2405546` (`sbatch --test-only` first: accepted, "Job 2405545 to start … on nodes exa01 in partition gpu_partition") |
| script | `slurm/probe/probe.sbatch`, submitted by `slurm/probe/submit_probe.sh` |
| repo / commit | `.../fscratch/repos/generative-inverse-heat-dissipation` on `ticket/T3.2-picasso-probe` at `29106ef` |
| request | `--constraint=a100 --gres=gpu:1 --cpus-per-task=8 --mem=48G --qos=short --time=01:50:00` |
| node / GPU | `exa02`, `NVIDIA A100-SXM4-40GB` (40,960 MiB; 39.52 GiB usable), driver 610.57.04 |
| environment | `.../fscratch/conda_envs/ihdm`, torch `2.14.0+cu130`, CUDA 13.0, Python 3.11.16 |
| submitted / started / ended | 09:11:28 / 11:02:19 / 11:37:10 CEST, 2026-09-23 |
| **queue wait** | **1 h 50 min 51 s** (`Reason=Priority` throughout; the initial estimate of 10:17 slipped twice) |
| **wall time** | **34 min 51 s**, `State=COMPLETED` |
| MaxRSS (`.batch` step) | 19,240,912 K = **18.35 GB** of the 48 GB requested → the array can ask for 32 GB |
| model | 61,056,257 parameters (`manifest.json`, both runs) |

`sacct -j 2405546 -n -P -o JobID,State,Elapsed,MaxRSS,NodeList` →
`2405546|COMPLETED|00:34:51||exa02`, `2405546.batch|COMPLETED|00:34:51|19240912K|exa02`.

Six of the seven steps succeeded. Step 3 (batch 24) failed with the OOM it was written to provoke;
that is the measurement, not a defect, and it did not abort the worker.

## 2. Training: throughput and memory at batch 16

Three epochs of the 3,200-image train split at batch 16 = 600 iterations. `it/s` and `img/s` are the
median of the logger's per-window values over steps ≥ 100 (the first windows include one-off
allocation and autotune costs); `peak GB` is the maximum of `gpu_mem_peak_gb`, which is
`torch.cuda.max_memory_allocated`, i.e. process-local and exclusive of the CUDA context and the
allocator's cache.

| run | iters | it/s | img/s | peak GB (torch) | device peak (`nvidia-smi`) | last train loss | last eval loss |
|---|---|---|---|---|---|---|---|
| `lsun_church_A0_s1` | 600 | **1.7093** | **27.349** | **28.544** | 33.29 GiB | 2.4896 @ 600 | **0.9119** @ 600 |
| `ixi_A0_s1` | 600 → 800 (resumed) | **1.7058** | **27.293** | **28.545** | 33.28 GiB | 1.0053 @ 800 | **0.4613** @ 800 |

An independent check that does not use the logger at all: between the end of the step-0 evaluation
(11:02:43.965) and the start of the step-200 evaluation (11:04:41.306) the trainer ran exactly 200
iterations in **117.341 s = 1.7043 it/s = 27.27 img/s**. The two routes agree to 0.3 %.

**The two datasets are indistinguishable in cost**, as they must be: at a fixed $192^2$ resolution
the activation footprint and the per-step FLOPs do not depend on the images. The 0.2 % gap between
them is run-to-run jitter.

**T2.3's memory model is confirmed.** The affine fit from the RTX 3060,
$\text{peak}(B) = 1.7046\,B + 1.2997$ GB, predicts **28.57 GB** at batch 16 against the
**28.544 / 28.545 GB** measured here — an error of **0.09 %**, on a different GPU, a different
driver and a different CUDA version. The per-sample activation cost of 1.705 GB is a property of
the network at this resolution, not of the card.

**Device-level memory is the number with less headroom than it looks.** The allocator peak is
28.54 GB, but `nvidia-smi` shows the process holding **33.29 GiB** of the 39.52 GiB usable during
training — the difference is the CUDA context plus the caching allocator's reserved-but-unallocated
blocks. Headroom at batch 16 is therefore **≈ 6.2 GiB**, not the ≈ 11 GiB the allocator figure
alone would suggest. It is still comfortable, and nothing in the array changes it.

### 2.1 Cost of each periodic operation (measured from the log timestamps)

These are what turn a per-iteration rate into a per-run wall time, and the probe's dense cadence is
what makes them separable.

| operation | cadence in the array | measured cost | evidence |
|---|---|---|---|
| training iteration (batch 16) | every step | **0.5868 s** (1.7043 it/s) | 200 iterations in 117.341 s |
| evaluation (25 `ref` batches of 16, EMA weights) | `eval_every=500` | **7.35 s** | 7.32, 7.28, 7.47 s (church); 7.31, 7.45, 7.36, 7.40 s (ixi) |
| sample grid (8 fixed seeds × 200 reverse steps + PNG) | `grid_every=2500` | **27.9 s** | 27.75 s and 28.0 s (church), 28.3 s (ixi), each isolated by subtracting the 200 iterations in the same interval |
| EMA checkpoint (233 MiB) | `ckpt_every=2500` | **1.4 s** | 1.29, 1.44, 1.66, 1.27, 1.39, 1.26, 1.30 s |
| rolling resume checkpoint (932 MiB full state) | `resume_every=500` | **1.0 s** | 12.68 s and 12.70 s for "save + 20 iterations", minus 11.72 s of iterations |
| process start → first training step | once | **17 s** | 11:02:19 worker start → 11:02:36.6 step-0 evaluation |

## 3. The batch-24 memory probe (D4″)

`ixi,A0` at `--config.training.batch_size=24 --config.eval.batch_size=24`, 40 iterations.

**Outcome: CUDA OOM at step 0, after 8 s.** No `metrics.jsonl` was written — the failure precedes
the first log event — so the evidence is the traceback, recorded verbatim in
`docs/RESULTS/picasso_probe/probe_2405546.log`:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 864.00 MiB. GPU 0 has a total
capacity of 39.52 GiB of which 730.88 MiB is free. Including non-PyTorch memory, this process has
38.79 GiB memory in use. Of the allocated memory 37.15 GiB is allocated by PyTorch, and 1.13 GiB
is reserved by PyTorch but unallocated.
```

So batch 24 had already allocated **37.15 GiB** and needed at least **38.0 GiB** to get through the
forward pass — against D4″'s threshold of **≤ 30 GB**, which it misses by more than 7 GiB, and
against the card's 39.52 GiB usable, which it also misses. T2.3's fit predicted 42.21 GB; the run
died at ≥ 38.0 GB, so the fit over-predicts the requirement by roughly 10 % while getting the
verdict — "does not fit" — exactly right.

**Batch 32 was therefore not attempted**: the specification only permits it if 24 peaks below
30 GB, and 24 did not run at all.

### Decision D4″ / D4‴ — applied

| quantity | value | status |
|---|---|---|
| `training.batch_size` | **16** | unchanged in `configs/spectral/arms.py` (it already read 16) |
| `eval.batch_size` | **16** (follows `training.batch_size`) | unchanged |
| `training.n_iters` | **40,000** | unchanged; the batch-24 alternative of 27,500 is not needed |

The only edit made to `arms.py` is the comment on the `batch_size` line, which now cites this
measurement instead of the pre-probe estimate. `pytest tests/train -q` → 105 passed.

## 4. Resume and extension

Step 2's exact command re-run with `--config.training.n_iters=800` on the same workdir.

- `metrics.jsonl` of `ixi_A0_s1` contains exactly one `resume` event:
  `{"from": ".../ixi_A0_s1/checkpoints-meta/checkpoint.pth", "kind": "resume", "step": 601}`.
- `checkpoints/ema_iter_000800.pt` exists (244,406,339 B, written 11:20).
- `ckpt` events at steps `[200, 400, 600, 800]`, `eval` at `[0, 200, 400, 600, 800]`,
  `grid` at `[200, 400, 600, 800]`, 41 `train` lines, **zero `abort` events**.
- The extension cost 165 s for 200 iterations including process start, one evaluation, one grid,
  one EMA checkpoint and `full_final.pt`.

**The resume step is 601, not 600.** `train.py` logs `initial_step = state['step']`, and `state`
is saved after the optimiser step, so the stored step is one past the last completed iteration.
The resumed loop is `range(601, 801)` — **no iteration is repeated and none is skipped**, which is
the behaviour the array needs. The ticket's wording ("a `resume` event at 600") is off by this
one-step convention, not the code.

## 5. Offline sampler on the A100

`ihdm.cli.sample_ckpt` on `ixi_A0_s1/checkpoints/ema_iter_000800.pt`, 200-step chains,
`start_level=200`, `prior_noise=True`, `delta=0.0125`, **no `--amp`** (matching T2.2's 3060
measurement so the two are comparable). `s/chain` is `timing.s_per_chain` from each `request.json`.

| step | source | chains | sampling batch | wall (s) | **s / chain / image** | samples/s | device peak |
|---|---|---|---|---|---|---|---|
| 5a | 4 held-out seeds × 8 | 32 | 32 | 103.57 | **3.236** | 0.3090 | 18.32 GiB |
| 5b | 128 train seeds × 1 | 128 | 64 | 410.41 | **3.206** | 0.3119 | 35.67 GiB |
| 5c | the same 128 train seeds | 128 | 128 | 410.87 | **3.210** | 0.3115 | **36.85 GiB** |

**The plateau is already reached at sampling batch 32.** Going to 64 buys 0.9 % and going to 128
buys 0.8 %, both inside the run-to-run spread of the three measurements (0.030 s, or 0.9 %). The
cost is set by the 200 sequential network evaluations, not by the batch — the same conclusion T2.2
drew on the 3060 at batch 20, now confirmed on hardware six times faster.

**Use batch 32 or 64 for the array's sampling jobs, never 128.** Batch 128 is no faster and leaves
only **3.1 GiB** of the card free (36.85 of 39.52 GiB), which is an OOM waiting for a slightly
larger checkpoint or a co-scheduled process; batch 32 leaves 21 GiB.

**Against the 3060**: 18.15 s/chain (T2.2, batch 20) → **3.21 s/chain**, a **5.65×** speedup.
Training scaled by **6.0×** (4.55 img/s at batch 4 on the 3060 → 27.3 img/s at batch 16 here). Both
are well above the 3–4× factor T2.1 assumed and T2.3 propagated, so every A100 estimate inherited
from that factor was **pessimistic**.

Artefacts of each sampler call: `samples.npy`, `seeds.npy`, `seed_idx.npy`, `request.json`,
`preview.png` — all five present in all three output directories.

## 6. Artefact check against `04-run-artifacts.md` §3

`ls -R` of both runs (full listing in the trimmed log):

| artefact required by §3 | `lsun_church_A0_s1` | `ixi_A0_s1` |
|---|---|---|
| `manifest.json` | present | present |
| `config.json` | present | present |
| `metrics.jsonl` | present (31 lines) | present (41 lines) |
| `tensorboard/` | 1 event file | 2 event files (one per invocation) |
| `checkpoints/ema_iter_XXXXXX.pt` | 200, 400, 600 | 200, 400, 600, **800** |
| `checkpoints/full_final.pt` | 977,764,597 B | 977,772,405 B |
| `checkpoints-meta/checkpoint.pth` | present | present |
| `grids/iter_XXXXXX.png` | 200, 400, 600 | 200, 400, 600, 800 |
| `grids/seeds.npy` | present | present |
| `DONE` | present | present |

**Every artefact of §3 is present in both runs.** `manifest.json` carries `run_id`, `git_sha`
`29106ef…`, `gpu` `NVIDIA A100-SXM4-40GB`, `torch` `2.14.0+cu130`, `cuda` `13.0`,
`slurm_job_id` `2405546`, `n_params` 61,056,257 and `batch_size` 16. Each `metrics.jsonl` train
line carries `loss`, `lr`, `it_per_s`, `img_per_s`, `gpu_mem_peak_gb`, `grad_norm`, `wall_s` and a
`loss_per_octave` dictionary finite in all eight octaves.

The `mem_b24` directory has `manifest.json`, `config.json`, `grids/seeds.npy` and an empty
`checkpoints/`: the OOM happened before the first log event, so there is no `metrics.jsonl` and no
`DONE`. That is the expected shape of a run that died at step 0.

Sample images: `docs/RESULTS/picasso_probe/lsun_church_A0_s1_grid_iter_000600.png`,
`ixi_A0_s1_grid_iter_000800.png`, `ixi_sampler_preview_b32.png`. At 600–800 iterations the network
has seen 3–4 epochs and the samples are not expected to be meaningful; the grids are here as
evidence that the grid writer runs on the cluster, not as a quality claim.

## 7. Derived budget for the 30-run array (T3.3)

### 7.1 One 40,000-iteration run at batch 16

Built from §2.1, with the array's cadences (`ckpt_every=2500`, `resume_every=500`,
`eval_every=500`, `grid_every=2500`):

| component | count | unit cost | total |
|---|---|---|---|
| training iterations | 40,000 | 0.5868 s | 23,470 s |
| evaluations | 81 (step 0 plus every 500) | 7.35 s | 595 s |
| sample grids | 16 | 27.9 s | 446 s |
| EMA checkpoints | 16 | 1.4 s | 22 s |
| rolling resume checkpoints | 80 | 1.0 s | 80 s |
| startup + `full_final.pt` + `DONE` | 1 | ≈ 22 s | 22 s |
| | | | **24,635 s = 6.84 h** |

> **A100-hours per 40k-iteration run at batch 16: 6.84 h.**
> Training alone is 6.52 h; the periodic artefacts add 4.7 %.

- **`--time` for the array:** 1.5 × 6.84 h = 10.26 h → **`--time=11:00:00`**, which requires QOS
  `medium` (3 days) — `short` caps at 2 h and cannot hold a training task.
- **Total A100-hours for 30 runs, training only: 30 × 6.84 = 205 A100-h.**
- `--mem`: 18.35 GB measured, so **`--mem=32G`** is enough (48 GB was 62 % wasted).
- Queue wait was 1 h 51 min for a single 8-core, 1-GPU task under `Reason=Priority`. A 30-task
  array will pay this per task, so the wall request should not be trimmed to the bone.

### 7.2 Sampling, at the measured 3.21 s per 200-step chain per image

Both plans are costed so `main` can choose; neither is chosen here.

**(a) `05-metrics.md` as written** — 16 checkpoints × 1k LSD samples, 2k at the final checkpoint,
40 × 50 diversity, 5k for FID/KID/$M$:

| set | chains |
|---|---|
| intermediate LSD, 16 checkpoints × 1,000 | 16,000 |
| final LSD, 2,000 | 2,000 |
| diversity, 40 seeds × 50 | 2,000 |
| FID / KID / memorisation, 5,000 | 5,000 |
| **total per run** | **25,000** |

25,000 × 3.21 s = 80,250 s = **22.29 A100-h per run** → **669 A100-h over 30 runs**.

**(b) Reduced plan** — 4 checkpoints × 500, one shared 2k final set serving both the final LSD and
the FID/KID/$M$ computation, 40 × 50 diversity:

| set | chains |
|---|---|
| intermediate LSD, 4 checkpoints × 500 | 2,000 |
| shared final set (LSD + FID/KID/$M$), 2,000 | 2,000 |
| diversity, 40 seeds × 50 | 2,000 |
| **total per run** | **6,000** |

6,000 × 3.21 s = 19,260 s = **5.35 A100-h per run** → **161 A100-h over 30 runs**.

### 7.3 The two totals side by side

| plan | training | sampling | **total A100-h** | sampling ÷ training |
|---|---|---|---|---|
| `05-metrics.md` as written | 205 | 669 | **874** | **3.26×** |
| reduced | 205 | 161 | **366** | **0.78×** |

**T2.3's warning survives the measurement, at a smaller magnitude.** T2.3 projected ≈ 32–42 A100-h
of sampling per run against ≈ 10–13 h of training, i.e. ≈ 3×; measured, it is **22.3 h against
6.8 h, still ≈ 3.3×**. The absolute numbers came down by a factor of ≈ 1.6 because the real A100
speedup is 5.65–6.0×, not the 3–4× that was assumed — but the *ratio*, which is what decides
whether the evaluation plan has to be cut, is unchanged. Sampling remains the dominant cost.

The reduced plan removes **508 A100-h** (58 % of the campaign) and costs: LSD-versus-iteration on 4
points instead of 16, which makes $T_\tau$ of `05-metrics.md` §2 a much coarser quantity (its
resolution becomes 10,000 iterations, not 2,500), 500-sample instead of 1,000-sample LSD at those
points, and FID/KID computed on 2k rather than 5k samples, which widens their bootstrap CIs. Those
are the three losses `main` is trading against the 508 hours; T3.2 does not choose between them.

## 8. Filesystem note for T3.3

`quota` read live on 2026-09-23 at 09:10 CEST showed FSCRATCH **over** its file quota (254.7k
against a 250.0k soft limit, 7-day grace running, hard limit 400.0k), not the ≈ 34k of headroom
`picasso_setup.md` §5 recorded the previous day. The orchestrator cleared the extracted conda
package cache and the pip cache the same morning, bringing it back to 247.5k / 250k with the grace
reset. Consequences for the array, per the orchestrator:

- **Budget under ~2k new files** — checkpoints only (30 runs × 16 EMA + `full_final.pt` + the
  rolling checkpoint ≈ 540 files, plus manifests, configs, metrics, grids and TensorBoard events).
- **Evaluation artefacts go to `$LOCALSCRATCH`, with one archive per run copied back.** The sample
  archives are where a file-count blow-up would come from, not the checkpoints.
- Re-read `quota` immediately before submitting. It is account state and other work moves it.

Space is not the constraint: this probe's two runs cost 3.6 GB, and 30 runs of 16 EMA checkpoints
(233 MiB each) plus a `full_final.pt` (932 MiB) is ≈ 140 GB against ≈ 0.9 TB free.

## 9. What this probe does not establish

- **Loss quality.** 600–800 iterations is 3–4 epochs; the losses in §2 show the loop learns and
  never goes non-finite, and nothing more. The learning-rate question of D4‴ is T2.3's, unfinished.
- **The N4 correction.** The cluster copies of `ixi` and `oasis1` are pre-N4. That cannot move
  it/s or peak GB — tensor shapes are unchanged — but the loss values in §2 will shift once the
  orchestrator re-syncs the corrected data before T3.3.
- **The plateau gate (D10).** Nothing here says whether 40,000 iterations is enough; that needs
  LSD at successive checkpoints of a real run.
- **Multi-task contention.** All numbers are from a single task with one GPU on `exa02`. A 30-task
  array sharing `exa[01-04]` may see lower per-task throughput from memory-bandwidth and
  filesystem contention.
- **loginexa as a pre-flight node.** It is unusable with this environment: its V100-DGXS is
  compute capability 7.0 and the env's `torch 2.14.0+cu130` ships
  `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — no `sm_70`, so every CUDA launch fails
  with "no kernel image is available for execution on the device".
