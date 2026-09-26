# Evaluation plan: 30 runs on Picasso (T5.1)

Scripts: `slurm/eval/` (see its `README.md`). Contract: D16 counts, `05-metrics.md` §7–§9.
Measured with the A100 timing job **2432221** (exa03, A100-SXM4-40GB, torch 2.14.0+cu130, code
`b74ff5f`) on the fixture `~/execs/ihdm/fixtures/array_2408239/ixi_A0_s2` (array-1 run, lr 2e-4,
EMA 5,000 and 7,500), and with the prepare job **2432211**. The contract default stays
`--amp off` until `main` decides (§4).

## 1. Was D16's 3.2 s/chain measured with autocast?

**No.** `docs/RESULTS/picasso_probe.md` §5 records the sampler timing with
"**no `--amp`** (matching T2.2's 3060 measurement so the two are comparable)": 3.236 / 3.206 /
3.210 s per 200-step chain at batch 32 / 64 / 128. The timing job reproduces it through
`evaluate_run`'s own cache path: **3.233 s/chain** at batch 32 without AMP. D16's cost model
(7.1 A100-h per run) therefore holds for `amp=off`. The array is not under-budgeted. AMP is an
option to save hours, not a correction.

## 2. Timing table (job 2432221)

Same 500 frozen training seeds (`eval_seeds_500.npy`, sha `e51c999e…`), `rng_seed` 2026, EMA
7,500, 200-step chains, drawn through `run_eval.draw_set`. The difference columns are paired
against the fp32 set on the same seeds and the same noise stream. The CI is the gate's
bootstrap (1,000 seed resamples, 95 %).

| mode | s/chain, batch 32 (500 chains) | s/chain, batch 64 (128 chains) | speed-up (b32) | peak GiB (b32) | LSD at 7,500 | LSD(off) − LSD(mode) [95 % CI] | \|ΔLSD\| / ixi floor 0.047 | pixels changed | mean / max \|Δ\| (u8 levels) |
|---|---|---|---|---|---|---|---|---|---|
| `off` | **3.233** | 3.209 | 1 | 8.21 (16.06 at b64) | 0.25763 | — | — | — | — |
| `fp16` | **2.425** | 2.395 | **1.333×** | 5.68 (10.99 at b64) | 0.25768 | −5.1e-5 [−1.0e-4, −6.5e-6] | **0.11 %** | 18.8 % | 0.26 / 27 |
| `bf16` | **2.424** | not measured¹ | **1.334×** | 5.68 | 0.25695 | +6.8e-4 [+3.6e-4, +9.9e-4] | **1.45 %** | 59.8 % | 1.66 / 147 |

Inception (clean mode, `fid_batch` 64): **7.2 s for 2,000 samples**, 4.1 s for the 800-image
reference, 5.5 s to build the extractor. The four reference matrices of the prepare job took
11.6 / 4.2 / 2.4 / 2.4 s. Model load: 1.6 s per checkpoint.

¹ The deadline guard skipped bf16 at batch 64 (projected 507 s, 390 s left in the 2 h wall).
It is not needed: batch 64 is ≤ 1.3 % faster than batch 32 in both measured modes, confirming
T3.2's plateau, and bf16 equals fp16 at batch 32.

**Plateau gate on 5,000 vs 7,500.** The job ran `evaluate_run --ckpts 5000 --gate 5000,7500
--n-lsd 500 --skip-inception` in `off`, exit 0, 1,628 s. It drew exactly one new 500-seed set
(at 5,000; 1,617 s). The 7,500 set came from the probe's cache.

- LSD 0.25175 → 0.25763. Difference LSD(5000) − LSD(7500) = **−0.00587**, 95 % CI
  [−0.00838, −0.00326], `extend = False`.
- The CI half-width at 500 seeds is **±0.0026** on an LSD of 0.26. That is the gate's measured
  resolution, 4.6× tighter than the ±0.012 that T4.3 §6 extrapolated from 32 seeds at an LSD
  of 0.77.
- Weight quality does not matter here (an lr-2e-4 array-1 run), so the sign of the difference
  means nothing. The job checks the command, its cost (1,000 chains, one of them cached) and
  its resolution.

The AMP speed-up is modest (1.33×) because the `off` path already runs its convolutions in TF32:
`torch.backends.cudnn.allow_tf32` is `True` by default (recorded in the probe's setup block).
fp16 and bf16 are equally fast.

## 3. Sizing

Chains per run (D16): 8 × 500 (LSD) + 2,000 (final) + 40 × 50 (held-out) = **8,000**, at
batch 32.

The fixed cost per run is an estimate, **0.25 h**. Its parts:

- 8 checkpoint loads (13 s);
- Inception on 2,000 samples plus the extractor build (13 s);
- the KID/FID/recall bootstrap with 200 resamples at n = 2,000, which is ≤ 213 s (measured on
  the workstation CPU with 2 threads under `nice`);
- memorisation, PCA, diversity, inherited band and LSD, a few minutes on the GPU;
- the tar of ≈ 0.55 GB and its copy to `$HOME`.

The first array task's log checks this estimate. The 30 % margin covers it several times over.

| mode | sampling h/run | total h/run | `--time` (× 1.3, rounded up to 15 min) | campaign A100-h (30 runs) | makespan at 8 concurrent tasks (4 rounds) |
|---|---|---|---|---|---|
| `off` | 7.185 | **7.43** | **09:45:00**² | **223** | 29.7 h |
| `fp16` | 5.389 | **5.64** | **07:30:00** | **169** | 22.6 h |
| `bf16` | 5.387 | **5.64** | **07:30:00** | **169** | 22.6 h |

The gate job costs 2 × 1,000 chains, which is 1.80 A100-h in `off` (`--time 03:00:00`). The
array reuses its four LSD sets through the gate tar, so the gate adds nothing net to the
campaign. The prepare job took 35 s and the timing job 2 h.

`--time` is set on the command line by `submit_eval.sh`. Since T3.5 its default is **computed**
from `N_ITERS` and `AMP` with the rule of this table (`common.sh` `ihdm_eval_time_limit`), and an
explicit `TIME_LIMIT` still wins. At 40k: `07:30:00` for `fp16`/`bf16`, `09:45:00` for `off`.
The 60k sizing is in §9.

² Until T3.5 this row said `10:00:00`, which was also the launcher's fixed default. The table's own
rule (7.43 h × 1.3 = 9.66 h, rounded up to 15 min) gives `09:45:00`; `main` ruled on 2026-09-26
that the one rule applies everywhere. The 15 min difference sits inside the × 1.3 margin.

No count cut is needed: the D16 plan fits the QOS `medium_uma` wall (3 days) with room to spare,
and 223 A100-h is D16's 214 A100-h plus the Inception and bookkeeping overhead.

## 4. AMP recommendation (for `main` to decide)

**Recommend `fp16`.**

- **Speed.** fp16 saves 25 % of the sampling time: 1.80 h per run, 54 A100-h and ≈ 7 h of
  makespan.
- **Effect on LSD.** It moves the LSD by 5e-5. That is 0.1 % of the `ixi` subject-grouped noise
  floor (0.047, `metrics_bracket.md` §1) and 2 % of the gate's measured resolution at 500 seeds
  (±0.0026, §2). The paired CI excludes zero because 500 seeds on the same noise stream
  resolve a shift that small. The shift is three orders of magnitude below any effect the
  experiment can report.
- **Effect on pixels.** 81 % of pixels are bitwise identical to the fp32 samples. The rest move
  by 0.26 grey levels on average and 27 at most.

**bf16 is dominated.** It is no faster than fp16. Its LSD shift is 13× larger (6.8e-4, 1.45 % of
the floor, 27 % of the gate's resolution). It changes 60 % of the pixels, by up to 147 levels, i.e. some chains diverge
visibly, as expected from a 7-bit mantissa.

**Risk of fp16 and its guard.** fp16 has a 65,504 overflow ceiling that fp32 and bf16 do not.
A late checkpoint with larger activations could overflow. Before this ticket, such an overflow
would have passed the [0, 1] clamp as NaN and become an arbitrary byte in the uint8 cache, with
no error. `draw_set` now refuses any non-finite draw (`test_a_non_finite_draw_is_refused_*`), so
an overflow fails the task loudly. The fallback is to resubmit that index with `AMP=off`; its
`samples/` tree is separate, so the two precisions cannot mix.

**What `fp16` costs scientifically.** All 30 runs must use the same mode; the arm contrasts are
then paired exactly as under fp32. The fp32 pilot numbers of T4.3 (`metrics_bracket.md` §6) are
not directly comparable with fp16 runs, but no conclusion rests on that comparison. If `main`
prefers bit-for-bit continuity with the fp32 pilot and the T3.2 probe, `off` costs 54 A100-h
more and nothing else.

## 5. Prepare job 2432211: the one-writer files

COMPLETED in 35 s on exa03. The job took these steps in order:

1. For each dataset, it checked `splits.json` against the workstation copy.
2. It drew the lists into a `$LOCALSCRATCH` shadow and matched them against
   `slurm/eval/expected_seed_lists.csv`, before any FSCRATCH write.
3. It wrote 6 files per dataset (24 in total), and nothing else on FSCRATCH. The FSCRATCH file
   count went from 248.6k to 248.7k.

| dataset | `eval_seeds_500` sha256 | `eval_seeds_final_2000` sha256 | matches local | `_features_inception_ref.npy` sha256 | ref features |
|---|---|---|---|---|---|
| `ixi` | `e51c999eb99c3308…` | `28ee6357f2bdd8c2…` | yes (also = T4.3's lists) | `21bcf0530b4ac497…` | 800 × 2048, 11.6 s |
| `lsun_bedroom` | `50eda5fa51e9efc3…` | `f9c946198c56d548…` | yes | `d59b41239ef9407b…` | 800 × 2048, 4.2 s |
| `lsun_church` | `50eda5fa51e9efc3…` | `f9c946198c56d548…` | yes | `77a145365382acb8…` | 800 × 2048, 2.4 s |
| `oasis1` | `5423693e1cd3b5c6…` | `e9e6b2cca37eb957…` | yes | `ae030222a5bacf4d…` | 800 × 2048, 2.4 s |

`lsun_bedroom` and `lsun_church` share their lists because both training splits are the same
3,200-entry index array, and the D17 draw depends only on that array. The timing job also drew
the `ixi` lists independently on its own node, and they matched (`e51c999e…`, `28ee6357…`).

## 6. Submission order and exact commands

Run everything from main's cluster clone after the merge. The launcher's `IHDM_REPO_DIR`
defaults to `~/fscratch/repos/generative-inverse-heat-dissipation`.

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation
# 0. the one-writer files already exist (job 2432211); a rerun is idempotent and writes nothing.
#    Only if the datasets are ever rebuilt:  bash slurm/eval/submit_eval.sh prepare
# 1. plateau gate on cells 0 and 3, as soon as the v2 training array <TRAIN> is queued:
TRAIN_ARRAY=<TRAIN> bash slurm/eval/submit_eval.sh gate --test-only
TRAIN_ARRAY=<TRAIN> bash slurm/eval/submit_eval.sh gate
#    -> ~/execs/ihdm/eval/gate/<run_id><amp>_gate_<early>_<late>.json (6-digit steps; the files
#       of job 2432703 keep the pre-T3.5 name <run_id>_amp-fp16_gate.json = 35k/40k); main decides.
# 2. the evaluation array, after main's decision (AMP per §4; --time computed from N_ITERS, AMP):
TRAIN_ARRAY=<TRAIN> bash slurm/eval/submit_eval.sh array --test-only
AMP=fp16 TRAIN_ARRAY=<TRAIN> bash slurm/eval/submit_eval.sh array                 # 07:30:00 at 40k
# 3. resubmit failed indices (each resumes from its partial tar):
ARRAY_SPEC='6,17' bash slurm/eval/submit_eval.sh array
```

If the runs are extended past 40,000 (D10, D22), submit the array with `N_ITERS=<n>` (§9). The
worker then requires `ema_iter_<n>.pt` to be the largest EMA checkpoint. Use `aftercorr` on the
extension array, not on the original one.

## 7. File budget

| where | files | space |
|---|---|---|
| FSCRATCH | 24, written once (done); nothing per run | 52 MB |
| `$HOME` `~/execs/ihdm/eval/` | 2 per run (tar + `summary.json`) = 60; gate 4; prepare 1; timing 5 | ≈ 0.55 GB per run tar ≈ 17 GB |
| `$HOME` `~/execs/ihdm/logs/` | 2 per task = 60; gate 2; prepare 2; timing 2 | small |
| `$HOME` T5.1's rsynced tree `~/execs/ihdm/wt/T5.1/` | ≈ 300 (delete after the merge) | small |

## 8. What T5.2 collects

Per run: `~/execs/ihdm/eval/<run_id>[_amp-<mode>]_summary.json` (plain) and the tar
`<run_id>[_amp-<mode>].tar`. The tar holds the shadow run:

- `metrics[_amp-<mode>]/` with `ckpt_<step>.json` × 8 (× 12 at 60k), `final.json`, `summary.json`, the `.npy`
  sidecars (memorisation per sample, PCA components) and, for cells 0 and 3, `gate.json`;
- `samples[_amp-<mode>]/<step>/{lsd,final,heldout}/` with `samples.npy`, `seeds.npy`,
  `seed_idx.npy` and `request.json`;
- links to the run's `config.json`, `manifest.json`, `checkpoints/`, `grids/` and
  `metrics.jsonl`.

Gate: `~/execs/ihdm/eval/gate/<run_id>[_amp-<mode>]_gate_<early:06d>_<late:06d>.json`; the 35k/40k
gate of job 2432703 is `<run_id>_amp-fp16_gate.json` (pre-T3.5 name).

T_tau is not computed in the array, because `--a0-final-lsd` is unknown while the other arms
run. T5.2/T6.1 compute it from `lsd_by_step` and the A0 run's `final.lsd` in the summaries.
T5.2 should check that every summary has `checkpoint_steps` equal to every 5,000 up to the run
length (the eight D16 steps at 40k, twelve at 60k),
`sampling.amp` equal to the decided mode, and seed-list digests equal to §5.

## 9. Sizing at 60,000 iterations (D22, T3.5)

The runs are extended from 40k to 60k by resume (D22). The evaluated checkpoints become every
5,000 up to the run length (`run_eval.evaluated_steps`), so a 60k run is evaluated at 12
checkpoints. The per-checkpoint counts of D16 do not change: 500 LSD samples per checkpoint, and
the final 2,000 set and the 40 × 50 held-out set at the largest checkpoint only.

Chains per run: 12 × 500 + 2,000 + 40 × 50 = **10,000** (8,000 at 40k). Rule, as in §3: measured
s/chain × chains + 0.25 h fixed, × 1.3, rounded up to 15 min (`common.sh` `ihdm_eval_time_limit`,
pinned by `tests/metrics/test_run_eval.py::test_eval_time_limit_follows_the_counts`).

| run length | mode | chains | sampling h/run | total h/run | `--time` | campaign A100-h (30 runs) | makespan at 8 concurrent (4 rounds) |
|---|---|---|---|---|---|---|---|
| 40k | `off` | 8,000 | 7.184 | 7.43 | `09:45:00` (was `10:00:00`) | 223 | 29.7 h |
| 40k | `fp16` | 8,000 | 5.389 | 5.64 | `07:30:00` | 169 | 22.6 h |
| **60k** | `off` | 10,000 | 8.981 | **9.23** | **`12:00:00`** | **277** | 36.9 h |
| **60k** | **`fp16`** (D20) | 10,000 | 6.736 | **6.99** | **`09:15:00`** | **210** | 27.9 h |
| 60k | `bf16` | 10,000 | 6.733 | 6.98 | `09:15:00` | 209 | 27.9 h |

`9.23 h × 1.3 = 11.9997 h` for 60k `off` rounds up to exactly `12:00:00`, with no slack beyond the
margin itself. The array reuses the LSD sets of the gate tars it unpacks (35k/40k and 55k/60k on
cells 0 and 3), 4 × 500 chains on 2 of 30 runs. The table ignores that saving.

**Training extension.** 20,000 more steps per run at 1.69–1.71 it/s take ≈ 3.3 h. That is
**≈ 98 A100-h** for the 30 runs, submitted with `TIME_LIMIT=05:00:00` (× 1.5).

**Gate 55k vs 60k (informational, D22).** It draws 2 × 500 chains per cell, 0.67 h in `fp16`
(`--time 03:00:00`, unchanged). The output is named by its step pair, so it never overwrites the
35k/40k gate.

**Campaign after D22 (fp16).** Extension ≈ 98 A100-h, plus the evaluation ≈ 210 A100-h, plus the
55k/60k gate, ≈ 1.4 A100-h (the 35k/40k gate, job 2432703, took 1 h 21 m).

**Commands (`main`, after the 40k archive).**

```bash
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation
export PYTHONPYCACHEPREFIX=/tmp/ihdm_pyc_$$
PY=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/ihdm/bin/python
RUNS=/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm
# extend in place
N_ITERS=60000 TIME_LIMIT=05:00:00 bash slurm/array/submit_array.sh --test-only
N_ITERS=60000 TIME_LIMIT=05:00:00 bash slurm/array/submit_array.sh              # -> <EXT>
# health at 60k, once every task has COMPLETED
$PY -m ihdm.cli.check_array --run-root $RUNS --cells slurm/array/cells.csv --n-iters 60000
# informational gate 55k vs 60k (writes ..._gate_055000_060000.{json,tar})
N_ITERS=60000 AMP=fp16 TRAIN_ARRAY=<EXT> bash slurm/eval/submit_eval.sh gate --test-only
N_ITERS=60000 AMP=fp16 TRAIN_ARRAY=<EXT> bash slurm/eval/submit_eval.sh gate
# evaluation array at 60k (--time 09:15:00 computed)
N_ITERS=60000 AMP=fp16 TRAIN_ARRAY=<EXT> bash slurm/eval/submit_eval.sh array --test-only
N_ITERS=60000 AMP=fp16 TRAIN_ARRAY=<EXT> bash slurm/eval/submit_eval.sh array
```
