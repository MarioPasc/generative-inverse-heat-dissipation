# Why array 2408239 hit non-finite losses — diagnosis from the saved weights (T3.4, D19)

## 1. Question and fixtures

Array 2408239 (recipe v1: lr 2e-4 after a 1,000-step warm-up, batch 16, fp16 AMP, clip 1.0)
aborted 7 of the 10 runs that reached full lr, 367–888 steps after the warm-up ended
(`docs/RESULTS/submissions.md` §7). Every abort logged `"loss": "nan"`; the train loss one log
step earlier was 0.28–0.77. The question: was it an **fp16 forward overflow** (an activation of
the network beyond 65,504, harmless in fp32), a **true divergence** (weights that no longer fit
the data in any precision), or **other**?

The guard saved the rolling checkpoint *after* the non-finite step, which the `GradScaler` had
turned into a no-op, so the `model` weights of that checkpoint are exactly the weights that
produced the non-finite loss. Fixtures (Picasso `$HOME`, read only; also in the workstation
archive `…/training/array_2408239_failed/runs/`):

| run | abort (loop step) | checkpoint `step` | `checkpoint.pth` sha256 | schedule | data `images_sha256` |
|---|---|---|---|---|---|
| `ixi_A0_s1` | 1382 | 1383 | `eafa5ab8e03c9dd8…` | `log_W2` `ad9c8c1163f5d790…` | `b666e407e9afcc03…` |
| `lsun_church_A3_s1` | 1367 | 1368 | `106d19fd3f545606…` | `ixi_W8` `8bfe56bd759253d5…` | `299c076853b0dfd2…` |

Both ran commit `5bc28a6` on an A100-SXM4-40GB with torch 2.14.0+cu130; the schedule hashes equal
the repository's current `schedules/` (checked), so the recomputation uses the same arrays.

## 2. Method (`python -m ihdm.cli.diagnose_nan`, run on a loginexa V100)

The training loss of `scripts/losses.py: get_inverse_heat_loss_fn` is recomputed at the saved
weights with explicit levels, noise and dropout seeds — blur to levels $k$ and $k-1$
(`DCTBlur`, the run's schedule), add $\mathcal N(0, 0.01^2)$, predict the difference, per-sample
sum of squared errors over the $192^2$ pixels — in four conditions: fp16 autocast (as in
training) or fp32, and the network in train mode (dropout 0.1 on, as in training) or eval mode.

1. **Parameters**: global and per-tensor norms, max |w|, non-finite count, distance of the live
   weights to their EMA, Adam's largest second moment.
2. **Survey**: 64 random training batches of 16 with levels drawn as in training; every
   condition sees the same images, levels, noise and dropout seed; live and EMA weights.
3. **Level sweep**: one random batch of 16 at every level $k = 1…200$ (fp16, train mode); every
   failing level re-run in fp32.
4. **Replay**: the exact 16 images of the failing step (the train loader's index sequence is
   replayed from `config.seed`: a shuffling, `drop_last` loader over a generator seeded with the
   seed, one batch per step, re-created at each epoch end), at every level with 2 noise/dropout
   draws; failures re-run in fp32. The levels, noise and dropout masks of the real step came from
   the CUDA generator and cannot be replayed; the sweep covers all of them.
5. **Hooks**: for the first failing input found (replay first), forward hooks on every module
   record, in execution order, each output's max |value| and non-finite count, in fp16 and in
   fp32 on the same input; attention blocks also record the max |q·k| logit recomputed in fp32
   from their input (what the fp16 `einsum` of `QKVAttention` must hold below 65,504). The
   train-mode pair may not share dropout masks across dtypes, so the same input is also profiled
   in eval mode, where the pair is matched.

**Classification rules** (fixed before the run, `diagnose_nan.classify`): (1) `true_divergence`
if any weight is non-finite, any fp32 loss is non-finite, or the fp32 median survey loss exceeds
10× the last logged train losses; (2) `fp16_forward_overflow` if some fp16 loss is non-finite
while every fp32 loss on the same inputs is finite (the hook profile then names the module);
(3) `other` otherwise, including "not reproduced".

## 3. Results

Runs: `harness.sh diag` on loginexa GPUs 2 and 3, 2026-09-25 11:06–11:22, code `677cec4`, V100
with the cu126 overlay; reports `~/execs/ihdm/logs/loginexa/diag_{ixi_A0_s1,lsun_church_A3_s1}.json`
(704 s and 931 s). Verbatim verdict lines:

```
run ixi_A0_s1 checkpoint step 1383 abort {'step': 1382, 'kind': 'abort', 'reason': 'non-finite training loss', 'loss': 'nan'}
params: total norm 201.5, max |w| 1.085, non-finite 0, lr in optimizer [0.0002]
survey live fp16_train: 64/64 batches, 272/1024 samples non-finite; median batch loss 1.6172401905059814
survey live fp32_train: 0/64 batches, 0/1024 samples non-finite; median batch loss 1.5511428713798523
survey live fp16_eval: 63/64 batches, 271/1024 samples non-finite; median batch loss 1.6048554182052612
survey live fp32_eval: 0/64 batches, 0/1024 samples non-finite; median batch loss 1.5347827672958374
survey ema fp16_train: 0/64 batches non-finite
sweep: fp16 non-finite samples 772 at levels [26, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]; fp32 non-finite 0
replay: fp16 non-finite samples 820 at levels [12, 19, 23, 24, 26, 32, 37, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]; fp32 non-finite 0
hooks (replay level 12 draw 1): first non-finite fp16 module output_blocks.9.2.conv; modules >= 65504 in fp32: []
CLASSIFICATION fp16_forward_overflow: 2135 non-finite fp16 losses, 0 non-finite fp32 losses on the same inputs | first non-finite fp16 output: output_blocks.9.2.conv (Conv2d) | fp32 median batch loss 1.551 vs logged [0.44221675395965576, 0.5370804667472839, 0.6657921075820923, 0.4849953353404999, 0.42022505402565]

run lsun_church_A3_s1 checkpoint step 1368 abort {'step': 1367, 'kind': 'abort', 'reason': 'non-finite training loss', 'loss': 'nan'}
CLASSIFICATION fp16_forward_overflow: 3879 non-finite fp16 losses, 0 non-finite fp32 losses on the same inputs | first non-finite fp16 output: output_blocks.4.2.conv (Conv2d); its fp32 activation reaches the fp16 limit | fp32 median batch loss 2.751 vs logged [2.0256409645080566, 0.5454163551330566, 0.5914579629898071, 0.7005817890167236, 0.7674493193626404]
```

| measurement | `ixi_A0_s1` @1383 | `lsun_church_A3_s1` @1368 |
|---|---|---|
| weights: global L2 norm / max \|w\| / non-finite | 201.5 / 1.085 / 0 | 202.3 / 1.064 / 0 |
| largest live–EMA relative distance | `out.2.bias` 1.31; deep GroupNorm biases 0.57–0.69 | `out.2.bias` 10.6; `input_blocks.18–19` norm/out biases 3.7–5.3 |
| survey, live, fp16 train mode (as trained) | **64/64 batches, 272/1024 samples (26.6 %) non-finite** | **64/64 batches, 422/1024 samples (41.2 %)** |
| survey, live, fp16 eval mode | 63/64 batches, 271/1024 samples | 64/64 batches, 419/1024 samples |
| survey, live, fp32 (train / eval) | **0/1024 non-finite**; median batch loss 1.551 / 1.535 | **0/1024**; median 2.751 / 2.744 |
| survey, EMA, all four conditions | 0/1024 non-finite; median 0.389 (train) / 0.351 (eval) | 0/1024; median 0.483 / 0.459 |
| last five logged v1 train losses (steps 1150–1350) | 0.442, 0.537, 0.666, 0.485, 0.420 | 2.026, 0.545, 0.591, 0.701, 0.767 |
| level sweep (16 random images × levels 1–200, fp16) | 772/3200 non-finite, levels 26 and 75–152 ($\sigma_B$ 1.0 and 3.5–27 px); fp32 0 | 1289/3200, levels 1–191 ($\sigma_B$ 0.5–19 px); fp32 0 |
| replay of the failing step's 16 images (2 draws per level) | 820/6400, levels 12–149; all 16 images fail at levels 83–109; fp32 0 | 1749/6400, levels 2–200; all 16 fail at levels 133–158; fp32 0 |
| first non-finite fp16 module (train mode, replay input) | `output_blocks.9.2.conv` (the Upsample conv 48² → 96², 256 channels) | `output_blocks.4.2.conv` (the Upsample conv 24² → 48², 256 channels) |
| that module's fp32 max \|output\| (train / eval mode) | 63,676 / 65,177 = **97.2 % / 99.5 % of 65,504** | 87,149 / 86,983 = **133 % of 65,504** |
| next largest fp32 activations | `output_blocks.4.2.conv` 62,085–62,518; `output_blocks.5.0.skip_connection` 56,140 | `output_blocks.5.0.skip_connection` 70,188 (107 %); `output_blocks.6` 41,913 |
| max \|attention logit\| (fp32, any block) | 270 (`input_blocks.12`) | 309 (`input_blocks.16`) |

**Classification: fp16 forward overflow in the decoder's upsampling convolutions, reached during
a loss spike of the live weights at lr 2e-4** — not a true divergence. Evidence, rule by rule:
the weights are finite and small (max |w| ≈ 1.1); every fp32 loss on every input is finite (0 of
1024 survey samples, 0 in the sweeps and replays), while 27–41 % of the training samples give a
NaN in fp16; the first non-finite fp16 output is an Upsample convolution of the decoder whose
fp32 output sits at 97–133 % of the fp16 maximum, and the NaN follows from the `GroupNorm32`
of the next block (`x.float()` of an `inf`); the attention logits (≤ 310) are nowhere near the
fp16 limit, so the hypothesis of an attention `einsum` overflow is rejected.

The **weights, however, were not healthy**: their fp32 loss is 2.3–5× the train losses logged
30–230 steps earlier (ixi 1.55 against 0.42–0.67; lsun 2.75 against 0.55–0.77, after a 2.03 at
step 1,150) and 4–5.7× their own EMA's (ixi 1.55 against 0.39; lsun 2.75 against 0.48), and the
largest live–EMA distances sit in the output bias and the deepest normalisation biases. The v1
run was in one of the loss spikes that the logs show throughout the lr ramp (2.83 at step 800,
1.37 at 900 in `ixi_A0_s1`; 1.83 and 1.91 in `lsun_church_A0_s2`), and at 2e-4 a spike carried
the decoder's residual stream (a sum of block outputs and skip connections that no normalisation
bounds) past 65,504. Under rule (1) the fp32 loss would have to exceed 10× the logged one to
count as divergence; it is 3.5–5.7×, so the pre-registered label is `fp16_forward_overflow`, and
the mechanism is **an optimisation spike made fatal by fp16**.

Consequences for recipe v2:

- **The skip policy alone would not have saved array 1.** At these weights a batch of 16 has
  every sample finite with probability $0.734^{16} = 0.7\%$ (ixi) or $0.588^{16} = 0.02\%$ (lsun),
  and a skipped step leaves the weights unchanged, so the D19 guard would have logged ten
  consecutive `skip` lines and aborted. The skip policy protects against isolated overflows; what
  prevents the state is a smaller step, i.e. the lr.
- **lr 1e-4 removes the event in the window where v1 failed**: check S (§4) ran 3,000 steps at
  full lr on the two same-seed cells with zero non-finite losses; v1 failed them after 382 and
  367. The spikes remain visible at 1e-4 (train loss 0.89 at step 1,250 in ixi, 1.07 in lsun, with
  pre-clip norms 2,483 and 1,148), but finite.
- **bf16 is not needed** for recipe v2 on this evidence: S passed at 1e-4, and the overflow is
  a consequence of the spike rather than of values that approach 65,504 in normal operation (the
  EMA weights at the same steps, and the v2 weights, have their largest activations measured in
  §3a). The question of fp16 overflow "at 5e-5" did not arise.

### 3a. Activation headroom of healthy and v2 weights

(filled from the `headroom` harness item)

## 4. Stability check S (pre-registered in D19): **PASSED at lr 1e-4**

Recipe v2 (`configs/spectral/arms.py` at this ticket: lr 1e-4, batch 16, warm-up 1,000, clip 1.0,
fp16 AMP, EMA 0.999, K = 200), the two cells that failed first under v1, from scratch to step
4,000 on loginexa's V100s (GPUs 2 and 3), `--config.training.n_iters=4000
--config.training.resume_every=100` (both outside the recipe), everything else production.
Runs: `~/execs/ihdm/loginexa_runs/S_lr1e-4/{ixi_A0_s1,lsun_church_A3_s1}`; code `cdedbcf`
(identical trainer to this branch's head); both directories pass `check_run.py --lr 1e-4` with 0
problems (cadence, schema, manifest hashes, checkpoints, grids, `DONE`).

| run | skip / abort events | `done` | eval loss @1,000 | eval loss @4,000 | criterion | v1 same seed |
|---|---|---|---|---|---|---|
| `ixi_A0_s1` | **0 / 0** | `n_skipped: 0` | 0.3115 | **0.1939** | pass (−38 %) | abort at 1,382 |
| `lsun_church_A3_s1` | **0 / 0** | `n_skipped: 0` | 0.4815 | **0.3657** | pass (−24 %) | abort at 1,367 |

Both runs went 2,618 and 2,633 steps past the v1 abort steps, i.e. 3,000 steps at full lr against
v1's 367–888 steps to failure. The eval loss (25 `ref` batches, random levels, EMA weights) is
noisy between evaluation points — ixi 3.882, 0.335, 0.312, 0.306, 0.271, 0.214, 0.322, 0.247,
0.194 at 0, 500, …, 4,000; lsun 3.734, 0.627, 0.482, 0.669, 0.451, 0.446, 0.351, 0.454, 0.366 —
which is why the criterion compares the two pre-registered points only.

**Traces against the v1 same-seed curves** (v2: this check; v1: the fixtures' `metrics.jsonl`,
which end at the abort; train loss = mean over the 50-step window):

`ixi_A0_s1`

| step | lr | v2 train loss | v2 eval loss | v2 pre-clip `grad_norm` | v2 `amp_scale` | v1 train loss | v1 eval loss |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.8487 | 3.8820 | — | 32768 | 3.8502 | 3.8990 |
| 250 | 2.5e-5 | 0.5199 | — | 1334.7 | 256 | 0.5857 | — |
| 500 | 5.0e-5 | 0.4325 | 0.3349 | 465.4 | 128 | 0.5032 | 0.3285 |
| 750 | 7.5e-5 | 0.5758 | — | 727.6 | 128 | 0.6881 | — |
| 1000 | 1e-4 | 0.4498 | 0.3115 | 794.6 | 128 | 0.5502 | 0.5356 |
| 1250 | 1e-4 | 0.8935 | — | 2483.4 | 64 | 0.6658 | — |
| 1500 | 1e-4 | 0.6755 | 0.3062 | 662.0 | 64 | (aborted 1382) | |
| 2000 | 1e-4 | 0.4084 | 0.2710 | 263.1 | 64 | | |
| 2500 | 1e-4 | 0.2711 | 0.2143 | 69.9 | 512 | | |
| 3000 | 1e-4 | 0.3013 | 0.3220 | 269.5 | 64 | | |
| 3500 | 1e-4 | 0.2413 | 0.2473 | 90.8 | 1024 | | |
| 4000 | 1e-4 | 0.2270 | 0.1939 | 29.1 | 512 | | |

`lsun_church_A3_s1`

| step | lr | v2 train loss | v2 eval loss | v2 pre-clip `grad_norm` | v2 `amp_scale` | v1 train loss | v1 eval loss |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.7345 | 3.7337 | — | 32768 | 3.7372 | 3.8376 |
| 250 | 2.5e-5 | 0.8889 | — | 345.5 | 256 | 1.3247 | — |
| 500 | 5.0e-5 | 0.8281 | 0.6269 | 807.7 | 256 | 0.9475 | 0.6064 |
| 750 | 7.5e-5 | 0.6328 | — | 652.7 | 128 | 0.7294 | — |
| 1000 | 1e-4 | 0.5432 | 0.4815 | 662.5 | 64 | 0.7953 | 0.7320 |
| 1250 | 1e-4 | 1.0650 | — | 1147.9 | 128 | 0.5915 | — |
| 1500 | 1e-4 | 0.8316 | 0.6695 | 691.8 | 128 | (aborted 1367) | |
| 2000 | 1e-4 | 0.6156 | 0.4514 | 117.2 | 128 | | |
| 2500 | 1e-4 | 0.6199 | 0.4458 | 183.0 | 256 | | |
| 3000 | 1e-4 | 0.5111 | 0.3512 | 25.3 | 256 | | |
| 3500 | 1e-4 | 0.4910 | 0.4544 | 29.0 | 256 | | |
| 4000 | 1e-4 | 0.3726 | 0.3657 | 13.9 | 512 | | |

Per 1,000-step window (every 50-step train line):

| run | window | v2 train loss median / max | v2 pre-clip `grad_norm` median (min–max) | v2 `amp_scale` range | v1 train loss median / max |
|---|---|---|---|---|---|
| ixi | 0–999 | 0.515 / 3.849 | 728 (316–2,599) | 128–32,768 | 0.777 / 3.861 |
| ixi | 1,000–1,999 | 0.450 / 1.682 | 413 (130–2,483) | 64–256 | 0.537 / 0.666 (to 1,350) |
| ixi | 2,000–2,999 | 0.300 / 0.804 | 111 (30–354) | 64–512 | — |
| ixi | 3,000–4,000 | 0.266 / 0.316 | 60 (13.5–320) | 64–1,024 | — |
| lsun | 0–999 | 0.889 / 4.000 | 829 (346–3,939) | 64–32,768 | 1.029 / 3.989 |
| lsun | 1,000–1,999 | 0.763 / 1.140 | 389 (71–6,412) | 64–128 | 0.746 / 2.026 (to 1,350) |
| lsun | 2,000–2,999 | 0.468 / 0.659 | 48 (14.7–457) | 128–1,024 | — |
| lsun | 3,000–4,000 | 0.447 / 0.582 | 23 (13.9–75.6) | 256–512 | — |

Two observations the report keeps:

- **Every step is clipped.** The pre-clip global norm of the unscaled gradients is 300–6,400
  during the first 2,000 steps and never below 13.5 afterwards, against `grad_clip = 1.0`. This
  is the released loss's scale — a per-sample *sum* of squared errors over $192^2 = 36{,}864$
  pixels, not a mean — not an artefact of v2. Clipping every step to unit norm makes the update
  Adam on normalised gradients; Adam's per-parameter step stays of order lr, so the clip does
  not limit the step size. v1 logged the post-clip norm, identically 1.0, which is why array 1
  carried no gradient information.
- **The loss scale settles at 64–1,024.** The `GradScaler` starts at 65,536 and halves on every
  step whose fp16 gradients overflow; that the scale settles near 2^6–2^10 is the same fact seen
  from the fp16 side (large per-element gradients of a summed loss). It is the released code's
  loss scaling, unchanged by v2.

**Session boundaries and scaler back-offs.** Each run needed four 24-minute sessions; every
session resumed from the last 100-step rolling checkpoint (≤ 100 steps replayed), restarted the
data loader from the beginning of its seeded order and the `GradScaler` at 65,536:

| run | session | steps (first → last train line) | `amp_scale` first → last line | back-offs (lower bound¹) | of which re-calibration after the restart | null `grad_norm` lines² |
|---|---|---|---|---|---|---|
| ixi | 1 | 0 → 1,150 | 32,768 → 64 | 10 | 1 | 1 |
| ixi | 2 | 1,101 → 2,300 | 256 → 64 | 10 | 8 | 0 |
| ixi | 3 | 2,301 → 3,400 | 512 → 64 | 10 | 7 | 0 |
| ixi | 4 | 3,401 → 4,000 | 1,024 → 512 | 7 | 6 | 0 |
| lsun | 1 | 0 → 1,200 | 32,768 → 64 | 10 | 1 | 1 |
| lsun | 2 | 1,201 → 2,400 | 128 → 128 | 9 | 9 | 0 |
| lsun | 3 | 2,401 → 3,500 | 1,024 → 256 | 8 | 6 | 1 |
| lsun | 4 | 3,501 → 4,000 | 512 → 512 | 7 | 7 | 0 |

¹ counted from the 50-step `amp_scale` samples, starting from 65,536 at each session (a back-off
and a growth inside the same window would cancel; no growth was seen — growth needs 2,000
consecutive clean steps). ² a log step whose fp16 gradients overflowed (the scaler skipped it);
these are gradient overflows with a finite loss, not `skip` events. The A100 array runs each
cell as **one continuous job**, so S is conservative on data order (four restarts of the
seeded order instead of none) and neutral on the forward-overflow question, which depends on
the weights and the inputs, not on the scaler state or the order of the batches (ruling of
`main`); each session covered ≈ 6 epochs of the 3,200 training images.

V100 throughput and memory at batch 16 (H7): 0.868 it/s (ixi median of the 50-step windows,
0.857–0.896) and 0.873 it/s (lsun, 0.862–0.900), against 1.71 on the A100; peak allocated
28.516 GiB in both (A100: 28.544 GiB); `nvidia-smi` 30,189 MiB used with expandable segments.
Grids at 4,000 (`grids/iter_004000.png`, EMA): head silhouettes with gyral texture (ixi, A0 from
the flat level-200 prior) and coarse church masses inherited from the seed's $W/8$ blur (lsun, A3);
no NaN, black or saturated tiles.
