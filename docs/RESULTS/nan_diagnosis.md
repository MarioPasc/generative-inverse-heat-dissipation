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

(filled from the loginexa runs below)
