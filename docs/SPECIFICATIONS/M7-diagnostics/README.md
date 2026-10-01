# M7 — Post-hoc diagnostics (wave W14, 2026-10-01)

Not part of the pre-registered experiment. Two questions raised by the final results
(`docs/RESULTS/results_discussion.md` §3, §8) and approved by Mario on 2026-10-01:

1. **Why do the photograph models fail at 192²?** (Churches A0: Inception precision 0.003 against
   0.63 on IXI A0.) The A0 arm keeps the paper's schedule rules but differs from the released LSUN
   config in resolution and framing, data size, colour, U-Net width, K, lr and length. The
   diagnostic changes **one factor at a time** on Churches A0, seed 1, with the 60k recipe
   unchanged:
   - `lsun_church_r128`: the same 4,000 source rows, whole-scene resize to 128² (the paper's
     `Resize(128)` + `CenterCrop(128)` framing), grayscale; σ_B,max = W/2 = 64, log spacing.
   - `lsun_church_n32k`: 192² native centre crops as `lsun_church`, the same 800 `ref` and 40
     `seed` images, and 32,000 `train` images (the original 3,200 plus 28,800 new rows).
2. **Is the under-dispersion a sampler setting?** δ sweep on four existing 60k checkpoints
   (IXI and Churches, A0 and A3, seed 1) at δ/σ ∈ {1.25, 2, 3}, no training.

| ticket | what | agent | depends on |
|---|---|---|---|
| T7.1 | diagnostic datasets, config support, training submission of 2 runs | opus55-xhigh | — |
| T7.2 | `evaluate_run --delta`, 128²-capable evaluation, δ-sweep job and results | opus55-high | — |
| T7.3 | evaluation of the 2 diagnostic runs and the baseline (late window), write-up | later wave | T7.1 trained, T7.2 merged |

**Pre-registered reading of T7.1/T7.3 (written 2026-10-01, before any diagnostic run exists).**
Each run is read at checkpoints 45k, 50k, 55k and 60k, with 500 training-seeded samples per
checkpoint against its own 800-image `ref` split. The quantities are the late-window means of
Inception precision, recall and KID, the variance ratio, and the sample grid. The baseline is
`lsun_church_A0_s1`, evaluated the same way. A factor **lifts the failure** if its late-window
precision is ≥ 0.10 **and** ≥ 10× the baseline's, **and** its KID is ≤ 0.5× the baseline's. The
possible outcomes and their readings:

| r128 lifts? | n32k lifts? | reading |
|---|---|---|
| yes | no | resolution and framing |
| no | yes | data size |
| yes | yes | either change suffices |
| no | no | the budget or recipe (lr, length, model width), which the diagnostic does not test |

One seed per factor, so the reading is descriptive.

**Pre-registered reading of T7.2.** The under-dispersion is "partly a sampler setting" if, on IXI
A0, some δ > 1.25σ raises the variance ratio of the 500-seed LSD set by ≥ 0.10 without raising KID.
If the variance ratio rises while KID and precision worsen, the added variance is noise.
Descriptive, one seed.
