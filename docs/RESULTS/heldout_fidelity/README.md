# Fidelity from held-out seeds (T7.5, M7, post hoc, CPU only)

**Question** (`docs/SPECIFICATIONS/M7-diagnostics/T7.5-heldout-seed-fidelity.md`): every fidelity
number of the experiment (tables 1a, 2 and 3) is computed on *training-seeded* samples, whose prior
state is a blurred training image. At W/8 (arm A3) that prior hands over the training image's
coarse layout, and 56–76% of the samples have their own seed as nearest training image. **Does
A3's fidelity gain over A0 (IXI: precision +0.088, KID −63%) depend on the seed being a training
image, or does it hold when the prior state comes from a subject the model never saw?**

**What the check separates.** Held-out seeds close the *memorisation* route: the prior cannot hand
over a training image. They do not close *inheritance*: at W/8 the prior still hands over the
coarse layout of a real brain, unseen but real. The check therefore separates "the gain is tied
to the training set" from "the gain generalises to unseen prior states". It cannot show "better
generation from scratch", which W/8 by construction never does.

## Protocol (pre-registered on 2026-10-03, before any number of this ticket existed)

No new sampling: every sample is read from the production evaluation tars (job 2488269, EMA 60k,
fp16, sample batch 32, δ = 0.0125). For each of the 24 runs (A0, A1, A3 on IXI and Churches; A0,
A3 on OASIS-1 and Bedrooms) two sample sets are compared:

| set | prior states | samples | stored as |
|---|---|---|---|
| **H**, held-out-seeded | the 40 `seed` images (MRI: slice 5 of each of the 40 seed subjects), never in `train` | 40 × 50, rng 2026 | `samples_amp-fp16/060000/heldout/` |
| **F**, training-seeded | 2,000 `train` images drawn with replacement (`eval_seeds_final_2000.npy`) | 2,000 × 1, rng 0 | `samples_amp-fp16/060000/final/` |

**References.**

- **R⁻** (primary): the `ref` split minus every image of the 40 seed subjects. MRI: 80 − 40
  subjects × 10 slices = **400** images. Photographs: 800 − 40 = **760**. One R⁻ per dataset,
  shared by every arm and both sets.
- **R5** (sensitivity, MRI only): the slice-5 image of each of the 320 `train` subjects (**320**
  images), the population at the held-out seeds' slice level. R⁻ is conservative for A3, because
  A0's samples drift across slice levels and an all-slice reference does not penalise that. R5 is
  liberal for A3.

Checked by the code on every run: the seed split lies inside `ref` on all four datasets (by
subject on IXI and OASIS-1, by image on Churches and Bedrooms). No R⁻ or R5 row belongs to a seed
subject. Each run's held-out `seed_idx` equals the expected 40 images, and its stored `seeds.npy`
is byte-identical to `images[seed_idx]`. Every final-set seed lies in `train`.

**Metrics.** CPU Inception features (`ihdm.metrics.inception.inception_features`, `mode="clean"`,
`device="cpu"`) of every sample set and reference, all recomputed on one machine with one
extractor.

| role | metric |
|---|---|
| **primary** | precision (k = 5) against R⁻, for H and for F |
| **secondary** | density against R⁻; KID against R⁻ (`kid_from_features`, 100 subsets, max size 1,000, rng 0); on MRI, the precision and KID of H against R5 |
| **descriptive** | recall and coverage. They are capped, because H has only 40 distinct prior states. |

Per-sample precision indicators and density counts use `prdc`'s own distance helpers. Their means
reproduce `recall_coverage`'s aggregates to 1e-12; the code asserts this on every set. The
**within-run 95% CI** is a percentile bootstrap, 1,000 draws, rng 0. H resamples the 40 seeds as
whole clusters of 50 samples; F resamples samples.

**Contrasts.** $\Delta_H(s) = m_H(\text{A3}, s) - m_H(\text{A0}, s)$ and
$\Delta_F(s) = m_F(\text{A3}, s) - m_F(\text{A0}, s)$ per run seed $s$, and the same with A1 in
place of A3. The statistics are `ihdm.stats.paired_delta` (10,000 seed resamples, rng 0) and the
exact `ihdm.stats.permutation_test`, as in tables 2a/2b: p_min is 0.1 at 3 seeds and 1/3 at 2.
The generalisation gap $m_F - m_H$ is reported for every run.

**Reading rule (IXI; primary endpoint: precision against R⁻).** Let $G_F$ and $G_H$ be the means
of $\Delta_F(s)$ and $\Delta_H(s)$ over the 3 run seeds. Apply the first rule that matches:

1. $G_F \le 0$, or not all three $\Delta_F(s) > 0$: *"the comparator shows no precision gain on
   R⁻; not evaluable on precision"*. Then apply the rule to KID, with the gain defined as A0 − A3,
   and label the result secondary.
2. All three $\Delta_H(s) > 0$ and $G_H \ge 0.5\,G_F$: **"the fidelity gain generalises to unseen
   seeds"**.
3. All three $\Delta_H(s) > 0$ and $G_H < 0.5\,G_F$: **"partly tied to training seeds"**.
4. Otherwise: **"not shown on unseen seeds"**.

OASIS-1 takes the same rule with its 2 run seeds and is read descriptively (transfer). Churches and
Bedrooms are descriptive only, because the photograph models fail; the rule is applied to them
mechanically. A0 sanity expectation: at W/2 the prior carries 0.3% of the variance, so A0's
$m_F - m_H$ should be near 0. It is reported, not gated on.

**Amendments made by main on 2026-10-03, before any contrast of this ticket was computed.** They
followed a question raised by the agent.

- *KID fallback.* When rule 1 fires on precision, the **whole** rule, rule 1 included, is applied
  to KID (gain A0 − A3). If the KID comparator also fails ($G_F^{\text{KID}} \le 0$, or not all
  $\Delta_F^{\text{KID}}$ gains $> 0$), the outcome is **"not evaluable on precision or KID"**.
  Without this amendment, a failing KID comparator would satisfy $G_H \ge 0.5\,G_F$ trivially.
  `kid_comparator_gain_positive` is reported in either case.
- *R5 is H only.* The R5 sensitivity reports, per run seed, $\Delta_H(\text{R5})$, its mean and
  whether all seeds agree in sign. The ratio rule is **not** applied against a $G_F$ on R5,
  because F against R5 is contaminated: the training seeds include the R5 images. F against R5
  is printed descriptively only.

Neither amendment changes the outcome below: precision is evaluable on every dataset, so the KID
fallback never runs.

## Reproduction anchors (ticket step 7; run before the full run)

**(a) Local IXI `ref` features against the production cache** (`ixi/_features_inception_ref.npy`,
computed on an A100 GPU), 800 rows in the same (sorted) order. The ticket sets no threshold for
this anchor; it is reported only.

| max abs difference | min row-wise cosine |
|---|---|
| 8.1e-3 | 0.999998 |

**(b) F against the full 800-image `ref`, local features against the stored `final.json`.** The
gate is |Δ| ≤ 0.01 on precision, recall, density and coverage, and KID inside its stored 95%
interval.

| run | metric | stored | local | abs. difference | ok |
|---|---|---|---|---|---|
| `ixi_A0_s1` | precision | 0.5740 | 0.5730 | 0.0010 | yes |
| `ixi_A0_s1` | recall | 0.2575 | 0.2600 | 0.0025 | yes |
| `ixi_A0_s1` | density | 0.3122 | 0.3107 | 0.0015 | yes |
| `ixi_A0_s1` | coverage | 0.4263 | 0.4275 | 0.0013 | yes |
| `ixi_A0_s1` | KID | 0.04638 [0.04397, 0.04878] | 0.04634 | 3.6e-5 | yes |
| `lsun_church_A3_s1` | precision | 0.0590 | 0.0585 | 0.0005 | yes |
| `lsun_church_A3_s1` | recall | 0.0288 | 0.0300 | 0.0013 | yes |
| `lsun_church_A3_s1` | density | 0.0158 | 0.0157 | 0.0001 | yes |
| `lsun_church_A3_s1` | coverage | 0.0500 | 0.0500 | 0.0000 | yes |
| `lsun_church_A3_s1` | KID | 0.22027 [0.21573, 0.22601] | 0.22022 | 5.3e-5 | yes |

**Both anchors pass; worst deviation 0.0025 (`ixi_A0_s1` recall).** The JSON holds the same
comparison for every one of the 24 runs (`runs.<id>.metrics.F_fullref` against
`runs.<id>.provenance.stored_inception`). Over all 24 the worst deviation is 0.0050
(`ixi_A1_s2` recall), and every local KID lies inside its stored interval.

## Results per run (against R⁻)

| run | precision H [95% CI] | precision F [95% CI] | density H [95% CI] | density F [95% CI] | KID H | KID F | recall H / F | coverage H / F |
|---|---|---|---|---|---|---|---|---|
| `ixi_A0_s1` | 0.590 [0.545, 0.632] | 0.602 [0.580, 0.622] | 0.277 [0.248, 0.307] | 0.302 [0.284, 0.317] | 0.0583 | 0.0474 | 0.170 / 0.302 | 0.395 / 0.585 |
| `ixi_A0_s2` | 0.711 [0.668, 0.750] | 0.683 [0.665, 0.704] | 0.396 [0.361, 0.433] | 0.384 [0.363, 0.404] | 0.0615 | 0.0518 | 0.285 / 0.383 | 0.482 / 0.625 |
| `ixi_A0_s3` | 0.683 [0.648, 0.715] | 0.680 [0.659, 0.701] | 0.369 [0.335, 0.402] | 0.389 [0.367, 0.409] | 0.0540 | 0.0424 | 0.258 / 0.345 | 0.512 / 0.677 |
| `ixi_A3_s1` | 0.750 [0.703, 0.792] | 0.755 [0.736, 0.773] | 0.414 [0.366, 0.468] | 0.478 [0.455, 0.501] | 0.0551 | 0.0183 | 0.087 / 0.510 | 0.512 / 0.885 |
| `ixi_A3_s2` | 0.775 [0.724, 0.823] | 0.775 [0.755, 0.794] | 0.466 [0.409, 0.525] | 0.518 [0.494, 0.540] | 0.0526 | 0.0162 | 0.080 / 0.403 | 0.552 / 0.902 |
| `ixi_A3_s3` | 0.750 [0.700, 0.796] | 0.741 [0.722, 0.759] | 0.419 [0.371, 0.472] | 0.470 [0.447, 0.492] | 0.0538 | 0.0175 | 0.080 / 0.475 | 0.512 / 0.858 |
| `ixi_A1_s1` | 0.676 [0.608, 0.735] | 0.649 [0.629, 0.671] | 0.354 [0.302, 0.408] | 0.374 [0.355, 0.397] | 0.0514 | 0.0168 | 0.102 / 0.502 | 0.502 / 0.840 |
| `ixi_A1_s2` | 0.754 [0.705, 0.794] | 0.750 [0.730, 0.770] | 0.422 [0.368, 0.481] | 0.485 [0.462, 0.508] | 0.0544 | 0.0171 | 0.085 / 0.372 | 0.532 / 0.877 |
| `lsun_church_A0_s1` | 0.007 [0.004, 0.011] | 0.005 [0.003, 0.009] | 0.002 [0.001, 0.003] | 0.001 [0.001, 0.002] | 0.2430 | 0.2458 | 0.009 / 0.009 | 0.013 / 0.008 |
| `lsun_church_A0_s2` | 0.000 [0.000, 0.000] | 0.002 [0.000, 0.004] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.3775 | 0.3741 | 0.000 / 0.000 | 0.000 / 0.003 |
| `lsun_church_A0_s3` | 0.004 [0.001, 0.007] | 0.004 [0.002, 0.007] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.3100 | 0.3134 | 0.009 / 0.000 | 0.011 / 0.012 |
| `lsun_church_A3_s1` | 0.038 [0.025, 0.051] | 0.059 [0.049, 0.070] | 0.015 [0.009, 0.022] | 0.016 [0.013, 0.020] | 0.2271 | 0.2201 | 0.039 / 0.029 | 0.063 / 0.053 |
| `lsun_church_A3_s2` | 0.083 [0.059, 0.106] | 0.083 [0.070, 0.095] | 0.029 [0.018, 0.040] | 0.034 [0.027, 0.041] | 0.1325 | 0.1340 | 0.111 / 0.121 | 0.112 / 0.133 |
| `lsun_church_A3_s3` | 0.123 [0.094, 0.154] | 0.130 [0.116, 0.144] | 0.049 [0.034, 0.064] | 0.052 [0.044, 0.062] | 0.1358 | 0.1326 | 0.117 / 0.130 | 0.138 / 0.163 |
| `lsun_church_A1_s1` | 0.115 [0.089, 0.142] | 0.115 [0.099, 0.129] | 0.044 [0.030, 0.061] | 0.043 [0.036, 0.051] | 0.1749 | 0.1748 | 0.084 / 0.086 | 0.111 / 0.128 |
| `lsun_church_A1_s2` | 0.052 [0.038, 0.067] | 0.057 [0.047, 0.068] | 0.015 [0.011, 0.021] | 0.018 [0.014, 0.022] | 0.1812 | 0.1836 | 0.051 / 0.018 | 0.043 / 0.055 |
| `oasis1_A0_s1` | 0.660 [0.626, 0.697] | 0.662 [0.642, 0.683] | 0.483 [0.432, 0.533] | 0.447 [0.423, 0.473] | 0.0305 | 0.0251 | 0.347 / 0.435 | 0.665 / 0.805 |
| `oasis1_A0_s2` | 0.755 [0.722, 0.781] | 0.748 [0.728, 0.767] | 0.616 [0.559, 0.671] | 0.581 [0.554, 0.611] | 0.0251 | 0.0206 | 0.375 / 0.477 | 0.738 / 0.855 |
| `oasis1_A3_s1` | 0.828 [0.782, 0.870] | 0.842 [0.826, 0.857] | 0.702 [0.618, 0.786] | 0.769 [0.739, 0.803] | 0.0394 | 0.0114 | 0.140 / 0.458 | 0.693 / 0.968 |
| `oasis1_A3_s2` | 0.856 [0.821, 0.886] | 0.827 [0.809, 0.843] | 0.693 [0.618, 0.764] | 0.742 [0.709, 0.774] | 0.0394 | 0.0121 | 0.142 / 0.593 | 0.640 / 0.978 |
| `lsun_bedroom_A0_s1` | 0.013 [0.009, 0.018] | 0.013 [0.009, 0.018] | 0.003 [0.002, 0.004] | 0.003 [0.002, 0.004] | 0.2066 | 0.2046 | 0.000 / 0.032 | 0.012 / 0.013 |
| `lsun_bedroom_A0_s2` | 0.012 [0.005, 0.020] | 0.010 [0.006, 0.015] | 0.003 [0.001, 0.004] | 0.002 [0.001, 0.003] | 0.2761 | 0.2706 | 0.001 / 0.000 | 0.008 / 0.005 |
| `lsun_bedroom_A3_s1` | 0.140 [0.110, 0.170] | 0.135 [0.120, 0.150] | 0.041 [0.031, 0.052] | 0.044 [0.039, 0.051] | 0.1341 | 0.1306 | 0.024 / 0.067 | 0.083 / 0.095 |
| `lsun_bedroom_A3_s2` | 0.109 [0.081, 0.139] | 0.112 [0.098, 0.126] | 0.032 [0.022, 0.044] | 0.035 [0.030, 0.041] | 0.1465 | 0.1404 | 0.038 / 0.055 | 0.071 / 0.093 |

H's intervals are wider than F's because H holds 40 independent prior states, not 2,000.

## Generalisation gap $m_F - m_H$ (against R⁻)

| run | precision | density | KID | | run | precision | density | KID |
|---|---|---|---|---|---|---|---|---|
| `ixi_A0_s1` | +0.012 | +0.025 | −0.0109 | | `lsun_church_A0_s1` | −0.002 | −0.001 | +0.0029 |
| `ixi_A0_s2` | −0.027 | −0.012 | −0.0097 | | `lsun_church_A0_s2` | +0.002 | +0.000 | −0.0034 |
| `ixi_A0_s3` | −0.003 | +0.020 | −0.0116 | | `lsun_church_A0_s3` | +0.001 | +0.000 | +0.0034 |
| `ixi_A3_s1` | +0.005 | +0.064 | −0.0367 | | `lsun_church_A3_s1` | +0.021 | +0.001 | −0.0070 |
| `ixi_A3_s2` | −0.000 | +0.052 | −0.0364 | | `lsun_church_A3_s2` | +0.000 | +0.005 | +0.0015 |
| `ixi_A3_s3` | −0.009 | +0.051 | −0.0363 | | `lsun_church_A3_s3` | +0.007 | +0.004 | −0.0033 |
| `ixi_A1_s1` | −0.027 | +0.020 | −0.0346 | | `lsun_church_A1_s1` | −0.001 | −0.001 | −0.0002 |
| `ixi_A1_s2` | −0.004 | +0.063 | −0.0373 | | `lsun_church_A1_s2` | +0.005 | +0.002 | +0.0024 |
| `oasis1_A0_s1` | +0.002 | −0.035 | −0.0054 | | `lsun_bedroom_A0_s1` | +0.001 | +0.000 | −0.0019 |
| `oasis1_A0_s2` | −0.007 | −0.034 | −0.0045 | | `lsun_bedroom_A0_s2` | −0.002 | −0.001 | −0.0054 |
| `oasis1_A3_s1` | +0.014 | +0.067 | −0.0280 | | `lsun_bedroom_A3_s1` | −0.005 | +0.004 | −0.0034 |
| `oasis1_A3_s2` | −0.029 | +0.049 | −0.0273 | | `lsun_bedroom_A3_s2` | +0.003 | +0.004 | −0.0061 |

A negative KID gap means H is farther from R⁻ than F.

**A0 sanity value.** IXI A0 precision gap: +0.012, −0.027 and −0.003, mean −0.006, so near 0 as
expected. Its KID gap is −0.011 on every seed, which means H is somewhat farther from R⁻ than F
even at W/2. OASIS-1 A0 precision gap: +0.002 and −0.007.

## Contrasts arm − A0 (against R⁻)

| dataset | arm | set | metric | per-seed Δ | mean [95% CI] | p (p_min) |
|---|---|---|---|---|---|---|
| IXI | A3 | H | precision | +0.1605, +0.0650, +0.0670 | +0.0975 [+0.0650, +0.1605] | 0.100 (0.100) |
| IXI | A3 | F | precision | +0.1535, +0.0915, +0.0605 | +0.1018 [+0.0605, +0.1535] | 0.100 (0.100) |
| IXI | A3 | H | density | +0.1372, +0.0698, +0.0506 | +0.0859 [+0.0506, +0.1372] | 0.100 (0.100) |
| IXI | A3 | F | density | +0.1764, +0.1343, +0.0814 | +0.1307 [+0.0814, +0.1764] | 0.100 (0.100) |
| IXI | A3 | H | KID | −0.0033, −0.0089, −0.0002 | −0.0041 [−0.0089, −0.0002] | 0.200 (0.100) |
| IXI | A3 | F | KID | −0.0290, −0.0356, −0.0249 | −0.0298 [−0.0356, −0.0249] | 0.100 (0.100) |
| IXI | A1 | H | precision | +0.0865, +0.0440 | +0.0652 [+0.0440, +0.0865] | 0.667 (0.333) |
| IXI | A1 | F | precision | +0.0480, +0.0665 | +0.0572 [+0.0480, +0.0665] | 0.667 (0.333) |
| IXI | A1 | H | density | +0.0765, +0.0254 | +0.0509 [+0.0254, +0.0765] | 0.667 (0.333) |
| IXI | A1 | F | density | +0.0720, +0.1007 | +0.0864 [+0.0720, +0.1007] | 0.667 (0.333) |
| IXI | A1 | H | KID | −0.0070, −0.0071 | −0.0071 [−0.0071, −0.0070] | 0.333 (0.333) |
| IXI | A1 | F | KID | −0.0306, −0.0346 | −0.0326 [−0.0346, −0.0306] | 0.333 (0.333) |
| OASIS-1 | A3 | H | precision | +0.1675, +0.1005 | +0.1340 [+0.1005, +0.1675] | 0.333 (0.333) |
| OASIS-1 | A3 | F | precision | +0.1795, +0.0790 | +0.1292 [+0.0790, +0.1795] | 0.333 (0.333) |
| OASIS-1 | A3 | H | density | +0.2196, +0.0773 | +0.1485 [+0.0773, +0.2196] | 0.333 (0.333) |
| OASIS-1 | A3 | F | density | +0.3224, +0.1610 | +0.2417 [+0.1610, +0.3224] | 0.333 (0.333) |
| OASIS-1 | A3 | H | KID | +0.0089, +0.0143 | +0.0116 [+0.0089, +0.0143] | 0.333 (0.333) |
| OASIS-1 | A3 | F | KID | −0.0137, −0.0085 | −0.0111 [−0.0137, −0.0085] | 0.333 (0.333) |
| Churches | A3 | H | precision | +0.0305, +0.0825, +0.1195 | +0.0775 [+0.0305, +0.1195] | 0.100 (0.100) |
| Churches | A3 | F | precision | +0.0535, +0.0810, +0.1260 | +0.0868 [+0.0535, +0.1260] | 0.100 (0.100) |
| Churches | A3 | H | KID | −0.0158, −0.2450, −0.1742 | −0.1450 [−0.2450, −0.0158] | 0.100 (0.100) |
| Churches | A3 | F | KID | −0.0257, −0.2401, −0.1808 | −0.1489 [−0.2401, −0.0257] | 0.100 (0.100) |
| Churches | A1 | H | precision | +0.1075, +0.0520 | +0.0798 [+0.0520, +0.1075] | 0.333 (0.333) |
| Churches | A1 | F | precision | +0.1090, +0.0555 | +0.0823 [+0.0555, +0.1090] | 0.333 (0.333) |
| Bedrooms | A3 | H | precision | +0.1265, +0.0970 | +0.1118 [+0.0970, +0.1265] | 0.333 (0.333) |
| Bedrooms | A3 | F | precision | +0.1210, +0.1020 | +0.1115 [+0.1020, +0.1210] | 0.333 (0.333) |
| Bedrooms | A3 | H | KID | −0.0725, −0.1296 | −0.1011 [−0.1296, −0.0725] | 0.333 (0.333) |
| Bedrooms | A3 | F | KID | −0.0740, −0.1302 | −0.1021 [−0.1302, −0.0740] | 0.333 (0.333) |

Every contrast (photograph density and A1 KID included, and F against R5) is in
`heldout_fidelity.json` under `contrasts`. With 3 seeds per cell the smallest attainable p is 0.1,
and with 2 seeds it is 1/3, so no contrast here can reach p < 0.05.

## The pre-registered reading, applied literally (IXI, precision against R⁻)

| quantity | value |
|---|---|
| $\Delta_F(s)$, s = 1, 2, 3 | +0.1535, +0.0915, +0.0605 (all > 0) |
| $G_F$ | +0.1018 |
| $\Delta_H(s)$, s = 1, 2, 3 | +0.1605, +0.0650, +0.0670 (all > 0) |
| $G_H$ | +0.0975 |
| $0.5\,G_F$ | +0.0509 |

| rule | condition | holds |
|---|---|---|
| 1 | $G_F \le 0$ or not all $\Delta_F > 0$ | no |
| 2 | all $\Delta_H > 0$ and $G_H \ge 0.5\,G_F$ (0.0975 ≥ 0.0509) | **yes** ← this outcome |
| 3 | all $\Delta_H > 0$ and $G_H < 0.5\,G_F$ | — |
| 4 | otherwise | — |

**Reading: under the pre-registered rule, the IXI outcome is "the fidelity gain generalises to
unseen seeds" (primary endpoint, precision against R⁻; $G_H = 0.0975$ against $G_F = 0.1018$,
ratio 0.96).** Rule 1 did not fire, so the KID fallback was not run
(`kid_comparator_gain_positive` = true).

**R5 sensitivity (H only).**

| dataset | metric | $\Delta_H(\text{R5})$ per seed | mean | signs agree |
|---|---|---|---|---|
| IXI | precision | +0.2040, +0.1480, +0.1270 | +0.1597 | yes (all > 0) |
| IXI | KID | −0.0056, −0.0075, −0.0033 | −0.0055 | yes (all < 0, A3 better) |
| OASIS-1 | precision | +0.2605, +0.2080 | +0.2343 | yes (all > 0) |
| OASIS-1 | KID | −0.0258, −0.0217 | −0.0237 | yes (all < 0, A3 better) |

The precision gain on H holds under both references, conservative (R⁻) and liberal (R5). As
expected, it is larger against R5.

| run | precision H vs R5 [95% CI] | KID H vs R5 | precision F vs R5 (desc.) | KID F vs R5 (desc.) |
|---|---|---|---|---|
| `ixi_A0_s1` | 0.450 [0.414, 0.485] | 0.0276 | 0.442 | 0.0226 |
| `ixi_A0_s2` | 0.532 [0.496, 0.566] | 0.0273 | 0.498 | 0.0244 |
| `ixi_A0_s3` | 0.520 [0.490, 0.548] | 0.0255 | 0.475 | 0.0208 |
| `ixi_A3_s1` | 0.654 [0.601, 0.704] | 0.0220 | 0.460 | 0.0206 |
| `ixi_A3_s2` | 0.680 [0.631, 0.726] | 0.0198 | 0.466 | 0.0226 |
| `ixi_A3_s3` | 0.647 [0.595, 0.697] | 0.0221 | 0.428 | 0.0246 |
| `ixi_A1_s1` | 0.580 [0.530, 0.626] | 0.0192 | 0.395 | 0.0209 |
| `ixi_A1_s2` | 0.644 [0.582, 0.699] | 0.0223 | 0.462 | 0.0224 |
| `oasis1_A0_s1` | 0.547 [0.509, 0.587] | 0.0371 | 0.514 | 0.0349 |
| `oasis1_A0_s2` | 0.609 [0.579, 0.639] | 0.0315 | 0.572 | 0.0319 |
| `oasis1_A3_s1` | 0.808 [0.759, 0.856] | 0.0114 | 0.610 | 0.0217 |
| `oasis1_A3_s2` | 0.817 [0.776, 0.857] | 0.0098 | 0.586 | 0.0191 |

**The other datasets (same rule, not the endpoint).**

- **OASIS-1** (descriptive, transfer, 2 seeds): "the fidelity gain generalises to unseen seeds".
  $\Delta_H$ = +0.1675 and +0.1005, $G_H$ = +0.1340 against $G_F$ = +0.1292.
- **Churches** (descriptive only; mechanical): "generalises". $\Delta_H$ = +0.0305, +0.0825 and
  +0.1195, $G_H$ = +0.0775 against $G_F$ = +0.0868. A3's precision is 0.04–0.13, so the model
  still fails.
- **Bedrooms** (descriptive only; mechanical): "generalises". $\Delta_H$ = +0.1265 and +0.0970,
  $G_H$ = +0.1118 against $G_F$ = +0.1115.

**Description.**

- **Precision.** On IXI, A3's precision against R⁻ is the same on unseen seeds as on training
  seeds: the A3 gap $m_F - m_H$ is +0.005, −0.000 and −0.009, against A0's +0.012, −0.027 and
  −0.003. Its precision gain over A0 survives intact: 0.0975 on H against 0.1018 on F.
- **Density.** The gain shrinks from +0.131 (F) to +0.086 (H).
- **KID does not transfer the same way.** On IXI, A3's KID gain over A0 falls from −0.0298 on F
  (−63% of A0's 0.0472) to −0.0041 on H (−7% of A0's 0.0579). That is 14% of the F gain, and
  seed 3's Δ is −0.0002. On OASIS-1, A3's KID on H is *worse* than A0's (+0.0089, +0.0143).
- **A gap specific to the W/8 arms.** The KID gap $m_F - m_H$ is ≈ −0.036 for both W/8 arms on
  IXI (A3 and A1) against −0.011 for A0. Recall on H is 0.08–0.10 for A3 and A1 against
  0.17–0.29 for A0.
- **The likely reason.** A W/8 chain stays near its prior state (`heldout_grid.png`: A3's two
  samples keep the seed's layout and slice level; A0's wander to other levels). Its 2,000 H
  samples are therefore 40 tight clusters around 40 layouts. KID and recall compare whole
  distributions and penalise that collapse; precision and density ask only whether each sample
  lies on the real manifold. The F set has 2,000 distinct prior states, so it does not collapse.
- **What this means for the KID result.** The pattern is consistent with the −63% KID gain of
  table 2 coming largely from the 2,000-seed sampling design: at W/8 each sample copies the coarse
  layout of its seed image, so the sample set inherits the training set's coarse diversity.
  - This is *not tested*. H and F differ in their number of prior states (40 against 2,000), and
    no available set separates that from "unseen against training".
  - The KID gain is therefore not shown to be a gain in per-sample fidelity, whereas the precision
    gain is.
  - This is a description, not a re-reading: the pre-registered endpoint is precision.
  - *(Wording softened by main at merge, 2026-10-03; the agent's version said "is therefore in
    large part a property of the … sampling design".)*
- **R5.** Against R5, which removes A0's slice-level drift penalty, A3's H KID is lower than
  A0's on every seed of both MRI datasets (IXI −0.0055, OASIS-1 −0.0237).

![held-out grid](heldout_grid.png)

`heldout_grid.png`: IXI and Churches, run seed 1, A0 and A3. The top row of each block is the
first 8 held-out seed images (dataset idx 5, 35, 195, 275, 335, 355, 395, 515 on IXI); the next
two rows are the first two of each seed's 50 samples.

## Caveats

- **The limit of the check.** Unseen seeds still pass a real coarse layout to the chain: at W/8
  the prior carries the low band of a real, if unseen, brain. A3's held-out samples can therefore
  be precise by inheriting that layout, not by generating one. The check rules out ties to the
  *training set*. It does not rule out inheritance, and it says nothing about generation from
  scratch.
- **Few prior states in H.** H has 40 prior states. Recall, coverage and KID on H are bounded by
  that, and so, for the W/8 arms, by the clustering described above; the README reports them, but
  the rule does not use them.
- **Different set sizes and seeds.** H (40 × 50, rng 2026) and F (2,000 × 1, rng 0) differ in
  sampling noise seed and in structure. The gap $m_F - m_H$ mixes "training versus unseen seed"
  with "40 versus 2,000 prior states". The A0 gap is the closest available control, because A0's
  prior carries almost nothing.
- **F against R5 is contaminated.** The final-set seeds are drawn from all of `train`, including
  the R5 images, so F samples can inherit R5 images. Those columns are descriptive only.
- **Permutation floors.** p_min = 0.1 at 3 seeds and 1/3 at 2, so the p-values cannot show
  significance at 0.05. The reading is a sign-and-ratio rule over seeds, as pre-registered.
- **Photographs.** The photograph models fail (precision ≤ 0.14 in every run); their rows are
  descriptive.
- **The two KID columns are not comparable across references.** R⁻ (400 or 760 images) and R5
  (320) are different populations.

## Provenance

| what | value |
|---|---|
| analysis code | `f5ef21c2406eee40e72f1d29de2fed448e15e40d` (branch `ticket/T7.5-heldout-fidelity`; `git_dirty` false) |
| command | `python -m ihdm.cli.heldout_fidelity --eval-dir …/evaluation/eval_2488269 --data-root /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project --work …/_heldout_fidelity --out docs/RESULTS/heldout_fidelity` (`OMP_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`) |
| extractor | clean-fid torchscript Inception, `mode="clean"`, `device="cpu"`, weights `/tmp/inception-2015-12-05.pt`, k = 5 |
| hardware and time | local workstation CPU, 2 threads. 62 ms per image; 124 s per 2,000-image set. Anchors about 10 min (the references of 4 datasets and the 2 anchor runs). Full invocation 5,975 s (22 new runs plus 2 cached runs, the metrics and the bootstrap) |
| evaluation tars | `/media/mpascual/Sandisk2TB/research/spectral_allocation_heat_diffusion_project/evaluation/eval_2488269/<run_id>_amp-fp16.tar`. Read-only; byte sizes in the JSON (`runs.<id>.provenance.tar_bytes`) |
| sample sets | step 60,000 for every run. sha256 of each extracted `heldout/samples.npy` and `final/samples.npy` in `runs.<id>.provenance`. The sampling signatures (checkpoint sha256, rng, batch, δ) are copied from each `request.json` |

Reference sets (`references.<dataset>` in the JSON, sha256 of the sorted indices as little-endian
int64):

| dataset | ref | R⁻ | R5 | R⁻ sha256 | R5 sha256 |
|---|---|---|---|---|---|
| IXI | 800 | 400 | 320 | `5e016e53c5cd…` | `de9b80243106…` |
| OASIS-1 | 800 | 400 | 320 | in JSON | in JSON |
| Churches | 800 | 760 | — | in JSON | — |
| Bedrooms | 800 | 760 | — | in JSON | — |

Scratch (not committed):
`/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_heldout_fidelity/`. It holds
`features/` (per-run H and F features and the reference features, `.npy`), `runs/` (per-run
sidecars), `grid/` (grid excerpts), `extract/` (the small extracted members), `anchors.json`, the
logs and `tables.md`. The extracted sample arrays were deleted once their features existed.
`heldout_fidelity.json` holds every number above.
