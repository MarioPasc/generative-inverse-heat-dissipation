# δ sweep (T7.2, M7 diagnostic, post hoc, not pre-registered as an experiment)

**Question** (`docs/SPECIFICATIONS/M7-diagnostics/README.md`, question 2): is the under-dispersion
of the samples a sampler setting? The sampler adds noise of sd δ to the prior and at every reverse
step. δ is used only at sampling time and never in training. We re-sample four existing 60k
checkpoints at δ/σ ∈ {1.25, 2, 3} (σ = 0.01) with no training.

**Protocol.** Each run is evaluated at checkpoint 60,000 (EMA weights) on one A100, with fp16
autocast (D20) and sample batch 32. The commands come from `slurm/diag_eval/delta_sweep.sbatch`:
`python -m ihdm.cli.evaluate_run --ckpts 60000 --amp fp16 --delta <δ> --final-from-lsd
--n-seeds 40 --n-per-seed 5`. Two sample sets are drawn per (run, δ):

- **LSD set.** The 500 frozen training seeds (`eval_seeds_500.npy`), sampled with rng seed 2026.
  The noise stream is the same at every δ, so the three conditions of a run are paired. This set
  gives the LSD, the octaves and the variance ratio. Through `--final-from-lsd` it also gives
  KID, FID, precision, recall, density, coverage and M.
- **Held-out set.** 40 seed subjects × 5 samples. It gives D_pix, D_lp and the inherited band.

**Not comparable with table 1a.** Inception metrics and M here are computed on the 500-seed LSD
set. They are not computed on the 2,000-seed production final set that tables 1a and 2 use.
At δ = 1.25σ they therefore differ from the published values. For example, on `ixi_A0_s1` KID is
0.0418 here and 0.0464 in production, and precision is 0.584 here and 0.574 in production. Only
the δ-to-δ contrasts inside this folder are paired and like for like. The held-out set is 40 × 5,
not the production 40 × 50.

**Provenance.**

- Jobs: array job 2550585, four tasks on four nodes:

  | task | run | job id | node | wall time |
  |---|---|---|---|---|
  | 0 | `ixi_A0_s1` | 2550609 | exa04 | 1:31:59 |
  | 1 | `ixi_A3_s1` | 2550613 | exa01 | 1:35:03 |
  | 2 | `lsun_church_A0_s1` | 2550705 | exa02 | 1:39:06 |
  | 3 | `lsun_church_A3_s1` | 2550585 | exa03 | 1:33:54 |

  The total is 6.3 A100-h.
- Code: `b0306769fad5dd301d4f1b76b01fbed2ce89445e`.
- Checkpoint sha256 values are in `delta_sweep.json` (`tasks[*].rows[*].checkpoint_sha256`).
- Raw trees: `~/execs/ihdm/diag_eval/delta_sweep/<run>_delta_sweep.tar` on Picasso.
- Every number below is in `delta_sweep.json`, written by `slurm/diag_eval/collect.py merge`.

## Reproduction anchor

At δ = 1.25σ the 500-seed LSD must equal the stored `lsd_060000` within 0.003. It does for all
four runs, and to every digit that `write_json` keeps. The table compares against both forms of
the stored value. The 4-digit column is the value in `docs/RESULTS/tables/t7_t_tau.md`. The
6-digit column is the production `summary.json`.

| run | stored (t7) | stored (summary) | reproduced | difference vs t7 | ok |
|---|---|---|---|---|---|
| `ixi_A0_s1` | 0.2497 | 0.249665 | 0.249665 | −0.000035 | yes |
| `ixi_A3_s1` | 0.08400 | 0.083997 | 0.083997 | −0.000003 | yes |
| `lsun_church_A0_s1` | 1.445 | 1.44537 | 1.44537 | +0.00037 | yes |
| `lsun_church_A3_s1` | 0.6416 | 0.641611 | 0.641611 | +0.000011 | yes |

## Results per run × δ

Column definitions:

- **oct. mean** is the mean over the 8 octave bins of the signed log10 ratio P_samples / P_ref.
- **64–96** is the finest octave.
- Both octave columns are negative when the samples carry too little variance.
- **I_w** is the measured inherited share over all modes. **pred.** is its linear-Gaussian
  prediction.
- The full octave profiles and the low-band shares are in the JSON.

| run | δ/σ | δ | LSD | var. ratio | oct. mean | 64–96 | KID [95% CI] | FID | precision | recall | density | coverage | M | D_pix | D_lp | I_w | pred. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ixi_A0_s1` | 1.25 | 0.0125 | 0.2497 | 0.651 | −0.197 | −0.212 | 0.0418 [0.0383, 0.0461] | 58.1 | 0.584 | 0.386 | 0.296 | 0.274 | 0.932 | 0.00936 | 0.00022 | −0.767 | 0.003 |
| `ixi_A0_s1` | 2.00 | 0.02 | 0.6814 | 3.394 | +0.257 | +1.518 | 0.2748 [0.2690, 0.2823] | 315.8 | 0.000 | 0.095 | 0.000 | 0.000 | 2.171 | 0.09720 | 0.00153 | −2.885 | 0.003 |
| `ixi_A0_s1` | 3.00 | 0.03 | 0.5987 | 3.028 | +0.385 | +1.019 | 0.3995 [0.3907, 0.4112] | 356.8 | 0.000 | 0.000 | 0.000 | 0.000 | 2.179 | 0.08391 | 0.00129 | −2.779 | 0.003 |
| `ixi_A3_s1` | 1.25 | 0.0125 | 0.0840 | 0.882 | −0.047 | −0.108 | 0.0175 [0.0155, 0.0199] | 34.0 | 0.698 | 0.571 | 0.476 | 0.517 | 0.871 | 0.00725 | 0.00001 | +0.076 | 0.080 |
| `ixi_A3_s1` | 2.00 | 0.02 | 0.3955 | 1.329 | +0.123 | +0.665 | 0.4171 [0.4059, 0.4285] | 312.3 | 0.000 | 0.237 | 0.000 | 0.000 | 1.346 | 0.02955 | 0.00003 | −0.357 | 0.080 |
| `ixi_A3_s1` | 3.00 | 0.03 | 0.8147 | 3.813 | +0.239 | +1.666 | 0.3530 [0.3468, 0.3609] | 350.0 | 0.000 | 0.000 | 0.000 | 0.000 | 2.382 | 0.11314 | 0.00140 | −3.274 | 0.080 |
| `lsun_church_A0_s1` | 1.25 | 0.0125 | 1.4454 | 0.285 | −1.253 | −2.368 | 0.2438 [0.2373, 0.2511] | 233.3 | 0.014 | 0.015 | 0.003 | 0.007 | 0.537 | 0.00229 | 0.00071 | +0.769 | 0.018 |
| `lsun_church_A0_s1` | 2.00 | 0.02 | 1.0814 | 3.358 | −0.003 | +1.386 | 0.4177 [0.4125, 0.4245] | 363.7 | 0.000 | 0.000 | 0.000 | 0.000 | 1.935 | 0.12628 | 0.00028 | −2.185 | 0.018 |
| `lsun_church_A0_s1` | 3.00 | 0.03 | 1.2740 | 3.970 | −0.132 | +1.233 | 0.4620 [0.4571, 0.4689] | 367.6 | 0.000 | 0.045 | 0.000 | 0.000 | 2.291 | 0.15906 | 0.00012 | −2.949 | 0.018 |
| `lsun_church_A3_s1` | 1.25 | 0.0125 | 0.6416 | 0.648 | −0.488 | −0.773 | 0.2190 [0.2092, 0.2285] | 212.7 | 0.052 | 0.040 | 0.017 | 0.029 | 0.672 | 0.00304 | 0.00002 | +0.793 | 0.289 |
| `lsun_church_A3_s1` | 2.00 | 0.02 | 1.1859 | 4.564 | −0.387 | +2.014 | 0.4744 [0.4709, 0.4780] | 364.0 | 0.000 | 0.000 | 0.000 | 0.000 | 2.417 | 0.18369 | 0.00004 | −3.806 | 0.289 |
| `lsun_church_A3_s1` | 3.00 | 0.03 | 1.6537 | 4.809 | −0.885 | +2.077 | 0.4122 [0.4098, 0.4149] | 326.9 | 0.000 | 0.000 | 0.000 | 0.000 | 2.473 | 0.19473 | 0.00001 | −4.183 | 0.289 |

## Pre-registered reading (M7-diagnostics/README.md, applied literally)

> The under-dispersion is "partly a sampler setting" if, on IXI A0, some δ > 1.25σ raises the
> variance ratio of the 500-seed LSD set by ≥ 0.10 without raising KID. If the variance ratio
> rises while KID and precision worsen, the added variance is noise.

- On `ixi_A0_s1` both values above 1.25σ raise the variance ratio by more than 0.10:
  - δ = 2σ: 0.651 → 3.394, a rise of 2.743.
  - δ = 3σ: 0.651 → 3.028, a rise of 2.377.
- Both also raise KID: 0.0418 → 0.2748 and 0.3995. The bootstrap CIs do not overlap.
- **The "partly a sampler setting" criterion is not met.**
- The second clause applies: the variance ratio rises while KID worsens and precision falls from
  0.584 to 0.000. **By the pre-registered reading, the variance added by a larger δ is noise.**
- One seed per run, so the reading is descriptive.

## Description (not confirmatory)

At δ = 1.25σ all four runs are under-dispersed (variance ratio 0.29 to 0.88). Already at 2σ every
run is over-dispersed (variance ratio 1.3 to 4.6). Precision, density and coverage fall to zero in
every run at both δ > 1.25σ.

The added variance goes mostly to the finest scales. In the 64–96 c/img octave the log10 ratio
moves from −0.11…−2.37 to +0.67…+2.08. On Churches the coarse octaves (0.5–4 c/img) end further below zero at
2σ and 3σ; on A3 the 1–2 c/img bin goes from −0.01 to −1.38 at 2σ. LSD
therefore does not fall monotonically with variance.

The grid is coarse (no value between 1.25σ and 2σ), so a smaller increase in δ is not excluded.
Within the tested range, a larger δ does not give the variance back at the scales where it is
missing.
