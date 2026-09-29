## Table 5 — The A2′ control on Churches: Churches-matched against IXI-matched spacing

**PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated): not for quoting.**

**Status: complete (6/6 runs)**

| endpoint | A2′−A0: mean Δ [95% CI] | p | A2−A0: mean Δ [95% CI] | p | A2′−A2: mean Δ [95% CI] | p |
|---|---:|---:|---:|---:|---:|---:|
| LSD (final, 2k set) | -0.4074 [-0.5500, -0.2647] | 0.333 | -0.06205 [-0.3459, +0.2218] | 0.667 | -0.3453 [-0.4865, -0.2041] | 0.333 |
| T_τ (steps) | 0 [0, 0] | 1.000 | not computable: T_τ not reached (A2 s2) | — | not computable: T_τ not reached (A2 s2) | — |
| KID (headline) | -0.05071 [-0.1368, +0.03541] | 0.667 | -0.1371 [-0.1960, -0.07814] | 0.333 | +0.08634 [+0.05913, +0.1136] | 0.333 |
| FID | -39.84 [-91.54, +11.85] | 0.667 | -82.21 [-107.1, -57.32] | 0.333 | +42.36 [+15.55, +69.18] | 0.333 |
| recall | +0.001875 [-0.007500, +0.01125] | 0.667 | +0.1412 [+0.08375, +0.1988] | 0.333 | -0.1394 [-0.1875, -0.09125] | 0.333 |
| coverage | +0.03375 [+0.01125, +0.05625] | 0.333 | +0.08500 [+0.05625, +0.1137] | 0.333 | -0.05125 [-0.1025, 0] | 0.667 |
| M | +0.04644 [+0.04456, +0.04831] | 0.333 | +0.03114 [+0.02266, +0.03962] | 0.333 | +0.01530 [+0.004940, +0.02566] | 0.333 |
| M_lp | +0.03228 [+0.02941, +0.03514] | 0.667 | +0.04948 [+0.01963, +0.07933] | 0.333 | -0.01720 [-0.04992, +0.01552] | 0.667 |
| seed-NN fraction | +4.337e-19 [-1.000e-03, +0.001000] | 1.000 | +0.003000 [+0.001000, +0.005000] | 0.333 | -0.003000 [-0.004000, -0.002000] | 0.333 |
| D_pix | +0.001863 [+0.001824, +0.001902] | 0.333 | +0.001315 [+0.001054, +0.001577] | 0.333 | +5.478e-04 [+3.249e-04, +7.707e-04] | 0.333 |
| D_lp | +2.295e-04 [+1.973e-04, +2.616e-04] | 0.667 | +3.919e-04 [+1.258e-04, +6.580e-04] | 0.333 | -1.624e-04 [-3.963e-04, +7.150e-05] | 0.667 |
| inherited share as pre-registered (biased under a non-zero mean image; see inherited_band_audit.md) | -0.05121 [-0.05873, -0.04369] | 0.333 | -0.04309 [-0.05914, -0.02705] | 0.333 | -0.008121 [-0.03169, +0.01544] | 0.667 |
| within-seed share I_w (D23) | -0.03762 [-0.03840, -0.03683] | 0.333 | -0.02655 [-0.03184, -0.02127] | 0.333 | -0.01106 [-0.01556, -0.006560] | 0.333 |
| seed-mean bias fraction G_b (D23) | +0.01435 [+0.006066, +0.02263] | 0.333 | +0.01707 [+0.006199, +0.02794] | 0.333 | -0.002718 [-0.02187, +0.01644] | 0.667 |
| T_τ, 2k-set threshold (sensitivity) | 0 [0, 0] | 1.000 | not computable: T_τ not reached (A2 s2) | — | not computable: T_τ not reached (A2 s2) | — |

- A2′ uses the spacing variance-matched to Churches, A2 the one matched to IXI, both at σ_B,max = 96. A2′−A2 isolates whether matching the spacing to the data's own spectrum matters on photographs.
- Every contrast is paired over seeds 1 and 2 (the seeds A2 and A2′ have): the CI is [min, max] of two per-seed Δ and the permutation p has floor p_min = 1/3 (2 vs 2 seeds, 6 assignments). Nothing in this table can reach p < 0.33.
- Churches is undertrained (final LSD ≈ 1.0–1.45, oscillating across checkpoints, blurry samples); its contrasts are reported with that caveat.

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 (collection INCOMPLETE, created 2026-09-29T07:38:12.518998+00:00, git 9fb5c84635); tables: git 4adfff670e, 2026-09-29T09:15:36.919074+00:00</sub>
