## Table 3 — Interaction: Δ_IXI − Δ_Churches for A3 vs A0 (the headline)

**PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated): not for quoting.**

**Status: complete (12/12 runs)**

| endpoint | Δ IXI (A3−A0) | Δ IXI / A0 | Δ Churches (A3−A0) | Δ Churches / A0 | Δ_IXI − Δ_Churches | 95% CI | perm. p | p_min | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LSD (final, 2k set) | -0.1640 | -67.7% | -0.3707 | -30.9% | +0.2066 | [-0.1210, +0.6496] | 0.700 | 0.100 | not detectable at this budget |
| T_τ (steps) | -10k | -66.7% | 0 | +0.0% | -10k | [-30k, 0] | 1.000 | 0.100 | not detectable at this budget |
| KID (headline) | -0.02925 | -63.3% | -0.1493 | -47.9% | +0.1201 | [-0.002164, +0.2099] | 0.300 | 0.100 | not detectable at this budget |
| FID | -28.68 | -50.7% | -89.08 | -34.6% | +60.40 | [-2.185, +105.2] | 0.300 | 0.100 | not detectable at this budget |
| recall | +0.1250 | +40.7% | +0.09125 | +3128.6% | +0.03375 | [-0.05417, +0.1217] | 0.600 | 0.100 | not detectable at this budget |
| coverage | +0.3079 | +65.9% | +0.1038 | +1660.0% | +0.2042 | [+0.1350, +0.2733] | 0.100 | 0.100 | CI excludes 0 |
| M | -0.03638 | -4.0% | +0.1333 | +24.5% | -0.1697 | [-0.1941, -0.1394] | 0.100 | 0.100 | CI excludes 0 |
| M_lp | -0.7907 | -77.1% | -0.5775 | -83.6% | -0.2131 | [-0.2805, -0.1518] | 0.100 | 0.100 | CI excludes 0 |
| seed-NN fraction | +0.6588 | +1051.3% | +0.6138 | +6949.1% | +0.04500 | [+0.03933, +0.05433] | 0.100 | 0.100 | CI excludes 0 |
| D_pix | -0.002647 | -23.2% | +1.299e-04 | +4.1% | -0.002777 | [-0.003486, -0.002039] | 0.100 | 0.100 | CI excludes 0 |
| D_lp | -2.646e-04 | -94.5% | -8.248e-04 | -96.6% | +5.602e-04 | [+3.087e-04, +7.700e-04] | 0.100 | 0.100 | CI excludes 0 |
| inherited share (measured) | +0.9224 | -108.2% | +0.04480 | +5.9% | +0.8776 | [+0.8005, +0.9397] | 0.100 | 0.100 | CI excludes 0 |
| T_τ, 2k-set threshold (sensitivity) | 0 | +0.0% | 0 | +0.0% | 0 | [0, 0] | 1.000 | 0.100 | not detectable at this budget |

- Δ = A3 − A0 paired by seed (3 seeds on each dataset). The interaction CI resamples the three IXI and the three Churches per-seed Δ independently (10,000 draws, percentile). A negative value on a lower-is-better endpoint (LSD, T_τ, KID, FID) means A3 helps IXI more than Churches, the direction the claim predicts.
- The interaction is on each metric's raw scale, as pre-registered; where the two datasets' A0 levels differ (Churches' LSD is about five times IXI's) the raw difference is dominated by the larger scale. 'Δ / A0' (mean Δ over the mean A0 value, same seeds) is a descriptive, scale-free reading and carries no test.
- p: exact permutation of the dataset label over the six pooled per-seed Δ (C(6,3) = 20 assignments), so p cannot fall below p_min = 0.1.
- Only the development pair enters: OASIS-1 and Bedrooms (2 seeds, A0/A3 only) are in the transfer table 6, without p-values.
- Churches is undertrained (final LSD ≈ 1.0–1.45, oscillating across checkpoints, blurry samples); its contrasts are reported with that caveat.

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 (collection INCOMPLETE, created 2026-09-29T07:38:12.518998+00:00, git 9fb5c84635); tables: git e12bebbfa8, 2026-09-29T08:03:16.310965+00:00</sub>
