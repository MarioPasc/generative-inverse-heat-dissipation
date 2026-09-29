## Table 4 — Decomposition of the A3 effect into terminal blur (A1) and spacing (A2)

**PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated): not for quoting.**

**Status: complete (20/20 runs)**

| dataset | endpoint | Δ A3 (s1,s2) | Δ A1 | Δ A2 | share A1 | share A2 | A1 + A2 | reading |
|---|---|---:|---:|---:|---:|---:|---:|---|
| IXI | LSD (final, 2k set) | -0.1694 | -0.1530 | -0.02826 | 0.90 | 0.17 | 1.07 | ok |
| IXI | T_τ (steps) | -15k | -15k | -15k | 1.00 | 1.00 | 2.00 | A3−A0 not detectable (table 2): shares not interpretable |
| IXI | KID (headline) | -0.03183 | -0.03204 | -0.005454 | 1.01 | 0.17 | 1.18 | ok |
| IXI | FID | -30.99 | -30.05 | -5.167 | 0.97 | 0.17 | 1.14 | ok |
| IXI | recall | +0.1375 | +0.1306 | +0.06125 | 0.95 | 0.45 | 1.40 | ok |
| IXI | coverage | +0.3412 | +0.2875 | +0.05562 | 0.84 | 0.16 | 1.01 | ok |
| IXI | M | -0.03960 | -0.02529 | +5.410e-04 | 0.64 | -0.01 | 0.63 | ok |
| IXI | M_lp | -0.8078 | -0.7792 | -0.04892 | 0.96 | 0.06 | 1.03 | ok |
| IXI | seed-NN fraction | +0.6612 | +0.6058 | +0.01650 | 0.92 | 0.02 | 0.94 | ok |
| IXI | D_pix | -0.002623 | -0.002619 | +1.995e-04 | 1.00 | -0.08 | 0.92 | ok |
| IXI | D_lp | -2.593e-04 | -2.584e-04 | +1.161e-05 | 1.00 | -0.04 | 0.95 | ok |
| IXI | inherited share as pre-registered (biased under a non-zero mean image; see inherited_band_audit.md) | +0.8926 | +0.9164 | -0.1168 | 1.03 | -0.13 | 0.90 | ok |
| IXI | within-seed share I_w (D23) | +0.07868 | +0.07855 | -0.005984 | 1.00 | -0.08 | 0.92 | ok |
| IXI | seed-mean bias fraction G_b (D23) | -0.8155 | -0.8394 | +0.1110 | 1.03 | -0.14 | 0.89 | ok |
| IXI | T_τ, 2k-set threshold (sensitivity) | 0 | 0 | 0 | — | — | — | A3−A0 not detectable (table 2): shares not interpretable |
| Churches | LSD (final, 2k set) | -0.4284 | -0.4517 | -0.06205 | 1.05 | 0.14 | 1.20 | ok |
| Churches | T_τ (steps) | 0 | 0 | — | — | — | — | not computable: T_τ not reached (A2 s2) |
| Churches | KID (headline) | -0.1334 | -0.1308 | -0.1371 | 0.98 | 1.03 | 2.01 | ok |
| Churches | FID | -80.78 | -84.16 | -82.21 | 1.04 | 1.02 | 2.06 | ok |
| Churches | recall | +0.07000 | +0.04438 | +0.1412 | 0.63 | 2.02 | 2.65 | ok |
| Churches | coverage | +0.08188 | +0.08562 | +0.08500 | 1.05 | 1.04 | 2.08 | ok |
| Churches | M | +0.1486 | +0.1491 | +0.03114 | 1.00 | 0.21 | 1.21 | ok |
| Churches | M_lp | -0.5556 | -0.5507 | +0.04948 | 0.99 | -0.09 | 0.90 | ok |
| Churches | seed-NN fraction | +0.6138 | +0.6067 | +0.003000 | 0.99 | 0.00 | 0.99 | ok |
| Churches | D_pix | +5.196e-04 | +5.622e-04 | +0.001315 | 1.08 | 2.53 | 3.61 | A3−A0 not detectable (table 2): shares not interpretable |
| Churches | D_lp | -7.195e-04 | -7.172e-04 | +3.919e-04 | 1.00 | -0.54 | 0.45 | ok |
| Churches | inherited share as pre-registered (biased under a non-zero mean image; see inherited_band_audit.md) | +0.02925 | +0.03258 | -0.04309 | 1.11 | -1.47 | -0.36 | ok |
| Churches | within-seed share I_w (D23) | -0.01049 | -0.01135 | -0.02655 | 1.08 | 2.53 | 3.61 | A3−A0 not detectable (table 2): shares not interpretable |
| Churches | seed-mean bias fraction G_b (D23) | -0.03953 | -0.04370 | +0.01707 | 1.11 | -0.43 | 0.67 | ok |
| Churches | T_τ, 2k-set threshold (sensitivity) | 0 | 0 | — | — | — | — | not computable: T_τ not reached (A2 s2) |

- All Δ are against A0 and use only seeds 1 and 2, the seeds A1 and A2 have, so the three Δ share their A0 values. share A1 = Δ_A1 / Δ_A3 and share A2 = Δ_A2 / Δ_A3; a sum near 1 means the two knobs add up to the A3 effect, a sum far from 1 means they interact. No CI: A1 and A2 have 2 seeds.
- A share is flagged 'not interpretable' when the 3-seed A3−A0 interval of table 2 contains 0: a ratio over an undetectable denominator carries no information.

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 (collection INCOMPLETE, created 2026-09-29T07:38:12.518998+00:00, git 9fb5c84635); tables: git 4adfff670e, 2026-09-29T09:15:36.919074+00:00</sub>
