## Table 1a — Cell table, one row per run: spectral and Inception endpoints

**PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated): not for quoting.**

**Status: incomplete: 24/30 runs** (missing: oasis1_A3_s1, oasis1_A3_s2, lsun_bedroom_A0_s1, lsun_bedroom_A0_s2, lsun_bedroom_A3_s1, lsun_bedroom_A3_s2)

| dataset | arm | seed | LSD final (2k) | LSD 60k (500) | T_τ | KID | KID 95% CI | FID | FID 95% CI | N_ref | recall | coverage | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| IXI | A0 | 1 | 0.2527 | 0.2497 | 5k | 0.04638 | [+0.04397, +0.04878] | 57.20 | [+56.58, +60.24] | 800 | 0.2575 | 0.4263 | ok |
| IXI | A0 | 2 | 0.2449 | 0.2404 | 35k | 0.05102 | [+0.04859, +0.05327] | 60.48 | [+59.81, +63.40] | 800 | 0.3375 | 0.4637 | ok |
| IXI | A0 | 3 | 0.2289 | 0.2271 | 5k | 0.04113 | [+0.03905, +0.04335] | 51.89 | [+51.35, +54.76] | 800 | 0.3262 | 0.5125 | ok |
| IXI | A3 | 1 | 0.08115 | 0.08400 | 5k | 0.01802 | [+0.01655, +0.01927] | 28.78 | [+29.21, +31.30] | 800 | 0.4700 | 0.7700 | ok |
| IXI | A3 | 2 | 0.07772 | 0.08047 | 5k | 0.01572 | [+0.01463, +0.01683] | 26.91 | [+27.60, +29.15] | 800 | 0.4000 | 0.8025 | ok |
| IXI | A3 | 3 | 0.07549 | 0.07686 | 5k | 0.01703 | [+0.01599, +0.01811] | 27.84 | [+28.51, +30.08] | 800 | 0.4263 | 0.7538 | ok |
| IXI | A1 | 1 | 0.1153 | 0.1180 | 5k | 0.01657 | [+0.01534, +0.01779] | 29.72 | [+30.36, +32.12] | 800 | 0.4763 | 0.7000 | ok |
| IXI | A1 | 2 | 0.07633 | 0.07907 | 5k | 0.01675 | [+0.01571, +0.01788] | 27.85 | [+28.47, +30.19] | 800 | 0.3800 | 0.7650 | ok |
| IXI | A2 | 1 | 0.2340 | 0.2284 | 5k | 0.04500 | [+0.04321, +0.04735] | 55.84 | [+55.65, +58.56] | 800 | 0.2863 | 0.4425 | ok |
| IXI | A2 | 2 | 0.2071 | 0.2042 | 5k | 0.04148 | [+0.03932, +0.04370] | 51.50 | [+51.00, +54.44] | 800 | 0.4313 | 0.5587 | ok |
| Churches | A0 | 1 | 1.452 | 1.445 | 5k | 0.2465 | [+0.2432, +0.2506] | 227.0 | [+226.9, +231.2] | 800 | 0.008750 | 0.007500 | ok |
| Churches | A0 | 2 | 1.127 | 1.129 | 5k | 0.3745 | [+0.3701, +0.3800] | 292.4 | [+291.7, +296.4] | 800 | 0 | 0.001250 | ok |
| Churches | A0 | 3 | 1.024 | 1.023 | 5k | 0.3141 | [+0.3097, +0.3194] | 253.6 | [+253.2, +258.3] | 800 | 0 | 0.01000 | ok |
| Churches | A3 | 1 | 0.6373 | 0.6416 | 5k | 0.2203 | [+0.2157, +0.2260] | 200.7 | [+200.8, +206.4] | 800 | 0.02875 | 0.05000 | ok |
| Churches | A3 | 2 | 1.085 | 1.092 | 5k | 0.1340 | [+0.1312, +0.1379] | 157.1 | [+157.3, +163.1] | 800 | 0.1200 | 0.1225 | ok |
| Churches | A3 | 3 | 0.7684 | 0.7796 | 5k | 0.1329 | [+0.1299, +0.1368] | 147.9 | [+148.8, +153.6] | 800 | 0.1338 | 0.1575 | ok |
| Churches | A1 | 1 | 0.6641 | 0.6656 | 5k | 0.1753 | [+0.1710, +0.1802] | 166.7 | [+167.0, +172.3] | 800 | 0.08125 | 0.1263 | ok |
| Churches | A1 | 2 | 1.012 | 1.015 | 5k | 0.1841 | [+0.1802, +0.1876] | 184.4 | [+184.6, +189.7] | 800 | 0.01625 | 0.05375 | ok |
| Churches | A2 | 1 | 1.107 | 1.101 | 5k | 0.1683 | [+0.1642, +0.1720] | 169.6 | [+169.6, +175.0] | 800 | 0.09250 | 0.1212 | ok |
| Churches | A2 | 2 | 1.348 | 1.347 | not reached | 0.1785 | [+0.1743, +0.1816] | 185.3 | [+185.4, +190.6] | 800 | 0.1988 | 0.05750 | ok |
| Churches | A2′ | 1 | 0.9024 | 0.9014 | 5k | 0.2819 | [+0.2778, +0.2858] | 238.8 | [+238.3, +242.9] | 800 | 0.001250 | 0.01875 | ok |
| Churches | A2′ | 2 | 0.8620 | 0.8581 | 5k | 0.2377 | [+0.2335, +0.2429] | 200.9 | [+200.4, +205.3] | 800 | 0.01125 | 0.05750 | ok |
| OASIS-1 | A0 | 1 | 0.2434 | 0.2472 | 60k | 0.02569 | [+0.02455, +0.02744] | 38.10 | [+38.41, +40.63] | 800 | 0.4300 | 0.6887 | ok |
| OASIS-1 | A0 | 2 | 0.2183 | 0.2209 | 50k | 0.02141 | [+0.02025, +0.02270] | 34.15 | [+34.47, +36.56] | 800 | 0.4400 | 0.7612 | ok |
| OASIS-1 | A3 | 1 | — | — | — | — | — | — | — | — | — | — | missing: not evaluated |
| OASIS-1 | A3 | 2 | — | — | — | — | — | — | — | — | — | — | missing: not evaluated |
| Bedrooms | A0 | 1 | — | — | — | — | — | — | — | — | — | — | missing: not evaluated |
| Bedrooms | A0 | 2 | — | — | — | — | — | — | — | — | — | — | missing: not evaluated |
| Bedrooms | A3 | 1 | — | — | — | — | — | — | — | — | — | — | missing: not evaluated |
| Bedrooms | A3 | 2 | — | — | — | — | — | — | — | — | — | — | missing: not evaluated |

- LSD final is the 2,000-sample final set (rng_seed 0, with replacement); LSD 60k (500) is the 500 frozen seeds of the curve at the last checkpoint.
- T_τ uses the primary threshold (same-seed A0 LSD at the last checkpoint, 500 seeds); table 7 gives both thresholds.
- KID is the headline Inception metric. FID against N_ref = 800 is biased upward and its bootstrap interval over resampled samples is shifted upward (T4.3), so it can lie above the point estimate.

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 (collection INCOMPLETE, created 2026-09-29T07:38:12.518998+00:00, git 9fb5c84635); tables: git e12bebbfa8, 2026-09-29T08:03:16.310965+00:00</sub>
