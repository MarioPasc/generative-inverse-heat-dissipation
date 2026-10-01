# Exploratory readings of the 30-run results

**Exploratory: defined after seeing the data (2026-10-01), not pre-registered.** Same statistics as tables 2-6: with 3 seeds a 95% CI is [min, max] of the per-seed Δ, so 'CI excludes 0' means 'all seeds agree in sign'; p_min is 0.1 at 3 vs 3 seeds and 1/3 at 2 vs 2. Transfer rows give signs only.

Command: `PYTHONPATH=$PWD python -m ihdm.analysis.exploratory --results <results_dir> --out docs/RESULTS/exploratory/`. Source: `/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results` (collection COMPLETE).

Definitions. e_b = log10 P̄_S(b) − log10 P̄_R(b) on the eight octaves of the final 2k set (`final.json` `lsd_octaves`); octave RMS = sqrt(mean e_b²) (tracks LSD, which uses 43 finer bins); level ē = mean e_b; shape = sd(e_b), so RMS² = ē² + sd². 'Low' = 0.5-4 c/img (a σ_B,max = 24 prior keeps d_K² ≈ 0.54 at 1 c/img, 0.08 at 2 and 5e-5 at 4), 'high' = 4-96 c/img. Variance ratio = ΣP_S/ΣP_R (non-DC). ρ = (1 − I_w)/(1 − I) from table 1c. Precision and density: Naeem et al. (2020), k = 5, Inception pool features, as stored by every run.

## X1. Seed means per cell

| dataset | arm | n | LSD (final, 2k; pre-registered, for reference) | LSD, mean of 45k-60k (500 seeds) | octave RMS error (final) | octave level ē (final) | octave shape sd(e) (final) | octave RMS, 0.5-4 c/img | octave RMS, 4-96 c/img | log10 variance ratio (final) | precision (Inception, k = 5) | density (Inception, k = 5) | regeneration ratio ρ = (1 − I_w)/(1 − I) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IXI T1 | A0 | 3 | 0.2422 | 0.2348 | 0.2158 | -0.1873 | 0.1071 | 0.2846 | 0.1609 | -0.1776 | 0.6342 | 0.3735 | 0.3427 |
| IXI T1 | A3 | 3 | 0.07812 | 0.08379 | 0.06821 | -0.04645 | 0.04992 | 0.03325 | 0.08235 | -0.05284 | 0.7220 | 0.4795 | 0.2848 |
| IXI T1 | A1 | 2 | 0.09581 | 0.08801 | 0.08463 | -0.05768 | 0.06192 | 0.03612 | 0.1032 | -0.07185 | 0.6623 | 0.4164 | 0.2849 |
| IXI T1 | A2 | 2 | 0.2205 | 0.2147 | 0.1961 | -0.1707 | 0.09609 | 0.2417 | 0.1627 | -0.1740 | 0.6367 | 0.3809 | 0.3480 |
| LSUN Churches | A0 | 3 | 1.201 | 1.178 | 1.158 | -1.049 | 0.4866 | 0.6781 | 1.363 | -0.5323 | 0.003167 | 7.000e-04 | 0.06493 |
| LSUN Churches | A3 | 3 | 0.8303 | 0.8513 | 0.8264 | -0.6168 | 0.5483 | 0.05723 | 1.044 | -0.1856 | 0.08883 | 0.03310 | 0.09332 |
| LSUN Churches | A1 | 2 | 0.8378 | 0.7929 | 0.8340 | -0.6274 | 0.5484 | 0.05913 | 1.054 | -0.1887 | 0.08350 | 0.02955 | 0.09284 |
| LSUN Churches | A2 | 2 | 1.228 | 1.272 | 1.211 | -1.057 | 0.5861 | 0.5535 | 1.468 | -0.4859 | 0.05750 | 0.02160 | 0.08273 |
| LSUN Churches | A2′ | 2 | 0.8822 | 0.8881 | 0.8349 | -0.7789 | 0.3004 | 0.6145 | 0.9418 | -0.4819 | 0.03100 | 0.01080 | 0.09399 |
| OASIS-1 T1 | A0 | 2 | 0.2308 | 0.2510 | 0.2217 | -0.1924 | 0.1101 | 0.2903 | 0.1676 | -0.1787 | 0.6833 | 0.5195 | 0.4490 |
| OASIS-1 T1 | A3 | 2 | 0.1118 | 0.1159 | 0.1061 | -0.07951 | 0.07004 | 0.03216 | 0.1317 | -0.08724 | 0.8355 | 0.8325 | 0.3372 |
| LSUN Bedrooms | A0 | 2 | 1.430 | 1.333 | 1.388 | -1.245 | 0.6115 | 0.6905 | 1.671 | -0.5545 | 0.01175 | 0.002450 | 0.06331 |
| LSUN Bedrooms | A3 | 2 | 0.9849 | 0.9930 | 0.9978 | -0.7247 | 0.6858 | 0.06057 | 1.261 | -0.1647 | 0.1205 | 0.03900 | 0.07431 |

## X2. Contrasts, transfer signs and the A3 interaction

**LSD, mean of 45k-60k (500 seeds)** (lower is better)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | -0.1510 | [-0.1629, -0.1325] | -0.1629 / -0.1578 / -0.1325 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | -0.1577 | [-0.1608, -0.1545] | -0.1545 / -0.1608 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | -0.03092 | [-0.03755, -0.02430] | -0.02430 / -0.03755 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A3−A0 | 3 | -0.3267 | [-0.6688, -0.07132] | -0.6688 / -0.07132 / -0.2399 | 0.200 (min 0.100) | CI excludes 0 |
| LSUN Churches A1−A0 | 2 | -0.4588 | [-0.7047, -0.2130] | -0.7047 / -0.2130 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | +0.02012 | [-0.2225, +0.2628] | -0.2225 / +0.2628 | 1.000 (min 0.333) | not detectable at this budget |
| LSUN Churches A2′−A0 | 2 | -0.3636 | [-0.5666, -0.1605] | -0.5666 / -0.1605 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | -0.3837 | [-0.4233, -0.3441] | -0.3441 / -0.4233 | 0.333 (min 0.333) | CI excludes 0 |
| OASIS-1 T1 A3−A0 (transfer) | 2 | -0.1351 | — | -0.1588 / -0.1115 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | -0.3402 | — | -0.6082 / -0.07227 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | +0.1756 | [-0.07801, +0.5161] | — | 0.500 (min 0.100) | not detectable at this budget |

**octave RMS error (final)** (≈ LSD on 8 bins; lower is better)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | -0.1476 | [-0.1586, -0.1336] | -0.1586 / -0.1506 / -0.1336 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | -0.1389 | [-0.1524, -0.1255] | -0.1255 / -0.1524 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | -0.02748 | [-0.03917, -0.01579] | -0.01579 / -0.03917 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A3−A0 | 3 | -0.3317 | [-0.8268, +0.04599] | -0.8268 / +0.04599 / -0.2142 | 0.300 (min 0.100) | not detectable at this budget |
| LSUN Churches A1−A0 | 2 | -0.4131 | [-0.7914, -0.03482] | -0.7914 / -0.03482 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | -0.03651 | [-0.3519, +0.2789] | -0.3519 / +0.2789 | 1.000 (min 0.333) | not detectable at this budget |
| LSUN Churches A2′−A0 | 2 | -0.4122 | [-0.5837, -0.2407] | -0.5837 / -0.2407 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | -0.3757 | [-0.5196, -0.2318] | -0.2318 / -0.5196 | 0.333 (min 0.333) | CI excludes 0 |
| OASIS-1 T1 A3−A0 (transfer) | 2 | -0.1156 | — | -0.1192 / -0.1120 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | -0.3903 | — | -0.5800 / -0.2006 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | +0.1841 | [-0.1909, +0.6765] | — | 0.700 (min 0.100) | not detectable at this budget |

**octave level ē (final)** (0 = right total variance; negative = samples too alike)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | +0.1409 | [+0.1285, +0.1514] | +0.1514 / +0.1426 / +0.1285 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | +0.1365 | [+0.1309, +0.1421] | +0.1309 / +0.1421 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | +0.02343 | [+0.01148, +0.03538] | +0.01148 / +0.03538 | 0.667 (min 0.333) | CI excludes 0 |
| LSUN Churches A3−A0 | 3 | +0.4319 | [+0.2004, +0.7743] | +0.7743 / +0.2004 / +0.3211 | 0.100 (min 0.100) | CI excludes 0 |
| LSUN Churches A1−A0 | 2 | +0.4944 | [+0.2375, +0.7512] | +0.7512 / +0.2375 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | +0.06469 | [-0.1496, +0.2790] | +0.2790 / -0.1496 | 0.667 (min 0.333) | not detectable at this budget |
| LSUN Churches A2′−A0 | 2 | +0.3428 | [+0.2184, +0.4672] | +0.4672 / +0.2184 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | +0.2781 | [+0.1882, +0.3680] | +0.1882 / +0.3680 | 0.333 (min 0.333) | CI excludes 0 |
| OASIS-1 T1 A3−A0 (transfer) | 2 | +0.1129 | — | +0.1124 / +0.1134 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | +0.5205 | — | +0.6701 / +0.3709 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | -0.2911 | [-0.6305, -0.06247] | — | 0.100 (min 0.100) | CI excludes 0 |

**octave shape sd(e) (final)** (0 = right allocation across scales)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | -0.05716 | [-0.06108, -0.05076] | -0.05965 / -0.06108 / -0.05076 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | -0.04880 | [-0.06408, -0.03352] | -0.03352 / -0.06408 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | -0.01463 | [-0.01721, -0.01205] | -0.01205 / -0.01721 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A3−A0 | 3 | +0.06173 | [-0.3220, +0.3906] | -0.3220 / +0.3906 / +0.1167 | 0.700 (min 0.100) | not detectable at this budget |
| LSUN Churches A1−A0 | 2 | +0.009281 | [-0.2941, +0.3126] | -0.2941 / +0.3126 | 0.667 (min 0.333) | not detectable at this budget |
| LSUN Churches A2−A0 | 2 | +0.04692 | [-0.2271, +0.3210] | -0.2271 / +0.3210 | 0.667 (min 0.333) | not detectable at this budget |
| LSUN Churches A2′−A0 | 2 | -0.2388 | [-0.3750, -0.1026] | -0.3750 / -0.1026 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | -0.2857 | [-0.4236, -0.1478] | -0.1478 / -0.4236 | 0.333 (min 0.333) | CI excludes 0 |
| OASIS-1 T1 A3−A0 (transfer) | 2 | -0.04008 | — | -0.04636 / -0.03381 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | +0.07434 | — | -0.05251 / +0.2012 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | -0.1189 | [-0.4478, +0.2644] | — | 0.700 (min 0.100) | not detectable at this budget |

**octave RMS, 0.5-4 c/img** (the bands a W/8 prior carries)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | -0.2514 | [-0.2642, -0.2346] | -0.2642 / -0.2554 / -0.2346 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | -0.2577 | [-0.2617, -0.2536] | -0.2617 / -0.2536 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | -0.05209 | [-0.06846, -0.03571] | -0.03571 / -0.06846 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A3−A0 | 3 | -0.6209 | [-0.7257, -0.5324] | -0.6047 / -0.7257 / -0.5324 | 0.100 (min 0.100) | CI excludes 0 |
| LSUN Churches A1−A0 | 2 | -0.6644 | [-0.7156, -0.6132] | -0.6132 / -0.7156 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | -0.1700 | [-0.2615, -0.07846] | -0.07846 / -0.2615 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A0 | 2 | -0.1090 | [-0.1308, -0.08709] | -0.08709 / -0.1308 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | +0.06103 | [-0.008636, +0.1307] | -0.008636 / +0.1307 | 0.667 (min 0.333) | not detectable at this budget |
| OASIS-1 T1 A3−A0 (transfer) | 2 | -0.2581 | — | -0.2663 / -0.2499 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | -0.6299 | — | -0.6371 / -0.6227 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | +0.3695 | [+0.2839, +0.4713] | — | 0.100 (min 0.100) | CI excludes 0 |

**octave RMS, 4-96 c/img** (bands no prior carries)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | -0.07856 | [-0.08911, -0.06643] | -0.08911 / -0.08014 / -0.06643 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | -0.06469 | [-0.08303, -0.04635] | -0.04635 / -0.08303 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | -0.005172 | [-0.01285, +0.002505] | +0.002505 / -0.01285 | 1.000 (min 0.333) | not detectable at this budget |
| LSUN Churches A3−A0 | 3 | -0.3186 | [-0.9727, +0.2025] | -0.9727 / +0.2025 / -0.1854 | 0.400 (min 0.100) | not detectable at this budget |
| LSUN Churches A1−A0 | 2 | -0.4138 | [-0.9275, +0.09988] | -0.9275 / +0.09988 | 0.667 (min 0.333) | not detectable at this budget |
| LSUN Churches A2−A0 | 2 | +4.174e-04 | [-0.4481, +0.4489] | -0.4481 / +0.4489 | 1.000 (min 0.333) | not detectable at this budget |
| LSUN Churches A2′−A0 | 2 | -0.5260 | [-0.7612, -0.2908] | -0.7612 / -0.2908 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | -0.5264 | [-0.7397, -0.3131] | -0.3131 / -0.7397 | 0.333 (min 0.333) | CI excludes 0 |
| OASIS-1 T1 A3−A0 (transfer) | 2 | -0.03585 | — | -0.03245 / -0.03925 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | -0.4097 | — | -0.6595 / -0.1600 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | +0.2400 | [-0.2780, +0.8912] | — | 0.700 (min 0.100) | not detectable at this budget |

**log10 variance ratio (final)** (0 = right total variance)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | +0.1247 | [+0.1125, +0.1315] | +0.1315 / +0.1301 / +0.1125 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | +0.1121 | [+0.09688, +0.1273] | +0.09688 / +0.1273 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | +0.009930 | [-0.006133, +0.02599] | -0.006133 / +0.02599 | 1.000 (min 0.333) | not detectable at this budget |
| LSUN Churches A3−A0 | 3 | +0.3467 | [+0.3224, +0.3617] | +0.3561 / +0.3617 / +0.3224 | 0.100 (min 0.100) | CI excludes 0 |
| LSUN Churches A1−A0 | 2 | +0.3570 | [+0.3559, +0.3582] | +0.3582 / +0.3559 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | +0.05985 | [+0.03547, +0.08423] | +0.03547 / +0.08423 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A0 | 2 | +0.06391 | [+0.06078, +0.06703] | +0.06078 / +0.06703 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | +0.004059 | [-0.01720, +0.02532] | +0.02532 / -0.01720 | 1.000 (min 0.333) | not detectable at this budget |
| OASIS-1 T1 A3−A0 (transfer) | 2 | +0.09141 | — | +0.08851 / +0.09431 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | +0.3898 | — | +0.3962 / +0.3834 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | -0.2220 | [-0.2415, -0.1982] | — | 0.100 (min 0.100) | CI excludes 0 |

**precision (Inception, k = 5)** (higher is better)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | +0.08783 | [+0.05500, +0.1370] | +0.1370 / +0.05500 / +0.07150 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | +0.03150 | [+0.01350, +0.04950] | +0.04950 / +0.01350 | 0.667 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | +0.006000 | [+0.002000, +0.01000] | +0.002000 / +0.01000 | 0.667 (min 0.333) | CI excludes 0 |
| LSUN Churches A3−A0 | 3 | +0.08567 | [+0.05400, +0.1235] | +0.05400 / +0.07950 / +0.1235 | 0.100 (min 0.100) | CI excludes 0 |
| LSUN Churches A1−A0 | 2 | +0.08050 | [+0.05300, +0.1080] | +0.1080 / +0.05300 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | +0.05450 | [+0.03050, +0.07850] | +0.07850 / +0.03050 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A0 | 2 | +0.02800 | [+0.009000, +0.04700] | +0.009000 / +0.04700 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | -0.02650 | [-0.06950, +0.01650] | -0.06950 / +0.01650 | 0.667 (min 0.333) | not detectable at this budget |
| OASIS-1 T1 A3−A0 (transfer) | 2 | +0.1522 | — | +0.2035 / +0.1010 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | +0.1088 | — | +0.1170 / +0.1005 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | +0.002167 | [-0.04833, +0.05267] | — | 0.900 (min 0.100) | not detectable at this budget |

**density (Inception, k = 5)** (higher is better)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | +0.1060 | [+0.06970, +0.1505] | +0.1505 / +0.09790 / +0.06970 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | +0.05420 | [+0.05180, +0.05660] | +0.05660 / +0.05180 | 0.667 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | +0.01865 | [-0.002200, +0.03950] | -0.002200 / +0.03950 | 1.000 (min 0.333) | not detectable at this budget |
| LSUN Churches A3−A0 | 3 | +0.03240 | [+0.01470, +0.05060] | +0.01470 / +0.03190 / +0.05060 | 0.100 (min 0.100) | CI excludes 0 |
| LSUN Churches A1−A0 | 2 | +0.02890 | [+0.01670, +0.04110] | +0.04110 / +0.01670 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | +0.02095 | [+0.009700, +0.03220] | +0.03220 / +0.009700 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A0 | 2 | +0.01015 | [+0.002700, +0.01760] | +0.002700 / +0.01760 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | -0.01080 | [-0.02950, +0.007900] | -0.02950 / +0.007900 | 0.667 (min 0.333) | not detectable at this budget |
| OASIS-1 T1 A3−A0 (transfer) | 2 | +0.3130 | — | +0.4001 / +0.2258 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | +0.03655 | — | +0.04020 / +0.03290 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | +0.07363 | [+0.03473, +0.1181] | — | 0.100 (min 0.100) | CI excludes 0 |

**regeneration ratio ρ = (1 − I_w)/(1 − I)** (1 = the chain regenerates what the blur removed)

| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |
|---|---:|---:|---:|---|---:|---|
| IXI T1 A3−A0 | 3 | -0.05791 | [-0.06337, -0.05104] | -0.06337 / -0.05104 / -0.05931 | 0.100 (min 0.100) | CI excludes 0 |
| IXI T1 A1−A0 | 2 | -0.05706 | [-0.06497, -0.04915] | -0.06497 / -0.04915 | 0.333 (min 0.333) | CI excludes 0 |
| IXI T1 A2−A0 | 2 | +0.006004 | [-0.009604, +0.02161] | -0.009604 / +0.02161 | 1.000 (min 0.333) | not detectable at this budget |
| LSUN Churches A3−A0 | 3 | +0.02839 | [+0.01330, +0.04704] | +0.04704 / +0.02485 / +0.01330 | 0.200 (min 0.100) | CI excludes 0 |
| LSUN Churches A1−A0 | 2 | +0.03715 | [+0.02509, +0.04921] | +0.04921 / +0.02509 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2−A0 | 2 | +0.02704 | [+0.02166, +0.03242] | +0.02166 / +0.03242 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A0 | 2 | +0.03830 | [+0.03751, +0.03910] | +0.03751 / +0.03910 | 0.333 (min 0.333) | CI excludes 0 |
| LSUN Churches A2′−A2 | 2 | +0.01126 | [+0.006679, +0.01585] | +0.01585 / +0.006679 | 0.333 (min 0.333) | CI excludes 0 |
| OASIS-1 T1 A3−A0 (transfer) | 2 | -0.1118 | — | -0.1137 / -0.1100 | — | sign only (§8) |
| LSUN Bedrooms A3−A0 (transfer) | 2 | +0.01100 | — | +0.01737 / +0.004634 | — | sign only (§8) |
| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | -0.08630 | [-0.1036, -0.07095] | — | 0.100 (min 0.100) | CI excludes 0 |

## X3. The checkpoint curves (500-seed set, seed means)

| dataset | arm | n | LSD 5k | LSD peak (step) | LSD 60k | var. ratio 5k | var. ratio at peak | var. ratio 60k | level ē 5k | ē at peak | ē 60k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IXI T1 | A0 | 3 | 0.226 | 0.293 (20k) | 0.239 | 0.64 | 0.55 | 0.67 | -0.159 | -0.240 | -0.182 |
| IXI T1 | A3 | 3 | 0.163 | 0.163 (5k) | 0.080 | 0.76 | 0.76 | 0.89 | -0.088 | -0.088 | -0.044 |
| IXI T1 | A1 | 2 | 0.205 | 0.205 (5k) | 0.099 | 0.69 | 0.69 | 0.85 | -0.113 | -0.113 | -0.056 |
| IXI T1 | A2 | 2 | 0.189 | 0.241 (15k) | 0.216 | 0.69 | 0.60 | 0.67 | -0.136 | -0.197 | -0.166 |
| LSUN Churches | A0 | 3 | 0.644 | 1.460 (15k) | 1.199 | 0.60 | 0.26 | 0.29 | -0.464 | -1.277 | -1.049 |
| LSUN Churches | A3 | 3 | 0.743 | 1.158 (20k) | 0.838 | 0.68 | 0.59 | 0.64 | -0.550 | -0.874 | -0.627 |
| LSUN Churches | A1 | 2 | 0.854 | 1.066 (20k) | 0.840 | 0.64 | 0.60 | 0.64 | -0.640 | -0.809 | -0.634 |
| LSUN Churches | A2 | 2 | 1.011 | 1.606 (10k) | 1.224 | 0.46 | 0.28 | 0.32 | -0.801 | -1.369 | -1.056 |
| LSUN Churches | A2′ | 2 | 0.604 | 1.441 (25k) | 0.880 | 0.74 | 0.23 | 0.33 | -0.390 | -1.292 | -0.777 |
| OASIS-1 T1 | A0 | 2 | 0.240 | 0.296 (20k) | 0.234 | 0.60 | 0.55 | 0.66 | -0.196 | -0.240 | -0.194 |
| OASIS-1 T1 | A3 | 2 | 0.184 | 0.184 (5k) | 0.113 | 0.66 | 0.66 | 0.82 | -0.138 | -0.138 | -0.081 |
| LSUN Bedrooms | A0 | 2 | 1.461 | 1.747 (15k) | 1.439 | 0.31 | 0.23 | 0.28 | -1.230 | -1.534 | -1.253 |
| LSUN Bedrooms | A3 | 2 | 1.371 | 1.371 (5k) | 0.977 | 0.62 | 0.62 | 0.69 | -1.029 | -1.029 | -0.717 |

- 360 checkpoint records (every run, every evaluated step). Octave RMS against LSD: r = 0.9994. The level ē² carries a mean 65% (median 71%) of the octave RMS².
- Within runs (each run's mean removed), LSD correlates with |ē| at r = 0.967 and with |log10 variance ratio| at r = 0.763.

Figure: `fx1_dispersion.pdf` / `.png` (top: variance ratio against iteration; bottom: LSD against the octave level |ē| over every checkpoint, with the line LSD = |ē| that a pure broadband deficit would follow).

