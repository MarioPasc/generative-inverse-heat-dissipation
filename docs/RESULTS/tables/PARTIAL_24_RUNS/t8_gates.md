## Table 8 — Plateau gates (D10/D17, D22)

**PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated): not for quoting.**

**Status: complete (4/4 gates)**

| run | checkpoints | LSD early | LSD late | early − late | 95% CI | relative | extend |
|---|---|---:|---:|---:|---:|---:|---|
| ixi_A0_s1 | 35k vs 40k | 0.2671 | 0.2576 | +0.009540 | [+0.007621, +0.01138] | +3.57% | yes |
| ixi_A0_s1 | 55k vs 60k | 0.2476 | 0.2497 | -0.002067 | [-0.003625, -6.988e-04] | -0.83% | no |
| lsun_church_A0_s1 | 35k vs 40k | 1.350 | 1.505 | -0.1551 | [-0.1673, -0.1429] | -11.49% | no |
| lsun_church_A0_s1 | 55k vs 60k | 1.365 | 1.445 | -0.08008 | [-0.09093, -0.07045] | -5.87% | no |

- Paired bootstrap over the 500 frozen evaluation seeds (1,000 resamples, fp16); a run is extended only when the CI of LSD(early) − LSD(late) lies above zero.
- 35k vs 40k (job 2432703) said extend on IXI and not on Churches; all 30 runs were extended to 60k (D22). The informational 55k vs 60k gate (job 2486891) found no further gain on either run, so 60k is the final length.

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 (collection INCOMPLETE, created 2026-09-29T07:38:12.518998+00:00, git 9fb5c84635); tables: git 4adfff670e, 2026-09-29T09:15:36.919074+00:00</sub>
