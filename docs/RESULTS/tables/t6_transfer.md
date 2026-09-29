## Table 6 — Transfer: the sign of A3−A0 on the development and the transfer dataset

**Status: complete (20/20 runs)**

| development → transfer | endpoint | Δ dev (3 seeds) | Δ dev / A0 | Δ transfer (2 seeds) | transfer per seed (s1 / s2) | Δ transfer / A0 | same sign |
|---|---|---:|---:|---:|---|---:|---|
| IXI → OASIS-1 | LSD (final, 2k set) | -0.1640 | -67.7% | -0.1191 | -0.1233 / -0.1148 | -51.6% | yes |
| IXI → OASIS-1 | T_τ (steps) | -10k | -66.7% | -50k | -55k / -45k | -90.9% | yes |
| IXI → OASIS-1 | KID (headline) | -0.02925 | -63.3% | -0.01148 | -0.01397 / -0.008997 | -48.8% | yes |
| IXI → OASIS-1 | FID | -28.68 | -50.7% | -13.72 | -15.87 / -11.57 | -38.0% | yes |
| IXI → OASIS-1 | recall | +0.1250 | — | +0.06937 | +0.008750 / +0.1300 | — | yes |
| IXI → OASIS-1 | coverage | +0.3079 | — | +0.2062 | +0.2400 / +0.1725 | — | yes |
| IXI → OASIS-1 | M | -0.03638 | -4.0% | -0.08684 | -0.1035 / -0.07022 | -8.7% | yes |
| IXI → OASIS-1 | M_lp | -0.7907 | -77.1% | -0.8119 | -0.8162 / -0.8077 | -72.2% | yes |
| IXI → OASIS-1 | seed-NN fraction | +0.6588 | — | +0.5387 | +0.5275 / +0.5500 | — | yes |
| IXI → OASIS-1 | D_pix | -0.002647 | -23.2% | -0.003385 | -0.003425 / -0.003345 | -28.0% | yes |
| IXI → OASIS-1 | D_lp | -2.646e-04 | -94.5% | -2.003e-04 | -1.861e-04 / -2.145e-04 | -92.4% | yes |
| IXI → OASIS-1 | inherited share as pre-registered (biased under a non-zero mean image; see inherited_band_audit.md) | +0.9224 | — | +0.5461 | +0.5291 / +0.5631 | — | yes |
| IXI → OASIS-1 | within-seed share I_w (D23) | +0.07938 | — | +0.1254 | +0.1269 / +0.1239 | — | yes |
| IXI → OASIS-1 | seed-mean bias fraction G_b (D23) | -0.8446 | — | -0.4232 | -0.4047 / -0.4416 | — | yes |
| IXI → OASIS-1 | T_τ, 2k-set threshold (sensitivity) | 0 | +0.0% | — | — | — | not computable: T_τ not reached (A0 s1) |
| Churches → Bedrooms | LSD (final, 2k set) | -0.3707 | -30.9% | -0.4447 | -0.6353 / -0.2541 | -31.1% | yes |
| Churches → Bedrooms | T_τ (steps) | 0 | +0.0% | 0 | 0 / 0 | +0.0% | yes |
| Churches → Bedrooms | KID (headline) | -0.1493 | -47.9% | -0.1023 | -0.07418 / -0.1305 | -43.1% | yes |
| Churches → Bedrooms | FID | -89.08 | -34.6% | -90.04 | -81.01 / -99.08 | -37.5% | yes |
| Churches → Bedrooms | recall | +0.09125 | — | +0.04375 | +0.03375 / +0.05375 | — | yes |
| Churches → Bedrooms | coverage | +0.1038 | — | +0.08438 | +0.08125 / +0.08750 | — | yes |
| Churches → Bedrooms | M | +0.1333 | +24.5% | +0.1498 | +0.1551 / +0.1445 | +30.7% | yes |
| Churches → Bedrooms | M_lp | -0.5775 | -83.6% | -0.5493 | -0.5458 / -0.5528 | -84.2% | yes |
| Churches → Bedrooms | seed-NN fraction | +0.6138 | — | +0.7490 | +0.7490 / +0.7490 | — | yes |
| Churches → Bedrooms | D_pix | +1.299e-04 | +4.1% | -4.556e-04 | -1.970e-04 / -7.141e-04 | -15.9% | no |
| Churches → Bedrooms | D_lp | -8.248e-04 | -96.6% | -9.812e-04 | -9.913e-04 / -9.711e-04 | -97.2% | yes |
| Churches → Bedrooms | inherited share as pre-registered (biased under a non-zero mean image; see inherited_band_audit.md) | +0.04480 | — | -0.01252 | -0.01819 / -0.006849 | — | no |
| Churches → Bedrooms | within-seed share I_w (D23) | -0.002623 | — | +0.009925 | +0.004292 / +0.01556 | — | no |
| Churches → Bedrooms | seed-mean bias fraction G_b (D23) | -0.04737 | — | +0.02225 | +0.02240 / +0.02210 | — | no |
| Churches → Bedrooms | T_τ, 2k-set threshold (sensitivity) | 0 | +0.0% | 0 | 0 / 0 | +0.0% | yes |

- Signs and magnitudes only, no p-values (05 §8): a transfer cell has 2 seeds. 'Δ / A0' divides the mean Δ by the mean A0 value over the same seeds, so the magnitude can be read across datasets whose metric scales differ; it is left blank for proportions near 0 and the signed inherited share.
- The development Δ uses the 3 seeds of IXI or Churches; the transfer Δ the 2 seeds of OASIS-1 or Bedrooms. The A3 schedule is the IXI-matched one on every dataset, transferred frozen (D12).
- Inherited band (D23, inherited_band_audit.md): the pre-registered share 1 − ΣV/ΣP_ref is biased under a non-zero mean image, its model expectation is I − T with T = Σ(1−d)²μ²/ΣP_ref. I_w = 1 − M/(M−1)·D_pix(W²−1)/ΣP_ref is the within-seed share (expectation I whatever the mean image) and G_b = 1 − pre-registered share − D_pix(W²−1)/ΣP_ref the seed-mean bias fraction (expectation T + (1−I)/M). ΣP_ref and T come from the ref split (inherited_band_constants.json).

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results (collection COMPLETE, created 2026-09-29T11:57:08.292774+00:00, git cff6585925); tables: git 4c8c81b19a, 2026-09-29T11:57:51.909550+00:00</sub>
