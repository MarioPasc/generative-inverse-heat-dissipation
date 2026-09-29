## Table 6 — Transfer: the sign of A3−A0 on the development and the transfer dataset

**PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated): not for quoting.**

**Status: incomplete: 14/20 runs** (missing: oasis1_A3_s1, oasis1_A3_s2, lsun_bedroom_A0_s1, lsun_bedroom_A0_s2, lsun_bedroom_A3_s1, lsun_bedroom_A3_s2)

| development → transfer | endpoint | Δ dev (3 seeds) | Δ dev / A0 | Δ transfer (2 seeds) | transfer per seed (s1 / s2) | Δ transfer / A0 | same sign |
|---|---|---:|---:|---:|---|---:|---|
| IXI → OASIS-1 | LSD (final, 2k set) | -0.1640 | -67.7% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | T_τ (steps) | -10k | -66.7% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | KID (headline) | -0.02925 | -63.3% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | FID | -28.68 | -50.7% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | recall | +0.1250 | — | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | coverage | +0.3079 | — | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | M | -0.03638 | -4.0% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | M_lp | -0.7907 | -77.1% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | seed-NN fraction | +0.6588 | — | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | D_pix | -0.002647 | -23.2% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | D_lp | -2.646e-04 | -94.5% | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | inherited share as pre-registered (biased under a non-zero mean image; see inherited_band_audit.md) | +0.9224 | — | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | within-seed share I_w (D23) | +0.07938 | — | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | seed-mean bias fraction G_b (D23) | -0.8446 | — | — | — | — | incomplete: 2/4 runs |
| IXI → OASIS-1 | T_τ, 2k-set threshold (sensitivity) | 0 | +0.0% | — | — | — | incomplete: 2/4 runs |
| Churches → Bedrooms | LSD (final, 2k set) | -0.3707 | -30.9% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | T_τ (steps) | 0 | +0.0% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | KID (headline) | -0.1493 | -47.9% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | FID | -89.08 | -34.6% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | recall | +0.09125 | — | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | coverage | +0.1038 | — | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | M | +0.1333 | +24.5% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | M_lp | -0.5775 | -83.6% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | seed-NN fraction | +0.6138 | — | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | D_pix | +1.299e-04 | +4.1% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | D_lp | -8.248e-04 | -96.6% | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | inherited share as pre-registered (biased under a non-zero mean image; see inherited_band_audit.md) | +0.04480 | — | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | within-seed share I_w (D23) | -0.002623 | — | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | seed-mean bias fraction G_b (D23) | -0.04737 | — | — | — | — | incomplete: 0/4 runs |
| Churches → Bedrooms | T_τ, 2k-set threshold (sensitivity) | 0 | +0.0% | — | — | — | incomplete: 0/4 runs |

- Signs and magnitudes only, no p-values (05 §8): a transfer cell has 2 seeds. 'Δ / A0' divides the mean Δ by the mean A0 value over the same seeds, so the magnitude can be read across datasets whose metric scales differ; it is left blank for proportions near 0 and the signed inherited share.
- The development Δ uses the 3 seeds of IXI or Churches; the transfer Δ the 2 seeds of OASIS-1 or Bedrooms. The A3 schedule is the IXI-matched one on every dataset, transferred frozen (D12).
- Inherited band (D23, inherited_band_audit.md): the pre-registered share 1 − ΣV/ΣP_ref is biased under a non-zero mean image, its model expectation is I − T with T = Σ(1−d)²μ²/ΣP_ref. I_w = 1 − M/(M−1)·D_pix(W²−1)/ΣP_ref is the within-seed share (expectation I whatever the mean image) and G_b = 1 − pre-registered share − D_pix(W²−1)/ΣP_ref the seed-mean bias fraction (expectation T + (1−I)/M). ΣP_ref and T come from the ref split (inherited_band_constants.json).

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 (collection INCOMPLETE, created 2026-09-29T07:38:12.518998+00:00, git 9fb5c84635); tables: git 4adfff670e, 2026-09-29T09:15:36.919074+00:00</sub>
