# Intrinsic dimension of the four training splits (T8.0)

Regenerate with

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" nice -n 19 env PYTHONPATH=$PWD \
  python -m ihdm.cli.intrinsic_dim --data-root $IHDM_DATA_ROOT --out docs/RESULTS/intrinsic_dimension
```

It takes 23 s on the workstation's CPU (two BLAS threads), one dataset at a time. Two reruns
gave byte-identical `intrinsic_dimension.json`, `id_vs_n.svg`, `id_vs_n.pdf` and `id_vs_n.png`.
Every number below comes from `intrinsic_dimension.json`, which was produced by commit `3e4cdb6`.

## Question

Does registered brain MRI (IXI, OASIS-1) have a lower intrinsic dimension than the photographs
(LSUN Churches, LSUN Bedrooms)? The question matters for the paper because data of lower intrinsic
dimension are easier to learn (Pope et al., ICLR 2021). That is one of the paper's reasons why the
photograph models did not reach usable quality at our budget (`00-framing.md` §2).

## Method

- **Data.** The `train` split of each dataset holds 3,200 images of 192 × 192 grey levels
  (`images.npy`, indices from `splits.json`). We scale each image to [0, 1], remove its mean (DC),
  and then centre across images. Everything is float64.
- **Local (nonlinear) dimension.** We use the maximum-likelihood estimator of Levina & Bickel
  (2004), with the MacKay & Ghahramani (2005) average of the *inverse* local estimates:

  $$\hat m_k = \Big[\tfrac1N\sum_{i=1}^N \tfrac{1}{k-1}\sum_{j=1}^{k-1}\log\tfrac{T_k(x_i)}{T_j(x_i)}\Big]^{-1},$$

  where $T_j(x)$ is the Euclidean distance from $x$ to its $j$-th nearest neighbour, the point
  itself excluded.
  - The distances come from the float64 Gram matrix of the full split.
  - We use k ∈ {5, 10, 20}.
- **Sample-size dependence.** We use N ∈ {320, 1,000, 3,200}.
  - For N < 3,200 we average 10 random subsets and report mean ± sample sd (ddof = 1). The
    subsets come from `numpy.random.default_rng(2026)`, re-created for each dataset, so all four
    datasets use the same positions in their sorted train lists.
  - Each subset uses the corresponding block of the full distance matrix. Euclidean distances do
    not depend on the centring, so the block equals the distances computed on the subset alone
    (this is tested).
  - MRI also gets a "one slice per subject" variant: slice 5 of each of the 320 training subjects
    (N = 320).
- **Linear measures, for contrast.** Both come from the eigenvalues $\lambda$ of the Gram matrix,
  which are the nonzero PCA eigenvalues up to a factor 1/N.
  - The participation ratio is $(\sum\lambda)^2/\sum\lambda^2$.
  - We also count the components that hold 50%, 90% and 95% of the variance.
- **Code:** `ihdm/analysis/intrinsic_dim.py` and `ihdm/cli/intrinsic_dim.py`. The tests in
  `tests/analysis/test_intrinsic_dim.py` check three things:
  - the estimate is within 20% of d for d ∈ {5, 10} on a random linear subspace of
    $\mathbb R^{1000}$ (N = 2,000);
  - the Gram distances equal the direct distances;
  - the participation ratio of a diagonal covariance equals its closed form.

**References.**

- E. Levina, P. J. Bickel. *Maximum likelihood estimation of intrinsic dimension.* NeurIPS 17
  (2004).
- D. J. C. MacKay, Z. Ghahramani. *Comments on "Maximum likelihood estimation of intrinsic
  dimension" by E. Levina and P. Bickel* (2005), inference.org.uk/mackay/dimension.
- P. Pope, C. Zhu, A. Abdelkader, M. Goldblum, T. Goldstein. *The intrinsic dimension of images
  and its impact on learning.* ICLR 2021, arXiv:2104.08894.

## Results

### MLE intrinsic dimension, k = 10 (the headline; figure `id_vs_n`)

| dataset | N = 320 (10 subsets) | N = 1,000 (10 subsets) | N = 3,200 (full split) | change 320 → 3,200 | one slice per subject, N = 320 |
|---|---:|---:|---:|---:|---:|
| IXI | 17.1 ± 0.6 | 17.8 ± 0.3 | **17.2** | +0% | 17.4 |
| OASIS-1 | 18.6 ± 0.5 | 21.1 ± 0.3 | **22.0** | +18% | 21.1 |
| LSUN Churches | 22.5 ± 1.3 | 26.1 ± 2.0 | **30.1** | +34% | — |
| LSUN Bedrooms | 20.8 ± 1.2 | 25.1 ± 1.0 | **30.8** | +49% | — |

### MLE intrinsic dimension, k = 5 and k = 20

| dataset | k = 5, N = 320 | k = 5, N = 1,000 | k = 5, N = 3,200 | k = 5, one slice | k = 20, N = 320 | k = 20, N = 1,000 | k = 20, N = 3,200 | k = 20, one slice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IXI | 17.6 ± 0.5 | 17.6 ± 0.3 | 15.9 | 18.6 | 16.5 ± 0.4 | 17.5 ± 0.2 | 17.9 | 15.9 |
| OASIS-1 | 19.9 ± 0.6 | 22.0 ± 0.5 | 21.4 | 23.4 | 17.2 ± 0.5 | 19.6 ± 0.2 | 21.7 | 19.0 |
| LSUN Churches | 24.1 ± 1.4 | 28.5 ± 2.3 | 32.4 | — | 20.0 ± 1.0 | 23.9 ± 1.4 | 27.6 | — |
| LSUN Bedrooms | 23.4 ± 1.7 | 28.3 ± 1.2 | 34.0 | — | 18.0 ± 0.8 | 22.4 ± 0.6 | 27.1 | — |

The change from N = 320 to 3,200 is −10%, +7%, +34% and +45% at k = 5, and +8%, +26%, +38% and
+51% at k = 20 (IXI, OASIS-1, Churches, Bedrooms).

### Linear (PCA) measures, N = 3,200

| dataset | participation ratio | components for 50% | for 90% | for 95% | rank | `sha256_images` |
|---|---:|---:|---:|---:|---:|---|
| IXI | **42.5** | 22 | 635 | 1,101 | 3,199 | `b666e407e9af…` |
| OASIS-1 | **60.7** | 32 | 815 | 1,366 | 3,199 | `8396e1c113ab…` |
| LSUN Churches | **25.0** | 18 | 859 | 1,482 | 3,199 | `299c076853b0…` |
| LSUN Bedrooms | **29.3** | 14 | 479 | 992 | 3,199 | `7e4c98d15446…` |

The rank is 3,199 because the 3,200 rows are centred.

### Agreement with the scratch measurement of 2026-10-06

All of the following match within ±1, most to the printed digit:

- MLE k = 10, N = 3,200: 17.2 / 22.0 / 30.1 / 30.8, the same as the scratch values.
- MLE k = 20, N = 3,200: 17.9 / 21.7 / 27.6 / 27.1, the same as the scratch values.
- Participation ratio: 42.5 / 60.7 / 25.0 / 29.3, the same as the scratch values.
- Components for 90%: 635 / 815 / 859 / 479, the same as the scratch values.

At N = 320, the scratch used **one** random draw, which gave 16.5 / 19.1 / 21.1 / 19.5 at k = 10
(+4%, +15%, +43%, +58% to N = 3,200). This page reports the mean of ten draws instead, which gives
+0%, +18%, +34% and +49%. The qualitative pattern is the same. If the paper quotes percentages, it
should quote the ones on this page.

## Reading

This is stated as the ticket words it. It is not a hypothesis test. `intrinsic_dimension.json`
→ `reading` checks each statement against the numbers (k = 10), and all three hold.

- **The local (MLE) dimension** at N = 3,200 is lower for MRI than for photographs. The MRI
  estimates also change less from N = 320 to N = 3,200 than the photograph estimates do; this is
  the estimator's known signature of a higher-dimensional set.
- **The linear measures do not show MRI as lower-dimensional.** The participation ratio is higher
  for MRI.
- **The wording the paper can use:** "a lower intrinsic (local) dimension". Do not write "a
  lower-dimensional data set".

The numbers behind each statement (k = 10):

- MLE at N = 3,200: 17.2 (IXI) and 22.0 (OASIS-1) against 30.1 (Churches) and 30.8 (Bedrooms).
- Change from N = 320 to 3,200: +0% and +18% against +34% and +49%. The same order holds at
  k = 5 in absolute value (|−10%|, +7% against +34%, +45%) and at k = 20 (+8%, +26% against
  +38%, +51%).
- Participation ratio: 42.5 and 60.7 against 25.0 and 29.3. The components for 90% of the variance
  are mixed: 635 and 815 against 859 and 479.

The separation needs the full training sets. At N = 320 the estimates overlap: at k = 10,
OASIS-1's one-slice value (21.1) lies above the Bedrooms random-subset mean (20.8 ± 1.2), and across
k ∈ {5, 10, 20} the N = 320 values of all four datasets span 15.9–24.1.

## Caveats

- **The estimator depends on N.** The MLE is biased downward when the sample under-resolves the
  neighbourhoods of a high-dimensional set, and the bias shrinks as N grows. The photograph values
  at N = 3,200 are therefore lower bounds that are still rising; the comparison is valid only at
  equal N and must give N. The estimate also depends on k (for example, Bedrooms reads 34.0 / 30.8 /
  27.1 at k = 5 / 10 / 20). IXI is not monotone in N at k = 5 (17.6 → 15.9).
- **The MRI slices are not independent.** Each subject contributes 10 slices (320 subjects ×
  10). Neighbouring slices of one subject are near neighbours of each other, which can pull the
  MRI estimates down. The one-slice-per-subject variant (N = 320) removes this, but only at a
  sample size where no dataset is separated from the others. The photographs are one image per
  scene.
- **Only 192² grayscale.** Every split is the project's 192 × 192 single-channel preprocessing.
  The photographs are native centre crops converted to grey (`photo_diagnostic/README.md`), and the
  MRI slices are registered to MNI space. The numbers describe these pixel spaces under Euclidean
  distance, not the datasets in general, nor colour or full-scene LSUN.
- The random-subset sd measures subset-to-subset variability only. It is not a confidence interval
  for the dimension.

## Files

- `intrinsic_dimension.json` holds every number above:
  - the per-subset values;
  - the ten leading eigenvalues;
  - provenance: the git SHA, the rng description, the numpy version and each dataset's
    `sha256_images` from `meta.json`;
  - the `reading` checks.
- `id_vs_n.{svg,pdf,png}`: the k = 10 estimate (mean ± sd) against N on a log axis.
  - MRI is drawn with solid lines and the photographs with dashed lines.
  - The hollow markers left of N = 320 are the MRI one-slice-per-subject values.
  - The figure is 5.5 × 2.0 in, with all text at 7 pt.
