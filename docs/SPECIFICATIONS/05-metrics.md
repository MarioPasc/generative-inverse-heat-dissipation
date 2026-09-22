# 05 — Metric definitions (frozen; units of the theory)

All images are $192^2$ grayscale in $[0,1]$; $W = 192$; "mode" means a 2-D DCT-II (orthonormal)
coefficient $(i, j)$, $i, j \in \{0, \dots, W-1\}$, with radial index $n = \sqrt{i^2 + j^2}$; a mode
with index $n$ carries $n/2$ **cycles per image**. Octave bins (in cycles per image) are the ones of
every table in the project: $[0.5, 1), [1, 2), [2, 4), [4, 8), [8, 16), [16, 32), [32, 64), [64, 96]$
(the last bin is the corner band up to the maximal radial index).

Reference set = the `ref` split of the same dataset (800 images), never the training split.

## 1. Per-mode variance and the radial profile

For a stack $X \in \mathbb R^{N \times W \times W}$: mean-centre across images
($\bar X = \frac1N \sum_i X_i$, $\tilde X_i = X_i - \bar X$), take the DCT of each $\tilde X_i$, and
define the **per-mode variance** $P(i, j) = \frac1N \sum_k \hat{\tilde X}_k(i, j)^2$ (this is
`mode_power` of `projects/GenAI/analysis/delta_star_from_psd.py`). The DC mode is excluded from
every quantity below. The **radial profile** $\bar P(b)$ is the mean of $P$ over the modes whose
$n/2$ falls in bin $b$; profiles are reported both on the octave bins and on 48 log-spaced bins
between 0.5 and 96 cycles per image (the fine profile is what LSD uses).

## 2. Log-spectral distance (LSD) and $T_\tau$

For a sample stack $S$ (2k samples, or 1k at intermediate checkpoints) and the reference stack $R$:

$$\mathrm{LSD}(S, R) = \sqrt{\frac{1}{B} \sum_{b=1}^{B} \left(\log_{10} \bar P_S(b) - \log_{10} \bar P_R(b)\right)^2}, \quad B = 48 \text{ log-spaced bins.}$$

Report also the per-octave profile $\log_{10} \bar P_S(b) - \log_{10} \bar P_R(b)$ on the 8 octave
bins (sign kept: positive = the samples carry too much variance in that band), and the total
variance ratio $\sum P_S / \sum P_R$.

$T_\tau$ (per dataset, per seed): with $\mathrm{LSD}^{A0}_{\text{final}}$ the LSD of the A0 run
with the same seed at its last checkpoint, $T_\tau(\text{arm})$ is the smallest checkpoint step $s$
at which $\mathrm{LSD}_{\text{arm}}(s) \le \mathrm{LSD}^{A0}_{\text{final}}$; $+\infty$ (reported
as "not reached") if never. Final LSD = LSD at the last checkpoint.

Samples for LSD start from **training** seeds (Alg. 2), one sample per seed, `prior_noise` as in
the run's config.

## 3. Within-seed diversity

For each of the 40 held-out seed images $x_s$ (one slice per seed subject, index 5), draw 50
samples $y_{s,1..50}$ with the same prior state (the seed blurred to level $K$, plus prior noise if
enabled) and independent sampling noise. Diversity in pixel space:

$$D_{\text{pix}}(s) = \frac{1}{50} \sum_{m=1}^{50} \|y_{s,m} - \bar y_s\|_2^2 \Big/ (W^2 - 1), \quad \bar y_s = \tfrac1{50}\textstyle\sum_m y_{s,m},$$

with the DC (image mean) removed from every $y$ first (non-DC variance per pixel). Low-pass
diversity $D_{\text{lp}}(s)$: the same after blurring every $y$ with the DCT heat kernel at
$\sigma_B = 16$ px. Report the mean over the 40 seeds and the per-seed values.

## 4. Memorisation ratio $M$

For a set $Y$ of samples (the 40 × 50 diversity samples) and the training split $X_{\text{train}}$
(3200 images), with $d(y) = \min_{x \in X_{\text{train}}} \|y - x\|_2$ (pixel space, DC removed) and
the same distance $d(r)$ for held-out real images $r \in X_{\text{ref}} \setminus \text{seeds}$:

$$M = \frac{\operatorname{median}_{y \in Y} d(y)}{\operatorname{median}_{r} d(r)}.$$

$M = 1$: as far from the training set as a new real image; $M < 1$: closer than real held-out
images (copying). Also $M_{\text{lp}}$ after the $\sigma_B = 16$ low-pass on both numerator and
denominator, and the fraction of samples whose nearest training neighbour is the seed's own subject
(MRI) or the seed image (photographs). Computed with `torch.cdist` in chunks on the GPU or CPU.

## 5. Inherited band (mechanism check, eq. (13.3) of learning/03)

For each seed $s$ with samples $y_{s,m}$: per-mode variance of the samples about their seed,
$V_s(i,j) = \frac1{50}\sum_m (\hat y_{s,m}(i,j) - \hat x_s(i,j))^2$, averaged over seeds and
normalised by the population per-mode variance $P_{\text{ref}}(i,j)$; plotted (radial profile)
against the prediction line $d_{K}^2(n) = (1 - e^{-\lambda_n t_K})^2$ with $\lambda_n$ the DCT
Laplacian eigenvalue of the mode and $t_K = \sigma_{B,\max}^2/2$. A straight line through the
origin means the model inherits exactly the modes the prior hands it. Report the inherited share:
$\sum_{n} P_{\text{ref}}(n)\, e^{-2\lambda_n t_K} / \sum_n P_{\text{ref}}(n)$ (prediction) against
the measured $1 - \sum V / \sum P_{\text{ref}}$ restricted to the low band.

## 6. PCA around the seed (figure only)

PCA fitted on the training split (DC removed, 3200 × 36864 → the first 2 components via
randomised SVD); the seed and its 50 samples are projected; one panel per dataset and arm for two
seeds, arrows from the seed to the sample centroid.

## 7. FID, KID, recall, coverage (final checkpoint only; D6′)

`clean-fid` FID and KID between 5k samples (training seeds, 1 per seed, drawn with replacement
over the 3200 training images with distinct sampling noise) and the 800 reference images, both
grayscale replicated to 3 channels and resized by clean-fid's own pipeline; `prdc` recall and
coverage on the same Inception features with $k = 5$. Reported with a bootstrap CI over samples.
Never compared across datasets; compared across arms within a dataset only.

## 8. Statistics (T4.3 / T6.1)

Cell = (dataset, arm), $s$ seeds. For a metric $m$: $\Delta^{(a)}_{\mathcal D} = m(a, \mathcal D)
- m(\text{A0}, \mathcal D)$ with seeds paired (seed 1 with seed 1). Interaction estimand:
$\Delta_{\text{MRI}} - \Delta_{\text{photo}}$. Bootstrap CI: resample seeds with replacement within
each cell (10,000 draws; percentile interval at 95%); for LSD also resample samples. Permutation
test: permute arm labels across the pooled seeds of the two cells (all $\binom{n}{s}$ assignments
when $n \le 6$, else 10,000 random), two-sided $p$. A CI containing zero is reported as "not
detectable at this budget". Transfer: sign agreement of $\Delta^{(3)}$ between IXI and OASIS-1 and
between Churches and Bedrooms, per metric.

## 9. Result files

`<run_root>/<run_id>/metrics/ckpt_<step>.json` with keys `lsd`, `lsd_octaves`, `variance_ratio`,
`n_samples`; `<run_id>/metrics/final.json` adding `diversity_pix`, `diversity_lp`, `M`, `M_lp`,
`seed_nn_fraction`, `inherited_measured`, `inherited_predicted`, `fid`, `kid`, `recall`,
`coverage`, each with its CI where defined; plus `metrics/summary.json` merging all checkpoints.
All JSON is written with `ihdm.metrics.io.write_json` (sorted keys, floats rounded to 6 digits).
