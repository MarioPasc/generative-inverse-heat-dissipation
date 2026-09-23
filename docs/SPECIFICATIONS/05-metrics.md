# 05 — Metric definitions (frozen; units of the theory)

All images are $192^2$ grayscale in $[0,1]$; $W = 192$; "mode" means a 2-D DCT-II (orthonormal)
coefficient $(i, j)$, $i, j \in \{0, \dots, W-1\}$, with radial index $n = \sqrt{i^2 + j^2}$; a mode
with index $n$ carries $n/2$ **cycles per image**. Octave bins (in cycles per image) are the ones of
every table in the project: $[0.5, 1), [1, 2), [2, 4), [4, 8), [8, 16), [16, 32), [32, 64), [64, 96]$;
modes above 96 cycles per image (the corner modes, up to 135) are excluded, as in every table of
the project.

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

Computed on **training-seeded** samples (the copying the design worries about is copying of the
training seeds handed over by the prior; a held-out-seeded sample can never be closer to the
training set than a new real image, so $M$ on the §3 set would be ≈ 1 by construction). $Y$ = the
5k final-checkpoint samples of §7 (one per training seed, seeds drawn with replacement). With
$d(y) = \min_{x \in X_{\text{train}}} \|y - x\|_2$ over the **full** training split including the
sample's own seed (pixel space, DC removed), and the same distance $d(r)$ for held-out real
images $r \in X_{\text{ref}} \setminus \text{seed subjects}$:

$$M = \frac{\operatorname{median}_{y \in Y} d(y)}{\operatorname{median}_{r} d(r)}.$$

$M = 1$: as far from the training set as a new real image; $M < 1$: closer than real held-out
images (copying). Also $M_{\text{lp}}$ after the $\sigma_B = 16$ low-pass on both numerator and
denominator, and `seed_nn_fraction`: the fraction of samples whose nearest training image is its
own seed (MRI: any slice of the seed's subject; photographs: the seed image). Computed with
`torch.cdist` in chunks on the GPU or CPU. The held-out-seeded 40 × 50 set is used for §3 and §5
only.

## 5. Inherited band (mechanism check, eq. (13.3) of learning/03)

For each seed $s$ with samples $y_{s,m}$: per-mode variance of the samples **about their prior
state** $d_K(i,j)\,\hat x_s(i,j)$ with $d_K = e^{-\lambda t_K}$ (the released `DCTBlur` kernel at
$\sigma_{B,\max}$), $V_s(i,j) = \frac1{50}\sum_m (\hat y_{s,m}(i,j) - d_K(i,j)\hat x_s(i,j))^2$
(*amended 2026-09-23, T4.1: the residual about the seed itself has expectation $2(1-d_K)P$ under
the linear-Gaussian model, not $(1-d_K^2)P$, so it is inconsistent with the prediction line
below*), averaged over seeds and normalised by the population per-mode variance
$P_{\text{ref}}(i,j)$; plotted (radial profile)
against the **linear-Gaussian prediction** $1 - d_K^2(n) = 1 - e^{-2\lambda_n t_K}$ with
$\lambda_n$ the DCT Laplacian eigenvalue of the mode and $t_K = \sigma_{B,\max}^2/2$: the chain
regenerates the removed part of each mode, whose variance is $(1 - d_K^2) P$, while the surviving
part $d_K x_{\text{seed}}$ is constant across the 50 samples and contributes no variance (at
$d_K = 0.5$ the prediction is $0.75$). The measured curve lying on the line means the model
inherits exactly the modes the prior hands it; curvature is the finding. Report the inherited
share: $\sum_{n} P_{\text{ref}}(n)\, e^{-2\lambda_n t_K} / \sum_n P_{\text{ref}}(n)$ (prediction)
against the measured $1 - \sum V / \sum P_{\text{ref}}$, **both over all non-DC modes**; the
low-band variant (modes with $\sigma_n = \sqrt{2/\lambda_n} \ge 8$ px, i.e. ≤ 5.4 cycles per
image at $W = 192$) masks both sides and is reported beside it (T4.1). LSD note: of the 48
log-spaced bins, five hold no mode of the $192^2$ grid; the LSD is the RMS over the 43 populated
bins, a fixed functional of $W$ and the bin count.

## 6. PCA around the seed (figure only)

PCA fitted on the training split (DC removed, 3200 × 36864 → the first 2 components via
randomised SVD); the seed and its 50 samples are projected; one panel per dataset and arm for two
seeds, arrows from the seed to the sample centroid.

## 7. FID, KID, recall, coverage (final checkpoint only; D6′)

`clean-fid` FID and KID between 5k samples (training seeds, 1 per seed, drawn with replacement
over the 3200 training images with distinct sampling noise) and the 800 reference images, both
grayscale replicated to 3 channels and resized by clean-fid's own pipeline; `prdc` recall and
coverage on the same Inception features with $k = 5$. Reported with a bootstrap CI over samples.
Never compared across datasets; compared across arms within a dataset only. **FID against an
800-image reference is biased upward (bias ∝ $1/N_{\text{ref}}$)**; the bias cancels in
differences between arms because the reference is identical for all arms, but absolute values are
not comparable to the paper's: **KID (unbiased) is the headline of the two**, every FID is printed
with its $N_{\text{ref}}$, and no absolute FID is quoted against the paper without that caveat.
The same 5k training-seeded samples are the set on which §4's $M$ is computed.

## 8. Statistics (T4.3 / T6.1)

Cell = (dataset, arm), $s$ seeds. For a metric $m$: $\Delta^{(a)}_{\mathcal D} = m(a, \mathcal D)
- m(\text{A0}, \mathcal D)$ with seeds paired (seed 1 with seed 1). Interaction estimand:
$\Delta_{\text{MRI}} - \Delta_{\text{photo}}$. Bootstrap CI: resample seeds with replacement within
each cell (10,000 draws; percentile interval at 95%); for LSD also resample samples. Permutation
test: permute arm labels across the pooled seeds of the two cells (all $\binom{n}{s}$ assignments
when $n \le 6$, else 10,000 random), two-sided $p$. A CI containing zero is reported as "not
detectable at this budget". Transfer: sign agreement of $\Delta^{(3)}$ between IXI and OASIS-1 and
between Churches and Bedrooms, per metric.

## 8a. Sampling budget (D16, 2026-09-23; supersedes the per-checkpoint counts above)

Measured on the A100: 3.2 s per 200-step chain per image at sampling batch 32–64. Per run:

| set | seeds | samples | chains | used for |
|---|---|---|---|---|
| intermediate LSD | training, 500 distinct, `rng_seed` = checkpoint step | 1 per seed | 8 × 500 = 4,000 | LSD at 5k, 10k, …, 40k; $T_\tau$ at 5k resolution; the plateau gate (35k vs 40k) |
| final shared | training, 2,000 with replacement, `rng_seed` 0 | 1 per seed | 2,000 | final LSD, KID (headline) and FID, recall/coverage, $M$ and $M_{\text{lp}}$, `seed_nn_fraction` |
| held-out | the 40 seed subjects (slice 5) | 50 per seed | 2,000 | within-seed diversity, inherited band, PCA figure |

Total 8,000 chains ≈ 7.1 A100-hours per run (≈ 214 A100-h for 30 runs) against 669 A100-h for the
plan as first written. Sampling batch 32 (18 GB) or 64 (36 GB); the batch is recorded in
`request.json` and pinned per run because samples are reproducible only at a fixed batch.

**Common random numbers (D17, reviewer, 2026-09-23).** The 500 intermediate seeds are ONE fixed
list per dataset (500 distinct training indices drawn once with `rng 2026` and written to
`$IHDM_DATA_ROOT/<id>/eval_seeds_500.npy` by `evaluate_run` on first use), reused at every
checkpoint of every run of that dataset; the sampling noise uses the same `rng_seed = 2026` and
the same sampling batch (32) everywhere, so noise streams are identical across checkpoints and
arms. Consequences: (a) the plateau gate is a **paired** test: the per-sample log-spectral
distance (each sample's 48-bin radial log profile against the reference profile, RMS over bins)
is computed for the same 500 seeds at 35k and 40k, and the run is extended only if the bootstrap
95% CI over seeds of the paired improvement $\mathrm{LSD}_{35k} - \mathrm{LSD}_{40k}$ excludes
zero; (b) $T_\tau$ and the A3-versus-A0 LSD contrast are paired at the sample level as well as at
the training-seed level. The 2k final set keeps "with replacement over the training split" with
`rng_seed 0`.

## 9. Result files

`<run_root>/<run_id>/metrics/ckpt_<step>.json` with keys `lsd`, `lsd_octaves`, `variance_ratio`,
`n_samples`; `<run_id>/metrics/final.json` adding `diversity_pix`, `diversity_lp`, `M`, `M_lp`,
`seed_nn_fraction`, `inherited_measured`, `inherited_predicted`, `fid`, `kid`, `recall`,
`coverage`, each with its CI where defined; plus `metrics/summary.json` merging all checkpoints.
All JSON is written with `ihdm.metrics.io.write_json` (sorted keys, floats rounded to 6 digits).
