# Audit of the inherited-band endpoint (T6.3)

**Question.** T6.1 reports Churches' measured inherited share on A0 as 0.76 against a predicted
0.018. T6.2 reports the MRI radial ratio at 0.46–1.11 above 3 cycles per image with spikes up to
18 in the lowest bins, Churches A0 at 0.00–0.13 against a prediction of about 1, and
`inherited_measured` negative on every σ_B,max = 96 MRI run. A share should lie in [0, 1], and
measured and predicted should roughly agree where the theory holds. Which link is wrong: the
metric code, the `final.json` fields, the `index.csv` copy, the readers, or the interpretation?

**Verdict.** No link of the pipeline has a bug. `evaluate_run` computes exactly the estimator the
specification defines, and `index.csv`, the tables and the figure read it faithfully. The
estimator itself is mis-specified. It measures the residual of the samples about the prior state
$d_K\hat x_s$. That residual equals the linear-Gaussian prediction only when (a) the population
mean image has no non-DC content and (b) the model regenerates the population variance the blur
removed. On MRI, (a) fails: the mean image of the registered brains carries 1.5 × (IXI) and 0.94 ×
(OASIS-1) the non-DC population variance. A perfect model would then read $I - T$ with
$T = 1.40$ (IXI, σ = 96), which explains the negative values and the low-bin spikes mode by mode.
On Churches, (b) fails: the samples vary within a seed by only 5–8 % of the population variance
against a predicted 96 %. The 0.76 is therefore regeneration that did not happen, not inheritance.
T6.1 and T6.2 describe the same numbers from two sides, and they do not contradict each other.
T6.1's reading that "the prediction fits IXI A3/A1", T6.2's caption sentence "below it, the
samples inherit more of the seed", and T4.1's attribution of the pilot's −1.75 to undertraining
are misreadings. The recommended fix (accepted by `main` as D23) touches only the reading. It
needs two reference-side numbers per (dataset, σ_B,max), computed from the `ref` split, and no
sample from the evaluation tars.

Status of claims: VERIFIED = checked numerically on the files named; DERIVED = follows from the
stated model; LIKELY = interpretation. All numbers are on intensities in [0, 1] (uint8 / 255),
orthonormal DCT-II, per-image DC removed, W = 192.

---

## 1. The definition, quoted

`docs/SPECIFICATIONS/05-metrics.md` §5 (as amended on 2026-09-23 by D-T4.1-1):

> For each seed $s$ with samples $y_{s,m}$: per-mode variance of the samples **about their prior
> state** $d_K(i,j)\,\hat x_s(i,j)$ with $d_K = e^{-\lambda t_K}$ (the released `DCTBlur` kernel
> at $\sigma_{B,\max}$), $V_s(i,j) = \frac1{50}\sum_m (\hat y_{s,m}(i,j) - d_K(i,j)\hat x_s(i,j))^2$
> […], averaged over seeds and normalised by the population per-mode variance
> $P_{\text{ref}}(i,j)$; plotted (radial profile) against the **linear-Gaussian prediction**
> $1 - d_K^2(n) = 1 - e^{-2\lambda_n t_K}$ with $\lambda_n$ the DCT Laplacian eigenvalue of the
> mode and $t_K = \sigma_{B,\max}^2/2$: the chain regenerates the removed part of each mode, whose
> variance is $(1 - d_K^2) P$, while the surviving part $d_K x_{\text{seed}}$ is constant across
> the 50 samples and contributes no variance […]. Report the inherited share:
> $\sum_{n} P_{\text{ref}}(n)\, e^{-2\lambda_n t_K} / \sum_n P_{\text{ref}}(n)$ (prediction)
> against the measured $1 - \sum V / \sum P_{\text{ref}}$, **both over all non-DC modes**; the
> low-band variant (modes with $\sigma_n = \sqrt{2/\lambda_n} \ge 8$ px […]) masks both sides.

The prediction is eq. (13.3) of `learning/03-terminal-blur-scaffolding.md`:
$I(\sigma_{B,\max}) = \sum_{i\neq0} d_{K,i}^2 P_i / \sum_{i\neq0} P_i$, "the expected share of
the dataset's between-image variance that a sample inherits from one training image".

D-T4.1-1 (T4.1 log §2) moved the residual from the raw seed to $d_K\hat x_s$, because the raw-seed
residual has expectation $2(1-d)P$. That derivation, and every synthetic test of
`tests/metrics/test_spectral.py`, uses **zero-mean** fields (T4.1 log, D-T4.1-5: "the synthetic
fields of the tests are zero-mean").

## 2. What each field is

| where | field | formula (non-DC modes; $d = e^{-\lambda\sigma_{\max}^2/2}$) | code |
|---|---|---|---|
| `final.json` | `inherited_measured` | $1 - \sum V/\sum P_{\text{ref}}$, $V = \operatorname{mean}_{s,m}(\hat y_{sm} - d\hat x_s)^2$ | `spectral.inherited_band` → `run_eval._inherited_record` |
| `final.json` | `inherited_predicted` | $I = \sum d^2 P_{\text{ref}}/\sum P_{\text{ref}}$ | `spectral.power.inherited_share` |
| `final.json` | `inherited_{measured,predicted}_low_band` | the same two sums restricted to $\lambda \le 2/8^2$ ($n \le 10.8$, ≤ 5.4 c/img) | same, `low_band_sigma_px = 8` |
| `final.json` | `radial.measured` | unweighted mean of $V(i,j)/P_{\text{ref}}(i,j)$ over the modes of each populated log bin (43 of 48) | `radial_profile(ratio, log_bin_edges())`, nan bins dropped by `_finite_curve` |
| `final.json` | `radial.predicted` | unweighted bin mean of $1 - d^2$ | same |
| `final.json` | `diversity_pix` | $D = \operatorname{mean}_s \frac{1}{M(W^2-1)}\sum_m\lVert y_{sm} - \bar y_s\rVert^2$ | `diversity.within_seed_diversity` |
| `index.csv` | `inherited_measured`, `inherited_predicted`, `D_pix` | copies of the three `final.json` fields | `collect.py` l. 838–839 |
| tables | `inherited_measured`, `inherited_predicted`, `inherited_*_low` | from `index.csv`, cross-checked against `summary.json`; low band from `summary.json` | `tables._SUMMARY_KEYS`, `_SUMMARY_ONLY` |
| figure 5 | solid / dashed curves | `radial.measured` (mean over seeds) / `radial.predicted` | `figures._band_panel` |

$V$ and $P_{\text{ref}}$ are both in [0, 1] units (`_DcRemoved` divides uint8 by 255;
`run_eval` passes `mode_power(reference / 255)`). $P_{\text{ref}}$ is the variance about the
population mean (`mode_power` centres across images), so it does **not** contain the mean image
$\mu$; $V$ is a second moment about $d\hat x_s$ and does.

## 3. Evidence

### 3.1 The plumbing is faithful (VERIFIED)

- `index.csv` equals `final.json` for `inherited_measured` and `inherited_predicted` on all 24
  evaluated runs of `_results_partial_24` (|Δ| < 1e-5, the 6-digit rounding of `write_json`).
- Rerunning `inherited_band` and `within_seed_diversity` on the pilot held-out samples
  (`_runs_local/pilot_ixi_A0_s1/samples/000750/heldout`, 4 × 10) against the IXI `ref` split
  reproduces that run's `final.json`: share −1.753442 against −1.75344, $D_{\text{pix}}$
  0.00394851 against 0.00394851.
- The IXI, OASIS-1 and Churches `ref` spectra, recomputed locally, reproduce `inherited_predicted`
  and the low-band prediction of every run to all printed digits (IXI 0.00343 / 0.07954, low
  0.01076 / 0.24925; Churches 0.01791 / 0.28862, low 0.02699 / 0.43488). The local datasets'
  `sha256_images` equal the `dataset_sha256` of the runs' `summary.json`.
- The readers use the right fields under the right names; the sign convention ($1 - \sum V/\sum P$)
  is the same in the spec, the code, `final.json`, `index.csv`, the tables and the figure.

### 3.2 The mean-image term (DERIVED, then VERIFIED on the reference spectra)

Write each mode of the population as $x = \mu + \xi$ with $\operatorname{Var}\xi = P$. The blur
damps the mean as much as the fluctuation: the prior state is $d\mu + d\xi$. A model whose samples
have the population's law and correlation $d$ with the seed, i.e. the linear-Gaussian model of §5
written without assuming $\mu = 0$, draws
$$y = \mu + d(x - \mu) + \varepsilon,\qquad \operatorname{Var}\varepsilon = (1-d^2)P .$$
Then $y - d x = (1-d)\mu + \varepsilon$ and
$$\mathbb E\,V = (1-d)^2\mu^2 + (1-d^2)P,\qquad
\mathbb E\Big[1-\frac{\sum V}{\sum P}\Big] = I - T,\qquad
T = \frac{\sum_{i\neq0}(1-d_i)^2\mu_i^2}{\sum_{i\neq0}P_i}.$$
The chain has to add $(1-d)\mu$ to every sample of every seed: that is the restoration of the
population's mean structure, the same for all seeds, carrying no variance and no inheritance. The
pre-registered estimator counts it as regenerated variance. Its prediction line and $I$ omit it.
The measured ratio of a perfect model is $(1-d^2) + (1-d)^2\mu^2/P$, which exceeds 1 wherever
$\mu^2/P > d^2/(1-d)^2$.

Reference constants, from each dataset's 800-image `ref` split (VERIFIED against `final.json`
where both exist):

| dataset | $\sum P_{\text{ref}}$ | $\sum\mu^2$ | $I$ (96) | $T$ (96) | $I - T$ (96) | $I$ (24) | $T$ (24) | $I - T$ (24) | $T_{\text{low}}$ (96 / 24) |
|---|---|---|---|---|---|---|---|---|---|
| IXI | 1254.13 | 1896.80 | 0.00343 | **1.4015** | −1.398 | 0.07954 | 0.2113 | −0.132 | 4.349 / 0.619 |
| OASIS-1 | 1015.14 | 955.65 | 0.00167 | **0.8785** | −0.877 | 0.04260 | 0.1917 | −0.149 | 3.326 / 0.643 |
| Churches | 1862.91 | 221.51 | 0.01791 | 0.0607 | −0.043 | 0.28862 | 0.0016 | 0.287 | 0.091 / 0.002 |
| Bedrooms | 1726.48 | 82.39 | 0.01669 | 0.0254 | −0.009 | 0.29587 | 0.0011 | 0.295 | 0.034 / 0.001 |

The term is carried by a handful of modes, the even low modes of the centred brain oval. At
σ = 96 on IXI, $(0,2)$ has $\mu^2/P = 25.7$ and holds 55.9 % of $T$, $(2,0)$ has 18.7 and 18.3 %,
$(1,0)$ has 5.7 and 6.9 %, and $(0,4)$ has 4.1 and 5.5 %. OASIS-1 shows the same pattern:
$(0,2)$ 22.8 / 53.3 %, $(2,0)$ 17.5 / 13.5 %, $(0,4)$ 11.1 / 9.9 %. On Churches, 96.8 % of the
small $T$ is the vertical gradient $(1,0)$ (sky above ground, $\mu^2/P = 0.75$).

### 3.3 An exact split of every stored value (VERIFIED)

For each seed and mode, $\frac1M\sum_m(y_m - c)^2 = \frac1M\sum_m(y_m-\bar y)^2 + (\bar y - c)^2$
with $c = d\hat x_s$. Summing over the non-DC modes and averaging over seeds, with Parseval on the
per-image-DC-removed images, gives
$$\sum V = D_{\text{pix}}(W^2-1) + \sum B,\qquad B = \operatorname{mean}_s(\bar y_s - d\hat x_s)^2 .$$
On the pilot samples the two sides agree to 1e-15 relative (3453.18507734057 both), and
$D_{\text{pix}}(W^2-1)$ equals the directly summed within-seed variance to 3e-9. With
$\sum P_{\text{ref}}$ from §3.2, every run therefore splits, from `final.json` alone, into:

- the within-seed fraction $G_w = D_{\text{pix}}(W^2-1)/\sum P_{\text{ref}}$, with expectation
  $(1-\tfrac1M)(1-I)$ under the model. It is written as the within-seed share
  $I_w = 1 - \tfrac{M}{M-1}G_w$, which has expectation $I$ **whatever the mean image**;
- the seed-mean bias fraction $G_b = 1 - \texttt{inherited\_measured} - G_w$, with expectation
  $T + (1-I)/M$.

Real values, A0 and A3 of every dataset in `_results_partial_24` (OASIS-1 A3 and all Bedrooms runs
are not yet evaluated), $M = 50$, 40 seeds:

| run | σ | `inherited_measured` (pre-reg.) | `inherited_predicted` $I$ | $I - T$ | low band meas. / pred. | $G_w$ (exp.) | $G_b$ (exp.) | $I_w$ | var. ratio | LSD |
|---|---|---|---|---|---|---|---|---|---|---|
| ixi_A0_s1 | 96 | −0.7675 | 0.0034 | −1.398 | −2.931 / 0.0108 | 0.336 (0.977) | 1.431 (1.421) | 0.657 | 0.650 | 0.253 |
| ixi_A0_s2 | 96 | −0.8787 | 0.0034 | −1.398 | −3.245 / 0.0108 | 0.332 (0.977) | 1.547 (1.421) | 0.661 | 0.660 | 0.245 |
| ixi_A0_s3 | 96 | −0.9106 | 0.0034 | −1.398 | −3.281 / 0.0108 | 0.336 (0.977) | 1.575 (1.421) | 0.657 | 0.684 | 0.229 |
| ixi_A3_s1 | 24 | 0.0755 | 0.0795 | −0.132 | −0.046 / 0.2492 | 0.253 (0.902) | 0.671 (0.230) | 0.742 | 0.880 | 0.081 |
| ixi_A3_s2 | 24 | 0.0634 | 0.0795 | −0.132 | −0.051 / 0.2492 | 0.261 (0.902) | 0.676 (0.230) | 0.734 | 0.890 | 0.078 |
| ixi_A3_s3 | 24 | 0.0716 | 0.0795 | −0.132 | −0.052 / 0.2492 | 0.257 (0.902) | 0.672 (0.230) | 0.738 | 0.887 | 0.076 |
| oasis1_A0_s1 | 96 | −0.3897 | 0.0017 | −0.877 | −2.217 / 0.0065 | 0.432 (0.978) | 0.958 (0.899) | 0.559 | 0.649 | 0.243 |
| oasis1_A0_s2 | 96 | −0.4547 | 0.0017 | −0.877 | −2.450 / 0.0065 | 0.446 (0.978) | 1.008 (0.899) | 0.545 | 0.676 | 0.218 |
| lsun_church_A0_s1 | 96 | 0.7684 | 0.0179 | −0.043 | 0.668 / 0.0270 | 0.056 (0.962) | 0.176 (0.080) | 0.943 | 0.290 | 1.453 |
| lsun_church_A0_s2 | 96 | 0.7767 | 0.0179 | −0.043 | 0.694 / 0.0270 | 0.052 (0.962) | 0.172 (0.080) | 0.948 | 0.279 | 1.127 |
| lsun_church_A0_s3 | 96 | 0.7202 | 0.0179 | −0.043 | 0.617 / 0.0270 | 0.080 (0.962) | 0.200 (0.080) | 0.918 | 0.312 | 1.024 |
| lsun_church_A3_s1 | 24 | 0.7942 | 0.2886 | 0.287 | 0.778 / 0.4349 | 0.073 (0.697) | 0.133 (0.016) | 0.925 | 0.659 | 0.637 |
| lsun_church_A3_s2 | 24 | 0.8095 | 0.2886 | 0.287 | 0.777 / 0.4349 | 0.055 (0.697) | 0.136 (0.016) | 0.944 | 0.642 | 1.085 |
| lsun_church_A3_s3 | 24 | 0.7961 | 0.2886 | 0.287 | 0.774 / 0.4349 | 0.067 (0.697) | 0.137 (0.016) | 0.931 | 0.656 | 0.768 |

"var. ratio" and "LSD" are `final.json: variance_ratio` ($\sum P_{\text{samples}}/\sum
P_{\text{ref}}$) and `lsd` of the 2,000 training-seeded final samples, for context.

Reading the table:

- **MRI at σ = 96 (negative values).** The bias fraction is 1.43–1.57 on IXI (expected 1.42) and
  0.96–1.01 on OASIS-1 (0.90). The negative pre-registered share is the mean-image term almost
  entirely. The within-seed fraction is a third of the model's (0.33 against 0.98 on IXI).
- **Churches (0.76).** $T$ is small. The samples of one seed vary by 5–8 % of $\sum
  P_{\text{ref}}$ against an expected 96 % (σ = 96) or 70 % (σ = 24), and the whole residual about
  the prior state is 18–28 % of $\sum P_{\text{ref}}$. At σ = 96 the prior state holds 1.8 % of the
  population variance ($I$), so a sample cannot *inherit* 76 % of it. The 0.76 is the part of the
  population variance the chain did not regenerate: near-deterministic chains (low within-seed
  variance) and under-powered samples. Both are consistent with `variance_ratio` 0.28–0.31 and
  LSD 1.0–1.45 on the A0 final set.
- **IXI A3 and A1: the apparent fit is a cancellation.** Measured 0.063–0.076 against $I =
  0.0795$ looks like agreement, but it is the sum of four terms of the size of the result:
  $I = +0.080$, mean term $-0.211$, under-dispersion $+0.645$ ($0.902 - 0.257$), and excess
  seed-mean bias $-0.44$ ($0.672 - 0.230$).

### 3.4 The radial curve: the spikes are μ²/P (VERIFIED)

Mean measured curve over seeds, the pre-registered line, and the mean-corrected expectation
$(1-d^2) + (1-d)^2\mu^2/P$ (unweighted bin means, like the measured curve):

| c/img | IXI A0 meas. | $1-d^2$ | corrected | OASIS-1 A0 meas. | corrected | IXI A3 meas. | $1-d^2$ | corrected |
|---|---|---|---|---|---|---|---|---|
| 0.53 | 2.22 | 0.915 | 2.35 | 2.56 | 2.53 | 0.026 | 0.143 | 0.159 |
| 1.02 | **17.58** | 1.000 | **22.87** | **15.44** | **20.87** | 1.76 | 0.460 | 2.02 |
| 1.14 | 1.17 | 1.000 | 2.33 | 0.56 | 1.51 | 0.19 | 0.537 | 0.67 |
| 1.97 | 3.89 | 1.000 | 4.24 | 5.50 | 6.10 | 2.50 | 0.921 | 2.57 |
| 2.19 | 1.44 | 1.000 | 2.99 | 1.55 | 2.91 | 2.18 | 0.949 | 2.18 |

The 1.02 c/img bin holds the $(0,2)$ and $(2,0)$ modes. Below 3 c/img the log of the measured
curve correlates with the log of the corrected expectation at 0.91 (IXI A0), 0.98 (IXI A3) and
0.94 (OASIS-1 A0). Above 3 c/img, $\mu^2/P \le 0.28$ on MRI, the corrected expectation is at most
1.03, and the measured 0.47–1.09 is the within-seed under-dispersion of §3.3. On Churches the
corrected line differs from $1-d^2$ only in the lowest bin (1.10 against 0.915 at 0.53 c/img).
The measured Churches A0 curve lies at 0.04–0.11 above 3 c/img. In that range $d_K$ is below
$e^{-44}$, so the prior state holds nothing of the seed and nothing there can be inherited.

T6.1's share and T6.2's curve are the same measurement. The share is one minus the
$P$-weighted mean of the per-mode ratio; the curve shows unweighted per-bin means. They cannot be
compared by eye, but neither contradicts the other.

### 3.5 Consequence for the pre-registered contrast and interaction (DERIVED from §3.2–3.3)

The pre-registered A3−A0 contrast of `inherited_measured` on IXI is +0.92 (T6.1, table 3), and
the IXI−Churches interaction is +0.88. From $I$ and $T$ alone, a perfect model gives
$\Delta(I - T) = +1.27$ on IXI. Only $+0.08$ of that is $\Delta I$; $+1.19$ is the mean term
shrinking with σ_B,max. On Churches, $\Delta I = +0.27$. The mechanism the endpoint was
pre-registered to measure predicts an interaction of $0.076 - 0.271 = -0.19$ on $\Delta I$. The
pre-registered interaction is positive because $T$ moves with σ on MRI; that $T$ moves is a
property of the data, not of the model.

On the corrected $I_w$ (partial collection, descriptive): IXI A0 0.658, A3 0.738
(per-seed Δ +0.085, +0.073, +0.081); Churches A0 0.936, A3 0.934 (per-seed Δ −0.018, −0.003,
+0.013). $\Delta I_w$ on IXI (+0.079) matches $\Delta I$ (+0.076). On Churches $I_w$ does not move,
because it already sits near 1 from the under-dispersion (LIKELY).

### 3.6 Caveat on the prediction itself (DERIVED; for `main`, possibly **[ask Daniel]**)

The line $1 - d^2$ and $I$ come from a model in which the regenerated part is independent of the
seed with variance $(1-d^2)P$. An exact Gaussian posterior sampler of the IHDM prior state
$u_K = d x + \delta\eta$, with prior noise $\delta = 0.0125$ in [0, 1] units, instead pins every
mode whose SNR $d^2P/\delta^2$ exceeds 1. Its within-seed share is $I_{w,\text{post}} = 1 -
\sum P\,r(s)/\sum P$ with $r(s) = (2s+1)/(s+1)^2$, $s = d^2P/\delta^2$. This gives 0.079 (IXI,
σ = 96), 0.249 (IXI, 24), 0.040 / 0.162 (OASIS-1) and 0.331 / 0.601 (Churches), with 5–6 modes
above SNR 1 at σ = 96 and 49–58 at σ = 24. The measured $I_w$ (0.55–0.74 on MRI, 0.92–0.95 on
Churches) exceeds this reference too, so the under-dispersion finding does not depend on which
reference is used. The size of the gap does depend on it, and the report should say which
reference it uses.

## 4. Verdict, by link

| link | verdict |
|---|---|
| `ihdm/metrics/spectral.py` (`inherited_band`) | correct implementation of §5 as amended; no bug |
| `ihdm/metrics/run_eval.py` → `final.json` | correct; both shares, the low band and the radial arrays are written as specified |
| `collect.py` → `index.csv` | correct copy (VERIFIED on 24 runs) |
| `tables.py`, `figures.py` | read the right fields; the tables label the column "inherited share", which it is not (see below) |
| **05-metrics §5 / D-T4.1-1 (the estimator)** | **mis-specified**: the residual is centred on $d\hat x_s$, which is the expected sample only when $\mu = 0$ at every non-DC mode, and the quantity is normalised by $P_{\text{ref}}$, so under-regeneration reads as inheritance. The zero-mean synthetic tests could not reveal either problem |
| T6.1 | misreading: "Churches … may be … a sign convention" (it is not); "the linear-Gaussian prediction fits only IXI A3/A1" (the fit is a cancellation, §3.3) |
| T6.2 | misreading in the caption: "below it, the samples inherit more of the seed" is impossible where $d \approx 0$ (§3.4); its hunch that the low-bin spikes drive the negative share is right, and they are $\mu^2/P$ |
| T4.1 (`metrics_bracket.md` §3) | misreading: the pilot's −1.75 is "the expected reading at 750 iterations"; $T = 1.40$ accounts for most of it on any model |

## 5. Fix

**Recommended, and accepted by `main` as D23: reading only, no recomputation from the tars.**

1. Keep `inherited_measured` (pre-registered) in every table, labelled "as pre-registered (biased
   under a non-zero mean image)". Its model expectation is $I - T$, not $I$.
2. Add the within-seed share $I_w = 1 - \frac{M}{M-1}D_{\text{pix}}(W^2-1)/\sum P_{\text{ref}}$
   against $I$ (`inherited_predicted`), and the bias fraction $G_b$ against $T + (1-I)/M$, in the
   cell table, the contrasts and the interaction, with the same statistics as every other
   endpoint.
3. Figure 5: draw $(1-d^2) + (1-d)^2\mu^2/P$ as the prediction, keep $1-d^2$ as a thin labelled
   reference, and replace the caption sentence with "below the line, the chain adds less than the
   variance the blur removed; only where $d_K$ is appreciable can copying the seed contribute".

Inputs: `diversity_pix`, `n_per_seed`, `sigma_max`, `inherited_measured` and
`inherited_predicted`, all already in `final.json` / `summary.json`. Also needed per (dataset,
σ_B,max): $\sum P_{\text{ref}}$, $T$ and the radial corrected curve. These come from the `ref`
split of `images.npy`, which is available locally and on Picasso, and are checked against the
runs' `dataset_sha256`. No sample, no tar, no GPU.

**Optional follow-up (needs the held-out samples from the evaluation tars).** Two per-mode
quantities: the residual re-centred on $d\hat x_s + (1-d)\hat\mu_{\text{ref}}$ (the prior state of
a population with a mean), and the per-mode within-seed variance, as radial curves. Both need
`heldout/samples.npy` and `seeds.npy` of each run (40 × 50 × 192² uint8). The cost is a CPU
re-read (`inherited_band` on that stack takes seconds), with no resampling. This per-mode split is
the only part of the audit that the stored scalars cannot give.

**Implementation status.** The D23 reading fix is not implemented in this ticket; see the T6.3
log (`docs/AGENT-LOGS/M6-analysis/T6.3-inherited-band-audit.md` §6).

The findings are pinned by `tests/metrics/test_inherited_audit.py`. Its synthetic tests pin four results: the mean term
enters as exactly $I - T$; the radial ratio is $(1-d^2) + (1-d)^2\mu^2/P$ mode for mode;
$1 - D_{\text{pix}}(W^2-1)/\sum P$ recovers $I$ with a mean image present; and a model that
regenerates $g^2$ of the removed variance reads $1 - g^2(1-I)$. Its integration test checks the
IXI A0 and Churches A0 decomposition on `_results_partial_24`.
