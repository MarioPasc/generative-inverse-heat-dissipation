# The dynamic range of the log-spectral distance, and the inherited band on a pilot checkpoint

Produced by ticket T4.1 on 2026-09-23 from `ihdm.metrics.spectral` at base commit `3ec0d92c`;
contract `05-metrics.md` §1, §2 and §5; harness `docs/HARNESSES/metrics.md` §2. Every number below
is measured on this machine (RTX 3060, `ihdm` conda environment) and is reproducible from the
paths quoted.

The point of this file is the **bracket**: what LSD means numerically. A metric with no scale is a
number nobody can read, so the reference-versus-reference distance (the floor a perfect model
would sit at, given the same number of samples) is recorded here once, for all four datasets,
beside the LSD of a deliberately bad model. Everything the report says about LSD differences
between arms is read against this range.

Definitions, restated: for a stack $X$, $P(i,j)$ is the per-mode variance in the orthonormal DCT-II
basis after mean-centring across images, DC excluded; $\bar P(b)$ is its mean over the modes whose
radial index $n$ satisfies $n/2 \in b$ cycles per image; and

$$\mathrm{LSD}(S, R) = \sqrt{\frac{1}{B}\sum_{b}\left(\log_{10}\bar P_S(b) - \log_{10}\bar P_R(b)\right)^2}.$$

The grid is 48 log-spaced bins between 0.5 and 96 cycles per image. **Five of those bins hold no
mode of the $192^2$ DCT grid** (indices 1, 2, 4, 5, 8: the integer radii jump $1, \sqrt2, 2,
\sqrt5, \dots$ while the bins step by $192^{1/48} = 1.1157$), so $B = 43$ in every number below;
the dropped bins depend only on the image size, so the same 43 are used for the samples and for
the reference. See the ticket log, decision D-T4.1-2.

## 1. Reference-versus-reference: the noise floor

The `ref` split of each dataset is 800 images. The floor of H-METRICS §2 is `lsd(ref[:400],
ref[400:])`: what the metric reads when the "samples" are real images drawn from the same
population as the reference, with 400 on each side.

| dataset | floor `lsd(ref[:400], ref[400:])` | interleaved `lsd(ref[0::2], ref[1::2])` | `lsd(train, ref)` (3200 vs 800) | variance ratio of the floor |
|---|---|---|---|---|
| `ixi` | **0.0474** | 0.0315 | 0.0169 | 0.963 |
| `oasis1` | **0.0491** | 0.0375 | 0.0217 | 1.041 |
| `lsun_church` | **0.0230** | 0.0230 | 0.0112 | 0.976 |
| `lsun_bedroom` | **0.0367** | 0.0317 | 0.0181 | 0.934 |

All four are below the 0.05 that H-METRICS §2 asks for, but **the two MRI floors are barely
below it, and not for a reason the metric can fix**: the `ref` split of an MRI dataset is 80
subjects × 10 slices in subject order, so `ref[:400]` and `ref[400:]` are *disjoint groups of 40
subjects*. The floor therefore mixes the estimator's own noise with a real between-subject-group
difference. Two controls make the point:

* splitting the same 800 images **interleaved** (so slices of the same subject fall on both sides)
  drops IXI from 0.0474 to 0.0315 and OASIS-1 from 0.0491 to 0.0375, while `lsun_church` — where
  there are no subjects and the two splits are statistically identical — does not move at all
  (0.02302 → 0.02299);
* the per-octave profile of the MRI floors is concentrated in the coarse bins
  (IXI: $-0.061$ and $-0.064$ in $[0.5,1)$ and $[1,2)$ cycles per image, against $-0.003$ to
  $-0.017$ everywhere above 4 c/img; OASIS-1: $+0.057$ and $+0.088$), which is the
  between-scanner anterior–posterior intensity component D15′ describes, while the photograph
  floors are flat across the eight octaves.

**Consequence for the report.** When a run's final LSD is compared against "the floor", say which
floor. The subject-grouped number (0.047 / 0.049) is the honest bound on *how well any model of
80 unseen subjects can be told from another 80*; the interleaved number (0.031 / 0.037) is the
bound on the *estimator* at $N = 400$. Both are far below anything a model at these budgets will
produce, so the choice does not change a conclusion; it changes a sentence.

## 2. The pilot checkpoint: the other end of the range

Run `pilot_ixi_A0_s1` (A0: $\sigma_{B,\max} = 96$, log schedule, $K = 200$, `prior_noise=True`,
$\delta = 1.25\sigma = 0.0125$), checkpoint `ema_iter_000750.pt` — **750 training iterations**, a
model that has barely started. 64 samples, one per training seed, Alg. 2, `--batch 32`, 19.3 s per
200-step chain on the RTX 3060 (1 235.8 s for 64 chains), against the **full 800-image `ref`
split** of `ixi`.

| quantity | value |
|---|---|
| `lsd` (64 samples vs 800 reference images) | **0.7775** |
| `variance_ratio` $\sum P_S/\sum P_R$ | 0.593 |
| bins used | 43 of 48 |

Per-octave profile, $\log_{10}\bar P_S(b) - \log_{10}\bar P_R(b)$ (positive = the samples carry
too much variance in that band):

| 0.5–1 | 1–2 | 2–4 | 4–8 | 8–16 | 16–32 | 32–64 | 64–96 |
|---|---|---|---|---|---|---|---|
| **+0.885** | −0.395 | −0.329 | −0.252 | −0.770 | −1.053 | −0.652 | −0.962 |

The shape is the expected signature of an untrained network: a factor of 7.7 too much variance in
the coarsest octave and a factor of 11 too little at 16–32 cycles per image, for 59% of the
reference's total variance. It is quoted as a number, not as a judgement of the model.

**The bracket, therefore: 0.047 (IXI floor) to 0.78 (a 750-iteration model), a range of 1.2
decades.** LSD differences between arms will live in the lower part of it, and any difference
below ≈ 0.03 on MRI is inside the floor.

## 3. Inherited band on the held-out seeds

4 held-out seed subjects × 10 samples each from the same checkpoint (`--source seed`, `--batch 20`,
19.0 s per chain, 761.3 s), $\sigma_{B,\max} = 96$, against $P_{\text{ref}}$ measured on the
800-image `ixi` `ref` split.

The measured quantity is the per-mode variance of the samples about the **prior state**
$d_K \hat x_s$, $d_K(i,j) = e^{-\lambda_{ij}\sigma_{B,\max}^2/2}$ (the released `DCTBlur` kernel at
the terminal blur), normalised by $P_{\text{ref}}$; the prediction is
$1 - d_K^2 = 1 - e^{-2\lambda_n t_K}$, $t_K = \sigma_{B,\max}^2/2$. The residual is taken about
$d_K\hat x_s$ rather than about the raw seed because the raw-seed residual has expectation
$2(1-d)P$, not $(1-d^2)P$; ticket log D-T4.1-1, approved by `main`.

| band | measured $1 - \sum V/\sum P_{\text{ref}}$ | predicted $\sum P e^{-2\lambda t_K}/\sum P$ |
|---|---|---|
| all non-DC modes (36 863) | **−1.752** | **0.00343** |
| low band, $\sigma_n \ge 8$ px (101 modes, ≤ 5.40 c/img) | −6.708 | 0.01076 |

The low band is the modes whose characteristic blur scale $\sigma_n = \sqrt{2/\lambda_n} =
\sqrt2\,W/(\pi n)$ is at least 8 px, i.e. $n \le 10.80$ at $W = 192$; **both** sums are restricted
when the restriction is used, since restricting only the measured side inflates it by
$\sum_{\text{all}} P/\sum_B P = 3.13$ on IXI (which is exactly the ratio of the two predicted
columns, 0.01076 / 0.00343).

A negative measured share means $\sum V > \sum P_{\text{ref}}$: the samples are *further* from the
prior state than two independent reference images are from each other. At 750 iterations that is
the expected reading and it is consistent with the LSD profile above — the radial curve makes the
same statement mode by mode:

| cycles per image | 0.53 | 1.02 | 2.19 | 4.23 | 8.17 | 15.75 | 30.39 | 65.43 | 90.88 |
|---|---|---|---|---|---|---|---|---|---|
| measured $V/P_{\text{ref}}$ | 28.9 | 1.73 | 0.29 | 1.80 | 0.79 | 0.134 | 0.144 | 0.183 | 0.104 |
| predicted $1 - d_K^2$ | 0.915 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

29× too much variance in the coarsest bin and 6–10× too little above 15 cycles per image: the same
two facts the octave profile of §2 reports (+0.885 at 0.5–1, −1.053 at 16–32). At
$\sigma_{B,\max} = 96$ the prediction is flat at 1 above one cycle per image, because a blur of 96
px erases everything else, so the figure has almost no dynamic range in this arm; the predicted
share on the same spectrum at $\sigma_{B,\max} = 24$ is 0.0795, which is where the mechanism
figure of the A3 arm will be legible.

**What this section does and does not establish.** It establishes that the pipeline runs end to
end on real sampler output in the `sample_ckpt` layout: `(S, M, H, W)` `uint8` samples,
`seeds.npy`, a reference spectrum from the `ref` split, finite numbers on both sides. It does not
say anything about the model, which has seen 750 iterations.

## 4. Reproducing this file

```bash
export IHDM_DATA_ROOT=/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project
export IHDM_RUN_ROOT=$IHDM_DATA_ROOT/_runs_local
R=$IHDM_RUN_ROOT/pilot_ixi_A0_s1
python -m ihdm.cli.sample_ckpt --run $R --ckpt ema_iter_000750.pt --source train \
    --n-seeds 64 --n-per-seed 1 --batch 32 --out $IHDM_RUN_ROOT/t41_lsd_samples
python -m ihdm.cli.sample_ckpt --run $R --ckpt ema_iter_000750.pt --source seed \
    --n-seeds 4 --n-per-seed 10 --batch 20 --out $IHDM_RUN_ROOT/t41_seed_samples
pytest tests/metrics -q -m integration
```

`--batch 40` on the second command runs out of memory on the 12 GB RTX 3060 at $192^2$
(`torch.OutOfMemoryError`, 8.9 GiB already in use, 2.11 GiB more requested, before the first
chain); 20 is the largest value tried that fits beside a desktop session. The chain cost is flat
in the batch on this GPU (19.3 s per chain at 32, 18.2 s at 20 per T2.2), so nothing is lost.

## 5. T4.2 — memorisation, diversity and PCA on the pilot

Same run and checkpoint as sections 1–4 (`pilot_ixi_A0_s1`, `ema_iter_000750.pt`, A0:
$\sigma_{B,\max} = 96$, $K = 200$, `prior_noise=True`), same two sample folders, dataset `ixi`
(N4-corrected). Corpus = the 3 200-image `train` split; held-out = the 400 rows with
`index["split"] == "ref"`, i.e. the `ref` split minus the 40 seed subjects. Distances are
Euclidean over the 36 864 pixels of a $192^2$ image in $[0, 1]$, per-image DC removed.

**Only 46 of the 64 training-seeded pilot samples are used.** The other 18 have a `seed_idx`
that is not a row of the current `splits["train"]` (6 of them are in `splits["seed"]`), so they
are not training-seeded and $M$ is not defined on them; see
`docs/AGENT-LOGS/M4-metrics/T4.2-memorisation-and-diversity.md` §6.

### 5.1 The memorisation ratio

| set as "samples" | $n$ | $M$ | $M_{lp}$ ($\sigma_B = 16$) | `seed_nn_fraction` |
|---|---|---|---|---|
| **pilot, training-seeded** | 46 | **1.752** | **8.722** | **0.000** |
| pilot, all samples (seed provenance ignored) | 64 | 1.769 | 8.813 | n/a |
| **control: 200 real `ref` images** | 200 | **1.001** | **0.992** | n/a |

The control is the calibration of the metric: real held-out images give $M = 1$ to 0.1% and
$M_{lp} = 1$ to 0.8%, which is the identity H-METRICS §1 asks for, measured here on the real
data rather than on a synthetic field.

Medians behind the ratios (pixel space): $d(\text{samples}) = 56.64$,
$d(\text{held-out}) = 32.34$. In the low-pass band: $d(\text{samples}) = 34.52$,
$d(\text{held-out}) = 3.96$.

**What this says, and it is not what the ticket expected.** $M$ at 750 iterations is **above**
one, not below: the samples are 1.75× *further* from the training set than a new real brain is,
and 8.7× further once both sides are blurred to $\sigma_B = 16$ px. They are not copies of
anything — the nearest training image is never the sample's own seed (0 of 46), and each sample
sits 67.1 from its own seed, 1.18× further than from its nearest training image. The mechanism
is visible in the spectrum: the samples carry 7% more non-DC pixel variance than the `ref`
images (0.0919 vs 0.0855) but **57% more in the $\sigma_B = 16$ band** (0.0639 vs 0.0406), and
real registered brains are nearly identical at that scale (their nearest-neighbour distance is
3.96, an eighth of the pixel-space one), so an excess of coarse structure is punished hard.

The reading for the report: **$M$ is not one-sided.** $M \ll 1$ is copying; $M \gg 1$ is a model
that has not reached the data manifold at all, which is what an undertrained checkpoint looks
like. $M$ is only interpretable as "copying or not" for a run whose LSD and KID say it fits;
until then it is a distance-to-manifold diagnostic. Nothing here is evidence for or against H1,
because 750 iterations is 1.9% of a run.

### 5.2 Within-seed diversity (4 held-out seeds × 10 samples)

| quantity | value | reference scale | share |
|---|---|---|---|
| $D_{\text{pix}}$ (mean over 4 seeds) | $3.996\times10^{-3}$ | non-DC pixel variance of the 400 `ref` images, $8.552\times10^{-2}$ | 4.67% |
| $D_{\text{lp}}$ ($\sigma_B = 16$) | $1.464\times10^{-4}$ | the same after the low-pass, $4.064\times10^{-2}$ | 0.36% |

Per seed: $D_{\text{pix}} = [3.2, 3.5, 3.7, 5.5]\times10^{-3}$,
$D_{\text{lp}} = [0.89, 1.06, 1.28, 2.62]\times10^{-4}$. Units are squared $[0,1]$ intensity per
pixel; the $1/M$ normalisation of `05-metrics.md` §3 makes these estimate $(1 - 1/10) = 0.9$
times the true within-seed variance at this $M = 10$ (0.98 at the $M = 50$ of the real runs).

Ten samples per seed and four seeds are a pilot, not a measurement: the number to quote is that
the chain moves by 4.7% of the real pixel variance and by 0.36% of the real coarse variance when
only the sampling noise changes. The coarse share being an order of magnitude below the pixel
share is the signature the terminal-blur hypothesis predicts (the prior fixes the coarse modes
and the chain regenerates only the fine ones), but at $\sigma_{B,\max} = 96$ and 750 iterations
it cannot be separated from a model that has simply not learned to move.

### 5.3 PCA around the seed (2 components, fitted on the 3 200 training images)

Explained variance $[122.7, 98.8]$; training score standard deviation $[11.08, 9.94]$.

| seed | seed score | sample centroid | arrow length | within-seed sample sd |
|---|---|---|---|---|
| 0 | $(-4.20, +22.60)$ | $(-4.03, -8.80)$ | 31.40 | $(0.61, 0.35)$ |
| 1 | $(-7.92, +6.12)$ | $(+3.06, -15.32)$ | 24.09 | $(0.52, 0.56)$ |
| 2 | $(-5.41, +8.30)$ | $(+1.27, -13.19)$ | 22.50 | $(0.35, 0.51)$ |
| 3 | $(-4.28, +3.45)$ | $(-0.41, -11.96)$ | 15.88 | $(0.72, 0.73)$ |

The four arrows all point the same way and land in the same place ($-4$ to $+3$ on PC1, $-15$ to
$-9$ on PC2), 1.5 training standard deviations from where the seeds are, while the spread of the
10 samples of one seed is 3–7% of the training standard deviation. At this checkpoint the model
maps every seed onto one region of the training PCA plane: mode collapse plus a systematic
offset, not a copy of the seed. The panel is the figure `05-metrics.md` §6 asks for; it will be
worth reading at 40k iterations.

### 5.4 Cost and peak memory of the nearest-neighbour path

Measured on this machine (RTX 3060 12 GB, 31 GB RAM, `chunk = 256`, fresh process per row;
"host RSS" is the process high-water mark, of which 0.70–0.72 GiB is the `uint8` input itself).

| call | device | wall | CUDA allocated / reserved | host RSS |
|---|---|---|---|---|
| `nn_distances(2000, 3200)` at $192^2$ | cuda | 0.68 s | 0.960 / 1.393 GiB | 1.84 GiB |
| `nn_distances(2000, 3200)` at $192^2$ | cpu | 1.57 s | — | 1.95 GiB |
| `memorisation_ratio(2000, 3200, 400)` at $192^2$ (both ratios) | cuda | 6.6 s | 0.960 / 1.428 GiB | 3.53 GiB |
| `memorisation_ratio(2000, 3200, 400)` at $192^2$ (both ratios) | cpu | 8.5 s | — | 3.52 GiB |
| `memorisation_ratio(46, 3200, 400)`, the pilot call above | cuda | 3.8 s | 0.960 GiB | 2.93 GiB |
| `pca_around_seed` on the 3 200 × 36 864 training split | cuda | 0.48 s | — | — |

Both device paths return the same nearest neighbours and the same median distance to four
decimals at this scale. The CUDA figure is dominated by the resident corpus (3 200 × 36 864
`float32` = 0.44 GiB) plus one query chunk; the whole array's evaluation therefore fits with
10 GB to spare, and the CPU fallback fits with 27 GB to spare.

### 5.5 Reproducing section 5

```bash
export IHDM_DATA_ROOT=/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project
export IHDM_RUN_ROOT=$IHDM_DATA_ROOT/_runs_local
pytest tests/metrics/test_memorisation.py tests/metrics/test_diversity.py -q -m integration
```

The tables above come from two scratchpad scripts (not committed, listed verbatim in the T4.2
log §4) that call `memorisation_ratio`, `within_seed_diversity` and `pca_around_seed` on the
same paths the integration tests use; the integration tests assert the shapes, the finiteness
and the ranges, not the values, because the values belong to one pilot checkpoint.

## 6. T4.3 — the whole evaluation on the pilot, the Inception metrics and the plateau gate

Same run and checkpoints as sections 1–5 (`pilot_ixi_A0_s1`, A0: $\sigma_{B,\max} = 96$,
$K = 200$, `prior_noise=True`), dataset `ixi` (N4-corrected), but drawn by `evaluate_run` itself
rather than by hand, so the seeds are the **frozen lists of D17** and not an ad-hoc draw. This is
the first evaluation in which every sample's seed is provably a row of `splits["train"]`
(`assert_in_split` on load), which is what T4.2 §6 asked for.

```bash
export IHDM_DATA_ROOT=/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project
python -m ihdm.cli.evaluate_run --run $IHDM_DATA_ROOT/_runs_local/pilot_ixi_A0_s1 \
  --ckpts 250,750 --n-lsd 32 --n-final 64 --n-seeds 4 --n-per-seed 10 \
  --fid-batch 32 --sample-batch 20 --device cuda
python -m ihdm.cli.evaluate_run --run $IHDM_DATA_ROOT/_runs_local/pilot_ixi_A0_s1 \
  --ckpts 250,750 --n-lsd 32 --n-final 64 --n-seeds 4 --n-per-seed 10 --gate 250,750 --skip-inception
```

### 6.1 The frozen evaluation seed lists (D17)

Written once per dataset by `ensure_seed_lists`, reused by every checkpoint and every arm.

| list | file | n | distinct | rule | sha256 |
|---|---|---|---|---|---|
| intermediate | `ixi/eval_seeds_500.npy` | 500 | 500 | `splits['train']` without replacement, `default_rng(2026)` | `e51c999eb99c3308…` |
| final | `ixi/eval_seeds_final_2000.npy` | 2000 | 1462 | `splits['train']` with replacement, `default_rng(0)` | `28ee6357f2bdd8c2…` |

A run evaluated with fewer samples takes a **prefix** of the list, so its noise streams stay a
subset of a full run's. Both digests are recorded in every `metrics/summary.json`.

### 6.2 LSD at the two checkpoints (32 training-seeded samples each, 800-image `ref`)

| step | LSD | variance ratio | $n$ |
|---|---|---|---|
| 250 | **0.7731** | 0.987 | 32 |
| 750 | **0.7622** | 0.595 | 32 |
| 750, the 64-sample final set | **0.7894** | 0.498 | 64 |

The 64-sample final value (0.7894) sits within 1.5% of T4.1's independent 0.7775 on its own
64-sample draw of the same checkpoint, which is the cross-check that the two sampling paths
agree. The variance ratio falling from 0.99 to 0.59 between 250 and 750 while the LSD barely
moves is the substance: the model is losing total variance without moving its *shape* closer to
the reference.

Octave profiles ($\log_{10}$ sample minus reference, positive = too much variance):

| step | 0.5–1 | 1–2 | 2–4 | 4–8 | 8–16 | 16–32 | 32–64 | 64–96 |
|---|---|---|---|---|---|---|---|---|
| 250 | +0.764 | −0.482 | −0.452 | +0.106 | +0.120 | −0.603 | −0.851 | −1.544 |
| 750 | +0.892 | −0.321 | −0.350 | −0.283 | −0.783 | −1.055 | −0.641 | −0.946 |

Between 250 and 750 the model gives up the 8–32 cycles-per-image band (+0.12 → −0.78 and
−0.60 → −1.06) and piles further variance into the coarsest octave. At 750 iterations that is
a model still collapsing onto the low modes, not the terminal-blur effect the experiment is
about.

### 6.3 The final checkpoint: memorisation, diversity, inherited band

| quantity | T4.3 (64 samples, frozen list) | T4.2 (46 samples, ad-hoc draw) |
|---|---|---|
| $M$ | **1.754** | 1.752 |
| $M_{lp}$ ($\sigma_B = 16$) | **8.707** | 8.722 |
| `seed_nn_fraction` | **0.031** (2 of 64) | 0.000 (0 of 46) |
| median $d$(samples) / $d$(held-out) | 56.73 / 32.34 | 56.64 / 32.34 |
| $D_{\text{pix}}$ (4 seeds × 10) | **3.949e-3** | 3.996e-3 |
| $D_{\text{lp}}$ | **1.485e-4** | 1.464e-4 |
| inherited measured / predicted (all non-DC) | **−1.753 / 0.00343** | −1.752 / 0.00343 |
| inherited, low band ($\sigma_n \ge 8$ px) | **−6.714 / 0.01076** | −6.708 / 0.01076 |
| PCA explained variance | **[122.70, 98.79]** | [122.7, 98.8] |

Every quantity reproduces T4.2's to within the sampling noise of two different draws of the same
checkpoint, on a seed set that is now verifiably inside the training split. $M = 1.75$ is read as
T4.2 established: **two-sided**, and $\gg 1$ here means off-manifold, not "no copying".

The inherited radial curve is written with **43 of 48 bins**; the five bins that hold no mode of
the $192^2$ grid are dropped rather than written, because `write_json` refuses `nan` (T4.1 §6).

### 6.4 Inception metrics (`05` §7; KID is the headline)

800-image `ref` reference, 64 training-seeded samples, features cached at
`ixi/_features_inception_ref.npy`, 200 bootstrap resamples over samples.

| quantity | value | 95% interval |
|---|---|---|
| **KID** | **0.4315** | [0.4195, 0.4495] |
| FID ($N_{\text{ref}} = 800$) | 348.86 | [348.70, 363.13] |
| precision / recall | 0.000 / 0.000 | — |
| density / coverage ($k = 5$) | 0.000 / 0.000 | — |

A KID of 0.43 is enormous (two halves of the real `ref` split give $|KID| < 0.05$ at this sample
size) and precision, recall, density and coverage are all exactly zero: at 750 iterations not one
of the 64 samples has a real neighbour within the $k = 5$ manifold radius, and no real image has
a sample neighbour. This is the same statement as $M \gg 1$ and as the octave profile, measured
by a third instrument. **No absolute FID here is comparable to the paper's**: 348.86 is against
800 references, and the $1/N_{\text{ref}}$ bias is why `05` §7 makes KID the headline.

The FID interval is strongly asymmetric about its point estimate: the lower end sits 0.16 below
348.86 while the upper end sits 14.3 above it. A bootstrap resample holds duplicate samples,
which inflates FID's finite-sample bias, so the replicate distribution is shifted upward and the
percentile interval measures spread rather than location; on other sample sizes the interval can
lie entirely above the point estimate, which is asserted in
`tests/metrics/test_inception.py::test_the_fid_bootstrap_interval_sits_above_the_point_estimate`
and recorded in the `notes` block of every `final.json`. KID, being unbiased, is the metric whose
location the interval is about — and it is also the headline for the separate
$1/N_{\text{ref}}$ reason above.

**Inception weights path: `/tmp/inception-2015-12-05.pt`.** `cleanfid.features.feature_extractor`
hard-codes this directory (`/tmp` on Linux) and passes `download=True`; it is not
`~/.cache`. T5.1's Picasso worker must stage the file into `/tmp` on **every** compute node, or
run the first evaluation on a node with outbound network access. The path is printed by the CLI
before any work and recorded in `final.json["inception"]["weights_path"]`.

### 6.5 The paired plateau gate (D10 / D17)

```
metrics/gate.json: step_a=250, step_b=750, 32 paired seeds, 43 radial bins, 1000 resamples
```

| quantity | value |
|---|---|
| LSD at 250 / 750 | 0.7731 / 0.7622 |
| paired difference $\mathrm{LSD}_{250} - \mathrm{LSD}_{750}$ | **+0.0109** |
| 95% bootstrap interval over resampled seeds | **[−0.0502, +0.0492]** |
| `extend` | **False** |
| relative change (the 5% quantity D10 was first worded with) | 1.41% |

The point improvement is positive but the interval straddles zero by a factor of five, so the
gate says "not detectable at this budget" and does **not** extend. Note what the two rules would
have decided differently: D10's original wording ("extend if the LSD changes by more than 5%
between the last two checkpoints") gives 1.41% and also does not extend, but it would have made
the decision from a point estimate with no uncertainty attached. With 32 seeds the interval is
$\pm 0.05$ on an LSD of 0.77; at the real budget of 500 seeds it narrows by $\sqrt{500/32} = 4.0$
to roughly $\pm 0.012$, which is the resolution the 35k-versus-40k decision will actually have.

### 6.6 Cost

| stage | count | wall | per unit |
|---|---|---|---|
| LSD set, step 250 | 32 chains | 605.9 s | 18.93 s/chain |
| LSD set, step 750 | 32 chains | 581.2 s | 18.16 s/chain |
| final set, step 750 | 64 chains | 1162.5 s | 18.16 s/chain |
| held-out set, step 750 | 40 chains | 724.1 s | 18.10 s/chain |
| Inception: 800 ref features + 64 sample features + 200 resamples | — | ≈ 13 s | — |
| plateau gate, 1000 resamples, 32 seeds | — | ≈ 3 s | — |
| **total** | 168 chains | **51 min** | 18.3 s/chain |

Measured on the RTX 3060 at sampling batch 20, **without AMP** (`SampleRequest.amp` defaults to
`False` and `evaluate_run` does not set it). D16 budgeted **3.2 s/chain on the A100** at batch
32–64; the 5.7× gap is partly the card and partly AMP, and T5.1 should confirm the A100 figure
through `evaluate_run` before the 214 A100-hour evaluation budget is treated as settled. A rerun
of the same command reuses every cached set and costs seconds: the second invocation above
(the gate) redrew nothing.

The Inception bootstrap is affordable only because `fid_from_features`' dense `scipy.linalg.sqrtm`
(9.5–10.0 s per call at 2048 dimensions, independent of the sample count) is replaced inside the
bootstrap by the algebraically identical Gram-matrix form (0.003 s at $n = 64$, 0.24 s at
$n = 2000$), which `tests/metrics/test_inception.py` pins to the frozen estimator to a relative
1e-6. Without it, 200 resamples would cost 33 minutes per run.

### 6.7 The FID licence check (H-METRICS §3) — it passes

`EXPERIMENT_PLAN` §6.1 makes the Inception metrics interpretable at $192^2$ grayscale only if FID
rises monotonically along a blur ladder. Measured on 64 real `ixi` `ref` images, each blurred with
the released DCT heat kernel and compared against the unblurred set:

| $\sigma_B$ (px) | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| FID | −0.000 | 55.23 | 120.82 | 210.03 | 316.95 |
| KID | −0.0052 | +0.0298 | +0.0967 | +0.1928 | +0.3662 |

Both are monotone and a $\sigma_B = 1$ blur is already resolved (FID 55), so **FID and KID may be
interpreted on this data**, within the $N_{\text{ref}}$ caveat of §6.4. Two disjoint halves of the
same 64 images give $\mathrm{KID} = +0.0026$, the noise floor at this sample size.

The check is `tests/metrics/test_inception.py::test_the_fid_licence_check_of_the_blur_ladder`
(marked `integration`). A **uniform intensity shift is not a usable perturbation**: +40 grey
levels on these images gives $\mathrm{KID} = -0.0012$, inside the noise floor, so brightness
alone does not move the Inception features at $n = 64$. Only the blur ladder is quoted.

### 6.8 The Inception extractor is not bitwise deterministic

Two calls of `inception_features` on the same 64 images, same device, same batch, differ by up to
**2.0e-3** in absolute value (mean 9.5e-5) on features of order 0.1–1; CUDA against CPU differs
by up to **6.6e-3**. The cause is cuDNN algorithm selection inside the torchscript Inception, not
this code.

The consequence is the reason `_features_inception_ref.npy` exists and is keyed by the dataset's
`sha256_images`: **every arm of a dataset must be scored against one stored reference feature
matrix**, never against a freshly computed one, or a few units of FID would separate two arms for
no reason other than which node computed the reference. T5.1 must therefore let the first
evaluation of a dataset write the cache and every later job read it, rather than letting 30
concurrent array tasks each compute their own.
