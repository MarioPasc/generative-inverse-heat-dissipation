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
