<!--
STATUS: UNVERIFIED first draft. Stopped by Mario on 2026-10-07 ("Stop the draft") before the fact-check
(T8.7) and the bibliography verification (T8.6) ran; both prompts are in projects/GenAI/code/wave-W20-prompts/.
Do not port to LaTeX before those checks.

Paper draft, markdown stage (Mario, 2026-10-07: "markdown first, LaTeX later").
Written by [Orchestrator-GenAI] against docs/SPECIFICATIONS/M8-paper/00-framing.md (claims ledger C1–C8,
"must not claim" list) and the course harnesses (projects/GenAI/project/HARNESSES/).
Every number carries an HTML comment with its ledger id and source file, for the fact-check pass.
Citations: [@key]. Keys marked ‡ are new and wait for the verified entries of T8.6
(docs/PAPER/related_work.bib); all other keys are in Overleaf proyecto/references.bib.
Figure numbering of the paper: Fig. 1 = F1 (visual abstract), Fig. 2 = FM (method), Fig. 3 = F2 (arms),
Fig. 4 = F3 (fidelity against diversity), Table 1 = T1. Captions here are the short versions; the long
generated captions are in docs/RESULTS/paper/{F1,FM,F2,F3,T1}.md.
Word budget (8 pages, NIPS 2015 style, four figures and one table ≈ 2 pages): about 3,600 words of body text.
-->

# Matching the Prior and Level Spacing of Inverse Heat Dissipation Models to Brain MRI

Saúl De los Reyes Pérez-Cejuela, Rodrigo Laso Mielgo, Mario Pascual González — Universidad Autónoma de Madrid

## Abstract

Inverse heat dissipation models (IHDM) generate images by reversing a blur: sampling starts from a
training image blurred to a terminal length-scale and restores detail from coarse to fine through a
fixed sequence of blur levels. Both settings were chosen for natural photographs. Brain MRI differs
from photographs in two ways that bear on them. Its between-image variance is concentrated in
mid frequencies, and registered subjects share their coarse anatomy. We ask whether IHDM benefits
from matching the two settings to these properties. In a 2 × 2 factorial on IXI (3,200 axial T1
slices) we cross the terminal blur (the default W/2 against W/8) with the level spacing (logarithmic
against fitted to the IXI spectrum), and we repeat the endpoints of the default and the matched
configuration on OASIS-1. The matched configuration lowers KID by 63% and the log-spectral distance
by 68% on IXI (three seeds, all agree in sign), and by 49% and 52% on OASIS-1. Nearly all of the gain
comes from the terminal blur. The price is copying: 72% of samples have their own seed subject as
nearest training image, against 6% at the default. Per-sample realism holds on 40 subjects the models
never saw, but most of the population-level gain does not.
<!-- C1: t1_main.md; results_discussion §4, §7. C2: table 4. C4: tables 1b. C5: heldout_fidelity/README.md. ~190 words -->

## 1 Introduction

Inverse heat dissipation models [@rissanen2023ihdm] replace the additive noise of diffusion models
[@ho2020ddpm; @song2021sde] with the heat equation. The forward process blurs an image; in the basis
of the discrete cosine transform (DCT) the blur multiplies every mode by a factor that decays with its
squared frequency, so fine detail disappears first and coarse structure last. A network learns to undo
one small step of blur at a time, and sampling runs the chain from a strongly blurred state back to a
sharp image. Two settings decide how the chain is laid out. The *prior* fixes where it starts: a
training image blurred to a terminal length-scale $\sigma_{B,\max}$, plus a little noise. The *level
spacing* fixes how the $K$ blur levels between $\sigma_{B,\min}$ and $\sigma_{B,\max}$ are distributed.
The released configuration uses $\sigma_{B,\max} = W/2$ for a $W \times W$ image and logarithmic
spacing.

Both defaults fit natural photographs. Logarithmic spacing gives every frequency octave the same
number of levels. That is the efficient allocation when every octave carries the same between-image
variance, which holds in the continuum for the $1/f^2$ power spectrum of natural images
[@field1987relations; @ruderman1994statistics]. A terminal blur of $W/2$ erases almost all of the seed
image, which suits photographs, where no two scenes share a layout.

Brain MRI departs from both assumptions (Fig. 1). After rigid registration, IXI holds 29.8% of its
between-image variance in the 8–16 cycles-per-image (c/img) octave and 4.1% in the coarsest octave,
whereas LSUN Churches holds 23.8% in its coarsest octave. Registered subjects also share their coarse
anatomy: the mean training slice is itself recognisably a brain. MRI is also a narrower distribution
than photographs: the local intrinsic dimension of the training set, estimated by maximum likelihood
[@levina2004maximum‡], is 17.2 for IXI and 22.1 for OASIS-1 against 30.1 and 30.8 for LSUN Churches and
Bedrooms.
<!-- data_profile/ixi.npz octave_shares_train; lsun_church.npz; intrinsic_dimension/README.md (k = 10, N = 3,200) -->

We therefore match the two settings to MRI. A terminal blur of $W/8$ lets the prior hand over the
subject's coarse anatomy, which subjects share and which is cheap to inherit. A level spacing fitted to
the IXI spectrum places the levels so that each removes the same between-image variance. Our
contributions are:

- a controlled comparison on brain MRI that separates the two settings in a 2 × 2 factorial, with a
  second cohort (14 MRI training runs, 2–3 seeds per configuration);
- the finding that the terminal blur carries nearly all of the fidelity gain, together with its price
  in copying and within-seed diversity;
- an analysis of fidelity against diversity that separates per-sample realism from population-level
  diversity, using samples started from subjects the models never saw.
<!-- 14 runs = IXI 3+2+2+3 + OASIS-1 2+2 -->

> **Figure 1** (F1, `f1_visual_abstract.pdf`, 5.5 × 2.85 in). *IHDM's defaults assume natural-image
> statistics; brain MRI departs from both.* (a) Heat states of one IXI slice on a common frequency axis;
> bars: IXI between-image variance per octave; dashed: LSUN Churches; dotted: the equal share of a
> $1/f^2$ spectrum, which logarithmic spacing assumes. Shading: variance kept by the default ($W/2$)
> and the matched ($W/8$) prior; rugs: the 200 levels of each schedule. (b) The mean of 3,200 training
> slices and noise-free priors of held-out subjects. (c) The four configurations on one unseen subject.

## 2 Objectives

- **O1 (prior).** Does a terminal blur matched to the shared anatomy of MRI ($W/8$ instead of $W/2$)
  improve sample fidelity on brain MRI, and what does it cost in copying and diversity?
- **O2 (spacing).** Does a level spacing fitted to the IXI spectrum improve fidelity over logarithmic
  spacing, on its own and on top of $W/8$?
- **O3 (generalisation).** Do the effects hold on a second cohort (OASIS-1), and does the gain hold for
  subjects the model never saw?
<!-- 00-framing.md §3 -->

## 3 Related work and state of the art

**Blurring as a forward process.** IHDM [@rissanen2023ihdm] is one of several generators whose
forward process is a deterministic degradation rather than additive noise alone. Cold diffusion
[@bansal2023cold‡] shows that sampling can invert arbitrary degradations, blur included; blurring
diffusion models [@hoogeboom2023blurring‡] combine the heat equation with Gaussian noise in a single
diffusion process, and inversion by direct iteration [@delbracio2023inversion‡] learns to undo blur by
small regression steps. All of these inherit their schedules from natural-image practice.

**Schedules and the data spectrum.** In diffusion models the noise schedule decides which frequencies
are generated at which step, because white noise drowns the weak, high-frequency modes of a $1/f^2$
spectrum first [@falck2025fourier‡]. Several works adapt the schedule to the data: to image resolution
[@chen2023importance‡; @hoogeboom2023simple‡], by a cosine rule [@nichol2021improved‡], or by learning
it [@kingma2021variational‡]. Our spacing arm is the heat-equation analogue: we fit the blur levels to
the measured spectrum of the training set. Spectral discrepancies are also an established diagnostic
of generators [@durall2020watch; @schwarz2021frequency]; our log-spectral distance follows that line.

**Starting from a real image, and memorisation.** A prior that keeps part of a real image is close to
SDEdit [@meng2022sdedit‡], which noises an input to an intermediate level and denoises it with a
diffusion model. Diffusion models can reproduce training images [@carlini2023extracting;
@somepalli2023diffusion‡], and brain-MRI generators in particular have been shown to memorise
[@akbar2025beware‡; @dar2023investigating‡]; we measure copying with a nearest-neighbour test in the
spirit of [@meehan2020nonparametric‡]. Diffusion models now dominate medical image synthesis
[@kazerouni2023diffusion‡], including brain MRI [@pinaya2022brain‡].

**Diffusion and flow matching.** DDPM [@ho2020ddpm], score-based models [@song2021sde] and flow
matching [@lipman2023flow‡; @liu2023flow‡; @albergo2023building‡] are the reference generators. We
train none of them; Section 5.6 argues when IHDM is preferable and when it is not.
<!-- ~360 words; final grouping to follow T8.6 related_work_notes.md "Suggested grouping" -->

## 4 Methodology

### 4.1 Data

We use T1-weighted brain MRI from IXI [@ixi] as the development set and OASIS-1 [@marcus2007oasis] as
an independent cohort. Each volume is rigidly registered to MNI152 space, corrected for bias field with
N4 [@tustison2010n4itk‡], and scaled by its foreground 99th percentile. We take 10 axial slices per
subject at 192 × 192 pixels <!-- VERIFY: crop or pad, and which slice range (data_profile.md, ihdm/preprocess) --> and split by subject (stratified by site on IXI): each dataset has
3,200 training and 800 reference images from 400 subjects. LSUN Churches and Bedrooms [@yu2015lsun],
converted to grayscale and centre-cropped to 192 × 192, serve as natural-image references (Appendix A).
<!-- results_discussion §2 "What was run"; data_profile.md -->

### 4.2 IHDM, the prior and the level spacing

Let $u_0 \in \mathbb{R}^{W\times W}$ be an image and $\tilde u_{0,i}$ its coefficient on the orthonormal
2-D DCT-II mode $i = (i_1, i_2)$, which carries $c_i = \tfrac12\sqrt{i_1^2+i_2^2}$ cycles per image.
The heat equation with Neumann boundaries is diagonal in this basis: blurring to time $t$ multiplies
mode $i$ by

$$d_i(t) = e^{-\lambda_i t}, \qquad \lambda_i = \frac{\pi^2 (i_1^2 + i_2^2)}{W^2}, \qquad t = \sigma_B^2/2, \tag{1}$$

so a blur of length-scale $\sigma_B$ keeps $d = e^{-1}$ of a mode at $c \approx 43.2/\sigma_B$ c/img
for $W = 192$. IHDM defines $K$ levels $0 < \sigma_{B,1} < \dots < \sigma_{B,K} = \sigma_{B,\max}$
and the factorised forward process

$$q(\mathbf u_k \mid \mathbf u_0) = \mathcal N\big(\mathbf u_k;\, F(t_k)\,\mathbf u_0,\, \sigma^2 I\big), \tag{2}$$

where $F(t)$ applies Eq. (1) to every mode and $\sigma = 0.01$. The reverse process
$p_\theta(\mathbf u_{k-1}\mid\mathbf u_k) = \mathcal N(\mu_\theta(\mathbf u_k,k),\,\delta^2 I)$, with
$\delta = 1.25\sigma$, is trained by regressing $\mu_\theta(\mathbf u_k,k)$ onto $F(t_{k-1})\mathbf u_0$
at a level $k$ drawn uniformly [@rissanen2023ihdm]. Sampling starts from the prior: a training image
blurred to $\sigma_{B,\max}$ plus noise. Figure 2 shows the process octave by octave.

**The prior.** At $\sigma_{B,\max} = W/2 = 96$ px the prior keeps 0.3% of IXI's between-image
variance; at $W/8 = 24$ px it keeps 8.0%, mostly below 2 c/img, where the head outline lies.
<!-- inherited_band_constants.json: I = 0.003432 (W/2), 0.07954 (W/8), ixi ref split -->

**The level spacing.** Because levels are drawn uniformly, the spacing alone decides how much training
effort each frequency band receives. With $P_i$ the between-image variance of mode $i$, the
data-dependent part of the regression target at level $k$ is

$$R_k = \sum_i \big(d_i(t_{k-1}) - d_i(t_k)\big)^2 P_i . \tag{3}$$

Logarithmic spacing gives every octave the same number of levels. Our *matched* spacing instead places
the levels so that $R_2 = \dots = R_K$ on the IXI training split, with the same endpoints
$\sigma_{B,1} = 0.5$ px and $\sigma_{B,\max}$; it is fitted separately for each terminal blur. Against
logarithmic spacing it moves 25 of the 200 levels to another octave at $W/2$ and 19 at $W/8$.
<!-- ihdm/spectral/schedules.py; results_discussion §2 "How far the matched spacing moves the levels" (net reallocation) -->

> **Figure 2** (FM, `fm_method.pdf`, 5.5 × 3.4 in). *IHDM octave by octave.* Top: an IXI slice $\mathbf x$
> is a weighted sum $\sum_{i,j} w_{i,j}\varphi_{i,j}$ of DCT modes, grouped into octaves (colours; the
> mean in grey); the inset shows each octave as a quarter-annulus of the index plane. Rows: noise-free
> heat states; each arrow is a macro-step over the levels that act on one octave (default 31/26/26/26/
> 27/26/26/12 and matched 0/5/30/38/47/42/29/9 of the 200 levels, octaves 0.5–1 to 64–96 c/img). The
> reverse row is the ideal path. Dashed: where generation starts at $W/2$ and $W/8$. Right: IXI's
> variance per octave and the levels each spacing gives it.

### 4.3 Configurations and training

The 2 × 2 factorial gives four configurations: *default* (A0: $W/2$, logarithmic), *+prior* (A1: $W/8$,
logarithmic), *+spacing* (A2: $W/2$, matched) and *matched* (A3: $W/8$, matched). We train default and
matched with three seeds and +prior and +spacing with two on IXI, and default and matched with two
seeds on OASIS-1, using the IXI-fitted spacing on both cohorts. Every run uses the released IHDM code
with a 61 M-parameter U-Net, $K = 200$, batch 16, Adam at learning rate $10^{-4}$ with 1,000 warm-up
steps, gradient clipping at 1.0, mixed precision, an exponential moving average of 0.999 and 60,000
iterations, about 10 h on one A100 GPU per run. The whole campaign, including the natural-image runs
and evaluation, took about 500 A100-hours.
<!-- results_discussion §2; configs/spectral/arms.py; submissions.md §8 -->

### 4.4 Measures of fidelity and diversity

All measures are computed with the EMA weights at 60k iterations. **Fidelity** is measured on 2,000
training-seeded samples against the 800-image reference split. The *log-spectral distance* (LSD) is
the root mean square, over 43 log-spaced frequency bins, of $\log_{10}\bar P_S(b) - \log_{10}\bar P_R(b)$,
the log-ratio of sample to reference variance; it needs no pretrained network and shows where in
frequency a configuration errs. *KID* [@binkowski2018kid], unbiased at our sample sizes, is our
headline Inception measure; FID [@heusel2017fid] is reported as secondary because it is biased upward
at 800 references [@chong2020effectively‡]. *Precision* [@kynkaanniemi2019improved‡] measures
per-sample realism: the share of samples inside the real-image manifold estimated with $k = 5$
nearest neighbours. Features come from Inception-v3 via clean-fid [@parmar2022cleanfid].

**Diversity** is measured at two levels. Across images, *recall* [@kynkaanniemi2019improved‡] and
*coverage* [@naeem2020prdc] ask whether every real image has a sample nearby. Within a seed, we start
50 samples from the same prior state for each of 40 held-out subjects and measure their per-pixel
variance $D_{pix}$; as a share of the reference variance it gives the measured within-seed share
$1-I_w$, which an ideal sampler would bring to $1-I$, the variance the prior removed. The ratio
$\rho = (1-I_w)/(1-I)$ is the share of the removed variance that the chain regenerates. **Copying** is
the *seed-NN fraction*: the share of training-seeded samples whose nearest training image, in pixel
space, belongs to their own seed subject.

Two known confounds shape the reading. KID and FID mix fidelity with diversity, so we report precision
next to them. LSD responds mostly to the overall variance of the samples (Section 5.4), so a lower LSD
largely means more varied samples.
<!-- results_discussion §2a.2, §2a.4, §2a.5, §2a.6; harness H-P3 acceptance 1 and 4 -->

### 4.5 Development protocol and statistics

We developed the pipeline on problems of increasing cost before the campaign (Appendix B). Unit tests
on synthetic fixtures fixed the data, schedule and metric contracts; a local pilot of 525 iterations
fixed the batch size at 16 from a measured memory model; a 750-iteration checkpoint calibrated every
metric (for instance, the copying ratio of real reference images is 1.001); a three-epoch probe on one
A100 fixed the throughput and the sampling cost; and a smoke run of all 30 configurations on a
development GPU caught a numerical overflow. The ladder missed one failure: every pilot ended inside
the learning-rate warm-up, so the first full array diverged at learning rate $2\times10^{-4}$ and was
rerun at $10^{-4}$.

With two or three seeds per configuration we report seed means with the per-seed range, never
confidence intervals. "All seeds agree in sign" is our strongest statement, and it is not significance:
the smallest attainable permutation p-value is 0.1 with three seeds against three. KID and LSD were
pre-registered as endpoints in the project proposal; every other measure is secondary, and analyses
added after the results are labelled post hoc.
<!-- harness H-P1; ORCHESTRATOR-SESSION §12 (lesson); T1.md caption (p_min) -->

## 5 Results and discussion

### 5.1 The matched configuration on MRI, and which setting carries it

Table 1 and Figure 3 compare the four configurations. Matched against default lowers KID by 63% and
LSD by 68% on IXI, with all three seeds agreeing in sign, and FID by 51%. It also raises recall from
0.307 to 0.432 and coverage from 0.47 to 0.78, so the gain is not bought with fewer kinds of images.
On OASIS-1 the signs hold, with KID −49% and LSD −52% over two seeds.
<!-- C1; t1_main.md; results_discussion §4 table (coverage 0.467 → 0.775), §7 -->

Nearly all of the gain is the terminal blur. On seeds 1–2, +prior alone carries 90% of the matched
configuration's LSD gain and 101% of its KID gain. +spacing alone lowers LSD and KID by 11% each, both
seeds agreeing, which is about a sixth of the terminal blur's effect; on top of $W/8$ the two seeds
disagree, so we detect no effect of the spacing there. The spacing changes the schedule half as much
at $W/8$ as at $W/2$ (19 against 25 levels moved), which may explain part of this.
<!-- C2; tables/t4_decomposition.md; results_discussion §6, §2 -->

The per-octave error locates the gain (Fig. 3c). The default configuration lacks variance at 1–2 and
2–4 c/img, where the head outline and the ventricles lie: its samples carry 50% and 41% of the real
variance there. The +spacing configuration narrows the deficit; with the $W/8$ prior it disappears
($|e| \le 0.05$). The prior can hand over part of the 1–2 c/img octave, but at 2–4 c/img it keeps at
most 8.5% of a mode's variance, so there the $W/8$ chain *generates* the missing variance.
<!-- C3; F2.md spectrum table; results_discussion §4 (d_K^2 table) -->

> **Table 1** (T1, `t1_main.tex`). *The four configurations on IXI and the two on OASIS-1.* Seed mean
> (n seeds) with the per-seed range. † pre-registered endpoint. Precision and recall against the full
> 800-image reference; seed-NN: share of training-seeded samples whose nearest training image is their
> own seed subject; $D_{pix}$: within-seed per-pixel variance over 40 unseen seeds × 50 samples.

> **Figure 3** (F2, `f2_arms.pdf`, 5.5 × 3.1 in). *Matching the prior to brain anatomy drives fidelity.*
> (a) Samples of the four configurations from the same seeds and noise (two training seeds, one unseen
> subject) and the prior states at $W/2$ and $W/8$. (b) Per-sample realism (precision against R⁻, the
> reference without the seed subjects) against copying (seed-NN fraction); small markers are runs, large
> markers means; left strip: unseen subjects. (c) Per-octave error $\log_{10}$ of the sample-to-real
> variance ratio over 2,000 samples; shading: the share of variance the $W/8$ prior hands over.

### 5.2 The price: copying and within-seed diversity

The $W/8$ prior makes each sample a variant of its seed. On IXI, 72% of training-seeded samples have
their own seed subject as nearest training image, against 6% at the default (Fig. 3b); on OASIS-1, 56%
against 2%. Within a seed, samples vary less: $D_{pix}$ falls by 23% on IXI and 28% on OASIS-1. Most of
this loss (at least 90% on IXI) lies in bands the prior does not keep, so it is not the inheritance our
spectral argument predicted; our reading is that the $W/8$ chain conditions more strongly on its seed.
A data-dependent prior of this kind is also a privacy risk.
<!-- C4; t1_main.md (seed-NN 0.063 → 0.721; 0.018 → 0.557); results_discussion §5, §10 item 3 -->

### 5.3 Per-sample realism against population diversity

Fidelity measured on training-seeded samples could come from the copied anatomy. We therefore start
50 samples from each of 40 subjects held out from training (post hoc; the reading rule was fixed before
the numbers). The precision gain of $W/8$ holds: +0.098 on unseen subjects against +0.102 on training
seeds, all three seeds positive. Most of the KID gain does not: only 14% of it carries over on IXI, and
on OASIS-1 its sign flips.
<!-- C5; heldout_fidelity/README.md; results_discussion §4.1 -->

Figure 4a shows why both families of measures are needed. From 2,000 training seeds, matched against
default moves up and to the right: precision 0.655 → 0.757, recall 0.343 → 0.463. From 40 unseen seeds
it moves up and to the left: precision 0.661 → 0.759, but recall 0.238 → 0.083. The 40 seeds all come
from one slice level, which accounts for part of the drop; against a slice-matched reference recall
still falls, from 0.277 to 0.171. The population diversity of $W/8$ samples is inherited from the seed
set. In deployment those seeds are training images, which is the copying of Section 5.2.
<!-- F3.md; R5 sensitivity in F3.md -->

> **Figure 4** (F3, `f3_fidelity_diversity.pdf`, 5.5 × 2.3 in). *Seeding from many training images
> raises realism and diversity together; within a seed, every model under-disperses.* (a) Precision
> against recall (Inception, $k = 5$, against R⁻). Filled: 2,000 training-seeded samples; hollow:
> 40 unseen seeds × 50. Circles IXI, squares OASIS-1. (b) Shares of the real between-image variance:
> outline, the within-seed share an ideal sampler would regenerate ($1-I$); fill, the measured share
> ($1-I_w$, dots: runs); $\rho$ above; diamond, the total variance ratio of the training-seeded samples.

### 5.4 Every model under-disperses

Figure 4b shows a deficit common to all configurations. An ideal sampler would regenerate all of the
variance its prior removed; our models regenerate 28–45% of it ($\rho$ = 0.34 at the default and 0.28
at matched on IXI). Across all 30 runs, 227 of the 240 final per-octave errors are negative, and the
13 positive ones are at most +0.045. The total variance of the matched samples is nevertheless closer
to the real one (89% against 66% at the default), because at $W/8$ most of it comes from the 2,000
distinct seeds. LSD tracks this deficit: over all checkpoints, its mean log-variance error explains on
average 65% of the squared error. A lower LSD in our tables therefore mostly means more varied samples.
Raising the sampling noise does not cure the deficit: at $\delta = 2\sigma$ samples turn into noise (one
seed, post hoc).
<!-- C6; results_discussion §8 (227/240; level 65%), §2a.6 (rho), exploratory.json log_vr; §10 item 6 -->

### 5.5 Natural images

We also trained the default and matched configurations on LSUN Churches and Bedrooms (Appendix A). At
our budget these models do not reach usable quality (Inception precision ≤ 0.12), as expected: they
saw 3% of the images of the original LSUN model with a U-Net 2.6 times smaller, and the photographs
have a higher local intrinsic dimension than registered MRI (Section 1); data of lower intrinsic
dimension are easier to learn [@pope2021intrinsic‡]. The cross-domain comparison
planned in our proposal is therefore not reported. We did not attempt to replicate the sample quality
of [@rissanen2023ihdm].
<!-- C7; results_discussion §3, §4 table (Bedrooms A3 precision 0.120), §10 item 8; 00-framing §2 -->

### 5.6 When is IHDM the right tool?

We trained no diffusion or flow-matching baseline, so this comparison is argued, not measured. IHDM is
not the tool for unconditional sample quality: its published FIDs (18.96 on CIFAR-10) trail those of
DDPM (3.17) and flow matching (6.35) on the same benchmark [@rissanen2023ihdm; @ho2020ddpm;
@lipman2023flow‡] (cited; to verify). Its strength is an explicit, predictable knob: the terminal blur
is a known low-pass filter, so the share of a seed's variance a sample keeps can be computed before
training (0.3% at $W/2$ and 8.0% at $W/8$ on IXI; measured). On brain MRI this turns IHDM into an
anatomy-preserving sampler that produces realistic variants even of subjects it never saw (precision
gain +0.098; measured), without training a conditional model; SDEdit [@meng2022sdedit‡] obtains a
similar effect from a diffusion model through a noise level that acts on every frequency at once. The
same property rules IHDM out for synthetic data meant to be shared, since 72% of $W/8$ samples copy
their seed subject (measured). Sampling also costs 200 network evaluations per image.
<!-- C8; harness H-P4 U1, U2, N1, N2, N3. FIDs: Rissanen Tab.; Ho 2020 Tab. 1; Lipman 2023 Tab. 1 — VERIFY -->

## 6 Conclusions and future work

IHDM's natural-image defaults are not the best choice for brain MRI. Matching the terminal blur to
the shared anatomy of registered brains ($W/8$) lowers KID by 63% and LSD by 68% on IXI and keeps its
signs on OASIS-1; fitting the level spacing to the MRI spectrum adds a small gain at $W/2$ and nothing
detectable at $W/8$. The gain has a price: samples become variants of their seed, most of them copy
their seed subject, and the population-level part of the gain depends on seeding from many real
images. Every configuration under-disperses, which our spectral measure mostly reflects.

These conclusions rest on two or three seeds, a budget far below the original paper's, and one
imaging domain at usable quality. Future work: a dose–response over the terminal blur ($W/4$), a
diffusion or flow-matching baseline at the same budget, held-out seed banks for privacy, and patient
cohorts, where a prior that keeps a subject's anatomy might also keep its lesions.

---

## Appendix A — Natural-image reference runs

Text and table: `docs/RESULTS/paper/A1.md`, `a1_table.{tex,md}`; figure `a1_natural_images.pdf`
(5.5 × 1.7 in). To be ported verbatim at the LaTeX stage (about 155 words, already fact-checked in W17).

## Appendix B — Development protocol

| step | scale and cost | what it caught or decided |
|---|---|---|
| unit and integration tests | CPU, seconds; 1,439 tests | contracts of data, schedules and metrics |
| local pilot (RTX 3060) | 525 iterations at batch 4, minutes | memory model 1.705·B + 1.30 GB; batch 16 instead of the planned 128 |
| metric pilot | a 750-iteration checkpoint, 64 samples | LSD bracket (floor 0.047 against 0.78); copying ratio of real images 1.001; Inception features respond monotonically to blur on brains |
| cluster probe | 3 epochs on one A100 | batch 16 fits (28.5 GB), batch 24 does not; 1.71 it/s; resume verified; sampling cost cut the evaluation plan by 3× |
| smoke harness | all 30 configurations for a few steps, no queue | every configuration builds and trains; the fp16 overflow diagnosed |
| zero-training probes | sampling only, about 6 A100-h | a larger sampling noise does not cure under-dispersion |
| analysis-only probes | CPU | the bias of the pre-registered inherited-share estimator; fidelity from held-out seeds |
| one-factor alternatives | 2 runs, about 15 A100-h | data size and framing for the natural-image runs |

The ladder missed the fp16 instability at learning rate $2\times10^{-4}$, because every pilot stopped
inside the 1,000-step warm-up; the first 30-run array failed and was rerun at $10^{-4}$.
<!-- harness H-P1; test count: fast suite at W19 (1,439 passed) -->

## Appendix C — Deviations from the proposal

- Learning rate $10^{-4}$ instead of $2\times10^{-4}$, after the first array diverged.
- 60,000 iterations instead of 20,000: the pre-registered plateau gate at 35k → 40k triggered an
  extension; the 55k/60k gate showed no further gain.
- One natural-image run needed three attempts and carries six skipped mixed-precision steps.
- Sampling in half precision (LSD moves by $5\times10^{-5}$).
- The pre-registered inherited-share estimator is biased when the mean image is not zero; we read the
  corrected within-seed share $I_w$.
- Training compute was about twice the proposal's estimate (about 300 A100-hours).

## Author contributions (CRediT)

[TEAM TO DECIDE — the 14 CRediT roles per author.]

## Declaration regarding the use of AI tools

During the preparation of this manuscript, the authors used Anthropic Claude (Claude Code; models
Claude Opus 5.5, Claude Fable 5.1, Claude Sonnet 5 and 5.5) for coding, literature support, analysis
support and AI-assisted language editing. The use of the tool was limited to [TEAM TO WORD: e.g.
implementing modules, tests and figures to written specifications that the authors designed, reviewed
and verified; searching and verifying references; drafting text that the authors revised]. The authors
independently verified the output and retain full responsibility for the accuracy, integrity,
originality, and final content of the manuscript.
