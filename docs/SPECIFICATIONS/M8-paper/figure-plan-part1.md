# Figure plan, part 1: the visual abstract (F1) and the A0–A3 figure (F2)

Written 2026-10-06 by [Orchestrator-GenAI] for Mario. Design only: no code yet. Approval turns §4
into tickets T8.1 and T8.2.

Sources:

- the results: `docs/RESULTS/results_discussion.md`;
- the metric definitions: §2a there;
- the course rules: `projects/GenAI/project/HARNESSES/`.

---

## 0. Principles, and the decision to make the main text about MRI

**The page has room for three figures** (F1–F3) and one compact table (T1). Each figure must carry
a message a reader can state in one sentence after looking at it for ten seconds, and every number
on a figure must come from a generated file.

**Mario's question (2026-10-06):** "If we got so much worse results on the churches dataset, why
don't we compare for only the MRI images?"

**Resolved on 2026-10-06 (Mario):** the paper is framed as "are the natural-image defaults right
for MRI, or does MRI benefit from matching them to its own spectrum and anatomy?" (see
[`00-framing.md`](00-framing.md)).

1. **The MRI comparison is a complete design on its own.** The four arms on IXI form a 2 × 2
   factorial (terminal blur W/2 or W/8 × spacing log or IXI-matched), with 3/2/2/3 seeds. It
   answers "does each MRI-matched change help, against the natural-image default?". OASIS-1 (A0,
   A3) adds an independent cohort.
2. **The natural-image runs** are out of reach at our budget, as expected (`00-framing.md` §2:
   compute, intrinsic dimension, framing). They appear as 2–3 sentences and one appendix table,
   *not* as a failure figure. The proposal's cross-domain interaction is stated as not reported.
3. **Churches stays in F1 as the natural-image reference for the data.** The figure contrasts its
   spectrum and its prior with MRI's, which is exactly what the defaults were tuned for, and needs
   no trained model.

**Colour, size and type.**

- Colour = arm, everywhere (A0 blue, A1 green, A2 amber, A3 orange; `ihdm/analysis/style.py`).
  Datasets are greyscale or hatched.
- Width 5.5 in; fonts ≥ 7 pt, in the house sans (DejaVu Sans).
- Vector PDF, with image thumbnails embedded at ≥ 300 dpi.

---

## 1. F1 — the visual abstract

### 1.1 The message (one sentence)

*IHDM generates an image by putting frequency bands back, coarse to fine. The prior decides how many
bands it is handed, and the level spacing decides where it spends its steps. Both defaults were
tuned on natural photographs. Brain MRI places its variance at other scales and shares its coarse
anatomy across subjects, so we match both settings to MRI.*

### 1.2 Layout (full width, 5.5 × ~2.7 in)

```
┌──────────────────────────────────────────────────────┬─────────────────────────────┐
│ (a) one frequency axis for everything (log c/img)    │ (b) the prior: what the     │
│                                                      │     sampler starts from     │
│  IXI   [u96] [u24] [u8]  [u2]  [x]   ← heat states   │       x   W/2   W/8   mean  │
│  LSUN  [u96] [u24] [u8]  [u2]  [x]     at c≈43/σ     │ IXI  [ ]  [ ]   [ ]   [ ]   │
│          ↓     ↓     ↓     ↓     ↓                   │ LSUN [ ]  [ ]   [ ]   [ ]   │
│  ───── variance share per octave (bars: IXI ▇,       ├─────────────────────────────┤
│        LSUN ░) ── 1/f² line (equal per octave) ──    │ (c) the four arms (2 × 2)   │
│        shaded: what the W/2 and W/8 priors keep (d²) │            log    IXI-match │
│  ───── reverse steps per octave: A0 rug ||||||||||   │   W/2      A0      A2       │
│                                  A3 rug    ||||||||  │   W/8      A1      A3       │
│  ◄── forward: heat equation erases fine bands        │ ∂u/∂t = Δu                  │
│  ──► generation: each step restores one band         │ û_i(t) = e^{-λ_i t} û_i(0)  │
└──────────────────────────────────────────────────────┴─────────────────────────────┘
       0.5      1       2       4       8      16      32      64  c/img
```

### 1.3 Panel (a): one frequency axis carries the image, the spectrum, the prior and the steps

**The mapping that makes one axis possible** (derived; put it in the caption or the Methodology).

- **The blur.** The heat blur multiplies DCT mode $i$ by $d_i = e^{-\lambda_i t}$, with
  $t=\sigma_B^2/2$ and $\lambda_i=(\pi n_i/W)^2$.
- **The length-scale.** Write $\sigma_n=\sqrt{2/\lambda}$; a mode at $c$ cycles per image has
  $n = 2c$, so

  $$\sigma_n=\frac{\sqrt2\,W}{2\pi c}\approx\frac{43.2}{c}\ \text{px}\quad(W=192),\qquad d=\exp\!\big(-\sigma_B^2/\sigma_n^2\big).$$

- **What one level does.** A level at $\sigma_B$ acts mostly on modes near $c \approx 43.2/\sigma_B$,
  where $d = e^{-1}$.
- **What the prior keeps.** The prior at $\sigma_{B,\max}$ keeps half of a mode's variance
  ($d^2 = \tfrac12$) up to

  $$c_{1/2}=\frac{43.2}{\sigma_{B,\max}\sqrt{2/\ln 2}}\approx\frac{25.4}{\sigma_{B,\max}}.$$

  - W/2 (96 px): $c_{1/2} = 0.26$ c/img, i.e. nothing on the 0.5–96 axis ($d^2$ = 0.085 at
    0.5 c/img).
  - W/8 (24 px): $c_{1/2} = 1.06$ c/img; $d^2$ = 0.86, 0.54, 0.085 and 0.008 at 0.5, 1, 2 and
    2.8 c/img.

**The three strips:**

1. **Heat states** (two rows: one IXI image, one Churches image).
   - The image blurred to $\sigma_B$ ∈ {96, 24, 8, 2} px, plus the image itself.
   - Each thumbnail sits above its $c = 43.2/\sigma_B$: 0.45, 1.8, 5.4 and 21.6 c/img, with the
     full image at the right end.
   - Arrows run down to the axis.
   - **These are the chain's actual intermediate states** $u_k$ without noise, not ideal
     low-passes. This answers "how the image looks up to that point" with the model's own operator.
   - u96 is the A0 prior and u24 the A3 prior; label both.
2. **Variance share per octave.**
   - Bars of the 8 octave shares (`docs/RESULTS/data_profile/<dataset>.npz`,
     `octave_shares_train`):
     - IXI: 4.1 / 11.3 / 11.0 / 16.8 / **29.8** / 18.3 / 7.8 / 1.0%;
     - Churches: **23.8** / 21.3 / 16.1 / 12.2 / 9.6 / 7.5 / 6.1 / 3.4%.
   - A dashed line at 12.5% reads "1/f²: the same share in every octave, what log spacing assumes".
   - Two light fills show $d^2(c)$ for σ = 96 and σ = 24, the "prior keeps" regions.
   - A corner annotation gives the share handed over:
     - IXI: 0.3% (W/2) → 8.5% (W/8);
     - Churches: 1.8% → 29.1% (`inherited_train`, `data_profile.md` §3).
   - This shows the "variance between octaves" Mario asked for, and why log spacing fits neither
     dataset exactly.
3. **Reverse steps per octave.**
   - Two rugs of 200 ticks at $c_k = 43.2/\sigma_k$, one for A0 (`schedules/log_W2.npy`) and one
     for A3 (`schedules/ixi_W8.npy`), in the arm colours.
   - The level count per octave is printed in small type above each rug (`data_profile.md` §5).
   - The A3 rug stops near 1.8 c/img: below that, the prior hands the bands over.
   - **This is where "what we introduce" becomes visible:** a shorter chain that starts with the
     coarse anatomy, and levels moved toward 2–8 px.

**Two direction arrows under the axis:** "forward (training): the heat equation erases fine bands"
(←) and "generation: each reverse step restores the band between two levels" (→).

### 1.4 Panel (b): the prior, and the inherent anatomy of MRI

- **The layout.** 2 rows (IXI, Churches) × 4 columns:
  1. an example image $x$;
  2. its W/2 prior $u(96)$;
  3. its W/8 prior $u(24)$;
  4. **the mean of the 3,200 training images**.
- **The examples.** The same `ref` examples as the data figure (rng 2026; MRI on slice 5). The
  priors are shown without the δ-noise and labelled "noise-free".
- **The message.**
  - The mean brain *is* a brain: registration gives MRI a shared anatomy.
  - The mean church is a sky-over-ground gradient.
  - At W/2 every prior is a featureless blob.
  - At W/8 the brain's prior hands over its head outline, only 8.5% of what makes subjects
    different; the church's prior hands over its scene layout, 29%.

### 1.5 Panel (c): the four arms

- A 2 × 2 table (rows: terminal blur W/2 or W/8; columns: spacing log or IXI-matched) with A0–A3 in
  the arm colours. This key serves every later figure.
- Underneath, two lines of equations: the heat equation, and the inherited share
  $I = \sum_i d_i^2P_i/\sum_iP_i$.

### 1.6 Alternatives considered

- **The proposal's log-log radial spectrum** (Fig. 1 there) instead of octave bars: rejected for
  the main panel.
  - On log-log axes the MRI/photograph difference reads as a slope (α 3.2 against 2.3).
  - The quantity the method uses is variance *per octave*: the prior keeps the low octaves, and the
    spacing allocates levels per octave.
  - The bars show that directly, and the 1/f² assumption becomes one flat line.
  - The log-log curve can stay as a thin inset if space allows.
- **Ideal low-pass thumbnails** instead of heat states: rejected. Heat states are what the model
  actually sees, and they tie the figure to the method.
- **All four rugs** (A0–A3): rejected for F1, as too dense. Only A0 against A3 (the paper's default
  against ours), with A1/A2 in panel (c).

### 1.7 Acceptance criteria (F1)

1. It is legible at 5.5 in on paper: every text ≥ 7 pt and every thumbnail ≥ 0.35 in.
2. Every annotated number is read from a file at draw time, never typed: the octave shares, the
   inherited shares, the level counts.
3. The frequency mapping is verified numerically. For the IXI example, the octave energy that
   $u(\sigma)$ keeps must match $\exp(-2\sigma^2/\sigma_n^2)$, using the released `DCTBlur`
   operator.
4. The output is byte-stable vector PDF plus PNG, with a caption draft in
   `docs/RESULTS/paper/README.md`.

---

## 2. F2 — A0–A3 on brain MRI

### 2.1 The message (one sentence)

*On brain MRI the terminal blur, not the spacing, drives sample quality. It works by handing each
sample its seed's coarse anatomy, which shows as copying on training seeds, yet the realism it buys
survives on subjects the model never saw.*

### 2.2 Choosing the two orthogonal metrics

Mario asked for two orthogonal metrics, "orthogonal as in they measure different things", not a
panel of metrics.

| pair (y × x) | measures | verdict |
|---|---|---|
| KID × seed-NN fraction | the population match (fidelity *and* diversity) × copying | ✗ Not orthogonal in practice. T7.5 shows that 86% of W/8's KID gain disappears on held-out seeds, so the KID gain is largely the inherited diversity of 2,000 training seeds, the same inheritance that drives copying. The two axes would show one effect twice. |
| LSD × seed-NN | the spectral variance match × copying | ✗ LSD is mostly dispersion (the broadband level carries a mean 65%, median 71%, of its square, §8), again inheritance-driven. Keep it for panel (c), where its *shape* is informative. |
| precision × recall | fidelity × diversity | reserved for F3 (the professor's rule 3) |
| **precision × seed-NN fraction** | **per-sample realism × provenance (copying)** | ✓ **Chosen.** Realism and copying are different constructs. Precision's W/8 gain is shown not to depend on copying: 96% of it holds on held-out seeds, where copying a training image is impossible (T7.5). It maps directly onto H1's trade-off. |

**The costs of this choice, to be stated in the caption:**

- **Precision was exploratory** in the main experiment; the pre-registered fidelity headline was
  KID. KID and LSD go to T1, so nothing pre-registered is hidden.
- **The spacing barely moves precision** (A2 − A0 = +0.006). Its effect is spectral (−11% LSD and
  KID) and is shown in panel (c). This is a finding, not a defect of the figure: the spacing changes
  the *spectrum* of the samples, not their per-sample realism.

### 2.3 Layout (full width, 5.5 × ~3.0 in)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ (a) samples, same seed and same noise in every arm (common random numbers)    │
│          seed   W/2 prior   A0     A2   │  W/8 prior   A1     A3              │
│ train 1  [ ]      [ ]       [ ]    [ ]  │    [ ]       [ ]    [ ]             │
│ train 2  [ ]      [ ]       [ ]    [ ]  │    [ ]       [ ]    [ ]             │
│ held-out [ ]      [ ]       [ ]    [ ]  │    [ ]       [ ]    [ ]             │
├─────────────────────────────────────────┬─────────────────────────────────────┤
│ (b) fidelity vs memorisation             │ (c) where the variance is wrong    │
│ held-out │  precision                    │  log10 P_synth/P_real per octave   │
│  seeds   │    ▲ A3 ■   ↗ terminal blur   │  0 ─────────────── real ────────   │
│  ○ ■ ▲   │    ▲ A1   ↗                   │  A0 ●── deep 1–4 c/img deficit     │
│  ◆ ●     │  ◆ A2 ● A0  → spacing         │  A2 ◆── shallower                  │
│ (no seed │──────────────────────── seed  │  A1 ▲, A3 ■ ── deficit gone         │
│  to copy)│      NN fraction (copying) →  │  shaded: W/8 prior hand-over (d²)  │
└─────────────────────────────────────────┴─────────────────────────────────────┘
```

### 2.4 Panel (a): samples

- **What it shows.** For each row, the seed image, the two prior states (W/2, W/8, noise-free),
  and the sample of each arm. Run seed 1 for all four arms.
- **Common random numbers were verified on 2026-10-06.** The four IXI arms share the same
  `final/seed_idx.npy` (rng 0, batch 32) and `heldout/seed_idx.npy` (rng 2026, batch 32). So
  "sample $i$" differs between arms *only* by the arm.
- **Selection rule, pre-declared so the rows are not cherry-picked:**
  - training rows: the samples at positions 0 and 1 of the final set (seed images 3311 and 2488);
  - held-out row: sample 0 of held-out seed 0 (image 5);
  - if a pre-declared seed is a top or bottom slice with almost no brain, the rule moves to the
    next position, and the skip is printed in the caption.
- **Reading.**
  - Within a row the W/2 columns differ from the seed (new anatomy; fig. 6 shows they even change
    slice level).
  - The W/8 columns reproduce the seed's slice level, head shape and ventricles.
  - The held-out row shows the same behaviour for a subject the model never saw.

### 2.5 Panel (b): the fidelity–memorisation plane

- **The axes.** y: precision against R⁻ (the `ref` split without the 40 seed subjects), the same
  reference on both sets. x: seed-NN fraction of the training-seeded final set
  (`seed_nn_fraction`).
- **The points.** One small marker per run and one large marker per arm mean (n = 3/2/2/3 on IXI).
  - Arrows from A0 to A1 (the terminal blur alone) and from A0 to A2 (the spacing alone).
  - Thin lines complete the parallelogram to A3; an additive effect is a parallelogram.
  - OASIS-1, A0 → A3, is a grey arrow (transfer).
- **The held-out inset.** A narrow strip on the left, sharing the y axis, carries the
  held-out-seed precision of each arm.
  - It is labelled "held-out seeds: no training seed to copy", because the seed-NN fraction is 0
    by construction there.
  - Reading: A3 sits at the same height in the strip as in the plane. Its realism does not come
    from copying.
- **The data.**
  - `docs/RESULTS/heldout_fidelity/heldout_fidelity.json` for A0, A1 and A3 (F and H against R⁻);
  - `$IHDM_DATA_ROOT/_results/index.csv` for the seed-NN fraction;
  - **a required extension:** T7.5 did not evaluate A2. T8.2 runs
    `python -m ihdm.cli.heldout_fidelity --runs ixi_A2_s1,ixi_A2_s2` into its own output folder,
    reusing the cached reference features. The T7.5 outputs stay untouched. This is about 10 CPU
    minutes.
- **Values to expect** (seed means, from T7.5, against R⁻):

  | arm | precision, F | precision, H | seed-NN |
  |---|---|---|---|
  | A0 | 0.655 | 0.661 | 0.063 |
  | A1 | 0.700 | 0.715 | read from `index.csv` |
  | A3 | 0.757 | 0.758 | 0.721 |
  | A2 | still to compute | still to compute | read from `index.csv` |

  If the drawn values differ, the figure is wrong.

### 2.6 Panel (c): the spectrum of real against synthetic

This is the "power spectrum of real and synthetic images" Mario suggested, drawn as a *ratio* to
the real spectrum.

- **Why a ratio.** On raw log-log axes the arms overlap within a line width. The ratio
  $\log_{10}\bar P_{\text{synth}}/\bar P_{\text{real}}$ per octave makes a 0.1-decade difference
  visible, and real = 0 is a single line.
- **The data.** IXI, the four arms: seed means as lines with markers, and single runs as dots, from
  `final.json` `lsd_octaves` (the final 2,000-sample set). This is fig. 2's IXI panel, restyled.
- **The overlays.**
  - The W/8 hand-over $d^2(c)$ for σ = 24 as a light fill (≥ 0.5 below 1.06 c/img, ≥ 0.05 below
    2.2).
  - Two anatomical labels: "head outline 0.5–2" and "ventricles and white-matter ring 2–4".
- **Reading.**
  - A0 has a deep deficit at 1–4 c/img (−0.30, −0.39): its samples are too alike in head shape and
    ventricles.
  - A2 shallows it (−0.24, −0.35); this is the spacing's spectral effect.
  - A1 and A3 remove it.
  - Above 4 c/img every arm stays too low (the broadband under-dispersion of F3).

### 2.7 Caption draft (F2)

> **The terminal blur, not the spacing, drives fidelity on brain MRI, by handing each sample its
> seed's coarse anatomy.**
>
> (a) Samples of the four arms from the same seed images and the same sampling noise (IXI, run
> seed 1, 60k iterations; two training seeds and one held-out seed, chosen by a pre-declared rule).
>
> (b) Per-sample fidelity (Inception precision, k = 5, against the reference split without the
> seed subjects) against copying (share of samples whose nearest training image is their own
> seed). Small markers are runs (n = 3/2/2/3) and large markers are arm means. The left strip
> gives precision on held-out seeds, where copying is impossible. OASIS-1 (grey) is the transfer
> cohort.
>
> (c) Per-octave spectral error of 2,000 samples against the reference (0 = real). The shaded band
> is what the W/8 prior hands over.
>
> Precision is an exploratory endpoint; the pre-registered KID and LSD are in Table 1.

### 2.8 Acceptance criteria (F2)

1. **CRN and the selection rule:** the CRN check and the selection rule are asserted in code, and
   any skipped seed is logged and printed in the caption.
2. **Numbers:** every plotted number equals its source file. The A0, A1 and A3 precisions equal
   T7.5's JSON to every printed digit.
3. **A2:** the extension uses the T7.5 CLI unchanged, and its outputs are listed in the README.
4. **Legibility and output:** legible at 5.5 in; thumbnails ≥ 0.45 in; byte-stable PDF and PNG;
   caption draft in `docs/RESULTS/paper/README.md`.

---

## 3. T1 — the compact main table (the numbers that F2 does not plot)

- **Rows:** IXI A0, A2, A1, A3; OASIS-1 A0, A3.
- **Columns:** n; KID ↓ (pre-registered headline); LSD ↓ (pre-registered H2); precision ↑;
  recall ↑; seed-NN ↓; within-seed diversity $D_{pix}$ ↑.
- **Format:** mean, with the per-seed [min, max] below.
- **Source:** `docs/RESULTS/tables/tables.json`, read at build time, never typed.
- **Overlap with F3.** T1 must stay ≤ 0.3 pages. F3 (part 2) may take over the recall and $D_{pix}$
  columns, in which case T1 drops them.

---

## 4. Tickets to write after approval (wave W16, 2 agents, CPU only, no Picasso)

| ticket | agent | owns | inputs |
|---|---|---|---|
| T8.1 F1 | opus55-high | `ihdm/analysis/paper_figures.py` (the F1 functions), `docs/RESULTS/paper/f1_*` | `data_profile/*.npz`; `schedules/*.npy`; `$IHDM_DATA_ROOT/{ixi,lsun_church}/images.npy` and `splits.json`; the `DCTBlur` operator |
| T8.2 F2 + T1 | opus55-high | a separate module `ihdm/analysis/paper_f2.py` (to keep file ownership disjoint), `docs/RESULTS/paper/f2_*` and `t1_*` | the SanDisk evaluation tars (final and held-out members of the IXI run-seed-1 arms); the T7.5 JSON and cached features; `_results/index.csv`, `final.json`, `tables.json` |

- **Shared:** `ihdm/analysis/style.py` gains `PAPER_WIDTH_IN = 5.5`. That is the only shared edit;
  main makes it before the wave, on the base commit.
- **CLI:** `python -m ihdm.cli.paper_figures --fig {f1,f2,t1} --out docs/RESULTS/paper/`. Main
  writes the CLI stub before the wave; each agent fills its own branch of it.
- **Rules:** one heavy process at `nice 19`, no GPU.

## 5. Open points for Mario (please answer in one go)

1. ~~MRI-only Results~~: resolved on 2026-10-06 by the reframing (§0, `00-framing.md`).
2. Do you approve **precision × seed-NN** for F2(b), with KID and LSD in T1 (§2.2)?
3. F1 keeps **Churches as a data contrast**. Are one IXI and one Churches example enough for the
   heat-state strip, or do you want OASIS-1 as well?
4. Is the **selection rule** for the F2(a) rows acceptable, or do you prefer a random draw with a
   fixed seed?
