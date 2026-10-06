# Figure plan, part 1: F1 visual abstract, F2 the four arms on MRI, T1, appendix A1 (natural images)

**Version 2, 2026-10-06**, by [Orchestrator-GenAI].

- **Why a new version.** It was redesigned after Mario's reframing (`00-framing.md`): the paper asks
  whether IHDM's natural-image defaults suit brain MRI, or whether MRI benefits from matching the
  prior and the spacing to its own properties.
- **Where version 1 is.** It is in git history (`9afcfb7`). It used LSUN Churches as a co-equal
  example in F1; version 2 makes MRI the protagonist.
- **Status.** Design only, no code. Approval turns §5 into tickets.

---

## 0. Principles

- **MRI is the protagonist.**
  - Natural images appear only as *the reference the defaults were tuned for*: one spectral
    outline and the 1/f² line in F1.
  - The trained LSUN models are in appendix A1 (≤ 3/4 page; it does not count toward the 8 pages).
- **Each figure carries one sentence.** Every number on a figure is read from a generated file at
  draw time.
- **Name the arms by what is matched.** In captions and text:
  - **default** (A0: W/2, log);
  - **+prior** (A1: W/8, log);
  - **+spacing** (A2: W/2, IXI-matched);
  - **matched** (A3: W/8, IXI-matched).

  Keep the codes in parentheses on first use and in the tables. *(Proposal; Mario to confirm.)*
- **Colour = arm, everywhere.** default blue `#2a78d6`, +prior green `#1baf7a`, +spacing amber
  `#eda100`, matched orange `#eb6834`. MRI data are greyscale, and the natural-image reference is a
  thin dashed outline.
- **Format.** Width 5.5 in; fonts ≥ 7 pt in the house sans; vector PDF with thumbnails at ≥ 300 dpi.

---

## 1. F1 — the visual abstract (MRI-centric)

### 1.1 The message (one sentence)

*IHDM generates by putting frequency bands back, coarse to fine. Its defaults, a prior that erases
everything above ~0.3 c/img and equal steps per octave, assume natural-image statistics. Brain MRI
puts its variance at mid frequencies and shares its coarse anatomy across subjects, so we match the
prior to the anatomy and the steps to the spectrum.*

### 1.2 Layout (full width, 5.5 × ~2.7 in)

```
┌───────────────────────────────────────────────────┬──────────────────────────────┐
│ (a) one frequency axis (log c/img)                │ (b) MRI's anatomical prior    │
│  IXI   [u96] [u24] [u8]  [u2]  [x]   heat states  │  mean of 3,200   subj1 subj2 subj3
│          ↓     ↓     ↓     ↓    ↓    at c≈43/σ    │  training slices              │
│  variance share per octave                        │   [mean]   W/2: [ ]  [ ]  [ ] │
│    ▇ IXI bars   ┄┄ natural-image reference        │            W/8: [ ]  [ ]  [ ] │
│    (LSUN Churches outline)  ── 1/f² (equal share) ├──────────────────────────────┤
│    shading: kept by the prior: default W/2 · W/8  │ (c) what we test              │
│  reverse steps per octave                         │        log       IXI-matched  │
│    default  ||||||||||||||||||||||||||||          │  W/2  default    +spacing     │
│    matched              |||||||||||||||           │  W/8  +prior     matched      │
│  ◄ forward: heat equation erases fine bands       │  unseen subject → default │ matched
│  ► generation: each step restores one band        │  KID −63% · seed copying 6% → 72% │
└───────────────────────────────────────────────────┴──────────────────────────────┘
          0.5     1      2      4      8     16     32     64   c/img
```

### 1.3 Panel (a): IHDM on one frequency axis (MRI only, with a natural-image reference)

**The mapping that makes one axis possible** (derived; it also goes in the Methodology box).

- **The blur.** The heat blur multiplies DCT mode $i$ by $d_i=e^{-\lambda_i t}$, with
  $t=\sigma_B^2/2$ and $\lambda_i=(\pi n_i/W)^2$. This is the released `DCTBlur`, checked on
  2026-10-06.
- **The length-scale.** A mode at $c$ c/img has $\sigma_n=\sqrt{2/\lambda}\approx 43.2/c$ px
  (W = 192), and $d=\exp(-\sigma_B^2/\sigma_n^2)$.
- **What one level does.** A level at $\sigma_B$ acts mainly near $c\approx43.2/\sigma_B$.
- **What the prior keeps.** The prior keeps half a mode's variance up to
  $c_{1/2}\approx25.4/\sigma_{B,\max}$: 0.26 c/img at W/2 (the default) and 1.06 c/img at W/8.

**The three strips:**

1. **Heat states of one IXI slice.**
   - $u(\sigma_B)$ for σ_B ∈ {96, 24, 8, 2} px, plus the image, each above
     $c = 43.2/\sigma_B$ (0.45, 1.8, 5.4 and 21.6 c/img; the image at the right end).
   - These are the chain's own intermediate states, without noise.
   - Label u96 "default prior" and u24 "matched prior".
   - The example is the data figure's IXI `ref` slice (rng 2026, slice 5).
2. **Variance share per octave.**
   - IXI as filled bars: 4.1 / 11.3 / 11.0 / 16.8 / **29.8** / 18.3 / 7.8 / 1.0%
     (`data_profile/ixi.npz` `octave_shares_train`).
   - The natural-image reference:
     - a dashed line at 12.5%, "1/f²: equal share per octave (what log spacing assumes)";
     - a thin dashed outline of LSUN Churches, "natural photographs (LSUN Churches)": 23.8 /
       21.3 / … / 3.4%.
   - The shading: $d^2(c)$ for the default prior (σ = 96) and the matched prior (σ = 24), with
     the share handed over printed in a corner, 0.3% → 8.5% of the between-subject variance
     (`inherited_train`).
3. **Reverse steps per octave.**
   - Two rugs of 200 ticks at $c_k=43.2/\sigma_k$: default (`schedules/log_W2.npy`, blue) and
     matched (`schedules/ixi_W8.npy`, orange). The level count per octave is printed above each
     rug.
   - The matched rug stops at 1.8 c/img: below that the prior provides the bands, so no step is
     spent there.

**The two direction arrows under the axis:** "forward (training): the heat equation erases fine
bands" (←) and "generation: each step restores one band" (→).

### 1.4 Panel (b): MRI's anatomical prior

- **The mean of the 3,200 IXI training slices.** One thumbnail labelled "shared anatomy: the mean
  slice is a brain".
- **Three subjects' prior states.** The subjects are the first three `ref` subjects at slice 5, a
  pre-declared choice.
  - Top row: the default prior $u(96)$. All three are near-identical blobs: the prior hands over
    0.3% of what distinguishes subjects.
  - Bottom row: the matched prior $u(24)$. Each subject's own head outline and ventricle layout
    sits on top of the shared anatomy: 8.5% of the between-subject variance.
  - Noise-free; the caption says so.
- **The wording is deliberate.** Do *not* write "the prior carries no identity". F2 shows that W/8
  samples are close copies of their seed: 72% have their own seed subject as nearest training
  image.

### 1.5 Panel (c): what we test, plus a one-line result

- **The 2 × 2 key.** Rows: the prior (default W/2, matched W/8). Columns: the spacing (log,
  IXI-matched). Cells: default, +spacing, +prior, matched, in the arm colours. This key serves
  every later figure.
- **The teaser.** One *unseen* subject (held-out seed 1, dataset index 35; pre-declared), shown as
  the seed, the default sample and the matched sample.
  - The same seed and the same noise in both arms (common random numbers verified 2026-10-06).
  - An unseen subject, so the teaser cannot be a copy of a training image.
- **Two numbers under the teaser**, with their cost made visible:
  - "KID −63% (IXI, 3 seeds)";
  - "seed copying 6% → 72%".

### 1.6 Alternatives considered

| alternative | decision | reason |
|---|---|---|
| Churches thumbnails and bars as a co-equal second dataset (version 1) | rejected | it competes with MRI for the reader's attention; one dashed outline is enough to show what the defaults were tuned for |
| the proposal's log-log radial spectrum | rejected for the main panel | the method acts per octave (the prior keeps low octaves, the spacing allocates levels per octave); bars show that directly and make the 1/f² assumption a flat line |
| ideal low-pass thumbnails | rejected | heat states are what the model actually holds |
| a teaser seeded from a training image | rejected | a W/8 sample from a training seed is close to that training image; a held-out seed shows the effect without the copy |

### 1.7 Acceptance criteria (F1)

1. **Legible at 5.5 in:** text ≥ 7 pt, thumbnails ≥ 0.35 in.
2. **Numbers from files:** every annotated number is read from a file (octave shares, inherited
   shares, level counts, the KID and seed-NN changes), never typed.
3. **The frequency mapping is verified numerically:** the octave energy that $u(\sigma)$ keeps
   equals $\exp(-2\sigma^2/\sigma_n^2)$ under `DCTBlur`.
4. **Output:** byte-stable PDF and PNG, with a caption draft in `docs/RESULTS/paper/README.md`.

---

## 2. F2 — the four arms on brain MRI

### 2.1 The message (one sentence)

*Matching the prior to MRI's anatomy drives the gain; matching the spacing adds a small spectral
correction. The prior works by handing each sample its seed's coarse anatomy, which shows as
copying on training seeds, yet the realism it buys survives on subjects the model never saw.*

### 2.2 The two orthogonal metrics

**Unchanged from version 1. Precision × seed-NN fraction.**

- **The y axis** is per-sample realism. Its gain from the matched prior holds on unseen subjects
  (96% of it, T7.5).
- **The x axis** is provenance, i.e. copying.
- **KID × seed-NN was rejected.** 86% of the matched prior's KID gain vanishes on held-out seeds,
  so KID would plot the inheritance effect twice.
- **The pre-registered KID and LSD go in T1.**
- **Stated in the caption:**
  - precision is an exploratory endpoint;
  - the spacing barely moves it (+0.006), because its effect is spectral (panel (c), T1).

### 2.3 Layout (full width, 5.5 × ~3.0 in)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ (a) same seed and same noise in every arm (IXI, run seed 1, 60k)              │
│              default prior (W/2)              │ matched prior (W/8)           │
│          seed   prior   default  +spacing     │ prior   +prior   matched      │
│ train 1  [ ]    [ ]     [ ]      [ ]          │ [ ]     [ ]      [ ]          │
│ train 2  [ ]    [ ]     [ ]      [ ]          │ [ ]     [ ]      [ ]          │
│ unseen   [ ]    [ ]     [ ]      [ ]          │ [ ]     [ ]      [ ]          │
├───────────────────────────────────────────┬───────────────────────────────────┤
│ (b) realism vs copying                     │ (c) spectrum, synthetic vs real   │
│ unseen │ precision                         │ log10 P_synth/P_real per octave   │
│ seeds  │   ■ matched  ↗ +prior              │ 0 ─────────── real ─────────      │
│ ○ ■ ▲  │   ▲ +prior                         │ default ● deep 1–4 c/img deficit  │
│ ◆ ●    │ ◆ ● default  → +spacing            │ +spacing ◆ shallower              │
│(cannot │ OASIS-1: default → matched (grey)   │ +prior ▲, matched ■: deficit gone │
│ copy)  │──────── seed-NN fraction →         │ shaded: what the matched prior keeps
└───────────────────────────────────────────┴───────────────────────────────────┘
```

### 2.4 The panels (the data and the rules carried over from version 1)

- **(a) Samples.**
  - Common random numbers were verified on 2026-10-06: the same `final/seed_idx.npy` (rng 0) and
    `heldout/seed_idx.npy` (rng 2026), batch 32, in all four IXI arms.
  - The rows follow a pre-declared rule: final-set positions 0 and 1 (images 3311 and 2488) and
    held-out seed 0 (image 5). A near-empty top or bottom slice moves the rule to the next
    position, and the skip is logged and printed in the caption.
  - The column headers group by prior: "default prior (W/2)" / "matched prior (W/8)".
- **(b) Realism against copying.**
  - The axes: y = precision against R⁻ (the `ref` split without the seed subjects); x = seed-NN
    fraction (`_results/index.csv`).
  - The marks: run markers and arm means; arrows "+prior" (default → +prior) and "+spacing"
    (default → +spacing); the parallelogram closed to "matched"; OASIS-1 default → matched as a
    grey arrow.
  - The left strip, labelled "unseen subjects (cannot copy)", gives the held-out precision.
  - The data: the T7.5 JSON for default, +prior and matched. **+spacing (A2) still needs its
    held-out evaluation:** T8.2 runs `python -m ihdm.cli.heldout_fidelity --runs
    ixi_A2_s1,ixi_A2_s2` into its own folder.
  - Expected seed means (F / H): default 0.655 / 0.661; +prior 0.700 / 0.715; matched
    0.757 / 0.758.
- **(c) The spectrum, synthetic against real.**
  - The data: IXI `final.json` `lsd_octaves` per arm (seed-mean lines and per-run dots).
  - The overlays: the shading $d^2(c)$ at σ = 24 ("kept by the matched prior"), and the labels
    "head outline 0.5–2" and "ventricles and white-matter ring 2–4".
  - Expected values: default −0.30 / −0.39 at 1–2 / 2–4 c/img; +spacing −0.24 / −0.35; +prior and
    matched about 0 there; every arm below 0 above 4 c/img.

### 2.5 Caption draft (F2)

> **Matching the prior to brain anatomy drives fidelity; matching the spacing adds a small spectral
> correction.**
>
> (a) Samples of the four configurations from the same seed images and the same sampling noise
> (IXI, run seed 1, 60k iterations; two training seeds and one unseen subject, chosen by a
> pre-declared rule).
>
> (b) Per-sample realism (Inception precision, k = 5, against the reference split without the seed
> subjects) against copying (the share of samples whose nearest training image is their own seed).
> Small markers are runs (n = 3/2/2/3) and large markers are means. The left strip gives realism on
> unseen subjects, where copying is impossible. Grey: the OASIS-1 cohort.
>
> (c) Per-octave spectral error of 2,000 samples against real images (0 = real). The shading is
> what the matched prior hands over.
>
> Precision is exploratory; the pre-registered KID and LSD are in Table 1.

### 2.6 Acceptance criteria (F2)

The criteria of version 1 are unchanged:

1. CRN and the selection rule are asserted in code.
2. Every plotted number equals its file.
3. The A2 extension uses the T7.5 CLI unchanged.
4. Legible at 5.5 in; byte-stable PDF and PNG; a caption draft.

---

## 3. T1 — the compact main table

- **Rows:** IXI default, +spacing, +prior, matched; OASIS-1 default, matched.
- **Columns:** n; KID ↓ (pre-registered); LSD ↓ (pre-registered); precision ↑; recall ↑; seed-NN ↓;
  $D_{pix}$ ↑.
- **Format:** mean, with the per-seed [min, max] below.
- **Source:** `tables.json`, read at build time.
- **Size:** ≤ 0.3 pages. F3 (part 2) may take over the recall and $D_{pix}$ columns.

---

## 4. Appendix A1 — natural-image reference runs (≤ 3/4 page, outside the 8 pages)

### 4.1 Purpose and message

- **What it says.** The same configurations were trained on LSUN Churches and Bedrooms. The
  training objective converged, but at our budget the samples do not reach usable quality.
- **What the reader learns.** What differs from the paper, and the two one-seed checks of the main
  trade-offs (data size, resolution and framing).
- **The tone** is "out of reach at this budget, as expected" (`00-framing.md` §2), not "failure".

### 4.2 Figure A1 (5.5 × ~1.9 in)

```
┌───────────────────────────────┬───────────────────────────────┐
│ (a) training loss ↓ (log)     │ (b) LSD ↓ (500 samples, same  │
│  Churches —   Bedrooms ┄      │     seeds and noise each ckpt)│
│  plateau from ~25k it.        │  Churches — oscillates 1.2–1.5│
│  ● marks: checkpoints of (c)  │  Bedrooms ┄                   │
│  x: iterations (top: epochs)  │  MRI default for scale: ·····  │
├───────────────────────────────┴───────────────────────────────┤
│ (c) one seed per dataset, its sample at 5k, 15k, 30k, 45k, 60k│
│  Churches  [seed] [5k] [15k] [30k] [45k] [60k]                 │
│  Bedrooms  [seed] [5k] [15k] [30k] [45k] [60k]                 │
└───────────────────────────────────────────────────────────────┘
```

- **(a) Training loss against iteration.**
  - The default configuration (A0) on both datasets: seed-mean thick, runs thin. A running mean
    over 1,000 iterations, on a log scale.
  - The data: `$IHDM_DATA_ROOT/_results/runs/<run>/metrics.canonical.jsonl`, the same history as
    fig. 7, with `canonical_records`.
  - A second x axis on top gives **epochs**: 3,200 images / batch 16 = 200 iterations per epoch, so
    60k iterations = 300 epochs.
  - Dots mark the five checkpoints shown in (c).
- **(b) LSD against iteration.** Why this panel is needed, even though Mario asked for the loss
  only:
  - **The loss shows the optimisation converged.** It cannot show that the *samples* did, because
    about 98% of each level's regression target is the added training noise, i.e. the denoising part (`learning/03` §6). *("Irreducible" was wrong; corrected 2026-10-06 after T8.3: the network removes most of it.)*
  - **A loss-only panel would therefore suggest a convergence the samples never reached.**
  - **One image-space curve makes the appendix honest.** The 500-sample LSD at each 5k checkpoint
    (common random numbers across checkpoints; `summary.json` `lsd_by_step`) oscillates instead of
    settling.
  - A thin dotted IXI default curve gives the scale: about 0.24 against 1.2–1.5.
- **(c) Samples across training.**
  - One seed per dataset: the first index of the frozen `eval_seeds_500.npy`. Its sample at 5k,
    15k, 30k, 45k and 60k.
  - **Source: the evaluation LSD sets in the SanDisk tars**
    (`samples_amp-fp16/<step>/lsd/samples.npy` and `seeds.npy`). The same seed image *and* the same
    sampling noise at every checkpoint, so the differences between thumbnails reflect training only.
  - **Fallback: the trainer grids** (`training/array_2475478_60k/runs/<run>/grids/iter_*.png`, one
    tile per seed). They have the same seed images and EMA weights, but the noise is not
    guaranteed fixed.

### 4.3 Table A1 — what differs from the paper's LSUN Churches model

| | Rissanen et al. (LSUN Churches 128², App. B) | ours (Churches, Bedrooms) |
|---|---|---|
| training images | ≈ 126k, RGB | 3,200, grayscale |
| framing | whole scene, 128² | native-resolution 192² centre crop |
| U-Net | (1,2,3,4,5), 2 res-blocks: 160 M parameters | (1,2,2,2), 4 res-blocks: 61 M |
| levels K | 400 | 200 |
| optimiser | lr 2e-5, batch 32 | lr 1e-4, batch 16 |
| iterations (images seen) | 1M (32M) | 60k (0.96M, 3%) |

The paper did not train on LSUN Bedrooms.

### 4.4 Text (about 120 words; draft)

> The four configurations were also trained on LSUN Churches and Bedrooms, with the recipe of §3,
> as natural-image references. Table A1 lists how this differs from the paper's LSUN model. The
> training loss plateaus after about 25k iterations (125 epochs; Fig. A1a), but the samples remain
> blurry (Fig. A1c) and the log-spectral distance oscillates rather than settles (Fig. A1b).
> Inception precision stays at or below 0.12 in every configuration, against 0.63–0.84 on MRI. Two
> one-seed checks of the main trade-offs point to resolution and framing, not data size: the
> paper's whole-scene 128² framing raises precision from 0.009 to 0.196, while ten times more
> images (32,000) leave it at 0.019. Registered brain slices have a lower intrinsic dimension than
> photographs (Sec. X), which is consistent with MRI being learnable at this budget. We therefore
> do not compare the two domains.

The numbers come from `results_discussion.md` §3, `photo_diagnostic/README.md` and T8.0.

### 4.5 Acceptance criteria (A1)

1. It fits in ≤ 3/4 page with the table and the text.
2. Every number is read from files, including the 0.12 and the 0.63–0.84.
3. The epoch axis uses 200 iterations per epoch, stated in the caption.
4. Run 11's abandoned segment is excluded through `canonical_records`. It does not affect A0,
   which is checked anyway.

---

## 5. Tickets after approval (≤ 2 agents at a time, CPU only)

| wave | ticket | agent | owns |
|---|---|---|---|
| W16 | T8.1 F1 | opus55-high | `ihdm/analysis/paper_f1.py`, `docs/RESULTS/paper/f1_*` |
| W16 | T8.2 F2 + T1 | opus55-high | `ihdm/analysis/paper_f2.py`, `docs/RESULTS/paper/{f2,t1}_*`, the A2 held-out extension |
| W17 | T8.0 intrinsic dimension + T8.3 appendix A1 | one opus55-medium | `ihdm/analysis/intrinsic_dim.py`, `ihdm/analysis/paper_a1.py`, `docs/RESULTS/{intrinsic_dimension,paper/a1_*}` |

**Before W16, main adds to the base commit:**

- `PAPER_WIDTH_IN = 5.5` and the arm display names in `ihdm/analysis/style.py`;
- the CLI stub `python -m ihdm.cli.paper_figures --fig {f1,f2,t1,a1}`.

## 6. Decisions of 2026-10-06 (Mario)

- **The template** is the NIPS 2015 style linked by the course (`nips15submit_e.sty`, text width
  5.5 in), in the Overleaf `proyecto/` (`df734f3`).
- **The arm names** default / +prior / +spacing / matched are **accepted**.
- **The unseen-subject teaser in F1** is **kept**.
- **F2(b) = precision × seed-NN**, chosen by Mario on 2026-10-06. By his rule (keep LSD in A1 only
  if LSD is one of F2's orthogonal metrics), **A1 drops the LSD panel**:
  - A1 = loss against iteration (with epochs) + thumbnails across checkpoints;
  - its caption must say that the loss is dominated by denoising, so its plateau
    does not imply that the samples converged;
  - the thumbnails carry that evidence.
- **Every paper figure is also saved as an Inkscape-editable SVG** (Mario, 2026-10-06): text kept
  as text, rasters inlined, deterministic ids. The SVG is the editable master; the PDF is the LaTeX
  input.
- **Part 2 (F3) is deferred.**
- **Wave W16:** T8.1 (F1) and T8.2 (F2 + T1). Wave W17: T8.0 (intrinsic dimension) + T8.3 (A1).

**Correction found by T8.2 (2026-10-06): the spacing's effect on precision depends on the
reference.**

- §2.2 said "the spacing barely moves precision (+0.006)". That value is against the *full* 800-image
  `ref`, the production value in T1.
- Against R⁻, the reference F2(b) uses, +spacing raises precision by **+0.025** on training seeds
  and **+0.033** on unseen subjects (seeds 1–2, both positive).
- Over the same seeds and reference, +prior raises it by about +0.057 and +0.065.
- So the paper must not say that the spacing leaves realism unchanged. Say instead: "the spacing's
  effect on precision is small, positive in both seeds, and depends on the reference: +0.006
  against the full reference, +0.025 against R⁻". Its LSD and KID effect is −11%, and only 2 seeds
  carry it.
- The F2 caption (`docs/RESULTS/paper/F2.md`) already reports the R⁻ values.

## 7. Open points for Mario (as first asked, kept for the record)

1. Do you approve **precision × seed-NN** for F2(b), with KID and LSD in T1?
2. Do you approve the **arm names** default / +prior / +spacing / matched, with the codes A0–A3 in
   parentheses?
3. **F1 teaser:** is the unseen subject (default against matched, with KID −63% and copying
   6% → 72%) OK, or do you want F1 without results?
4. **A1(b):** may the appendix keep the LSD panel beside the loss (§4.2 explains why the loss alone
   would mislead)?
5. May W16 start (T8.1, T8.2), followed by W17 (T8.0 + T8.3)?
