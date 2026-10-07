# Wave W21 — F2 as the single results figure (T8.8), 2026-10-07

**Orchestrator:** [Orchestrator-GenAI] (Claude Opus 5.5), session 5.

**Mandate.** On 2026-10-07, after W19, Mario stopped the paper draft and redesigned the results
figure. One figure replaces F2 + F3:

- the samples;
- "Fidelity vs Diversity";
- a memorisation plot;
- the spectrum with an image per octave.

Main raised two objections, and Mario chose among the options offered.

- **"Realism vs Copying" cannot be renamed "Diversity vs Fidelity".** Its x-axis is copying, which
  moves opposite to recall. Mario's choice: a real precision–recall panel, F3(a).
- **A 2-D PCA cannot show a nearest-neighbour property.** Mario's choice: a distance-to-seed plot.
- **The other choices:**
  - band-pass images per octave, not cumulative ones, which would repeat F1(a) and FM;
  - the per-slice-height prior goes to future work.

## Setup

| ticket | agent | base | worktree / branch | prompt |
|---|---|---|---|---|
| T8.8 F2 rework | opus55-xhigh | `0274cc0` | `wt/T8.8` / `ticket/T8.8-f2-rework` | `wave-W21-prompts/T8.8-prompt.md` |

**One mid-flight exchange.** Band images as wide as an octave (0.65 in) cannot be square within the
4.8 in height. Main accepted square images at the strip height (0.403 in), centred in their octave,
plus a common crop around the head (main's optional suggestion, which the agent adopted).

## Verdict

**T8.8: ACCEPT.**

- The log file table equals `git diff --name-only 0274cc0..HEAD` (10 files); the worktree is clean.
- Main re-ran the F2 and seed-distance tests: 53 passed. The agent's fast suite passed 1461.
- A cached rebuild by main is byte-identical for the PDF, SVG, PNG and `t1_main.*`. `F2.md`
  differs only in the `--out` path of the recorded command.
- **T1 and `f2_a2_heldout/` are unchanged** against the base.
- **The anchor.** The recomputed M and seed-NN equal `index.csv` for all 14 runs (|ΔM| ≤ 1.8e-6), and
  $D_{\text{ref}}$ equals `final.json` `d_heldout_median` (32.3351).
- **The PNG was viewed by main.** 5.5 × 4.8 in, three rows; Mario's two-row placement was rendered
  and rejected (it was cramped, and it split (b) across rows); Title Case is applied; the
  configurations legend is gone.

## The finding of the distance plot (post hoc; IXI seed means, units of $D_{\text{ref}}$)

| configuration | F: own seed | F: nearest other subject | H: own seed | H: nearest training image |
|---|---:|---:|---:|---:|
| default | 1.287 | 0.910 | 1.282 | 0.902 |
| matched | 0.893 | 0.940 | 0.867 | 0.927 |

The grey band (the interquartile range of real R⁻ images to the training set) is 0.936–1.078.

- **$W/8$ samples inherit their seed, and unseen seeds just as much as training seeds.** The distance to
  the own seed falls from 1.29 to 0.89 for F and from 1.28 to 0.87 for H.
- **$W/8$ does not bring samples closer to the training set.** The distance to the nearest other
  training image rises slightly: 0.91 → 0.94 for F and 0.90 → 0.93 for H.
- **The "copying" of C4 is not replication.**
  - A matched sample sits at 0.89 of the typical distance between a new real brain and its nearest
    training brain from its own seed, and at 0.94 from the nearest other subject. The 72% seed-NN
    fraction measures which of these two is smaller, not near-duplication.
  - The paper should say "samples are variants of their seed" and "their nearest training image is
    their own seed subject", and avoid "copy" without that qualification.
- **Every configuration sits slightly closer to the training set than a new real image** (0.90–0.94
  against the band's lower edge 0.936), including the default. Our reading, not tested: pixel L2 is
  two-sided, and under-dispersed, smoother samples are close to everything (as for M, §2a.5).
- **OASIS-1** (`F2.md`, not plotted): H's distance to the training set goes 0.991 → 0.925, the
  opposite direction, with both values inside its band.

## Merge record

| commit | content |
|---|---|
| `integration/W21` cut from `main` `0274cc0` | — |
| `183a45c` | merge(T8.8) |
| `fbe1337` | the T8.8 prompt pasted into its log |
| next docs commit | this file |

## Follow-ups

1. The paper draft (`docs/PAPER/paper_draft.md`, stopped) refers to F2's old panels, to Fig. 4 = F3,
   and uses "copy" loosely. It must be revised to the new F2 and to the finding above before any
   LaTeX port.
2. Overleaf still holds the old `f2_arms.pdf`.
3. Carried over: the `creator` keyword; `paper_f1`'s uncaught `ScheduleError`.
