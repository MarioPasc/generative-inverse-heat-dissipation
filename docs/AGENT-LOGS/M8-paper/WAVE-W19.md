# Wave W19 — F3, fidelity against diversity (T8.5), 2026-10-07

**Orchestrator:** [Orchestrator-GenAI] (Claude Opus 5.5), session 5.

**Mandate.** Part 2 of the figure plan, for the professor's rule 3 (harness H-P3). On 2026-10-07
Mario chose the design among three options:

- "Two panels: P–R + within-seed" (chosen);
- a forest plot of matched − default;
- no figure, a table only.

He also chose the order: F3 first, then the paper text, in markdown first.

## Setup

| ticket | agent | base | worktree / branch | prompt |
|---|---|---|---|---|
| T8.5 F3 fidelity against diversity | opus55-high | `8898dd0` | `wt/T8.5` / `ticket/T8.5-fidelity-diversity` | `wave-W19-prompts/T8.5-prompt.md` |

CPU only; no images and no eval tars. Main computed every expectation of the ticket from the
result files before spawning.

**One mid-flight exchange.** The agent reported four points.

1. **The OASIS-1 matched F expectations of the ticket were wrong.** Main had averaged per-run
   values already rounded to 3 decimals. The exact values are precision 0.834 (not 0.835) and
   recall 0.525 (not 0.526). Accepted. Main added one condition: values sitting exactly on the
   ±0.0005 bound are asserted at 4 decimals, computed by hand from the per-run values.
2. **The PDF/SVG Creator string.** It names `paper_f2`, because `save_paper_figure` hard-codes it.
   Accepted, as the same follow-up as FM's.
3. **The expectation check.** The agent wrote its own short check, because `paper_f2._check` is
   private.
4. **The denominator check passed.** `log_vr` and $I_w$ both use the non-DC $\sum P_{\text{ref}}$
   of the full `ref` split (800 images); precision and recall use R⁻ (400). The caption says so.

## Verdict

**T8.5: ACCEPT.**

- The log file table equals `git diff --name-only 8898dd0..HEAD` (8 files); the worktree is clean.
- Main re-ran `test_paper_f3.py`: 23 passed. The agent's fast suite passed 1439 (1 skipped,
  27 deselected).
- A fresh CLI build in main's scratchpad is byte-identical to the committed PDF, SVG and PNG.
- **The PNG was viewed by main.** IXI, default → matched:
  - **(a) training seeds (F):** precision 0.655 → 0.757, recall 0.343 → 0.463 (up and right);
  - **(a) unseen seeds (H):** precision 0.661 → 0.759, recall 0.238 → 0.083 (up and left);
  - **(b) within-seed diversity:**
    - ideal $1-I$: 0.997 → 0.920;
    - measured $1-I_w$: 0.342 → 0.262 (−23%);
    - ρ: 0.343 → 0.285;
    - total variance ratio of F: 0.665 → 0.885.
  - OASIS-1 reads the same, with ρ 0.449 → 0.337.
- **The caption** (199 words) states:
  - the two sample sets and the reference;
  - the H caveat (40 prior states at one slice level);
  - the seed counts;
  - the post hoc and exploratory labels.

  For the paper text, two wording details:
  - "H gains the same precision" should read "almost the same" (+0.098 against +0.102);
  - the H recall prints 0.083, which is 0.0825 rounded.
- **R5 sensitivity** (`F3.md`, not plotted): against the slice-5 reference, H recall still falls
  from default to matched, but less: IXI 0.277 → 0.171, OASIS-1 0.342 → 0.291. **The recall
  collapse is therefore partly the slice-level confound and partly real**: W/8's samples cluster
  tightly around 40 seeds.
- **Cosmetic, accepted.** Panel (a) has no colour key for +spacing and +prior; the colours are
  named on panel (b)'s axis and in F2.

## Merge record

| commit | content |
|---|---|
| `integration/W19` cut from `main` `8898dd0` | — |
| `49e0d61` | merge(T8.5) |
| `5f35f05` | the T8.5 prompt pasted into its log |
| `58eaf7e` | the paper README row; the M8 README status; this file |
| next docs commit | the suite result and the merge note |

**Full suite on `integration/W19`:** 1439 passed, 1 skipped, 27 deselected (937.84 s).

**Merged to `main`** on Mario's instruction ("Merge all work on main", 2026-10-07). It is not pushed
yet, and `f3_fidelity_diversity.pdf` is not yet in Overleaf.

## Follow-ups

1. Add a `creator` keyword to `save_f1` and `save_paper_figure`. The FM files name `paper_f1` and
   the F3 files name `paper_f2`. Carried over from W18.
2. `ihdm.cli.paper_f1` does not catch `ScheduleError`. Carried over from W18.

## Lesson

**An expectation table must be computed from unrounded per-run values.** Main averaged rounded
values and was off by 0.001; the agent caught it because the build asserts at ±0.0005.
