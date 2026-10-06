# Wave W16 — paper figures F1 and F2, table T1 (2026-10-06)

**Orchestrator:** [Orchestrator-GenAI] (Claude Opus 5.5).

**The user's prompt, verbatim:**

> For now, change the template to NeurIPS 2015 and proceed with the figure generation of the
> graphical abstract and the first part. Let's not dive onto the second part for now, so we skip 1,
> i accept 2, we keep the unseen subject teaser 3, we keep the LSD on the loss panel only if it
> really matters and we have included it as one of the orthogonal metrics 4.

**Later in the same turn:**

> Also! for the visual abstract, tell the agent via message that it should save it in a
> inkscape-compatible format, so that I can modify it if I want to.

**Answered by AskUserQuestion:** F2(b) = "Precision × copying (Recommended)".

## Setup

| ticket | agent | base | worktree / branch | prompt |
|---|---|---|---|---|
| T8.1 F1 | opus55-high | `cc819fc` | `wt/T8.1` / `ticket/T8.1-visual-abstract` | `wave-W16-prompts/T8.1-prompt.md` |
| T8.2 F2 + T1 | opus55-high | `9b00297` (= `cc819fc` + the T8.2 ticket, docs only) | `wt/T8.2` / `ticket/T8.2-arms-figure` | `wave-W16-prompts/T8.2-prompt.md` |

Both ran on CPU only, with disjoint file ownership. The template switch to NIPS 2015 was done by
main in the Overleaf clone (`df734f3`).

## Interventions

1. **T8.1, mid-flight.** Also save an Inkscape-editable SVG: `svg.fonttype = "none"`, rasters
   inlined, deterministic ids.
2. **T8.2, mid-flight.** The unchanged T7.5 CLI rejects A2.
   - The agent proposed extending `ARMS` at run time.
   - Accepted on three conditions: a module subcommand (`paper_f2 a2-heldout`) instead of
     `python -c`; the cache's `tables.md` and `anchors.json` checked unchanged by sha256; the same
     R⁻ sha256.
   - The agent reports that a first run, before the wrapper, rewrote the cache's `tables.md`. It
     restored it from backup and verified the sha256.
3. **T8.1, RETURN (review round 1).**
   - The rug counts were per σ_B octave, so they were misaligned with the frequency-octave bars;
     they are now per frequency octave.
   - The caption title now names the two assumptions.
   - The panel (c) label now reads "IXI-matched".
   - Fixed in `3cdd1ab`, then re-reviewed by main and accepted.

## Verdicts

**T8.1: ACCEPT** (after RETURN).

- The log file table equals the diff (8 files).
- 32/32 tests re-run by main.
- The PNG was viewed twice.
- The SVG has 79 `<text>` and 15 `<image>` elements.
- The agent's fast suite: 1251 passed.

**T8.2: ACCEPT.**

- The log file table equals the diff (15 files).
- 31/31 tests re-run by main.
- The PNG was viewed.
- The OASIS-1 arrow was checked against `F2.md`: 0.705 → 0.834, copying 2% → 56%.
- T1 was checked against `results_discussion.md` §4 (all IXI and OASIS-1 values).
- The agent's fast suite: 1254 passed.

## Findings worth keeping

- **The spacing's precision effect depends on the reference:** +0.006 on the full `ref`, against
  +0.025 (F) and +0.033 (H) on R⁻, both seeds positive. The plan wording was corrected
  (`figure-plan-part1.md` §6).
- **The matched schedule spends its steps where IXI's variance is.** Per frequency octave, its rug
  peaks at 8–16 c/img (47 levels), the octave holding 29.8% of IXI's variance. The default schedule
  stays flat at 26–27 levels per octave.

## Merge record

| commit | content |
|---|---|
| `integration/W16` cut from `9b00297` | — |
| `42e74a8` | merge(T8.1) |
| `faca324` | merge(T8.2) |
| `3c0c981` | the prompts pasted into the logs |
| next docs commit | the paper README, the plan correction, this file |

The full fast suite on `integration/W16`: see the docs commit message and the session log.
