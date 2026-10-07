# Wave W18 — the method figure FM (T8.4), 2026-10-07

**Orchestrator:** [Orchestrator-GenAI] (Claude Opus 5.5), session 5.

**Mandate.** Mario's brief of 2026-10-07 for a didactic method figure, written into ticket T8.4
by the previous session (`ccc57a6`). Its handoff (`ORCHESTRATOR-SESSION.md` §21) said to spawn it,
review it as W16 and W17 were reviewed, merge it and copy it to Overleaf. On 2026-10-07 Mario asked
the new session to "continue with the task at hand, review the contents of the subagents and just
keep working with me until we have a draft".

## Setup

| ticket | agent | base | worktree / branch | prompt |
|---|---|---|---|---|
| T8.4 method figure FM | opus55-xhigh | `ccc57a6` | `wt/T8.4` / `ticket/T8.4-method-figure` | `wave-W18-prompts/T8.4-prompt.md` |

The agent ran on CPU only. Before spawning, main changed one line of the prepared prompt: the
`--work` scratch path, from the previous session's scratchpad to this one's.

**One mid-flight exchange.** The agent reported four assumptions, and main accepted all four with
conditions:

1. **Private helpers.** A public `octave_masks` in `paper_fm`, tested equal to
   `ihdm.spectral.power._octave_masks`. FM also gets its own inch canvas.
2. **The PDF/SVG Creator string.** It reads `ihdm.cli.paper_f1`, because the agent reuses
   `save_f1`. This is now a follow-up.
3. **6 explicit macro-steps plus an ellipsis**, because a $p_\theta$ label at 7 pt needs a pitch of
   ≥ 0.6 in.
   - The ellipsis must name what it folds, "−/+(32–96 c/img)".
   - All 8 level counts stay in the panel and the caption.
   - `stixsans` mathtext is set locally, not in `paper_style`.
4. **The prior markers** use the arm colours, as in F1, and are labelled W/2 and W/8.

## Verdict

**T8.4: ACCEPT.**

- The log file table equals `git diff --name-only ccc57a6..HEAD` (8 files); the worktree is
  clean.
- Main re-ran `test_paper_fm.py`: 53 passed. The agent's fast suite passed 1416 (1 skipped,
  27 deselected).
- A fresh CLI build in main's scratchpad is byte-identical to the committed PDF, SVG and PNG.
- **The PNG was viewed by main.** The exact-content items of the ticket all hold:
  - the representative modes lie in their octaves: (0,1), (1,1); (0,2), (2,2); …; (0,16), (16,16);
  - the forward arrows read $q(\mathbf u_k\mid\mathbf u_0)$, and the reverse arrows read
    $p_\theta(\mathbf u_{k-1}\mid\mathbf u_k)$;
  - the states sit at $\sigma_B = 43.2/c_b$ (1.35 … 86.4 px);
  - the matched prior (24 px) falls inside the −(1–2) step, and the default prior (96 px) beyond
    the last state;
  - the level counts are 31/26/26/26/27/26/26/12 and 0/5/30/38/47/42/29/9;
  - the IXI shares are 4.1/11.3/11.0/16.8/29.8/18.3/7.8/1.0%;
  - the octaves use a violet ramp, distinct from every arm colour;
  - the caption (194 words) states the macro-step, the level counts and the ideal reverse path.

## A finding raised in review (main; affects F1 and FM)

The dotted "1/f²: equal share" line in F1 (already in Overleaf) and in FM sits at 100/8 = 12.5% for
every band. That is the continuum idealisation, and it is wrong in two ways.

- **The last band is a partial octave.** In the continuum a $1/f^2$ spectrum gives each full octave
  13.2% and the 64–96 band 7.7%.
- **On the 192² DCT grid the coarsest octaves are far from flat.** A per-mode variance $\propto 1/c^2$
  puts 23.7/15.2/12.2/11.2/10.7/10.5/10.4/6.1% in the octaves 0.5–1 … 64–96, because the low
  octaves hold few modes (3, 11, 41).
  - LSUN Churches holds 23.8% in its coarsest octave, almost exactly the grid's $1/f^2$ value.
  - IXI holds 4.1%.

The visual message of both figures is unchanged: IXI departs from any $1/f^2$ reference. But the
label "1/f²" is attached to a line that does not equal the $1/f^2$ share on the grid the bars are
measured on. **[ask Mario]** whether to correct the reference line in F1 and FM, in a small
follow-up ticket.

## Merge record

| commit | content |
|---|---|
| `integration/W18` cut from `main` `20e95c3` | — |
| `f152260` | merge(T8.4) |
| `29679fc` | the T8.4 prompt pasted into its log |
| next docs commit | the paper README row; this file |

## Follow-ups

1. Add a `creator` keyword to `save_f1` and `save_paper_figure`. The FM files currently say
   `ihdm.cli.paper_f1`.
2. `ihdm.cli.paper_f1` does not catch `ScheduleError`: an out-of-range schedule ends in a
   traceback. Found by the agent while reading the code; not run.
3. The $1/f^2$ reference line, as described in the finding above.
