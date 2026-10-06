# Wave W17 — intrinsic dimension (T8.0) and appendix A1 (T8.3), 2026-10-06

**Orchestrator:** [Orchestrator-GenAI] (Claude Opus 5.5).

**Mandate.** Mario's request on 2026-10-06 to proceed with the figures of the first part, after
which W16 had delivered F1, F2 and T1. A1 belongs to part 1 (`figure-plan-part1.md` §4). T8.0
supports A1's text and `00-framing.md` §2.

## Setup

| ticket | agent | base | worktree / branch | prompt |
|---|---|---|---|---|
| T8.0 intrinsic dimension | opus55-high | `a2e0b04` | `wt/T8.0` / `ticket/T8.0-intrinsic-dimension` | `wave-W17-prompts/T8.0-prompt.md` |
| T8.3 appendix A1 | opus55-high | `a2e0b04` | `wt/T8.3` / `ticket/T8.3-appendix-natural-images` | `wave-W17-prompts/T8.3-prompt.md` |

Both ran on CPU only, with disjoint ownership and no mid-flight messages.

## Verdicts

**T8.0: ACCEPT.**

- The log file table equals the diff (9 files); main re-ran 25/25 tests; the agent's fast suite
  passed 1310.
- The scratch numbers are reproduced: MLE k = 10, N = 3,200 gives 17.18 / 22.05 / 30.10 / 30.83;
  the participation ratio 42.5 / 60.7 / 25.0 / 29.3; the components for 90% are identical.
- **A finding.** Main's scratch N-scaling percentages (+4/+15/+43/+58%) came from one random
  draw. The 10-subset means give +0/+18/+34/+49%: the same pattern.
- `00-framing.md` §2 was updated to the committed values.

**T8.3: ACCEPT, with two deviations, both accepted.**

- The log file table equals the diff (10 files); main re-ran 54/54 tests; the agent's fast suite
  passed 1339; the PNG was viewed by main.
- **Deviation 1.** `tables.json` holds no precision, so the per-cell precision comes from
  `index.csv`. The values equal exploratory X1.
- **Deviation 2.** The caption does not call the loss noise "irreducible".
  - The agent showed that the network removes most of it: the loss goes from 3.80 at step 0
    (Nσ² = 3.69) to 0.32 at the end.
  - The correct statement, also in `learning/03` §6, is that 98% of each level's target is the
    *denoising* part. The loss is dominated by denoising, so its plateau ≠ sample convergence.
  - Main's own wording was corrected in `results_discussion.md` §2 and `figure-plan-part1.md`.
    The ticket and prompt keep the old wording, as historical records.
- **Main's FIXUP at merge.** It filled the intrinsic-dimension placeholder of `A1.md` with the
  T8.0 numbers, including "linear PCA measures do not show this".

## Merge record

| commit | content |
|---|---|
| `integration/W17` cut from `a2e0b04` | — |
| `81c0946` | merge(T8.0) |
| `1542bc4` | merge(T8.3) |
| next docs commits | the prompts pasted into the logs; the A1 placeholder filled; the "irreducible" correction; the framing §2 numbers; the paper README; this file |

## Lessons

- **When a scratch number enters a plan, record whether it is a single draw.** Here it changed the
  quoted percentages, though not the conclusion.
- **Agents that check a premise against data before writing it into a caption save the paper from
  a wrong statement.** Here: "irreducible noise".
