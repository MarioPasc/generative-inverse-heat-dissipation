# Wave W15 — T7.5 fidelity from held-out seeds (2026-10-03)

**Orchestrator:** [Orchestrator-GenAI], session 4 (Claude Opus 5.5).

**The user's prompt, verbatim:**

> Proceed with (c.1) and clearly state that we were not trying to replicate the original article,
> state clearly in which things we fell short since we could not afford that much compute, but that
> we have included some experiments to check the main trade-off areas (dataset size and iamge
> resolution).

Item (c.1) refers to `results_discussion.md` §11.3, item 1: fidelity from unseen seeds.

## Setup

| item | value |
|---|---|
| base | `25e4d06`: the docs revision and the T7.5 ticket, committed by main on `main` |
| ticket | `docs/SPECIFICATIONS/M7-diagnostics/T7.5-heldout-seed-fidelity.md`, protocol and reading rule pre-registered |
| prompt | `projects/GenAI/code/wave-W15-prompts/T7.5-prompt.md`, pasted into the agent log §1 at merge |
| agent | opus55-high |
| worktree | manual: `projects/GenAI/code/wt/T7.5` |
| branch | `ticket/T7.5-heldout-fidelity` |
| compute | local CPU only, `nice 19`, 2 threads, no GPU: about 1.7 h for about 100k Inception passes |
| data | the existing evaluation tars on the SanDisk (`evaluation/eval_2488269/`) |

## Interventions

1. **The agent's question (before any contrast).** The KID fallback of the reading rule did not
   check the comparator's own KID gain, so a failing comparator would satisfy $G_H \ge 0.5\,G_F$
   trivially.
   - Ruling: apply the whole rule, rule 1 included, to KID; "not evaluable on precision or KID" if
     both comparators fail.
   - Recorded in the ticket's "Amendments", the agent log §2 and the results README.
2. **Ruling on R5.** The sensitivity reference applies to H only. F against R5 is contaminated,
   because the training seeds include the R5 images, so it is printed descriptively only.
3. **FIXUP at merge, by main.** One sentence of the results README was softened. The agent had
   written that the table-2 KID gain "is therefore in large part a property of the sampling design";
   it now reads "is consistent with …, *not tested*".

## Verdict

**ACCEPT, with that FIXUP.** The checks behind the verdict:

- **Log against the diff.** The log's file table equals `git diff --name-only 25e4d06..HEAD`: 7
  files, all inside the ownership set. `git status` was clean.
- **Tests.** Main re-ran the 38 T7.5 tests: 38 passed.
- **The agent's full fast suite:** 1223 passed, 1 skipped.
- **Independent recomputation by main.** With the cached features, main computed precision against
  its own R⁻ selection for `ixi_A0_s1` and `ixi_A3_s1`:
  - H: 0.5895 → 0.7500, Δ +0.1605;
  - F: 0.6015 → 0.7550, Δ +0.1535.

  Both deltas equal the agent's.
- **Anchors.**
  - (a) The local IXI reference features against the A100 cache: max |Δ| 8.1e-3, minimum cosine
    0.999998.
  - (b) The worst deviation is 0.0025 on the two gated runs, and 0.0050 over all 24 runs. Every KID
    lies inside its stored interval.

## Result

The pre-registered reading on IXI, with precision against R⁻, is **"the fidelity gain generalises
to unseen seeds"**:

- $G_H$ = +0.0975 against $G_F$ = +0.1018;
- the per-seed $\Delta_H$ are +0.1605, +0.0650 and +0.0670.

KID mostly does not carry over: only 14% of the gain on IXI, and the sign flips on OASIS-1 against
R⁻. This is consistent with W/8 samples clustering on their 40 prior states, which is untested.
R5 favours A3 on every seed. Integrated into `results_discussion.md` §4.1, with the consequences
in §1, §9, §10 item 9 and §11.

## Merge record

| commit | content |
|---|---|
| `integration/W15` cut from `25e4d06` | — |
| `575231f` | merge(T7.5), `--no-ff`, no conflict |
| `ba66a2b` | the prompt pasted into the agent log |
| next docs commit | the FIXUP, the ticket amendment, `results_discussion.md` §4.1 and this file |

Merging `integration/W15` into `main` waits for Mario's approval.

## Lessons

- **Pre-register the fallback branches of a reading rule as carefully as the main branch.** The
  agent found the hole before any number existed, which is the only time it could be fixed
  cleanly.
- **On held-out-seeded sets with few prior states, use per-sample endpoints.** Precision and
  density are not distorted by those structural features; distribution metrics (KID, recall) are.
  This choice turned out to matter: KID and precision disagree.
