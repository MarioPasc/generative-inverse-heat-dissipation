# Wave W14 — post-hoc diagnostics (M7), 2026-10-01

Orchestrator: Claude Opus 5.5 (session 3, `[Orchestrator-GenAI]`). Skills: `parallel-agents`,
with `picasso-sbatch` for the agents. At most 2 agents at once, counting reviewers (Mario,
2026-10-01). The orchestrator does not write code.

## User prompts (verbatim)

```
CONTEXT: You are going to act as an expert in AI and workflows for our generative AI project. You must first gather all the context needed to work on this, for that, read the handoff from the previous agent: projects/GenAI/code/docs/AGENT-LOGS/ORCHESTRATOR-SESSION.md and all the related files. Specifically, you should read our proposal projects/GenAI/project/6aab9fca47dc3902a0dbfcef/propuesta/templateArxiv.tex and check the actual stage of the code. TASK: Your task will mainly be to keep iterating until the completion of the project from the handoff that the previous orchestrator. You don't have to write the final project file. But rather, I expect a md document where you expose the metrics and discuss them. Its ok if the results are slighly negative or negligible, this is a course project, not a research paper. ACCEPTANCE CRITERIA: You have thought step by step, you have checked facts before taking any decision. Iterate for as long as you need.
```
```
One important thing: Should we be seeing good results for lsun in the A0 condition? Since that is like replicating the paper, essentially
```
```
I think it'd be correct to launch a job in Picasso that takes the A0 setting (the one that is a "copy" from the paper) and train with 64², 128², ... resolutions for lsun churches, if the 128² setting (the paper one) yields correct results (the ones from the paper), then we can say that the "bad quality" of the 192² run in /media/mpascual/Sandisk2TB/research/spectral_allocation_heat_diffusion_project/results/runs/lsun_church_A0_s1 is because of the upgraded image resolution. Don't you think the same? Maybe I can remove the results from Picasso, if they are copied correctly locally, to free up space.
```
Answers to the orchestrator's questions: "Two-factor diagnostic (Recommended)"; δ sweep "Yes, run it".
```
Also, don't code anything yourself, using /parallel-agents and /picasso-sbatch spawn at most 2 agents at a time and prompt them really carefully. You are supposed to review their work. :-)
```

## Before the wave

- **Results write-up.** `docs/RESULTS/results_discussion.md`, commits `48363e2` → `9d02e0c`. An
  independent fact-check agent (opus55-xhigh, read-only) found these errors in the first draft:
  - the H1d inheritance reading;
  - the 2–4 c/img "handed over" claim;
  - "a tenth" for the spacing's share;
  - three dispersion claims;
  - the compute totals;
  - the cause of the stale proposal numbers.

  Main re-verified each one before correcting the document.
- **Exploratory module.** `ihdm/analysis/exploratory.py` (`93785d7`) was written by main before
  Mario's no-coding instruction. The fact-check agent reviewed it and found no correctness bug.
  Follow-up for an agent:
  - the tests do not check `add_endpoints` values or the curve statistics;
  - `main` does not catch ValueError from a non-finite octave error or a variance ratio of 0;
  - the "low" band 0.5–4 c/img includes 2–4 c/img, which a σ = 24 prior barely carries
    ($d_K^2$ ≈ 0.008 at 2.8 c/img). Split it at 2 c/img or relabel it.
- **Picasso copies verified** against the SanDisk by file name and size, plus 3 sha256 spot
  checks:
  - runs: 1,712 files, 236 GB;
  - results: 696 files;
  - eval tars: 73 files.

  Mario has the cleanup command; it keeps the 4 run directories used by T7.2 and T7.3.

## Wave record

| ticket | agent | model / effort | branch / worktree | base | verdict | merged |
|---|---|---|---|---|---|---|
| T7.1 photo diagnostic | `aea7e4dee451cf398` | opus55-xhigh | `ticket/T7.1-photo-diagnostic` / `wt/T7.1` | `48363e2` | pending | — |
| T7.2 δ override + sweep (first attempt) | `a88cefa46e144a780` | opus55-high | `ticket/T7.2-delta-sweep` / `wt/T7.2` | `48363e2` | stopped twice by an API safeguard flag ("reasoning_extraction"); nothing committed | — |
| T7.2 (redo) | `ad3b28ec290ce214c` | opus55-high | same branch and worktree | `48363e2` | pending | — |

The exact prompts are in `projects/GenAI/code/wave-W14-prompts/` (git-ignored in the TFM repo).
Main pastes them into each log's §1 at merge time.

## Interventions

1. **T7.2 design assumptions accepted, with conditions.**
   - W rule: bands fixed in cycles per image, with an explicit W < 128 legacy branch kept only for
     the 96² test fixture.
   - δ-tagged trees.
   - `--final-from-lsd`, which brings the sweep to ≈ 7.3 A100-h.
   - `paired_lsd_gate` at W ≠ 192 recorded as a follow-up.
2. **T7.2 restarted.** The first agent was interrupted while writing the verbatim prompt into its
   log, then again on resume. The redo prompt carries the approved decisions. Both agents now
   write "Prompt: pasted by main at merge time." in §1.
3. **T7.1: `lsun_church_n32k` holds 32,800 images, not 32,840.** The seed split is a subset of
   ref; the ticket was amended in `18c9daa`. Indices 0–3999 equal `lsun_church`; the train shard
   has 119,915 rows.
4. **T7.1: near-duplicate leakage.**
   - 6 of the 800 ref images have re-encoded near-copies among the new train rows (32×32
     thumbnail r 0.994–0.9997, confirmed visually). The baseline has 0 of 800 at the same
     threshold.
   - Decision (b): reject any new row whose thumbnail correlates above 0.95 with a ref image;
     report the rejection count, the correlations and a QC sheet; message main if more than 30
     rows are rejected.
   - Reason: n32k must differ from `lsun_church` in data size only.
   - The SHA-1 rule had already dropped 46 exact copies of `lsun_church` images, because the HF
     test and train shards overlap.
