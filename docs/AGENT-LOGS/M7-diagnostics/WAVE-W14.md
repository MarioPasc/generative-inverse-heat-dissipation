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
5. **T7.2 GO gate (head `87133bd`; code at `b030676`): accepted, GO given.**
   - Every file in the diff lies inside T7.2's ownership; `ihdm/sampling` is untouched.
   - The Picasso copy's sha256 equals local for `run_eval.py`, `spectral.py` and
     `delta_sweep.sbatch`.
   - Main re-ran `tests/metrics`, `tests/sampling` and `tests/analysis`: 517 passed, 6 skipped.
     `test_regression_t72.py`: 4 passed, 0 skipped, so the default path is byte-identical at
     W = 96 and W = 192.
   - δ reaches the lsd, final and heldout draws. An explicit 0.0125 gets its own tree, so the
     anchor is a real reproduction.
   - The job: an array of 4 tasks, 2,100 chains each, `--time 02:30:00`, 10 A100-h requested. It
     works on a `$LOCALSCRATCH` shadow run, writes nothing on FSCRATCH, and copies back to
     `~/execs/ihdm/diag_eval/delta_sweep/`.
   - Smoke: V100 loginexa only. The A100 queue `--test-only` estimate was 2026-10-24.
   - Follow-ups:
     - `paired_lsd_gate` at W ≠ 192;
     - rows for `lsun_church_r128` and `lsun_church_n32k` in `slurm/eval/expected_seed_lists.csv`
       (goes to T7.3);
     - Mario to delete `~/execs/ihdm/diag_eval/smoke_loginexa`.
6. **T7.2 submitted: array job 2550585** (tasks 0–3: ixi_A0_s1, ixi_A3_s1, lsun_church_A0_s1,
   lsun_church_A3_s1), code `b0306769`, PENDING at submission. Agent head `c9563b8`, status
   waiting-for-queue; main resumes it when the tasks finish.
7. **T7.1 GO gate (head `c83ff4a`, code `b483fcf`): accepted, GO given.**
   - Every file in the diff lies inside T7.1's ownership. The Picasso copy's sha256 equals local
     for 5 files, and `slurm/array/train_array.sbatch` is unchanged.
   - Config dump against `lsun_church` A0:
     - r128 differs only in the dataset, `image_size` 128, the schedule `log_W2_128` and σ_max 64;
     - n32k differs only in the dataset id;
     - A3 is refused on r128.
   - Datasets:
     - r128's `splits.json` equals `lsun_church`'s;
     - n32k's ref, seed and idx 0–3999 are pixel-identical to `lsun_church`, with 32,000 train;
     - r128's centre crops correlate with `lsun_church` at median 0.988;
     - both QC sheets eyeballed: the 14 rejected rows are the same 5 photos, and the r128 pairs are
       the same scenes.
   - Main re-ran `tests/{train,spectral,preprocess,slurm,data}`: 523 passed. The agent's full
     suite: 1095 passed, 1 skipped.
   - A100 smoke 2550583: `check_run` PASS. r128 runs at 3.665 it/s and 13.4 GB, n32k at
     1.712 it/s and 28.5 GB.
   - `--time` is 06:00 for r128 and 13:00 for n32k.
   - Declined: a `git_sha` fallback in `ihdm/train`; the `GIT_SHA` file is enough.
   - Mario to delete `fscratch/runs/ihdm_diag_smoke` (4.1 GB).
8. **T7.1 complete** (head `1739d58`).
   - Submitted: array 2550808 (r128, `--time` 06:00) and array 2550811 (n32k, 13:00), both
     PENDING (Priority). The first-task check passes to main.
   - The log's file table equals the diff (19 files). Verdict: **ACCEPT**.
   - Merged into `integration/W14` (worktree `wt/integration-W14`, branched from main `b90612e`)
     as `0246650`; the prompt was pasted by main as `29c9529`. Fast suite on integration: 1095
     passed, 1 skipped.
9. **T7.4** (sonnet5-xhigh, base `cf574ea`, head `6c8a5ea`): the exploratory module hardened.
   - Octave bands are now 0.5–2 / 2–4 / 4–96 c/img; ValueError paths become AnalysisError (exit
     1); value tests were added.
   - Main's independent JSON diff of the regenerated outputs: 0 non-band changes over 30 runs and
     99 contrasts; `oct_rms_high`, the curves and the curve statistics are identical.
   - New: Churches A2′−A2 on the 2–4 band shows CI excludes 0 (+0.084).
   - Main re-ran `tests/analysis`: 186 passed. Verdict: **ACCEPT**.
   - Merged as `41f843e`; prompt pasted as `63c4d96`; `tests/analysis` + `tests/train`: 451
     passed.
10. **First-task check by main** (2026-10-01, ~12:43 local). Starts were far earlier than the
    `--test-only` estimates.
    - The δ-sweep tasks 0–2 started at 11:20–11:25 and task 3 at 12:33. Training: 2550808_0 at
      12:33 (node exa03), 2550811_1 at 12:34 (exa04).
    - Both training logs show the right `CELL` line and run id, with `N_ITERS` 60000.
    - `config.json`: r128 has `image_size` 128, σ_max 64; n32k has 192, 96. Both have lr 1e-4 and
      batch 16.
    - `metrics.jsonl`: an eval line at step 0; finite losses (r128 1.69 → 0.19–0.23 at 1.8k; n32k
      3.81 → 0.59 at 800); the lr rises over the warm-up; 3.65 it/s (r128) and 1.71 it/s (n32k),
      as in the smoke.
    - ETA: r128 ≈ 17:15, n32k ≈ 22:20 local.
11. **The δ sweep finished.** Array 2550585: 4/4 COMPLETED in 1:32–1:39 each, 6.3 A100-h.
    - **Anchors:** all four reproduce `lsd_060000` to 6 digits; the largest difference is
      +0.00037 on Churches A0.
    - **Result:** at δ ≥ 2σ every run is over-dispersed (variance ratio 1.3–4.8), with KID up
      (CIs disjoint) and precision 0. The pre-registered reading gives "noise": the
      under-dispersion is not a sampler setting within the tested grid. The added variance goes to
      64–96 c/img.
    - **Review: RETURN for one fix.** The README column labelled "I_w" held the pre-registered,
      biased `inherited_measured`. Main asked for it to be relabelled, for the D23 I_w (M = 5) and
      ρ to be added, and for a note on the 5-versus-50 samples per seed.
12. **T7.2 fix verified** (head `1fb48f9`; fix commit `aab6bd3`). Verdict: **ACCEPT**.
    - Only T7.2's own files changed. Main re-ran `test_diag_collect` + `test_regression_t72`:
      9 passed.
    - ρ at 1.25σ matches production for IXI A0 (0.345 vs 0.344) and both Churches runs (0.058
      vs 0.058; 0.106 vs 0.105); IXI A3 is 0.290 vs 0.281, within 5-sample noise.
    - Merged into `integration/W14` as `2def5ef`; prompts (redo + superseded first) pasted as
      `e9474c8`.
    - Then, on integration:
      - main appended dated M7 amendments to 03, 04 and 05 (`bee7057`), from the agents' §6
        proposals;
      - main updated `results_discussion.md` §8, §10 and §11 with the δ result (`031f548`).
