# Wave W10 — T3.4 training recovery ‖ T5.1 evaluation array (2026-09-25)

| field | value |
|---|---|
| Orchestrator | session 2, Claude Opus 5.5 (1M), xhigh; skill `parallel-agents`; prompts in `ORCHESTRATOR-SESSION.md` §9 |
| Base commit | `368fdde` (D19, tickets T3.4 and T5.1) |
| Integration branch | `integration/W10` (manual worktree `wt/integration-W10`; the fork is nested in the TFM repo, so `isolation: worktree` cannot be used) |
| Agents | `T3.4` opus55-xhigh, `wt/T3.4`, `ticket/T3.4-training-recovery-loginexa`, head `a77680b` (26 commits, 35 files) · `T5.1` opus55-high, `wt/T5.1`, `ticket/T5.1-evaluation-array`, head `aef7ef7` (4 commits, 19 files) |
| Verdicts | T5.1 **ACCEPT** · T3.4 **ACCEPT** |
| Merges | `7afe5ee` merge(T5.1), then `cd07dbd` merge(T3.4), both `--no-ff` via `rtk proxy git merge`; no conflict |
| Logs | `M3-picasso/T3.4-training-recovery-loginexa.md`, `M5-evaluation/T5.1-evaluation-array.md` |

## Why this decomposition

Array 2408239 failed 30/30 (D19). The recovery (diagnosis, recipe v2, loginexa harness) was the
critical path. The evaluation scripts and their cost check needed only T4.3's code and one A100
job, and so fit into the queue's dead time. The two ownership sets were disjoint:

- T3.4: trainer, `scripts/losses.py` hook, `ihdm/train`, configs, `slurm/array`, `slurm/loginexa`, training/picasso harness docs.
- T5.1: `slurm/eval`, the `evaluate_run`/`run_eval`/`chain` AMP flag, the metrics harness doc.

Shared resources were split as well:

- loginexa GPUs → T3.4.
- The RTX 3060 and the A100 queue → T5.1.
- FSCRATCH write-free for both, except T5.1's 24 one-writer files.

## Verification by the orchestrator

- **T5.1.** Tree clean; the log's file table equals `git diff --name-only 368fdde..aef7ef7` (19 = 19) and every path is inside the ownership set. Re-ran the suite in the worktree: `606 passed, 21 deselected` (403 s); `ruff` clean. Code read:
  - `--amp` writes each mode to its own `samples*/`/`metrics*/` tree, and the legacy fp32 cache stays valid.
  - A non-finite draw is refused, not cast to bytes.
  - The worker keeps a shadow run on `$LOCALSCRATCH`, resumes from its tar, refuses a run without `DONE` or without the last checkpoint, and exports `PYTHONPYCACHEPREFIX`.

  Jobs: prepare 2432211 COMPLETED (24 files, seed-list hashes equal to local for all four datasets); timing 2432221 COMPLETED (1 h 48 m).
- **T3.4.** Tree clean; file table = diff (35 = 35); all inside ownership. The edit to the released `scripts/losses.py` only keeps `clip_grad_norm_`'s return value and exposes the scaler (optimisation unchanged).
  - Guard: the skip is counted only when the scaler is enabled; on abort, the resume checkpoint is kept only when the skipped steps were no-ops.
  - Recipe check: it compares content (schedule values and sha, data `images_sha256`), not paths. The legacy-manifest fallback accepts a stale hash only if `config.json` is identical key by key, so it cannot mask a changed key. Array-1 directories (no `recipe_sha256`) are caught by the key-wise path on `optim.lr`.
  - Harness evidence (`docs/RESULTS/loginexa_harness.md`) is verbatim; H2 put all 30 cells through the real worker (cells 11–29 had never executed).
- **Integration.** Full suite on `integration/W10` after both merges (`cd07dbd`), CUDA visible, `nice 19`, `OMP_NUM_THREADS=2`: `699 passed, 21 deselected, 9 warnings in 445.93s`; `ruff check ihdm tests`: All checks passed.

## Results that matter downstream

- **Diagnosis** (`docs/RESULTS/nan_diagnosis.md`). Array 1 died of fp16 forward overflow in the decoder upsampling convs (`output_blocks.9.2.conv`, `4.2.conv`) during an lr-2e-4 loss spike.
  - Fp32 activations at the spike weights reached 143–213 % of 65,504; the EMA stayed ≤ 0.53 % and the v2 weights at step 4,000 ≤ 0.33 %.
  - Attention logits ≤ 310.
  - The skip policy alone would not have saved array 1; the lr is the fix. No bf16.
- **Check S** passed at lr 1e-4 with 0 skips. Eval loss from step 1k to 4k: 0.3115 → 0.1939 (`ixi_A0_s1`) and 0.4815 → 0.3657 (`lsun_church_A3_s1`). The pre-clip grad norm fell from 400–6,400 to 14–76.
- **V100.** 0.87 it/s, 28.52 GiB peak at batch 16 (A100: 1.71 it/s).
- **Evaluation cost** (A100, job 2432221):

  | mode | s/chain | h per run | `--time` | A100-h (30 runs) |
  |---|---|---|---|---|
  | `off` | 3.233 | 7.43 | 10:00:00 | 223 |
  | `fp16` | 2.425 | 5.64 | 07:30:00 | 169 |

  - fp16 moves the LSD by 5.1e-5, 0.11 % of the `ixi` floor.
  - bf16 is equally fast but moves the LSD 13× more.
  - D16's 3.2 s/chain was measured without AMP, so the D16 budget holds. The gate's resolution at 500 seeds is ±0.0026.

## Interventions

1. **Archive and wipe.** Array 1 was archived to the SanDisk (82/82 run files and every log verified by name and size). My delete on Picasso was blocked by the permission check; Mario ran it himself. The rsync also caught T3.4's live overlay log, which is not array-1 evidence.
2. **T3.4 reported that the run root was not empty.** The wipe was pending on the archive; no change.
3. **FSCRATCH `.pyc` incident** (T3.4). The overlay build's imports wrote 892 `.pyc` + 118 `__pycache__` into the base env on FSCRATCH (248.8k → 249.8k). T3.4 deleted exactly those. Approved `PYTHONPYCACHEPREFIX` in the training worker; relayed the rule to T5.1 before any of its jobs ran.
4. **T5.1 timing independent of prepare, and gate flags.** Accepted, on condition that the job checks the shadow data and seed-list hashes, and that a test pins the gate's 1,000-chain draw. Both conditions were met.
5. **Mario reported the workstation was slow.** Two T5.1 processes were on the display GPU and one T3.4 process used about 5 cores. I reniced them; both agents were limited to ≤ 1 local process at `nice 19`, `OMP 2`, with heavy compute on loginexa or the A100.
6. **T3.4 recipe check compared `blur_schedule_file` paths** → rejected. It now compares content, with tests both ways.
7. **T3.4 S sessions reset the scaler and the data order at each resume** → accepted. S answers the forward-overflow question, and the session boundaries are reported.
8. **Headroom threshold.** Asked T3.4 to report whether v2's live weights come within 25 % of 65,504 anywhere. Answer: ≤ 0.33 %, so no activation logging is added to the array.

## Decisions taken at the merge

- **D20 (evaluation precision): `--amp fp16` for all 30 runs**, `TIME_LIMIT=07:30:00`.
  - Evidence: 1.33× speed-up, 54 A100-h and ≈ 7 h of makespan saved; |ΔLSD| 5.1e-5 = 0.11 % of the `ixi` floor and 2 % of the gate's resolution. The EMA and v2 weights sit ≥ 190× below the fp16 ceiling, and a non-finite draw now fails the task loudly.
  - Rule: a run whose fp16 draw fails is re-evaluated **entirely** in `off` and flagged in T5.2/T6.1. The gate job uses the same mode as the array, so its sample sets are reusable.
  - Recorded in `00-overview.md`.

## Follow-ups

- **Before submitting the v2 array.**
  - Merge `integration/W10` into `main` and push; `git pull` the cluster clone.
  - Rerun `sbatch --test-only`; submit with `N_ITERS=40000`.
  - FSCRATCH headroom is ≈ 1.3k files against ≈ 1.2k for the array: freeing files first is Mario's call.
- **After the submission.** Submit the gate job with `TRAIN_ARRAY=<id>` and `AMP=fp16`. Delete `~/execs/ihdm/wt/{T3.4,T5.1}` (≈ 600 HOME files) and `~/execs/ihdm/fixtures/array_2408239/` (2.3 GB) once nothing refers to them.
- **Kept on purpose.** The loginexa overlay (`~/execs/ihdm/overlay/ihdm-v100`, 5.2k files, 5.3 GB) and the S runs (`~/execs/ihdm/loginexa_runs`, 4.6 GB).
- **Low priority.** `paired_lsd_gate` raises `StatsError` at tiny n and the CLI does not catch it (`ihdm/stats`; irrelevant at n = 500).
- **Downstream readers.** T5.2 must read `metrics[_amp-<mode>]/` and check `sampling.amp`. T6.2 must accept the `skip`/`done` events, `amp_scale`, and a pre-clip `grad_norm` (13–6,400).

## What the decomposition got wrong

- **Local compute budget.** The tickets named the loginexa GPUs and the RTX 3060 as resources, but not the workstation itself. Both agents ran heavy local processes on Mario's desktop until he noticed. Future prompts must state a local budget: ≤ 1 process, `nice 19`, and the display GPU only for short checks.
- **"Write nothing on FSCRATCH" was stated for artefacts only.** Python's bytecode cache is an implicit write into the env, and it cost ≈ 1k files of a 1.3k-file margin. `PYTHONPYCACHEPREFIX` is now a standing rule for every Picasso process.
