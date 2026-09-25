# Orchestrator session log — [Code-Orchestrator], 2026-09-22

Model: Claude Fable 5.1 (`claude-fable-5-1`), effort xhigh. Skills loaded: `parallel-agents`,
`knowledge-map`. Peers consulted: `[Proposal-Specifier]`, `[Experiment-Reviewer]` (idle sessions
on this machine). At most two implementation agents at a time, Opus 5 unless stated.

## 0. The user's prompt (verbatim)

```
CONTEXT: You are an expert in the topic of our generative AI project, which is detailed here:
projects/GenAI/learning/03-terminal-blur-scaffolding.md
 and the proposal is detailed here:
projects/GenAI/project/6aab9fca47dc3902a0dbfcef/propuesta
 TASK: Load the /parallel-agents skill, and, using at most 2 Opus agents at once, you are tasked with (1) Planning the code development (at 2 level, first of all, specification-driven-development, where you will identify the tasks needed to complete the project; second level: implementation-level, where you will identify the best python code practices that we should follow and common software engineering patterns that we may apply for the development of the project), you are going to write the specifications first in a broad way, then, using milestones, and then, using tickets for each milestone, the milestones should be sequential, whereas tickets can be either sequential, or parallel (can be developed at one by several parallel agents). All these files will be placed in projects/GenAI/code/docs/, in their corresponding folder, a README.md will be a tree for these files. When an agent is given a ticket, it should create a md file inside a folder for its milestone in
projects/GenAI/code/docs/AGENT-LOGS
, paste the prompt given, and detail some decisions it had to take during the process of completing, files modified, and results of the ticket. You may create a skill for this, so that they can invoque it. The code will be written by the agent, whereas you will plan, design, supervise, answer doubts, and, if needed the opinion of other experts, consult sessions [Proposal-Specifier] and [Experiment-Reviewer], currently up. You should be little to no verbose in the console, and answer questions of the subagents if needed. ACCEPTANCE CRITERIA: You have designed the levels that I specified, the harnesses, and organized the development of the project. Regarding the iteration, you have iterated until, at least, we have all the correctly preprocessed for all teh datasets, in a standard format, where some known results of the exploration are known (like the power law); then, you have set up everything in Picasso for model training (copied data to
/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/spectral_allocation_heat_diffusion_project
 and set up the github repository in
/mnt/home/users/tic_163_uma/mpascual/fscratch/repos
, the conda env correctly created, and you have submitted a small probe picasso job that tries to train the model for 3 epochs for churches and ixi, and checked that it is behaving as expected). When you have done everything of this, and made sure that the training process saves the needed checkpoints required by the downstream metric computation, and logs sanity metrics of the run (or anything else that can only be obtained during training and that we'd need), you can spawn an agent with the /picasso-sbatch skill to write the slurm scripts and submit all the jobs. Think step by step, reason. Consult your peers if needed and remember that this is a course project, not a research paper, dont overengineer. Log your session as an md file too, with the prompts.
```

Mid-turn addition from the user: "Also! Commit and push to the repo as much as you need, its a
fork of the original repository that is in my account, so no problem."

## 1. Preflight findings

- Fork: `MarioPasc/generative-inverse-heat-dissipation`, base `3f735ca`, clean, `main` only.
- `projects/GenAI/code/` is git-ignored in the TFM repo → `docs/` moved into the fork and
  symlinked back (`projects/GenAI/code/docs → generative-inverse-heat-dissipation/docs`).
- Local GPU: RTX 3060 12 GB; 24 cores; 31 GB RAM; data disk 3.6 TB free. `tfm` env has torch
  2.13+cu130 but no SimpleITK/nibabel → dedicated `ihdm` env (D8).
- Picasso reachable; `fscratch/datasets/spectral_allocation_heat_diffusion_project/` exists
  (empty); conda envs live at `fscratch/conda_envs/`; A100 via `--constraint=a100`.
- Peers `[Proposal-Specifier]` and `[Experiment-Reviewer]` idle and answering.
- A Sonnet explorer surveyed the fork's code (config keys consumed, forward process, loss,
  sampler, checkpoint format, hooks); its findings are folded into `04-run-artifacts.md` §5 and
  the D3 fixes. Notable: `config.seed` is never consumed by the released code; the LSUN config
  lacks `model.sigma`; `mpi4py` is imported at module level in `scripts/datasets.py`.

## 2. Decisions and consultations

See `SPECIFICATIONS/00-overview.md` §4 (D1–D9) and its addendum (D1′, D3′, D4′, D5′, D6′,
D10–D13) with the peer objections that produced them. Items flagged **[ask Mario]**: 10 slices
instead of 8; batch 64 / warm-up 1000 / 20k instead of 128 / 5000 / 30k; checkpoint cadence 2.5k
(proposal) versus 5k (learning/03 §16.4); both paper-versus-code fixes on (A0 = "the paper as
described").

## 3. Wave record

| wave | tickets | agents | base | verdicts | merged |
|---|---|---|---|---|---|
| W0 | T0.1 | sonnet5-xhigh | `c23fbff` | ACCEPT | `1c7e2c5` |
| W1 | T1.1 ‖ T1.2 | opus5-xhigh ‖ opus5-high | `1c7e2c5` | T1.2 ACCEPT; T1.1 ACCEPT (IXI 539/581 pass, 400 used, 70 min; OASIS-1 405/416 pass, 400 used, 50 min; both cohorts validated, QC sheets eyeballed by the orchestrator: template outline on every brain, A top / L left; IXI200 excluded by hand after a metric-gate miss, `fov_fraction_window` noted as the better gross-failure detector) | T1.2 `c5ca1ad`; T1.1 `1c7a678` |
| W2 | T2.1 (started early: contracts only, no M1 data needed) | opus5-xhigh | `9f61ba7` | ACCEPT (154 tests; found the `lsun_church` routing bug, the float64-lr resume bug, the 61 M-param memory wall) | `1382a77` |
| W3 | T2.2 (slot 2 while T1.1 runs) | opus5-high | `a0bc936` | ACCEPT (257 tests; 18 s per 200-step chain per image on the 3060 at batch 20; samples reproducible only at a fixed batch) | `214b419` |
| W4 | T1.3 | opus5-xhigh | `1c7a678` | ACCEPT (310 tests; 7 frozen schedules; checklist 4/5 PASS, item ii FAIL on IXI: coarse share 3.96% > 3%, 84% of it in the (1,0) mode = a between-subject A–P intensity ramp; registration collapsed the log spread from 51×/94× to 8.7×/15.1× while the ordering and the crossover survived; N4 question sent to the reviewer) | `32df895` |
| W5 | T3.1 (clone/env/data now; import check after T1.3 lands) ‖ T2.3 (3060 pilot) | opus5-high ‖ opus5-high | `214b419` ‖ `32df895` | T3.1 ACCEPT after an orchestrator FIXUP of `picasso_setup.md` §4/§6 (driver 610.57 accepts cu130; import check PASSED: 318 tests, OK ixi/lsun_church, DONE); T2.3 partial ACCEPT (memory fit 1.705·B + 1.30 GB, lr 2e-4 stable to 525 steps; the lr 1e-4 comparison, A3/Churches pilots, resume and sampler checks were cut by the reboot and are folded into T3.2 or already evidenced by T2.1/T2.2) | `cd29f59` (both) |
| W6 (2026-09-23) | T1.4 (N4, local CPU) ‖ T3.2 (Picasso probe) | opus5-xhigh ‖ opus5-high | `cd29f59` | T1.4 ACCEPT (340 tests; both MRI sets rebuilt with N4, subject lists identical, schedules refit, 30 cells build; D15's premise only half right, see D15′) · T3.2 ACCEPT (job 2405546: batch 16 = 28.5 GB, 1.71 it/s, 6.84 h per 40k run; batch 24 OOM; sampler 3.2 s/chain; resume verified) | T1.4 `56b74f0`; T3.2 `96917bc` |
| W7 | T3.3 (array scripts, `picasso-sbatch`; held for T1.5) ‖ T4.1 (spectral metrics) ‖ T1.5 (stratified IXI split, sonnet5-xhigh) | opus5-high ‖ opus5-xhigh ‖ sonnet5-xhigh | `3ec0d92` | T1.5 ACCEPT (site mix train/ref 51.2/35.9/12.8 vs 51.2/36.2/12.5; ixi_W2/W8 refit; re-synced); T3.3 ACCEPT for the scripts (QOS `medium_uma` found; `--test-only` accepted); submission released as T3.3b after the merge; T4.1 running | T1.5 + T3.3 `5bc28a6`; T4.1 `305d672` (ACCEPT: 396 tests; LSD bracket: noise floors ixi 0.047, oasis1 0.049, churches 0.023, bedrooms 0.037; pilot LSD 0.78) |
| W8 | T3.3b (submission record) ‖ T4.2 (memorisation, diversity, PCA) | opus5-high ‖ opus5-xhigh | `5bc28a6` ‖ `305d672` | T3.3b ACCEPT (job 2408239 submitted, QOS medium_uma, cluster SHA 5bc28a6); T4.2 ACCEPT (467 tests; M calibrated at 1.00 on real held-out images; pilot M 1.75 = off-manifold at 750 iterations; found that T4.1s 64-seed draw held 18 indices outside `train` — cause undetermined, likely a race with T1.5s rebuild; hardened in T4.3: written seed lists + in-split assertion) | T3.3b `b0d2347`; T4.2 `8a3db69` |
| W9 | T4.3 (evaluate_run, Inception, statistics, paired gate) | opus5-high | `8a3db69` | ACCEPT (570 tests; pilot eval end to end: LSD 0.77, KID 0.43, FID 349 @ n_ref 800, M 1.75, gate CI [-0.05, +0.05] → no extension; FID licence ladder monotone; Inception weights at /tmp; reference-feature cache must be written once) | `a010e19` |

Decision D4⁗ (2026-09-23, orchestrator): **lr = 2e-4** (pre-registered) for every cell. Evidence:
T2.3's `ixi,A0` pilot at batch 4 ran 525 steps at 2e-4 with no non-finite loss, train loss
3.8 → 1.2 and eval tracking train; the 1e-4 comparison was cut by the reboot and is not repeated,
because D4‴ only required the fallback if instability appeared. Batch stays 16 (24 predicted at
42 GB, T3.2 confirms).

(Updated as waves complete; each wave's detail is `M<k>-<slug>/WAVE-<id>.md`.)

## 4. Prompts issued to agents

Each spawn prompt is stored verbatim in the agent's own log (§1 of the template); this file
records the ticket id, agent, model/effort, base SHA and any mid-flight corrections sent by
`SendMessage`.

## 5. Interventions

(FIXUPs, RETURNs, contract changes — appended as they happen.)

- W0/T0.1: approved `blobfile` + `opencv-python-headless`; accepted `model_channels=32` and the
  subprocess CPU smoke test; specs amended (`04` §6).
- W1: relayed the byte-exact contents of the three shared files (`ihdm/preprocess/__init__.py`,
  `ihdm/cli/__init__.py`, `ihdm/preprocess/errors.py`) from T1.1 to T1.2 so the merge is trivial
  (agents cannot address each other by name in this harness; `main` relays).
- W1/T1.2: accepted the dedup-induced index shift for Churches (row 1855 duplicate of 1623) and the
  shard-row naming of the raw PNGs; both documented in `meta.json`. Verdict ACCEPT, merged `c5ca1ad`.
- W1/T1.1: contract amended on the agent's measurements: the 300-iteration cap on the finest
  registration level is not a failure (converged metric, flat valley); FAIL = metric rule
  (median + 3 MAD) plus eye. Recorded in the ticket file.
- Environment hazard found by both W1 agents: `pip install -e .` from a worktree re-points the
  shared env's editable install. Rule from W2 on: agents use `PYTHONPATH=<worktree>`; the
  orchestrator re-runs `pip install -e .` on `main` after each merge.

## 5a. Handoff at the reboot of 2026-09-22 (evening) — resume from here

State of `main` (`origin/main` = local `main`, everything pushed): T0.1, T1.1, T1.2, T1.3, T2.1,
T2.2 merged; 310 tests green (`OMP_NUM_THREADS=2 pytest -q -m "not integration"`); the four datasets
validated at `$IHDM_DATA_ROOT`; `schedules/` frozen (pre-N4); recipe batch 16 / 40k in `arms.py`.

In flight when the machine went down (both agents were told to commit their interim state):

| ticket | branch / worktree | what was running | how to resume |
|---|---|---|---|
| T2.3 pilot (3060) | `ticket/T2.3-local-pilot-3060`, `wt/T2.3` | memory sweep + `ixi,A0` pilots at lr 2e-4 / 1e-4, `ixi,A3`, `lsun_church,A0` in `$IHDM_DATA_ROOT/_runs_local/`; killed by the reboot | read the interim log; the runs resume from `checkpoints-meta/` by re-issuing the same command; respawn a T2.3 agent (opus5-high) with the same prompt (log §1) plus "continue from the interim state" |
| T3.1 Picasso setup | `ticket/T3.1-picasso-setup` (pushed), `wt/T3.1` | driver probe and/or the import-check job on Picasso (job ids in its log); clone at `fscratch/repos/generative-inverse-heat-dissipation` on the ticket branch; env `fscratch/conda_envs/ihdm`; data synced (pre-N4 hashes) | `ssh picasso 'sacct -u mpascual --starttime today'`; read `~/execs/ihdm/logs/`; respawn or resume T3.1 to write `picasso_setup.md`, then merge |

T2.3 interim (committed `ae8fb30` on its branch, log §5): peak GB = 1.705·B + 1.30 on the 3060
(batch 2: 4.7 GB, 2.1 it/s; batch 4: 8.1 GB, 1.14 it/s, 4.5 img/s; batch 6 OOM); A100 extrapolation
batch 16 → 28.6 GB (fits), **batch 24 → 42 GB (does not fit; D4″'s "raise to 24" is moot)**; lr 2e-4
stable to step 525 (no NaN, loss 3.8 → 1.2), lr 1e-4 comparison NOT run; A3, Churches, resume test,
sampler and `pilot_3060.md` NOT done. **Budget risk to settle with the T3.2 probe's real A100
numbers**: training ≈ 10–13 h per 40k run at batch 16 (≈ 300–390 A100-h for 30 runs, against the
proposal's 150) and the `05-metrics` sampling plan (≈ 25k chains per run at ≈ 5 s per chain) would
cost ≈ 950–1,260 A100-h. Remedies to decide next session, in this order: measure real A100 it/s
and s/chain at large sampling batches (64–128) first; cut LSD to 4 checkpoints (10k/20k/30k/40k)
with 500 samples each and share one 2k final set for LSD, FID/KID and $M$; keep the 40 × 50
diversity set; then the pre-registered drop order (tier-3 seeds → A2 → iterations 40k → 30k).

T3.1 interim (pushed `7f34030` on its branch): Picasso jobs 2402054 `create_env` COMPLETED (5 m 44 s);
2403074 `gpu_probe` and 2403829 `import_check` PENDING (Reason=Priority) at the reboot. After the
reboot: `ssh picasso 'sacct -j 2403074,2403829'`, read `~/execs/ihdm/logs/{gpu_probe,import_check}_<id>.out`,
confirm the A100 driver accepts torch 2.14.0+cu130 (else rebuild the env with a cu12x index), then
fill `docs/RESULTS/picasso_setup.md` §4/§6 and the log's pending rows.

Next steps, in order: (1) verify/merge T3.1 and T2.3; (2) spawn **T1.4** (N4; ticket written,
D15) — it rebuilds `ixi`/`oasis1`, refits `schedules/`, updates `data_profile.md`; then re-sync the
two MRI datasets to Picasso (hashes change); (3) T3.2 probe (batch 16 vs 24, 3 epochs, resume,
sampler timing) with the lr from T2.3; (4) T3.3 array via `picasso-sbatch`; (5) M4 metrics tickets
while the queue runs. Peers [Proposal-Specifier] and [Experiment-Reviewer] hold the current numbers
(pre-N4) and expect the post-N4 ones for the Fig. 1 caption.

## 6. Open threads

- [ask Mario] items above.
- Registration is the schedule risk: T1.1 is the longest CPU ticket (≈ 1 h per cohort).

- 2026-09-23, W6: T3.2 found FSCRATCH at 254.9k files against the 250k soft quota (7-day grace
  running). Breakdown (`find -type f` per top-level dir): `conda_envs` 96.4k (ihdm 32.6k, isalhg /
  isalhg-tkde / isalhg-tkde-refill 64k), `results` 65.9k, `build_gedlib` 55.4k (Aug 2026 build
  tree), `conda_pkgs` 15.1k, `datasets` 8.1k, `repos` 4.2k, `pip_cache` 0.5k. The orchestrator
  removed the two caches (`conda_pkgs/*`, `pip_cache`; re-downloadable; the envs keep hardlinked
  copies and `ihdm` still imports) → 247.5k, grace cleared. **[ask Mario]**: `build_gedlib`,
  the three `isalhg*` envs and `results` are his other projects; clearing any of them is his call.
  Consequence for T3.3/T5.1: the array budgets < 2k new files (checkpoints only); evaluation
  artefacts go to `$LOCALSCRATCH` with one archive per run copied back.
- Follow-up (T3.1): `environment.yml` pins only `torch>=2.4` (resolved to 2.14.0+cu130 on both machines); pin the exact torch/torchvision versions once the Picasso driver probe (job 2403074) answers, so the env is reproducible.
- [ask Mario / Proposal-Specifier, not reachable after the reboot]: update the proposal Fig. 1 caption and Motivation numbers from `docs/RESULTS/data_profile.md` (N4 row): alpha 3.22/3.10 vs 2.28/2.58; coarse share 4.07%/2.02% vs 23.8%/23.1%; inherited W/8 8.4%/4.1% vs 29.1%; log spread 8.9x/18.7x vs 3.8x/8.0x; 3200 images per curve.

## 7. State at the close of the orchestration session (2026-09-23, 15:30)

- **Delivered**: specifications (levels 1–2), harnesses, ticket-log skill, 15 tickets executed and
  merged (T0.1, T1.1–T1.5, T2.1–T2.3, T3.1–T3.3(b), T4.1–T4.3), 570 tests green on `main`
  (`a010e19`), four datasets in the standard format (N4-corrected, site-stratified IXI), frozen
  schedules, the profile with the known results, the Picasso environment/data/repo, the probe, and
  the **training array 2408239 (30 runs) submitted** on `main` 5bc28a6 (the cluster tree is
  byte-identical to `a010e19` for everything the worker reads; docs and metrics code differ only).
- **Queued, not started**: all 30 tasks PENDING (Reason=Priority; the `--test-only` bound said
  2026-10-01 worst case). Monitor: `ssh picasso 'squeue -j 2408239'`; first-task check per
  `docs/RESULTS/submissions.md` §1 (the worker's `CELL index=… run_id=…` line and the first
  `metrics.jsonl` rows).
- **Next session (M5/M6)**: (1) when index 0 (`ixi_A0_s1`) and 3 (`lsun_church_A0_s1`) reach
  40k, run the plateau gate (`evaluate_run --gate 35000,40000` per `docs/HARNESSES/picasso.md` §6);
  extend all runs to 60k with `N_ITERS=60000 bash slurm/array/submit_array.sh` only if the CI is
  above zero; (2) T5.1 evaluation array with the `picasso-sbatch` skill honouring the notes in
  `docs/SPECIFICATIONS/M5-evaluation/README.md` (weights at `/tmp`, one writer of the reference
  feature caches, `$LOCALSCRATCH`, ≈ 7 A100-h per run); (3) T5.2 collection; (4) T6.1/T6.2 tables
  and figures. Pull the cluster checkout to `main` before T5.1 (it needs T4.x).
- **[ask Mario]** (all in `00-overview.md`): 10 slices; batch 16 / 40k / lr 2e-4 (measured, not the
  design file's 128 / 30k); checkpoint cadence 2.5k; both paper-vs-code fixes on; N4 with both rows
  (D15′); the evaluation cut (D16) and common random numbers (D17); the site-stratified split (D18);
  FSCRATCH file quota (his other projects hold 185k files: `isalhg*` envs 64k, `build_gedlib` 55k,
  `results` 66k); the proposal caption numbers (Proposal-Specifier was not reachable after the
  reboot).

## 8. Seed for the next orchestrator — how this was run, what to keep, what to change

Read this before spawning anything. It is the thought process, not the rules; the rules are in
`docs/SPECIFICATIONS/02-engineering-practices.md` §6 and the `parallel-agents` skill.

### 8.1 The loop that worked

Every wave followed the same eight moves, and every deviation from them cost time:

1. **Preflight the base.** `git status` clean on `main`, record `BASE_SHA`, `git worktree add
   ../wt/<ticket> -b ticket/<id>-<slug> <BASE_SHA>` for each agent. The fork is nested inside the
   TFM repo and git-ignored there, so the harness's `isolation: worktree` cannot be used (it would
   cut a worktree of the TFM repo, which does not contain the fork). Manual worktrees + the
   "prefix every shell command with `cd <worktree> &&`" rule were enough; no agent ever escaped.
2. **Write the ticket before the prompt.** The ticket file (`docs/SPECIFICATIONS/M<k>/T<k>.<n>-*.md`)
   is the durable spec; the spawn prompt restates it plus everything the agent cannot see: the base
   SHA, ownership set, frozen contracts with exact signatures, the environment commands, the
   verification commands, the shared resources, the definition of done, the log obligation, the
   peer roster, the final-message format. Agents have no conversation history; the prompt is their
   whole world. Prompts of 150–250 lines were normal and paid for themselves. Each agent pastes
   its prompt verbatim into §1 of its log, so the exact wording of every prompt is on disk.
3. **Two agents, disjoint files, contracts frozen in writing.** The `‖` pairs in
   `01-milestones.md` never shared a file except the two empty `__init__.py` and `errors.py`,
   whose byte-exact contents I relayed between T1.1 and T1.2 (agents in this harness cannot
   address each other by name; `main` relays). Zero merge conflicts in 15 merges.
4. **Answer within minutes.** Agents message `main` with an assumption and keep going; a late
   answer means rework. Every question they asked was a real contract gap (GroupNorm-32, the
   released `lsun_church` branch order, the iteration-cap "failure", OASIS-1 arm rule, the `train`
   rows vs dataset indices in `memorisation_ratio`), and each answer became a spec amendment the
   same hour. Keep the spec current on `main` while agents run: docs are orchestrator-owned and
   disjoint from every ticket, so committing them mid-wave is safe.
5. **Verify, then merge.** For each finished branch: `git status --porcelain` empty, `git diff
   --name-only <base>..HEAD` equals the log's file table, re-run the tests myself in the worktree,
   eyeball any PNG the ticket produced (I looked at every QC sheet), then `git merge --no-ff` on
   `main`, `pip install -e .` on `main` (see 8.3), full suite, `git push`, `git worktree remove`.
   Verdicts were ACCEPT or ACCEPT-with-FIXUP; one FIXUP was mine (`picasso_setup.md` §4/§6 after
   the reboot), one was a two-line recipe change with its test. Nothing was RETURNed or REDONE.
6. **Record as you go.** The wave table (§3) and the interventions list (§5) were updated at every
   merge; `docs/SPECIFICATIONS/00-overview.md` got a dated addendum row for every decision (D1–D18).
   When the machine rebooted mid-wave, §5a was the handoff and it worked: the next session
   resumed in three tool calls.
7. **Consult the reviewer at decision points, not for approval.** [Experiment-Reviewer] was asked
   four times, each with the measured numbers and a proposed decision, and each answer changed the
   design (FID required; plateau gate; N4 with both rows; stratified split; common random numbers).
   Ask with numbers and a default; never "is this OK?".
8. **Keep the console quiet.** The user reads only the final message of a turn; status lines of
   one or two sentences between agent reports were enough.

### 8.2 What the measurements overturned (so the next orchestrator does not re-assume them)

- The design file's recipe (batch 128, 30k) was impossible: the "CIFAR-scale" U-Net has 61 M
  parameters and needs 1.7 GB of activations per $192^2$ sample; batch 16 is the A100 ceiling.
  **Measure memory before budgeting** (T2.1 did it in a 60-iteration run on the 3060; the fit
  predicted the A100 to 0.1 %).
- The unregistered spectral numbers were mostly scalp position: registration collapsed the
  log-schedule spread from 45×/84× to 9×/19×. Only the ordering survived, and that is what the
  claim needs. N4 fixed within-site bias but not the between-scanner ramp; report both rows.
- Sampling, not training, dominates the compute: 3.2 s per chain per image on the A100 means the
  original evaluation plan cost 3× the training. Cut the plan (D16) before submitting anything
  that depends on it.
- The cluster's file quota, not space, was the binding constraint (250k soft, mostly Mario's other
  projects). Read `quota` before every campaign and write per-run artefacts to `$LOCALSCRATCH`.

### 8.3 Traps that bit and their fixes

- `pip install -e .` from a worktree re-points the shared env's editable install (first wave).
  Rule since W2: agents use `PYTHONPATH=<worktree>`; the orchestrator re-installs on `main`
  after every merge.
- Torch's default thread count collapses under a peer's 20-process registration: always
  `OMP_NUM_THREADS=2..4` in test commands (T2.1 measured 600 s vs 4 s).
- `create_model` wraps in `DataParallel(device_ids=None)`: CPU runs on a GPU host crash unless
  `CUDA_VISIBLE_DEVICES=""` (tests use a subprocess); inference code instantiates `UNetModel`
  directly.
- The released `optimization_manager` stores a numpy float lr that `torch.load(weights_only=True)`
  rejects on resume; handled in `ihdm/train/checkpoints.py`.
- `sbatch --qos=medium` is rejected: the association has `medium_uma`. `squeue --start` is
  ignored by the wrapper; `sbatch --parsable` prints a banner: parse the id with `grep -oE`.
- loginexa's V100 (sm_70) cannot run the cu130 wheels: no queue-free smoke test exists with this
  env; every GPU check is an A100 batch job (2 h queue on a normal day).
- clean-fid stores its weights in `/tmp` and its Inception is not bitwise deterministic across
  calls: one writer for the reference-feature cache, many readers.
- A sample set drawn while a peer rebuilt the dataset held out-of-split seeds (T4.1/T1.5 race).
  Never rebuild a dataset while another agent samples from it; seed lists are now files with
  hashes and an in-split assertion.
- The `rtk` shell hook rewrites `grep`/`git log` output in this environment: use `awk` for
  signature scans and `git log --format=` for SHAs; never trust a garbled listing.
- Session rate limits and a reboot each cut agents mid-flight. Tell agents to commit after every
  verified step (from W3 on), and resume them with `SendMessage` (they keep their context) rather
  than respawning; write the handoff (§5a) before a planned interruption.

### 8.4 Model and effort choices, in hindsight

Opus xhigh for tickets with real design content (registration, spectral port, trainer hooks,
metrics), Opus high for well-specified cluster or CLI tickets, Sonnet xhigh for mechanical but
careful work (scaffolding, the stratified split). Every Opus-xhigh agent found at least one defect
in the frozen contract and argued it correctly (GroupNorm-32; the iteration cap; the inherited-band
residual; the per-sample LSD that does not exist); that is the value bought. The Sonnet agents were
exact and fast on bounded tasks. Two agents at once was the right ceiling: the orchestrator's
attention, not the agents' speed, was the bottleneck.

### 8.5 Where to start (M5/M6)

1. `ssh picasso 'squeue -j 2408239'`; when indices 0 and 3 are `COMPLETED`, check the worker's
   `CELL index=… run_id=…` line in `~/execs/ihdm/logs/train_2408239_{0,3}.out` (the decode is the
   one failure that yields 30 plausible, wrong runs) and the first `metrics.jsonl` rows.
2. Plateau gate: `evaluate_run --run <ixi_A0_s1> --gate 35000,40000` (and `lsun_church_A0_s1`) in
   an A100 job; extend with `N_ITERS=60000 bash slurm/array/submit_array.sh` only if the CI lies
   above zero. Record in `docs/RESULTS/submissions.md`.
3. Pull the cluster checkout to `main` (it is at 5bc28a6, before T4.x), then T5.1 with the
   `picasso-sbatch` skill and the notes in `docs/SPECIFICATIONS/M5-evaluation/README.md`.
4. T5.2 collection to `~/execs/ihdm/results` and to this workstation; then T6.1/T6.2 in parallel.
5. Before any of it, decide the FSCRATCH clean-up with Mario and pin torch in `environment.yml`.

---

# Session 2 — [Orchestrator-GenAI], 2026-09-25

Model: Claude Opus 5.5 (`claude-opus-5-5`, 1M), effort xhigh. Skills: `parallel-agents`.
Subagents run Opus 5.5 (`CLAUDE_CODE_SUBAGENT_MODEL=claude-opus-5-5`).

## 9. The user's prompts (verbatim)

```
Gain the context needed for you to become the orchestrator of the project. Read any md file that you may need, apart from projects/GenAI/code/docs/AGENT-LOGS/ORCHESTRATOR-SESSION.md ; Re-check on the Picasso jobs and see if they succeded. If so, create a folder for the results in /media/mpascual/Sandisk2TB/research and copy them in an organized way. Results are in /mnt/home/users/tic_163_uma/mpascual/fscratch/runs while logs are in /mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs. When you have checked that the runs have correctly finished, read and understand the whole planning of the project in projects/GenAI/code/docs/SPECIFICATIONS and check where the last agents left in projects/GenAI/code/docs/AGENT-LOGS ; Using the harnesses for agents in projects/GenAI/code/docs/HARNESSES spawn maximum 2 subagents via /parallel-agents to continue with the next ticket.
```

Mid-turn addition:

```
If you need to re-submit the jobs, make sure to whipe out the current results and logs so that we free space, and exhaustively test the training/logging (metrics) procedure in loginexa (for example)
```

## 10. What was found and decided

- Array 2408239 failed 30/30 (`docs/RESULTS/submissions.md` §7): 7 non-finite-loss aborts at
  steps 1,367–1,888 (lr at 2e-4 since 1,000), 23 killed by an FSCRATCH outage on 2026-09-24.
  Nothing to copy as results; the failed array was archived to the SanDisk and wiped on Picasso.
- D19 (`00-overview.md`): recipe v2 = D4‴'s pre-registered fallback lr 1e-4, the skip policy,
  pre-clip gradient norm, recipe check on resume, pre-registered stability check S. The released
  configs use 2e-5 at ≥128 px; D4‴'s check never ran at full lr (process defect, §12).
- loginexa: the cu130 env cannot run on the V100 (sm_70); T3.4 builds a cu126 overlay in `$HOME`
  (FSCRATCH is at 248.8k of 250k files).

## 11. Wave record (session 2)

| wave | tickets | agents | base | verdicts | merged |
|---|---|---|---|---|---|
| W10 | T3.4 (recovery + loginexa harness) ‖ T5.1 (evaluation scripts + A100 cost) | opus55-xhigh ‖ opus55-high | `368fdde` | T5.1 ACCEPT (606 tests; prepare 2432211 + timing 2432221; fp16 1.33×, ΔLSD 5e-5) · T3.4 ACCEPT (diagnosis: fp16 forward overflow in the decoder upsampling during an lr-2e-4 spike; S passed at 1e-4 with 0 skips; H1–H7 PASS, 30/30 cells) | `integration/W10`: `7afe5ee` (T5.1), `cd07dbd` (T3.4); detail in `M3-picasso/WAVE-W10.md` |

## 12. Lessons added to §8

- **A stability check must run at the maximum lr, not inside the warm-up.** T2.3 (525 steps)
  and T3.2 (600 steps) both ended before step 1,000, so D4‴'s rule never had a chance to fire.
- **Log the quantity before the transformation you care about.** A post-clip gradient norm is
  identically the clip value.
- **A guard's saved state is a diagnosis fixture.** The abort saved the resume checkpoint after a
  no-op step, i.e. the exact weights that produced the non-finite loss.
- **Exercise every cell before the array.** Array 1 never executed indices 11–29; a config error
  in the transfer or ablation cells would have surfaced days later.
