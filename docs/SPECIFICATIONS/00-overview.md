# 00 — Project overview and top-level specification

Status: frozen 2026-09-22 by the Code-Orchestrator session. Changes go in dated addenda at the end.

## 1. What is being built

A course-project experiment (GenAI, M.Sc. AI, UAM) on **Inverse Heat Dissipation Models (IHDM)**
(Rissanen, Heinonen & Solin, ICLR 2023, arXiv:2206.13397). The scientific question, fixed in the
proposal (`projects/GenAI/project/.../propuesta/templateArxiv.tex`) and in
`projects/GenAI/learning/03-terminal-blur-scaffolding.md` §16 and §21:

> Should the terminal blur $\sigma_{B,\max}$ and the spacing of the $K$ blur levels depend on the
> spectrum of the data? Brain MRI is the case where the photograph defaults are farthest from what
> they assume.

Two knobs of the released code are changed, everything else fixed:

| knob | paper default | brain configuration |
|---|---|---|
| terminal blur `blur_sigma_max` | $W/2 = 96$ px | $W/8 = 24$ px |
| level spacing `blur_schedule` | log-spaced, $K=200$ | variance-matched, fitted once on the IXI training split |

Five arms, four datasets, two roles:

| arm | `blur_sigma_max` | schedule | role | seeds |
|---|---|---|---|---|
| A0 | 96 | log | the paper | 3 |
| A3 | 24 | IXI-matched in $[0.5, 24]$ | the brain configuration; transferred frozen | 3 |
| A1 | 24 | log | ablation: terminal blur alone | 2 |
| A2 | 96 | IXI-matched in $[0.5, 96]$ | ablation: spacing alone | 2 |
| A2′ | 96 | LSUN-Churches-matched in $[0.5, 96]$ | control, photographs only | 2 |

| role | MRI | photographs |
|---|---|---|
| development pair (A0, A1, A2, A3, A2′ on photographs) | IXI T1 | LSUN Churches |
| transfer pair (A0, A3 only) | OASIS-1 T1 | LSUN Bedrooms |

Total: 12 + 8 + 2 (A2′ Churches) + 8 (transfer) = **30 training runs** of 20k iterations at native
$192^2$, grayscale.

Endpoints (proposal §Measurements): **within-seed diversity** and the **memorisation ratio $M$**
for the terminal blur; the **log-spectral distance (LSD)** at checkpoints, its per-octave profile,
the final LSD and $T_\tau$ for the spacing; the **inherited-band check** as the mechanism figure.
FID/KID/prdc are optional extras, implemented only if time remains.

## 2. Scope of the code work

In scope (this repository, Mario's fork of `AaltoML/generative-inverse-heat-dissipation`):

1. A preprocessing pipeline that turns the four raw sources into **one standard on-disk format**
   (`03-data-format.md`), with subject-level splits, quality-control sheets and a validator.
2. The **spectral profile** of the final training splits (the power-law exponent $\alpha$, octave
   shares, inherited shares at $W/2$ and $W/8$, per-level spread under each schedule) and the
   **frozen schedule arrays** for the arms, plus the report's data figure.
3. The **training harness**: the released code with a dataset backend for the standard format, the
   schedule loader, the arm configs, checkpoint cadence, run manifest and sanity logging, the two
   paper-versus-code fixes, and an offline sampler that draws from a checkpoint given seed images.
4. The **Picasso deployment**: environment, data transfer, repository clone, a probe job, then the
   full SLURM array.
5. The **metrics harness** and the **analysis** (statistics, tables, figures) for the report.

Out of scope: any change to the U-Net or the loss beyond the two documented fixes; the report's
LaTeX; the TFM knowledge base (`projects/GenAI/learning/*`); FID/KID unless time remains.

## 3. Constraints that every ticket honours

- **Released code unchanged** except at the documented hook points (`04-run-artifacts.md` §5).
  No refactor of `model_code/`; no renaming of existing config keys.
- **One recipe in every cell.** Architecture, optimiser, batch size, iteration count, noise
  levels and seeds are identical across arms and datasets; only `blur_sigma_max` and
  `blur_schedule` differ. The recipe is frozen after the local pilot (M2) and never retuned per arm.
- **Same seeds across cells**: seed values $\{1, 2, 3\}$; a two-seed cell uses $\{1, 2\}$.
- **Held-out data for every metric.** Nothing is measured against the training split except the
  nearest-neighbour distance that defines $M$ (by construction).
- **Native $192^2$**: a 192-px window of the native frame; no resampling other than the rigid
  MNI152 registration of the MRI volumes (one interpolation per volume).
- **Splits by subject** for MRI (slices of one subject never straddle a split), by image for
  photographs; 80/20; 40 seed subjects (or images) inside the reference 20%.
- **Same image count per dataset**: 4000 images each; 3200 train / 800 reference.
- **Course project, not a paper**: correctness and reproducibility over generality. No new
  abstraction without two concrete users.

## 4. Decisions taken by the orchestrator (2026-09-22)

Each was sent to the [Experiment-Reviewer] and [Proposal-Specifier] sessions for challenge;
objections, if any, are recorded as addenda below.

| id | decision | rationale |
|---|---|---|
| D1 | 4000 images per dataset: MRI 400 subjects × 10 axial slices, MNI $z \in [-10, +35]$ mm at 5 mm spacing; photographs 4000 images (2000 more fetched from the same HF shards) | equal counts across datasets are required by the proposal; 5 mm spacing keeps adjacent slices from being near-duplicates; the band is the "central 40% of $z$" rule of the CPU analyses, expressed in MNI coordinates |
| D2 | Storage as `uint8`, loader divides by 255 | quantisation variance $1.3\times10^{-6}$ against training-noise variance $10^{-4}$; photographs are 8-bit anyway, so MRI and photographs get symmetric treatment |
| D3 | Fix both paper-versus-code items in every arm: train levels $1..K$ inclusive; add $\delta$-noise to the prior draw | level $K$ is called at sampling time, so it must be trained; the prior with noise is what the paper states; both are config-toggled (`model.train_level_max_inclusive`, `sampling.prior_noise`) so the choice is auditable |
| D4 | Recipe: CIFAR-scale U-Net (`model_channels=128`, `channel_mult=(1,2,2,2)`, `num_res_blocks=4`, `attention_levels=(2,3)`), 1 channel, $K=200$, $\sigma=0.01$, $\delta=1.25\sigma$, batch 32, AMP, EMA 0.999, grad clip 1.0, warm-up 1000, lr $2\times10^{-4}$, 20k iterations | the design file's recipe with the warm-up shortened for a 20k run; batch 32 is the paper's batch at $128^2$ and fits the local 12 GB pilot; the pilot may lower lr to $10^{-4}$ if unstable, once, for all cells |
| D5 | Checkpoints: EMA weights every 2,500 iterations (8 per run), full state at the end, rolling resume checkpoint; run manifest at start; all metrics computed offline from EMA checkpoints | keeps the training job simple and the storage bounded (≈160 MB per EMA checkpoint); the sampling budget (2k samples per checkpoint) is a separate array job |
| D6 | Endpoints as in §1; FID/KID/prdc deferred | the proposal names LSD, $M$, diversity and the inherited band as the endpoints; Inception metrics need a licence check that costs time |
| D7 | Local data root mirrors the Picasso one: `/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/` locally, `$FSCRATCH/datasets/spectral_allocation_heat_diffusion_project/` on Picasso; env var `IHDM_DATA_ROOT` | one relative layout, two roots |
| D8 | Dedicated conda env `ihdm` from `environment.yml` (torch ≥ 2.4 with CUDA, SimpleITK, nibabel, ml_collections, absl, tensorboard, pytest); `mpi4py` dropped | the fork's pins are from 2023 and the local `tfm` env lacks the imaging libraries |
| D9 | `docs/` lives inside the fork (symlinked from `projects/GenAI/code/docs`), so specifications, harnesses and agent logs travel with the code and are versioned | `projects/GenAI/code/` is git-ignored in the TFM repository |

### Addendum 2026-09-22 — after the [Experiment-Reviewer] and [Proposal-Specifier] replies

| id | change | source |
|---|---|---|
| D1′ | **10 slices per subject at MNI152NLin2009cAsym 1 mm voxel $z$ indices `round(linspace(55, 128, 10))` = {55, 63, 71, 79, 87, 96, 104, 112, 120, 128}** (30–70% of the atlas head extent, ≈8 mm apart, from $z \approx -23$ to $+50$ mm; the plane used for the report's Fig. 1 is index 5, voxel 96). 400 subjects × 10 = 4000. Bedrooms: T1.2 counts the test shard first and takes the remainder from a train shard if it holds fewer than 4000 usable rows, recording it in the raw-data README. **[ask Mario]**: the CPU analyses used 8 slices; 10 are needed for OASIS-1 (416 subjects) to reach 4000 | Specifier §1, Reviewer §1a |
| D3′ | Both fixes on, both config keys kept (`model.train_level_max_inclusive=True`, `sampling.prior_noise=True`); the report states that A0 is "the paper as described", not "the code as released"; a released-code A0 can be run once if asked | Reviewer §3 |
| D4′ | **Batch 64** if the Picasso probe measures peak memory ≤ 30 GB at $192^2$ with AMP, else 32; recorded once in `configs/spectral/arms.py`. lr $2\times10^{-4}$, warm-up 1000, 20k iterations pre-registered. **[ask Mario]**: the design file said batch 128 / warm-up 5000 / 30k | Reviewer §4 |
| D5′ | Checkpoint cadence 2,500 (the proposal) rather than 5,000 (learning/03 §16.4); 2k samples per checkpoint for LSD, reducible to 1k at intermediate checkpoints if the evaluation budget bites. **[ask Mario]**: he has not been told of the 2.5k/5k discrepancy | Specifier §1 |
| D6′ | **FID and KID (`clean-fid`, grayscale replicated to 3 channels) and recall/coverage (`prdc`) at the final EMA checkpoint are required**, not optional: they are the only quantities comparable to the paper. Ticket T4.4 | Reviewer §6a |
| D10 | **Plateau gate with a resume rule.** All 30 runs are submitted at 20k iterations in tier order (A0 and A3 first). When the first A0 runs finish, their LSD-versus-iteration curve is checked: if the LSD changes by more than 5% between the last two EMA checkpoints, every run is extended to 30k by resubmitting with `--config.training.n_iters=30000` and resuming from its rolling checkpoint (the trainer's loop is `range(initial_step, n_iters + 1)`, so extension is native). No cell is ever re-trained from scratch with a different count | Reviewer §4 |
| D11 | Two seed sources in the offline sampler: `train` (Alg. 2, for the fidelity/LSD samples) and `seed` (the 40 held-out seed subjects, for diversity, $M$ and the inherited band); seeds are passed as an array, never drawn inside the sampler | Reviewer §1c, §6c; Specifier §3 |
| D12 | The A3 and A2 arrays are built by running the variance matching with the endpoint set to $W/8$ or $W/2$ (bisection on the registered IXI training split), never by truncating or rescaling another array; the schedule array and its SHA-256 are stored in every run manifest and inside every checkpoint | Reviewer §7a, §7c |
| D13 | LSD, $T_\tau$, $M$, diversity, the inherited band and the PCA figure are defined in `05-metrics.md` in the theory's units (cycles per image, the octave bins of the tables) | Reviewer §6b |

### Addendum 2026-09-22 (evening) — after T2.1 measured the memory of the CIFAR-scale U-Net at $192^2$

| id | change | source |
|---|---|---|
| D4″ | The recipe's U-Net has **61.1 M parameters** and needs ≈ 1.75 GB of activations per sample at $192^2$ with AMP (measured on the RTX 3060: 4.7 GB peak at batch 2, OOM at batch 8 on 11.6 GB), so batch 64 and batch 32 do not fit an A100 40 GB (≈ 29 GB at batch 16, ≈ 57 GB at batch 32). **Batch = 16**, raised to 24 only if the Picasso probe measures ≤ 30 GB peak at 24. The model is not changed (the released `use_checkpoint` flag of the ResBlocks is left off: gradient checkpointing would buy batch at the cost of throughput, and samples per GPU-hour is what the budget is about). **[ask Mario]** | T2.1 log §6 |
| D4‴ | **Iterations = 40,000** at batch 16 (640k samples seen, the sample budget the reviewer asked for; ≈ 200 epochs of 3200 images), pre-registered; the plateau gate of D10 still applies (extension to 60k by resume). At batch 24 the count is 27,500 (11 × 2,500). lr stays $2\times10^{-4}$ unless the local pilot (T2.3) shows instability at batch 16 over 1k iterations, in which case $10^{-4}$ is used everywhere; T2.3 runs both for 1k iterations and reports the loss curves | Reviewer §4; T2.1 |
| D5″ | Checkpoint cadence stays 2,500 → 16 EMA checkpoints per run at 233 MB each (measured) ≈ 3.7 GB per run, ≈ 112 GB for the array plus ≈ 21 GB of `full_final.pt`; Picasso FSCRATCH has ≈ 0.9 TB free (read live before the array). LSD uses 1k samples at intermediate checkpoints and 2k at the final one | T2.1; H-PICASSO §1 |
| D15 | **N4 bias-field correction** on the registered MRI volumes (MRI only; default N4 parameters; foreground mask = the dilated template brain mask; never tuned against the coarse share), because T1.3 found 84% of IXI's coarse-bin variance in the single (1,0) mode, an anterior–posterior between-subject intensity ramp (multi-site / coil residue, not anatomy). The claim is stated on N4 data; the uncorrected numbers are a **sensitivity row** in every table; the crossover is recomputed after the refit; IXI site labels are recorded and the coarse share is printed per site before and after. Ticket T1.4. The conclusion already held without N4 (3.96% vs 23.8% coarse share; 10.3% vs 29.1% inherited at $W/8$) | Reviewer (evening reply); T1.3 log |
| D15′ (outcome, 2026-09-23) | N4 applied and kept (T1.4), but **its premise was only half right**: per site N4 lowers the coarse share (Guys 3.51 → 3.11%, HH 4.12 → 2.90%, IOP 3.21 → 2.62%), yet the pooled IXI share stays at 4.07% (was 3.96%) because the (1,0) ramp is largely a **between-scanner** component that a within-volume correction cannot remove; OASIS-1 falls 2.44 → 2.02%. The report words it as "a between-subject anterior–posterior intensity component, part within-site bias field and part between-scanner, which N4 halves within a site but not across sites"; histogram standardisation (Nyúl–Udupa) is named as the next step and not done. Final numbers (N4): $\alpha$ IXI 3.22 / OASIS-1 3.10 vs Churches 2.28 / Bedrooms 2.58; coarse share 4.07% / 2.02% vs 23.8% / 23.1%; inherited at $W/8$ 8.4% / 4.1% vs 29.1%; log spread 8.9× / 18.7× vs 3.8× / 8.0×; the uncorrected numbers are the sensitivity row in `docs/RESULTS/data_profile.md`. Checklist item (ii) (IXI < 3%) fails at 4.07% in both rows; the ordering MRI ≪ photographs (6×) is what the claim needs and holds | T1.4 log §5 |
| D16 (2026-09-23, after the A100 probe) | **Measured on the A100** (T3.2, job 2405546): batch 16 → 28.5 GB peak, 1.71 it/s, **6.84 h per 40k-iteration run** (205 A100-h for 30 runs; `--time=11:00:00`, QOS `medium`, `--mem=32G`); batch 24 OOMs (37 GB); the sampler runs at **3.2 s per 200-step chain per image** from batch 32 up (plateau; never batch 128). The `05-metrics.md` sampling plan as written (25k chains per run) would cost 669 A100-h, so the **evaluation plan is cut to 8k chains per run** (≈ 7.1 h, ≈ 214 A100-h): LSD at the 8 checkpoints 5k, 10k, …, 40k with 500 training-seeded samples each; one shared final set of 2k training-seeded samples for the final LSD, KID/FID, recall/coverage and $M$; the 40 × 50 held-out-seeded set for diversity, the inherited band and the PCA. $T_\tau$ resolution becomes 5k iterations; the plateau gate (D10) compares the LSD at 35k and 40k. Campaign ≈ 420 A100-h; the pre-registered drop order applies if the queue is slow | T3.2 log §5; T2.3 |
| D14 | `scripts/datasets.get_dataset` routes the standard-format ids before the released torchvision branches (T2.1 found that `lsun_church` matched the released LMDB branch first); the released `optimization_manager` stores a `numpy.float64` lr that torch ≥ 2.6 `weights_only` loading rejects, handled in `ihdm/train/checkpoints.py`; `metrics.jsonl` never contains `NaN` tokens (`null`) | T2.1 log §6 |

### Addendum 2026-09-25 — the first training array failed (orchestrator, session 2)

| id | change | source |
|---|---|---|
| D19 | **Array 2408239 (recipe v1) failed 30/30.** 7 runs were aborted by the non-finite-loss guard at steps 1,367–1,888, i.e. 367–888 steps after the lr reached its 2e-4 maximum at step 1,000; the train loss one log step earlier was healthy (0.35–0.49), so the event is sudden, not a slow divergence. 4 runs were killed by an FSCRATCH outage (stale file handle, 2026-09-24 10:54) after 750–8,750 steps without a non-finite loss; 19 failed at start in the same outage. The logged `grad_norm` was measured after clipping and is identically 1.0, so it carried no information. **Process defect:** D4‴'s stability check never reached full lr — T2.3 ran 525 steps at batch 4 and T3.2 600 steps at batch 16, both inside the 1,000-step warm-up. **Evidence on the lr:** the released configs use 2e-5 at batch 32 with 5,000 warm-up for LSUN Churches 128² and FFHQ 256², 1e-4 at batch 32 for AFHQ 256²; 2e-4 is the released value only at 28–32 px with batch 128. **Decision (recipe v2):** D4‴'s pre-registered fallback, **lr = 1e-4 for every cell**, everything else unchanged (batch 16, warm-up 1,000, clip 1.0, fp16 AMP, EMA 0.999, 40k). Pre-registered stability check **S** (T3.4): the two cells that failed first under v1 (`ixi,A0,s1` at 1,382 and `lsun_church,A3,s1` at 1,367) reach step 4,000 (3,000 steps at full lr, 3.4× the latest v1 failure offset) with zero non-finite losses; if S fails, lr = 5e-5 once; if that fails, stop (**[ask Mario]**). Power of S: v1 failed 7 of the 10 runs that reached the high-lr window within 900 steps, so an unchanged hazard would fail S with probability ≈ 0.9. **Guard change:** under an enabled GradScaler a non-finite loss makes the step a no-op (the released code's behaviour), so it is logged as `{"kind": "skip"}` and training continues; abort (exit 3) after 10 consecutive or 100 total skips, or at the first non-finite loss when the scaler is disabled (non-AMP or CPU), because then the update is not a no-op. `grad_norm` becomes the pre-clip norm; the GradScaler scale is logged; a resume whose recipe differs from the run's manifest aborts before writing. **Rerun:** all 30 runs from scratch after the loginexa harness of T3.4 passes; the array-1 run directories and logs were archived to `/media/mpascual/Sandisk2TB/research/spectral_allocation_heat_diffusion_project/training/array_2408239_failed/` and wiped from Picasso (Mario's request) | orchestrator; `docs/RESULTS/submissions.md` §7 |
| D20 (evaluation precision, W10 merge) | **The evaluation array samples with `--amp fp16` for all 30 runs** (`TIME_LIMIT=07:30:00`). Measured on an A100 (T5.1, job 2432221): 2.425 against 3.233 s/chain (1.33×; 169 against 223 A100-h for the campaign, ≈ 7 h less makespan); on the same 500 seeds and noise stream the LSD moves by 5.1e-5, 0.11 % of the `ixi` noise floor (0.047) and 2 % of the gate's measured resolution (±0.0026). bf16 is equally fast and moves the LSD 13× more. The EMA and recipe-v2 weights stay ≤ 0.53 % of the fp16 ceiling (T3.4 `nan_diagnosis.md` §3a), and a non-finite draw now fails the task (T5.1). Rule: a run whose fp16 draw fails is re-evaluated entirely with `--amp off` and flagged; the gate job uses the same mode as the array. D16's 3.2 s/chain was measured without AMP, so the D16 budget was correct as written | T5.1 `evaluation_plan.md` §2–§4; T3.4 |
| D21 (2026-09-26, array 2 at 16/30) | **(a) Training stays at 40k: Mario's decision, a deviation from the pre-registered gate.** Gate job 2432703 (fp16, 500 seeds): `ixi_A0_s1` LSD 0.2671 → 0.2576 from 35k to 40k, difference +0.0095, 95 % CI [+0.0076, +0.0114], so D10/D17 say extend. `lsun_church_A0_s1` went 1.3496 → 1.5047, difference −0.155, CI [−0.167, −0.143]. Reasons for not extending: ≈ 100 A100-h; the IXI gain is 3.6 % relative, below D10's original 5 % threshold; Churches did not improve. The report states the deviation and that the IXI LSD was still falling at 40k. **(b) Run 11 (`lsun_church_A3_s3`) is retrained from scratch** with the same recipe and seed (attempt 2). Attempt 1 had sporadic fp16 forward overflows from step 27,883 (6 skips) while its loss was healthy (eval 0.2965 at 28k). The overflow became persistent at 30,111 (29 skips in 26 steps), and the guard aborted at 30,136 (exit 3). The abort overwrote the rolling checkpoint (step 30,137), so no good resume point existed. Attempt 1 is moved to `fscratch/runs/ihdm_failed/lsun_church_A3_s3_attempt1/`. No other of the 16 finished runs logged a single skip. Fallback if attempt 2 aborts: retry once from its last good rolling checkpoint (see c); if that fails, report the cell with 2 seeds, flagged. **(c) Abort path:** under an enabled scaler the abort state goes to `checkpoints-meta/abort_step_<step>.pth` and never over `checkpoint.pth`; the `abort` line's `resume_saved` becomes `abort_state` (path or null). This applies to tasks started after the cluster pull (25–29 and run 11's attempt 2); the recipe is unchanged. **(d) From the training grids:** Churches samples are still blurry at 40k (A0 especially), consistent with its LSD. `lsun_church_A3_s1` shows a saturated vertical stripe at the right edge of every grid sample at 40k; seeds 2 and 3 do not. Both are flagged for T6.2 and the report | Mario (a, b, c); orchestrator; `docs/RESULTS/submissions.md` §8 |

## 5. Acceptance criteria of the orchestration (what "done" means for this planning session)

1. Levels 1 (specification, milestones, tickets) and 2 (engineering practices) written under
   `docs/SPECIFICATIONS/`; harnesses under `docs/HARNESSES/`; `docs/README.md` is the tree.
2. All four datasets preprocessed into the standard format, validated, with QC sheets; the
   spectral profile of the training splits reproduces the known results (MRI $\alpha$ steeper than
   photographs; coarse-octave share ≈1% on MRI against ≈25% on photographs; the log-schedule
   spread; the per-octave level counts of the matched schedules) and the schedule arrays are frozen.
3. Training harness verified locally (smoke test on CPU, pilot on the RTX 3060) with checkpoints,
   manifest and sanity metrics as specified; the offline sampler loads a pilot checkpoint.
4. Picasso: data copied, repository cloned, env created, a 3-epoch probe on `lsun_church` and
   `ixi` run and checked (artefacts present, iteration rate measured, resume works).
5. The full array (30 runs) submitted through the `picasso-sbatch` skill.
6. Every ticket has its log under `docs/AGENT-LOGS/`; the orchestrator's own log is
   `docs/AGENT-LOGS/ORCHESTRATOR-SESSION.md`.

## 6. Reading order for a new agent

1. This file. 2. `01-milestones.md`. 3. `02-engineering-practices.md`. 4. The contracts
`03-data-format.md` and `04-run-artifacts.md`. 5. Your ticket under `M<k>-<slug>/`. 6. The harness
files your ticket names. 7. `docs/AGENT-LOGS/TEMPLATE.md` (or the `ticket-log` skill).

The scientific background is not repeated here; read `projects/GenAI/learning/03-terminal-blur-scaffolding.md`
§16 and §21 in the TFM repository when a ticket points there.
