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
