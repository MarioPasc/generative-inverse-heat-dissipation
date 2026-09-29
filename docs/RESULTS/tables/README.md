# Tables of the report (T6.1)

This folder holds the tables the report cites and states how each one was computed. They are
generated. Never edit a `t*.md`, `t*.tex` or `tables.json` by hand; rerun the command instead.

```bash
cd <repository>
PYTHONPATH=$PWD python -m ihdm.cli.analyse --results <results_dir> --out docs/RESULTS/tables/
```

- **Input:** one folder written by `python -m ihdm.cli.collect_results`
  (`docs/RESULTS/collection.md`): `collection.json`, `index.csv`, `runs/<run_id>/summary.json`
  and `gates/`. Nothing else is read.
- **Output:** per table, `<name>.md` and `<name>.tex` (a `booktabs` float with
  `\label{tab:<name>}`, needs `\usepackage{booktabs}`), plus one `tables.json`. The JSON holds
  the raw numbers of every row, the provenance (results folder, collection verdict and git sha,
  the analysing git sha) and the method. The command overwrites its own files and leaves every
  other file here alone, including this README and `PARTIAL_24_RUNS/`.
- **Exit codes:** 0 every table complete; 1 the folder cannot be analysed (collection verdict
  `FAIL`, or `index.csv` disagrees with a run's `summary.json`); 2 `--results` is not a results
  folder; 3 tables written but at least one is incomplete.
- **Checks before any statistic:** the collection verdict must not be `FAIL`; every row of
  `index.csv` must be a design cell (`configs/spectral/arms.py` `EXPERIMENT_CELLS`) with its own
  identity; a run counts as evaluated only if it has both a non-empty `index.csv` row and a
  `summary.json`; every number of `index.csv` must equal the run's `summary.json` (read through
  `ihdm.stats.cell_table`).
- **Missing runs:** a table that needs a run the collection lacks is still written. It is marked
  **`incomplete: n/N runs`** in its status line, its LaTeX caption and `tables.json`, and the
  rows that need the missing run carry no statistic. No row is dropped. When the collection is
  not complete, every file also carries a "PARTIAL COLLECTION … not for quoting" banner.
- **Layout:** some tables are wide. Every `.tex` compiles with `pdflatex` (checked on the
  partial output), but on a landscape A4 page with 1 cm margins, 2a, 2b and 5 still run 7–62 pt
  over the text width. The report may need `\resizebox{\textwidth}{!}{…}` around
  the `tabular`, or a landscape page.

## Read this first: what "CI excludes 0" means with 3 seeds

`05-metrics.md` §8 asks for a percentile bootstrap over seeds (10,000 draws, 95%). With **3
seeds**, the bootstrap mean of a resample equals the smallest of the three paired differences
whenever all three draws hit it, which happens with probability 1/27 = 3.7% > 2.5%. The 95%
interval is therefore **exactly [min, max] of the three per-seed differences**. The consequences:

- "CI excludes 0" means **"all three seeds agree in sign"**, nothing more. Under a symmetric null,
  that happens with probability 2 × (1/2)³ = 25%, so the nominal 95% interval covers the null only
  75% of the time. It is a sign test at p = 0.25 two-sided (0.125 one-sided).
- With **2 seeds** (A1, A2, A2′ and the transfer cells) the interval is [min, max] of two values
  and its coverage under the null is 50%.
- The permutation p-value cannot go below its floor: **p_min = 0.1** with 3 vs 3 seeds
  (C(6,3) = 20 assignments) and **1/3** with 2 vs 2 (C(4,2) = 6). Every p is printed with its
  p_min. At these seed counts no p-value can fall below 0.05, so the permutation test can never
  reject at the conventional level.

The tables follow §8's wording: a CI containing zero is reported as "not detectable at this
budget", and one that excludes zero as "CI excludes 0". Neither is a significance statement.

## The tables

| # | file | what it shows | method |
|---|---|---|---|
| 1a | `t1a_cells_fidelity` | one row per run: final LSD (2k set), LSD at the last checkpoint (500 seeds), T_τ, KID with CI, FID with CI and N_ref, recall, coverage | values as collected; T_τ as in table 7 |
| 1b | `t1b_cells_mechanism` | one row per run: M, M_lp, seed-NN fraction, D_pix, D_lp, inherited share measured/predicted (all modes and the σ_n ≥ 8 px low band), `n_skipped` | values as collected; low band from `summary.json` via `cell_table` |
| 2a, 2b | `t2a_contrasts_ixi`, `t2b_contrasts_lsun_church` | A3−A0, A1−A0, A2−A0 (and A2′−A0 on Churches) for every endpoint: n seeds, mean Δ, 95% CI, per-seed Δ, permutation p, p_min | paired contrast (below) |
| 3 | `t3_interaction` | **the headline**: Δ_IXI − Δ_Churches for A3 vs A0, every endpoint, with CI and p | interaction (below) |
| 4 | `t4_decomposition` | share of the A3 effect carried by A1 (terminal blur alone) and by A2 (spacing alone), per development dataset | ratio of seed-matched means, no CI |
| 5 | `t5_a2p_control` | A2′−A0, A2−A0 and A2′−A2 on Churches | paired contrast, 2 seeds |
| 6 | `t6_transfer` | sign and magnitude of A3−A0 on IXI vs OASIS-1 and on Churches vs Bedrooms | per-seed Δ, means, Δ / A0; no p-values |
| 7 | `t7_t_tau` | T_τ per run under the primary and the sensitivity threshold, with the exploratory settling step | 05 §2 (below) |
| 8 | `t8_gates` | the plateau gates: 35k vs 40k and 55k vs 60k | as written by the gate jobs |
| 9 | `t9_settle_exploratory` | **exploratory**: settling step per arm, A3−A0 and the interaction, descriptively | see "Exploratory" |

**Endpoints of tables 2–6**, in this order: final LSD (2k set; lower is better), T_τ (lower is
earlier), KID (the headline Inception metric; lower is better), FID (lower is better; see the
caveat below), recall and coverage (higher is better), M and M_lp (1 = as far from the training
set as held-out real images, below 1 = copying), seed-NN fraction (share of samples whose nearest
training image is their own seed), D_pix and D_lp (higher = more within-seed diversity), the
measured inherited share (read against the predicted share of table 1b), and T_τ under the 2k
threshold (sensitivity row). Definitions: `docs/SPECIFICATIONS/05-metrics.md` §2–§7.

## Statistical method (`05-metrics.md` §8, D13, D16, D17)

- **Paired contrast** (tables 2, 5): Δ = m(arm) − m(reference), paired by seed over the seeds both
  cells have (seed 1 with seed 1, …). The CI comes from `ihdm.stats.paired_delta`: a percentile
  bootstrap over seeds, 10,000 draws, α = 0.05, `rng_seed` 0, so reruns are identical. p comes
  from `ihdm.stats.permutation_test`: an exact two-sided permutation of the arm labels over the
  pooled seeds of the two cells, restricted to the paired seeds so that its statistic equals the
  mean Δ. For A1, A2 and A2′ this uses A0 seeds 1 and 2.
- **Interaction** (table 3): Δ_IXI − Δ_Churches from the per-seed A3−A0 differences (3 + 3).
  The CI comes from `ihdm.stats.interaction`, which resamples the two sides independently
  (10,000 draws, percentile). p comes from an exact permutation of the dataset label over the six
  pooled differences (20 assignments, p_min = 0.1). Only the development pair enters. OASIS-1 and
  Bedrooms have 2 seeds and A0/A3 only, and they appear in table 6.
- **Decomposition** (table 4): share A1 = Δ_A1 / Δ_A3 and share A2 = Δ_A2 / Δ_A3. All three Δ
  are taken against A0 on seeds 1 and 2, the seeds A1 and A2 have, so they share their A0 values.
  A sum of the two shares near 1 means the two knobs add up to the A3 effect. There is no CI,
  because A1 and A2 have 2 seeds. A share is flagged "not interpretable" when the 3-seed A3−A0
  interval of table 2 contains 0.
- **A2′ control** (table 5): A2′ uses spacing matched to Churches, A2 spacing matched to IXI,
  both at σ_B,max = 96. A2′−A2 is the same paired contrast between two non-A0 arms. It asks
  whether matching the spacing to the data's own spectrum matters on photographs. Everything
  here has 2 seeds (p_min = 1/3).
- **Transfer** (table 6): the sign of the mean A3−A0 on the development dataset (3 seeds) and on
  the transfer dataset (2 seeds), whether the signs agree, the per-seed transfer differences,
  and Δ / mean(A0) as a magnitude that can be compared across datasets. No p-values (§8: "sign
  agreement"). The A3 schedule is the IXI-matched one on every dataset, transferred frozen (D12).
- **Not applied: the sample-level resampling for LSD.** §8 asks to resample samples as well as
  seeds for LSD. The sample stacks are not in `results/` (they stay in the evaluation tars), and
  the ticket's only input is `results/`, so every LSD interval resamples seeds only. The seed
  spread already contains each seed's own sampling noise. A two-level bootstrap would widen the
  intervals somewhat, not change their centre.

## T_τ (table 7)

`05-metrics.md` §2: "$T_\tau(\text{arm})$ is the smallest checkpoint step $s$ at which
$\mathrm{LSD}_{\text{arm}}(s) \le \mathrm{LSD}^{A0}_{\text{final}}$", with
$\mathrm{LSD}^{A0}_{\text{final}}$ "the LSD of the A0 run with the same seed at its last
checkpoint", and "+∞ (reported as 'not reached') if never". The curve is the run's 12 checkpoint
LSDs at 5k, 10k, …, 60k in `index.csv`. Each point uses the 500 frozen evaluation seeds with the
same noise stream at every checkpoint and in every arm (D17).

- **Primary threshold:** the same-seed A0 run's LSD at 60k **on the curve's own estimator**, the
  500-seed set (`lsd_060000`). Curve and threshold then come from one sample set. For A0 itself,
  T_τ is at most 60k by construction. It is earlier when the A0 curve dipped below its final
  value before the end.
- **Sensitivity threshold:** the A0 run's 2,000-sample final LSD (`lsd_final`). Two earlier
  documents (`docs/RESULTS/evaluation_plan.md` §8, `docs/RESULTS/collection.md`) named this one.
  It was replaced as primary (`main`, 2026-09-29) because it comes from another sample set. The
  set difference is 0.003 on IXI A0 s1 (0.2527 vs 0.2497), as large as the gate's measured
  resolution (±0.0026), and that difference alone can decide T_τ. Both are printed.
- **T_τ is degenerate on this data, and it is reported as pre-registered anyway.** The LSD
  curves are non-monotone (next section), so the first crossing fires at the first checkpoint:
  **T_τ = 5k in 20 of the 24 runs evaluated on 2026-09-29**. The exceptions are
  `ixi_A0_s2` (35k), `oasis1_A0_s1` (60k), `oasis1_A0_s2` (50k) and `lsun_church_A2_s2` (never
  reached). A T_τ contrast is "not computable" whenever a seed of either cell never reaches the
  threshold; nothing is imputed. At this budget T_τ carries no information about the spacing.
  Table 9 gives an exploratory alternative.

## Honesty notes

- **Training length 60k.** Every run was first trained to 40k. The pre-registered plateau gate
  (D10/D17: extend when the 95% CI of LSD(35k) − LSD(40k) over the 500 frozen seeds lies above
  zero) then triggered the extension of all 30 runs to 60k by resume (D22). An informational
  55k-vs-60k gate found no further gain. Numbers from `results/gates/` (table 8):

  | run | checkpoints | LSD early → late | early − late [95% CI] | extend |
  |---|---|---|---|---|
  | `ixi_A0_s1` | 35k vs 40k | 0.2671 → 0.2576 | +0.0095 [+0.0076, +0.0114] | yes |
  | `lsun_church_A0_s1` | 35k vs 40k | 1.3496 → 1.5047 | −0.1551 [−0.1673, −0.1429] | no |
  | `ixi_A0_s1` | 55k vs 60k | 0.2476 → 0.2497 | −0.0021 [−0.0036, −0.0007] | no |
  | `lsun_church_A0_s1` | 55k vs 60k | 1.3653 → 1.4454 | −0.0801 [−0.0909, −0.0705] | no |

  The 35k/40k gate said extend on IXI only. D21 at first kept 40k, and D22 then reversed it and
  extended every run, following the gate.
- **fp16 evaluation sampling** (D20). All 30 runs were sampled with `--amp fp16`. On the same
  500 seeds and noise stream, fp16 moves the LSD by 5.1e-5, 0.1% of the IXI noise floor (0.047,
  `docs/RESULTS/metrics_bracket.md` §1).
- **Churches is undertrained.** Its final LSD is 1.0–1.45 for A0 (1.45 / 1.13 / 1.02), against
  0.23–0.25 on IXI. It oscillates across checkpoints (A0 s1: 1.35–1.50 over 35k–60k; A3 s1: 0.22
  at 40k, 0.91 at 45k), and the samples are blurry. Its contrasts are reported and
  interpreted with that caveat.
- **Run 11** (`lsun_church_A3_s3`) was trained in 3 attempts. The final one resumed from its
  30,001 rolling state (D21) and ran with no skip or abort after the resume. The run carries 6
  skipped steps: isolated fp16 overflow no-ops before 30,001, kept in the canonical history
  (`n_skipped` in table 1b).
- **`lsun_church_A3_s1`:** a saturated right-edge stripe appeared in its training grid at 40k
  and is no longer visible at 60k (T6.2).
- **FID.** Against an 800-image reference, FID is biased upward (bias ∝ 1/N_ref). Its bootstrap
  interval over resampled samples is shifted upward too, because a resample holds duplicates
  (T4.3), so the printed FID CI can lie above the point estimate. The bias cancels in
  differences between arms of one dataset. **KID is the headline.** No absolute FID is comparable
  to the paper's.
- **Raw scales in the interaction.** The interaction is on each metric's raw scale, as
  pre-registered. Churches' A0 LSD is about five times IXI's (≈ 1.20 vs ≈ 0.24). On raw LSD,
  Churches' A3 gain (−0.37) is therefore larger than IXI's (−0.16), although relative to A0 it
  is −31% against −68%. The raw interaction on LSD is dominated by the dataset with the larger
  scale. Table 3 prints Δ / A0 beside it (see "Exploratory").
- **The LSD curve is non-monotone**, an observation to explain and not a claim. On the 24
  evaluated runs, every curve is higher somewhere in 10k–30k than at 60k. In 14 of them the 5k
  value is already at or below the 60k value. IXI A0 runs 0.21–0.24 at 5k, 0.28–0.34 at its
  10k–30k peak and 0.23–0.25 at 60k. Churches A0 runs 0.40–0.78 at 5k, 1.43–1.66 at its peak and
  1.02–1.45 at 60k. So an early model matches the reference's radial variance profile better
  than the models of the middle of training. Why is open. The per-octave profiles
  (`ckpt_<step>.json` `lsd_octaves`) and the samples at 5k against 20k (T6.2) are where to look.
  This is what makes the first-crossing T_τ fire at 5k.

## Exploratory (not pre-registered)

Everything in this section was **defined after seeing the data (2026-09-29) and is not
pre-registered**. It is descriptive only: no CI and no p-value, so none of it is confirmatory.

- **Settling step** (table 9, and the last column of table 7). This is the first checkpoint after
  which the run's LSD curve stays at or below the primary T_τ threshold through the last step
  (`ihdm.analysis.tables.t_settle`). It is not reached when the 60k value is above the threshold.
  Table 9 gives, per dataset and arm, the per-seed settling steps of the arm and of A0, their
  means and the mean paired difference. Its last row is the A3−A0 difference on IXI minus that
  on Churches. It exists because the pre-registered first crossing is degenerate on these
  curves.
- **Δ / A0 columns in table 3** (and in table 6, where §8's "magnitudes" asks for them). Each is
  the mean paired Δ divided by the mean A0 value over the same seeds, a scale-free reading beside
  the pre-registered raw-scale interaction, which stays the headline column. They are printed
  only for positive ratio-scale endpoints (LSD, T_τ, KID, FID, M, M_lp, D_pix, D_lp). They are
  blank for recall, coverage and the seed-NN fraction, whose A0 values sit near 0 on Churches
  (a +0.09 recall gain would read as +3,000%), and for the inherited share, which is negative
  on IXI A0 and would flip the sign.

## `PARTIAL_24_RUNS/`

The output of this command on the **24-run partial collection** of 2026-09-29
(`$IHDM_DATA_ROOT/_results_partial_24/`, collection verdict `INCOMPLETE`). The tier-3 cells 24–29
were not yet evaluated: `lsun_bedroom_A0_s1/s2`, `oasis1_A3_s1/s2` and `lsun_bedroom_A3_s1/s2`.
It is a sample of the format, kept to show how incomplete tables are marked. Every file in it is
bannered "not for quoting". The report quotes the tables in this folder, which `main` writes
from the final 30-run collection.
