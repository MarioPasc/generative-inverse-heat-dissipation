# 01 — Milestones and tickets

Milestones are **sequential**: M(k+1) starts when M(k)'s exit criteria hold. Tickets inside a
milestone are either parallel (‖, disjoint file ownership, run by two agents at once) or
sequential (→). At most two implementation agents run concurrently. Ticket files live in
`docs/SPECIFICATIONS/M<k>-<slug>/T<k>.<n>-<slug>.md`; each ticket ends with its acceptance list.

One stated exception to strict sequencing: M4 (metrics harness) may start once M3's probe job is
*queued*, because its inputs (the run-artefact contract) are frozen and the Picasso queue wait is
dead time. M3 still closes only when the full array is submitted.

| milestone | goal | tickets | exit criteria |
|---|---|---|---|
| **M0 Scaffolding** | the repository can host the new code: env, package skeleton, the data-format module, the standard-format dataset backend, tests, docs tree, ticket-log skill | T0.1 | `conda env create` works; `pytest` green; `ihdm.data.format` writes/reads/validates a synthetic dataset; `scripts/datasets.get_dataset` serves it; `python train.py` smoke config runs 5 iterations on CPU |
| **M1 Data** | the four datasets in the standard format at `IHDM_DATA_ROOT`, validated, with QC sheets; the spectral profile; the frozen schedules; the report's data figure | T1.1 ‖ T1.2 → T1.3 | four validated dataset folders of 4000 images; `docs/RESULTS/data_profile.md` with the known results reproduced; `schedules/*.npy` + `schedules.json`; `docs/RESULTS/fig_data.pdf` |
| **M2 Training harness** | the released trainer reads the standard format and a schedule file, runs every arm from one config factory, saves the specified artefacts, logs sanity metrics; the offline sampler works; a pilot on the RTX 3060 fixes the recipe | T2.1 ‖ T2.2 → T2.3 | smoke test green; pilot run of 1k iterations of A0 and A3 on `ixi` at $192^2$ with measured it/s, peak memory and stable loss; `sample_ckpt` draws 50 samples from 4 seeds of a pilot checkpoint; recipe frozen in `configs/spectral/arms.py` |
| **M3 Picasso** | env, data and repo on Picasso; a 3-epoch probe on `lsun_church` and `ixi` verified; the 30-run array submitted | T3.1 → T3.2 → T3.3 | probe artefacts complete and checked; it/s on A100 recorded; array submitted with job ids recorded in `docs/RESULTS/submissions.md` |
| **M4 Metrics harness** | LSD (+ octave profile, $T_\tau$), $M$, within-seed diversity, inherited band; statistics; tested on synthetic data and on the pilot checkpoint | T4.1 ‖ T4.2 → T4.3 | `pytest tests/metrics` green; `evaluate_run` produces `metrics/*.json` for the pilot run; statistics module reproduces a hand-computed bootstrap on toy data |
| **M5 Evaluation runs** | sampling and metrics for every checkpoint of every run on Picasso; results copied to `$HOME` | T5.1 → T5.2 | `results/<run_id>/metrics.json` for all 30 runs; copied to `~/execs/ihdm/results` |
| **M6 Analysis** | tables, CIs, permutation tests, figures for the report | T6.1 ‖ T6.2 | `docs/RESULTS/` holds the interaction tables with CIs, the LSD-vs-iteration curves, the diversity/M figure, the PCA-around-seed figure, the inherited-band figure |

## Ticket index

| id | title | mode | model / effort | depends on | owns (write access) |
|---|---|---|---|---|---|
| T0.1 | Scaffolding, env, data-format module, dataset backend | single | sonnet5-xhigh | — | `environment.yml`, `pyproject.toml`, `ihdm/__init__.py`, `ihdm/paths.py`, `ihdm/data/**`, `scripts/datasets.py` (new branch only), `tests/**`, `configs/spectral/smoke.py`, `.gitignore`, `README.md` (one section) |
| T1.1 | MRI pipeline: rigid MNI152 registration, slicing, windowing, splits, QC | ‖ with T1.2 | opus5-xhigh | T0.1 | `ihdm/preprocess/mri.py`, `ihdm/preprocess/registration.py`, `ihdm/preprocess/qc.py`, `ihdm/cli/preprocess_mri.py`, `tests/preprocess/test_mri*.py` |
| T1.2 | Photograph pipeline: fetch to 4000, centre crop, grayscale, splits, QC | ‖ with T1.1 | opus5-high | T0.1 | `ihdm/preprocess/photos.py`, `ihdm/preprocess/fetch_hf.py`, `ihdm/cli/preprocess_photos.py`, `tests/preprocess/test_photos*.py` |
| T1.3 | Spectral profile of the training splits, frozen schedules, data figure | single | opus5-xhigh | T1.1, T1.2 | `ihdm/spectral/**`, `ihdm/cli/profile_data.py`, `ihdm/cli/build_schedules.py`, `schedules/**`, `docs/RESULTS/data_profile.md`, `docs/RESULTS/fig_data.*`, `tests/spectral/**` |
| T2.1 | Trainer: schedule loader, arm config factory, checkpoint cadence, manifest, sanity logging, the two fixes | ‖ with T2.2 | opus5-xhigh | T1.3 | `train.py`, `scripts/losses.py`, `scripts/sampling.py` (prior-noise flag only), `scripts/utils.py`, `configs/spectral/arms.py`, `ihdm/train/**`, `tests/train/**` |
| T2.2 | Offline sampler from a checkpoint with seed images; checkpoint tools | ‖ with T2.1 | opus5-high | T1.3 | `ihdm/sampling/**`, `ihdm/cli/sample_ckpt.py`, `tests/sampling/**` |
| T2.3 | Local pilot on the RTX 3060: it/s, memory, stability, recipe freeze | single, GPU | opus5-high | T2.1, T2.2 | `docs/RESULTS/pilot_3060.md`, `configs/spectral/arms.py` (recipe values only) |
| T3.1 | Picasso: env, repo clone, data rsync, import check job | single | opus5-high | T2.3 | `slurm/**` (env + sync scripts), `docs/RESULTS/picasso_setup.md` |
| T3.2 | Picasso probe: 3 epochs of `lsun_church` and `ixi`, resume test, artefact check | single | opus5-high | T3.1 | `slurm/probe/**`, `docs/RESULTS/picasso_probe.md` |
| T3.3 | Full array submission through the `picasso-sbatch` skill | single | opus5-high (+ skill) | T3.2 | `slurm/array/**`, `docs/RESULTS/submissions.md` |
| T4.1 | Spectral metrics: LSD, octave profile, $T_\tau$, inherited band | ‖ with T4.2 | opus5-xhigh | T2.2 | `ihdm/metrics/spectral.py`, `tests/metrics/test_spectral.py` |
| T4.2 | Memorisation ratio $M$, within-seed diversity, PCA-around-seed | ‖ with T4.1 | opus5-xhigh | T2.2 | `ihdm/metrics/memorisation.py`, `ihdm/metrics/diversity.py`, `tests/metrics/test_mem*.py` |
| T4.3 | `evaluate_run` CLI, statistics (bootstrap over seeds, permutation test), result schema | single | opus5-high | T4.1, T4.2 | `ihdm/cli/evaluate_run.py`, `ihdm/stats/**`, `tests/stats/**` |
| T5.1 | Evaluation array on Picasso (sampling + metrics per checkpoint) | single | opus5-high (+ skill) | T3.3, T4.3 | `slurm/eval/**` |
| T5.2 | Collect results, copy to `$HOME`, integrity check | single | sonnet5-high | T5.1 | `ihdm/cli/collect_results.py`, `docs/RESULTS/collection.md` |
| T6.1 | Interaction tables, CIs, permutation tests, transfer signs | ‖ with T6.2 | opus5-xhigh | T5.2 | `ihdm/analysis/tables.py`, `docs/RESULTS/tables/**` |
| T6.2 | Figures: LSD vs iteration, diversity and $M$, PCA around seed, inherited band, sample grids | ‖ with T6.1 | opus5-high | T5.2 | `ihdm/analysis/figures.py`, `docs/RESULTS/figures/**` |

## Waves (two agents at once)

| wave | tickets | notes |
|---|---|---|
| W0 | T0.1 | contracts first |
| W1 | T1.1 ‖ T1.2 | both write only their own dataset folders under `IHDM_DATA_ROOT`; CPU only; T1.1 may use up to 20 of the 24 cores, T1.2 at most 4 |
| W2 | T1.3 ‖ T2.1 | T2.1 codes against the schedule-file contract; T1.3 produces the arrays |
| W3 | T2.2 ‖ T2.3 | T2.3 holds the GPU; T2.2 verifies loading on the pilot checkpoint after T2.3 reports the path |
| W4 | T3.1 → T3.2 → T3.3 | sequential; queue-bound |
| W5 | T4.1 ‖ T4.2, then T4.3 | may overlap with W4's queue wait |
| W6 | T5.1 → T5.2 | after the training array finishes |
| W7 | T6.1 ‖ T6.2 | |

## Milestone tickets not yet written in full

M5 and M6 tickets are sketched in their folders and are completed by the orchestrator when M4
closes, because their inputs depend on what M4 actually produced.
