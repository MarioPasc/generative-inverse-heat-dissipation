# Collection of the evaluation results (T5.2)

`python -m ihdm.cli.collect_results` turns the outputs of the 30 evaluated runs into one folder,
`results/`, which T6.1 (tables, statistics) and T6.2 (figures) read and nothing else. It refuses
to publish that folder unless every integrity check passes. Code: `ihdm/analysis/collect.py`
(logic), `ihdm/cli/collect_results.py` (shell). Tests: `tests/analysis/test_collect.py`,
`tests/cli/test_collect_results.py`.

## 1. Inputs (read only)

| input | where on Picasso | naming |
|---|---|---|
| plain summary, one per run | `~/execs/ihdm/eval/` | `<run_id>_amp-fp16_summary.json` |
| tar of the shadow run, one per run | `~/execs/ihdm/eval/` | `<run_id>_amp-fp16.tar`; members `<run_id>/metrics_amp-fp16/{summary,final,ckpt_<step:06d>}.json` and the `.npy` sidecars; `samples_amp-fp16/` and the links are not read |
| gates, legacy name | `~/execs/ihdm/eval/gate/` | `<run_id>_amp-fp16_gate.json` = 35k/40k (job 2432703) |
| gates, pair name | `~/execs/ihdm/eval/gate/` | `<run_id>_amp-fp16_gate_<early:06d>_<late:06d>.json` (55k/60k: job 2486891) |
| run directories | `~/fscratch/runs/ihdm/<run_id>/` | `manifest.json`, `config.json`, `metrics.jsonl`, `grids/iter_060000.png` |
| cell table | the repository | `slurm/array/cells.csv` (30 rows) |
| seed-list digests | the repository | `slurm/eval/expected_seed_lists.csv` (`evaluation_plan.md` §5) |

The gate JSON is flat: `step_a`, `step_b`, `lsd_a`, `lsd_b`, `difference` (`point`, `ci_low`,
`ci_high`, `alpha`, `n`, `n_boot`, `excludes_zero`), `extend`, `n_seeds`, `amp`,
`seed_list_sha256`, … (checked on `ixi_A0_s1_amp-fp16_gate.json`, 2026-09-28). The plain gate
JSONs are authoritative. The `gate.json` inside an eval tar holds only the last pair unpacked
(T3.5 §6), so the collection never reads it.

## 2. Output

```
results/
├── collection.json   provenance: date, git sha of the collecting code, n_iters, amp, every input
│                     path with sha256 and size (summaries, tars, raw metrics.jsonl, manifests,
│                     configs, grids, gates, cells.csv, expected_seed_lists.csv), check verdicts
├── index.csv         one row per cell, in cells.csv order
├── gates/            <run_id>_gate_<early:06d>_<late:06d>.json, byte copies of both naming schemes
└── runs/<run_id>/    summary.json, ckpt_<step:06d>.json (12 at 60k), final.json,
                      final_memorisation_per_sample_{d,nn}.npy, final_pca_{components,mean,
                      train_scores}.npy, manifest.json, config.json, metrics.canonical.jsonl,
                      grid_final.png
```

- Every file is a byte copy of its input, except `metrics.canonical.jsonl`. That file holds the
  lines of `metrics.jsonl` that `validate_run.canonical_records` keeps, copied line for line. The
  raw file is not copied (it stays in the training archives); its sha256 is in `collection.json`.
- Samples are never copied; the tars are the archive.
- The folder is staged in `.<out>.partial-<pid>` next to `--out` and renamed into place at the
  end. `--out` must not exist: results are never overwritten.

### `index.csv` columns

`index, run_id, dataset, arm, seed, tier, n_iters, amp, n_checkpoints, checkpoint_steps,
final_step, lsd_final, lsd_final_ckpt, kid, kid_ci_low, kid_ci_high, fid, fid_ci_low, fid_ci_high,
fid_n_reference, recall, coverage, precision, density, M, M_lp, seed_nn_fraction, D_pix, D_lp,
inherited_measured, inherited_predicted, n_skipped, n_resumes, lsd_005000 … lsd_060000`, then per
gate pair found `gate_<early>_<late>_{diff,ci_low,ci_high,extend}` (at 60k:
`gate_035000_040000_*`, `gate_055000_060000_*`; empty on runs without a gate).

- `lsd_final` is `final.json` `lsd`, the 2,000-sample final set. `lsd_final_ckpt` is the
  500-seed set at the last checkpoint (`intermediate_lsd`).
- `checkpoint_steps` is `;`-separated.
- `n_skipped` counts the `skip` events of the canonical history: 6 on run 11, 0 elsewhere.
  `n_resumes` counts the `resume` events it keeps.
- T_tau is not a column. T6.1 computes it from `lsd_<step>` and the A0 run's `lsd_final`
  (`evaluation_plan.md` §8).

## 3. Checks

All must pass. A problem in an input that exists is `FAIL`; an absent input is `MISSING`, which
also fails unless `--allow-missing` is given.

| id | check | pinned by (`tests/analysis/test_collect.py::…`) |
|---|---|---|
| C1 | all 30 cells present; summary `run.{run_id,dataset,arm,seed}` and manifest identity equal the row; no summary for a run outside the table | `test_c1_a_missing_run_fails_and_publishes_nothing`, `test_c1_allow_missing_publishes_a_partial_folder`, `test_c1_a_missing_run_directory_is_missing`, `test_c1_a_summary_of_another_cell_fails`, `test_c1_a_manifest_of_another_cell_fails`, `test_c1_a_summary_for_a_run_that_is_not_a_cell_fails` |
| C2 | `checkpoint_steps == evaluated_steps(n_iters)` (12 at 60k), `final_step`, `lsd_by_step` keys, every `ckpt_<step>.json` with its step, `final.json` at `n_iters`, manifest `n_iters` | `test_c2_a_40k_summary_fails_at_60k`, `test_c2_a_missing_checkpoint_record_fails`, `test_c2_a_checkpoint_record_of_the_wrong_step_fails`, `test_c2_a_manifest_of_another_length_fails` |
| C3 | `sampling.amp` and every result file's `amp` equal `--amp` | `test_c3_a_run_sampled_in_another_precision_fails`, `test_c3_a_result_file_of_another_precision_fails`, `test_c3_collecting_another_precision_finds_no_run` |
| C4 | summary seed-list digests, every `ckpt` and `final` `seed_list_sha256` equal `expected_seed_lists.csv` | `test_c4_a_foreign_seed_list_fails`, `test_c4_a_checkpoint_on_another_list_fails` |
| C5 | summary `run.config_sha256` equals manifest `config_sha256`; manifest `recipe_sha256` equals the hash of `config.json`'s recipe | `test_c5_a_config_hash_that_disagrees_with_the_manifest_fails`, `test_c5_a_recipe_hash_that_disagrees_with_the_config_fails` |
| C6 | strict JSON everywhere (no `NaN`/`Infinity`, which `json.loads` accepts); every `05-metrics.md` §9 key, the Inception block with FID/KID CIs, recall, coverage, `n_reference` | `test_c6_a_nan_token_fails`, `test_c6_a_nan_token_in_the_summary_fails`, `test_c6_a_missing_section_9_key_fails` (4 cases), `test_c6_a_final_without_inception_fails`, `test_c6_an_inception_block_without_its_ci_fails`, `test_c7_a_nan_token_in_the_history_fails` |
| C7 | canonical `metrics.jsonl` ends with `done` at `n_iters` and holds no `abort` | `test_c7_an_abort_not_abandoned_by_a_resume_fails`, `test_c7_a_history_that_stops_at_40k_fails`, `test_run_11_keeps_its_canonical_history_and_six_skips` |
| C8 | the tar's `summary.json` equals the plain one byte for byte; the five `.npy` sidecars and `grids/iter_<n_iters>.png` are present; the tar is readable | `test_c8_a_plain_summary_that_differs_from_the_tar_fails`, `test_c8_a_missing_sidecar_fails`, `test_c8_an_unreadable_tar_fails`, `test_c8_a_missing_final_grid_fails` |
| C9 | gates of `--amp` under both names: strict, keys, the name's pair equals `step_a`/`step_b`, `amp`, intermediate seed list of the dataset, no pair twice per run; cells 0 and 3 hold 35k/40k and `(n_iters − 5k, n_iters)`; gates of other precisions are ignored | `test_c9_a_missing_required_gate_is_missing`, `test_c9_a_gate_whose_content_contradicts_its_name_fails`, `test_c9_a_gate_of_another_precision_inside_ours_fails`, `test_c9_a_legacy_and_a_pair_named_gate_of_the_same_pair_fail`, `test_c9_a_gate_on_a_foreign_seed_list_fails`, `test_c9_gates_of_another_precision_are_ignored` |

Exit codes: **0** complete and published; **1** a check failed or an input is absent (nothing
published, except under `--allow-missing`); **2** unusable arguments (`--out` exists, a bad cell
table, `--n-iters` not a multiple of 5,000); **3** only absent inputs, and `--allow-missing`
published a partial folder marked `"complete": false`.

## 4. Commands

### 4.1 Dry run, any time (writes only under `/tmp`)

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation
export PYTHONPYCACHEPREFIX=/tmp/ihdm_pyc_$$
PY=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/ihdm/bin/python
EVAL=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/eval
RUNS=/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm
PYTHONPATH=$PWD $PY -m ihdm.cli.collect_results --eval-dir $EVAL --run-root $RUNS \
    --cells slurm/array/cells.csv --n-iters 60000 --amp fp16 --allow-missing \
    --out /tmp/ihdm_results_dry_$$ ; echo "exit=$?"
rm -rf /tmp/ihdm_results_dry_$$ /tmp/ihdm_pyc_$$
```

### 4.2 The collection, once every eval task is COMPLETED (login node)

The collection reads each tar once, to hash it and extract ≈ 20 small members, so it reads
≈ 17 GB from `$HOME`. It writes ≈ 30 × 1 MB into `~/execs/ihdm/results/` and nothing on FSCRATCH.

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation
git pull                                   # the merged T5.2; collection.json records this sha
export PYTHONPYCACHEPREFIX=/tmp/ihdm_pyc_$$
PY=/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/ihdm/bin/python
EVAL=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/eval
RUNS=/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm
PYTHONPATH=$PWD $PY -m ihdm.cli.collect_results --eval-dir $EVAL --run-root $RUNS \
    --cells slurm/array/cells.csv --n-iters 60000 --amp fp16 \
    --out /mnt/home/users/tic_163_uma/mpascual/execs/ihdm/results \
    | tee /mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/collect_results_$(date +%Y%m%d_%H%M).log
echo "exit=${PIPESTATUS[0]}"               # must be 0 and print VERDICT: COMPLETE
rm -rf /tmp/ihdm_pyc_$$
```

If it exits 1, nothing was written. Read the failing check's lines, fix the cause (usually by
resubmitting an eval index, `evaluation_plan.md` §6), and rerun. If an old `results/` must be
replaced, move it aside first (`mv results results_superseded_<date>`); the command never
overwrites.

### 4.3 Copies (workstation, then the SanDisk)

`$IHDM_DATA_ROOT/../results/` on the workstation is
`/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/../results/`, i.e.
`/media/mpascual/MeningD2/results/`.

```bash
# on the workstation
SRC=picasso:/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/results/
WS="$IHDM_DATA_ROOT/../results"
SANDISK=/media/mpascual/Sandisk2TB/research/spectral_allocation_heat_diffusion_project/results
test ! -e "$WS" && test ! -e "$SANDISK" || echo "a results folder exists already: move it aside"
rsync -a "$SRC" "$WS/"
rsync -a "$WS/" "$SANDISK/"
# verification: both must print nothing (content compared by checksum)
rsync -a --checksum --dry-run --itemize-changes "$SRC" "$WS/"
rsync -a --checksum --dry-run --itemize-changes "$WS/" "$SANDISK/"
python -c "import json,sys; c=json.load(open(sys.argv[1])); print(c['verdict'], c['git_sha'])" "$WS/collection.json"
```

The tars and the plain summaries are not part of `results/`. They stay in `~/execs/ihdm/eval/`
until `main` archives them next to the training archives on the SanDisk.
