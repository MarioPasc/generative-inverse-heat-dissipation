# `slurm/eval/`: evaluation of the 30 training runs on Picasso (T5.1)

Launcher: `submit_eval.sh`. Workers: `prepare_eval.sbatch`, `gate.sbatch`, `eval_array.sbatch`,
`timing.sbatch`. Shared shell helpers are in `common.sh`. The Python helpers are
`prepare_eval.py`, `timing_probe.py` and `expected_seed_lists.csv`. The sizing, the timing
table and the AMP recommendation are in `docs/RESULTS/evaluation_plan.md`.

All jobs run on one A100 (`--constraint=a100 --gres=gpu:1`, QOS `medium_uma`, account
`tic_163_uma`). They read code from `IHDM_REPO_DIR`, which defaults to main's cluster clone
`~/fscratch/repos/generative-inverse-heat-dissipation`, and use the env `IHDM_ENV_PREFIX`
(default `~/fscratch/conda_envs/ihdm`). Logs go to `~/execs/ihdm/logs/eval_*`.

## Order

Submit in this order. Each step says what it waits for.

1. **prepare**, once per campaign, before anything else:
   `bash slurm/eval/submit_eval.sh prepare`.
   It writes the one-writer inputs and exits. If its 24 files already exist from an earlier
   COMPLETED run, a rerun reads them back and writes nothing.
2. **gate**, on the first finished A0 runs of `ixi` (cell 0) and `lsun_church` (cell 3):
   `PREPARE_JOB=<prep> TRAIN_ARRAY=<train> bash slurm/eval/submit_eval.sh gate`.
   It gets `afterok:<train>_0:<train>_3:<prep>` and writes
   `~/execs/ihdm/eval/gate/<run_id>_gate.json`.
3. **main decides** from the two `gate.json` files whether to extend the campaign (D10). The
   decision stays with `main`. If the runs are extended, the array waits for the extended runs.
   The worker accepts `N_ITERS=<n>` and checks for `ema_iter_<n>.pt` and `DONE`.
4. **array**, one task per row of `slurm/array/cells.csv`:
   `PREPARE_JOB=<prep> TRAIN_ARRAY=<train> bash slurm/eval/submit_eval.sh array`.
   It gets `afterok:<prep>,aftercorr:<train>`, so eval task *i* starts when training task *i*
   COMPLETES. Once the prepare job has finished, drop `PREPARE_JOB`: the workers check the files
   themselves.

`--dry-run` prints the `sbatch` line and submits nothing. `--test-only` runs `sbatch --test-only`
without the dependency flags and submits nothing. Every real submission runs `--test-only`
first. It then parses the job id banner-safely (the last number on the last line) and cancels
the job if SLURM recorded `Dependency=(null)` for a requested dependency.

Resubmitting by index: `ARRAY_SPEC='6,17' bash slurm/eval/submit_eval.sh array`. A task
whose `~/execs/ihdm/eval/<run_id>_summary.json` exists is skipped unless `FORCE_EVAL=1`. A task
killed by TIMEOUT still writes its partial tar, and the next submission unpacks that tar and
resumes from the signature-checked sample cache. It first unpacks the gate job's tar, so the
two gate LSD sets are never drawn twice.

## What each job does

| job | reads | writes | wall |
|---|---|---|---|
| `prepare_eval` | the 4 datasets; `~/execs/ihdm/cache/inception-2015-12-05.pt` | per dataset: `eval_seeds_500.{npy,json}`, `eval_seeds_final_2000.{npy,json}`, `_features_inception_ref.{npy,json}`, 24 files in total and nothing else on FSCRATCH; `~/execs/ihdm/eval/prepare_<job>.json` | 1 h |
| `gate` | cells 0 and 3 (read only), the seed lists | `~/execs/ihdm/eval/gate/<run_id>_gate.{json,tar}` | 3 h |
| `eval_array` | one run (read only), the 6 cache files of its dataset | `~/execs/ihdm/eval/<run_id>.tar` and `<run_id>_summary.json` | see plan |
| `timing` | the fixture `~/execs/ihdm/fixtures/array_2408239/ixi_A0_s2` (read only), `ixi` | `~/execs/ihdm/eval/timing_<job>/` (JSON records, one tar of samples) | 2 h |

**One writer.** The torchscript Inception is not bitwise deterministic across calls (T4.3
§6), so every run of a dataset must be scored against one stored reference matrix. Only
`prepare_eval` writes it. `eval_array` and `gate` refuse to start while any of the six files of
their dataset is missing, because `evaluate_run` would otherwise write its own copy. Before
`prepare_eval` touches FSCRATCH, it draws the seed lists into a shadow copy of each dataset on
`$LOCALSCRATCH`. It compares their sha256 with `expected_seed_lists.csv`, which was computed on
the workstation with the same code, and stops with exit 3 on any mismatch.

**Local disk.** Every evaluation runs on a *shadow run* in `$LOCALSCRATCH`. The shadow run
links `config.json`, `manifest.json`, `checkpoints/`, `grids/` and `metrics.jsonl` from the
real run, so the thousands of sample files land on the node and never in the run directory.
Each run comes back as one tar plus one plain `summary.json`. A `trap` on EXIT and TERM writes
the tar even on TIMEOUT.

**Inception weights.** clean-fid hard-codes `/tmp/inception-2015-12-05.pt` and would try to
download to it, but the compute nodes are offline. `common.sh` therefore copies the file from
`~/execs/ihdm/cache/` to `/tmp` under a temporary name, checks its sha256 (`f58cb9b6…`) and
renames it into place.

**pycache.** Every worker exports
`PYTHONPYCACHEPREFIX=${TMPDIR:-/tmp}/ihdm_pycache_<job>_<task>` on its first line, before any
python starts. The env on FSCRATCH ships no `.pyc` files, and the first import would otherwise
write about 900 of them into the env (T3.4). `ihdm_env_setup` refuses to continue when the
prefix is unset or points into FSCRATCH.

## The gate command, and why `--ckpts` names only the earlier step

```
evaluate_run --run <shadow> --ckpts 35000 --gate 35000,40000 --n-lsd 500 --skip-inception
```

`evaluate_run` draws the final (2,000) and held-out (40 × 50) sets at the **largest available**
checkpoint of the run, and only when that step is among `--ckpts`. `--ckpts all` or
`--ckpts 35000,40000` therefore costs 5,000 chains, while the form above draws exactly the
two 500-seed LSD sets (1,000 chains). The gate step draws the 40,000 set itself. The
`summary.json` of this command covers step 35,000 only; the gate's result is `gate.json`. The
behaviour is pinned by
`tests/metrics/test_run_eval.py::test_the_gate_command_draws_only_the_two_lsd_sets` and by its
contrast test `test_naming_the_last_step_in_ckpts_also_draws_the_final_sets`.

## `--amp`

`AMP` (`off` | `fp16` | `bf16`, default `off`, the D16 contract) is passed through to
`evaluate_run --amp`. The files are keyed by precision:

- the samples go to `samples_amp-<mode>/` and the results to `metrics_amp-<mode>/`;
- the returned files are `<run_id>_amp-<mode>.tar` and `<run_id>_amp-<mode>_summary.json`;
- `off` keeps the historical names `samples/` and `metrics/`, so no two precisions ever share a
  cache or a result file.

Change the mode only on `main`'s decision; see `docs/RESULTS/evaluation_plan.md`.

## File budget

| where | files |
|---|---|
| FSCRATCH | 24 (prepare), once; nothing per run |
| `$HOME` | 2 per run (tar + summary) + 2 logs per task = 120 for 30 runs; 4 for the gate + 2 logs; prepare and timing about 15 |
