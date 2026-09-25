# `slurm/array/` — the 30-run training array (T3.3)

The experiment itself: 30 runs × 40,000 iterations at batch 16 on one A100 each, producing the EMA
checkpoints every 2,500 iterations that every downstream metric reads. Submission record:
`docs/RESULTS/submissions.md`; log: `docs/AGENT-LOGS/M3-picasso/T3.3-full-array-submission.md`;
harness: `docs/HARNESSES/picasso.md` §5–§6.

## Files

| file | what |
|---|---|
| `cells.csv` | the run table: `index,run_id,dataset_id,arm,seed,tier`, indices 0–29 in tier order |
| `make_cells.py` | derives `cells.csv` from `configs.spectral.arms.EXPERIMENT_CELLS` (`--write` / `--check`) |
| `train_array.sbatch` | worker, A100: decodes its row by `$SLURM_ARRAY_TASK_ID` and runs `train.py` |
| `submit_array.sh` | launcher, login node: validates the array spec, `sbatch --test-only`, submits, captures the job id |
| `../../tests/slurm/test_cells.py` | pins the 30 rows, the bijection with `EXPERIMENT_CELLS` and the tier order |

`cells.csv` is generated, not hand-written. After any change to `EXPERIMENT_CELLS`:

```bash
python slurm/array/make_cells.py --write    # regenerate
python slurm/array/make_cells.py --check    # fail if the committed file is stale
pytest tests/slurm/test_cells.py
```

## Tier order (why the indices are what they are)

Sorted by tier, then arm, then dataset, then seed, so the lowest indices — which the scheduler
starts first — are the cells the plateau gate D10 is read from.

| indices | tier | cells |
|---|---|---|
| 0–11 | 1 | `{ixi, lsun_church} × {A0, A3} × {1,2,3}` — the primary contrast on the development pair |
| 12–21 | 2 | `{ixi, lsun_church} × {A1, A2} × {1,2}`, then `lsun_church × A2p × {1,2}` |
| 22–29 | 3 | `{oasis1, lsun_bedroom} × {A0, A3} × {1,2}` — the transfer pair |

Index 0 is `ixi_A0_s1` and index 3 is `lsun_church_A0_s1`: the two runs D10 needs.

## Resources (D16, measured on the A100 by the T3.2 probe, job 2405546)

| flag | value | why |
|---|---|---|
| `--constraint=a100 --gres=gpu:1` | one A100 | `exa[01-04]` advertise an **untyped** `gpu:8` with the feature `a100`; `--gres=gpu:A100:1` matches no node and a bare `--constraint=dgx` would also match the B200 nodes |
| `--time=11:00:00` | 11 h | 1.5 × the measured 6.84 h for 40k iterations at 1.71 it/s |
| `--qos=medium_uma` | 3-day wall | the probe's `short` caps at 2 h and would TIMEOUT every task |
| `--cpus-per-task=8` | 8 | data loading; `OMP_NUM_THREADS`/`MKL_NUM_THREADS` follow `SLURM_CPUS_PER_TASK` |
| `--mem=32G` | host RAM | probe MaxRSS 18.4 GB |
| `--array=0-29%8` | 8 concurrent | QOS `medium_uma` allows gpu=23 per user; 8 is one node's worth and leaves the cluster usable |

Nothing else is overridden: batch 16, lr **1e-4** (recipe v2, D19; array 2408239 ran 2e-4),
`ckpt_every=2500` and the skip limits are frozen in `configs/spectral/arms.py` (D4″, D4‴, D5″,
D19). The only run-varying flags are `--config.seed`, `--config.training.n_iters` and
`--workdir`.

## Exit codes, the skip policy and the recipe check (D19, T3.4)

| `train.py` exit | worker `status=` | meaning | what to do |
|---|---|---|---|
| 0 | `ok` | finished, or already past `N_ITERS` | nothing |
| 3 | `FAILED_nonfinite_guard` | the non-finite-loss guard aborted: 10 consecutive or more than 100 skipped steps (`training.max_consecutive_skips`, `training.max_skips`), or one non-finite loss with the `GradScaler` disabled | read the `skip`/`abort` lines of `metrics.jsonl`; the rolling checkpoint holds the weights of the last non-finite loss (a diagnosis fixture: `python -m ihdm.cli.diagnose_nan --fixture <run>`); do **not** resubmit blindly — the D19 lr ladder is 1e-4 → 5e-5 → stop |
| 4 | `FAILED_recipe_mismatch` | the resume was refused: the run directory was trained with another recipe (any config key except `training.n_iters` and the cadences `ckpt_every`, `resume_every`, `log_every`, `eval_every`, `grid_every` and their released aliases); nothing was written, not even a `resume` line | point the task at an empty run root, or at the run's own recipe; the refused keys are in the `.err` log (`Refusing to resume: … optim.lr: 0.0002 -> 0.0001`) |
| 1 | `FAILED` | a Python error (or a setup `[FATAL]` in the worker) | read the `.err` log |
| 137 / 143 | `FAILED` | killed (TIMEOUT, `scancel`, node failure) | resubmit the index: it resumes |

**Skip policy.** Under fp16 AMP with an enabled `GradScaler`, a non-finite loss makes the
optimiser step a no-op (the scaler skips it and halves its scale); the trainer logs
`{"kind": "skip", "loss": null, "n_skipped": k, "consecutive": c}` and continues. The count `k`
survives a resume (it is replayed from `metrics.jsonl`) and appears in the final
`{"kind": "done", "n_skipped": k}` line. A handful of skips over 40k steps is benign; a `skip`
line in the first v2 array is still worth a look, because v1 had none before its aborts.

**Recipe check.** Every resume compares the invocation's recipe with the run's
(`manifest.json: recipe_sha256`, key diff from `config.json`). It is what makes a blanket
resubmission safe: a run root that still holds array-1 (lr 2e-4) directories refuses to
resume them (exit 4) instead of continuing them under the v2 label. The schedule file path is
part of the recipe, so a resume must run from the same clone path.

**Logged scalars.** `grad_norm` on every train line is the pre-clip global norm of the unscaled
gradients (array 1 logged the post-clip value, identically 1.0; recipe v2 measured ≈ 400–1000 at
step 500 against the clip of 1.0, i.e. every step is clipped); `amp_scale` is the `GradScaler`
scale after the step (a power of two; `null` without AMP). `null` `grad_norm` marks a step whose
fp16 gradients overflowed (the scaler skipped it), normal for the first steps of a run and of
every resume (the scale restarts at 65536).

**`.pyc` files.** The worker exports `PYTHONPYCACHEPREFIX` to node-local `$TMPDIR`/`/tmp`:
the env ships no compiled `.pyc` for sympy, triton, pip, PIL, …, so without it the first task
writes ≈ 0.9k `.pyc` files and ≈ 0.1k `__pycache__` directories into the env on FSCRATCH, which
sits at its file quota (measured by T3.4, 2026-09-25).

**Tested on loginexa before submission.** `slurm/loginexa/harness.sh` item H2 runs this worker
with bash and hand-set SLURM variables on every one of the 30 cells (`N_ITERS=2`, V100 overlay);
see `docs/HARNESSES/training.md` §7 and `docs/RESULTS/loginexa_harness.md`.

Output goes to `$IHDM_RUN_ROOT` on FSCRATCH, not `$LOCALSCRATCH`: a run writes ≈ 40 files over
6.8 h, which is not the thousands-of-small-files pattern the manual's §4.9 rule is about, and
staging would break the resume rule below — the trainer reads the previous submission's rolling
checkpoint out of the run directory. T5.1's evaluation, which does write thousands of samples,
uses `$LOCALSCRATCH`.

## Submit

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation
git fetch && git checkout ticket/T3.3-full-array-submission && git log -1

quota                                          # file-count headroom: ~40 files per run
bash slurm/array/submit_array.sh --dry-run     # print the sbatch line, submit nothing
bash slurm/array/submit_array.sh --test-only   # sbatch --test-only, submit nothing
bash slurm/array/submit_array.sh               # --test-only, then the real submission
```

The launcher derives `0-29` from `cells.csv` rather than hard-coding it, refuses an `ARRAY_SPEC`
naming an index the table does not contain, runs `sbatch --test-only` and aborts on failure, and
parses the job id past the Lua wrapper's ANSI banner (last number of the last line).

## Monitor

```bash
squeue                                                    # Picasso's wrapper rejects `-u`
squeue -j <id> -o "%.14i %.12j %.8T %.10M %.6D %R"        # task by task
squeue --start -j <id>                                    # estimated start of pending tasks
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,NodeList | head -40
sacct -j <id> -X -n -P -o State | sort | uniq -c          # state histogram
tail -f ~/execs/ihdm/logs/train_<id>_0.out

# which runs have finished (H-PICASSO §5):
for r in $(cut -d, -f2 slurm/array/cells.csv | tail -n +2); do
    test -f "$IHDM_RUN_ROOT/$r/DONE" && echo "DONE $r" || echo "---- $r"
done
```

`MaxRSS` is recorded on the `.batch` step, so drop `-X` when you want memory. A PENDING task's
`REASON` is the diagnosis: `Resources`/`Priority` are normal queue pressure, while
`ReqNodeNotAvail` or `BadConstraints` means the request can never be satisfied.

## Resubmit one index (resume after a TIMEOUT or a node failure)

`train.py` resumes from `checkpoints-meta/checkpoint.pth` (written every 500 iterations) and its
loop is `range(initial_step, n_iters + 1)`, so resubmitting an index simply continues that run:

```bash
ARRAY_SPEC='6' bash slurm/array/submit_array.sh            # one index
ARRAY_SPEC='6,22,27' bash slurm/array/submit_array.sh      # several
```

Every task that did not end `COMPLETED`, in one line:

```bash
ARRAY_SPEC="$(sacct -j <id> -n -X -o JobID,State \
    | awk '$2 != "COMPLETED" {split($1, a, "_"); print a[2]}' | paste -sd, -)%8" \
    bash slurm/array/submit_array.sh
```

A blanket `bash slurm/array/submit_array.sh` is also safe: the worker skips, in seconds and before
python starts, any cell whose `DONE` marker exists and whose highest `ema_iter_XXXXXX.pt` is at
least `N_ITERS`.

## Extend every run (the plateau gate, D10)

When the first A0 runs finish, compare their LSD at 35,000 and 40,000 iterations. If the relative
change exceeds 5% on either dataset, extend the whole array — no cell is ever retrained from
scratch with a different count:

```bash
N_ITERS=60000 bash slurm/array/submit_array.sh
```

Each task resumes from its rolling checkpoint and trains on to 60,000; a run already at 60,000 is
skipped. Record the decision and the new job id in `docs/RESULTS/submissions.md`.

## Cancel

```bash
scancel <id>          # the whole array
scancel <id>_22       # one task
scancel <id>_[22-29]  # a subset (tier 3)
```

## Environment overrides

All optional; the defaults are the Picasso paths.

| variable | default |
|---|---|
| `N_ITERS` | `40000` |
| `ARRAY_SPEC` | `0-29%8` (derived from `cells.csv`) |
| `MAX_CONCURRENT` | `8` |
| `QOS` / `TIME_LIMIT` / `CPUS` / `MEM` | `medium_uma` / `11:00:00` / `8` / `32G` |
| `IHDM_REPO_DIR` | `…/fscratch/repos/generative-inverse-heat-dissipation` |
| `IHDM_ENV_PREFIX` | `…/fscratch/conda_envs/ihdm` |
| `IHDM_DATA_ROOT` | `…/fscratch/datasets/spectral_allocation_heat_diffusion_project` |
| `IHDM_RUN_ROOT` | `…/fscratch/runs/ihdm` |
| `IHDM_CELLS` | `$IHDM_REPO_DIR/slurm/array/cells.csv` |
| `IHDM_LOGS_DIR` | `~/execs/ihdm/logs` |

The worker also accepts `IHDM_TASK_ID` in place of `$SLURM_ARRAY_TASK_ID`, which is how its decode
and skip/extend branches are exercised outside SLURM.
