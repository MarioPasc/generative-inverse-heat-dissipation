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
| `--qos=medium` | 3-day wall | the probe's `short` caps at 2 h and would TIMEOUT every task |
| `--cpus-per-task=8` | 8 | data loading; `OMP_NUM_THREADS`/`MKL_NUM_THREADS` follow `SLURM_CPUS_PER_TASK` |
| `--mem=32G` | host RAM | probe MaxRSS 18.4 GB |
| `--array=0-29%8` | 8 concurrent | QOS `medium` allows gpu=23 per user; 8 is one node's worth and leaves the cluster usable |

Nothing else is overridden: batch 16, lr 2e-4, `ckpt_every=2500` are frozen in
`configs/spectral/arms.py` (D4″, D4‴, D5″). The only run-varying flags are `--config.seed`,
`--config.training.n_iters` and `--workdir`.

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
| `QOS` / `TIME_LIMIT` / `CPUS` / `MEM` | `medium` / `11:00:00` / `8` / `32G` |
| `IHDM_REPO_DIR` | `…/fscratch/repos/generative-inverse-heat-dissipation` |
| `IHDM_ENV_PREFIX` | `…/fscratch/conda_envs/ihdm` |
| `IHDM_DATA_ROOT` | `…/fscratch/datasets/spectral_allocation_heat_diffusion_project` |
| `IHDM_RUN_ROOT` | `…/fscratch/runs/ihdm` |
| `IHDM_CELLS` | `$IHDM_REPO_DIR/slurm/array/cells.csv` |
| `IHDM_LOGS_DIR` | `~/execs/ihdm/logs` |

The worker also accepts `IHDM_TASK_ID` in place of `$SLURM_ARRAY_TASK_ID`, which is how its decode
and skip/extend branches are exercised outside SLURM.
