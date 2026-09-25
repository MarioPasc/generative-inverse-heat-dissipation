# Submissions — the 30-run training array on Picasso

Record of every submission of the training array of `00-overview.md` §1. Scripts:
`slurm/array/` (`cells.csv`, `train_array.sbatch`, `submit_array.sh`, `README.md`); ticket
T3.3; harness `docs/HARNESSES/picasso.md` §5–§6.

**Status (2026-09-25 13:40):** array 1 (**2408239**) FAILED 30/30 (§7). Array 2 (**2432693**,
recipe v2: lr 1e-4, D19 skip policy) was submitted at 13:39 and its first 8 tasks started at
once. The plateau-gate job **2432703** (fp16, D20) waits on tasks 0 and 3 (§8).

---

## 1. Array 1 — 30 runs at 40,000 iterations

| field | value |
|---|---|
| date | 2026-09-23 |
| job id | **2408239** (submitted 2026-09-23 ≈12:26 local, after `main` gave GO on the site-stratified IXI split) |
| array spec | `0-29%8` |
| `N_ITERS` | 40000 |
| batch size | 16 (frozen in `configs/spectral/arms.py`, D4″) |
| lr | 2e-4 (frozen, D4‴) |
| checkpoint cadence | every 2,500 iterations → 16 EMA checkpoints per run (D5″) |
| `--time` | `11:00:00` |
| `--qos` | `medium_uma` |
| resources | `--constraint=a100 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G --account=tic_163_uma` |
| logs | `~/execs/ihdm/logs/train_%A_%a.out` / `.err` |
| run root | `/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm` |
| repo SHA on the cluster | **`5bc28a640ad5369c553014dcb8976d06bd3985c8`** on `main` (verified `git status` clean at submission). `origin/main` is one commit further at `f16f925` (`docs: W7 verdicts…`), but `git diff 5bc28a6 f16f925` is **empty** — that commit changes no file, so the cluster tree is byte-identical to `origin/main` |
| data on the cluster | `ixi/images.npy` sha256 `b666e407e9afcc03fe2c51eb55ed99b0b72a34c80515e78874d64e155864bf4b` (T1.5 site-stratified re-sync) |
| schedules on the cluster | `schedules/ixi_W8.npy` sha256 `8bfe56bd759253d5b873ea6cd2080351b2051709719edd53337d3c930a7b916b`; `schedules/ixi_W2.npy` sha256 `9c80d11c614197750f62b2c8dcc760e6c1df416b0a85248a25d784d15e94dfc3` (both refit on the stratified split, T1.5) |

### The `sbatch` line

Exactly as the launcher built and ran it (`$U` = `/mnt/home/users/tic_163_uma/mpascual`,
`$REPO` = `$U/fscratch/repos/generative-inverse-heat-dissipation`):

```
sbatch --array=0-29%8 --job-name=ihdm-train --time=11:00:00 --qos=medium_uma \
  --ntasks=1 --cpus-per-task=8 --mem=32G --constraint=a100 --gres=gpu:1 \
  --account=tic_163_uma \
  --output=$U/execs/ihdm/logs/train_%A_%a.out \
  --error=$U/execs/ihdm/logs/train_%A_%a.err \
  --export=ALL,IHDM_REPO_DIR=$REPO,IHDM_ENV_PREFIX=$U/fscratch/conda_envs/ihdm,IHDM_DATA_ROOT=$U/fscratch/datasets/spectral_allocation_heat_diffusion_project,IHDM_RUN_ROOT=$U/fscratch/runs/ihdm,IHDM_CELLS=$REPO/slurm/array/cells.csv,N_ITERS=40000 \
  $REPO/slurm/array/train_array.sbatch
```

A100 selection is `--constraint=a100` with an **untyped** `--gres=gpu:1`: `exa[01-04]` advertise
`gpu:8` with the feature `a100`, so `--gres=gpu:A100:1` matches no node and a bare
`--constraint=dgx` would also match the B200 nodes.

The command that produced it, on the login node:

```bash
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation
git checkout main && git pull        # 5bc28a6
bash slurm/array/submit_array.sh     # prints quota, runs --test-only, submits
```

### `sbatch --test-only`

Run twice: once before the hold (2026-09-23 ≈11:50, on `ticket/T3.3-full-array-submission`) and
again by the launcher immediately before the real submission (≈12:26, on `main` at `5bc28a6`).

```
$ bash slurm/array/submit_array.sh --test-only          # first, before the hold
sbatch.orig: Job 2408178 to start at 2026-09-30T16:44:17 a using 8 processors on nodes exa02 in partition gpu_partition
[TEST-ONLY] nothing submitted

---- sbatch --test-only ----                            # second, inside the real submission
sbatch.orig: Job 2408238 to start at 2026-10-01T11:26:43 a using 8 processors on nodes exa01 in partition gpu_partition
```

Both probes were accepted and routed to `gpu_partition` on an A100 node (`exa02`, then
`exa01`). The ids 2408178 and 2408238 belong to the probes; the array actually submitted is
**2408239**. The dates the probes print are the scheduler's **worst-case** placement under the
current queue (4,325 jobs, 53 users, 3,838 running), not reservations; tasks usually start
earlier.

The same check on `--array=6` alone was also accepted (job 2408179), so the single-index
resubmission path in §4 is valid as written.

### QOS: `medium_uma`, not `medium`

The ticket and `docs/HARNESSES/picasso.md` §1 say `medium`. The account is not entitled to it:

```
$ sacctmgr -nP show assoc user=$USER format=Account,QOS
tic_163_uma|long_uma,medium_uma,short
mpascual|long_uma,medium_uma,short

$ sacctmgr -nP show qos format=Name,MaxWall,MaxTRESPU
medium|3-00:00:00|cpu=9000,gres/gpu=23,mem=50000000M
medium_uma|3-00:00:00|cpu=9000,gres/gpu=23,mem=50000000M
```

`medium` exists cluster-wide with identical limits but is outside our association, and a
submission with it is refused:

```
$ QOS=medium bash slurm/array/submit_array.sh --test-only
allocation failure: Invalid qos specification
FATAL: --test-only rejected the request (exit 1); nothing submitted
```

`medium_uma` gives the same 3-day wall and the same `gres/gpu=23` per-user cap, so `%8`
concurrency is unchanged.

### `quota` immediately before submission (2026-09-23, printed by the launcher)

```
			 HOME				       FSCRATCH
		 Space	 Limits	 			 File	 Limits
	 used	 quota	 limit	grace	  ║	 files	 quota	 limit	grace
home	27.10GB	 0.28TB	 0.75TB	none	  ║	  21.0k	  35.0k	 150.0k	none
fscratc	 0.47TB	 1.40TB	 1.68TB	none	  ║	 248.6k	 250.0k	 400.0k	none
```

Budget for the array: ≈ 40 files per run (16 EMA checkpoints, `full_final.pt`, the rolling
checkpoint, 16 grids, `seeds.npy`, `metrics.jsonl`, `manifest.json`, `config.json`,
tensorboard) × 30 runs ≈ 1.2k files, plus 60 log files in `$HOME`. That takes FSCRATCH from
248.6k to ≈ 249.8k against a 250.0k soft limit (hard 400.0k), i.e. the last runs land on the
soft limit and start its 7-day grace. Writes are not blocked — the hard limit is 400k — but
**nothing else may be written to FSCRATCH while the array runs**. T5.1's evaluation writes to
`$LOCALSCRATCH` and returns one archive per run, as planned in D16.

Space is not a constraint: 16 × 233 MB + ≈ 700 MB ≈ 4.4 GB per run ≈ 132 GB for the array,
against 0.93 TB free.

### `squeue` snapshot after submission

Immediately after `sbatch` (the launcher's own `squeue -j 2408239`):

```
JOBID      NAME             USER       STATUS      TIMELEFT     CPUS/NODES  REASON/NODES
2408239    ihdm-train       mpascual   [PD]        11:00:00     8/1         None/
2408239    ihdm-train       mpascual   [PD:29]     11:00:00     232         None/
```

A few minutes later the reason had resolved to the normal one:

```
2408239    ihdm-train       mpascual   [PD]        11:00:00     8/1         Priority/
2408239    ihdm-train       mpascual   [PD:29]     11:00:00     232         Priority/
```

`Reason=Priority` is queue pressure, not an unsatisfiable request — the diagnosis to worry
about would be `ReqNodeNotAvail` or `BadConstraints`, which never appeared. `sacct` reports the
array as one pending record, `2408239_[0-29%8]|PENDING`.

`scontrol show job 2408239` confirms the request that was accepted:

```
JobId=2408239 ArrayJobId=2408239 ArrayTaskId=0-29%8 ArrayTaskThrottle=8 JobName=ihdm-train
   Priority=27450 Nice=0 Account=tic_163_uma QOS=medium_uma
   JobState=PENDING Reason=Priority Dependency=(null)
   StartTime=Unknown EndTime=Unknown Deadline=N/A
   Partition=gpu_partition AllocNode:Sid=picasso3:15983
   ReqTRES=cpu=8,mem=32G,node=1,billing=8,gres/gpu=1
   AllocTRES=(null)
```

i.e. `QOS=medium_uma`, `ArrayTaskThrottle=8`, `gpu_partition`, one GPU, 8 cores, 32 GB — the
request as specified.

### Estimated start

`squeue --start` is not useful here: **Picasso's Lua `squeue` wrapper ignores `--start`** and
returns its ordinary listing, and `scontrol show job` reports `StartTime=Unknown` while
`Reason=Priority`. The only figure available is the one `sbatch --test-only` printed for an
identical request moments before submission:

```
Job 2408238 to start at 2026-10-01T11:26:43 using 8 processors on nodes exa01 in partition gpu_partition
```

That is a worst-case bound under a queue holding 4,325 jobs from 53 users, not a reservation.
Treat it as "hours to days", and re-read it with
`scontrol show job 2408239 | grep StartTime` once the scheduler commits to a slot.

### First task to run

Not observed within this session: all 30 tasks were still `PENDING` with `Reason=Priority` when
the record was written, and the queue may take hours. Nothing is blocking — the first task will
start on its own. To capture the evidence when it does:

```bash
squeue -j 2408239 -o "%.14i %.12j %.8T %.10M %.6D %R"     # look for RUNNING
tail -n 40 ~/execs/ihdm/logs/train_2408239_0.out          # header, CELL line, nvidia-smi
head -n 5 $IHDM_RUN_ROOT/ixi_A0_s1/metrics.jsonl          # first metrics lines
```

The worker prints `CELL index=... run_id=... dataset=... arm=... seed=... tier=...` before it
starts, which is the line to check first: it is the decoded tuple, and a wrong decode is the
one failure here that would otherwise produce a complete, plausible, wrong result set.

---

## 2. The cells

`slurm/array/cells.csv`, generated from `configs.spectral.arms.EXPERIMENT_CELLS` by
`slurm/array/make_cells.py` and pinned by `tests/slurm/test_cells.py`. Sorted by tier, then
arm, then dataset, then seed, so the lowest indices — the ones the scheduler starts first —
are the cells the plateau gate D10 is read from.

| index | run_id | dataset | arm | seed | tier |
|---|---|---|---|---|---|
| 0 | `ixi_A0_s1` | ixi | A0 | 1 | 1 |
| 1 | `ixi_A0_s2` | ixi | A0 | 2 | 1 |
| 2 | `ixi_A0_s3` | ixi | A0 | 3 | 1 |
| 3 | `lsun_church_A0_s1` | lsun_church | A0 | 1 | 1 |
| 4 | `lsun_church_A0_s2` | lsun_church | A0 | 2 | 1 |
| 5 | `lsun_church_A0_s3` | lsun_church | A0 | 3 | 1 |
| 6 | `ixi_A3_s1` | ixi | A3 | 1 | 1 |
| 7 | `ixi_A3_s2` | ixi | A3 | 2 | 1 |
| 8 | `ixi_A3_s3` | ixi | A3 | 3 | 1 |
| 9 | `lsun_church_A3_s1` | lsun_church | A3 | 1 | 1 |
| 10 | `lsun_church_A3_s2` | lsun_church | A3 | 2 | 1 |
| 11 | `lsun_church_A3_s3` | lsun_church | A3 | 3 | 1 |
| 12 | `ixi_A1_s1` | ixi | A1 | 1 | 2 |
| 13 | `ixi_A1_s2` | ixi | A1 | 2 | 2 |
| 14 | `lsun_church_A1_s1` | lsun_church | A1 | 1 | 2 |
| 15 | `lsun_church_A1_s2` | lsun_church | A1 | 2 | 2 |
| 16 | `ixi_A2_s1` | ixi | A2 | 1 | 2 |
| 17 | `ixi_A2_s2` | ixi | A2 | 2 | 2 |
| 18 | `lsun_church_A2_s1` | lsun_church | A2 | 1 | 2 |
| 19 | `lsun_church_A2_s2` | lsun_church | A2 | 2 | 2 |
| 20 | `lsun_church_A2p_s1` | lsun_church | A2p | 1 | 2 |
| 21 | `lsun_church_A2p_s2` | lsun_church | A2p | 2 | 2 |
| 22 | `oasis1_A0_s1` | oasis1 | A0 | 1 | 3 |
| 23 | `oasis1_A0_s2` | oasis1 | A0 | 2 | 3 |
| 24 | `lsun_bedroom_A0_s1` | lsun_bedroom | A0 | 1 | 3 |
| 25 | `lsun_bedroom_A0_s2` | lsun_bedroom | A0 | 2 | 3 |
| 26 | `oasis1_A3_s1` | oasis1 | A3 | 1 | 3 |
| 27 | `oasis1_A3_s2` | oasis1 | A3 | 2 | 3 |
| 28 | `lsun_bedroom_A3_s1` | lsun_bedroom | A3 | 1 | 3 |
| 29 | `lsun_bedroom_A3_s2` | lsun_bedroom | A3 | 2 | 3 |

Cost at the measured 6.84 h per run (D16): 205 A100-h for the array; 25.6 h of wall clock at
`%8` if every slot stays busy.

---

## 3. The plateau gate (D10)

When the first A0 runs finish — indices 0 (`ixi_A0_s1`) and 3 (`lsun_church_A0_s1`) — compare
their LSD at the 35,000 and 40,000 EMA checkpoints:

```bash
evaluate_run --ckpts 35000,40000 --n-lsd 500 --skip-inception   # T5.1, or by hand
```

**If the relative change exceeds 5% on either dataset, extend every run to 60,000 iterations:**

```bash
N_ITERS=60000 bash slurm/array/submit_array.sh
```

Each task resumes from `checkpoints-meta/checkpoint.pth` and trains on; no cell is ever
retrained from scratch with a different count. A run already at 60,000 is skipped in seconds
before python starts. Record the decision, the two LSD values and the new job id in §1 of a new
entry below.

**Gate outcome:** _pending._

---

## 4. Resubmission

`train.py` writes its rolling checkpoint every 500 iterations and its loop is
`range(initial_step, n_iters + 1)`, so resubmitting an index continues that run.

```bash
# one index (e.g. a TIMEOUT or a node failure on ixi_A3_s1):
ARRAY_SPEC='6' bash slurm/array/submit_array.sh

# several:
ARRAY_SPEC='6,22,27' bash slurm/array/submit_array.sh

# every task that did not COMPLETE:
ARRAY_SPEC="$(sacct -j 2408239 -n -X -o JobID,State \
    | awk '$2 != "COMPLETED" {split($1, a, "_"); print a[2]}' | paste -sd, -)%8" \
    bash slurm/array/submit_array.sh

# blanket resubmission: safe, finished cells are skipped before python starts
bash slurm/array/submit_array.sh
```

Cancel with `scancel 2408239`, one task with `scancel 2408239_22`, tier 3 with `scancel 2408239_[22-29]`.

---

## 5. Drop order if the queue is slow

Pre-registered (D16 references it; the ticket states it), applied in this order and recorded
here when used:

1. **Tier-3 seeds** — indices 23, 25, 27, 29 (the second seed of every transfer-pair cell).
   The transfer claim survives on one seed per cell.
2. **A2** — indices 16–19 (and, if still needed, the `A2p` control at 20–21). A2 is the
   secondary matched-schedule arm; A0 versus A3 carries the claim.
3. **Iterations** — the plateau gate's answer is then read at whatever checkpoint the finished
   runs reach, and the reported `n_iters` drops for every arm equally, never per arm.

Never dropped: tier 1 (indices 0–11), because it is the claim.

**Drops applied:** none.

---

## 6. Monitoring

```bash
squeue                                                    # Picasso's wrapper rejects `-u`
squeue -j 2408239 -o "%.14i %.12j %.8T %.10M %.6D %R"
squeue --start -j 2408239
sacct -j 2408239 --format=JobID,State,Elapsed,MaxRSS,NodeList | head -40
sacct -j 2408239 -X -n -P -o State | sort | uniq -c
for r in $(cut -d, -f2 slurm/array/cells.csv | tail -n +2); do
    test -f "$IHDM_RUN_ROOT/$r/DONE" && echo "DONE $r" || echo "---- $r"
done
```

`MaxRSS` is recorded on the `.batch` step, so drop `-X` when you want memory.

---

## 7. Outcome of array 1 (2408239): FAILED 30/30 (recorded 2026-09-25 by `main`)

Tasks started 2026-09-24 09:24 on `exa04`/`exa02` (≈ 21 h after submission). Throughput was
as the probe predicted: 50 steps per 29.3 s = 1.71 it/s, peak 28.5 GB.

| cause | indices | evidence |
|---|---|---|
| non-finite training loss, guard abort (exit 3) | 0, 2, 4, 5, 6, 8, 9 | abort at steps 1382, 1606, 1888, 1434, 1672, 1861, 1367 (`ixi_A0_s1`, `ixi_A0_s3`, `lsun_church_A0_s2`, `lsun_church_A0_s3`, `ixi_A3_s1`, `ixi_A3_s3`, `lsun_church_A3_s1`); the lr had been at 2e-4 since step 1,000; last logged train loss 0.35–0.49 |
| FSCRATCH outage while running (exit 1) | 1, 3, 7, 10 | `OSError: [Errno 116] Stale file handle` on `metrics.jsonl` / tensorboard at 2026-09-24 10:54; reached steps 8750, 7900, 5400, 750 with no non-finite loss |
| FSCRATCH outage at start (exit 1, 2 s) | 11–29 | `couldn't chdir … Stale file handle`; `[FATAL] no interpreter at …/conda_envs/ihdm/bin/python` |

The logged `grad_norm` is identically 1.0 in every run because it was read after
`clip_grad_norm_`; it carried no information (fixed in D19). The loss curves showed spikes
during the warm-up ramp from lr ≈ 1.2e-4 on (e.g. `ixi_A0_s1` 2.83 at step 800 and 1.37 at 900 against
≈ 0.4–0.6 around them; `lsun_church_A0_s2` 1.83 at 600 and 1.91 at 900).

**Decision:** D19 (`00-overview.md`) — recipe v2 with lr 1e-4, skip policy, pre-registered
stability check S, loginexa harness (T3.4), rerun of all 30 cells from scratch.

**Archive and wipe.** Everything under `fscratch/runs/ihdm/` (11 run directories, 82 files,
12 GB) and all of `~/execs/ihdm/logs/` were copied to
`/media/mpascual/Sandisk2TB/research/spectral_allocation_heat_diffusion_project/training/array_2408239_failed/`
on the workstation, then the run directories and the 60 `train_2408239_*` logs were deleted on
Picasso at Mario's request. The eight setup/probe logs cited in `picasso_setup.md` and
`picasso_probe.md` were kept. Fixtures for T3.4's diagnosis and T5.1's timing job (16 files,
2.3 GB, in `$HOME`, not FSCRATCH) are at `~/execs/ihdm/fixtures/array_2408239/`:
`ixi_A0_s1` and `lsun_church_A3_s1` (the resume checkpoints the guard saved at steps 1383 and
1368, i.e. the weights that produced the non-finite loss) and `ixi_A0_s2` (EMA 5,000 and 7,500).
They are removed when W10 closes.

---

## 8. Array 2 (2432693), recipe v2, 30 runs at 40,000 iterations (2026-09-25, `main`)

| field | value |
|---|---|
| job id | **2432693**, submitted 2026-09-25 13:39 (after W10 merged; Mario approved the merge, push and submission) |
| array spec | `0-29%8`; tasks 0–7 started 13:39:51 on `exa02`/`exa03`/`exa04` (the queue was nearly empty) |
| recipe | v2 (D19): **lr 1e-4**; batch 16, warm-up 1,000, clip 1.0, fp16 AMP, EMA 0.999, K = 200; skip policy 10 consecutive / 100 total; recipe check on resume (exit 4) |
| `N_ITERS` / `--time` / QOS | 40000 / `11:00:00` / `medium_uma`; resources as array 1 (§1) |
| repo SHA on the cluster | **`3e9dbf2b7ac5fb9e69117e0134ce4fe87f8fb8e7`** (`main`, clean; the W10 integration merge) |
| worker changes since array 1 | exports `PYTHONPYCACHEPREFIX` to node-local `/tmp` (the env on FSCRATCH has no `.pyc`); names exit 3 (guard abort) and 4 (recipe mismatch) in its status line |
| verification before submission | T3.4 loginexa harness H1–H7 PASS, all 30 cells through the real worker (`docs/RESULTS/loginexa_harness.md`); check S passed at lr 1e-4 on the two cells that failed first under v1 (`docs/RESULTS/nan_diagnosis.md` §4); integration suite 699 passed |
| `--test-only` | accepted: `Job 2432692 to start at 2026-09-27T21:39:49 … on nodes exa02` (worst-case bound; the tasks started immediately) |
| quota at submission | FSCRATCH 248.7k of 250.0k files (the ≈ 1.2k array files may cross the soft limit into its 7-day grace; Mario accepted this); `$HOME` 24.7k of 35.0k |

**Plateau gate (D10, D17, D20):** job **2432703**, `slurm/eval/submit_eval.sh gate` with
`TRAIN_ARRAY=2432693 AMP=fp16`, dependency `afterok:2432693_0:2432693_3`, `--time 03:00:00`. It
runs `evaluate_run --ckpts 35000 --gate 35000,40000 --n-lsd 500 --skip-inception --amp fp16` on
`ixi_A0_s1` and `lsun_church_A0_s1` and writes `~/execs/ihdm/eval/gate/<run_id>_amp-fp16_gate.json`.
Extend every run to 60k (`N_ITERS=60000 bash slurm/array/submit_array.sh`) only if the CI of
LSD(35k) − LSD(40k) lies above zero; the evaluation array (T5.1, `AMP=fp16 TIME_LIMIT=07:30:00`,
`docs/RESULTS/evaluation_plan.md` §6) is submitted after that decision.

**Outcome:** _running._

- **Early check (≈ 14:00, `main`).** Tasks 0–7 all passed step 2,000, beyond array 1's failure window (1,367–1,888).
- **Guard events.** Zero `skip` and zero `abort` events.
- **Cell decoding.** Each `CELL index=… run_id=…` line matches `cells.csv`.
- **Throughput and memory.** 1.69–1.70 it/s, 28.54 GB peak.
- **Gradient norm and loss scale.** The pre-clip `grad_norm` at step 2,000 was 135–829, and `amp_scale` was 32–128.
