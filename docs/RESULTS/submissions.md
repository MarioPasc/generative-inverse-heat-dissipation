# Submissions — the 30-run training array on Picasso

Record of every submission of the training array of `00-overview.md` §1. Scripts:
`slurm/array/` (`cells.csv`, `train_array.sbatch`, `submit_array.sh`, `README.md`); ticket
T3.3; harness `docs/HARNESSES/picasso.md` §5–§6.

---

## 1. Array 1 — 30 runs at 40,000 iterations

| field | value |
|---|---|
| date | 2026-09-23 |
| job id | **not submitted** — held by `main` on 2026-09-23 pending a site-balance check of the IXI train/ref split; submit with `bash slurm/array/submit_array.sh` and fill this row in |
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
| repo SHA on the cluster | `a84b56c9add752a06b16555140066c63427b5c34` (`ticket/T3.3-full-array-submission`) |

### The `sbatch` line

```
sbatch --array=0-29%8 --job-name=ihdm-train --time=11:00:00 --qos=medium_uma \
  --ntasks=1 --cpus-per-task=8 --mem=32G --constraint=a100 --gres=gpu:1 \
  --account=tic_163_uma \
  --output=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/train_%A_%a.out \
  --error=/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs/train_%A_%a.err \
  --export=ALL,IHDM_REPO_DIR=...,IHDM_ENV_PREFIX=...,IHDM_DATA_ROOT=...,IHDM_RUN_ROOT=...,IHDM_CELLS=...,N_ITERS=40000 \
  .../slurm/array/train_array.sbatch
```

### `sbatch --test-only` (2026-09-23, before any real submission)

```
$ bash slurm/array/submit_array.sh --test-only
sbatch.orig: Job 2408178 to start at 2026-09-30T16:44:17 a using 8 processors on nodes exa02 in partition gpu_partition
[TEST-ONLY] nothing submitted
```

Accepted, routed to `gpu_partition` on `exa02` (an A100 node). The estimated start of
2026-09-30 is the scheduler's worst case under the current queue (4,325 jobs, 53 users, 3,838
running), not a reservation; tasks usually start earlier.

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

### `quota` immediately before submission (2026-09-23)

```
			 HOME				       FSCRATCH
		 Space	 Limits	 			 File	 Limits
	 used	 quota	 limit	grace	  ║	 files	 quota	 limit	grace
home	27.10GB	 0.28TB	 0.75TB	none	  ║	  21.0k	  35.0k	 150.0k	none
fscratc	 0.47TB	 1.40TB	 1.68TB	none	  ║	 248.5k	 250.0k	 400.0k	none
```

Budget for the array: ≈ 40 files per run (16 EMA checkpoints, `full_final.pt`, the rolling
checkpoint, 16 grids, `seeds.npy`, `metrics.jsonl`, `manifest.json`, `config.json`,
tensorboard) × 30 runs ≈ 1.2k files, plus 60 log files in `$HOME`. That takes FSCRATCH from
248.5k to ≈ 249.8k against a 250.0k soft limit (hard 400.0k), i.e. the last runs land on the
soft limit and start its 7-day grace. Writes are not blocked — the hard limit is 400k — but
**nothing else may be written to FSCRATCH while the array runs**. T5.1's evaluation writes to
`$LOCALSCRATCH` and returns one archive per run, as planned in D16.

Space is not a constraint: 16 × 233 MB + ≈ 700 MB ≈ 4.4 GB per run ≈ 132 GB for the array,
against 0.93 TB free.

### `squeue` snapshot after submission

_pending — filled when `main` gives GO._

### First task to run

_pending — filled when a task reaches RUNNING; the first `metrics.jsonl` lines of that run go
here._

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
ARRAY_SPEC="$(sacct -j <id> -n -X -o JobID,State \
    | awk '$2 != "COMPLETED" {split($1, a, "_"); print a[2]}' | paste -sd, -)%8" \
    bash slurm/array/submit_array.sh

# blanket resubmission: safe, finished cells are skipped before python starts
bash slurm/array/submit_array.sh
```

Cancel with `scancel <id>`, one task with `scancel <id>_22`, tier 3 with `scancel <id>_[22-29]`.

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
squeue -j <id> -o "%.14i %.12j %.8T %.10M %.6D %R"
squeue --start -j <id>
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,NodeList | head -40
sacct -j <id> -X -n -P -o State | sort | uniq -c
for r in $(cut -d, -f2 slurm/array/cells.csv | tail -n +2); do
    test -f "$IHDM_RUN_ROOT/$r/DONE" && echo "DONE $r" || echo "---- $r"
done
```

`MaxRSS` is recorded on the `.batch` step, so drop `-X` when you want memory.
