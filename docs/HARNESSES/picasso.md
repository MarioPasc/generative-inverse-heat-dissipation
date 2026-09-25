# H-PICASSO — cluster harness

All commands from this workstation via `ssh picasso '...'` unless noted. Never run compute on the
login node (10-min CPU kill).

## 1. Before anything

```bash
ssh picasso 'quota; squeue -u $USER; sacctmgr -nP show assoc user=$USER format=Account,QOS'
```
Expected: FSCRATCH space and file quota with headroom for ≈ 30 runs × (16 EMA checkpoints × 233 MB (D5″)
+ full final ≈ 500 MB + samples ≈ 1 GB) ≈ 90 GB and ≈ 3k files; the QOS list includes `short` and
`medium`.

## 2. Environment and data (T3.1)

```bash
ssh picasso 'ls /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation && git -C /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation rev-parse HEAD'
ssh picasso 'ls /mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/ihdm/bin/python'
ssh picasso 'for d in ixi oasis1 lsun_church lsun_bedroom; do sha256sum /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/spectral_allocation_heat_diffusion_project/$d/images.npy; done'
```
Expected: the SHA equals local `origin/main`; the python exists; the four hashes equal the
`sha256_images` of each local `meta.json`.

## 3. Import-check job (T3.1)

`~/execs/ihdm/logs/import_check_<jobid>.out` contains: the GPU name (`A100`), `torch.cuda.is_available() True`,
the `pytest` summary line with 0 failures, `OK ixi`, `OK lsun_church`, and a 20-iteration run's
last `metrics.jsonl` line. `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` shows `COMPLETED`.

## 4. Probe (T3.2)

`sacct -j <jobid>` `COMPLETED`; the six steps' outputs listed in `docs/RESULTS/picasso_probe.md`;
peak memory at batch 64 recorded; the resume line present; sampler timing recorded.

## 5. Array (T3.3)

```bash
ssh picasso 'sbatch --test-only <launcher-generated worker>'    # before the real submission
ssh picasso 'squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"'
ssh picasso 'sacct -j <arrayjobid> --format=JobID,State,Elapsed,MaxRSS,NodeList | head -40'
ssh picasso 'for r in $(cut -d, -f2 <cells.csv> | tail -n +2); do test -f /mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm/$r/DONE && echo "DONE $r" || echo "---- $r"; done'
```
Expected over time: tasks move PENDING → RUNNING → COMPLETED; every finished run has `DONE`,
8 EMA checkpoints, no `abort` line; a `TIMEOUT` task is resubmitted by index and resumes.

## 6. Plateau gate (D10)

When the first A0 runs on `ixi` and `lsun_church` finish, run `evaluate_run --ckpts 35000,40000
--n-lsd 500 --skip-inception` on them (T5.1 or by hand) and compare the LSD at 35,000 and 40,000:
if the relative change exceeds 5% on either dataset, resubmit the whole array with `N_ITERS=60000`
(the worker resumes every run from its rolling checkpoint). Record the decision in
`docs/RESULTS/submissions.md`. File-quota note (2026-09-23): FSCRATCH sits at ≈ 247.5k of 250k
soft files after the caches were cleared; the array writes ≈ 40 files per run; evaluation
artefacts go to `$LOCALSCRATCH` and come back as one archive per run.

**`.pyc` files on FSCRATCH (T3.4, 2026-09-25).** The `ihdm` env holds no compiled `.pyc` for
sympy, mpmath, triton, pip, PIL, yaml, absl and parts of numpy, so the first process that imports
them writes ≈ 0.9k `.pyc` files and ≈ 0.1k `__pycache__` directories into the env, i.e. onto
FSCRATCH (measured: +1.0k files in one loginexa build, reverted). Every job and every loginexa
process therefore sets `PYTHONPYCACHEPREFIX` to node-local storage (`slurm/array/train_array.sbatch`,
`slurm/loginexa/common.sh`); a clean-up of the env's caches can be undone by a single run without it.

## 7. loginexa (V100) through the cu126 overlay

Measured 2026-09-23 (T3.2): loginexa carries 4× Tesla V100-DGXS-32GB (compute capability 7.0)
and the cluster env ships torch 2.14.0+cu130 compiled for sm_75/80/86/90/100/120 only (CUDA 13
dropped Volta), so every CUDA kernel launch there fails ("no kernel image is available").

Since T3.4 (2026-09-25) loginexa **is usable** through an overlay: a `--system-site-packages`
venv on top of the cluster env, in `$HOME` (`~/execs/ihdm/overlay/ihdm-v100`, ≈ 5.2k inodes,
5.2 GB; `$HOME` went 18.6k → 24.1k of its 35k soft file quota), holding only torch 2.14.0+cu126
and torchvision 0.29.0+cu126 (the same versions, built for sm_50–sm_90) and their CUDA 12 runtime
wheels; the C++ headers are pruned. Every other package (numpy, ml_collections, the editable
`ihdm`, …) is the cluster env's, so the harness tests the same code and libraries as the A100 jobs
except the CUDA build of torch. Build: `slurm/loginexa/build_overlay.sh` (run on loginexa, which
has internet; 2 min). Use: `slurm/loginexa/` (`common.sh` rules, `harness.sh` items,
`launch.sh` from the workstation), documented in `docs/HARNESSES/training.md` §7. Rules: every
process under `timeout 25m`; a GPU only when `nvidia-smi` shows < 1000 MiB on it, at most two;
run directories on loginexa's local `/tmp` or in `$HOME`, never on FSCRATCH.

V100 figures at batch 16 (production recipe, fp16 autocast): 0.88 it/s (A100 1.71), 28.5 GiB
peak allocated of 31.7 GiB — it fits, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
set by `common.sh` against fragmentation (an allocator OOM-and-retry was logged once without it).
