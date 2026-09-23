# H-PICASSO — cluster harness

All commands from this workstation via `ssh picasso '...'` unless noted. Never run compute on the
login node (10-min CPU kill).

## 1. Before anything

```bash
ssh picasso 'quota; squeue -u $USER; sacctmgr -nP show assoc user=$USER format=Account,QOS'
```
Expected: FSCRATCH space and file quota with headroom for ≈ 30 runs × (8 EMA checkpoints × 160 MB
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

## 7. loginexa (V100) is unusable with the `ihdm` env

Measured 2026-09-23 (T3.2): loginexa carries a Tesla V100-DGXS-32GB (compute capability 7.0) and the
cluster env ships torch 2.14.0+cu130 compiled for sm_75/80/86/90/100/120 only, so every CUDA kernel
launch there fails ("no kernel image is available"). Do not use the `test-picasso-loginexa` skill
with this env; a cu126 sibling env would cost ~32k more FSCRATCH files. All GPU checks go through
A100 batch jobs.
