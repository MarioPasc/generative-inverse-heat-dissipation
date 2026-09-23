# Picasso setup (T3.1)

Everything needed to put the experiment on Picasso and prove it runs there: the repository
clone, the `ihdm` conda environment, the four datasets, and one A100 job that exercises the
whole stack. The measured outcome of one full pass is in `docs/RESULTS/picasso_setup.md`.

Nothing here is edited on the cluster. Every change is committed on the workstation, pushed, and
pulled on Picasso.

## Conventions

| what | where |
|---|---|
| repository | `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation` |
| conda env | `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/ihdm` |
| datasets (`IHDM_DATA_ROOT`) | `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/spectral_allocation_heat_diffusion_project` |
| runs (`IHDM_RUN_ROOT`) | `/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm` |
| job logs | `/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/logs` (`~/execs/ihdm/logs`) |
| SSH | alias `picasso` (`picasso3.scbi.uma.es`), account `tic_163_uma` |

Each script reads these from environment variables (`IHDM_REPO_DIR`, `IHDM_ENV_PREFIX`,
`IHDM_DATA_ROOT`, `IHDM_RUN_ROOT`) and falls back to the paths above, so a second copy of the
experiment needs no edits to the scripts.

## Step 0 — read the live cluster state first

```bash
ssh picasso 'quota; squeue; sacctmgr -nP show assoc user=$USER format=Account,QOS'
```

FSCRATCH has both a space and a **file-count** quota, and a conda environment with torch is tens
of thousands of files. Check the headroom before building, not after.

## Step 1 — clone the repository

```bash
ssh picasso 'mkdir -p ~/execs/ihdm/logs \
    /mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm \
    /mnt/home/users/tic_163_uma/mpascual/fscratch/repos'
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos \
    && git clone https://github.com/MarioPasc/generative-inverse-heat-dissipation.git'
ssh picasso 'git -C /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation rev-parse HEAD'
```

Public read over HTTPS needs no credentials. To move the clone to another commit or branch:

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation \
    && git fetch origin && git checkout <branch> && git pull --ff-only'
```

## Step 2 — create the conda environment

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation \
    && sbatch --test-only slurm/setup/create_env.sbatch'
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation \
    && sbatch slurm/setup/create_env.sbatch'
```

`sbatch --parsable` does not return a bare job id on Picasso (a Lua wrapper prepends a warning
banner), so take the number from the output:

```bash
ssh picasso '... sbatch slurm/setup/create_env.sbatch' | grep -oE '[0-9]+' | tail -1
ssh picasso 'squeue -j <jobid>; sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS'
```

Why a batch job and not the login node: the login node kills any process above 10 minutes of CPU
and this build takes tens of minutes. The job requests `--constraint=download`, the only node
feature that carries outbound internet (`picasso4`); `conda env create` and `pip install` cannot
be assumed to reach conda-forge or PyPI from a generic compute node.

The job writes `conda`'s package cache and pip's cache under FSCRATCH (never `$HOME`, whose file
quota is the tightest) and runs `conda clean --all` afterwards. `IHDM_ENV_RECREATE=1 sbatch ...`
deletes the prefix and rebuilds it from scratch.

Log: `~/execs/ihdm/logs/create_env_<jobid>.out`. It ends with the python and torch versions, the
package versions, and `import ihdm` resolved from the checkout.

## Step 3 — copy the datasets

Run on the **workstation**, with the external disk mounted:

```bash
bash slurm/setup/sync_data.sh --dry-run    # see what would be copied
bash slurm/setup/sync_data.sh              # copy the four folders, then verify
bash slurm/setup/sync_data.sh --verify     # verify an existing copy, copy nothing
```

It `rsync`s `ixi`, `oasis1`, `lsun_church`, `lsun_bedroom` (with their `qc/` subfolders,
excluding the parent's `_cache/`), prints the cluster quota before and after, and compares the
remote `sha256sum images.npy` against the `sha256_images` recorded in each local `meta.json`. Any
mismatch exits non-zero and names the dataset.

## Step 4 — the import check

Needs `schedules/` (ticket T1.3) present in the clone, because the 20-iteration run builds an arm
from the frozen schedule arrays.

```bash
ssh picasso 'cd <repo> && git pull --ff-only'
ssh picasso 'cd <repo> && sbatch --test-only slurm/setup/import_check.sbatch'
ssh picasso 'cd <repo> && sbatch slurm/setup/import_check.sbatch'
ssh picasso 'sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS'
ssh picasso 'cat ~/execs/ihdm/logs/import_check_<jobid>.out'
```

One A100 (`--constraint=a100` plus an untyped `--gres=gpu:1`), QOS `short`, 30 minutes, 8 cores,
32 GB. Five steps, each reported separately so one failure still leaves the others' evidence:
`nvidia-smi`; torch version, bundled CUDA, device name and a GPU matmul; `pytest -q -m "not
integration"`; `ihdm.cli.validate_dataset ixi lsun_church`; and a 20-iteration `ixi,A0` run at
batch 4 into `$IHDM_RUN_ROOT/import_check/ixi_A0_s1`. The last lines are the run's file listing,
whether `DONE` exists, and the tail of `metrics.jsonl`. The job exits non-zero, and names the
failing steps, if any step failed.

Batch 4 rather than the pre-registered 16: the point is that the code runs, not that it fills the
card (T2.1 measured about 29 GB at batch 16 on an A100 40 GB).

## Re-running one piece

| goal | command |
|---|---|
| rebuild the environment | `IHDM_ENV_RECREATE=1 sbatch slurm/setup/create_env.sbatch` |
| re-verify the data only | `bash slurm/setup/sync_data.sh --verify` |
| re-run the import check | `sbatch slurm/setup/import_check.sbatch` (it overwrites the same workdir) |
| validate a dataset by hand | `python -m ihdm.cli.validate_dataset ixi` with `IHDM_DATA_ROOT` exported |
