# Picasso setup — measured result (T3.1)

One full pass of `slurm/setup/`: clone, environment, data, import check. Every number below was
read from the cluster, not assumed. How to repeat any step: `slurm/setup/README.md`.

Date of the pass: 2026-09-22. Host `picasso3.scbi.uma.es`, user `mpascual`, account
`tic_163_uma`, QOS available `long_uma,medium_uma,short`.

## 1. Repository

| field | value |
|---|---|
| path | `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation` |
| remote | `https://github.com/MarioPasc/generative-inverse-heat-dissipation.git` (public read, no credentials) |
| branch | `ticket/T3.1-picasso-setup` |
| SHA | `355d302cc450a9082331000f53f3ebe62bbe66dd` |

The setup scripts were pushed from the workstation and pulled here; nothing was edited on the
cluster.

## 2. Environment

| field | value |
|---|---|
| prefix | `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/ihdm` |
| job id | `2402054` (`sbatch --test-only` first: accepted, `to start at 2026-09-22T20:40:02 using 4 processors on nodes picasso4`) |
| state / elapsed | `COMPLETED` / `00:05:44` |
| MaxRSS | 4,572,056 K (4.57 GB) of the 16 GB requested |
| node | `picasso4` (`--constraint=download`, the only node with outbound internet) |
| files in the prefix | 36,396 |

`sacct -j 2402054 -X -n -P -o JobID,State,Elapsed` → `2402054|COMPLETED|00:05:44`.

Log excerpt, `~/execs/ihdm/logs/create_env_2402054.out`:

```
[env] conda env create -f .../environment.yml -p .../conda_envs/ihdm
...
Successfully built ihdm
Successfully installed ihdm-0.1.0
[clean] conda clean --all
==========================================
python:      Python 3.11.16
torch:       2.14.0+cu130 (bundled CUDA 13.0)
cuda avail:  False  (False on a CPU node is expected)
numpy        2.4.6
SimpleITK    2.5.6
nibabel      5.4.2
ml_collections 1.1.0
absl-py      2.5.0
opencv-python-headless 5.0.0.93
[check] the repository imports from the new environment
ihdm at /mnt2/fscratch/users/tic_163_uma/mpascual/repos/generative-inverse-heat-dissipation/ihdm/__init__.py
Duration:    0h 5m 45s
```

`environment.yml` pins only `torch>=2.4`, so pip resolved **torch 2.14.0 with CUDA 13.0 bundled**.
Whether that wheel runs on the A100 nodes is decided by their driver, and is reported in §4.

## 3. Data

`bash slurm/setup/sync_data.sh` from the workstation, four folders with their `qc/` subfolders,
the parent's `_cache/` excluded.

| dataset | size on Picasso | files | `sha256_images` (local `meta.json`) | remote `sha256sum images.npy` |
|---|---|---|---|---|
| `ixi` | 144 MB | 11 | `93e99e89d96b984210f88b02d3c1970bf893825e972bda6b1b44662223b4b46f` | identical |
| `oasis1` | 144 MB | 11 | `4392a77136ca80d08cb9a4c910876a8b26b7c3b7d69dfe53ebe5710ba655a680` | identical |
| `lsun_church` | 145 MB | 7 | `299c076853b0dfd261d4c758fea1ee265ddd98f975b43acaa59ac81fcc61b65a` | identical |
| `lsun_bedroom` | 144 MB | 7 | `7e4c98d1544620047967f69d2df2bdcf685e363a9eccfd71607666d6fe5687d6` | identical |

All four match. The script exits non-zero and names the dataset on any mismatch.

## 4. Import check (A100)

`<pending: blocked on ticket T1.3's schedules/, which the 20-iteration run needs>`

## 5. Quota

`ssh picasso quota`, read live around the data copy:

| moment | HOME space | HOME files | FSCRATCH space | FSCRATCH files |
|---|---|---|---|---|
| before the copy | 27.09 GB / 0.28 TB soft / 0.75 TB hard | 20.9k / 35.0k / 150.0k | 0.47 TB / 1.40 TB / 1.68 TB | 216.0k / 250.0k / 400.0k |
| after the copy | 27.09 GB | 20.9k | 0.47 TB | 216.1k |

The data costs 36 files and ~0.6 GB. The conda prefix is the expensive object: 36,396 files, which
is 15 % of the FSCRATCH soft file quota on its own. Package caches were kept out of `$HOME` (whose
file quota is the tightest at 35.0k soft) and cleaned afterwards, so the prefix is all that
remains. The 30-run array's own file budget (D5″: 16 EMA checkpoints per run) still has to be
checked against this headroom before T3.3.

## 6. Job ids

| job | id | what | state |
|---|---|---|---|
| `create_env.sbatch` | `2402054` | build the `ihdm` conda environment | `COMPLETED` |
| `import_check.sbatch` | `<pending>` | A100, tests, dataset validation, 20 iterations | `<pending>` |
