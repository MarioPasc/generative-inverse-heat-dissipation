# loginexa harness of the training and logging path (T3.4, D19)

The production training path (`slurm/array/train_array.sbatch` → `train.py` → `scripts/losses.py`
→ `ihdm/train/*`, recipe v2 of `configs/spectral/arms.py`) exercised on Picasso's loginexa node
before the v2 array is resubmitted. Procedure: `docs/HARNESSES/training.md` §7; scripts:
`slurm/loginexa/`. Every item below quotes its log verbatim (trimmed to the verdict lines; the
full logs are in `~/execs/ihdm/logs/loginexa/` on Picasso).

## Environment

| item | value |
|---|---|
| node | loginexa, 4× Tesla V100-DGXS-32GB (compute capability 7.0, 32768 MiB), driver 580.159.04, 20 cores, 251 GB RAM, kernel 6.8.0-88; shared: GPUs 0 and 1 held 17.4 GB and 13.5 GB of another user's work all day and were never used; GPUs 2 and 3 only |
| python | `~/execs/ihdm/overlay/ihdm-v100/bin/python` (3.11.16): `--system-site-packages` venv over `fscratch/conda_envs/ihdm`, holding torch 2.14.0+cu126 and torchvision 0.29.0+cu126 (arch list `sm_50 … sm_90`), cuDNN 9.10.2, CUDA 12.6 runtime wheels; every other package from the cluster env (numpy 2.4.6, …) |
| overlay build | `slurm/loginexa/build_overlay.sh`, 2 min on loginexa; 13,091 files installed, 3,231 after pruning the C++ headers (torch/include 9,529); 5.2 GB; 5,246 inodes with directories |
| `$HOME` file count (`quota`) | **18.6k before** (09:05, before any write of this ticket) → **24.1k after** the overlay (+ the rsynced tree, 288 entries) — soft limit 35k |
| FSCRATCH file count | 248.8k before; 249.8k right after the first overlay build (892 `.pyc` + 118 `__pycache__` written into the base env by the build's imports), **248.8k after the revert**; nothing written there since (`PYTHONPYCACHEPREFIX` on loginexa's `/tmp`) |
| environment of every process | `slurm/loginexa/common.sh`: `IHDM_DATA_ROOT=<fscratch>/datasets/spectral_allocation_heat_diffusion_project` (read only), `PYTHONPATH=~/execs/ihdm/wt/T3.4`, `PYTHONPYCACHEPREFIX=/tmp/ihdm_T3.4/pycache`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `OMP_NUM_THREADS=4`; `timeout 25m` on every launch, a GPU only if `nvidia-smi` shows < 1000 MiB |
| run directories | H1–H7 on loginexa's local `/tmp/ihdm_T3.4/…` (removed at the end); the S runs in `~/execs/ihdm/loginexa_runs/` |

## Results

(filled below as the items ran)
