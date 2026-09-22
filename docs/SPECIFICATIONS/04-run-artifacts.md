# 04 — Schedules, configs, run artefacts and hook points (frozen contracts)

Producers: T1.3 (schedules), T2.1 (trainer), T2.2 (sampler). Consumers: T2.3, T3.x, T4.x, T5.x.
Line numbers refer to the fork at base commit `3f735ca`.

## 1. Schedule files (`schedules/`)

| file | content |
|---|---|
| `schedules/<name>.npy` | `float64`, shape `(K+1,)` = `(201,)`; `s[0] == 0.0`; `s[1] == 0.5`; strictly increasing; `s[K] ∈ {96.0, 24.0}` |
| `schedules/schedules.json` | per name: `{"kind": "log"|"matched", "sigma_min": 0.5, "sigma_max": 96|24, "K": 200, "fitted_on": "<dataset_id>/train" or null, "n_images": 3200, "images_sha256": "...", "levels_per_octave": {"0.5-1": n, ..., "64-96": n}, "spread": <max/min per-level removed variance>, "sha256": "<hex of the .npy bytes>", "created": "...", "git_sha": "..."}` |

Names: `log_W2`, `log_W8`, `ixi_W2`, `ixi_W8`, `lsun_church_W2`. Also produced but not used by an
arm: `oasis1_W8`, `lsun_bedroom_W2` (for the profile tables only).

Semantics: `s[k]` is $\sigma_{B,k}$ in pixels; the heat time is $t_k = \sigma_{B,k}^2/2$
(`model_code/utils.py: DCTBlur.forward`). The log schedule is exactly
`np.exp(np.linspace(np.log(0.5), np.log(smax), K))` prepended with 0. A matched schedule places the
$K$ levels so that each step removes the same between-image variance of the fitting split,
computed with the DCT mode power on the $192^2$ grid (`ihdm/spectral/schedules.py`, ported from
`projects/GenAI/analysis/variance_matched_schedule.py` and `delta_star_from_psd.py` with
attribution); the endpoints are fixed at $0.5$ and $\sigma_{B,\max}$ and the matching is run for
each endpoint separately (D12).

`validate_schedule(path) -> list[str]` checks shape, dtype, monotonicity, endpoints and the hash
against `schedules.json`.

## 2. Config factory (`configs/spectral/arms.py`)

The released trainer takes `--config <file>[:<string>]` (ml_collections `config_flags`). Our
single factory:

```python
def get_config(spec: str) -> ml_collections.ConfigDict:
    """spec = "<dataset_id>,<arm>[,<key>=<value>...]", e.g. "ixi,A3" or "lsun_church,A2p"."""
```

It returns the released structure (`training`, `eval`, `data`, `model`, `optim`, `seed`, `device`)
with these values (frozen unless the ticket says "set by pilot/probe"):

| group | key | value |
|---|---|---|
| `data` | `dataset` | `<dataset_id>` (one of `ixi`, `oasis1`, `lsun_church`, `lsun_bedroom`) |
| | `root` | `str(ihdm.paths.data_root())` |
| | `split_train`, `split_eval` | `"train"`, `"ref"` |
| | `image_size`, `num_channels` | `192`, `1` |
| | `random_flip`, `centered`, `uniform_dequantization` | `False`, `False`, `False` |
| | `num_workers` | `4` |
| `model` | `K`, `sigma`, `blur_sigma_min` | `200`, `0.01`, `0.5` |
| | `blur_sigma_max` | `96.0` (A0, A2, A2′) or `24.0` (A1, A3) |
| | `blur_schedule_name` | `log_W2` (A0), `log_W8` (A1), `ixi_W2` (A2), `ixi_W8` (A3), `lsun_church_W2` (A2′) |
| | `blur_schedule_file` | absolute path of `schedules/<name>.npy`, resolved relative to the repository root |
| | `blur_schedule` | the loaded array (so the released code sees exactly what it expects) |
| | `blur_schedule_sha256` | hex |
| | `train_level_max_inclusive` | `True` (D3) |
| | `model_channels`, `channel_mult`, `num_res_blocks`, `attention_levels` | `128`, `(1, 2, 2, 2)`, `4`, `(2, 3)` |
| | `dropout`, `num_heads`, `num_head_channels`, `num_heads_upsample` | `0.1`, `1`, `-1`, `-1` |
| | `conv_resample`, `use_fp16`, `use_scale_shift_norm`, `resblock_updown`, `use_new_attention_order`, `skip_rescale`, `conditional`, `normalization`, `nonlinearity` | released CIFAR values |
| | `ema_rate` | `0.999` |
| `sampling` | `prior_noise` | `True` (D3): the prior draw adds $\mathcal N(0, \delta^2 I)$ with $\delta = 1.25\sigma$ |
| | `delta_factor` | `1.25` |
| `training` | `batch_size` | **set by probe** (64 or 32; D4′) |
| | `n_iters` | `20000` |
| | `ckpt_every` | `2500` |
| | `resume_every` | `500` |
| | `log_every` | `50` |
| | `eval_every` | `500` (25 batches of the `ref` split, EMA weights) |
| | `grid_every` | `2500` (8 samples from 8 fixed training seeds; PNG only) |
| `eval` | `batch_size` | same as `training.batch_size` |
| `optim` | `optimizer`, `lr`, `beta1`, `eps`, `weight_decay` | `Adam`, `2e-4`, `0.9`, `1e-8`, `0.0` |
| | `warmup`, `grad_clip`, `automatic_mp` | `1000`, `1.0`, `True` |
| `seed` | | `1` by default; overridden with `--config.seed=2` |
| `device` | | `cuda:0` if available |

The released keys `training.snapshot_freq`, `snapshot_freq_for_preemption`, `sampling_freq`,
`eval_freq`, `log_freq` are set to the values above under their new names *and* kept under the old
names for compatibility with `sample.py`/`evaluate.py`.

Arms A2′ are only valid for photograph datasets; the factory raises `ValueError` for
`"ixi,A2p"`. Bedrooms accepts only A0 and A3.

`run_id = f"{dataset_id}_{arm}_s{seed}"` (arm ∈ {A0, A1, A2, A3, A2p}); `workdir = <run_root>/<run_id>`.

## 3. Run directory (`<run_root>/<run_id>/`)

```
manifest.json                      # written before the first step; see §3.1
config.json                        # the resolved ConfigDict as JSON (arrays as lists)
metrics.jsonl                      # one JSON object per log event; see §3.2
tensorboard/                       # the same scalars, for eyeballing
checkpoints/ema_iter_002500.pt     # every ckpt_every iterations, and at n_iters; see §3.3
checkpoints/full_final.pt          # model + optimizer + ema + step at the end
checkpoints-meta/checkpoint.pth    # rolling resume checkpoint (released format), every resume_every
grids/iter_002500.png              # 8 samples (EMA) from 8 fixed training seeds, with the seeds row
grids/seeds.npy                    # the 8 seed images used for the grids (uint8), fixed by config.seed
DONE                               # empty file written after the final checkpoint
```

### 3.1 `manifest.json`

```json
{"run_id": "ixi_A3_s1", "dataset_id": "ixi", "arm": "A3", "seed": 1,
 "git_sha": "...", "git_dirty": false, "hostname": "...", "gpu": "NVIDIA A100-SXM4-40GB",
 "python": "3.11.x", "torch": "2.x", "cuda": "12.x", "started": "ISO", "slurm_job_id": "... or null",
 "data": {"root": "...", "meta": {...meta.json of the dataset...}, "images_sha256": "..."},
 "schedule": {"name": "ixi_W8", "file": "...", "sha256": "...", "values": [0.0, 0.5, ...]},
 "config_sha256": "...", "n_params": 12345678, "batch_size": 64, "n_iters": 20000}
```

### 3.2 `metrics.jsonl`

Every `log_every` steps: `{"step": s, "kind": "train", "loss": float, "lr": float, "it_per_s": float,
"img_per_s": float, "gpu_mem_peak_gb": float, "grad_norm": float, "wall_s": float,
"loss_per_octave": {"0.5-1": mean loss of the samples whose level fell in that octave, ...}}`.
Every `eval_every`: `{"step": s, "kind": "eval", "loss": float (EMA weights, 25 ref batches)}`.
Every checkpoint: `{"step": s, "kind": "ckpt", "path": "..."}`.
On resume: `{"step": s, "kind": "resume", "from": "..."}`. Also `nan` guard: if the loss is not
finite, log `{"kind": "abort", ...}` and exit non-zero.

### 3.3 EMA checkpoint (`ema_iter_XXXXXX.pt`)

`torch.save({"step": int, "ema_state_dict": <model state dict with EMA weights, keys without the
DataParallel "module." prefix>, "schedule": {"name", "sha256", "values"}, "run_id": str,
"config_sha256": str})`. Size ≈ 4 bytes × n_params. Loaded by `ihdm.sampling.load_ema_model`.

## 4. Sampler contract (`ihdm/sampling/`)

```python
@dataclass(frozen=True)
class SampleRequest:
    n_per_seed: int              # 1 for fidelity samples, 50 for the diversity endpoint
    delta: float | None = None   # None -> config delta_factor * sigma
    start_level: int | None = None  # None -> K; smaller values for the start-level sweep
    prior_noise: bool | None = None # None -> config.sampling.prior_noise
    batch_size: int = 64
    rng_seed: int = 0

def load_ema_model(ckpt_path: Path, config) -> torch.nn.Module            # eval mode, EMA weights
def sample_from_seeds(model, config, seeds_u8: np.ndarray, request: SampleRequest,
                      device) -> np.ndarray   # seeds (N,192,192) uint8 -> samples (N, n_per_seed, 192, 192) float32 in [0,1] (clipped)
def load_seed_images(dataset_root: Path, source: Literal["train", "seed"], n: int, rng_seed: int) -> tuple[np.ndarray, np.ndarray]
    # returns (images uint8 (n,192,192), idx (n,)); "train": n images drawn from the train split with rng_seed; "seed": one image per seed subject (MRI: the slice with index 5, the Fig. 1 plane), all 40 if n >= 40
```

CLI: `python -m ihdm.cli.sample_ckpt --run <workdir> --ckpt ema_iter_020000.pt --source train|seed
--n-seeds 2000 --n-per-seed 1 --out <dir>` writes `samples.npy` (uint8 after rounding, shape
`(N, n_per_seed, 192, 192)`), `seeds.npy`, `seed_idx.npy`, `request.json`.

The reverse loop is a parametrised copy of `scripts/sampling.get_sampling_fn_inverse_heat` that
starts at `start_level` and does not keep intermediates; the released function is left untouched.

## 5. Hook points in the released code (the only permitted edits)

| hook | file:where | change |
|---|---|---|
| dataset backend | `scripts/datasets.py: get_dataset` | new `elif config.data.dataset in NPY_DATASETS:` branch returning `DataLoader`s over `ihdm.data.dataset.NpyImageDataset`; the module-level `from mpi4py import MPI` becomes a lazy import inside `load_data` |
| level sampler | `scripts/losses.py: get_label_sampling_function` (~line 57) | `high = K + 1 if getattr(config.model, "train_level_max_inclusive", False) else K` |
| prior noise | `scripts/sampling.py: get_initial_sample` (~line 101) | after blurring to level K, add `torch.randn_like(x) * delta` when `config.sampling.prior_noise` is true (the `delta` argument is already passed in) |
| seeding | `train.py` top of `train()` | `torch.manual_seed(config.seed)`, `np.random.seed(config.seed)`, `random.seed(config.seed)`; DataLoader `generator` seeded |
| checkpoint cadence and naming | `train.py` (~lines 137–143) | `if step % config.training.ckpt_every == 0 or step == n_iters: ihdm.train.checkpoints.save_ema(...)`; full state at the end; the rolling resume checkpoint every `resume_every` |
| manifest and metrics | `train.py` after model creation and inside the loop | `ihdm.train.manifest.write(...)`, `ihdm.train.logging.MetricsLogger` fed with `loss`, `losses_batch`, `fwd_steps_batch`, timing, memory, lr |
| sample grids | `train.py` sampling block | replaced by `ihdm.train.grids.save_grid(...)` using 8 fixed training seeds; the released gif/video writers are not called |
| schedule from file | `configs/spectral/arms.py` | the factory loads the array; no change in `model_code/` |

Everything else in `model_code/`, `scripts/`, `sample.py`, `evaluate.py` is read-only.

## 6. Smoke config (`configs/spectral/smoke.py`)

`get_config()` for the synthetic dataset written by the test fixtures: `image_size=32`, `K=8`,
`model_channels=16`, `channel_mult=(1, 2)`, `num_res_blocks=1`, `attention_levels=()`, batch 4,
`n_iters=6`, `ckpt_every=3`, `resume_every=2`, `log_every=1`, `eval_every=3`, `grid_every=3`, CPU.
Used by `tests/train/test_smoke_train.py` (runs `train.train(config, tmp_workdir)` twice to test
resume) and by the Picasso import-check job.
