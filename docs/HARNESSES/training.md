# H-TRAIN — trainer harness

## 1. Smoke (CPU, < 2 min)

```bash
pytest tests/train -q
```
Expected: green. Covers arm configs, the level sampler (with/without the fix), the prior noise
flag, a 6-iteration run with the full artefact set, resume to 9 iterations, checkpoint format.

## 2. Artefact listing of any run

```bash
R=$IHDM_RUN_ROOT/<run_id>; ls $R; ls $R/checkpoints; tail -3 $R/metrics.jsonl; python -m json.tool $R/manifest.json | head -40
```
Expected: `manifest.json config.json metrics.jsonl tensorboard/ checkpoints/ checkpoints-meta/
grids/ [DONE]`; `checkpoints/ema_iter_XXXXXX.pt` every `ckpt_every`; the last `metrics.jsonl`
line has `kind`, `step`, `loss`, `lr`, `it_per_s`, `gpu_mem_peak_gb`, `loss_per_octave`; the
manifest has `git_sha`, `schedule.sha256`, `data.images_sha256`, `n_params`, `batch_size`.

## 3. Cadence check

```bash
python - <<'EOF'
import json,sys; R=sys.argv[1] if len(sys.argv)>1 else "."
kinds={}
for l in open(f"{R}/metrics.jsonl"):
    j=json.loads(l); kinds.setdefault(j["kind"],[]).append(j["step"])
print({k:(len(v),v[:3],v[-1]) for k,v in kinds.items()})
EOF
```
Expected: `train` every `log_every`, `eval` every `eval_every`, `ckpt` every `ckpt_every` plus the
final step, at most one `resume` per restart, no `abort`.

## 4. Resume

Kill a run with SIGTERM mid-way, restart the same command: the first new `metrics.jsonl` line is
`{"kind": "resume", ...}` at the step of `checkpoints-meta/checkpoint.pth`; later EMA checkpoints
continue the numbering; `--config.training.n_iters` larger than the saved step extends the run.

## 5. Seeding

Two runs with the same `--config.seed` and identical config produce identical `grids/seeds.npy`
and identical loss at step 1 (CPU: exactly; GPU: to 1e-4 with AMP).

## 5a. D19 events and scalars (T3.4)

`metrics.jsonl` also carries `skip` (non-finite loss, step skipped by the `GradScaler`),
`abort` (with `reason`, `n_skipped`, `consecutive`, `abort_state`: the path of the saved
state, which never overwrites the rolling checkpoint, D21) and a final
`{"kind": "done", "n_skipped": k}`; every train line carries the pre-clip `grad_norm` and
`amp_scale`. A resume with another recipe exits 4 and writes nothing. The whole contract is
checked line by line by `ihdm.train.validate_run` (CLI: `python slurm/loginexa/check_run.py <run>
--data-root $IHDM_DATA_ROOT --lr 1e-4`), which is also the §2–§3 check of any array run.

## 6. Pilot numbers (RTX 3060, `docs/RESULTS/pilot_3060.md`)

| quantity | how to read it |
|---|---|
| peak GB at batch 8/16/32 | must fit 12 GB at 16; extrapolate to A100 |
| it/s | A100 estimate = 3–4× |
| loss over 1k iterations | decreasing; no NaN; eval loss tracks train loss |
| grids at 1k | coarse structure visible (head silhouette / building mass) |
| per-octave losses | finite for every octave; the coarse octaves are not identically zero |

## 7. The loginexa harness (V100, no queue; T3.4)

Run before any submission of the training array, and after any change to `train.py`,
`scripts/losses.py`, `ihdm/train/`, `configs/spectral/arms.py` or `slurm/array/`. Picasso's
loginexa node has 4× V100-DGXS-32GB, no SLURM and a 30-minute limit per session; it is shared,
so a GPU is used only when `nvidia-smi` shows < 1000 MiB on it, at most two at once. Results of
the first run: `docs/RESULTS/loginexa_harness.md`.

```bash
# 0. once: the V100 overlay (torch 2.14.0+cu126 over the cluster env, in $HOME, ~5.2k inodes)
ssh picasso 'ssh loginexa "nohup timeout 25m bash ~/execs/ihdm/wt/T3.4/slurm/loginexa/build_overlay.sh \
    > ~/execs/ihdm/logs/loginexa/overlay_$(date +%Y%m%d_%H%M%S).log 2>&1 &"'
# 1. every time: ship the committed tree (never git on Picasso) and record its SHA
git ls-files -z | rsync -a --from0 --files-from=- ./ picasso:/mnt/home/users/tic_163_uma/mpascual/execs/ihdm/wt/T3.4/
ssh picasso "echo $(git rev-parse --short HEAD) > ~/execs/ihdm/wt/T3.4/.t34_sha"
# 2. one item per session, from the workstation; each prints LOG=<path>
bash slurm/loginexa/launch.sh h1 2                       # env, 25 production steps at batch 16, pytest tests/train
bash slurm/loginexa/launch.sh h2 2 0 7                   # the real worker on cells 0..7 (then 8..15, 16..23, 24..29)
bash slurm/loginexa/launch.sh h3 2 ixi,A0                # 300 its, shortened cadence, full artefact check
bash slurm/loginexa/launch.sh h3 3 lsun_church,A3
bash slurm/loginexa/launch.sh h4 2 ixi,A0                # SIGTERM + resume, 300 -> 400, recipe-mismatch refusals
bash slurm/loginexa/launch.sh h5 3                       # skip / abort / carried count / disabled scaler, CUDA scaler
bash slurm/loginexa/launch.sh h6 2 /tmp/ihdm_T3.4/h3/ixi_A0_s1   # load_ema_model + evaluate_run (+ --gate)
# several items back to back on one GPU (waits for the GPU to be free before each):
bash slurm/loginexa/queue.sh 2 h3:ixi,A0 h2:0:7 h2:16:23 h4:ixi,A0 h6:/tmp/ihdm_T3.4/h3/ixi_A0_s1
# a guard-saved run (exit 3) or any run directory: diagnosis and fp32 activation headroom
bash slurm/loginexa/launch.sh diag 3 <run_dir> [--weights ema]
bash slurm/loginexa/launch.sh headroom 3 ixi_A0_s1     # T3.4's four weight sets of one S cell
# a multi-session run (the S check): chains 24-minute sessions until DONE
bash slurm/loginexa/chain_s.sh 2 ixi,A0 1
# 3. poll
ssh picasso 'tail -n 30 <LOG>'
# 4. at the end: remove the local scratch of loginexa
ssh picasso 'ssh loginexa "rm -rf /tmp/ihdm_T3.4 /tmp/ihdm_pycache_h2_*"'
```

| item | pass criterion |
|---|---|
| H1 | `H1 overall PASS`: capability 7.0 and `sm_70` in the overlay's arch list; 25 production train steps (fp16 autocast, batch 16, real U-Net) with finite losses and a finite, varying pre-clip norm once the scaler has settled; `pytest tests/train` green on loginexa |
| H2 | for every one of the 30 cells: the worker's `CELL` line decodes the right row, `END TASK … status=ok exit=0`, and `H2[i] check_run PASS` (manifest schedule hash = file = `schedules.json`, data hash = `meta.json`, lr 1e-4, finite loss at batch 16, every artefact) |
| H3 | `H3 check_run PASS` on both cells: H-TRAIN §2–§3, the line-by-line schema of every kind, no `NaN` token, a varying pre-clip `grad_norm`, `amp_scale` on every train line, lr 1e-4 after warm-up, `full_final.pt`, `DONE`; look at `grids/iter_000300.png` |
| H4 | the resumed run logs exactly one `resume` at the rolling checkpoint's step and passes `check_run`; the extension to 400 passes; each recipe mismatch exits 4 with `run_dir=identical` |
| H5 | skips at the injected steps with `amp_scale` halved and `done n_skipped=2`; ten consecutive skips exit 3 with the abort state in `abort_step_<step>.pth` and the rolling checkpoint untouched (D21; the loginexa run of 2026-09-25 predates D21 and shows the old `resume_saved: true`); the resume carries the skips before the rolling checkpoint; the disabled scaler exits 3 at once with `abort_state: null` and a finite rolling checkpoint |
| H6 | `load_ema_model` loads both checkpoints; `evaluate_run` and `evaluate_run --gate` exit 0 and every result file is strict JSON with no `null`/`NaN` leaf |
| H7 | the V100's `it_per_s` and `gpu_mem_peak_gb` at batch 16, read from the S and H3 `metrics.jsonl` |

The injected non-finite losses of H5 come from `slurm/loginexa/train_inject.py`, a test-only
driver that monkeypatches `scripts.losses` in its own process; production (`train.py` with a
locked `config_flags` config) cannot reach it.
