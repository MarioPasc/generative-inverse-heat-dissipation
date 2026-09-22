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

## 6. Pilot numbers (RTX 3060, `docs/RESULTS/pilot_3060.md`)

| quantity | how to read it |
|---|---|
| peak GB at batch 8/16/32 | must fit 12 GB at 16; extrapolate to A100 |
| it/s | A100 estimate = 3–4× |
| loss over 1k iterations | decreasing; no NaN; eval loss tracks train loss |
| grids at 1k | coarse structure visible (head silhouette / building mass) |
| per-octave losses | finite for every octave; the coarse octaves are not identically zero |
