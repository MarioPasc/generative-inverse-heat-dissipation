# M5 — Evaluation runs on Picasso (sketch; completed by the orchestrator when M4 closes)

## T5.1 — Evaluation array

One SLURM array task per training run (30), each running `python -m ihdm.cli.evaluate_run --run
<workdir> --ckpts all` on an A100; `--time` from the sampler timing of T3.2 (2k samples × 200 steps
per checkpoint × 8 checkpoints + the 40 × 50 diversity samples + 5k FID samples); dependency on
the training job (`--dependency=afterok:<train job id>` per index, or submitted when `DONE`
exists). Pre-download the Inception weights on the login node into the cache path documented by
T4.3. Written with the `picasso-sbatch` skill. Budget note: if the evaluation exceeds the queue
budget, intermediate checkpoints use 1k LSD samples (D5′).

Facts from T4.3 that the T5.1 worker must honour: clean-fid hard-codes its weights at
`/tmp/inception-2015-12-05.pt`, so the worker stages that file on every compute node before
`evaluate_run` (copy from `$HOME` or the repo's cache dir); the reference Inception features are
cached per dataset at `<dataset>/_features_inception_ref.npy` and the torchscript Inception is not
bitwise deterministic across calls, so ONE job (a dependency step, or the first task with a lock)
writes the four caches and the 30 tasks only read them; samples and metrics go to `$LOCALSCRATCH`
and come back as one `tar` per run (FSCRATCH file quota); the per-run cost is ≈ 7.1 A100-h at
3.2 s per chain (T3.2's A100 measurement, no AMP) — verify with one `evaluate_run --ckpts final
--n-lsd 500` job before sizing the array; the plateau gate is `evaluate_run --gate 35000,40000`
on the first finished A0 runs of `ixi` and `lsun_church`.

## T5.2 — Collect results

`python -m ihdm.cli.collect_results --runs $IHDM_RUN_ROOT --out results/` gathering every
`metrics/summary.json`, `manifest.json`, `metrics.jsonl` and the final grids into one folder;
rsync to `~/execs/ihdm/results` on Picasso (permanent home) and to this workstation under
`$IHDM_DATA_ROOT/../results/`; integrity check (30 runs × 8 checkpoints present, hashes).
