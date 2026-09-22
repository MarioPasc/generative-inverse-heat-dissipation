# M5 — Evaluation runs on Picasso (sketch; completed by the orchestrator when M4 closes)

## T5.1 — Evaluation array

One SLURM array task per training run (30), each running `python -m ihdm.cli.evaluate_run --run
<workdir> --ckpts all` on an A100; `--time` from the sampler timing of T3.2 (2k samples × 200 steps
per checkpoint × 8 checkpoints + the 40 × 50 diversity samples + 5k FID samples); dependency on
the training job (`--dependency=afterok:<train job id>` per index, or submitted when `DONE`
exists). Pre-download the Inception weights on the login node into the cache path documented by
T4.3. Written with the `picasso-sbatch` skill. Budget note: if the evaluation exceeds the queue
budget, intermediate checkpoints use 1k LSD samples (D5′).

## T5.2 — Collect results

`python -m ihdm.cli.collect_results --runs $IHDM_RUN_ROOT --out results/` gathering every
`metrics/summary.json`, `manifest.json`, `metrics.jsonl` and the final grids into one folder;
rsync to `~/execs/ihdm/results` on Picasso (permanent home) and to this workstation under
`$IHDM_DATA_ROOT/../results/`; integrity check (30 runs × 8 checkpoints present, hashes).
