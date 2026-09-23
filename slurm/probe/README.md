# `slurm/probe/` — the T3.2 A100 probe

One job that produces every number T3.3 needs to size the 30-run array. Results:
`docs/RESULTS/picasso_probe.md`; log: `docs/AGENT-LOGS/M3-picasso/T3.2-picasso-probe.md`.

## Files

| file | what |
|---|---|
| `submit_probe.sh` | launcher, login node: makes the log/run dirs, runs `sbatch --test-only`, submits, captures the job id past the Lua wrapper's banner |
| `probe.sbatch` | worker, A100: the six steps, each wrapped so a failure (the expected batch-24 OOM) does not abort the rest |

## Run it

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/generative-inverse-heat-dissipation
git fetch && git checkout ticket/T3.2-picasso-probe
bash slurm/probe/submit_probe.sh --test-only    # always first
bash slurm/probe/submit_probe.sh
squeue -j <jobid>                               # `squeue -u` is rejected by Picasso's wrapper
sacct -j <jobid> -n -P -o JobID,State,Elapsed,MaxRSS,NodeList
```

Log: `~/execs/ihdm/logs/probe_<jobid>.out`. Runs: `$IHDM_RUN_ROOT` =
`.../fscratch/runs/ihdm/probe/{lsun_church_A0_s1,ixi_A0_s1,mem_b24}`.

## The six steps

1. `lsun_church,A0` seed 1, batch 16 (the config default), 600 iterations — three epochs of the
   3200-image train split — with `ckpt/grid/eval_every=200`, `log_every=20`.
2. `ixi,A0` seed 1, identical.
3. Memory probe: `ixi,A0` at `--config.training.batch_size=24 --config.eval.batch_size=24`,
   40 iterations. Both flags are needed: ml_collections applies CLI overrides *after* the
   factory, so the factory's "eval follows training" rule never sees them (T2.1 log §6).
   T2.3's fit `peak(B) = 1.7046 B + 1.2997` GB predicts 42.2 GB, i.e. an OOM on a 40 GB A100;
   the step is wrapped so the OOM is recorded, not fatal.
4. Resume/extension: step 2's command again with `n_iters=800`. `train.py` writes the rolling
   checkpoint in its finaliser, so `initial_step` is 600 and the first metrics line of the
   second invocation is `{"kind": "resume", "step": 600}`.
5. Offline sampler on `ema_iter_000800.pt` at sampling batch 32 (4 held-out seeds × 8), then 64
   and 128 over the same 128 training seeds. `request.json` carries `timing.s_per_chain`.
   Steps 5b/5c are only started if `128 × (the s/chain 5a measured)` still fits before the
   deadline; otherwise they are skipped and reported as skipped — a measurement is never
   shortened to make it fit.
6. Evidence: `ls -R` of each run directory, the `metrics.jsonl` kind/cadence histogram with the
   median it/s and img/s over steps ≥ 100, the sampler timings, the device-level
   `nvidia-smi memory.used` peak (sampled every 5 s in the background), and `sacct`.

## Resources

`--constraint=a100 --gres=gpu:1` (untyped GRES: exa[01-04] advertise `gpu:8` with the feature
`a100`; `--gres=gpu:A100:1` matches no node and `--constraint=dgx` would also match the B200
nodes), `--cpus-per-task=8 --mem=48G --qos=short --time=01:50:00` (`short` caps at 2 h).
`--output`/`--error` are explicit so nothing lands in the checkout.
