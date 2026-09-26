# H-METRICS — metric harness

## 1. Identities (in `pytest tests/metrics`)

| metric | identity | tolerance |
|---|---|---|
| LSD | $\mathrm{LSD}(X, X) = 0$ | exact |
| LSD | increases monotonically along a blur ladder $\sigma_B \in \{1,2,4,8\}$ on one half of a $1/f^2$ stack | strict |
| LSD octaves | RMS of the 8 octave differences within 10% of the 48-bin LSD on smooth spectra | 10% |
| $T_\tau$ | first step with LSD ≤ threshold on a hand-made dict; `None` when never | exact |
| inherited band | synthetic linear model: measured share = predicted share | 5% |
| $M$ | held-out images as "samples" → $M \approx 1$; training copies → $M \approx 0$ | 5% / < 0.05 |
| seed NN fraction | samples = their seeds → 1.0 | exact |
| diversity | identical samples → 0; Gaussian perturbation of variance $v$ → $v$ | 5% |
| low-pass | `dct_lowpass` equals the released `DCTBlur` | 1e-5 |
| bootstrap | percentile CI on a fixed sample equals a hand computation | 1e-9 |
| permutation | exact enumeration count $\binom{6}{3} = 20$ | exact |

## 2. Pilot numbers (after T4.x, on `runs/pilot_*`)

```bash
python -m ihdm.cli.evaluate_run --run $IHDM_RUN_ROOT/pilot_ixi_A0_s1 --ckpts final --n-lsd 200 --n-seeds 4 --n-per-seed 10 --skip-inception
python -m json.tool $IHDM_RUN_ROOT/pilot_ixi_A0_s1/metrics/final.json
```
Expected: every key of `05-metrics.md` §9 present and finite; LSD of an untrained-ish model is
large (> 0.3) and the reference-vs-reference LSD (run `lsd(ref[:400], ref[400:])`) is small
(< 0.05): this bracket is the metric's dynamic range and must be recorded once in
`docs/RESULTS/metrics_bracket.md` with the four datasets.

## 3. FID licence (T4.3, optional but recorded)

FID of a dataset's ref split against a blurred copy ($\sigma_B = 1, 2, 4$) increases
monotonically; FID(ref, ref-with-different-seed-subsets) is small. If not monotone at $192^2$
grayscale, FID is reported but not interpreted (EXPERIMENT_PLAN §6.1).

## 4. The evaluation array on Picasso (T5.1)

Scripts are in `slurm/eval/`; the order, the flags and the file budget are in
`slurm/eval/README.md`, and the sizing and AMP decision in `docs/RESULTS/evaluation_plan.md`.
What a green evaluation looks like, and how to check it:

| check | command | expected |
|---|---|---|
| one writer ran | `ls $IHDM_DATA_ROOT/<ds>/` | 6 cache files per dataset; `prepare_<job>.json` in `~/execs/ihdm/eval/` lists the digests of `slurm/eval/expected_seed_lists.csv` |
| no second writer | `stat -c %Y $IHDM_DATA_ROOT/<ds>/_features_inception_ref.npy` before and after the array | unchanged |
| the gate costs 1,000 chains | `pytest tests/metrics/test_run_eval.py -k gate_command` | green; on the cluster, `gate.tar` holds exactly two `samples.npy` |
| a task finished | `ls ~/execs/ihdm/eval/<run_id>_summary.json` | present; `checkpoint_steps` = every 5,000 up to `N_ITERS` (8 at 40k, 12 at 60k); `sampling.amp` = the decided mode |
| every set reused on a rerun | resubmit one index with `FORCE_EVAL=1` | every `sampling.log` entry `reused: true` |
| precisions never mix | `pytest tests/metrics/test_run_eval.py -k "precision or amp"` | green: an fp16/bf16 evaluation writes to `samples_amp-<mode>/`/`metrics_amp-<mode>/` and leaves the fp32 trees byte-identical; a set of another precision is refused, never overwritten |

Local dry runs of the workers (no SLURM needed) set `LOCALSCRATCH`, `IHDM_REPO_DIR`,
`IHDM_ENV_PREFIX`, `IHDM_DATA_ROOT`, `IHDM_RUN_ROOT`, `IHDM_EVAL_HOME`, `IHDM_INCEPTION_SRC` and
the tiny counts (`TIMING_N`, `GATE_N_LSD`, `EVAL_EXTRA_ARGS`); the exact invocations used in
T5.1 are in its log §4.

## 5. Evaluation of extended runs (T3.5, D22)

When the runs are extended, the evaluation follows the run length. `--ckpts all` selects every
multiple of `EVAL_STRIDE` = 5,000 up to the run's largest EMA checkpoint
(`run_eval.evaluated_steps`). On a 40k run that is exactly the old `EVALUATED_STEPS`. The gate pair
and the array's `--time` follow `N_ITERS` (`slurm/eval/README.md`, "Run length").

| check | command | expected |
|---|---|---|
| 40k behaviour unchanged | `pytest tests/metrics/test_run_eval.py -k "select_checkpoints or evaluated_steps"` | green: the D16 tuple is pinned verbatim, and `all` equals the pre-D22 rule on every run up to 40k (full, missing step, 42.5k, pilots) |
| 60k selects 12 steps | same | `[5000, …, 60000]`; a missing multiple of 5k is skipped with a warning |
| gate pair and names | `pytest tests/metrics/test_run_eval.py -k "gate_pair or gate_stem or gate_tar"` | green: 55k/60k at `N_ITERS=60000`, env values win; `<run_id><amp>_gate_055000_060000`; the worker unpacks the legacy `_gate.tar` first, then the pair-named tars in step order |
| time limits | `pytest tests/metrics/test_run_eval.py -k time_limit` | 40k fp16 `07:30:00`, 40k off `09:45:00`, 60k fp16 `09:15:00`, 60k off `12:00:00` |
| an extended run is scored to 60k | `~/execs/ihdm/eval/<run_id>_amp-fp16_summary.json` | `checkpoint_steps` has 12 entries; `final_step` 60000 |
| array health | `python -m ihdm.cli.check_array … --n-iters 60000` | `VERDICT: HEALTHY 30/30` (`docs/HARNESSES/picasso.md` §8) |
