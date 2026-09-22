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
