# Report figures (T6.2)

Drawn by `python -m ihdm.cli.figures` from a `results/` folder of `python -m ihdm.cli.collect_results` (`docs/RESULTS/collection.md`) and nothing else. Code: `ihdm/analysis/figures.py`, `ihdm/analysis/style.py`, `ihdm/cli/figures.py`.

| field | value |
|---|---|
| command | `python -m ihdm.cli.figures --results /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 --out docs/RESULTS/figures/PARTIAL_24_RUNS --no-pdf --png-dpi 100 --readme docs/RESULTS/figures/README.md` |
| results folder | `/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24` |
| collection complete | False (verdict `INCOMPLETE`, created 2026-09-29T07:38:12.518998+00:00, git `9fb5c84635ebe00184e14a81b6f12a91aa048901`) |
| runs in `index.csv` | 30 |

> **Partial collection.** Captions give the n/N runs each figure uses and name the absent runs; rerun the command on the complete collection before quoting any figure.

Style: one colour and one marker per arm in every figure (A0 blue circle, A3 orange square, A1 aqua triangle, A2 yellow diamond, A2′ pink inverted triangle; the `dataviz` reference palette, colour-blind checked); widths 3.3 in (single column) and 6.75 in (full width); PDFs are vector with no creation date.

## Figure 1. LSD versus iteration

Files: [`fig1_lsd_vs_iteration.png`](PARTIAL_24_RUNS/fig1_lsd_vs_iteration.png). Width 6.75 in. Coverage: **24/30 runs**.

**Caption draft.** Log-spectral distance (LSD, 43 populated log-spaced bins, 0.5-96 cycles per image) between 500 training-seeded samples and the 800-image reference split, at every checkpoint from 5k to 60k iterations (12 checkpoints; common random numbers across checkpoints and arms, D17). One panel per dataset (top: development pair, IXI and LSUN Churches; bottom: transfer pair, OASIS-1 and LSUN Bedrooms). Thick lines with markers: seed mean per arm; thin lines: individual seeds; no smoothing, linear y axis. Dashed line: the mean over the A0 seeds of their LSD at the last checkpoint on this 500-seed set. The pre-registered $T_\tau$ compares each seed's curve with its own A0 seed's value, so the line shows the level of the reference, not the per-seed thresholds, and no $T_\tau$ is marked on the curves. The curves are not monotone in iteration: the LSD at 5k is often at or below its value at the last checkpoint and rises before it falls, so a first crossing of the reference can occur at the first checkpoint. Shaded: the extension from 40k to 60k by resume at step 40,001 (D22); the pre-registered gate triggered it on IXI. The y axes are not shared: LSD is never compared across datasets. n = 24/30 runs. Absent (not evaluated or not collected): `lsun_bedroom_A0_s1`, `lsun_bedroom_A0_s2`, `oasis1_A3_s1`, `oasis1_A3_s2`, `lsun_bedroom_A3_s1`, `lsun_bedroom_A3_s2`.

## Figure 2. Per-octave LSD profile at the final checkpoint

Files: [`fig2_lsd_octaves.png`](PARTIAL_24_RUNS/fig2_lsd_octaves.png). Width 6.75 in. Coverage: **24/30 runs**.

**Caption draft.** Per-octave spectral error of the final sample set (2,000 training-seeded samples at 60k iterations) against the reference split: $\log_{10}\bar P_S(b) - \log_{10}\bar P_R(b)$ on the eight octave bands of the project, in cycles per image (05-metrics §2; sign kept: positive means the samples carry too much variance in that band; the DC mode and modes above 96 cycles per image are excluded). Lines with markers: seed mean per arm; small markers: individual seeds; arms are offset horizontally for legibility. One panel per dataset; y axes not shared. n = 24/30 runs. Absent (not evaluated or not collected): `lsun_bedroom_A0_s1`, `lsun_bedroom_A0_s2`, `oasis1_A3_s1`, `oasis1_A3_s2`, `lsun_bedroom_A3_s1`, `lsun_bedroom_A3_s2`.

## Figure 3. Diversity and memorisation

Files: [`fig3_diversity_memorisation.png`](PARTIAL_24_RUNS/fig3_diversity_memorisation.png). Width 6.75 in. Coverage: **24/30 runs**.

**Caption draft.** Within-seed diversity and memorisation at the final checkpoint (60k), per dataset (columns) and arm (x position); each marker is one training run (seed), the horizontal bar is the seed mean. Rows 1-2: $D_{\mathrm{pix}}$ and $D_{\mathrm{lp}}$, the non-DC per-pixel variance across 50 samples drawn from the same prior state, averaged over the 40 held-out seed subjects (05-metrics §3; $D_{\mathrm{lp}}$ after the $\sigma_B = 16$ px heat-kernel low-pass). Rows 3-4: $M$ and $M_{\mathrm{lp}}$, the median nearest-training-image distance of 2,000 training-seeded samples divided by that of held-out real images (05-metrics §4); the dotted line $M = 1$ is a new real image, $M < 1$ indicates copying. Axes are per panel. n = 24/30 runs. Absent (not evaluated or not collected): `lsun_bedroom_A0_s1`, `lsun_bedroom_A0_s2`, `oasis1_A3_s1`, `oasis1_A3_s2`, `lsun_bedroom_A3_s1`, `lsun_bedroom_A3_s2`.

## Figure 4. PCA around the seed

Files: [`fig4_pca_seed.png`](PARTIAL_24_RUNS/fig4_pca_seed.png). Width 3.3 in. Coverage: **4/4 runs**.

**Caption draft.** Samples around their seed in the plane of the first two principal components of the training split (3,200 images, DC removed; 05-metrics §6; the basis is the same for every run of a dataset). Each row is zoomed to its seeds and samples; grey points are the training images that fall in that window. Hollow star: a held-out seed image (seed subjects 1 and 2 of 40, slice 5); circles and triangles: its 50 samples drawn from the same prior state (seed blurred to the terminal level, plus prior noise); arrow: seed to sample centroid, drawn when the shift exceeds 4% of the window. Rows: IXI and LSUN Churches; columns: A0 ($\sigma_{B,\max}$ 96, log spacing) and A3 ($\sigma_{B,\max}$ 24, IXI-matched spacing), training seed 1 of each. Axes are shared within a row. n = 4/4 runs.

## Figure 5. Inherited band

Files: [`fig5_inherited_band.png`](PARTIAL_24_RUNS/fig5_inherited_band.png). Width 6.75 in. Coverage: **14/20 runs**.

**Caption draft.** The inherited band (05-metrics §5): per-mode variance of 50 samples about their prior state $d_K \hat x_s$, averaged over the 40 held-out seeds and divided by the population variance $P_{\mathrm{ref}}$, as a radial profile on the 43 populated log-spaced bins (solid; thin: single seeds), against the linear-Gaussian prediction $1 - d_K^2(n) = 1 - e^{-2\lambda_n t_K}$ with $t_K = \sigma_{B,\max}^2/2$ (dashed). The two terminal blurs: A0 ($\sigma_{B,\max}$ = 96 px, prediction $\approx 1$ above one cycle per image) and A3 ($\sigma_{B,\max}$ = 24 px, the prior keeps the low band). A measured curve on its dashed line means the model regenerates exactly the variance the prior removed; below it, the samples inherit more of the seed. Log axes, one y range for all panels. The bins below 3 cycles per image hold 1-6 DCT modes each on the $192^2$ grid, so the measured ratio there is noisy. n = 14/20 runs. Absent (not evaluated or not collected): `lsun_bedroom_A0_s1`, `lsun_bedroom_A0_s2`, `oasis1_A3_s1`, `oasis1_A3_s2`, `lsun_bedroom_A3_s1`, `lsun_bedroom_A3_s2`.

## Figure 6. Sample grids

Files: [`fig6_grids.png`](PARTIAL_24_RUNS/fig6_grids.png). Width 6.75 in. Coverage: **8/8 runs**.

**Caption draft.** Training sample grids at the last iteration (60k, EMA weights) of training seed 1, for A0 (left) and A3 (right) on each dataset. In each grid the top row holds 8 training images used as seeds and the bottom row the sample each seed produces from its prior state (the seed blurred to the terminal level), drawn with the EMA weights. The seeds are fixed by the run seed, so the A0 and A3 grids of a dataset share their top row. These are the trainer's monitoring grids (`ihdm/train/grids.py`), not the evaluation sets. `lsun_church_A3_s1` (Churches, A3, right column) was flagged at 40k for a saturated vertical stripe at the right edge of every sample (D21 d; seeds 2 and 3 do not show it). The stripe is not visible in the 60k grid shown here. n = 8/8 runs.

- `lsun_church_A3_s1` at 60k, checked on 2026-09-29 (T6.2): in each of the 8 samples the mean of the 3 right-most pixel columns differs from that of the 15 columns inside them by at most 14 grey levels (`lsun_church_A3_s2`: 17; `lsun_church_A0_s1`: 6), and no right-edge column averages above 206 of 255. Recheck if the grid file changes.

## Figure 7. Training sanity panel

Files: [`fig7_training_sanity.png`](PARTIAL_24_RUNS/fig7_training_sanity.png). Width 6.75 in. Coverage: **30/30 runs**.

**Caption draft.** Training sanity from each run's canonical history (`metrics.canonical.jsonl`: the segment abandoned by run 11's resume at 30,001 is already removed, D21). Top: training loss, running mean over 20 logged steps (1000 iterations), log scale; middle: throughput in iterations per second as logged (A100, batch 16). Both y ranges span the records from step 2,500 on; the warm-up and the first logged step lie above or below them. Bottom: the per-octave training loss averaged over the last 5,000 iterations, on the trainer's σ_B bands in pixels (labelled by their lower edge; bands an arm's blur range never visits are not logged and not drawn), log scale. Thin lines: single runs; thick lines in the bottom row: seed mean. The loss is the regression loss of each arm's own blur schedule, so its level is not comparable across arms; the panel checks convergence and stability only. Shaded: the 40k → 60k extension, resumed at step 40,001 (D22). Skipped fp16 steps kept in the canonical histories: `lsun_church_A3_s3` 6 skipped steps. n = 30/30 runs.
