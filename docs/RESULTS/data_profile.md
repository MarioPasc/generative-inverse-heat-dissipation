# Spectral profile of the four training splits, and the frozen schedules

Produced by `python -m ihdm.cli.profile_data` at `e9a9fb070923` on 2026-09-22. Ticket T1.3; contracts `04-run-artifacts.md` §1 and `05-metrics.md` §1.

Every curve and every table below is measured on the **training** split of each dataset (3200 registered images of $192^2$, values in $[0, 1]$); the `ref` split (800 images) is profiled beside it as a consistency check. The per-mode variance is mean-centred across images, in the orthonormal DCT-II basis, with the DC mode excluded; a mode of radial index $n$ carries $n/2$ cycles per image and modes above 96 cycles per image are excluded from the octave shares.

The **archived** columns are the pre-registration numbers of `worklog/sessions/2026-09-21_ihdm-knob-proposals/native192_profile_output_unprocessed.md` (480 head-centred crops of the *unprocessed* volumes per dataset, no rigid registration, no intensity harmonisation, no padding). They are not a target: registration changes the coarse variance by construction. They are printed so the size and the direction of that change are visible.

## 1. Spectral exponent

$\alpha$ is fitted by least squares of $\log \bar P$ on $\log n$ over the integer DCT radii $b \in [0.10 W, 0.70 W] = [20, 134]$, i.e. **10.0 – 67.0 cycles per image** — the window of `analysis/control_profile.py: fit_alpha`, held fixed across datasets because the brain spectrum is curved. The third column repeats the fit over the window the ticket's prose names (1 – 48 cycles per image); see the ticket log, decision D-T1.3-1.

| dataset | $\alpha$ (train) | $\alpha$ (ref) | $\alpha$ (train, 1–48 c/img window) | archived, unregistered |
|---|---|---|---|---|
| IXI T1 | **3.22** | 3.21 | 2.17 | 3.47 |
| OASIS-1 T1 | **3.11** | 3.08 | 2.12 | 3.16 |
| LSUN Churches | **2.28** | 2.28 | 2.27 | 2.27 |
| LSUN Bedrooms | **2.58** | 2.59 | 2.50 | 2.62 |

## 2. Octave shares of the between-image variance

Bins in cycles per image; the last bin is closed at 96 (the archived rows close it at 95.5, the analysis scripts' $N-1$ convention).

| dataset | 0.5-1 | 1-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-96 |
|---|---|---|---|---|---|---|---|---|
| IXI T1 | 4.0% | 14.8% | 10.4% | 16.6% | 28.3% | 17.5% | 7.4% | 1.0% |
| IXI T1, archived | 0.8% | 9.8% | 7.4% | 14.8% | 36.1% | 22.3% | 8.0% | 0.7% |
| OASIS-1 T1 | 2.4% | 6.7% | 9.4% | 22.9% | 28.9% | 19.7% | 8.0% | 2.0% |
| OASIS-1 T1, archived | 0.4% | 5.1% | 6.1% | 19.7% | 33.5% | 23.7% | 8.9% | 2.5% |
| LSUN Churches | 23.8% | 21.3% | 16.1% | 12.2% | 9.6% | 7.5% | 6.1% | 3.4% |
| LSUN Churches, archived | 25.1% | 21.0% | 15.4% | 11.9% | 9.5% | 7.5% | 6.2% | 3.5% |
| LSUN Bedrooms | 23.1% | 24.1% | 19.6% | 13.2% | 8.7% | 5.8% | 3.8% | 1.7% |
| LSUN Bedrooms, archived | 22.6% | 24.8% | 20.2% | 13.1% | 8.6% | 5.6% | 3.6% | 1.5% |

The coarsest bin is exactly three modes, and item (ii) of the checklist below reads it alone, so it is split here. $(1,0)$ is a ramp along the image rows (anterior–posterior on the MRI sets, whose slices carry `A` at the top and `L` on the image left), $(0,1)$ a ramp along the columns (left–right, the padded direction).

| dataset | $(0,1)$ left–right ramp | $(1,0)$ anterior–posterior ramp | $(1,1)$ diagonal | bin total |
|---|---|---|---|---|
| IXI T1 | 0.52% | 3.31% | 0.12% | 3.96% |
| OASIS-1 T1 | 0.23% | 2.14% | 0.07% | 2.44% |
| LSUN Churches | 5.27% | 16.09% | 2.44% | 23.80% |
| LSUN Bedrooms | 6.83% | 12.81% | 3.47% | 23.11% |

## 3. Inherited share at the three terminal blurs

$\sum_i P_i e^{-2\lambda_i t}/\sum_i P_i$ with $t = \sigma_{B,\max}^2/2$: the fraction of the between-image variance the prior hands to the sampler.

| dataset | $W/8$ (24 px) | $W/4$ (48 px) | $W/2$ (96 px) | archived (unregistered) |
|---|---|---|---|---|
| IXI T1 | 10.3% | 3.0% | 0.3% | 5.5% / 1.0% / 0.1% |
| OASIS-1 T1 | 5.1% | 1.6% | 0.2% | 2.8% / 0.5% / 0.0% |
| LSUN Churches | 29.1% | 13.2% | 1.8% | 30.1% / 13.9% / 1.9% |
| LSUN Bedrooms | 29.3% | 12.6% | 1.7% | 29.2% / 12.4% / 1.6% |

## 4. Per-level target spread — the crossover table

Spread = $\max_k R_k / \min_k R_k$ over levels $2 \dots K$, with $R_k = \sum_i (d_{k-1,i} - d_{k,i})^2 P_i$ the data-dependent part of the IHDM regression target. A schedule matched to a dataset has spread 1 on it by construction; the interesting numbers are off the diagonal.

| dataset | log ($W/2$) | IXI-matched ($W/2$) | own matched ($W/2$) | own vs IXI's, max / median of $\lvert s_{\text{own}}/s_{\text{IXI}} - 1 \rvert$ | archived log / IXI |
|---|---|---|---|---|---|
| IXI T1 | 8.7x | 1.0x | 1.002x | 0% / 0% | 50.8x / 1.0x |
| OASIS-1 T1 | 15.1x | 3.1x | 1.001x | 33% / 11% | 93.5x / 3.3x |
| LSUN Churches | 3.8x | 15.1x | 1.004x | 106% / 55% | 3.9x / 90.6x |
| LSUN Bedrooms | 8.0x | 16.4x | 1.004x | 129% / 72% | 8.9x / 92.8x |

## 5. Levels per $\sigma_B$ octave

Where each schedule spends its 200 levels. `oasis1_W2` is fitted by `profile_data` for this table and for item (v) of the checklist; no arm uses it, so it is not frozen.

| schedule | 0.5-1 px | 1-2 px | 2-4 px | 4-8 px | 8-16 px | 16-32 px | 32-64 px | 64-96 px | sum |
|---|---|---|---|---|---|---|---|---|---|
| `log_W2` | 27 | 26 | 26 | 26 | 27 | 26 | 26 | 16 | 200 |
| `log_W8` | 36 | 36 | 35 | 36 | 36 | 21 | 0 | 0 | 200 |
| `ixi_W2` | 19 | 29 | 37 | 33 | 26 | 24 | 23 | 9 | 200 |
| `ixi_W8` | 24 | 36 | 47 | 43 | 32 | 18 | 0 | 0 | 200 |
| `lsun_church_W2` | 19 | 21 | 23 | 26 | 28 | 31 | 33 | 19 | 200 |
| `oasis1_W8` | 25 | 37 | 45 | 45 | 33 | 15 | 0 | 0 | 200 |
| `lsun_bedroom_W2` | 15 | 18 | 22 | 26 | 32 | 34 | 34 | 19 | 200 |
| `oasis1_W2` (not frozen) | 21 | 31 | 39 | 38 | 29 | 19 | 16 | 7 | 200 |

## 6. Known-results checklist

| # | expected | outcome | measured |
|---|---|---|---|
| (i) | MRI alpha exceeds photograph alpha by at least 0.5 | **PASS** | min MRI alpha 3.11 - max photograph alpha 2.58 = +0.53; per dataset IXI T1 3.22, OASIS-1 T1 3.11, LSUN Churches 2.28, LSUN Bedrooms 2.58 |
| (ii) | the 0.5-1 c/img share is below 3% on both MRI sets and above 15% on both photograph sets | **FAIL** | IXI T1 3.96%, OASIS-1 T1 2.44%, LSUN Churches 23.80%, LSUN Bedrooms 23.11% |
| (iii) | the log-schedule spread is larger on MRI than on photographs | **PASS** | min MRI 8.7x vs max photograph 8.0x; per dataset IXI T1 8.7x, OASIS-1 T1 15.1x, LSUN Churches 3.8x, LSUN Bedrooms 8.0x |
| (iv) | the IXI schedule reduces OASIS-1's spread and increases Churches' | **PASS** | OASIS-1 15.1x -> 3.1x, Churches 3.8x -> 15.1x (log -> IXI-matched) |
| (v) | the two brain schedules agree within 35% (max) and the photograph schedule differs from IXI's by more than 50% | **PASS** | max abs(s_own / s_ixi - 1): OASIS-1 T1 33%, LSUN Churches 106%, LSUN Bedrooms 129% |

**4 of 5 items pass.** Failing: ii.

**Notes on the measured numbers** (facts from the tables above; no threshold was moved and no schedule was refitted to make an item pass).

- Registration raised the coarse-octave variance of both MRI sets by roughly a factor of five (0.8% → 3.96% on IXI, 0.4% → 2.44% on OASIS-1) and left the photograph sets unchanged. The ordering the design rests on is intact: the photograph sets still hold 6× the MRI sets' coarse share.
- On both MRI sets the coarse bin is carried by one mode, the anterior–posterior ramp $(1,0)$ (84% of the bin on IXI): a between-subject intensity gradient along the slice's rows, the signature of a residual receive-field / intensity-harmonisation difference rather than of coarse brain structure. The photograph sets spread the bin over all three modes.
- The log-schedule spread collapsed on both MRI sets (IXI 51× → 8.7×, OASIS-1 94× → 15.1×) while the photograph sets moved little. Item (iii) therefore holds by a margin of only 1.09× instead of the archived ~6×, and the design's contrast between the MRI and photograph arms is correspondingly weaker on the registered data than the proposal's numbers suggest.
- Item (ii) is the one that fails, on IXI alone (3.96 % against the 3 % threshold); OASIS-1 passes it at 2.44 % and both photograph sets clear the 15 % side by a wide margin.

## 7. Numbers for the figure caption

`docs/RESULTS/fig_data.pdf` / `.png`: four examples from the `ref` splits (rng 2026; MRI on the report's Fig. 1 plane, `slice == 5`) and four curves measured on the **3200-image** training splits.

- image count per curve: **3200**
- $\alpha$: IXI T1 3.22, OASIS-1 T1 3.11, LSUN Churches 2.28, LSUN Bedrooms 2.58
- MRI range $\alpha \approx 3.1$–$3.2$; photographs $\alpha \approx 2.3$–$2.6$

## 8. Provenance

| dataset | split | images | `sha256_images` |
|---|---|---|---|
| IXI T1 | train | 3200 | `93e99e89d96b9842…` |
| OASIS-1 T1 | train | 3200 | `4392a77136ca80d0…` |
| LSUN Churches | train | 3200 | `299c076853b0dfd2…` |
| LSUN Bedrooms | train | 3200 | `7e4c98d154462004…` |
| IXI T1 | ref | 800 | `93e99e89d96b9842…` |
| OASIS-1 T1 | ref | 800 | `4392a77136ca80d0…` |
| LSUN Churches | ref | 800 | `299c076853b0dfd2…` |
| LSUN Bedrooms | ref | 800 | `7e4c98d154462004…` |
