# Spectral profile of the four training splits, and the frozen schedules

Produced by `python -m ihdm.cli.profile_data` at `0eb8982f37c8` on 2026-09-23. Tickets T1.3 and T1.4; contracts `04-run-artifacts.md` §1 and `05-metrics.md` §1.

Every curve and every table below is measured on the **training** split of each dataset (3200 images of $192^2$, values in $[0, 1]$); the `ref` split (800 images) is profiled beside it as a consistency check. The per-mode variance is mean-centred across images, in the orthonormal DCT-II basis, with the DC mode excluded; a mode of radial index $n$ carries $n/2$ cycles per image and modes above 96 cycles per image are excluded from the octave shares.

The MRI datasets are **rigidly registered and N4 bias-field corrected** (decision D15, ticket T1.4): N4 runs on each registered volume, with SimpleITK's default parameters and the dilated template brain mask, before the foreground-p99 intensity scaling. The **no N4** columns and rows are the same quantities measured on the archived uncorrected datasets under `$IHDM_DATA_ROOT/_sensitivity/`, which hold the identical 400 subjects, splits and slices. They are the sensitivity row the reviewer asked for: the claim is stated on the corrected data, the uncorrected numbers are printed beside it, and the crossover table is recomputed rather than assumed. N4 is applied to the MRI side only, so the photograph numbers are identical in both branches and their `no N4` cells are left empty.

The **archived** columns are the pre-registration numbers of `worklog/sessions/2026-09-21_ihdm-knob-proposals/native192_profile_output_unprocessed.md` (480 head-centred crops of the *unprocessed* volumes per dataset, no rigid registration, no intensity harmonisation, no padding). They are not a target: registration changes the coarse variance by construction. They are printed so the size and the direction of that change are visible.

## 1. Spectral exponent

$\alpha$ is fitted by least squares of $\log \bar P$ on $\log n$ over the integer DCT radii $b \in [0.10 W, 0.70 W] = [20, 134]$, i.e. **10.0 – 67.0 cycles per image** — the window of `analysis/control_profile.py: fit_alpha`, held fixed across datasets because the brain spectrum is curved. The third column repeats the fit over the window the ticket's prose names (1 – 48 cycles per image); see the ticket log, decision D-T1.3-1.

| dataset | $\alpha$ (train) | $\alpha$ (ref) | $\alpha$ (train, 1–48 c/img window) | **no N4** (train) | archived, unregistered |
|---|---|---|---|---|---|
| IXI T1 | **3.22** | 3.22 | 2.14 | 3.22 | 3.47 |
| OASIS-1 T1 | **3.10** | 3.08 | 2.09 | 3.11 | 3.16 |
| LSUN Churches | **2.28** | 2.28 | 2.27 | &mdash; | 2.27 |
| LSUN Bedrooms | **2.58** | 2.59 | 2.50 | &mdash; | 2.62 |

## 2. Octave shares of the between-image variance

Bins in cycles per image; the last bin is closed at 96 (the archived rows close it at 95.5, the analysis scripts' $N-1$ convention).

| dataset | 0.5-1 | 1-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64-96 |
|---|---|---|---|---|---|---|---|---|
| IXI T1 | 4.1% | 11.2% | 10.9% | 16.8% | 29.8% | 18.3% | 7.8% | 1.0% |
| IXI T1, **no N4** | 4.0% | 14.8% | 10.4% | 16.6% | 28.3% | 17.5% | 7.4% | 1.0% |
| IXI T1, archived | 0.8% | 9.8% | 7.4% | 14.8% | 36.1% | 22.3% | 8.0% | 0.7% |
| OASIS-1 T1 | 2.0% | 5.4% | 9.1% | 23.2% | 29.7% | 20.3% | 8.3% | 2.1% |
| OASIS-1 T1, **no N4** | 2.4% | 6.7% | 9.4% | 22.9% | 28.9% | 19.7% | 8.0% | 2.0% |
| OASIS-1 T1, archived | 0.4% | 5.1% | 6.1% | 19.7% | 33.5% | 23.7% | 8.9% | 2.5% |
| LSUN Churches | 23.8% | 21.3% | 16.1% | 12.2% | 9.6% | 7.5% | 6.1% | 3.4% |
| LSUN Churches, archived | 25.1% | 21.0% | 15.4% | 11.9% | 9.5% | 7.5% | 6.2% | 3.5% |
| LSUN Bedrooms | 23.1% | 24.1% | 19.6% | 13.2% | 8.7% | 5.8% | 3.8% | 1.7% |
| LSUN Bedrooms, archived | 22.6% | 24.8% | 20.2% | 13.1% | 8.6% | 5.6% | 3.6% | 1.5% |

The coarsest bin is exactly three modes, and item (ii) of the checklist below reads it alone, so it is split here. $(1,0)$ is a ramp along the image rows (anterior–posterior on the MRI sets, whose slices carry `A` at the top and `L` on the image left), $(0,1)$ a ramp along the columns (left–right, the padded direction).

| dataset | $(0,1)$ left–right ramp | $(1,0)$ anterior–posterior ramp | $(1,1)$ diagonal | bin total | $(1,0)$ share of the bin |
|---|---|---|---|---|---|
| IXI T1 | 0.50% | 3.46% | 0.11% | 4.07% | 85% |
| IXI T1, **no N4** | 0.52% | 3.31% | 0.12% | 3.96% | 84% |
| OASIS-1 T1 | 0.22% | 1.75% | 0.06% | 2.02% | 86% |
| OASIS-1 T1, **no N4** | 0.23% | 2.14% | 0.07% | 2.44% | 88% |
| LSUN Churches | 5.27% | 16.09% | 2.44% | 23.80% | 68% |
| LSUN Bedrooms | 6.83% | 12.81% | 3.47% | 23.11% | 55% |

### 2.1 The coarse bin per acquisition site

IXI mixes three sites and two field strengths (Guys 1.5 T, HH 3 T, IOP 1.5 T); OASIS-1 is a single 1.5 T scanner. Each row is measured on that site's training images alone, so `coarse share` is the site's own between-image variance in the 0.5–1 c/img bin, not its contribution to the pooled one. This is the table decision D15 asks for: if the coarse bin were a multi-site artefact it would be small within a site and large across sites, and N4 should shrink the $(1,0)$ share everywhere.

| dataset | site | subjects | images | coarse share, N4 | coarse share, no N4 | $(1,0)$ share of the bin, N4 | no N4 |
|---|---|---|---|---|---|---|---|
| IXI T1 | Guys | 159 | 1590 | **3.11%** | 3.51% | **89%** | 89% |
| IXI T1 | HH | 120 | 1200 | **2.90%** | 4.12% | **70%** | 76% |
| IXI T1 | IOP | 41 | 410 | **2.62%** | 3.21% | **73%** | 78% |
| OASIS-1 T1 | WashU | 320 | 3200 | **2.02%** | 2.44% | **86%** | 88% |

## 3. Inherited share at the three terminal blurs

$\sum_i P_i e^{-2\lambda_i t}/\sum_i P_i$ with $t = \sigma_{B,\max}^2/2$: the fraction of the between-image variance the prior hands to the sampler.

| dataset | $W/8$ (24 px) | $W/4$ (48 px) | $W/2$ (96 px) | **no N4** $W/8$ / $W/4$ / $W/2$ | archived (unregistered) |
|---|---|---|---|---|---|
| IXI T1 | 8.4% | 2.7% | 0.3% | 10.3% / 3.0% / 0.3% | 5.5% / 1.0% / 0.1% |
| OASIS-1 T1 | 4.1% | 1.3% | 0.2% | 5.1% / 1.6% / 0.2% | 2.8% / 0.5% / 0.0% |
| LSUN Churches | 29.1% | 13.2% | 1.8% | &mdash; | 30.1% / 13.9% / 1.9% |
| LSUN Bedrooms | 29.3% | 12.6% | 1.7% | &mdash; | 29.2% / 12.4% / 1.6% |

## 4. Per-level target spread — the crossover table

Spread = $\max_k R_k / \min_k R_k$ over levels $2 \dots K$, with $R_k = \sum_i (d_{k-1,i} - d_{k,i})^2 P_i$ the data-dependent part of the IHDM regression target. A schedule matched to a dataset has spread 1 on it by construction; the interesting numbers are off the diagonal.

| dataset | log ($W/2$) | IXI-matched ($W/2$) | own matched ($W/2$) | own vs IXI's, max / median of $\lvert s_{\text{own}}/s_{\text{IXI}} - 1 \rvert$ | **no N4**: log / IXI-matched / max dev | archived log / IXI |
|---|---|---|---|---|---|---|
| IXI T1 | 8.9x | 1.0x | 1.002x | 0% / 0% | 8.7x / 1.0x / 0% | 50.8x / 1.0x |
| OASIS-1 T1 | 18.7x | 2.8x | 1.001x | 31% / 10% | 15.1x / 3.1x / 33% | 93.5x / 3.3x |
| LSUN Churches | 3.8x | 15.6x | 1.004x | 123% / 67% | 3.8x / 15.1x / 106% | 3.9x / 90.6x |
| LSUN Bedrooms | 8.0x | 17.0x | 1.004x | 145% / 86% | 8.0x / 16.4x / 129% | 8.9x / 92.8x |

## 5. Levels per $\sigma_B$ octave

Where each schedule spends its 200 levels. `oasis1_W2` is fitted by `profile_data` for this table and for item (v) of the checklist; no arm uses it, so it is not frozen. The `no N4` rows are the same fits on the uncorrected training splits, i.e. the arrays T1.3 froze and this refit replaced.

| schedule | 0.5-1 px | 1-2 px | 2-4 px | 4-8 px | 8-16 px | 16-32 px | 32-64 px | 64-96 px | sum |
|---|---|---|---|---|---|---|---|---|---|
| `log_W2` | 27 | 26 | 26 | 26 | 27 | 26 | 26 | 16 | 200 |
| `log_W8` | 36 | 36 | 35 | 36 | 36 | 21 | 0 | 0 | 200 |
| `ixi_W2` | 19 | 30 | 38 | 35 | 26 | 23 | 20 | 9 | 200 |
| `ixi_W8` | 24 | 37 | 47 | 42 | 32 | 18 | 0 | 0 | 200 |
| `lsun_church_W2` | 19 | 21 | 23 | 26 | 28 | 31 | 33 | 19 | 200 |
| `oasis1_W8` | 25 | 37 | 46 | 45 | 33 | 14 | 0 | 0 | 200 |
| `lsun_bedroom_W2` | 15 | 18 | 22 | 26 | 32 | 34 | 34 | 19 | 200 |
| `oasis1_W2` (not frozen) | 22 | 32 | 39 | 39 | 29 | 18 | 14 | 7 | 200 |
| `ixi_W2` **no N4** | 19 | 29 | 37 | 33 | 26 | 24 | 23 | 9 | 200 |
| `ixi_W8` **no N4** | 24 | 36 | 47 | 43 | 32 | 18 | 0 | 0 | 200 |
| `oasis1_W2` **no N4** | 21 | 31 | 39 | 38 | 29 | 19 | 16 | 7 | 200 |
| `oasis1_W8` **no N4** | 25 | 37 | 45 | 45 | 33 | 15 | 0 | 0 | 200 |

## 6. Known-results checklist

The claim is the `N4` column. The `no N4` column is the same five items evaluated on the archived uncorrected datasets, each scored against schedules refitted on those same uncorrected training splits.

| # | expected | outcome (N4) | outcome (no N4) | measured on the N4 data | measured without N4 |
|---|---|---|---|---|---|
| (i) | MRI alpha exceeds photograph alpha by at least 0.5 | **PASS** | PASS | min MRI alpha 3.10 - max photograph alpha 2.58 = +0.52; per dataset IXI T1 3.22, OASIS-1 T1 3.10, LSUN Churches 2.28, LSUN Bedrooms 2.58 | min MRI alpha 3.11 - max photograph alpha 2.58 = +0.53; per dataset IXI T1 3.22, OASIS-1 T1 3.11, LSUN Churches 2.28, LSUN Bedrooms 2.58 |
| (ii) | the 0.5-1 c/img share is below 3% on both MRI sets and above 15% on both photograph sets | **FAIL** | FAIL | IXI T1 4.07%, OASIS-1 T1 2.02%, LSUN Churches 23.80%, LSUN Bedrooms 23.11% | IXI T1 3.96%, OASIS-1 T1 2.44%, LSUN Churches 23.80%, LSUN Bedrooms 23.11% |
| (iii) | the log-schedule spread is larger on MRI than on photographs | **PASS** | PASS | min MRI 8.9x vs max photograph 8.0x; per dataset IXI T1 8.9x, OASIS-1 T1 18.7x, LSUN Churches 3.8x, LSUN Bedrooms 8.0x | min MRI 8.7x vs max photograph 8.0x; per dataset IXI T1 8.7x, OASIS-1 T1 15.1x, LSUN Churches 3.8x, LSUN Bedrooms 8.0x |
| (iv) | the IXI schedule reduces OASIS-1's spread and increases Churches' | **PASS** | PASS | OASIS-1 18.7x -> 2.8x, Churches 3.8x -> 15.6x (log -> IXI-matched) | OASIS-1 15.1x -> 3.1x, Churches 3.8x -> 15.1x (log -> IXI-matched) |
| (v) | the two brain schedules agree within 35% (max) and the photograph schedule differs from IXI's by more than 50% | **PASS** | PASS | max abs(s_own / s_ixi - 1): OASIS-1 T1 31%, LSUN Churches 123%, LSUN Bedrooms 145% | max abs(s_own / s_ixi - 1): OASIS-1 T1 33%, LSUN Churches 106%, LSUN Bedrooms 129% |

**4 of 5 items pass on the N4 data.** Failing: ii. Without N4: 4 of 5, failing ii.

**Notes on the measured numbers** (facts from the tables above; no threshold was moved, no N4 parameter was chosen by looking at a spectral number, and no schedule was refitted to make an item pass).

- **N4 moved the coarse bin of the MRI sets** from 3.96% to 4.07% on IXI and from 2.44% to 2.02% on OASIS-1, and the share of that bin carried by the single anterior–posterior ramp mode $(1,0)$ from 84% to 85% on IXI. The photograph sets are byte-identical in both branches: N4 corrects an MRI acquisition artefact and nothing was applied to them.
- Registration raised the coarse-octave variance of both MRI sets well above the archived pre-registration values (0.8% → 4.07% on IXI, 0.4% → 2.02% on OASIS-1) and left the photograph sets unchanged. The ordering the design rests on is intact: the photograph sets hold 6× the MRI sets' coarse share.
- The log-schedule spread is far below the archived values on both MRI sets (IXI 51× → 8.9×, OASIS-1 94× → 18.7×) while the photograph sets moved little. Item (iii) therefore holds by a margin of 1.12× instead of the archived ~6×, and the design's contrast between the MRI and photograph arms is correspondingly weaker on the processed data than the proposal's numbers suggest.
- Failing item(s): (ii) IXI T1 4.07%, OASIS-1 T1 2.02%, LSUN Churches 23.80%, LSUN Bedrooms 23.11%.

## 7. Numbers for the figure caption

`docs/RESULTS/fig_data.pdf` / `.png`: four examples from the `ref` splits (rng 2026; MRI on the report's Fig. 1 plane, `slice == 5`) and four curves measured on the **3200-image** training splits.

- image count per curve: **3200**
- $\alpha$: IXI T1 3.22, OASIS-1 T1 3.10, LSUN Churches 2.28, LSUN Bedrooms 2.58
- MRI range $\alpha \approx 3.1$–$3.2$; photographs $\alpha \approx 2.3$–$2.6$

## 8. Provenance

| dataset | split | images | `sha256_images` |
|---|---|---|---|
| IXI T1 | train | 3200 | `b666e407e9afcc03…` |
| OASIS-1 T1 | train | 3200 | `8396e1c113abb4df…` |
| LSUN Churches | train | 3200 | `299c076853b0dfd2…` |
| LSUN Bedrooms | train | 3200 | `7e4c98d154462004…` |
| IXI T1 | ref | 800 | `b666e407e9afcc03…` |
| OASIS-1 T1 | ref | 800 | `8396e1c113abb4df…` |
| LSUN Churches | ref | 800 | `299c076853b0dfd2…` |
| LSUN Bedrooms | ref | 800 | `7e4c98d154462004…` |
| IXI T1 (no N4) | train | 3200 | `93e99e89d96b9842…` |
| OASIS-1 T1 (no N4) | train | 3200 | `4392a77136ca80d0…` |
