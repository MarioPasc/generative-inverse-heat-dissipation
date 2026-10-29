# Domain brief — expert-code (IHDM fork: spectral, schedules, prior, figure code, results)

Sources read (whole files): model_code/utils.py (186), model_code/torch_dct.py (199), ihdm/paths.py (92),
ihdm/data/{format.py 673, dataset.py 142, errors.py 9}, ihdm/spectral/{power.py 374, schedules.py 599,
profile.py 661, errors.py, __init__.py}, ihdm/sampling/{chain.py 321, seeds.py 230, loader.py 293},
ihdm/metrics/spectral.py (848), ihdm/analysis/paper_fm.py (1264), ihdm/analysis/paper_f1.py (1740),
ihdm/cli/{paper_fm 126, paper_f1 136, profile_data 931, build_schedules 266}, configs/spectral/arms.py (466),
schedules/schedules.json, docs/RESULTS/data_profile.md, docs/RESULTS/paper/{README,FM,F1,F2,T1,t1_main,A1}.md,
results_discussion.md §0–§2a.1 and §2a.6, README.md, environment.yml, pyproject.toml, .gitignore.
Not read: paper/F3.md, a1_table.md (not needed); results_discussion §2a.2–§13 (only on demand);
ihdm/analysis/style.py, tables.py, preprocess/* (imports checked only).
Written: 2026-10-29. Base: REPO@960d6e7. Read-only checks run in env `ihdm` (CPU).

## Map
- `model_code/utils.py:10-27` DCTBlur: released heat forward op, torch, `blur(x, fwd_steps)` indexes a sigma table.
- `model_code/torch_dct.py` dct/idct/dct_2d/idct_2d (norm='ortho'), port of zh217/torch-dct.
- `ihdm/paths.py` data_root() (env IHDM_DATA_ROOT, default /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project), schedules_dir().
- `ihdm/data/format.py` read_dataset(root) -> (images uint8 (N,H,W) memmap, index DataFrame, splits dict, DatasetMeta).
- `ihdm/data/dataset.py` NpyImageDataset (training loader; not needed for the notebook).
- `ihdm/spectral/power.py` mode_power, eigenvalues, radial_index, octave_bins/octave_shares, radial_profile, radial_spectrum, fit_alpha, inherited_share, power_law. Pure NumPy/SciPy.
- `ihdm/spectral/schedules.py` ScheduleSpec, log_schedule, matched_schedule, residual_norms, per_level_removed (E_k), per_level_spread, levels_per_octave (sigma_B pixel octaves), validate_schedule.
- `ihdm/spectral/profile.py` split_power, profile_split, reference_example, make_figure (fig_data), ARCHIVED numbers. Imports ihdm.preprocess.mri -> nibabel.
- `ihdm/metrics/spectral.py` radial_log_profile, lsd (+ octave errors), inherited_band (needs samples), spectral_grid.
- `ihdm/sampling/chain.py` prior_state (blur to level K + optional N(0,delta^2) noise), reverse_chain, sample_from_seeds.
- `ihdm/analysis/paper_f1.py` NumPy heat_blur/heat_multiplier, mode_scale (43.2), cycles_of_sigma, kept_power, half_power_cycles, levels_per_frequency_octave, released_dct_blur, F1 drawing.
- `ihdm/analysis/paper_fm.py` dct_basis_1d, dct_mode, dct_weights, synthesise, octave_masks, octave_mode_counts, macro_step_sigma, verify_macro_steps, FM drawing.
- `ihdm/cli/profile_data.py` writes docs/RESULTS/data_profile.md, data_profile/<d>.npz, fig_data.{pdf,png}.
- `ihdm/cli/build_schedules.py` writes schedules/*.npy + schedules.json.
- `configs/spectral/arms.py` arm -> schedule: A0 log_W2 96, A1 log_W8 24, A2 ixi_W2 96, A3 ixi_W8 24, A2p lsun_church_W2 96.

## Key facts for the goal
- Heat op: DCT coef of mode (i,j) times exp(-(pi^2 (i^2+j^2)/W^2) * sigma^2/2) — `model_code/utils.py:15-27` (freqs = pi*linspace(0,N-1,N)/N; t = sigma^2/2). NumPy twin `paper_f1.heat_blur(images, sigma)` `ihdm/analysis/paper_f1.py:504-524`; verified here: |DCTBlur(f32) - heat_blur(f64)| <= 2.8e-7 on IXI 3705 at levels 0/50/150/200 of log_W2.
- Frequency mapping: mode at c c/img has sigma_n = W/(sqrt2 pi c) ≈ 43.2/c px; d = exp(-sigma_B^2/sigma_n^2); half-variance at c½ ≈ 25.4/sigma_B — `paper_f1.py:15-21, 392-405, 449-464`; `paper_fm.py:20-33`.
- Per-mode power P_i: mean-centre across images, ortho DCT-II, mean of squares, P[0,0]=0, float64 — `ihdm/spectral/power.py:65-116`. Units: pixel values in [0,1]; IXI train sum P = 1259.3, Churches 1873.3 (computed).
- Octave bins in c/img (0.5,1,2,...,64,96], last closed at 96, corner modes >96 c/img excluded; mode c = sqrt(i^2+j^2)/2 — `power.py:50-56, 173-213`. Mode counts per octave 3/11/41/158/619/2,443/9,709/16,151 (29,135 total; 7,728 corner) — `docs/RESULTS/paper/FM.md:21-33`.
- Octave shares (train, 3,200 imgs) — `docs/RESULTS/data_profile.md:28-37`: IXI 4.1/11.3/11.0/16.8/29.8/18.3/7.8/1.0 %; OASIS-1 2.0/5.4/9.1/23.2/29.7/20.3/8.3/2.1; Churches 23.8/21.3/16.1/12.2/9.6/7.5/6.1/3.4; Bedrooms 23.1/24.1/19.6/13.2/8.7/5.8/3.8/1.7. 1/f^2 equal share = 12.5%.
- Spectral exponent alpha (10–67 c/img window): IXI 3.22, OASIS 3.10, Churches 2.28, Bedrooms 2.58 — `data_profile.md:17-20`; fit_alpha `power.py:280-313`.
- Inherited share I = sum d^2 P / sum P, t = sigma_max^2/2, all non-DC modes incl. corner — `power.py:316-346`. Train split (data_profile.md:92-95): IXI 8.5% (W/8) / 2.7% (W/4) / 0.3% (W/2); OASIS 4.1/1.3/0.2; Churches 29.1/13.2/1.8; Bedrooms 29.3/12.6/1.7. npz exact: IXI [0.08458, 0.02715, 0.00338].
- Schedules are frozen float64 (201,), s[0]=0, s[1]=0.5, s[200]=96 or 24 — `schedules.py:1-15`, `schedules/schedules.json`. log_schedule = [0, exp(linspace(log .5, log smax, 200))] `schedules.py:131-155`; rebuilt log_W2 is bit-identical to the frozen file (checked).
- Levels per sigma_B octave (px bins 0.5–1…64–96) `schedules.py:399-434`: log_W2 27/26/26/26/27/26/26/16; log_W8 36/36/35/36/36/21/0/0; ixi_W2 19/30/38/35/26/23/20/9; ixi_W8 24/37/46/43/32/18/0/0; lsun_church_W2 19/21/23/26/28/31/33/19 — `data_profile.md:114-121`.
- Levels per FREQUENCY octave (c_k = 43.2/sigma_k, below 0.5 c/img folded into first) `paper_f1.py:777-808`: log_W2 31/26/26/26/27/26/26/12 (4 folded); ixi_W8 0/5/30/38/47/42/29/9 — `FM.md:23-35`, `F1.md:45`.
- E_k = R_k = sum_i (d_{k-1,i} - d_{k,i})^2 P_i, levels 2..K (199 values) `schedules.py:185-216, 352-370`; spread = max/min `schedules.py:373-396`. Spreads on IXI train: log_W2 8.9x, ixi_W2 1.0019x, log_W8 8.52x (computed, not in docs), ixi_W8 1.0042x; Churches: log_W2 3.8x, ixi_W2 15.5x; OASIS log 18.7x, ixi 2.8x; Bedrooms log 8.0x, ixi 16.9x — `data_profile.md:103-106`.
- Prior state = DCTBlur(x, K) [+ delta N(0,I) if prior_noise] `ihdm/sampling/chain.py:157-194`; config prior_noise=True, delta=1.25*0.01 `configs/spectral/arms.py:362-364, 390`. Figures draw it noise-free: `paper_f1.py:1078` (priors at 96 and 24 px of first 3 ref subjects).
- FM/F1 IXI slice = dataset index 3705 (IXI607, slice 5, split 'seed') via reference_example(root,'ixi',default_rng(2026)) `F1.md:23`, verified. fig_data examples (one shared rng 2026, order ixi,oasis1,church,bedroom): rows 3705, 975, 107, 2544 (verified).

## Entities and where they live
| Entity | Location | Note |
|---|---|---|
| DCTBlur(blur_sigmas, image_size, device) | model_code/utils.py:10-27 | x (B,C,H,W) float32; fwd_steps long (B,); imports model_code.unet+torch (~1.7 s import) |
| heat_blur(images, sigma) | analysis/paper_f1.py:504 | NumPy f64, (W,W) or (n,W,W); importing paper_f1 pulls matplotlib + analysis.style/tables |
| heat_multiplier(width, sigma) | paper_f1.py:486 | d (W,W) |
| kept_power(cycles, sigma, width) | paper_f1.py:428 | d^2 as function of c/img (continuous curve) |
| mode_scale / cycles_of_sigma | paper_f1.py:392 / 408 | 43.2 at W=192 |
| levels_per_frequency_octave(levels, width) | paper_f1.py:777 | levels = schedule[1:] |
| dct_mode / dct_weights / synthesise / octave_masks / octave_mode_counts | paper_fm.py:179/205/221/256/282 | band-pass images = synthesise(weights*mask) (F2 strip) |
| macro_step_sigma(low, width) | paper_fm.py:315 | 43.2/c_b: 86.4, 43.2, 21.6, 10.8, 5.40, 2.70, 1.35, 0.675 px |
| mode_power(images, chunk=256) | spectral/power.py:65 | (n,W,W) in [0,1] -> (W,W) f64; IXI train 1.9 s CPU |
| octave_shares(power) | power.py:185 | dict label->share, sums to 1 |
| radial_spectrum(power) | power.py:253 | (c = b/2 for b=1..W-1, mean P per band); fig_data plots var/var.sum() |
| inherited_share(power, sigma_max) | power.py:316 | float in [0,1] |
| ScheduleSpec(kind, sigma_max, sigma_min=.5, K=200, fitted_on) | schedules.py:76 | matched needs fitted_on string |
| log_schedule(spec) / matched_schedule(power, spec) | schedules.py:131 / 274 | log: instant; matched: > 3 min CPU (80x bisection) — load frozen instead |
| per_level_removed(power, schedule) / per_level_spread | schedules.py:352 / 373 | (199,) ; 0.07–0.11 s |
| levels_per_octave(schedule) | schedules.py:399 | dict of 8 px-octave counts |
| read_dataset(root, mmap=True) | data/format.py:449 | root = data_root()/'ixi' |
| reference_example(root, id, rng) | spectral/profile.py:288 | ref split; MRI restricted to slice==5 |
| split_power(root, id, split) | profile.py:227 | (power, n, sha) |
| prior_state / reverse_chain | sampling/chain.py:157 / 197 | not to be run (needs checkpoint) |
| lsd / radial_log_profile | metrics/spectral.py:524 / 418 | per-image DC removed; 48 log bins, 43 populated |
| make_figure (fig_data) | profile.py:598 | IEEE rcParams, Tol colours `profile.py:114-129` |

## Data and artefacts available locally
| What | Path | Format / size | Readable now? |
|---|---|---|---|
| Profile per dataset | docs/RESULTS/data_profile/{ixi,oasis1,lsun_church,lsun_bedroom}.npz (tracked) | 552 KB each: power_train/power_ref (192,192) f64, cycles_per_image (191,), radial_train/ref, octave_labels, octave_shares_train/ref (8,), alpha_*, inherited_train (3,) [W/8,W/4,W/2], spread_log/ixi/own | yes; power_train equals a fresh mode_power exactly (checked IXI, Churches) |
| No-N4 sensitivity | data_profile/{ixi,oasis1}_no_n4.npz | 280 KB; + schedule_W2/W8 | yes |
| Frozen schedules | schedules/*.npy (+ schedules.json) | 1.7 KB each, (201,) f64 | yes, all validate |
| Inherited constants (ref split) | docs/RESULTS/inherited_band_constants.json | 28 KB; per dataset, sigma 24/96: share_predicted, radial predicted | yes |
| Figures | docs/RESULTS/fig_data.png (404 KB); paper/{fm_method,f1_visual_abstract,f2_arms,a1_natural_images}.{png,pdf,svg}; figures/fig1–fig7 png/pdf | PNG 0.2–1 MB | yes |
| Tables | docs/RESULTS/tables/tables.json (212 KB), paper/t1_main.md, F2.md tables | JSON/MD | yes |
| F2(c) per-octave LSD | $DATA/_results/runs/<run>/final.json key lsd_octaves (+ radial) | ~KB per run; seed means in F2.md:106-111 | yes (MeningD2) |
| Datasets | $DATA/{ixi,oasis1,lsun_church,lsun_bedroom}/images.npy | uint8 (4000,192,192), 140.6 MB; index.csv, splits.json (train 3200 / ref 800 / seed 400 MRI or 40 photo), meta.json | yes; read 0.01–0.1 s, one slice 0.04–0.3 s |
| Eval tars | /media/mpascual/Sandisk2TB/research/spectral_allocation_heat_diffusion_project/evaluation/eval_2488269/<run>_amp-fp16.tar | 635 MB each | yes (do not need) |

$DATA = /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project.

## Runtime (CPU, measured here)
- Cheap (< 2 s): load npz; load schedules; octave_shares, inherited_share, radial_spectrum, per_level_removed/spread (~0.1 s), levels_per_octave, log_schedule, heat_blur of one slice, read_dataset + one slice.
- Seconds: mode_power on 3,200 train images ~1.9 s (+0.4–2.3 s to load from the external drive); DCTBlur first call ~4 s (torch warm-up).
- Expensive: matched_schedule > 3 min per schedule (timing run still going at write time); verify_mapping/kept_variance_per_octave on full train (blurs whole stack); anything in sampling/ (needs checkpoints, GPU).

## Inconsistencies and risks
- I(IXI, W/8) is 8.5% on the train split (data_profile.md:92, F1 caption) but 8.0% on the ref split (results_discussion.md:528, 290 "0.080"; inherited_band_constants.json share_predicted 0.0795447, n_ref 800). Both correct; the notebook must say which split.
- "Levels per octave" has two meanings: sigma_B pixel octaves (levels_per_octave; log_W2 27/26/.../16) vs frequency octaves via c=43.2/sigma (log_W2 31/26/.../12). FM/F1 print the frequency version; data_profile §5 the pixel version.
- "Removed variance" naming: per_level_removed docstring says "between-image variance removed by each step" (`schedules.py:353`) but the quantity is the squared-difference norm sum (d_{k-1}-d_k)^2 P (`schedules.py:200`), not sum (d_{k-1}^2 - d_k^2) P. data_profile.md:99 calls it correctly "the data-dependent part of the IHDM regression target".
- data_profile.md:3 says produced at 31c590e (2026-09-23); the IXI schedules/profile were refit later on the stratified split (git fc0692e "refit ixi_W2/ixi_W8 on the stratified split; regen profile"); schedules.json ixi_W2/W8 git_sha 31c590e. Numbers in npz/md/json agree with each other (checked).
- .gitignore ignores `*.npy` except `schedules/*.npy` (.gitignore:5-6): any `assets/*.npy` would be silently untracked; use .npz/.png/.json or add an exception.
- README.md is still the upstream README (MNIST/CIFAR); only the last section mentions the fork.
- profile.py imports ihdm.preprocess.mri (nibabel); paper_f1/paper_fm import matplotlib and set styles; importing them in a notebook is fine in env ihdm but heavier than spectral.power.
- `arms.py` comment says it validates locally because T1.3 "not merged yet" (`arms.py:212-215`) — stale comment.
- results_discussion §1 H1d says predicted I = 0.08 (IXI) and 0.29 (Churches) at W/8 — ref-split values.
- Licences: the repo states NO dataset licence (grep over docs/ found none). Sources recorded: LSUN from HF `tglcourse/lsun_church_train` and `pcuenq/lsun-bedrooms` (meta.json hf_dataset; docs/SPECIFICATIONS/03-data-format.md:121). IXI CC BY-SA 3.0 is UNVERIFIED in this repo; OASIS-1 terms UNVERIFIED.

## Open questions
- Which "matched schedule" the notebook shows: ixi_W2 (spacing alone, A2) or ixi_W8 (A3, used by FM/F1). Both frozen.
- Whether to load P from npz (recommended, exact) or recompute from images (needs drive; 2–4 s).
