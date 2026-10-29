# Tutorial assets

These are the only files that `ihdm_mri_tutorial.ipynb` reads besides the tracked results under
`docs/RESULTS/` and `schedules/`. All of them are rebuilt from the data root and the result files with

```bash
python -m ihdm.cli.build_tutorial_assets --force            # into assets/
python -m ihdm.cli.build_tutorial_assets --out /tmp/assets  # into another folder, for a byte comparison
```

The command reads the datasets under `$IHDM_DATA_ROOT` (default
`/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project`) and the per-run
`final.json` files under `$IHDM_DATA_ROOT/_results`. Its outputs carry no timestamp or commit hash,
so a second build from the same sources writes the same bytes.

## Images (`images/`)

Each PNG is one row of the dataset's `images.npy`, saved as an 8-bit grayscale PNG with no
resampling: uint8, 192 × 192 pixels, values as stored. `images/manifest.json` records the row,
subject, slice, split, source and SHA-256 of each file, and the notebook takes its labels from it.

| File | Dataset | Row | Subject, slice | Split | Where else it appears |
|---|---|---|---|---|---|
| `ixi_3705.png` | IXI T1 | 3705 | IXI607 (Guys), slice 5 ($z=+18$ mm) | seed (held out) | report Figure 1, `docs/RESULTS/paper/FM.md`, `F1.md` |
| `ixi_3311.png` | IXI T1 | 3311 | IXI541 (IOP), slice 1 ($z=-15$ mm) | train | row "train 1" of Figure 2a (`F2.md`) |
| `lsun_church_1180.png` | LSUN Churches | 1180 | church_01180 | train | the A1 figure (`docs/RESULTS/paper/A1.md`) |
| `lsun_church_107.png` | LSUN Churches | 107 | church_00107 | seed (held out) | `docs/RESULTS/fig_data.png` |

The IXI slices come from T1-weighted volumes that we rigidly registered to MNI152 2009c at 1 mm,
corrected with N4 and scaled to [0, 1] by the 99th percentile of the foreground
(`ihdm.preprocess.mri`, pipeline 1.1). The Churches images are photographs converted to grayscale,
resized to a 256 px short side when the short side exceeds 256 px, and centre-cropped to 192 × 192
(`ihdm.preprocess.photos`, pipeline 1.0).

**Licences.**

- IXI: the IXI dataset is distributed under CC BY-SA 3.0. Attribution: "IXI dataset,
  https://brain-development.org/ixi-dataset/". The two derived slices here are shared under the
  same licence, CC BY-SA 3.0.
- LSUN Churches: source Hugging Face `tglcourse/lsun_church_train` (shard
  `data/test-00000-of-00001-b3f98c62a94e5d25.parquet`, rows 1180 and 107), taken from LSUN
  (Yu et al., 2015, arXiv 1506.03365). Licence of the individual images not stated by the source;
  check before public release.

## Results (`results/f2_results.json`)

The numbers that Figure 2 of the report plots, per run, for the 14 brain-MRI runs (IXI: A0, A3 with
three seeds, A1, A2 with two; OASIS-1: A0, A3 with two). Every field names its source in the file's
`sources` entry, with the SHA-256 of the file it was read from.

| Field | Figure 2 panel | Source |
|---|---|---|
| `runs.<id>.precision_recall.{F,H}.{precision,recall}` | (b) top | `docs/RESULTS/heldout_fidelity/heldout_fidelity.json` (A0, A1, A3, OASIS-1) and `docs/RESULTS/paper/f2_a2_heldout/heldout_fidelity.json` (A2), keys `runs.<id>.metrics.{F_rminus,H_rminus}` |
| `runs.<id>.distance_median.{F_seed,F_train,H_seed,H_train}` | (b) bottom | parsed from the per-run table of `docs/RESULTS/paper/F2.md` |
| `distance_seed_mean.<dataset>.<configuration>` | (b) bottom | the "seed mean" rows of the same table; F2.md computed them from unrounded medians, so they can differ by 0.001 from the mean of the rounded per-run values |
| `distance_band_rminus`, `distance_pooled_iqr` | (b) bottom | parsed from the band and pooled-IQR tables of `docs/RESULTS/paper/F2.md` |
| `runs.<id>.lsd_octaves` (IXI only) | (c) | `lsd_octaves` of `$IHDM_DATA_ROOT/_results/runs/<id>/final.json`; the build checks that the seed means equal the panel (c) table of `F2.md` to its printed rounding (0.0005) |

The distances have no machine-readable source: `ihdm.analysis.seed_distance` wrote them to a scratch
cache that no longer exists, so the build reads the tables that `ihdm.cli.paper_f2` generated into
`F2.md`. Hand check: `ixi_A0_s1` has F precision 0.6015 and F recall 0.3025 in both
`heldout_fidelity.json` and `F2.md`.
