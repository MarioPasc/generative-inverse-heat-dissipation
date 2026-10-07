# FM — the method figure

Written by `ihdm.cli.paper_fm` (T8.4); do not edit by hand. Ticket: `docs/SPECIFICATIONS/M8-paper/T8.4-method-figure.md`.

## Files

| file | role | sha256 | identical to the previous run |
|---|---|---|---|
| `fm_method.pdf` | the version for LaTeX (vector, thumbnails embedded unresampled) | `7509656027d8985b…` | yes |
| `fm_method.svg` | editable master (Inkscape: text kept as text, thumbnails embedded) | `99b27292ccf58f8e…` | yes |
| `fm_method.png` | preview, 300 dpi | `e2ec027027e47ff0…` | yes |

Size 5.5 × 3.4 in; smallest font 7 pt; smallest mode thumbnail 0.26 in; smallest state thumbnail 0.42 in; text boxes overlapping: 0; text outside the figure: 0; embedded rasters at 310 dpi or more (PDF, SVG). Formulas use mathtext `stixsans` (FM only; F1 and F2 use the default DejaVu mathtext).

## Caption draft (194 words)

**IHDM, octave by octave.** Top: an IXI slice x is a weighted sum of orthonormal DCT-II modes φ_{i,j}; mode (i, j) carries c = ½√(i²+j²) cycles per image (c/img), and the modes fall into octaves (colours; the mean in grey), quarter-annuli of the (i, j) plane (inset, log radius). Rows: noise-free heat states of x. The forward process q(u_k | u_0) multiplies a mode at c by d = exp(−σ_B²/σ_n²), σ_n ≈ 43.2/c px; each state is drawn at σ_B = 43.2/c_b, where the octave starting at c_b keeps d = e⁻¹ and every finer mode d ≤ e⁻⁴. One arrow is therefore a macro-step over many of the K = 200 levels: default (log) 31/26/26/26/27/26/26/12 and matched 0/5/30/38/47/42/29/9 levels per octave, 0.5–1 to 64–96 c/img (right panel; 4 default levels below 0.5 c/img in the first). The ellipsis folds 32–96 c/img. The reverse row is the ideal path of p_θ(u_{k−1} | u_k): the same states read backwards. Dashed: the priors, where generation starts; the default (W/2) keeps almost nothing, the matched (W/8) the octaves below about 1.8 c/img. Right: IXI's between-image variance per octave against the 12.5% equal share of a 1/f² spectrum.

## Octaves: modes, variance and levels

Modes per octave of the 192 × 192 grid (`octave_masks`): the octaves partition the 29,135 non-DC modes with c ≤ 96 c/img; the DC mode and 7,728 corner modes above 96 c/img are in none.

| octave (c/img) | modes | IXI variance share | default levels (`log_W2`) | matched levels (`ixi_W8`) |
|---|---:|---:|---:|---:|
| 0.5–1 | 3 | 4.1% | 31 | 0 |
| 1–2 | 11 | 11.3% | 26 | 5 |
| 2–4 | 41 | 11.0% | 26 | 30 |
| 4–8 | 158 | 16.8% | 26 | 38 |
| 8–16 | 619 | 29.8% | 27 | 47 |
| 16–32 | 2,443 | 18.3% | 26 | 42 |
| 32–64 | 9,709 | 7.8% | 26 | 29 |
| 64–96 | 16,151 | 1.0% | 12 | 9 |
| total | 29,135 | 100.0% | 200 | 200 |

Levels per frequency octave: c_k = 43.2/σ_k binned on the octaves (levels below 0.5 c/img folded into the first: default 4, matched 0); the same function as F1's rugs. Levels per σ_B octave, checked against `data_profile.md` §5: `log_W2` 27/26/26/26/27/26/26/16 · `ixi_W8` 24/37/46/43/32/18/0/0.

## Macro-step check

At σ_b = 43.2/c_b (the released `DCTBlur` multiplier d = exp(−λσ_B²/2), float64): the modes exactly at c_b keep d = e⁻¹ = 0.367879; the released `DCTBlur` applied to φ_{0,2c_b} gives the same; the radial bin round(n) = 2c_b brackets e⁻¹; every mode at c ≥ 2c_b keeps d ≤ e⁻⁴ = 0.018316.

| octave (c/img) | σ_b (px) | d at c_b (modes) | d by DCTBlur | radial-bin mean d [min, max] (modes) | max d at c ≥ 2c_b (modes) | min d inside the octave | pass |
|---|---:|---|---:|---|---|---:|---|
| 0.5–1 | 86.43 | 0.367879 (2) | 0.367879 | 0.2904 [0.1353, 0.3679] (3) | 0.018316 (36,860) | 0.1353 | yes |
| 1–2 | 43.22 | 0.367879 (2) | 0.367879 | 0.3272 [0.2865, 0.3679] (4) | 0.018316 (36,849) | 0.03877 | yes |
| 2–4 | 21.61 | 0.367879 (2) | 0.367879 | 0.3569 [0.2865, 0.4437] (9) | 0.018316 (36,808) | 0.02209 | yes |
| 4–8 | 10.8 | 0.367879 (2) | 0.367879 | 0.3676 [0.3247, 0.4040] (13) | 0.018316 (36,650) | 0.02012 | yes |
| 8–16 | 5.402 | 0.367879 (2) | 0.367879 | 0.3677 [0.3456, 0.3901] (29) | 0.018316 (36,031) | 0.01853 | yes |
| 16–32 | 2.701 | 0.367879 (2) | 0.367879 | 0.3676 [0.3576, 0.3777] (48) | 0.018316 (33,588) | 0.01837 | yes |
| 32–64 | 1.35 | 0.367879 (2) | 0.367879 | 0.3679 [0.3622, 0.3736] (111) | 0.018316 (23,879) | 0.01833 | yes |
| 64–96 | 0.6752 | 0.367879 (2) | 0.367879 | 0.3677 [0.3652, 0.3706] (198) | 0.018311 (226) | 0.1054 | yes |

## Selected image and sources

- **image:** dataset index 3705, `reference_example(root, "ixi", np.random.default_rng(2026))`; the states are noise-free heat states (`paper_f1.heat_blur`, the released `DCTBlur` formula), the same slice as F1.
- **octave shares:** `docs/RESULTS/data_profile/ixi.npz`, `octave_shares_train` (checked against `octave_shares(power_train)`)
- **level counts:** `levels_per_frequency_octave` (paper_f1) of `schedules/log_W2.npy` and `schedules/ixi_W8.npy`; per sigma_B octave checked against `docs/RESULTS/data_profile.md` §5
- **mode counts:** `octave_masks(192)` (this module)
- **eval tars:** none read (no model sample is drawn)

## Command

```bash
python -m ihdm.cli.paper_fm --data-root /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project --out docs/RESULTS/paper --work /tmp/claude-1000/-home-mpascual-research-code-TFM/1a7523ce-2507-4f7b-9bc6-dae8070ec483/scratchpad/T8.4
```
