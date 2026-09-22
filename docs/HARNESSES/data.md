# H-DATA — dataset and schedule harness

## 1. Contract validation (after T1.1, T1.2)

```bash
for d in ixi oasis1 lsun_church lsun_bedroom; do
  python -m ihdm.cli.validate_dataset "$IHDM_DATA_ROOT/$d" || echo "FAIL $d"
done
```
Expected: four `OK <id>: 4000 images, 3200/800/40` lines, exit 0. Any violation is listed with
file/key/idx.

## 2. Counts and splits

```bash
python - <<'EOF'
from pathlib import Path; import os, json, pandas as pd
root = Path(os.environ["IHDM_DATA_ROOT"])
for d in ["ixi","oasis1","lsun_church","lsun_bedroom"]:
    ix = pd.read_csv(root/d/"index.csv"); sp = json.load(open(root/d/"splits.json"))
    print(d, len(ix), ix.split.value_counts().to_dict(), len(sp["train_subjects"]), len(sp["ref_subjects"]), len(sp["seed_subjects"]))
EOF
```
Expected: MRI `4000 {train: 3200, ref: 400, seed: 400}` (40 seed subjects × 10 slices are
labelled `seed`, the other 40 ref subjects `ref`), subjects 320/80/40; photographs
`{train: 3200, ref: 760, seed: 40}`, 3200/800/40 ids.

## 3. Eyeball gates (open the PNGs)

- `ixi/qc/registration_sheet.png`, `oasis1/qc/registration_sheet.png`: the green template
  brain-mask outline sits on the brain of every tile; the skull is outside it; no tile is
  rotated or mirrored relative to the template tile.
- `*/qc/orientation_sheet.png`: `A` at the top, `L` on the image left, for raw, registered and
  template tiles alike; the OASIS-1 raw tiles are not mirrored (compare the asymmetric template's
  left/right features against the registered subjects if in doubt).
- `*/qc/levels_sheet.png`: 10 slices from below the temporal lobes to above the ventricles.
- `*/qc/intensity_sheet.png`: no spike at 255 beyond the ≈1% by construction; MRI background is
  noisy, not zero (except the left–right padding columns).
- `lsun_*/qc/contact_sheet.png`: grayscale 192² crops, not resized, not padded.

## 4. Known results (after T1.3)

`docs/RESULTS/data_profile.md` checklist, all five items PASS:

| item | expected |
|---|---|
| $\alpha$ | MRI ≈ 3.5–3.9 > photographs ≈ 2.3–3.2 by ≥ 0.5 |
| 0.5–1 c/img share | MRI < 3%; photographs > 15% |
| log-schedule spread | MRI ≫ photographs (previously 45× and 84× vs 14×) |
| crossover | IXI schedule lowers OASIS-1's spread, raises Churches' |
| schedule agreement | IXI vs OASIS-1 within 35% max; IXI vs Churches > 50% |

If an item fails on the registered data, the number is reported and the orchestrator decides
(registration changes the coarse variance; the ordering must survive, the magnitudes may move).

## 5. Schedule files

```bash
python - <<'EOF'
from pathlib import Path; from ihdm.spectral.schedules import validate_schedule
for p in sorted(Path("schedules").glob("*.npy")): print(p.name, validate_schedule(p) or "OK")
EOF
```
Expected: 7 files, all `OK`; `schedules.json` lists the same 7 with hashes and the
levels-per-octave table (each row sums to 200).
