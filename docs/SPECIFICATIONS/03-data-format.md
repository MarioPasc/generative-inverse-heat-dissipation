# 03 — Standard dataset format (frozen contract)

Producer: T1.1 (MRI), T1.2 (photographs). Consumers: `ihdm/data/dataset.py`, T1.3 (profile),
T4.x (metrics). Owner of the module: T0.1 (`ihdm/data/format.py`). Any change goes through the
orchestrator.

## 1. Layout

```
$IHDM_DATA_ROOT/<dataset_id>/
├── images.npy        # uint8, shape (N, 192, 192), C-order, grayscale; value v means intensity v/255
├── index.csv         # one row per image, in images.npy order (see §2)
├── splits.json       # {"train": [...], "ref": [...], "seed": [...]} lists of image indices (see §3)
├── meta.json         # provenance and parameters (see §4)
└── qc/               # contact sheets and any per-dataset diagnostic (PNG/MD), free-form
```

`dataset_id ∈ {ixi, oasis1, lsun_church, lsun_bedroom}`. `N = 4000` for every dataset.

## 2. `index.csv`

Columns, in this order, no index column, UTF-8, header row:

| column | type | meaning |
|---|---|---|
| `idx` | int | row number, equals the position in `images.npy` |
| `subject` | str | MRI: subject id (`IXI002`, `OAS1_0001`); photographs: the image id (each image is its own subject) |
| `slice` | int | MRI: slice number within the subject's stack (0..9); photographs: 0 |
| `z_mm` | float | MRI: the MNI $z$ coordinate of the slice in mm; photographs: `nan` |
| `source` | str | the raw file the image came from (path relative to the raw root, or the HF shard + row) |
| `split` | str | `train`, `ref` or `seed` (`seed ⊂ ref` logically, but the column holds the finest label: an image of a seed subject is labelled `seed`) |

## 3. `splits.json`

```json
{"train": [idx, ...], "ref": [idx, ...], "seed": [idx, ...],
 "train_subjects": [...], "ref_subjects": [...], "seed_subjects": [...],
 "rule": "80/20 by subject, seed = 40 subjects drawn from ref with rng seed 2026",
 "rng_seed": 2026}
```

- `train` and `ref` partition `range(N)`; `seed ⊂ ref`; all three are sorted ascending.
- MRI: subjects are partitioned, then every slice of a subject inherits its subject's split.
  `len(train_subjects) = 320`, `len(ref_subjects) = 80`, `len(seed_subjects) = 40`.
- Photographs: `train` 3200 images, `ref` 800, `seed` 40 images.
- The partition is drawn with `np.random.default_rng(2026)` over the sorted subject list, so it is
  reproducible from `index.csv` alone.

## 4. `meta.json`

```json
{"dataset_id": "ixi", "n_images": 4000, "image_size": 192, "dtype": "uint8",
 "pipeline": "ihdm.preprocess.mri", "pipeline_version": "1.0", "git_sha": "...",
 "created": "2026-09-2xT..:..:..", "raw_root": "/media/.../UNPROCESSED_MRI/HEALTHY/IXI/T1",
 "parameters": { ... every parameter of the pipeline, e.g. registration settings, z band, window rule, intensity rule ... },
 "counts": {"subjects_total": 581, "subjects_used": 400, "subjects_failed_registration": 3, "slices_per_subject": 10},
 "sha256_images": "<hex of images.npy bytes>"}
```

Photographs record the HF dataset name, shard, the row ids and the crop rule instead of the
registration block.

## 5. Image conventions

- Grayscale, `uint8`. The consumer converts with `x.astype(np.float32) / 255.0` → $[0, 1]$.
- MRI: axial slices in **radiological display convention as stored by the template grid**: after
  rigid registration into MNI152NLin2009cAsym (RAS), a slice is the template's $(x, y)$ plane at the
  chosen $z$; the array's first axis runs anterior→posterior (top of the image = anterior), the
  second axis left→right (image left = subject's left, i.e. neurological convention). Record the
  convention in `meta.json.parameters.orientation` and draw it on the QC sheet.
- MRI intensity: per volume, scale by the 99th percentile of the foreground (the foreground mask
  is the template brain mask dilated by 10 mm, applied only for computing the percentile), clip to
  $[0, 1]$, quantise `np.round(x * 255)`. Background noise is **kept** (no low clip).
- Photographs: RGB → luminance (`PIL.Image.convert("L")`), 256-px short side as fetched, centre
  crop 192 × 192, no resize.
- Padding: MRI columns outside the acquisition's field of view keep the value of the resampler's
  default (0). `meta.json.counts.padding_fraction` records the mean fraction of exactly-zero
  pixels.

## 6. Module contract (`ihdm/data/format.py`)

```python
@dataclass(frozen=True)
class DatasetMeta: ...                    # mirrors meta.json; to_json()/from_json()

def write_dataset(root: Path, images: np.ndarray, index: pd.DataFrame, splits: dict, meta: DatasetMeta) -> None
def read_dataset(root: Path, mmap: bool = True) -> tuple[np.ndarray, pd.DataFrame, dict, DatasetMeta]
def validate_dataset(root: Path) -> list[str]     # returns the list of violations; empty = valid
def split_by_subject(subjects: list[str], rng_seed: int = 2026, train_frac: float = 0.8, n_seed: int = 40) -> dict
```

`validate_dataset` checks, at least: file presence; `images.npy` dtype/shape/`N`; `index.csv`
columns, row count and `idx == row`; `splits.json` partition and subset properties and the
subject-level consistency; every subject's slices in one split; `meta.json` required keys; the
`sha256_images` hash; no image entirely constant; `mean(images) ∈ (5, 200)`; MRI `z_mm` finite and
inside the band, photographs `nan`.

## 7. Dataset backend (`ihdm/data/dataset.py` + hook in `scripts/datasets.py`)

`NpyImageDataset(root, split, random_flip=False)` returns `(tensor[1, 192, 192] float32 in [0,1], {})`
per item, exactly what the released trainer consumes (`batch = next(train_iter)[0]`). Hooked into
`scripts/datasets.get_dataset` under `config.data.dataset in {"ixi", "oasis1", "lsun_church_npy", "lsun_bedroom_npy"}`
with `config.data.root = str(IHDM_DATA_ROOT)` and `config.data.split_train = "train"`,
`config.data.split_eval = "ref"`. The train loader shuffles; the eval loader does not. `num_workers`
from `config.data.num_workers` (default 4). No horizontal flip for MRI (anatomical left/right is
meaningful); flips for photographs off by default too (one recipe everywhere).
