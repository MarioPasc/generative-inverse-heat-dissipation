"""Rebuild the small files of ``assets/`` that the tutorial notebook reads.

Writes two kinds of file (ticket T9.1 §5):

- ``assets/images/<dataset>_<row>.png``: dataset images exactly as stored (uint8, no
  resampling), plus ``assets/images/manifest.json`` with the row, subject, slice and split of
  each one;
- ``assets/results/f2_results.json``: the numbers Figure 2 of the report plots, per run.
  Precision and recall come from the tracked held-out-fidelity JSONs; the median distances to the
  seed and to the training set, the R⁻ band and the pooled interquartile ranges are parsed from
  the generated tables of ``docs/RESULTS/paper/F2.md`` (their cache was scratch and no
  machine-readable copy exists); the octave errors come from ``lsd_octaves`` of each run's
  ``final.json`` under the results root, and their seed means are checked against ``F2.md``.

The outputs carry no timestamp and no commit hash, so two builds from the same sources are
byte-identical. Run as ``python -m ihdm.cli.build_tutorial_assets``; this is the only part of the
tutorial that reads the data drive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ihdm.data.format import read_dataset
from ihdm.paths import data_root, repo_root
from ihdm.spectral.power import OCTAVE_LABELS

logger = logging.getLogger(__name__)

# The four images of the notebook (T9.1 §5): IXI 3705 is the slice of the report's Figure 1
# (split seed), IXI 3311 the "train 1" row of Figure 2a; Churches 1180 (train) is the image of
# the A1 figure and Churches 107 a held-out seed image.
DEFAULT_IMAGES: tuple[str, ...] = ("ixi:3705", "ixi:3311", "lsun_church:1180", "lsun_church:107")

# The 14 brain-MRI runs that Figure 2 plots, by dataset and configuration.
CONFIGURATIONS: dict[str, str] = {
    "A0": "default",
    "A1": "+prior",
    "A2": "+spacing",
    "A3": "matched",
}
RUNS: tuple[str, ...] = (
    "ixi_A0_s1", "ixi_A0_s2", "ixi_A0_s3",
    "ixi_A1_s1", "ixi_A1_s2",
    "ixi_A2_s1", "ixi_A2_s2",
    "ixi_A3_s1", "ixi_A3_s2", "ixi_A3_s3",
    "oasis1_A0_s1", "oasis1_A0_s2",
    "oasis1_A3_s1", "oasis1_A3_s2",
)
# Figure 2c shows the octave errors of the IXI runs only.
LSD_DATASET = "ixi"
DATASET_NAMES: dict[str, str] = {"IXI": "ixi", "OASIS-1": "oasis1"}
DISTANCE_KEYS: tuple[str, ...] = ("F seed", "F train", "H seed", "H train")
# F2.md prints the seed means of the octave errors to three decimals.
LSD_TOLERANCE = 0.0005 + 1e-9

RESULTS_NAME = "f2_results.json"
MANIFEST_NAME = "manifest.json"

_RUN_ID = re.compile(r"^(?P<dataset>[a-z0-9]+)_(?P<arm>A\d)_s(?P<seed>\d+)$")


class TutorialAssetError(Exception):
    """A source of the tutorial assets is missing or disagrees with another source."""


@dataclass(frozen=True)
class ImageSpec:
    """One dataset image to export.

    Parameters
    ----------
    dataset_id : str
        Directory name of the standard-format dataset under the data root.
    row : int
        Row of ``images.npy`` (the ``idx`` column of ``index.csv``).
    """

    dataset_id: str
    row: int

    @classmethod
    def from_text(cls, text: str) -> ImageSpec:
        """Parse ``"<dataset_id>:<row>"``.

        Parameters
        ----------
        text : str
            For example ``"ixi:3705"``.

        Returns
        -------
        ImageSpec
            The parsed specification.

        Raises
        ------
        TutorialAssetError
            If ``text`` is not of the form ``<dataset_id>:<row>``.
        """
        dataset_id, sep, row = text.partition(":")
        if not sep or not dataset_id or not row.isdigit():
            raise TutorialAssetError(f"image spec {text!r} is not '<dataset_id>:<row>'")
        return cls(dataset_id=dataset_id, row=int(row))

    @property
    def file_name(self) -> str:
        """Name of the PNG written for this image."""
        return f"{self.dataset_id}_{self.row}.png"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        64 hex characters.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    """Repository-relative path when ``path`` is inside the repository, else the path as given."""
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as deterministic JSON (sorted keys, one-space indent, final newline).

    Parameters
    ----------
    path : Path
        Output file; its parent must exist.
    payload : dict[str, Any]
        JSON-serialisable content.
    """
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def export_images(root: Path, specs: Sequence[ImageSpec], out_dir: Path) -> dict[str, Any]:
    """Save each image as an 8-bit grayscale PNG, exactly as stored, and return the manifest.

    Parameters
    ----------
    root : Path
        Data root holding one standard-format dataset per ``dataset_id``.
    specs : Sequence[ImageSpec]
        Images to export.
    out_dir : Path
        Directory for the PNGs; created if missing.

    Returns
    -------
    dict[str, Any]
        Manifest: one entry per PNG with dataset, row, subject, slice, split, source and the
        SHA-256 of the PNG.

    Raises
    ------
    TutorialAssetError
        If a row is out of range, an image is not uint8, or a PNG does not read back identical.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    loaded: dict[str, tuple[np.ndarray, Any]] = {}
    for spec in specs:
        if spec.dataset_id not in loaded:
            images, index, _, _ = read_dataset(root / spec.dataset_id)
            loaded[spec.dataset_id] = (images, index)
        images, index = loaded[spec.dataset_id]
        if not 0 <= spec.row < images.shape[0]:
            raise TutorialAssetError(
                f"row {spec.row} outside {spec.dataset_id} (n = {images.shape[0]})"
            )
        image = np.asarray(images[spec.row])  # (H, W) uint8, copied out of the memmap
        if image.dtype != np.uint8:
            raise TutorialAssetError(f"{spec.dataset_id} images are {image.dtype}, not uint8")
        path = out_dir / spec.file_name
        Image.fromarray(image).save(path, format="PNG")  # 2-D uint8 -> mode "L"
        if not np.array_equal(np.asarray(Image.open(path)), image):
            raise TutorialAssetError(f"{path} does not read back identical to the stored image")
        row = index.loc[index["idx"] == spec.row].iloc[0]
        entries.append(
            {
                "file": spec.file_name,
                "dataset_id": spec.dataset_id,
                "row": spec.row,
                "subject": str(row["subject"]),
                "slice": int(row["slice"]),
                "split": str(row["split"]),
                "source": str(row["source"]),
                "shape": list(image.shape),
                "dtype": "uint8",
                "sha256": sha256_file(path),
            }
        )
        logger.info("wrote %s (%s, split %s)", path, row["subject"], row["split"])
    return {"images": entries, "rebuild": "python -m ihdm.cli.build_tutorial_assets"}


def _parse_run(run_id: str) -> tuple[str, str, int]:
    """Split ``"<dataset>_<arm>_s<seed>"`` into its parts."""
    match = _RUN_ID.match(run_id)
    if match is None:
        raise TutorialAssetError(f"run id {run_id!r} is not '<dataset>_<arm>_s<seed>'")
    return match["dataset"], match["arm"], int(match["seed"])


def load_precision_recall(
    heldout_paths: Sequence[Path], runs: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Precision and recall against R⁻ of sets F and H, per run.

    Parameters
    ----------
    heldout_paths : Sequence[Path]
        Held-out-fidelity JSONs with ``runs.<id>.metrics.{F_rminus,H_rminus}``.
    runs : Sequence[str]
        Run ids to extract.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        ``{run: {"F": {"precision", "recall"}, "H": {...}}}``.

    Raises
    ------
    TutorialAssetError
        If a run is missing from every file, or two files disagree on a run.
    """
    found: dict[str, dict[str, dict[str, float]]] = {}
    for path in heldout_paths:
        stored = json.loads(path.read_text())["runs"]
        for run in runs:
            if run not in stored:
                continue
            metrics = stored[run]["metrics"]
            values = {
                set_name: {
                    key: float(metrics[f"{set_name}_rminus"][key])
                    for key in ("precision", "recall")
                }
                for set_name in ("F", "H")
            }
            if run in found and found[run] != values:
                raise TutorialAssetError(f"{run}: {path} disagrees with an earlier held-out file")
            found[run] = values
    missing = [run for run in runs if run not in found]
    if missing:
        raise TutorialAssetError(f"runs missing from the held-out files: {missing}")
    return found


def _markdown_tables(text: str) -> list[list[list[str]]]:
    """Every pipe table of a Markdown text, as rows of stripped cells (separator rows dropped)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(set(cell) <= set("-: ") for cell in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _find_table(tables: list[list[list[str]]], *required: str) -> list[list[str]]:
    """The first table whose header holds every ``required`` column name."""
    for table in tables:
        if all(name in table[0] for name in required):
            return table
    raise TutorialAssetError(f"F2.md has no table with the columns {required}")


def _range(text: str) -> list[float]:
    """Parse ``"0.936–1.078"`` (en dash) into ``[0.936, 1.078]``."""
    low, sep, high = text.partition("–")
    if not sep:
        raise TutorialAssetError(f"{text!r} is not a range 'a–b'")
    return [float(low), float(high)]


def _strip(cell: str) -> str:
    """Drop Markdown bold and code marks from a table cell."""
    return cell.replace("**", "").replace("`", "").strip()


def parse_f2_distances(f2_text: str) -> dict[str, Any]:
    """The distance numbers of Figure 2b (bottom) from the generated tables of ``F2.md``.

    Parameters
    ----------
    f2_text : str
        Content of ``docs/RESULTS/paper/F2.md``.

    Returns
    -------
    dict[str, Any]
        ``per_run``: ``{run: {"F_seed", "F_train", "H_seed", "H_train"}}`` (medians in units of
        D_ref); ``band``: ``{dataset: [p25, p75]}`` of the R⁻ images; ``pooled_iqr``:
        ``{dataset: {configuration: {key: [p25, p75]}}}``.

    Raises
    ------
    TutorialAssetError
        If a table or a column is missing.
    """
    tables = _markdown_tables(f2_text)
    per_run_table = _find_table(tables, "run", *DISTANCE_KEYS)
    header = per_run_table[0]
    per_run: dict[str, dict[str, float]] = {}
    for cells in per_run_table[1:]:
        run = _strip(cells[header.index("run")])
        if run:
            per_run[run] = {
                key.replace(" ", "_"): float(cells[header.index(key)]) for key in DISTANCE_KEYS
            }

    band_table = _find_table(tables, "dataset", "R⁻ band (25th–75th percentile)")
    band_header = band_table[0]
    band = {
        DATASET_NAMES[cells[0]]: _range(cells[band_header.index("R⁻ band (25th–75th percentile)")])
        for cells in band_table[1:]
    }

    iqr_tables = [
        table
        for table in tables
        if table[0][:2] == ["dataset", "configuration"]
        and all(key in table[0] for key in DISTANCE_KEYS)
        and "run" not in table[0]
    ]
    if len(iqr_tables) != 1:
        raise TutorialAssetError(f"expected one pooled-IQR table in F2.md, found {len(iqr_tables)}")
    iqr_header = iqr_tables[0][0]
    pooled: dict[str, dict[str, dict[str, list[float]]]] = {}
    for cells in iqr_tables[0][1:]:
        dataset = DATASET_NAMES[cells[0]]
        pooled.setdefault(dataset, {})[cells[1]] = {
            key.replace(" ", "_"): _range(cells[iqr_header.index(key)]) for key in DISTANCE_KEYS
        }
    return {"per_run": per_run, "band": band, "pooled_iqr": pooled}


def parse_f2_lsd_means(f2_text: str) -> dict[str, dict[str, float]]:
    """Seed means of the octave errors per configuration, as printed in ``F2.md`` (panel c).

    Parameters
    ----------
    f2_text : str
        Content of ``docs/RESULTS/paper/F2.md``.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{configuration label: {octave label: mean}}``.

    Raises
    ------
    TutorialAssetError
        If the table is missing.
    """
    table = _find_table(_markdown_tables(f2_text), "configuration", "runs", *OCTAVE_LABELS)
    header = table[0]
    return {
        _strip(cells[0]): {label: float(cells[header.index(label)]) for label in OCTAVE_LABELS}
        for cells in table[1:]
    }


def load_lsd_octaves(results_root: Path, runs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``lsd_octaves`` of each run's ``final.json``, with the file's SHA-256.

    Parameters
    ----------
    results_root : Path
        Directory holding ``runs/<run>/final.json``.
    runs : Sequence[str]
        Run ids.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{run: {"lsd_octaves": {octave label: e_o}, "sha256": str}}``.

    Raises
    ------
    TutorialAssetError
        If a file is missing or lacks an octave.
    """
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        path = results_root / "runs" / run / "final.json"
        if not path.is_file():
            raise TutorialAssetError(f"missing {path}")
        octaves = json.loads(path.read_text())["lsd_octaves"]
        if set(octaves) != set(OCTAVE_LABELS):
            raise TutorialAssetError(f"{path}: octaves {sorted(octaves)} != {OCTAVE_LABELS}")
        out[run] = {
            "lsd_octaves": {label: float(octaves[label]) for label in OCTAVE_LABELS},
            "sha256": sha256_file(path),
        }
    return out


def check_lsd_seed_means(
    lsd: dict[str, dict[str, Any]], expected: dict[str, dict[str, float]]
) -> list[str]:
    """Compare the seed means of the octave errors with the table of ``F2.md``.

    Parameters
    ----------
    lsd : dict[str, dict[str, Any]]
        Output of :func:`load_lsd_octaves`.
    expected : dict[str, dict[str, float]]
        Output of :func:`parse_f2_lsd_means`.

    Returns
    -------
    list[str]
        One message per octave whose mean differs by more than the printed rounding; empty if
        all agree.
    """
    problems: list[str] = []
    for arm, label in CONFIGURATIONS.items():
        members = [run for run in lsd if _parse_run(run)[1] == arm]
        if not members:
            continue
        if label not in expected:
            problems.append(f"F2.md has no octave-error row for {label}")
            continue
        for octave in OCTAVE_LABELS:
            mean = float(np.mean([lsd[run]["lsd_octaves"][octave] for run in members]))
            printed = expected[label][octave]
            if abs(mean - printed) > LSD_TOLERANCE:
                problems.append(f"{label} {octave}: seed mean {mean:+.4f} != F2.md {printed:+.3f}")
    return problems


def build_results(
    *,
    heldout_paths: Sequence[Path],
    f2_md: Path,
    results_root: Path,
    runs: Sequence[str] = RUNS,
) -> dict[str, Any]:
    """Assemble the content of ``f2_results.json``.

    Parameters
    ----------
    heldout_paths : Sequence[Path]
        Held-out-fidelity JSONs (precision and recall).
    f2_md : Path
        ``docs/RESULTS/paper/F2.md`` (distances, band, pooled IQRs, octave-error means).
    results_root : Path
        Directory holding ``runs/<run>/final.json`` (octave errors).
    runs : Sequence[str]
        Run ids to include.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable content, with the source of every field.

    Raises
    ------
    TutorialAssetError
        If a source is missing, or the octave-error seed means disagree with ``F2.md``.
    """
    f2_text = f2_md.read_text()
    precision_recall = load_precision_recall(heldout_paths, runs)
    distances = parse_f2_distances(f2_text)
    lsd_runs = [run for run in runs if _parse_run(run)[0] == LSD_DATASET]
    lsd = load_lsd_octaves(results_root, lsd_runs)
    problems = check_lsd_seed_means(lsd, parse_f2_lsd_means(f2_text))
    if problems:
        raise TutorialAssetError("octave errors disagree with F2.md: " + "; ".join(problems))

    per_run: dict[str, Any] = {}
    for run in runs:
        dataset, arm, seed = _parse_run(run)
        if run not in distances["per_run"]:
            raise TutorialAssetError(f"{run} missing from the distance table of F2.md")
        entry: dict[str, Any] = {
            "dataset": dataset,
            "arm": arm,
            "configuration": CONFIGURATIONS[arm],
            "seed": seed,
            "precision_recall": precision_recall[run],
            "distance_median": distances["per_run"][run],
        }
        if run in lsd:
            entry["lsd_octaves"] = lsd[run]["lsd_octaves"]
        per_run[run] = entry

    return {
        "description": (
            "Numbers plotted by Figure 2 of the report (docs/RESULTS/paper/F2.md), per run; "
            "built by python -m ihdm.cli.build_tutorial_assets."
        ),
        "configurations": CONFIGURATIONS,
        "octaves": list(OCTAVE_LABELS),
        "runs": per_run,
        "distance_band_rminus": distances["band"],
        "distance_pooled_iqr": distances["pooled_iqr"],
        "units": {
            "precision_recall": "Inception-v3, k = 5, against R- (400 images)",
            "distance_median": "median over the samples of a run, in units of D_ref",
            "lsd_octaves": "e_o = log10 P_gen(o) - log10 P_real(o), 2,000 samples of set F vs R",
        },
        "sources": {
            "precision_recall": [
                {"path": _display_path(path), "sha256": sha256_file(path)}
                for path in heldout_paths
            ],
            "distance_median, distance_band_rminus, distance_pooled_iqr": {
                "path": _display_path(f2_md),
                "sha256": sha256_file(f2_md),
                "note": "parsed from the generated tables; no machine-readable source exists",
            },
            "lsd_octaves": {
                f"<results>/runs/{run}/final.json": lsd[run]["sha256"] for run in lsd_runs
            },
        },
    }


def _outputs_exist(out_dir: Path) -> bool:
    """True when both JSON outputs are already in ``out_dir``."""
    return (out_dir / "images" / MANIFEST_NAME).is_file() and (
        out_dir / "results" / RESULTS_NAME
    ).is_file()


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        ``0`` on success, ``1`` if a source is missing or inconsistent.
    """
    repo = repo_root()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument(
        "--results", type=Path, default=None,
        help="results root with runs/<run>/final.json (default: <data-root>/_results)",
    )
    parser.add_argument("--out", type=Path, default=repo / "assets")
    parser.add_argument(
        "--images", nargs="+", default=list(DEFAULT_IMAGES), metavar="DATASET:ROW",
        help="images to export",
    )
    parser.add_argument(
        "--heldout", type=Path, nargs="+",
        default=[
            repo / "docs/RESULTS/heldout_fidelity/heldout_fidelity.json",
            repo / "docs/RESULTS/paper/f2_a2_heldout/heldout_fidelity.json",
        ],
    )
    parser.add_argument("--f2-md", type=Path, default=repo / "docs/RESULTS/paper/F2.md")
    parser.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    out_dir: Path = args.out
    if not args.force and _outputs_exist(out_dir):
        logger.info("tutorial assets already in %s; --force to rebuild", out_dir)
        return 0
    results_root: Path = args.results if args.results is not None else args.data_root / "_results"
    try:
        specs = [ImageSpec.from_text(text) for text in args.images]
        manifest = export_images(args.data_root, specs, out_dir / "images")
        results = build_results(
            heldout_paths=args.heldout, f2_md=args.f2_md, results_root=results_root
        )
    except TutorialAssetError as exc:
        logger.error("FAILED: %s", exc)
        return 1
    write_json(out_dir / "images" / MANIFEST_NAME, manifest)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results" / RESULTS_NAME, results)
    logger.info(
        "wrote %d images and %s (%d runs) to %s",
        len(specs), RESULTS_NAME, len(results["runs"]), out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
