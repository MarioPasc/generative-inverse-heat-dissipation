"""Fast tests of ``ihdm.cli.build_tutorial_assets`` on the synthetic fixture (no data drive).

Invariants: an exported PNG reads back equal to the stored uint8 image; two builds from the same
sources are byte-identical; the build refuses octave errors whose seed means disagree with the
printed table of ``F2.md``. The held-out JSONs and ``F2.md`` are tracked files; the per-run
``final.json`` files live on the data drive, so the tests write stand-ins whose seed means equal
the printed ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ihdm.cli import build_tutorial_assets as cli
from ihdm.data.format import read_dataset
from ihdm.paths import repo_root


def _write_fake_results(root: Path, shift: float = 0.0) -> Path:
    """Write ``runs/<run>/final.json`` for the IXI runs, each equal to its F2.md seed mean."""
    means = cli.parse_f2_lsd_means((repo_root() / "docs/RESULTS/paper/F2.md").read_text())
    for i, run in enumerate(r for r in cli.RUNS if r.startswith("ixi_")):
        label = cli.CONFIGURATIONS[run.split("_")[1]]
        octaves = dict(means[label])
        if i == 0:
            octaves["1-2"] += shift
        run_dir = root / "runs" / run
        run_dir.mkdir(parents=True)
        (run_dir / "final.json").write_text(json.dumps({"lsd_octaves": octaves}))
    return root


def _build(data_root: Path, results: Path, out: Path) -> int:
    return cli.main(
        [
            "--data-root", str(data_root),
            "--results", str(results),
            "--out", str(out),
            "--images", "synthetic:3", "synthetic:10",
            "--log-level", "WARNING",
        ]
    )


def test_build_is_exact_and_byte_stable(synthetic_dataset: Path, tmp_path: Path) -> None:
    results = _write_fake_results(tmp_path / "results")
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert _build(synthetic_dataset.parent, results, out_a) == 0
    assert _build(synthetic_dataset.parent, results, out_b) == 0

    images, index, _, _ = read_dataset(synthetic_dataset)
    manifest = json.loads((out_a / "images" / cli.MANIFEST_NAME).read_text())
    for entry in manifest["images"]:
        stored = np.asarray(images[entry["row"]])
        exported = np.asarray(Image.open(out_a / "images" / entry["file"]))
        np.testing.assert_array_equal(exported, stored)
        assert entry["split"] == index.loc[index["idx"] == entry["row"], "split"].item()

    files = sorted(p.relative_to(out_a) for p in out_a.rglob("*") if p.is_file())
    assert len(files) == 4  # two PNGs, the manifest and the results JSON
    for rel in files:
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), rel

    payload = json.loads((out_a / "results" / cli.RESULTS_NAME).read_text())
    run = payload["runs"]["ixi_A0_s1"]
    # Hand-checked against F2.md: ixi_A0_s1 F precision 0.6015, recall 0.3025, F seed 1.329.
    assert run["precision_recall"]["F"] == {"precision": 0.6015, "recall": 0.3025}
    assert run["distance_median"]["F_seed"] == pytest.approx(1.329)
    assert payload["distance_band_rminus"]["ixi"] == [0.936, 1.078]
    # F2.md prints seed means from unrounded medians: 0.893, not the 0.892 of the rounded runs.
    assert payload["distance_seed_mean"]["ixi"]["matched"] == {
        "F_seed": 0.893, "F_train": 0.94, "H_seed": 0.867, "H_train": 0.927, "n": 3
    }
    assert len(payload["runs"]) == len(cli.RUNS)


def test_build_rejects_octave_errors_that_disagree_with_f2(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    # One run of A0 off by 0.01 moves its 3-seed mean by 0.0033, beyond the printed rounding.
    results = _write_fake_results(tmp_path / "results", shift=0.01)
    assert _build(synthetic_dataset.parent, results, tmp_path / "out") == 1
    assert not (tmp_path / "out" / "results" / cli.RESULTS_NAME).exists()


def test_failed_build_leaves_the_output_folder_unchanged(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert _build(synthetic_dataset.parent, _write_fake_results(tmp_path / "good"), out) == 0
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    bad = _write_fake_results(tmp_path / "bad", shift=0.01)
    # A different image and --force: a build that wrote PNGs before checking would add one.
    status = cli.main(
        [
            "--data-root", str(synthetic_dataset.parent), "--results", str(bad),
            "--out", str(out), "--images", "synthetic:5", "--force", "--log-level", "CRITICAL",
        ]
    )
    assert status == 1
    after = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert after == before


def test_unknown_dataset_label_raises_the_package_error() -> None:
    with pytest.raises(cli.TutorialAssetError):
        cli._dataset_id("BraTS")


def test_image_spec_rejects_malformed_text() -> None:
    with pytest.raises(cli.TutorialAssetError):
        cli.ImageSpec.from_text("ixi-3705")
