"""Tests of the intrinsic-dimension analysis (T8.0): estimator, distances, PCA measures, CLI."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ihdm.analysis.intrinsic_dim import (
    IDConfig,
    IntrinsicDimError,
    MissingInputError,
    components_for_fraction,
    distances_from_gram,
    linear_measures,
    mle_intrinsic_dim,
    participation_ratio,
    run,
)
from ihdm.cli.intrinsic_dim import EXIT_FAIL, EXIT_MISSING, main

DATASETS = ("ixi", "oasis1", "lsun_church", "lsun_bedroom")


def _subspace_points(d: int, n: int = 2000, ambient: int = 1000, noise: float = 1e-3,
                     seed: int = 0) -> np.ndarray:
    """Gaussian points on a random d-dimensional linear subspace of R^ambient, plus noise."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((ambient, d)))
    x = rng.standard_normal((n, d)) @ basis.T + noise * rng.standard_normal((n, ambient))
    return x - x.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------- 1. synthetic known dimension

@pytest.mark.parametrize("d", [5, 10])
@pytest.mark.parametrize("k", [5, 10, 20])
def test_mle_recovers_known_dimension(d: int, k: int) -> None:
    x = _subspace_points(d)
    estimate = mle_intrinsic_dim(distances_from_gram(x @ x.T), k)
    assert abs(estimate - d) <= 0.2 * d, f"d={d}, k={k}: estimate {estimate:.2f}"


def test_mle_separates_dimensions() -> None:
    lo = _subspace_points(5, n=1000)
    hi = _subspace_points(10, n=1000)
    assert (mle_intrinsic_dim(distances_from_gram(lo @ lo.T), 10)
            < mle_intrinsic_dim(distances_from_gram(hi @ hi.T), 10))


def test_mle_rejects_duplicates_and_bad_k() -> None:
    x = _subspace_points(3, n=50, ambient=20)
    x[1] = x[0]
    with pytest.raises(IntrinsicDimError, match="coincide"):
        mle_intrinsic_dim(distances_from_gram(x @ x.T), 5)
    y = _subspace_points(3, n=10, ambient=20)
    with pytest.raises(IntrinsicDimError):
        mle_intrinsic_dim(distances_from_gram(y @ y.T), 10)
    with pytest.raises(IntrinsicDimError):
        mle_intrinsic_dim(distances_from_gram(y @ y.T), 1)


def test_mle_matches_formula_by_hand() -> None:
    """The estimator equals the ticket's formula evaluated with an explicit loop."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((40, 6))
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    k = 7
    inv = []
    for i in range(len(x)):
        t = np.sort(np.delete(dist[i], i))[:k]
        inv.append(np.mean([np.log(t[k - 1] / t[j]) for j in range(k - 1)]))
    np.testing.assert_allclose(mle_intrinsic_dim(dist, k), 1.0 / np.mean(inv), rtol=1e-12)


# ---------------------------------------------------------------- 2. Gram-based distances

@pytest.mark.parametrize("shape", [(1, 5), (2, 3), (60, 200), (200, 30)])
def test_gram_distances_equal_direct(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, shape)
    direct = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    np.testing.assert_allclose(distances_from_gram(x @ x.T), direct, rtol=1e-9, atol=1e-9)


def test_gram_distances_independent_of_centring_and_subset() -> None:
    """A block of the full matrix equals the distances of the subset, centred on its own."""
    rng = np.random.default_rng(8)
    x = rng.uniform(0.0, 1.0, (120, 50))
    full = distances_from_gram((x - x.mean(0)) @ (x - x.mean(0)).T)
    idx = np.sort(rng.choice(120, 40, replace=False))
    sub = x[idx] - x[idx].mean(0)
    np.testing.assert_allclose(full[np.ix_(idx, idx)], distances_from_gram(sub @ sub.T),
                               rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------- 3. participation ratio

def test_participation_ratio_of_diagonal_covariance() -> None:
    """Data whose covariance is diag(lambda): the Gram route gives the closed form."""
    lam = np.array([9.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05])
    rng = np.random.default_rng(11)
    u, _ = np.linalg.qr(rng.standard_normal((50, lam.size)))  # orthonormal columns
    x = u * np.sqrt(lam)  # X^T X = diag(lam), so the nonzero spectrum of X X^T is lam
    meas = linear_measures(x @ x.T, (0.5, 0.9, 0.95))
    np.testing.assert_allclose(meas.participation_ratio, lam.sum() ** 2 / (lam**2).sum(),
                               rtol=1e-10)
    np.testing.assert_allclose(meas.total_variance, lam.sum(), rtol=1e-10)
    cum = np.cumsum(lam) / lam.sum()
    for f in (0.5, 0.9, 0.95):
        assert meas.components[f"{f:g}"] == int(np.argmax(cum >= f) + 1)
    assert meas.rank == lam.size


@pytest.mark.parametrize("m", [1, 3, 17])
def test_participation_ratio_of_equal_eigenvalues(m: int) -> None:
    np.testing.assert_allclose(participation_ratio(np.full(m, 2.5)), m, rtol=1e-12)
    assert components_for_fraction(np.full(m, 2.5), 1.0) == m


def test_participation_ratio_rejects_zero_spectrum() -> None:
    with pytest.raises(IntrinsicDimError):
        participation_ratio(np.zeros(4))


# ---------------------------------------------------------------- 4. CLI and the whole run

def _write_dataset(root: Path, name: str, n_subjects: int, rng: np.random.Generator,
                   size: int = 6) -> None:
    """A small dataset folder in the layout of the project (MRI: 10 slices per subject)."""
    folder = root / name
    folder.mkdir(parents=True)
    mri = name in ("ixi", "oasis1")
    slices = 10 if mri else 1
    n = n_subjects * slices
    images = rng.integers(0, 256, size=(n, size, size), dtype=np.uint8)
    np.save(folder / "images.npy", images)
    subjects = [f"{name}_{i // slices:04d}" for i in range(n)]
    with (folder / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "subject", "slice", "z_mm", "source", "split"])
        for i in range(n):
            writer.writerow([i, subjects[i], i % slices, "", f"src/{i}", "train"])
    (folder / "splits.json").write_text(json.dumps({
        "train": list(range(n)), "ref": [], "seed": [],
        "train_subjects": sorted(set(subjects)), "ref_subjects": [], "rng_seed": 2026}))
    (folder / "meta.json").write_text(json.dumps({"dataset_id": name, "n_images": n,
                                                  "sha256_images": f"sha-{name}"}))


@pytest.fixture
def small_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(2026)
    root = tmp_path / "data"
    for name in DATASETS:
        _write_dataset(root, name, 24 if name in ("ixi", "oasis1") else 240, rng)
    return root


SMALL = IDConfig(n_values=(40, 120, 240), n_subsets=3)


def test_cli_exits_2_on_missing_dataset_folder(small_root: Path, tmp_path: Path) -> None:
    for path in (small_root / "oasis1").iterdir():
        path.unlink()
    (small_root / "oasis1").rmdir()
    assert main(["--data-root", str(small_root), "--out", str(tmp_path / "out")]) == EXIT_MISSING
    assert not (tmp_path / "out").exists()


def test_cli_exits_2_on_missing_data_root(tmp_path: Path) -> None:
    assert main(["--data-root", str(tmp_path / "nowhere"),
                 "--out", str(tmp_path / "out")]) == EXIT_MISSING


def test_cli_exits_1_on_analysis_error(small_root: Path, tmp_path: Path) -> None:
    """Train indices beyond images.npy are an analysis error, raised before any Gram matrix."""
    splits = small_root / "ixi" / "splits.json"
    doc = json.loads(splits.read_text())
    doc["train"].append(10_000)
    splits.write_text(json.dumps(doc))
    assert main(["--data-root", str(small_root), "--out", str(tmp_path / "out")]) == EXIT_FAIL


def test_run_rejects_wrong_split_size(small_root: Path, tmp_path: Path) -> None:
    with pytest.raises(IntrinsicDimError, match="expects"):
        run(small_root, tmp_path / "out", IDConfig(n_values=(40, 120, 3200), n_subsets=3))


def test_run_missing_file_is_missing_input(small_root: Path, tmp_path: Path) -> None:
    (small_root / "lsun_bedroom" / "meta.json").unlink()
    with pytest.raises(MissingInputError):
        run(small_root, tmp_path / "out", SMALL)


def test_run_writes_every_number_and_a_byte_stable_figure(small_root: Path,
                                                          tmp_path: Path) -> None:
    first = run(small_root, tmp_path / "a", SMALL)
    second = run(small_root, tmp_path / "b", SMALL)
    for name in ("id_vs_n.svg", "id_vs_n.pdf", "id_vs_n.png", "intrinsic_dimension.json"):
        assert first["_files"][name] == second["_files"][name], name
    doc = json.loads((tmp_path / "a" / "intrinsic_dimension.json").read_text())
    assert doc["provenance"]["git_sha"]
    assert "2026" in doc["provenance"]["rng"]
    for name in DATASETS:
        entry = doc["datasets"][name]
        assert entry["sha256_images"] == f"sha-{name}"
        for k in (5, 10, 20):
            for n in (40, 120):
                cell = entry["mle"][f"k={k}"][f"N={n}"]
                assert cell["n_subsets"] == 3 and len(cell["values"]) == 3
                np.testing.assert_allclose(cell["mean"], np.mean(cell["values"]), rtol=1e-12)
                np.testing.assert_allclose(cell["sd"], np.std(cell["values"], ddof=1),
                                           rtol=1e-12)
            assert entry["mle"][f"k={k}"]["N=240"]["n_subsets"] == 1
        assert set(entry["linear"]["components_for_fraction"]) == {"0.5", "0.9", "0.95"}
        assert entry["linear"]["participation_ratio"] > 1.0
        one = entry.get("one_slice_per_subject")
        if name in ("ixi", "oasis1"):
            assert one["N"] == 24 and one["slice"] == 5 and set(one["mle"]) == {"k=5", "k=10",
                                                                                 "k=20"}
        else:
            assert one is None
    for statement in ("mle_lower_for_mri_at_full_n", "mri_changes_less_with_n",
                      "participation_ratio_higher_for_mri"):
        assert isinstance(doc["reading"][statement]["holds"], bool)
    svg = (tmp_path / "a" / "id_vs_n.svg").read_text()
    assert "<text" in svg  # svg.fonttype = none keeps the labels as text
    digest = hashlib.sha256((tmp_path / "a" / "id_vs_n.svg").read_bytes()).hexdigest()
    assert digest == first["_files"]["id_vs_n.svg"]
