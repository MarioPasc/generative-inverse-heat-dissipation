"""Tests of F2 and T1 (T8.2, T8.8): CRN and selection rules, expectations, band-pass images,
the seed-distance anchor, T1 byte identity, exit codes and byte stability."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ihdm.analysis import paper_f2 as pf
from ihdm.analysis.heldout_fidelity import index_sha256 as ref_sha256
from ihdm.analysis.heldout_fidelity import select_references
from ihdm.analysis.paper_f2 import (
    ARM_LABELS,
    PAPER_WIDTH_IN,
    CRNError,
    ExpectationError,
    Expectations,
    IndexRun,
    PaperF2Error,
    SampleSet,
    SelectionError,
    assert_crn,
    assert_precision_means,
    assert_selection,
    assert_spectrum_means,
    band_decomposition,
    foreground_fraction,
    head_crop,
    kept_variance,
    merge_heldout,
    prior_state,
    select_rows,
)
from ihdm.cli import paper_f2 as cli
from ihdm.data.format import DatasetMeta, write_dataset
from ihdm.metrics.memorisation import memorisation_ratio

REPO = Path(__file__).resolve().parents[2]
T75_JSON = REPO / "docs" / "RESULTS" / "heldout_fidelity" / "heldout_fidelity.json"
A2_JSON = REPO / "docs" / "RESULTS" / "paper" / "f2_a2_heldout" / "heldout_fidelity.json"
PAPER = REPO / "docs" / "RESULTS" / "paper"
#: The committed T1 (T8.2); T8.8 regenerates it with the same CLI and must not change a byte.
T1_SHA256 = {
    "t1_main.tex": "7e4dc907363c2e27f3769628ef6cad77d6f3c98112ebdd048e7f2186786a9f63",
    "t1_main.md": "1791695eae6a10c0f5371a3dd5d71fa53a9ddf7dbdfaea5fde0f8a6b904066b2",
}
DATA_ROOT = Path("/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project")
EVAL_DIR = Path("/media/mpascual/Sandisk2TB/research/spectral_allocation_heat_diffusion_project/"
                "evaluation/eval_2488269")

SIZE = 16
#: Synthetic MRI-like datasets: subject = image // 10, slice = image % 10. The seed subjects
#: 0, 3, 19 put their slice-5 images at 5, 35, 195 (the real held-out seeds of panel (a)); the
#: extra ref subject 370 puts its slice 5 at 3705, which reference_example(rng 2026) picks
#: among the four slice-5 ref images, as on the real data.
N_SLICES = 10
N_IMAGES = 3710
SEED_SUBJECTS = (0, 3, 19)
REF_SUBJECTS = (*SEED_SUBJECTS, 370)
FINAL_IDX = np.array([3311, 2488, 1995, 1073, 1205, 161], dtype=np.int64)
HELDOUT_IDX = np.array([5, 35, 195], dtype=np.int64)
PER_SEED = 4
IXI_CELLS = {"A0": 3, "A2": 2, "A1": 2, "A3": 3}
OASIS_CELLS = {"A0": 2, "A3": 2}
#: Per-run precision and recall of the synthetic tree: the committed held-out JSONs' (T7.5 and
#: the A2 extension), so that panel (b) is laid out as on the real data and T8.5's expectation
#: table (asserted by the build) holds.
REAL_METRICS = merge_heldout([T75_JSON, A2_JSON])[0]
PROFILES = {
    "A0": [-0.02, -0.30, -0.39, -0.18, -0.13, -0.16, -0.13, -0.18],
    "A2": [-0.01, -0.24, -0.35, -0.23, -0.14, -0.15, -0.09, -0.16],
    "A1": [-0.01, 0.04, -0.05, -0.17, -0.11, -0.06, -0.02, -0.09],
    "A3": [-0.01, 0.04, -0.04, -0.12, -0.07, -0.05, -0.01, -0.10],
}


# --------------------------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------------------------


def _images() -> np.ndarray:
    """Disc phantoms with texture: a bright disc (foreground well above 10%) plus noise, so that
    every image is distinct and no two images are at distance 0."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    disc = ((yy - SIZE / 2 + 0.5) ** 2 + (xx - SIZE / 2 + 0.5) ** 2) <= (SIZE / 3) ** 2
    noise = np.random.default_rng(11).integers(0, 50, (N_IMAGES, SIZE, SIZE))
    return np.clip(disc[None] * 180 + noise, 0, 255).astype(np.uint8)


def _write_dataset(root: Path, name: str, images: np.ndarray) -> None:
    """A standard-format dataset: train = every subject outside ``REF_SUBJECTS``."""
    idx = np.arange(N_IMAGES)
    subjects = [f"S{i // N_SLICES:03d}" for i in idx]
    slices = (idx % N_SLICES).tolist()
    in_ref = np.isin(idx // N_SLICES, REF_SUBJECTS)
    in_seed = np.isin(idx // N_SLICES, SEED_SUBJECTS)
    index = pd.DataFrame({"idx": idx, "subject": subjects, "slice": slices,
                          "z_mm": [float(s) for s in slices],
                          "source": [f"synthetic/{s}/{k}" for s, k in zip(subjects, slices,
                                                                          strict=True)],
                          "split": np.where(in_seed, "seed", np.where(in_ref, "ref", "train"))})
    splits = {"train": idx[~in_ref].tolist(), "ref": idx[in_ref].tolist(),
              "seed": idx[in_seed].tolist(),
              "train_subjects": sorted({subjects[i] for i in idx[~in_ref]}),
              "ref_subjects": sorted({subjects[i] for i in idx[in_ref]}),
              "seed_subjects": sorted({subjects[i] for i in idx[in_seed]}), "rng_seed": 2026}
    meta = DatasetMeta(dataset_id=name, n_images=N_IMAGES, image_size=SIZE, dtype="uint8",
                       pipeline="tests.analysis.test_paper_f2", pipeline_version="1.0",
                       git_sha="test", created="2026-10-07T00:00:00", raw_root=str(root),
                       parameters={}, counts={})
    write_dataset(root / name, images, index, splits, meta)


def _sample_set(name: str, images: np.ndarray, seed_idx: np.ndarray, per_seed: int, rng_seed: int,
                arm_seed: int, batch: int = 32) -> SampleSet:
    rng = np.random.default_rng(arm_seed)
    samples = rng.integers(0, 256, (seed_idx.size * per_seed, SIZE, SIZE), dtype=np.uint8)
    return SampleSet(name, seed_idx.copy(), images[seed_idx].copy(), samples,
                     {"rng_seed": rng_seed, "batch_size": batch})


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _sets(images: np.ndarray, arm_seed: int,
          final_idx: np.ndarray = FINAL_IDX) -> dict[str, SampleSet]:
    """The final set (6 training seeds x 1) and the held-out set (3 unseen seeds x 4)."""
    final = _sample_set("final", images, final_idx, 1, 0, arm_seed)
    # Half of the training-seeded samples are near-copies of their seed (seed-NN > 0).
    copies = np.clip(images[final_idx[::2]].astype(int) + 3, 0, 255).astype(np.uint8)
    samples = final.samples.copy()
    samples[::2] = copies
    return {"final": SampleSet("final", final.seed_idx, final.seeds, samples, final.signature),
            "heldout": _sample_set("heldout", images, HELDOUT_IDX, PER_SEED, 2026,
                                   arm_seed + 100)}


def _write_tar(path: Path, run_id: str, sets: dict[str, SampleSet]) -> None:
    base = f"{run_id}/samples_amp-fp16/060000"
    with tarfile.open(path, "w") as tar:
        for name, s in sets.items():
            members = {
                "samples.npy": _npy_bytes(s.samples),
                "seed_idx.npy": _npy_bytes(s.seed_idx),
                "seeds.npy": _npy_bytes(s.seeds),
                "request.json": json.dumps({"signature": s.signature}).encode(),
            }
            for member, data in members.items():
                info = tarfile.TarInfo(f"{base}/{name}/{member}")
                info.size = len(data)
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))


def _runs() -> list[tuple[str, str, str, int]]:
    out = []
    for dataset, cells in (("ixi", IXI_CELLS), ("oasis1", OASIS_CELLS)):
        for arm, n in cells.items():
            out += [(f"{dataset}_{arm}_s{s}", dataset, arm, s) for s in range(1, n + 1)]
    return out


def _memorisation(images: np.ndarray, final: SampleSet) -> tuple[float, float, float]:
    """M, seed-NN and M's denominator of a synthetic run, from the production metric (the
    independent reference the build's recomputation is anchored to)."""
    index = pd.DataFrame({"idx": np.arange(N_IMAGES), "slice": np.arange(N_IMAGES) % N_SLICES,
                          "subject": [f"S{i // N_SLICES:03d}" for i in range(N_IMAGES)]})
    ref = np.flatnonzero(np.isin(np.arange(N_IMAGES) // N_SLICES, REF_SUBJECTS))
    train = np.setdiff1d(np.arange(N_IMAGES), ref)
    heldout = np.setdiff1d(ref, np.flatnonzero(np.isin(np.arange(N_IMAGES) // N_SLICES,
                                                       SEED_SUBJECTS)))
    result = memorisation_ratio(final.samples, images[train], images[heldout],
                                index["subject"].to_numpy()[train].tolist(),
                                np.searchsorted(train, final.seed_idx), device="cpu")
    return result.M, result.seed_nn_fraction, result.d_heldout_median


def _references(data_root: Path, dataset: str) -> dict[str, object]:
    frame = pd.read_csv(data_root / dataset / "index.csv")
    splits = json.loads((data_root / dataset / "splits.json").read_text())
    refs = select_references(frame, splits, dataset)
    return {"n_ref": int(refs.ref_idx.size), "n_r_minus": int(refs.r_minus_idx.size),
            "ref_idx_sha256": ref_sha256(refs.ref_idx),
            "r_minus_idx_sha256": ref_sha256(refs.r_minus_idx)}


def build_tree(root: Path, broken_crn: bool = False) -> dict[str, Path]:
    """Write a complete synthetic input tree; return the CLI paths."""
    images = _images()
    data_root = root / "data"
    for dataset in ("ixi", "oasis1"):
        _write_dataset(data_root, dataset, images)
    eval_dir = root / "eval"
    eval_dir.mkdir()
    results = root / "results"
    (results / "runs").mkdir(parents=True)
    rows, t1a, t1b, held = [], [], [], {}
    for k, (rid, dataset, arm, seed) in enumerate(_runs()):
        final_idx = FINAL_IDX.copy()
        if broken_crn and rid == "ixi_A3_s1":
            final_idx[2], final_idx[3] = final_idx[3], final_idx[2]
        sets = _sets(images, k, final_idx)
        _write_tar(eval_dir / f"{rid}_amp-fp16.tar", rid, sets)
        m_ratio, seed_nn, d_heldout = _memorisation(images, sets["final"])
        metrics = REAL_METRICS[rid]
        row = {"index": len(rows), "run_id": rid, "dataset": dataset, "arm": arm, "seed": seed,
               "kid": 0.01 * (seed + 1), "lsd_final": 0.1 * seed,
               "precision": metrics["F_rminus"]["precision"] - 0.02, "recall": 0.3,
               "seed_nn_fraction": seed_nn, "D_pix": 0.011, "M": m_ratio}
        rows.append(row)
        t1a.append({"run_id": rid, "kid": row["kid"], "lsd_final": row["lsd_final"],
                    "recall": row["recall"]})
        t1b.append({"run_id": rid, "seed_nn_fraction": seed_nn, "D_pix": row["D_pix"]})
        held[rid] = {"metrics": {key: metrics[key] for key in ("F_rminus", "H_rminus", "H_r5")}}
        (results / "runs" / rid).mkdir()
        record: dict[str, object] = {"d_heldout_median": d_heldout}
        if dataset == "ixi":
            record["lsd_octaves"] = dict(zip(pf.LSD_OCTAVES, PROFILES[arm], strict=True))
        (results / "runs" / rid / "final.json").write_text(json.dumps(record))
    with (results / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    references = {d: _references(data_root, d) for d in ("ixi", "oasis1")}
    heldout = root / "heldout.json"
    heldout.write_text(json.dumps({"references": references, "runs": held}))
    tables = root / "tables.json"
    tables.write_text(json.dumps({"tables": {"t1a_cells_fidelity": {"rows": t1a},
                                             "t1b_cells_mechanism": {"rows": t1b}}}))
    return {"data_root": data_root, "results": results, "eval_dir": eval_dir,
            "heldout": heldout, "tables": tables}


def _argv(tree: dict[str, Path], out: Path, work: Path) -> list[str]:
    return ["--data-root", str(tree["data_root"]), "--results", str(tree["results"]),
            "--eval-dir", str(tree["eval_dir"]), "--heldout", str(tree["heldout"]),
            "--out", str(out), "--work", str(work), "--tables", str(tree["tables"])]


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> dict[str, Path]:
    return build_tree(tmp_path_factory.mktemp("t82"))


# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------


def test_paper_constants() -> None:
    assert PAPER_WIDTH_IN == 5.5
    assert ARM_LABELS == {"A0": "default", "A1": "+prior", "A2": "+spacing", "A3": "matched"}


def test_expectations_are_the_tickets() -> None:
    e = Expectations()
    assert e.rows == (3311, 2488, 5)
    assert e.rng_seed == {"final": 0, "heldout": 2026} and e.batch_size == 32
    assert e.precision == {"A0": (0.655, 0.661), "A1": (0.700, 0.715), "A3": (0.757, 0.758)}
    assert e.precision_tol == 5e-4


# --------------------------------------------------------------------------------------------
# CRN
# --------------------------------------------------------------------------------------------


def _crn_runs(changes: dict | None = None) -> dict[str, dict[str, SampleSet]]:
    images = _images()[:200]
    final_idx, held_idx = np.arange(6) + 10, np.arange(3) + 100
    runs = {}
    for k, rid in enumerate(("a", "b", "c", "d")):
        runs[rid] = {"final": _sample_set("final", images, final_idx, 1, 0, k),
                     "heldout": _sample_set("heldout", images, held_idx, PER_SEED, 2026, k)}
    for (rid, name), value in (changes or {}).items():
        runs[rid][name] = value(runs[rid][name], images)
    return runs


def test_crn_holds_on_identical_sets() -> None:
    record = assert_crn(_crn_runs(), Expectations())
    assert record["final"]["n_seeds"] == 6 and record["heldout"]["n_per_seed"] == PER_SEED
    assert record["final"]["rng_seed"] == 0 and record["heldout"]["rng_seed"] == 2026


@pytest.mark.parametrize("change, message", [
    (lambda s, im: SampleSet(s.name, s.seed_idx[::-1].copy(), im[s.seed_idx[::-1]], s.samples,
                             s.signature), "seed_idx differs"),
    (lambda s, im: SampleSet(s.name, s.seed_idx, s.seeds, s.samples,
                             {**s.signature, "rng_seed": 1}), "rng_seed"),
    (lambda s, im: SampleSet(s.name, s.seed_idx, s.seeds, s.samples,
                             {**s.signature, "batch_size": 16}), "batch_size"),
    (lambda s, im: SampleSet(s.name, s.seed_idx, 255 - s.seeds, s.samples, s.signature),
     "seed images differ"),
])
def test_crn_violations_raise(change, message) -> None:
    with pytest.raises(CRNError, match=message):
        assert_crn(_crn_runs({("c", "final"): change}), Expectations())


def test_crn_heldout_violation_raises() -> None:
    def bump(s: SampleSet, im: np.ndarray) -> SampleSet:
        return SampleSet(s.name, s.seed_idx, s.seeds, s.samples, {**s.signature, "rng_seed": 0})
    with pytest.raises(CRNError, match="d/heldout: rng_seed"):
        assert_crn(_crn_runs({("d", "heldout"): bump}), Expectations())


# --------------------------------------------------------------------------------------------
# Selection and skip rule
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fraction, expected", [(0.0, 0.0), (0.099, 0.099), (0.5, 0.5)])
def test_foreground_fraction(fraction: float, expected: float) -> None:
    image = np.zeros(1000, dtype=np.uint8)
    image[: int(round(fraction * 1000))] = 26  # 26/255 > 0.1
    np.testing.assert_allclose(foreground_fraction(image), expected)


def test_foreground_threshold_is_strict() -> None:
    image = np.full((10, 10), 25, dtype=np.uint8)  # 25/255 = 0.098 < 0.1
    assert foreground_fraction(image) == 0.0


def _sets_with(final_seeds: np.ndarray, held_seeds: np.ndarray) -> tuple[SampleSet, SampleSet]:
    final = SampleSet("final", np.arange(final_seeds.shape[0]) + 1000, final_seeds,
                      np.zeros_like(final_seeds), {})
    held = SampleSet("heldout", np.arange(held_seeds.shape[0]) + 5,
                     held_seeds, np.zeros((held_seeds.shape[0] * PER_SEED, SIZE, SIZE),
                                          dtype=np.uint8), {})
    return final, held


def test_selection_without_skips() -> None:
    full = np.full((5, SIZE, SIZE), 200, dtype=np.uint8)
    final, held = _sets_with(full, full[:3])
    rows = select_rows(final, held)
    assert [(r.label, r.position, r.sample_index) for r in rows] == [
        ("train 1", 0, 0), ("train 2", 1, 1), ("unseen", 0, 0)]
    assert [r.image_index for r in rows] == [1000, 1001, 5]
    assert all(not r.skipped for r in rows)


def test_skip_rule_moves_to_next_position() -> None:
    full = np.full((5, SIZE, SIZE), 200, dtype=np.uint8)
    final_seeds, held_seeds = full.copy(), full[:3].copy()
    final_seeds[1] = 0          # train 2 would be near-empty: skipped
    held_seeds[0] = 0           # unseen seed 0: one bright row of 16 = 6.25% < 10%
    held_seeds[0, 0, :] = 200
    final, held = _sets_with(final_seeds, held_seeds)
    rows = select_rows(final, held)
    assert [r.position for r in rows] == [0, 2, 1]
    assert [r.sample_index for r in rows] == [0, 2, PER_SEED]
    assert [s.position for s in rows[1].skipped] == [1]
    assert rows[2].skipped[0].image_index == 5
    np.testing.assert_allclose(rows[2].skipped[0].foreground, 1 / 16)
    verdict = assert_selection(rows, Expectations(rows=(1000, 1001, 5)))
    assert "skip rule moved" in verdict


def test_selection_runs_out_of_seeds() -> None:
    empty = np.zeros((2, SIZE, SIZE), dtype=np.uint8)
    final, held = _sets_with(empty, empty)
    with pytest.raises(SelectionError, match="passes the skip rule"):
        select_rows(final, held)


def test_selection_assertion_fails_on_other_images() -> None:
    full = np.full((5, SIZE, SIZE), 200, dtype=np.uint8)
    rows = select_rows(*_sets_with(full, full[:3]))
    assert "as expected" in assert_selection(rows, Expectations(rows=(1000, 1001, 5)))
    with pytest.raises(SelectionError, match="expected"):
        assert_selection(rows, Expectations())


# --------------------------------------------------------------------------------------------
# Heat blur
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 6, 15])
def test_kept_variance_matches_dctblur(n: int) -> None:
    from scipy.fft import dctn, idctn

    width, sigma = 32, 4.0
    coefs = np.zeros((width, width))
    coefs[n, 0] = 1.0
    image = idctn(coefs, norm="ortho")
    lo, hi = image.min(), image.max()
    as_uint8 = np.round((image - lo) / (hi - lo) * 255).astype(np.uint8)
    blurred = prior_state(as_uint8, sigma)
    before = dctn(as_uint8 / 255.0, norm="ortho")[n, 0]
    after = dctn(blurred, norm="ortho")[n, 0]
    np.testing.assert_allclose((after / before) ** 2, kept_variance(n / 2, sigma, width),
                               rtol=1e-5)


def test_prior_state_keeps_mean_and_range() -> None:
    image = _images()[0]
    for sigma in (24.0, 96.0):
        state = prior_state(image, sigma)
        assert state.shape == image.shape
        np.testing.assert_allclose(state.mean(), image.mean() / 255.0, rtol=1e-6)
        assert state.min() >= -1e-9 and state.max() <= 1.0 + 1e-9


def test_kept_variance_half_points() -> None:
    # The prior keeps half the variance at c = W sqrt(ln 2) / (2 pi sigma): 0.26 and 1.06 c/img.
    for sigma, c_half in ((96.0, 0.2650), (24.0, 1.0599)):
        np.testing.assert_allclose(kept_variance(c_half, sigma), 0.5, atol=1e-3)


# --------------------------------------------------------------------------------------------
# Expected seed means
# --------------------------------------------------------------------------------------------


def _points(values: dict[str, list[tuple[float, float]]]) -> dict[str, pf.ArmPoints]:
    return {arm: pf.ArmPoints("ixi", arm, tuple(f"ixi_{arm}_s{i + 1}" for i in range(len(v))),
                              np.zeros(len(v)), np.array([x[0] for x in v]),
                              np.array([x[1] for x in v])) for arm, v in values.items()}


def test_precision_means_pass_and_fail() -> None:
    good = _points({"A0": [(0.655, 0.661)] * 3, "A1": [(0.6996, 0.7154)] * 2,
                    "A3": [(0.7575, 0.7585)] * 3})
    assert len(assert_precision_means(good, Expectations())) == 6
    bad = _points({"A0": [(0.656, 0.661)] * 3, "A1": [(0.70, 0.715)] * 2,
                   "A3": [(0.757, 0.758)] * 3})
    with pytest.raises(ExpectationError, match="precision F default"):
        assert_precision_means(bad, Expectations())


@pytest.mark.skipif(not (T75_JSON.is_file() and A2_JSON.is_file()),
                    reason="the committed T7.5 and A2 JSONs are needed")
def test_expected_seed_means_on_the_committed_jsons() -> None:
    metrics, record = merge_heldout([T75_JSON, A2_JSON])
    assert record["identical_in_several_files"] == ["ixi_A0_s1", "ixi_A0_s2", "ixi_A0_s3"]
    index = [IndexRun(rid, "ixi", rid.split("_")[1], int(rid[-1]), {"seed_nn_fraction": "0"})
             for rid in metrics if rid.startswith("ixi_")]
    points = {arm: pf.arm_points(index, metrics, "ixi", arm) for arm in ("A0", "A1", "A2", "A3")}
    assert len(assert_precision_means(points, Expectations())) == 6
    assert points["A2"].n == 2 and points["A0"].n == 3 and points["A3"].n == 3


def test_spectrum_expectations() -> None:
    profiles = {arm: np.array([p, p]) for arm, p in PROFILES.items()}
    lines = assert_spectrum_means(profiles, Expectations())
    assert any("below 0 above 4 c/img" in line for line in lines)
    shifted = dict(profiles)
    shifted["A3"] = profiles["A3"] + np.array([0, 0.2, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ExpectationError, match="1-2 matched"):
        assert_spectrum_means(shifted, Expectations())
    positive = dict(profiles)
    positive["A1"] = profiles["A1"] + np.array([0, 0, 0, 0, 0, 0, 0.5, 0])
    with pytest.raises(ExpectationError, match="A1 is not below 0"):
        assert_spectrum_means(positive, Expectations())


# --------------------------------------------------------------------------------------------
# Held-out JSON merge
# --------------------------------------------------------------------------------------------


def _held(path: Path, runs: dict[str, float], digest: str = "abc") -> Path:
    path.write_text(json.dumps({
        "references": {"ixi": {"r_minus_idx_sha256": digest}},
        "runs": {r: {"metrics": {"F_rminus": {"precision": v}, "H_rminus": {"precision": v}}}
                 for r, v in runs.items()}}))
    return path


def test_merge_heldout(tmp_path: Path) -> None:
    a = _held(tmp_path / "a.json", {"ixi_A0_s1": 0.6, "ixi_A3_s1": 0.7})
    b = _held(tmp_path / "b.json", {"ixi_A0_s1": 0.6, "ixi_A2_s1": 0.65})
    metrics, record = merge_heldout([a, b])
    assert sorted(metrics) == ["ixi_A0_s1", "ixi_A2_s1", "ixi_A3_s1"]
    assert record["identical_in_several_files"] == ["ixi_A0_s1"]
    c = _held(tmp_path / "c.json", {"ixi_A0_s1": 0.6000001})
    with pytest.raises(PaperF2Error, match="differ"):
        merge_heldout([a, c])
    d = _held(tmp_path / "d.json", {"ixi_A2_s1": 0.65}, digest="other")
    with pytest.raises(PaperF2Error, match="reference sets differ"):
        merge_heldout([a, d])


# --------------------------------------------------------------------------------------------
# The A2 wrapper: ARMS restored, cache files untouched
# --------------------------------------------------------------------------------------------


def test_run_a2_heldout_guards_the_cache(tmp_path: Path, monkeypatch) -> None:
    from ihdm.cli import heldout_fidelity as t75

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "tables.md").write_text("T7.5 tables\n")
    reference = _held(tmp_path / "ref.json", {"ixi_A0_s1": 0.6})
    seen = {}

    def fake_main(argv: list[str]) -> int:
        args = dict(zip(argv[::2], argv[1::2], strict=True))
        work, out = Path(args["--work"]), Path(args["--out"])
        seen["arms"] = t75.ARMS
        (work / "tables.md").write_text("subset\n")
        (work / "features" / "ixi_A2_s1_F.npy").write_bytes(b"x")
        out.mkdir(parents=True, exist_ok=True)
        _held(out / "heldout_fidelity.json", {"ixi_A0_s1": 0.6, "ixi_A2_s1": 0.65})
        return 0

    monkeypatch.setattr(t75, "main", fake_main)
    before = t75.ARMS
    out = tmp_path / "out"
    assert pf.run_a2_heldout(tmp_path, tmp_path, cache, out, "ixi_A0_s1,ixi_A2_s1",
                             reference) == 0
    assert "A2" in seen["arms"] and t75.ARMS == before
    assert (cache / "tables.md").read_text() == "T7.5 tables\n"
    assert (cache / "features" / "ixi_A2_s1_F.npy").is_file()  # additions reach the cache
    assert (out / "tables.md").read_text() == "subset\n"
    checks = json.loads((out / "checks.json").read_text())
    assert checks["cache_unchanged"] and checks["check"]["identical_runs"] == ["ixi_A0_s1"]


# --------------------------------------------------------------------------------------------
# CLI: exit codes, end-to-end build, byte stability
# --------------------------------------------------------------------------------------------


def test_missing_input_exits_2(tmp_path: Path) -> None:
    argv = ["--data-root", str(tmp_path / "no"), "--results", str(tmp_path / "no"),
            "--eval-dir", str(tmp_path / "no"), "--heldout", str(tmp_path / "no.json"),
            "--out", str(tmp_path / "out"), "--work", str(tmp_path / "w"),
            "--tables", str(tmp_path / "no.json")]
    assert cli.main(argv) == 2
    assert not (tmp_path / "out").exists()


def test_missing_tar_exits_2(tree: dict[str, Path], tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    shutil.copytree(tree["eval_dir"], eval_dir)
    (eval_dir / "ixi_A1_s1_amp-fp16.tar").unlink()
    argv = _argv({**tree, "eval_dir": eval_dir}, tmp_path / "out", tmp_path / "w")
    assert cli.main(argv) == 2


def test_failing_crn_exits_1(tmp_path: Path, capsys) -> None:
    broken = build_tree(tmp_path / "broken", broken_crn=True)
    assert cli.main(_argv(broken, tmp_path / "out", tmp_path / "w")) == 1
    assert "common random numbers violated" in capsys.readouterr().err
    assert not (tmp_path / "out" / "f2_arms.pdf").exists()


def test_build_end_to_end_and_byte_stable(tree: dict[str, Path], tmp_path: Path) -> None:
    outs = [tmp_path / "run1", tmp_path / "run2"]
    for k, out in enumerate(outs):
        # Separate scratch folders: the second run extracts the tars again.
        assert cli.main(_argv(tree, out, tmp_path / f"work{k}")) == 0
    for name in ("f2_arms.pdf", "f2_arms.png", "f2_arms.svg", "t1_main.tex", "t1_main.md"):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
    svg = (outs[0] / "f2_arms.svg").read_text()
    assert "<dc:date>" not in svg and "data:image/png;base64" in svg
    for gid in ("a_thumb_r2_A3", "a_thumb_r0_u_matched", "b_pr", "b_pr_arrow_F_ixi",
                "b_pr_arrow_H_oasis1", "b_pr_mean_H_ixi_A3", "b_dist", "b_dist_real_band",
                "b_dist_mean_H_train_A3", "c_spectrum", "c_prior_fill", "c_band_0", "c_band_7"):
        assert f'id="{gid}"' in svg, gid
    for gone in ("key", "b_strip", "c_band_label_0"):  # the configurations key, the old (b)
        assert f'id="{gone}"' not in svg, gone
    assert "<text" in svg  # svg.fonttype none keeps text as text
    notes = (outs[0] / "F2.md").read_text()
    assert "rows are images (3311, 2488, 5), as expected; no skip" in notes
    assert "CRN, final set" in notes
    assert "each band scaled to its own range" in notes
    assert "M ixi_A0_s1: recomputed" in notes and "M ixi_A3_s1: recomputed" in notes
    tex = (outs[0] / "t1_main.tex").read_text()
    assert "\\toprule" in tex and "\\scriptsize [" in tex and "matched (A3)" in tex
    assert "resizebox" not in tex


def test_t1_cross_check_detects_a_mismatch(tree: dict[str, Path], tmp_path: Path) -> None:
    tables = json.loads(tree["tables"].read_text())
    tables["tables"]["t1a_cells_fidelity"]["rows"][0]["kid"] += 1e-3
    path = tmp_path / "tables.json"
    path.write_text(json.dumps(tables))
    rows = pf.t1_rows(pf.read_index(tree["results"]))
    with pytest.raises(PaperF2Error, match="differ from tables.json"):
        pf.cross_check_tables(rows, path)


def test_missing_oasis_tar_exits_2(tree: dict[str, Path], tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    shutil.copytree(tree["eval_dir"], eval_dir)
    (eval_dir / "oasis1_A3_s2_amp-fp16.tar").unlink()
    assert cli.main(_argv({**tree, "eval_dir": eval_dir}, tmp_path / "out", tmp_path / "w")) == 2


def test_missing_dataset_file_exits_2(tree: dict[str, Path], tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    shutil.copytree(tree["data_root"], data_root)
    (data_root / "oasis1" / "splits.json").unlink()
    assert cli.main(_argv({**tree, "data_root": data_root}, tmp_path / "out",
                          tmp_path / "w")) == 2


def _edited_index(tree: dict[str, Path], tmp_path: Path, run_id: str, key: str,
                  delta: float) -> dict[str, Path]:
    results = tmp_path / "results"
    shutil.copytree(tree["results"], results)
    rows = list(csv.DictReader((results / "index.csv").open(newline="")))
    for row in rows:
        if row["run_id"] == run_id:
            row[key] = str(float(row[key]) + delta)
    with (results / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # T1 cross-checks seed-NN against tables.json: change it there too, so the anchor fails.
    tables = json.loads(tree["tables"].read_text())
    for row in tables["tables"]["t1b_cells_mechanism"]["rows"]:
        if row["run_id"] == run_id and key in row:
            row[key] += delta
    (tmp_path / "tables.json").write_text(json.dumps(tables))
    return {**tree, "results": results, "tables": tmp_path / "tables.json"}


@pytest.mark.parametrize("run_id, key", [("ixi_A0_s1", "M"), ("ixi_A3_s1", "seed_nn_fraction")])
def test_anchor_mismatch_exits_1(tree: dict[str, Path], tmp_path: Path, capsys, run_id: str,
                                 key: str) -> None:
    edited = _edited_index(tree, tmp_path, run_id, key, 0.01)
    assert cli.main(_argv(edited, tmp_path / "out", tmp_path / "w")) == 1
    err = capsys.readouterr().err
    assert "distance anchor" in err and run_id in err
    assert not (tmp_path / "out" / "f2_arms.pdf").exists()


def test_fidelity_diversity_table_mismatch_exits_1(tree: dict[str, Path], tmp_path: Path,
                                                   capsys) -> None:
    report = json.loads(tree["heldout"].read_text())
    report["runs"]["ixi_A2_s1"]["metrics"]["H_rminus"]["recall"] += 0.01
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps(report))
    assert cli.main(_argv({**tree, "heldout": heldout}, tmp_path / "out", tmp_path / "w")) == 1
    assert "IXI +spacing recall H" in capsys.readouterr().err


def test_r_minus_must_match_the_heldout_json(tree: dict[str, Path], tmp_path: Path,
                                             capsys) -> None:
    report = json.loads(tree["heldout"].read_text())
    report["references"]["oasis1"]["r_minus_idx_sha256"] = "0" * 64
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps(report))
    assert cli.main(_argv({**tree, "heldout": heldout}, tmp_path / "out", tmp_path / "w")) == 1
    assert "R⁻ rebuilt from the dataset" in capsys.readouterr().err


def test_anchor_lines_name_the_required_runs(tree: dict[str, Path], tmp_path: Path) -> None:
    paths = pf.F2Paths(tree["data_root"], tree["results"], tree["eval_dir"], (tree["heldout"],),
                       tmp_path / "out", tmp_path / "w", tree["tables"])
    index = pf.read_index(tree["results"])
    _, record = merge_heldout([tree["heldout"]])
    distances = pf.load_distances(paths, index, record["references"])
    lines = pf.assert_anchor(distances, index, Expectations())
    assert sum(line.startswith("M ixi_A") for line in lines) == 2
    assert lines[-1].startswith("the other 12 runs also match")
    with pytest.raises(ExpectationError, match="anchor runs not computed"):
        pf.assert_anchor(distances, index, Expectations(anchor_runs=("ixi_A9_s1",)))
    # The band of real images is centred on 1 by construction (D_ref is its median).
    lo, hi = distances.real_quartiles["ixi"]
    assert lo <= 1.0 <= hi
    assert distances.sizes["ixi"] == {"n_train": 3670, "n_r_minus": 10, "n_train_subjects": 367}


# --------------------------------------------------------------------------------------------
# Panel (c): band-pass images
# --------------------------------------------------------------------------------------------


def test_band_images_sum_back_to_the_slice() -> None:
    from ihdm.analysis.paper_fm import mode_cycles

    image = np.random.default_rng(5).random((192, 192))
    bands, dc, corner = band_decomposition(image)
    assert len(bands) == 8
    np.testing.assert_allclose(sum(bands) + dc + corner, image, rtol=0, atol=1e-10)
    np.testing.assert_allclose(dc, np.full_like(image, image.mean()), atol=1e-10)
    # Each band holds only its octave's modes; the corner only modes above 96 c/img.
    from scipy.fft import dctn

    cycles = mode_cycles(192)
    for (lo, hi), band in zip(pf.OCTAVE_EDGES, bands, strict=True):
        coef = dctn(band, norm="ortho")
        inside = (cycles >= lo) & ((cycles < hi) | ((hi == 96.0) & (cycles == hi)))
        assert np.abs(coef[~inside]).max() < 1e-10
        assert np.abs(coef[inside]).max() > 0
    assert np.abs(dctn(corner, norm="ortho")[cycles <= 96.0]).max() < 1e-10
    assert np.abs(corner).max() > 0


def test_head_crop_is_a_centred_square() -> None:
    image = np.zeros((192, 192))
    image[20:192, 14:177] = 0.5  # the reference slice's head box (rows 20-191, cols 14-176)
    assert head_crop(image) == (20, 192, 9, 181)
    assert head_crop(np.zeros((8, 8))) == (0, 8, 0, 8)
    corner = np.zeros((10, 10))
    corner[0:2, 0:6] = 1.0  # a wide box at the edge: the square is clipped to the image
    assert head_crop(corner) == (0, 6, 0, 6)


@pytest.mark.skipif(not (DATA_ROOT / "ixi" / "images.npy").is_file(),
                    reason="the IXI dataset is needed")
def test_reference_slice_is_image_3705_and_decomposes_exactly() -> None:
    from ihdm.spectral.profile import reference_example

    bands = pf.load_band_images(DATA_ROOT, Expectations())
    assert bands.image_index == 3705 and bands.crop == (20, 192, 9, 181)
    image, _ = reference_example(DATA_ROOT, "ixi", np.random.default_rng(2026))
    parts, dc, corner = band_decomposition(np.asarray(image, dtype=np.float64))
    np.testing.assert_allclose(sum(parts) + dc + corner, image, rtol=0, atol=1e-10)


# --------------------------------------------------------------------------------------------
# T1 and the distance anchor on the real data
# --------------------------------------------------------------------------------------------


def test_committed_t1_is_byte_identical() -> None:
    for name, digest in T1_SHA256.items():
        assert hashlib.sha256((PAPER / name).read_bytes()).hexdigest() == digest, name


def test_committed_caption_length_and_content() -> None:
    text = (PAPER / "F2.md").read_text()
    caption = "\n".join(line for line in text.split("## Caption draft", 1)[1].splitlines()
                        if line.startswith(">"))
    words = pf.caption_words(caption)
    assert 180 <= words <= 230, words
    assert f"Caption length: {words} words." in text
    for phrase in ("R⁻", "(F, filled)", "(H, hollow)", "40 prior states at one slice level",
                   "R5", "n = 3/2/2/3", "n = 2/2", "post hoc",
                   "each band scaled to its own range"):
        assert phrase in caption, phrase


def test_caption_words_ignores_markup() -> None:
    assert pf.caption_words("> **Bold title.**\n>\n> (a) One → two, 0.5 c/img.") == 7


@pytest.mark.integration
@pytest.mark.skipif(not (EVAL_DIR.is_dir() and (DATA_ROOT / "_results" / "index.csv").is_file()),
                    reason="the IXI eval tars (SanDisk) and the data root are needed")
@pytest.mark.parametrize("run_id", ["ixi_A0_s1", "ixi_A3_s1"])
def test_anchor_reproduces_index_csv(run_id: str, tmp_path: Path) -> None:
    from ihdm.analysis import seed_distance as sd
    from ihdm.data.format import read_dataset

    images, frame, splits, _ = read_dataset(DATA_ROOT / "ixi")
    refs = select_references(frame, splits, "ixi")
    train_idx = np.sort(np.asarray(splits["train"], dtype=np.int64))
    train = np.asarray(images[train_idx])
    subjects = frame["subject"].astype(str).to_numpy()[train_idx]
    d_ref = float(np.median(sd.reference_distances(np.asarray(images[refs.r_minus_idx]), train)))
    folder = pf.extract_run(EVAL_DIR, run_id, tmp_path)
    final = pf.load_sample_set(folder / "final", "final")
    held = pf.load_sample_set(folder / "heldout", "heldout")
    run = sd.compute_run(run_id, final.samples, np.searchsorted(train_idx, final.seed_idx),
                         held.samples, held.seeds, train, subjects, d_ref)
    stored = {r.run_id: r for r in pf.read_index(DATA_ROOT / "_results")}[run_id]
    assert abs(run.m_ratio - stored.value("M")) <= 5e-3
    assert abs(run.seed_nn - stored.value("seed_nn_fraction")) <= 5e-3
    final_json = json.loads((DATA_ROOT / "_results" / "runs" / run_id / "final.json").read_text())
    np.testing.assert_allclose(d_ref, final_json["d_heldout_median"], rtol=1e-4)
