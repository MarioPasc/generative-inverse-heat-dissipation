"""Tests of F2 and T1 (T8.2): CRN and selection rules, expectations, exit codes, byte stability."""

from __future__ import annotations

import csv
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from ihdm.analysis import paper_f2 as pf
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
    foreground_fraction,
    kept_variance,
    merge_heldout,
    prior_state,
    select_rows,
)
from ihdm.cli import paper_f2 as cli

REPO = Path(__file__).resolve().parents[2]
T75_JSON = REPO / "docs" / "RESULTS" / "heldout_fidelity" / "heldout_fidelity.json"
A2_JSON = REPO / "docs" / "RESULTS" / "paper" / "f2_a2_heldout" / "heldout_fidelity.json"

SIZE = 16
N_IMAGES = 3312
FINAL_IDX = np.array([3311, 2488, 1995, 1073, 1205, 161], dtype=np.int64)
HELDOUT_IDX = np.array([5, 35, 195], dtype=np.int64)
PER_SEED = 4
IXI_CELLS = {"A0": 3, "A2": 2, "A1": 2, "A3": 3}
OASIS_CELLS = {"A0": 2, "A3": 2}
#: Per-arm (precision F, precision H, seed-NN) equal to the ticket's expected means.
PRECISION = {"A0": (0.655, 0.661, 0.06), "A2": (0.66, 0.68, 0.07), "A1": (0.70, 0.715, 0.66),
             "A3": (0.757, 0.758, 0.72)}
OASIS_PRECISION = {"A0": (0.70, 0.70, 0.02), "A3": (0.83, 0.84, 0.56)}
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
    """Disc phantoms: every image has a bright disc (foreground well above 10%)."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    disc = ((yy - SIZE / 2 + 0.5) ** 2 + (xx - SIZE / 2 + 0.5) ** 2) <= (SIZE / 3) ** 2
    base = (disc * 200).astype(np.uint8)
    out = np.repeat(base[None], N_IMAGES, axis=0)
    out[:, 0, 0] = (np.arange(N_IMAGES) % 251).astype(np.uint8)  # make images distinct
    return out


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


def _write_tar(path: Path, run_id: str, images: np.ndarray, arm_seed: int,
               final_idx: np.ndarray = FINAL_IDX) -> None:
    base = f"{run_id}/samples_amp-fp16/060000"
    sets = {
        "final": _sample_set("final", images, final_idx, 1, 0, arm_seed),
        "heldout": _sample_set("heldout", images, HELDOUT_IDX, PER_SEED, 2026, arm_seed + 100),
    }
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


def _values(dataset: str, arm: str) -> tuple[float, float, float]:
    return (PRECISION if dataset == "ixi" else OASIS_PRECISION)[arm]


def build_tree(root: Path, broken_crn: bool = False) -> dict[str, Path]:
    """Write a complete synthetic input tree; return the CLI paths."""
    images = _images()
    data_root = root / "data"
    (data_root / "ixi").mkdir(parents=True)
    np.save(data_root / "ixi" / "images.npy", images)
    eval_dir = root / "eval"
    eval_dir.mkdir()
    for k, arm in enumerate(("A0", "A2", "A1", "A3")):
        final_idx = FINAL_IDX.copy()
        if broken_crn and arm == "A3":
            final_idx[2], final_idx[3] = final_idx[3], final_idx[2]
        _write_tar(eval_dir / f"ixi_{arm}_s1_amp-fp16.tar", f"ixi_{arm}_s1", images, k,
                   final_idx)
    results = root / "results"
    (results / "runs").mkdir(parents=True)
    rows, t1a, t1b, held = [], [], [], {}
    for rid, dataset, arm, seed in _runs():
        prec_f, prec_h, seed_nn = _values(dataset, arm)
        row = {"index": len(rows), "run_id": rid, "dataset": dataset, "arm": arm, "seed": seed,
               "kid": 0.01 * (seed + 1), "lsd_final": 0.1 * seed, "precision": prec_f - 0.02,
               "recall": 0.3, "seed_nn_fraction": seed_nn, "D_pix": 0.011}
        rows.append(row)
        t1a.append({"run_id": rid, "kid": row["kid"], "lsd_final": row["lsd_final"],
                    "recall": row["recall"]})
        t1b.append({"run_id": rid, "seed_nn_fraction": seed_nn, "D_pix": row["D_pix"]})
        held[rid] = {"metrics": {"F_rminus": {"precision": prec_f},
                                 "H_rminus": {"precision": prec_h}}}
        if dataset == "ixi":
            (results / "runs" / rid).mkdir()
            octaves = dict(zip(pf.LSD_OCTAVES, PROFILES[arm], strict=True))
            (results / "runs" / rid / "final.json").write_text(json.dumps({"lsd_octaves": octaves}))
    with (results / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    references = {"ixi": {"r_minus_idx_sha256": "abc", "n_r_minus": 400}}
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
    for gid in ("a_thumb_r2_A3", "a_thumb_r0_u_matched", "b_arrow_A1", "b_arrow_A2",
                "b_arrow_oasis", "b_strip", "c_prior_fill", "key"):
        assert f'id="{gid}"' in svg, gid
    assert "<text" in svg  # svg.fonttype none keeps text as text
    notes = (outs[0] / "F2.md").read_text()
    assert "rows are images (3311, 2488, 5), as expected; no skip" in notes
    assert "CRN, final set" in notes
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
