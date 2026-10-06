"""Tests of appendix A1 (T8.3): CRN, epochs, running mean, plateau, exit codes, byte stability."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from ihdm.analysis import paper_a1 as pa
from ihdm.analysis.paper_a1 import (
    CHECKPOINTS,
    ITERS_PER_EPOCH,
    PAPER_WIDTH_IN,
    CRNError,
    LsdSet,
    PaperA1Error,
    assert_crn,
    epochs_to_iterations,
    index_sha256,
    iterations_to_epochs,
    plateau_step,
    running_mean,
)
from ihdm.cli import paper_a1 as cli

N_SEEDS = 4
SIZE = 16
SEED_IDX = np.array([7, 3, 11, 5], dtype=np.int64)
RNG_SEED = 2026
BATCH = 32
PHOTO_PRECISION = {("lsun_church", "A0"): [0.004, 0.002, 0.003],
                   ("lsun_church", "A3"): [0.05, 0.07, 0.06],
                   ("lsun_bedroom", "A0"): [0.01, 0.012],
                   ("lsun_bedroom", "A3"): [0.4321, 0.4321]}
MRI_PRECISION = {("ixi", "A0"): [0.61, 0.63, 0.62], ("ixi", "A3"): [0.7, 0.72, 0.71],
                 ("oasis1", "A0"): [0.68, 0.69], ("oasis1", "A3"): [0.88, 0.86]}


# --------------------------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------------------------


def _history(seed: int, scale: float) -> str:
    steps = np.arange(0, 60_001, 50)
    rng = np.random.default_rng(seed)
    loss = scale * (1.0 + 3.0 * np.exp(-steps / 3_000.0)) * (1 + 0.02 * rng.standard_normal(
        steps.size))
    lines = []
    for s, v in zip(steps, loss, strict=True):
        lines.append(json.dumps({"step": int(s), "kind": "train", "loss": float(v)}))
        if s == 40_000:
            lines.append(json.dumps({"step": 40_001, "kind": "resume"}))
    return "\n".join(lines) + "\n"


def _run_files(folder: Path, seed: int, scale: float) -> None:
    folder.mkdir(parents=True)
    (folder / "metrics.canonical.jsonl").write_text(_history(seed, scale))
    config = {
        "data": {"num_channels": 1, "image_size": 192},
        "model": {"channel_mult": [1, 2, 2, 2], "num_res_blocks": 4, "K": 200},
        "optim": {"lr": 1e-4},
        "training": {"batch_size": 16, "n_iters": 60_000},
    }
    (folder / "config.json").write_text(json.dumps(config))
    summary = {"env": {"n_train": 3_200},
               "seed_lists": {"intermediate_sha256": index_sha256(SEED_IDX),
                              "intermediate_path": "/x/eval_seeds_500.npy"}}
    (folder / "summary.json").write_text(json.dumps(summary))
    (folder / "manifest.json").write_text(json.dumps({"n_params": 61_056_257}))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def _request(step: int, rng_seed: int = RNG_SEED, batch: int = BATCH) -> dict:
    return {"set": "lsd", "step": step, "signature": {
        "rng_seed": rng_seed, "batch_size": batch, "n_per_seed": 1, "n_seeds": N_SEEDS,
        "set": "lsd", "seed_idx_sha256": index_sha256(SEED_IDX)}}


def _write_tar(path: Path, rid: str, overrides: dict[int, dict] | None = None) -> None:
    rng = np.random.default_rng(len(rid))
    seeds = rng.integers(0, 256, (N_SEEDS, 1, SIZE, SIZE), dtype=np.uint8)
    with tarfile.open(path, "w") as tar:
        _add(tar, f"{rid}/samples_amp-fp16/060000/final/samples.npy", b"unrelated")
        for step in CHECKPOINTS:
            base = f"{rid}/samples_amp-fp16/{step:06d}/lsd"
            samples = rng.integers(0, 256, (N_SEEDS, 1, SIZE, SIZE), dtype=np.uint8)
            request = _request(step)
            request["signature"].update((overrides or {}).get(step, {}))
            _add(tar, f"{base}/samples.npy", _npy_bytes(samples))
            _add(tar, f"{base}/seed_idx.npy", _npy_bytes(SEED_IDX))
            _add(tar, f"{base}/seeds.npy", _npy_bytes(seeds))
            _add(tar, f"{base}/request.json", json.dumps(request).encode())


def _index_csv(path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "dataset", "arm", "seed", "precision"])
        for (dataset, arm), values in {**PHOTO_PRECISION, **MRI_PRECISION}.items():
            for k, value in enumerate(values, start=1):
                writer.writerow([f"{dataset}_{arm}_s{k}", dataset, arm, k, value])


def _photo_json(path: Path) -> None:
    data = {"late_window": {"baseline": {"precision": 0.011}, "r128": {"precision": 0.222},
                            "n32k": {"precision": 0.033}},
            "tasks": {"baseline": {"env": {"n_train": 3_200}},
                      "n32k": {"env": {"n_train": 32_000}}}}
    path.write_text(json.dumps(data))


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> Path:
    """A synthetic _results folder, eval tars and photo_diagnostic.json; never modified."""
    root = tmp_path_factory.mktemp("a1")
    results = root / "results"
    for dataset, seeds in pa.LOSS_SEEDS.items():
        for seed in seeds:
            scale = 0.3 if dataset == "lsun_church" else 0.2
            _run_files(results / "runs" / f"{dataset}_A0_s{seed}", seed, scale)
    _index_csv(results / "index.csv")
    tars = root / "tars"
    tars.mkdir()
    for dataset in pa.DATASETS:
        rid = f"{dataset}_A0_s1"
        _write_tar(tars / f"{rid}_amp-fp16.tar", rid)
    _photo_json(root / "photo_diagnostic.json")
    return root


@pytest.fixture
def inputs(pristine, tmp_path) -> Path:
    """A private copy of the synthetic inputs that the test may damage."""
    root = tmp_path / "in"
    shutil.copytree(pristine, root)
    return root


def _args(root: Path, out: Path, work: Path) -> list[str]:
    return ["--results", str(root / "results"), "--eval-dir", str(root / "tars"),
            "--out", str(out), "--work", str(work),
            "--photo-diagnostic", str(root / "photo_diagnostic.json")]


def _digests(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


# --------------------------------------------------------------------------------------------
# Epochs, running mean, plateau
# --------------------------------------------------------------------------------------------


def test_iterations_per_epoch_is_the_frozen_contract():
    assert ITERS_PER_EPOCH == 3_200 // 16 == 200


@pytest.mark.parametrize(("iterations", "epochs"), [(0, 0.0), (200, 1.0), (5_000, 25.0),
                                                     (25_000, 125.0), (60_000, 300.0)])
def test_epoch_conversion(iterations, epochs):
    assert iterations_to_epochs(iterations) == pytest.approx(epochs)
    assert epochs_to_iterations(epochs) == pytest.approx(iterations)


def test_epoch_conversion_round_trips_arrays():
    its = np.array([0.0, 50.0, 12_345.0, 60_000.0])
    np.testing.assert_allclose(epochs_to_iterations(iterations_to_epochs(its)), its, rtol=1e-12)
    np.testing.assert_allclose(iterations_to_epochs(its, iters_per_epoch=100), its / 100)


@pytest.mark.parametrize("per_epoch", [0, -200])
def test_epoch_conversion_rejects_non_positive(per_epoch):
    with pytest.raises(PaperA1Error):
        iterations_to_epochs(100, iters_per_epoch=per_epoch)
    with pytest.raises(PaperA1Error):
        epochs_to_iterations(1, iters_per_epoch=per_epoch)


@pytest.mark.parametrize("window", [1, 2, 3, 20, 50])
@pytest.mark.parametrize("n", [1, 5, 37])
def test_running_mean_matches_brute_force(window, n):
    values = np.random.default_rng(n * 100 + window).standard_normal(n)
    expected = np.array([values[max(0, i - window + 1):i + 1].mean() for i in range(n)])
    np.testing.assert_allclose(running_mean(values, window), expected, rtol=1e-12, atol=1e-12)


def test_running_mean_of_a_constant_is_constant_and_empty_is_empty():
    np.testing.assert_allclose(running_mean(np.full(30, 0.25), 20), np.full(30, 0.25))
    assert running_mean(np.empty(0), 20).size == 0


def test_running_mean_rejects_a_zero_window():
    with pytest.raises(PaperA1Error):
        running_mean(np.ones(4), 0)


def test_running_mean_window_is_one_thousand_iterations(pristine):
    history = pa.read_history(pristine / "results", "lsun_church_A0_s1")
    assert history.window_records == 20  # 1,000 iterations at one record per 50
    np.testing.assert_allclose(history.smooth, running_mean(history.loss, 20))


def test_plateau_step_is_after_the_last_excursion():
    steps = np.arange(0, 10_001, 100, dtype=float)
    curve = np.where(steps < 3_000, 2.0, 1.0)
    curve[steps == 5_000] = 1.5  # one late excursion
    step, final = plateau_step(steps, curve, tolerance=0.1, final_window=1_000)
    assert final == pytest.approx(1.0)
    assert step == 5_100


def test_plateau_step_of_a_flat_curve_is_its_start():
    steps = np.arange(0, 1_001, 50, dtype=float)
    assert plateau_step(steps, np.ones_like(steps), final_window=200) == (0, 1.0)


def test_canonical_records_drop_an_abandoned_segment(tmp_path):
    folder = tmp_path / "runs" / "r"
    folder.mkdir(parents=True)
    records = [{"step": s, "kind": "train", "loss": 1.0} for s in (0, 50, 100, 150)]
    records += [{"step": 101, "kind": "resume"}, {"step": 150, "kind": "train", "loss": 2.0}]
    (folder / "metrics.canonical.jsonl").write_text("\n".join(map(json.dumps, records)))
    history = pa.read_history(tmp_path, "r")
    np.testing.assert_array_equal(history.steps, [0, 50, 100, 150])
    np.testing.assert_array_equal(history.loss, [1.0, 1.0, 1.0, 2.0])


# --------------------------------------------------------------------------------------------
# CRN
# --------------------------------------------------------------------------------------------


def _sets(**changes) -> dict[int, LsdSet]:
    seeds = np.zeros((N_SEEDS, SIZE, SIZE), dtype=np.uint8)
    out = {}
    for step in CHECKPOINTS:
        request = _request(step)
        idx = SEED_IDX.copy()
        these = seeds.copy()
        if step == CHECKPOINTS[-1]:
            request["signature"].update(changes.get("signature", {}))
            if "seed_idx" in changes:
                idx = changes["seed_idx"]
                request["signature"]["seed_idx_sha256"] = index_sha256(idx)
            if changes.get("seed_image"):
                these[0, 0, 0] = 255
            if "step" in changes:
                request["step"] = changes["step"]
        out[step] = LsdSet(step, idx, these, np.zeros((N_SEEDS, SIZE, SIZE), np.uint8), request)
    return out


def test_crn_holds_on_identical_sets():
    record = assert_crn(_sets(), index_sha256(SEED_IDX))
    assert record["steps"] == list(CHECKPOINTS)
    assert (record["rng_seed"], record["batch_size"], record["n_seeds"]) == (RNG_SEED, BATCH, 4)


@pytest.mark.parametrize("changes", [
    {"signature": {"rng_seed": RNG_SEED + 1}},
    {"signature": {"batch_size": BATCH * 2}},
    {"signature": {"n_per_seed": 2}},
    {"signature": {"seed_idx_sha256": "0" * 64}},
    {"seed_idx": SEED_IDX[::-1].copy()},
    {"seed_image": True},
    {"step": 1},
], ids=["rng_seed", "batch_size", "n_per_seed", "signature_sha", "seed_idx", "seed_image",
        "step"])
def test_crn_violations_raise(changes):
    with pytest.raises(CRNError):
        assert_crn(_sets(**changes))


def test_crn_requires_the_frozen_seed_list():
    with pytest.raises(CRNError, match="eval_seeds_500"):
        assert_crn(_sets(), "f" * 64)


def test_crn_needs_two_checkpoints():
    with pytest.raises(CRNError):
        assert_crn({5_000: _sets()[5_000]})


def test_cli_exits_1_when_a_checkpoint_breaks_crn(inputs, tmp_path, capsys):
    rid = "lsun_bedroom_A0_s1"
    _write_tar(inputs / "tars" / f"{rid}_amp-fp16.tar", rid,
               overrides={30_000: {"rng_seed": RNG_SEED + 7}})
    assert cli.main(_args(inputs, tmp_path / "out", tmp_path / "work")) == cli.EXIT_FAIL
    err = capsys.readouterr().err
    assert "common random numbers violated" in err and "step 30000: rng_seed" in err
    assert not (tmp_path / "out" / "a1_natural_images.pdf").exists()


# --------------------------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("victim", [
    "tars/lsun_church_A0_s1_amp-fp16.tar",
    "results/runs/lsun_bedroom_A0_s2/metrics.canonical.jsonl",
    "results/runs/lsun_church_A0_s3/summary.json",
    "results/index.csv",
    "photo_diagnostic.json",
])
def test_cli_exits_2_on_a_missing_input(inputs, tmp_path, victim):
    (inputs / victim).unlink()
    assert cli.main(_args(inputs, tmp_path / "out", tmp_path / "work")) == cli.EXIT_MISSING


def test_cli_exits_2_on_a_missing_tar_member(inputs, tmp_path):
    rid = "lsun_church_A0_s1"
    tar_file = inputs / "tars" / f"{rid}_amp-fp16.tar"
    with tarfile.open(tar_file) as tar:
        kept = [(m, tar.extractfile(m).read()) for m in tar.getmembers()
                if "/045000/lsd/seeds.npy" not in m.name]
    with tarfile.open(tar_file, "w") as tar:
        for member, data in kept:
            tar.addfile(member, io.BytesIO(data))
    assert cli.main(_args(inputs, tmp_path / "out", tmp_path / "work")) == cli.EXIT_MISSING


def test_cli_exits_1_when_a_run_contradicts_table_a1(inputs, tmp_path, capsys):
    path = inputs / "results" / "runs" / "lsun_church_A0_s2" / "config.json"
    config = json.loads(path.read_text())
    config["optim"]["lr"] = 2e-4
    path.write_text(json.dumps(config))
    assert cli.main(_args(inputs, tmp_path / "out", tmp_path / "work")) == cli.EXIT_FAIL
    assert "lr: expected 0.0001" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(pristine, tmp_path_factory) -> Path:
    """One successful build on the synthetic inputs."""
    out = tmp_path_factory.mktemp("a1_out")
    assert cli.main(_args(pristine, out, tmp_path_factory.mktemp("a1_work"))) == cli.EXIT_OK
    return out


def test_outputs_are_written(built):
    names = {p.name for p in built.iterdir()}
    assert names == {"a1_natural_images.svg", "a1_natural_images.pdf", "a1_natural_images.png",
                     "a1_table.tex", "a1_table.md", "A1.md"}


def test_outputs_are_byte_stable(pristine, built, tmp_path):
    out = tmp_path / "again"
    assert cli.main(_args(pristine, out, tmp_path / "fresh_work")) == cli.EXIT_OK
    assert _digests(out) == _digests(built)


def test_svg_is_editable(built):
    svg = (built / "a1_natural_images.svg").read_text()
    assert "<text" in svg and "font-family" in svg  # svg.fonttype = "none": text stays text
    assert svg.count("data:image/png;base64") == 2 * (1 + len(CHECKPOINTS))
    assert 'id="b_thumb_lsun_church_0"' in svg and 'id="a_loss"' in svg
    assert "<dc:date" not in svg


def test_text_reads_its_numbers_from_files(built):
    notes = (built / "A1.md").read_text()
    assert "at most 0.43 in every photograph configuration" in notes
    assert "against 0.62–0.87 on MRI" in notes
    assert "from 0.011 to 0.222" in notes and "(32,000) give 0.033" in notes
    assert "10× more training images" in notes
    assert "200 iterations per epoch" in notes
    assert "plateau does not" in notes or "not that the samples converged" in notes
    assert "PLACEHOLDER (Sec. X; T8.0)" in notes


def test_table_fits_the_text_width(built):
    tex = (built / "a1_table.tex").read_text()
    spec = tex.split("\\begin{tabular}{", 1)[1].split("}\n", 1)[0]
    widths_cm = [float(w) for w in spec.replace("@{}", "").replace("p{", " ").replace("cm}", " ")
                 .split()]
    sep_cm = 4 / 72.27 * 2.54 * 2 * (len(widths_cm) - 1)
    assert sum(widths_cm) + sep_cm <= PAPER_WIDTH_IN * 2.54
    assert "\\toprule" in tex and "did not train on LSUN Bedrooms" in tex
    md = (built / "a1_table.md").read_text()
    assert md.count("\n|") == 2 + len(pa.TABLE_A1_ROWS)


def test_layout_meets_the_format(pristine, tmp_path):
    from matplotlib import pyplot as plt

    losses = pa.load_losses(pristine / "results")
    rows = []
    for dataset in pa.DATASETS:
        rid = f"{dataset}_A0_s1"
        folder = pa.extract_checkpoints(pristine / "tars", rid, tmp_path)
        sets = {s: pa.load_lsd_set(folder / f"{s:06d}" / "lsd", s) for s in CHECKPOINTS}
        rows.append(pa.thumb_row(dataset, rid, sets, index_sha256(SEED_IDX)))
    with pa.paper_style():
        fig = pa.draw_a1(pa.A1Data(tuple(losses), tuple(rows)))
        audit = pa.audit_layout(fig)
        plt.close(fig)
    assert audit.problems() == []
    assert audit.width_in == pytest.approx(5.5)
    assert audit.height_in <= 2.0
    assert audit.min_font_pt >= 7.0
    assert audit.min_thumb_in >= 0.45
    assert rows[0].image_index == int(SEED_IDX[0])
