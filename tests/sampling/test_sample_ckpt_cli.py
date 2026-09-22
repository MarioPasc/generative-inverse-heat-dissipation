"""The ``python -m ihdm.cli.sample_ckpt`` command (``04-run-artifacts.md`` §4)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ihdm.cli.sample_ckpt import main
from ihdm.sampling.errors import SamplingError

#: A real run produced by the trainer, used by the integration test when it exists.
INTEGRATION_RUN = Path(
    os.environ.get(
        "IHDM_T22_INTEGRATION_RUN",
        "/tmp/claude-1000/-home-mpascual-research-code-TFM/"
        "afc690e9-0922-4501-b126-8f7d74370c45/scratchpad/runs_t22/lsun_church_A0_s1",
    )
)


def _assert_artifacts(out_dir: Path, n_seeds: int, n_per_seed: int, size: int) -> dict:
    """Check the five files of §4 and return the parsed ``request.json``."""
    samples = np.load(out_dir / "samples.npy")
    seeds = np.load(out_dir / "seeds.npy")
    seed_idx = np.load(out_dir / "seed_idx.npy")
    record = json.loads((out_dir / "request.json").read_text())

    assert samples.shape == (n_seeds, n_per_seed, size, size)
    assert samples.dtype == np.uint8
    assert seeds.shape == (n_seeds, size, size)
    assert seeds.dtype == np.uint8
    assert seed_idx.shape == (n_seeds,)
    assert (out_dir / "preview.png").stat().st_size > 0
    assert len(record["checkpoint"]["sha256"]) == 64
    assert record["env"]["git_sha"]
    assert record["timing"]["wall_s"] > 0
    assert record["timing"]["samples_per_s"] > 0
    return record


def test_cli_writes_the_five_artifacts(tiny_run, tmp_path, capsys):
    """The seed source produces samples, seeds, indices, the record and the preview."""
    workdir, ckpt = tiny_run
    out_dir = tmp_path / "out_seed"

    code = main(
        [
            "--run", str(workdir), "--ckpt", ckpt.name, "--source", "seed",
            "--n-seeds", "2", "--n-per-seed", "2", "--batch", "4",
            "--rng-seed", "0", "--device", "cpu", "--out", str(out_dir),
        ]
    )

    assert code == 0
    record = _assert_artifacts(out_dir, n_seeds=2, n_per_seed=2, size=32)
    assert record["args"]["source"] == "seed"
    assert record["request"]["start_level"] == 8
    assert "OK" in capsys.readouterr().out


def test_cli_train_source_and_start_level(tiny_run, tmp_path):
    """The train source and an explicit start level are recorded in ``request.json``."""
    workdir, ckpt = tiny_run
    out_dir = tmp_path / "out_train"

    main(
        [
            "--run", str(workdir), "--ckpt", ckpt.name, "--source", "train",
            "--n-seeds", "3", "--n-per-seed", "1", "--start-level", "2",
            "--delta", "0.0", "--device", "cpu", "--out", str(out_dir),
        ]
    )

    record = _assert_artifacts(out_dir, n_seeds=3, n_per_seed=1, size=32)
    assert record["request"]["start_level"] == 2
    assert record["request"]["delta"] == 0.0
    splits = json.loads((Path(record["args"]["run"]) / "config.json").read_text())
    dataset_root = Path(splits["data"]["root"]) / splits["data"]["dataset"]
    train_split = json.loads((dataset_root / "splits.json").read_text())["train"]
    assert set(np.load(out_dir / "seed_idx.npy").tolist()) <= set(train_split)


def test_cli_file_source(tiny_run, tmp_path):
    """``--source file`` takes the seeds from an array on disk."""
    workdir, ckpt = tiny_run
    seeds_file = tmp_path / "seeds_in.npy"
    np.save(seeds_file, np.random.default_rng(3).integers(0, 256, (5, 32, 32), dtype=np.uint8))
    out_dir = tmp_path / "out_file"

    main(
        [
            "--run", str(workdir), "--ckpt", ckpt.name, "--source", "file",
            "--seeds-file", str(seeds_file), "--n-seeds", "2", "--n-per-seed", "1",
            "--device", "cpu", "--out", str(out_dir),
        ]
    )

    _assert_artifacts(out_dir, n_seeds=2, n_per_seed=1, size=32)
    np.testing.assert_array_equal(
        np.load(out_dir / "seeds.npy"), np.load(seeds_file)[:2]
    )


def test_cli_file_source_without_array_raises(tiny_run, tmp_path):
    """``--source file`` without ``--seeds-file`` is refused."""
    workdir, ckpt = tiny_run
    with pytest.raises(SamplingError, match="requires --seeds-file"):
        main(
            [
                "--run", str(workdir), "--ckpt", ckpt.name, "--source", "file",
                "--n-seeds", "2", "--n-per-seed", "1", "--device", "cpu",
                "--out", str(tmp_path / "out"),
            ]
        )


def test_cli_is_reproducible(tiny_run, tmp_path):
    """Two invocations with the same ``--rng-seed`` write identical sample files."""
    workdir, ckpt = tiny_run
    common = [
        "--run", str(workdir), "--ckpt", ckpt.name, "--source", "seed",
        "--n-seeds", "2", "--n-per-seed", "2", "--rng-seed", "7", "--device", "cpu",
    ]
    main([*common, "--out", str(tmp_path / "a")])
    main([*common, "--out", str(tmp_path / "b")])

    np.testing.assert_array_equal(
        np.load(tmp_path / "a" / "samples.npy"), np.load(tmp_path / "b" / "samples.npy")
    )


@pytest.mark.integration
def test_cli_on_a_real_checkpoint(tmp_path):
    """The CLI runs on a real 192^2 EMA checkpoint produced by the trainer."""
    if not INTEGRATION_RUN.is_dir():
        pytest.skip(f"no real run at {INTEGRATION_RUN}")
    checkpoints = sorted((INTEGRATION_RUN / "checkpoints").glob("ema_iter_*.pt"))
    if not checkpoints:
        pytest.skip(f"no EMA checkpoint under {INTEGRATION_RUN}/checkpoints")
    out_dir = tmp_path / "out_real"

    code = main(
        [
            "--run", str(INTEGRATION_RUN), "--ckpt", checkpoints[-1].name, "--source", "seed",
            "--n-seeds", "2", "--n-per-seed", "2", "--batch", "4", "--out", str(out_dir),
        ]
    )

    assert code == 0
    record = _assert_artifacts(out_dir, n_seeds=2, n_per_seed=2, size=192)
    assert record["request"]["start_level"] == 200
    assert record["request"]["prior_noise"] is True
