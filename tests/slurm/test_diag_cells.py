"""The M7 diagnostic cell table and launcher (``slurm/diag_train/``, T7.1)."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pytest

from configs.spectral.arms import DIAGNOSTIC_CELLS, EXPERIMENT_CELLS

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "slurm" / "diag_train"
LAUNCHER = DIAG / "submit_diag.sh"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_cells_are_the_diagnostic_cells_in_order() -> None:
    rows = _rows(DIAG / "cells.csv")
    expected = [(d, a, s) for d, a, seeds in DIAGNOSTIC_CELLS for s in seeds]
    assert [(r["dataset_id"], r["arm"], int(r["seed"])) for r in rows] == expected
    assert [int(r["index"]) for r in rows] == list(range(len(expected)))
    assert all(r["run_id"] == f"{r['dataset_id']}_{r['arm']}_s{r['seed']}" for r in rows)
    assert all(r["tier"] for r in rows)


def test_cells_share_the_production_header_and_no_run() -> None:
    production = REPO / "slurm" / "array" / "cells.csv"
    header = production.read_text().splitlines()[0]
    assert (DIAG / "cells.csv").read_text().splitlines()[0] == header
    diag_runs = {r["run_id"] for r in _rows(DIAG / "cells.csv")}
    assert not diag_runs & {r["run_id"] for r in _rows(production)}
    assert not {d for d, _, _ in DIAGNOSTIC_CELLS} & {d for d, _, _ in EXPERIMENT_CELLS}


def _launch(tmp_path: Path, extra: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    data = tmp_path / "data"
    for dataset, _, _ in DIAGNOSTIC_CELLS:
        (data / dataset).mkdir(parents=True, exist_ok=True)
        for name in ("images.npy", "index.csv", "splits.json", "meta.json"):
            (data / dataset / name).touch()
    env = {**os.environ, "IHDM_REPO_DIR": str(REPO), "IHDM_DATA_ROOT": str(data),
           "IHDM_RUN_ROOT": str(tmp_path / "runs"), "IHDM_LOGS_DIR": str(tmp_path / "logs"),
           "TIME_LIMIT": "01:00:00", **extra}
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True,
                          text=True, timeout=60, check=False)


def test_launcher_dry_run_names_both_cells_and_the_right_logs(tmp_path: Path) -> None:
    out = _launch(tmp_path, {}, "--dry-run")
    assert out.returncode == 0, out.stderr
    assert "cell 0,lsun_church_r128_A0_s1" in out.stdout
    assert "cell 1,lsun_church_n32k_A0_s1" in out.stdout
    assert "--array=0-1" in out.stdout and "--time=01:00:00" in out.stdout
    assert "--constraint=a100 --gres=gpu:1" in out.stdout
    assert "diag_train_%A_%a.out" in out.stdout and "N_ITERS=60000" in out.stdout
    assert "slurm/array/train_array.sbatch" in out.stdout.replace("diag_train/../", "")
    assert "[DRY-RUN] nothing submitted" in out.stdout


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        ({"TIME_LIMIT": ""}, "set TIME_LIMIT"),
        ({"IHDM_RUN_ROOT": "/mnt/home/users/tic_163_uma/mpascual/fscratch/runs/ihdm"},
         "production run root"),
        ({"ARRAY_SPEC": "0-2"}, "no row 2"),
    ],
)
def test_launcher_refuses_unsafe_requests(tmp_path: Path, extra: dict[str, str], needle: str):
    out = _launch(tmp_path, extra, "--dry-run")
    assert out.returncode != 0
    assert needle in out.stderr


def test_launcher_refuses_a_missing_dataset(tmp_path: Path) -> None:
    out = _launch(tmp_path, {"IHDM_DATA_ROOT": str(tmp_path / "nowhere")}, "--dry-run")
    assert out.returncode != 0 and "missing" in out.stderr
