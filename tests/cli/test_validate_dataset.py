"""Tests for the ``ihdm.cli.validate_dataset`` command."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ihdm.cli.validate_dataset import main, resolve_root


def test_valid_dataset_prints_ok_and_exits_zero(synthetic_dataset, capsys):
    code = main([str(synthetic_dataset)])
    out = capsys.readouterr().out

    assert code == 0
    assert out.startswith("OK synthetic: 64 images, ")
    n_train, n_ref, n_seed = (int(part) for part in out.split(", ")[1].strip().split("/"))
    # `train` and `ref` partition the dataset; `seed` is drawn from `ref`, so it is
    # counted twice on purpose (see `ihdm.data.format.split_by_subject`).
    assert n_train + n_ref == 64
    assert 0 < n_seed <= n_ref


def test_missing_root_reports_violations_and_exits_one(tmp_path, capsys):
    code = main([str(tmp_path / "absent")])
    out = capsys.readouterr().out

    assert code == 1
    assert "FAIL" in out
    assert "missing file: images.npy" in out


@pytest.mark.parametrize("removed", ["meta.json", "splits.json", "index.csv"])
def test_corrupt_dataset_exits_one(synthetic_dataset, tmp_path, capsys, removed):
    root = tmp_path / "copy"
    shutil.copytree(synthetic_dataset, root)
    (root / removed).unlink()

    code = main([str(root)])
    out = capsys.readouterr().out

    assert code == 1
    assert f"missing file: {removed}" in out


def test_bare_id_is_resolved_under_data_root(synthetic_dataset, monkeypatch):
    monkeypatch.setenv("IHDM_DATA_ROOT", str(synthetic_dataset.parent))
    assert resolve_root("synthetic") == synthetic_dataset


def test_unresolvable_spec_is_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("IHDM_DATA_ROOT", str(tmp_path))
    assert resolve_root("nowhere") == Path("nowhere")


def test_several_roots_fail_if_any_is_invalid(synthetic_dataset, tmp_path, capsys):
    code = main([str(synthetic_dataset), str(tmp_path / "absent")])
    out = capsys.readouterr().out

    assert code == 1
    assert "OK synthetic:" in out
    assert "FAIL" in out
