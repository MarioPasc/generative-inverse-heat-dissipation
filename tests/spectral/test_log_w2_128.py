"""The post-freeze schedule ``log_W2_128`` of the M7 diagnostic (T7.1) and ``build_schedules --add``.

``log_W2_128`` is the A0 schedule of the 128² dataset ``lsun_church_r128``: K = 200 log-spaced
sigma_B from 0.5 to W/2 = 64 px, built by the same code path as ``log_W2`` (0.5 -> 96). Adding it
must leave the seven frozen entries of ``schedules.json`` byte-identical.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ihdm.cli import build_schedules
from ihdm.cli.build_schedules import ADDED_SCHEDULES, SCHEDULES, add
from ihdm.paths import repo_root
from ihdm.spectral.schedules import (
    ScheduleError,
    ScheduleSpec,
    log_schedule,
    save_schedule,
    validate_schedule,
)

BASE_COMMIT = "48363e27d180eb4cad5b3f83f18a58f18f9bcdba"
FROZEN = repo_root() / "schedules"


def _frozen_copy(tmp_path: Path) -> Path:
    """Copy the seven frozen arrays and the base ``schedules.json`` (without the added entry)."""
    out = tmp_path / "schedules"
    out.mkdir()
    for name, _ in SCHEDULES:
        shutil.copy(FROZEN / f"{name}.npy", out / f"{name}.npy")
    table = json.loads((FROZEN / "schedules.json").read_text())
    for name in ADDED_SCHEDULES:
        table.pop(name, None)
    (out / "schedules.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return out


def _formula(sigma_max: float, k: int = 200) -> np.ndarray:
    """The released log schedule, ``04-run-artifacts.md`` §1."""
    return np.concatenate([[0.0], np.exp(np.linspace(np.log(0.5), np.log(sigma_max), k))])


def test_the_added_spec_is_the_w2_log_schedule_of_a_128_image() -> None:
    assert ADDED_SCHEDULES == {"log_W2_128": ScheduleSpec(kind="log", sigma_max=64.0)}
    assert dict(SCHEDULES)["log_W2"] == ScheduleSpec(kind="log", sigma_max=96.0)


def test_add_appends_one_entry_and_keeps_every_other_byte(tmp_path: Path) -> None:
    out = _frozen_copy(tmp_path)
    before = (out / "schedules.json").read_text()

    entry = add("log_W2_128", out)

    table = json.loads((out / "schedules.json").read_text())
    assert sorted(table) == sorted([name for name, _ in SCHEDULES] + ["log_W2_128"])
    assert table["log_W2_128"] == entry
    rest = {name: value for name, value in table.items() if name != "log_W2_128"}
    assert json.dumps(rest, indent=2, sort_keys=True) + "\n" == before
    assert validate_schedule(out / "log_W2_128.npy") == []
    for name, _ in SCHEDULES:
        assert validate_schedule(out / f"{name}.npy") == []


def test_add_builds_the_released_formula_with_the_log_w2_code_path(tmp_path: Path) -> None:
    out = _frozen_copy(tmp_path)
    entry = add("log_W2_128", out)
    values = np.load(out / "log_W2_128.npy")

    assert values.dtype == np.float64 and values.shape == (201,)
    assert values[0] == 0.0 and values[1] == 0.5 and values[200] == 64.0
    np.testing.assert_array_equal(values, log_schedule(ScheduleSpec(kind="log", sigma_max=64.0)))
    np.testing.assert_allclose(values, _formula(64.0), rtol=1e-14, atol=0.0)
    assert entry["kind"] == "log" and entry["fitted_on"] is None and entry["spread"] is None
    assert entry["sigma_min"] == 0.5 and entry["sigma_max"] == 64.0 and entry["K"] == 200
    # Equal levels per octave: 7 octaves from 0.5 to 64, 199 gaps, the last level alone in 64-96.
    assert entry["levels_per_octave"]["64-96"] == 1
    assert sum(entry["levels_per_octave"].values()) == 200


def test_add_refuses_a_duplicate_and_an_unknown_name(tmp_path: Path) -> None:
    out = _frozen_copy(tmp_path)
    add("log_W2_128", out)
    with pytest.raises(ScheduleError, match="already"):
        add("log_W2_128", out)
    add("log_W2_128", out, force=True)
    with pytest.raises(ScheduleError, match="unknown"):
        add("log_W2_256", out)


def test_cli_add_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = _frozen_copy(tmp_path)
    assert build_schedules.main(["--out", str(out), "--add", "log_W2_128"]) == 0
    first = (out / "schedules.json").read_bytes()
    assert build_schedules.main(["--out", str(out), "--add", "log_W2_128"]) == 0
    assert "already added" in capsys.readouterr().out
    assert (out / "schedules.json").read_bytes() == first


@pytest.mark.parametrize("sigma_max", [64.0, 96.0, 24.0])
def test_validate_schedule_accepts_the_three_terminal_blurs(tmp_path: Path, sigma_max: float):
    spec = ScheduleSpec(kind="log", sigma_max=sigma_max)
    meta = {"kind": "log", "sigma_min": 0.5, "sigma_max": sigma_max, "K": 200, "fitted_on": None,
            "n_images": None, "images_sha256": None, "spread": None, "git_sha": "test"}
    entry = save_schedule(tmp_path / "s.npy", log_schedule(spec), meta)
    (tmp_path / "schedules.json").write_text(json.dumps({"s": entry}))
    assert validate_schedule(tmp_path / "s.npy") == []


def test_save_schedule_still_rejects_another_terminal_blur(tmp_path: Path) -> None:
    with pytest.raises(ScheduleError, match="s\\[K\\]"):
        save_schedule(tmp_path / "s.npy", log_schedule(ScheduleSpec(kind="log", sigma_max=50.0)),
                      {})


def test_the_committed_log_w2_128_is_the_formula() -> None:
    values = np.load(FROZEN / "log_W2_128.npy")
    np.testing.assert_array_equal(values, log_schedule(ScheduleSpec(kind="log", sigma_max=64.0)))
    assert validate_schedule(FROZEN / "log_W2_128.npy") == []


def _base_bytes(path: str) -> bytes | None:
    """Return ``path`` at the base commit, or ``None`` outside a git checkout of this history."""
    try:
        out = subprocess.run(["git", "show", f"{BASE_COMMIT}:{path}"], cwd=repo_root(),
                             capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def test_the_committed_files_keep_the_base_entries_byte_identical() -> None:
    base = _base_bytes("schedules/schedules.json")
    if base is None:
        pytest.skip("the base commit is not reachable from this checkout")
    table = json.loads((FROZEN / "schedules.json").read_text())
    rest = {name: value for name, value in table.items() if name not in ADDED_SCHEDULES}
    assert (json.dumps(rest, indent=2, sort_keys=True) + "\n").encode() == base
    for name, _ in SCHEDULES:
        assert (FROZEN / f"{name}.npy").read_bytes() == _base_bytes(f"schedules/{name}.npy")
