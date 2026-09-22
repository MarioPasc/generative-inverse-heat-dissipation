"""Unit tests of ``ihdm.spectral.schedules`` and of the frozen files in ``schedules/``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ihdm.paths import repo_root
from ihdm.spectral.power import power_law
from ihdm.spectral.schedules import (
    SIGMA_B_OCTAVE_LABELS,
    ScheduleError,
    ScheduleSpec,
    build_schedule,
    levels_per_octave,
    log_schedule,
    matched_schedule,
    per_level_removed,
    per_level_spread,
    residual_norms,
    save_schedule,
    validate_schedule,
)

FROZEN_NAMES = (
    "log_W2", "log_W8", "ixi_W2", "ixi_W8", "lsun_church_W2", "oasis1_W8", "lsun_bedroom_W2",
)


def _meta(spec: ScheduleSpec) -> dict[str, object]:
    """The non-derived half of a ``schedules.json`` entry."""
    return {
        "kind": spec.kind,
        "sigma_min": spec.sigma_min,
        "sigma_max": spec.sigma_max,
        "K": spec.K,
        "fitted_on": spec.fitted_on,
        "n_images": None,
        "images_sha256": None,
        "spread": None,
        "git_sha": "test",
    }


@pytest.mark.parametrize("sigma_max", [96.0, 24.0])
def test_log_schedule_equals_the_released_formula(sigma_max: float) -> None:
    """``s = [0] + exp(linspace(log 0.5, log sigma_max, K))``, exactly."""
    values = log_schedule(ScheduleSpec(kind="log", sigma_max=sigma_max))
    expected = np.concatenate(
        [[0.0], np.exp(np.linspace(np.log(0.5), np.log(sigma_max), 200))]
    )
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("sigma_max", [96.0, 24.0])
def test_schedule_contract(sigma_max: float) -> None:
    """Shape, dtype, endpoints and monotonicity of ``04-run-artifacts.md`` §1."""
    values = log_schedule(ScheduleSpec(kind="log", sigma_max=sigma_max))
    assert values.dtype == np.float64
    assert values.shape == (201,)
    assert values[0] == 0.0
    assert values[1] == 0.5
    assert values[200] == sigma_max
    assert np.all(np.diff(values) > 0)


def test_matched_schedule_equalises_the_removed_variance(white_power: np.ndarray) -> None:
    """On a white spectrum every step removes the same variance to within 1 %."""
    spec = ScheduleSpec(kind="matched", sigma_max=32.0, fitted_on="synthetic/train")
    values = matched_schedule(white_power, spec)
    removed = per_level_removed(white_power, values)
    assert removed.max() / removed.min() - 1.0 < 0.01


@pytest.mark.parametrize("alpha", [2.0, 3.5])
def test_matched_schedule_beats_the_log_schedule(alpha: float) -> None:
    """A matched schedule equalises the removed variance on its own spectrum.

    Levels 2 … K-1 carry exactly the same target. The last level does not: the march stops
    when the final heat time is within 1e-4 of ``t_last`` (the released procedure's
    criterion), and this port then pins ``s[K]`` to ``sigma_max`` exactly, as
    ``04-run-artifacts.md`` §1 requires. On the five frozen arrays that leaves a spread of
    1.002–1.005; on this stiffer synthetic spectrum it reaches 1.013.
    """
    power = power_law(64, alpha, 1.0)
    spec = ScheduleSpec(kind="matched", sigma_max=32.0, fitted_on="synthetic/train")
    matched = matched_schedule(power, spec)
    logged = log_schedule(ScheduleSpec(kind="log", sigma_max=32.0))
    removed = per_level_removed(power, matched)
    assert removed[:-1].max() / removed[:-1].min() < 1.0001
    assert per_level_spread(power, matched) < 1.05
    assert per_level_spread(power, matched) <= per_level_spread(power, logged)


@pytest.mark.parametrize("sigma_max", [32.0, 8.0])
def test_matched_schedule_keeps_its_own_endpoint(sigma_max: float) -> None:
    """D12: the march runs to the spec's terminal blur; nothing is truncated or rescaled."""
    power = power_law(64, 3.0, 1.0)
    values = matched_schedule(
        power, ScheduleSpec(kind="matched", sigma_max=sigma_max, fitted_on="synthetic/train")
    )
    assert values[0] == 0.0
    assert values[1] == 0.5
    assert values[200] == sigma_max
    assert np.all(np.diff(values) > 0)


def test_the_two_terminal_blurs_give_different_schedules() -> None:
    """The ``W/8`` array is not the ``W/2`` array truncated or rescaled."""
    power = power_law(64, 3.0, 1.0)
    w2 = matched_schedule(
        power, ScheduleSpec(kind="matched", sigma_max=32.0, fitted_on="synthetic/train")
    )
    w8 = matched_schedule(
        power, ScheduleSpec(kind="matched", sigma_max=8.0, fitted_on="synthetic/train")
    )
    assert not np.allclose(w8, w2[: len(w8)])
    assert not np.allclose(w8[1:], w2[1:] * (8.0 / 32.0))


def test_the_frozen_w8_arrays_are_not_the_w2_arrays_rescaled() -> None:
    """The same check on the committed IXI pair (D12, on the real fitting split)."""
    w2 = np.load(repo_root() / "schedules" / "ixi_W2.npy")
    w8 = np.load(repo_root() / "schedules" / "ixi_W8.npy")
    assert not np.allclose(w8[1:], w2[1:] * 0.25, rtol=1e-2)
    assert np.abs(w8[1:] / (w2[1:] * 0.25) - 1.0).max() > 0.1


@pytest.mark.parametrize("sigma_max", [96.0, 24.0])
def test_levels_per_octave_sums_to_k(sigma_max: float) -> None:
    """The eight counts cover the 200 levels of both terminal blurs."""
    counts = levels_per_octave(log_schedule(ScheduleSpec(kind="log", sigma_max=sigma_max)))
    assert list(counts) == list(SIGMA_B_OCTAVE_LABELS)
    assert sum(counts.values()) == 200
    if sigma_max == 24.0:
        assert counts["32-64"] == 0 and counts["64-96"] == 0


def test_levels_per_octave_rejects_a_level_outside_the_bins() -> None:
    """A schedule reaching past 96 px is not describable by the frozen bins."""
    values = np.concatenate([[0.0], np.linspace(0.5, 128.0, 200)])
    with pytest.raises(ScheduleError):
        levels_per_octave(values)


def test_log_schedule_spends_equal_levels_per_octave() -> None:
    """The paper's schedule is equal arc length in ``log sigma_B``."""
    counts = levels_per_octave(log_schedule(ScheduleSpec(kind="log", sigma_max=96.0)))
    full = [counts[b] for b in SIGMA_B_OCTAVE_LABELS[:7]]
    assert max(full) - min(full) <= 1


def test_residual_norms_rejects_bad_times() -> None:
    """Times must be a positive, strictly increasing 1-D array."""
    power = power_law(16, 2.0, 1.0)
    with pytest.raises(ScheduleError):
        residual_norms(power, np.array([]))
    with pytest.raises(ScheduleError):
        residual_norms(power, np.array([1.0, 0.5]))
    with pytest.raises(ScheduleError):
        residual_norms(power, np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "quadratic", "sigma_max": 96.0},
        {"kind": "log", "sigma_max": 0.25},
        {"kind": "log", "sigma_max": 96.0, "K": 0},
        {"kind": "matched", "sigma_max": 96.0},
        {"kind": "log", "sigma_max": 96.0, "fitted_on": "ixi/train"},
    ],
)
def test_schedule_spec_validates(kwargs: dict[str, object]) -> None:
    """Every malformed spec raises at construction."""
    with pytest.raises(ScheduleError):
        ScheduleSpec(**kwargs)  # type: ignore[arg-type]


def test_build_schedule_needs_a_spectrum_for_a_matched_arm() -> None:
    """``build_schedule`` refuses to invent a spectrum."""
    with pytest.raises(ScheduleError):
        build_schedule(ScheduleSpec(kind="matched", sigma_max=96.0, fitted_on="ixi/train"))


def test_save_and_validate_round_trip(tmp_path: Path) -> None:
    """A saved schedule validates, and its entry carries the hash and the octave table."""
    spec = ScheduleSpec(kind="log", sigma_max=96.0)
    values = log_schedule(spec)
    entry = save_schedule(tmp_path / "log_W2.npy", values, _meta(spec))
    (tmp_path / "schedules.json").write_text(json.dumps({"log_W2": entry}))
    assert validate_schedule(tmp_path / "log_W2.npy") == []
    assert entry["sha256"] == hashlib.sha256((tmp_path / "log_W2.npy").read_bytes()).hexdigest()
    assert sum(entry["levels_per_octave"].values()) == 200


def test_validate_schedule_reports_every_kind_of_violation(tmp_path: Path) -> None:
    """A missing file, a missing index, a wrong hash and a broken array are all reported."""
    absent = tmp_path / "absent.npy"
    assert validate_schedule(absent) == [f"{absent}: file not found"]

    spec = ScheduleSpec(kind="log", sigma_max=96.0)
    entry = save_schedule(tmp_path / "log_W2.npy", log_schedule(spec), _meta(spec))
    assert any("schedules.json" in m for m in validate_schedule(tmp_path / "log_W2.npy"))

    (tmp_path / "schedules.json").write_text(json.dumps({"log_W2": {**entry, "sha256": "0" * 64}}))
    assert any("sha256" in m for m in validate_schedule(tmp_path / "log_W2.npy"))

    (tmp_path / "schedules.json").write_text(json.dumps({"other": entry}))
    assert any("no entry" in m for m in validate_schedule(tmp_path / "log_W2.npy"))

    np.save(tmp_path / "broken.npy", np.arange(201, dtype=np.float32))
    problems = validate_schedule(tmp_path / "broken.npy")
    assert any("float64" in m for m in problems)
    assert any("s[1]" in m for m in problems)


def test_save_schedule_rejects_an_array_off_contract(tmp_path: Path) -> None:
    """``save_schedule`` never writes an array that would fail validation."""
    spec = ScheduleSpec(kind="log", sigma_max=96.0)
    with pytest.raises(ScheduleError):
        save_schedule(tmp_path / "bad.npy", np.linspace(0.0, 96.0, 201), _meta(spec))


def test_the_frozen_schedules_validate() -> None:
    """The seven committed files pass ``validate_schedule`` (``docs/HARNESSES/data.md`` §5)."""
    sched_dir = repo_root() / "schedules"
    found = sorted(p.stem for p in sched_dir.glob("*.npy"))
    assert found == sorted(FROZEN_NAMES)
    for name in FROZEN_NAMES:
        assert validate_schedule(sched_dir / f"{name}.npy") == []


def test_the_frozen_index_is_complete() -> None:
    """Every entry of ``schedules.json`` carries the fields of ``04-run-artifacts.md`` §1."""
    table = json.loads((repo_root() / "schedules" / "schedules.json").read_text())
    assert sorted(table) == sorted(FROZEN_NAMES)
    required = {
        "kind", "sigma_min", "sigma_max", "K", "fitted_on", "n_images", "images_sha256",
        "levels_per_octave", "spread", "sha256", "created", "git_sha",
    }
    for name, entry in table.items():
        assert required <= set(entry), name
        assert entry["sigma_min"] == 0.5
        assert entry["K"] == 200
        assert entry["sigma_max"] in (96.0, 24.0)
        assert sum(entry["levels_per_octave"].values()) == 200
        if entry["kind"] == "matched":
            assert entry["fitted_on"].endswith("/train")
            assert entry["n_images"] == 3200
            assert entry["spread"] < 1.01
        else:
            assert entry["fitted_on"] is None
