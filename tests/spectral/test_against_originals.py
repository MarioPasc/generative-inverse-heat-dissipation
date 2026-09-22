"""Integration test: the ported functions against the TFM analysis scripts they came from.

Skipped unless ``projects/GenAI/analysis/`` is on this machine. The originals are loaded by
path, never imported by the library code.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from ihdm.spectral.power import fit_alpha, inherited_share, mode_power, octave_shares, power_law
from ihdm.spectral.schedules import ScheduleSpec, matched_schedule, residual_norms

ANALYSIS = Path("/home/mpascual/research/code/TFM/projects/GenAI/analysis")
pytestmark = pytest.mark.integration


def _original(name: str) -> ModuleType:
    """Import one analysis module by path, or skip the test."""
    if not (ANALYSIS / f"{name}.py").is_file():
        pytest.skip(f"the TFM analysis scripts are not on this machine ({ANALYSIS})")
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - depends on the machine's environment
        pytest.skip(f"{name} cannot be imported here: {exc}")


@pytest.fixture(scope="module")
def field() -> np.ndarray:
    """A random stack the two implementations are compared on."""
    return np.random.default_rng(11).normal(size=(40, 32, 32))


def test_mode_power_matches_the_original(field: np.ndarray) -> None:
    """Up to the zeroed DC mode, the port equals ``delta_star_from_psd.mode_power``."""
    reference = _original("delta_star_from_psd").mode_power(field)
    reference[0, 0] = 0.0
    np.testing.assert_allclose(mode_power(field), reference, rtol=1e-6, atol=1e-12)


def test_fit_alpha_matches_the_original(field: np.ndarray) -> None:
    """The ported exponent fit equals ``control_profile.fit_alpha``."""
    power = mode_power(field)
    reference = _original("control_profile").fit_alpha(power)
    assert fit_alpha(power) == pytest.approx(reference, rel=1e-6)


def test_inherited_share_matches_the_original(field: np.ndarray) -> None:
    """The ported inherited share equals ``knob_evidence.inherited_share``."""
    power = mode_power(field)
    reference = _original("knob_evidence").inherited_share(power, 16.0)
    assert inherited_share(power, 16.0) == pytest.approx(reference, rel=1e-6)


def test_octave_shares_match_the_original_on_the_shared_bins() -> None:
    """The seven shared bins hold the same variance; only the normalisation differs.

    The original closes its last bin at the DCT radius ``N - 1`` (95.5 cycles per image) and
    this port at ``N`` (96, ``05-metrics.md`` §1), so the totals the shares are divided by
    differ by the annulus ``191 <= n <= 192``. The band *contents* are identical, which is
    what the ratios below check; the normalisation difference is then measured on a realistic
    ``1/f^2`` spectrum, where that annulus is negligible.
    """
    power = mode_power(np.random.default_rng(12).normal(size=(40, 192, 192)))
    reference = dict(_original("delta_star_from_psd").octave_shares(power))
    mine = octave_shares(power)
    labels = list(reference)[:-1]
    for label in labels:
        assert mine[label] / mine["8-16"] == pytest.approx(
            reference[label] / reference["8-16"], rel=1e-9
        ), label

    smooth = power_law(192, 2.0, 1.0)
    reference_smooth = dict(_original("delta_star_from_psd").octave_shares(smooth))
    mine_smooth = octave_shares(smooth)
    for label in labels:
        assert mine_smooth[label] == pytest.approx(reference_smooth[label], rel=2e-3), label


def test_residual_norms_match_the_original(field: np.ndarray) -> None:
    """The ported residual norms equal ``variance_matched_schedule.residual_norms``."""
    original = _original("variance_matched_schedule")
    power = mode_power(field)
    times = np.exp(np.linspace(np.log(0.5), np.log(16.0), 200)) ** 2 / 2
    np.testing.assert_allclose(
        residual_norms(power, times), original.residual_norms(power, times), rtol=1e-6
    )


def test_matched_schedule_matches_the_original(field: np.ndarray) -> None:
    """The ported march equals the original everywhere except the pinned terminal blur.

    The original stops when the last heat time is within 1e-3 of ``t_last``; the frozen
    contract requires ``s[K]`` to be exactly ``sigma_max``, so this port assigns the endpoint
    after the march. Levels 1 … K-1 are compared at ``rtol = 1e-6``; the last level is checked
    against the contract instead.
    """
    original = _original("variance_matched_schedule")
    power = mode_power(field)
    reference = original.matched_schedule(power, 32)
    mine = matched_schedule(
        power, ScheduleSpec(kind="matched", sigma_max=16.0, fitted_on="synthetic/train")
    )
    np.testing.assert_allclose(mine[1:-1], reference[:-1], rtol=1e-6)
    assert mine[-1] == 16.0
    assert reference[-1] == pytest.approx(16.0, rel=1e-3)
