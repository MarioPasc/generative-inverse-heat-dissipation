"""Tests of the exploratory readings (``ihdm.analysis.exploratory``) on the synthetic campaign."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ihdm.analysis.exploratory import (
    ENDPOINTS,
    OCTAVES,
    main,
    octave_summary,
    regeneration_ratio,
    run,
)
from tests.analysis.synthetic_eval import build_campaign
from tests.analysis.test_figures import collected_results
from tests.analysis.test_inherited import synthetic_constants


@pytest.mark.parametrize("errors", [
    np.zeros(8),
    np.full(8, -0.3),
    np.linspace(-1.5, 0.1, 8),
    np.array([0.02, -0.3, -0.4, -0.2, -0.1, -0.15, -0.12, -0.2]),
])
def test_octave_rms_splits_exactly_into_level_and_shape(errors):
    s = octave_summary(errors)
    np.testing.assert_allclose(s["oct_rms"] ** 2, s["oct_level"] ** 2 + s["oct_shape"] ** 2,
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(s["oct_rms_low"], np.sqrt(np.mean(errors[:3] ** 2)), rtol=1e-12)
    np.testing.assert_allclose(s["oct_rms_high"], np.sqrt(np.mean(errors[3:] ** 2)), rtol=1e-12)


def test_a_uniform_deficit_is_all_level():
    s = octave_summary(np.full(len(OCTAVES), -0.25))
    assert s["oct_shape"] == pytest.approx(0.0, abs=1e-15)
    assert s["oct_level"] == pytest.approx(-0.25)


@pytest.mark.parametrize("bad", [np.zeros(7), np.array([0.0] * 7 + [math.nan])])
def test_octave_summary_refuses_bad_input(bad):
    with pytest.raises(ValueError):
        octave_summary(bad)


def test_regeneration_ratio():
    # a chain that regenerates exactly the removed variance has I_w = I
    assert regeneration_ratio(0.08, 0.08) == pytest.approx(1.0)
    assert regeneration_ratio(0.66, 0.0034) == pytest.approx(0.34 / 0.9966)
    assert math.isnan(regeneration_ratio(math.nan, 0.1))
    with pytest.raises(ValueError):
        regeneration_ratio(0.5, 1.0)


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    root = tmp_path_factory.mktemp("expl")
    results = collected_results(build_campaign(root / "campaign"), root)
    constants = synthetic_constants(results, root / "constants.json")
    out = root / "out"
    return out, run(results, out, png_dpi=40, inherited_constants=constants)


def test_run_writes_the_report_json_and_figure(written):
    out, path = written
    assert path == out / "exploratory.md"
    for name in ("exploratory.md", "exploratory.json", "fx1_dispersion.png",
                 "fx1_dispersion.pdf"):
        assert (out / name).stat().st_size > 0, name
    text = path.read_text()
    assert "not pre-registered" in text
    assert text.count("**interaction**") == len(ENDPOINTS) - 1


def test_the_json_holds_every_run_and_endpoint(written):
    out, _ = written
    record = json.loads((out / "exploratory.json").read_text())
    assert len(record["per_run"]) == 30
    keys = {e["key"] for e in record["endpoints"]}
    assert keys == {e.key for e in ENDPOINTS}
    assert record["curve_stats"]["n_records"] == 30 * 12


def test_main_refuses_a_folder_that_is_not_results(tmp_path):
    assert main(["--results", str(tmp_path), "--out", str(tmp_path / "out")]) == 2
