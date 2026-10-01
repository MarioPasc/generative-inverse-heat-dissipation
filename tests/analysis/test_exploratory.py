"""Tests of the exploratory readings (``ihdm.analysis.exploratory``) on the synthetic campaign."""

from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest

from ihdm.analysis.exploratory import (
    ENDPOINTS,
    LATE_STEPS,
    OCTAVES,
    add_endpoints,
    curve_summary,
    main,
    octave_summary,
    regeneration_ratio,
    run,
)
from ihdm.analysis.tables import Results
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
    np.testing.assert_allclose(s["oct_rms_prior"], np.sqrt(np.mean(errors[:2] ** 2)), rtol=1e-12)
    np.testing.assert_allclose(s["oct_rms_mid"], np.sqrt(np.mean(errors[2:3] ** 2)), rtol=1e-12)
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


def test_add_endpoints_computes_known_values(tmp_path):
    """``add_endpoints`` on one hand-built run with non-constant, known inputs."""
    rid = "known_run"
    e = np.array([0.5, -1.5, 3.0, -2.0, 1.0, -1.0, 2.0, -0.5])
    final = {"lsd_octaves": dict(zip(OCTAVES, e.tolist(), strict=True)), "variance_ratio": 2.0}
    final_path = tmp_path / "runs" / rid / "final.json"
    final_path.parent.mkdir(parents=True)
    final_path.write_text(json.dumps(final))

    late_values = {"lsd_045000": 0.40, "lsd_050000": 0.35, "lsd_055000": 0.30,
                   "lsd_060000": 0.25}
    frame = pd.DataFrame([{"run_id": rid, "present": True, **late_values,
                          "inherited_within": 0.2, "inherited_predicted": 0.5}]
                         ).set_index("run_id", drop=False)
    results = Results(root=tmp_path, frame=frame, collection={}, gates=(), steps=LATE_STEPS)

    add_endpoints(results)
    row = results.frame.loc[rid]
    assert row["lsd_late"] == pytest.approx(np.mean(list(late_values.values())))
    assert row["log_vr"] == pytest.approx(math.log10(2.0))
    assert row["regen_ratio"] == pytest.approx((1 - 0.2) / (1 - 0.5))
    assert row["oct_rms_prior"] == pytest.approx(np.sqrt(np.mean(e[:2] ** 2)))
    assert row["oct_rms_mid"] == pytest.approx(np.sqrt(np.mean(e[2:3] ** 2)))
    assert row["oct_rms_high"] == pytest.approx(np.sqrt(np.mean(e[3:] ** 2)))


def test_curve_summary_computes_peak_and_within_run_correlations():
    """``curve_summary`` on a constructed, non-constant checkpoint frame (2 runs, 3 steps)."""
    data = {
        "r1": {"lsd": [0.10, 0.50, 0.30], "variance_ratio": [0.90, 0.50, 0.80],
               "oct_level": [-0.05, -0.30, -0.10], "oct_rms": [0.06, 0.35, 0.12]},
        "r2": {"lsd": [0.20, 0.60, 0.25], "variance_ratio": [0.85, 0.45, 0.75],
               "oct_level": [-0.08, -0.33, -0.09], "oct_rms": [0.09, 0.38, 0.11]},
    }
    steps = [0, 10_000, 20_000]
    rows = [{"run_id": rid, "dataset": "test_ds", "arm": "A0", "step": step,
            "lsd": series["lsd"][i], "variance_ratio": series["variance_ratio"][i],
            "oct_level": series["oct_level"][i], "oct_rms": series["oct_rms"][i]}
           for rid, series in data.items() for i, step in enumerate(steps)]
    ckpt = pd.DataFrame(rows)

    summary, stats = curve_summary(ckpt)
    row = summary.iloc[0]
    assert row["peak_step"] == 10_000
    assert row["lsd_peak"] == pytest.approx((0.50 + 0.60) / 2)
    assert row["variance_ratio_peak"] == pytest.approx((0.50 + 0.45) / 2)
    assert row["oct_level_peak"] == pytest.approx((-0.30 + -0.33) / 2)
    assert stats["n_records"] == 6

    # Ground truth computed independently of ``_within_run_corr``: demean each run's own
    # three-step series by its own mean, then take the pooled Pearson r over both runs.
    def demeaned(key: str, transform=lambda x: x) -> np.ndarray:
        parts = [np.array([transform(v) for v in series[key]]) for series in data.values()]
        return np.concatenate([part - part.mean() for part in parts])

    lsd_d = demeaned("lsd")
    abs_level_d = demeaned("oct_level", abs)
    abs_log_vr_d = demeaned("variance_ratio", lambda v: abs(math.log10(v)))
    expected_lsd_level = float(np.corrcoef(lsd_d, abs_level_d)[0, 1])
    expected_lsd_log_vr = float(np.corrcoef(lsd_d, abs_log_vr_d)[0, 1])
    assert stats["within_run_corr_lsd_abs_level"] == pytest.approx(expected_lsd_level)
    assert stats["within_run_corr_lsd_abs_log_vr"] == pytest.approx(expected_lsd_log_vr)


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


@pytest.fixture(scope="module")
def collected_root(pristine_campaign, tmp_path_factory):
    """The healthy 30-run campaign (``conftest.py``), already collected into one results folder."""
    return collected_results(pristine_campaign, tmp_path_factory.mktemp("expl_collected"))


def test_main_returns_1_on_a_non_finite_octave(collected_root, tmp_path, caplog):
    """A -inf in one run's final.json octaves is an AnalysisError, not a traceback (exit 1)."""
    corrupted = tmp_path / "results"
    shutil.copytree(collected_root, corrupted)
    run_id = sorted(p.name for p in (corrupted / "runs").iterdir())[0]
    final_path = corrupted / "runs" / run_id / "final.json"
    record = json.loads(final_path.read_text())
    record["lsd_octaves"]["2-4"] = float("-inf")
    final_path.write_text(json.dumps(record))

    with caplog.at_level("ERROR"):
        assert main(["--results", str(corrupted), "--out", str(tmp_path / "out")]) == 1
    assert run_id in caplog.text
