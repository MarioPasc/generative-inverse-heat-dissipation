"""Tests of ``ihdm.stats.tables``: the flattening of a summary and the cell table."""

from __future__ import annotations

import pytest

from ihdm.stats.errors import StatsError
from ihdm.stats.tables import IDENTITY_COLUMNS, cell_table, flatten_summary


def _summary(run_id: str, dataset: str, arm: str, seed: int, lsd: float) -> dict:
    """Return a minimal summary in the shape ``evaluate_run`` writes."""
    return {
        "run": {"run_id": run_id, "dataset": dataset, "arm": arm, "seed": seed},
        "git_sha": "deadbeef",
        "created": "2026-09-23T00:00:00+00:00",
        "lsd_by_step": {"5000": lsd + 0.1, "10000": lsd},
        "t_tau": None,
        "final": {
            "lsd": lsd,
            "M": 1.02,
            "diversity_pix": 3.9e-3,
            "inception": {"kid": 0.004, "fid": 31.2, "n_reference": 800},
            "lsd_octaves": {"0.5-1": 0.2, "1-2": -0.1},
        },
    }


def test_flatten_summary_produces_dotted_scalar_keys():
    """Nested mappings become dotted keys and provenance strings are dropped."""
    flat = flatten_summary(_summary("ixi_A0_s1", "ixi", "A0", 1, 0.20))
    assert flat["final.lsd"] == pytest.approx(0.20)
    assert flat["final.inception.kid"] == pytest.approx(0.004)
    assert flat["final.lsd_octaves.0.5-1"] == pytest.approx(0.2)
    assert flat["lsd_by_step.5000"] == pytest.approx(0.30)
    assert "git_sha" not in flat
    assert "created" not in flat


def test_flatten_summary_maps_a_missing_t_tau_to_nan():
    """``t_tau`` is ``null`` when the A0 threshold was not supplied; the table wants a number."""
    flat = flatten_summary(_summary("ixi_A0_s1", "ixi", "A0", 1, 0.20))
    assert flat["t_tau"] != flat["t_tau"]  # nan


def test_cell_table_has_the_identity_columns_first_and_is_sorted():
    """One row per run, identity first, metrics sorted, ordered by (dataset, arm, seed)."""
    summaries = {
        "ixi_A3_s2": _summary("ixi_A3_s2", "ixi", "A3", 2, 0.18),
        "ixi_A0_s1": _summary("ixi_A0_s1", "ixi", "A0", 1, 0.20),
        "lsun_church_A0_s1": _summary("lsun_church_A0_s1", "lsun_church", "A0", 1, 0.11),
    }
    frame = cell_table(summaries)

    assert list(frame.columns[: len(IDENTITY_COLUMNS)]) == list(IDENTITY_COLUMNS)
    assert len(frame) == 3
    assert frame["run_id"].tolist() == ["ixi_A0_s1", "ixi_A3_s2", "lsun_church_A0_s1"]
    assert frame.loc[0, "final.lsd"] == pytest.approx(0.20)
    assert frame.loc[2, "final.inception.fid"] == pytest.approx(31.2)
    metrics = [column for column in frame.columns if column not in IDENTITY_COLUMNS]
    assert metrics == sorted(metrics)


def test_cell_table_supports_a_paired_contrast_end_to_end():
    """The table is what ``paired_delta`` consumes: one column, keyed by seed within a cell."""
    from ihdm.stats.bootstrap import paired_delta

    summaries = {
        f"ixi_{arm}_s{seed}": _summary(f"ixi_{arm}_s{seed}", "ixi", arm, seed, value)
        for arm, values in (("A0", (0.30, 0.31, 0.29)), ("A3", (0.20, 0.22, 0.21)))
        for seed, value in enumerate(values, start=1)
    }
    frame = cell_table(summaries)
    cells = {
        arm: dict(zip(group["seed"], group["final.lsd"], strict=True))
        for arm, group in frame.groupby("arm")
    }
    delta = paired_delta(cells["A3"], cells["A0"], n_boot=500, rng_seed=0)
    assert delta.seeds == (1, 2, 3)
    assert delta.mean < 0.0
    assert delta.interval.high < 0.0


def test_cell_table_rejects_an_empty_mapping_and_a_summary_without_identity():
    """A table without runs, or a run without a dataset/arm, is a caller bug."""
    with pytest.raises(StatsError):
        cell_table({})
    with pytest.raises(StatsError, match="identity"):
        cell_table({"x": {"run": {"run_id": "x"}, "final": {"lsd": 0.1}}})
