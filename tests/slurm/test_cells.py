"""Tests for the SLURM array cell table (`slurm/array/cells.csv`, T3.3).

The table is what `slurm/array/train_array.sbatch` indexes with `$SLURM_ARRAY_TASK_ID`, so a
silent divergence between it and `configs.spectral.arms.EXPERIMENT_CELLS` would train the wrong
cell under the right run id — the failure mode the `picasso-sbatch` skill calls out as the most
expensive one available. These tests pin the row count, the bijection with `EXPERIMENT_CELLS`,
the tier order and the freshness of the committed file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from configs.spectral.arms import EXPERIMENT_CELLS
from slurm.array.make_cells import HEADER, build_cells, render_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
CELLS_PATH = REPO_ROOT / "slurm" / "array" / "cells.csv"

#: The tier each `(dataset_id, arm)` pair belongs to, written out independently of `make_cells`
#: so the test is a specification and not a restatement of the implementation.
EXPECTED_TIER: dict[tuple[str, str], int] = {
    ("ixi", "A0"): 1,
    ("ixi", "A3"): 1,
    ("lsun_church", "A0"): 1,
    ("lsun_church", "A3"): 1,
    ("ixi", "A1"): 2,
    ("ixi", "A2"): 2,
    ("lsun_church", "A1"): 2,
    ("lsun_church", "A2"): 2,
    ("lsun_church", "A2p"): 2,
    ("oasis1", "A0"): 3,
    ("oasis1", "A3"): 3,
    ("lsun_bedroom", "A0"): 3,
    ("lsun_bedroom", "A3"): 3,
}

#: Indices 0-11, spelled out: the ticket's tier-1 order (A0 before A3, ixi before lsun_church,
#: seed ascending). The plateau gate D10 is read off index 0 and index 3, so this order is load
#: bearing and not merely cosmetic.
EXPECTED_TIER1_RUN_IDS: tuple[str, ...] = (
    "ixi_A0_s1",
    "ixi_A0_s2",
    "ixi_A0_s3",
    "lsun_church_A0_s1",
    "lsun_church_A0_s2",
    "lsun_church_A0_s3",
    "ixi_A3_s1",
    "ixi_A3_s2",
    "ixi_A3_s3",
    "lsun_church_A3_s1",
    "lsun_church_A3_s2",
    "lsun_church_A3_s3",
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    """Return `cells.csv` as a list of dictionaries, in file order."""
    with CELLS_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_file_exists_with_the_frozen_header() -> None:
    """The worker reads fields by position, so the header is a frozen contract."""
    assert CELLS_PATH.is_file(), f"{CELLS_PATH} is missing"
    with CELLS_PATH.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == HEADER


def test_thirty_rows(rows: list[dict[str, str]]) -> None:
    """The array is `--array=0-29`: exactly 30 rows, or the spec and the submission disagree."""
    assert len(rows) == 30
    assert len(rows) == sum(len(seeds) for _, _, seeds in EXPERIMENT_CELLS)


def test_indices_are_dense_and_zero_based(rows: list[dict[str, str]]) -> None:
    """Row `i` must carry index `i`; the worker trusts the index column, not the line number."""
    assert [int(row["index"]) for row in rows] == list(range(len(rows)))


def test_one_row_per_experiment_cell(rows: list[dict[str, str]]) -> None:
    """The table is a bijection with `EXPERIMENT_CELLS` expanded over its seeds."""
    expected = {
        (dataset_id, arm, seed)
        for dataset_id, arm, seeds in EXPERIMENT_CELLS
        for seed in seeds
    }
    got = [(row["dataset_id"], row["arm"], int(row["seed"])) for row in rows]
    assert len(got) == len(set(got)), "a (dataset, arm, seed) triple is repeated"
    assert set(got) == expected


def test_run_id_matches_the_config_factory(rows: list[dict[str, str]]) -> None:
    """`run_id` must equal `arms.get_config`'s `config.run_id`, or artefacts land elsewhere."""
    for row in rows:
        assert row["run_id"] == f"{row['dataset_id']}_{row['arm']}_s{int(row['seed'])}"


def test_tiers_are_correct_and_non_decreasing(rows: list[dict[str, str]]) -> None:
    """Lower indices must hold lower tiers, so the queue starts the primary cells first."""
    tiers = [int(row["tier"]) for row in rows]
    assert tiers == sorted(tiers), "tiers are interleaved; low indices no longer start tier 1"
    for row in rows:
        assert int(row["tier"]) == EXPECTED_TIER[(row["dataset_id"], row["arm"])]
    assert [tiers.count(t) for t in (1, 2, 3)] == [12, 10, 8]


def test_tier1_order_is_exactly_the_ticket_order(rows: list[dict[str, str]]) -> None:
    """Tier 1 order decides which runs the plateau gate can be evaluated on first."""
    assert tuple(row["run_id"] for row in rows[:12]) == EXPECTED_TIER1_RUN_IDS


def test_seeds_ascend_within_each_dataset_and_arm(rows: list[dict[str, str]]) -> None:
    """Seed 1 before 2 before 3 inside every `(dataset, arm)` block."""
    seen: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        seen.setdefault((row["dataset_id"], row["arm"]), []).append(int(row["seed"]))
    for key, seeds in seen.items():
        assert seeds == sorted(seeds), f"seeds out of order for {key}"


def test_committed_file_is_not_stale() -> None:
    """`cells.csv` must be byte-identical to what `make_cells.py --write` produces now."""
    assert CELLS_PATH.read_text(encoding="utf-8") == render_csv(build_cells())
