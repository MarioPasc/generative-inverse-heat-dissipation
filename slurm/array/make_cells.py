"""Derive `slurm/array/cells.csv` from `configs.spectral.arms.EXPERIMENT_CELLS`.

`cells.csv` is the row table the SLURM worker indexes with `$SLURM_ARRAY_TASK_ID`, so it must
never drift from the config factory's own cell list. This module is the single place that maps
`EXPERIMENT_CELLS` (dataset-major, the order the factory documents its arms in) onto the tier
order of `docs/SPECIFICATIONS/M3-picasso/T3.3-full-array-submission.md` §1, which is what decides
which runs the queue starts first.

Run it (from the repository root) with::

    python slurm/array/make_cells.py --check     # fail if cells.csv is stale
    python slurm/array/make_cells.py --write     # regenerate cells.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

from configs.spectral.arms import EXPERIMENT_CELLS

__all__ = ["CELLS_CSV", "Cell", "build_cells", "render_csv"]


class CellTableError(Exception):
    """Raised when the derived cell table contradicts `EXPERIMENT_CELLS`."""


#: Where the generated table lives, relative to the repository root.
CELLS_CSV: Path = Path("slurm/array/cells.csv")

#: The CSV header, frozen: the worker reads fields by position.
HEADER: tuple[str, ...] = ("index", "run_id", "dataset_id", "arm", "seed", "tier")

#: Tier 1 and tier 2 are the development pair; tier 3 is the transfer pair (00-overview §1).
_DEVELOPMENT_DATASETS: frozenset[str] = frozenset({"ixi", "lsun_church"})

#: Sort precedence inside a tier: arm first, then dataset, then seed (ticket §1). One global arm
#: rank is enough because A0/A3 only ever meet each other and A1/A2/A2p only ever meet each other.
_ARM_RANK: dict[str, int] = {"A0": 0, "A3": 1, "A1": 2, "A2": 3, "A2p": 4}

#: Dataset precedence: the development pair before the transfer pair, MRI before photographs.
_DATASET_RANK: dict[str, int] = {"ixi": 0, "lsun_church": 1, "oasis1": 2, "lsun_bedroom": 3}

#: Arms that make a cell primary (the A0-versus-A3 contrast the plateau gate D10 is read from).
_PRIMARY_ARMS: frozenset[str] = frozenset({"A0", "A3"})


@dataclass(frozen=True, slots=True)
class Cell:
    """One training run of the array.

    Parameters
    ----------
    index
        The `--array` index, 0-based, assigned after the tier sort.
    run_id
        `f"{dataset_id}_{arm}_s{seed}"`, identical to `config.run_id` in `arms.py`.
    dataset_id
        One of `configs.spectral.arms.DATASETS`.
    arm
        One of `configs.spectral.arms.ARMS`.
    seed
        The training seed.
    tier
        1, 2 or 3; lower tiers occupy the lower indices and therefore start first.
    """

    index: int
    run_id: str
    dataset_id: str
    arm: str
    seed: int
    tier: int

    def as_row(self) -> tuple[str, ...]:
        """Return the cell as the CSV row the worker parses."""
        return (
            str(self.index),
            self.run_id,
            self.dataset_id,
            self.arm,
            str(self.seed),
            str(self.tier),
        )


def _tier_of(dataset_id: str, arm: str) -> int:
    """Return the tier a cell belongs to.

    Parameters
    ----------
    dataset_id
        The dataset id of the cell.
    arm
        The arm of the cell.

    Returns
    -------
    int
        1 for the primary arms on the development pair, 2 for the secondary arms on the
        development pair, 3 for every cell on the transfer pair.
    """
    if dataset_id not in _DEVELOPMENT_DATASETS:
        return 3
    return 1 if arm in _PRIMARY_ARMS else 2


def _sort_key(dataset_id: str, arm: str, seed: int) -> tuple[int, int, int, int]:
    """Return the total order of the array: tier, then arm, then dataset, then seed."""
    return (
        _tier_of(dataset_id, arm),
        _ARM_RANK[arm],
        _DATASET_RANK[dataset_id],
        seed,
    )


def build_cells() -> tuple[Cell, ...]:
    """Flatten `EXPERIMENT_CELLS` into the indexed, tier-ordered run table.

    Returns
    -------
    tuple of Cell
        One cell per `(dataset_id, arm, seed)` triple, indices `0 .. n-1` in tier order.

    Raises
    ------
    CellTableError
        If a dataset or arm of `EXPERIMENT_CELLS` has no rank here (a new dataset or arm was
        added to the factory and the tier map was not updated), or if a triple repeats.
    """
    triples: list[tuple[str, str, int]] = []
    for dataset_id, arm, seeds in EXPERIMENT_CELLS:
        if dataset_id not in _DATASET_RANK:
            raise CellTableError(f"dataset {dataset_id!r} has no rank in make_cells._DATASET_RANK")
        if arm not in _ARM_RANK:
            raise CellTableError(f"arm {arm!r} has no rank in make_cells._ARM_RANK")
        triples.extend((dataset_id, arm, int(seed)) for seed in seeds)

    if len(set(triples)) != len(triples):
        raise CellTableError("EXPERIMENT_CELLS contains a repeated (dataset, arm, seed) triple")

    triples.sort(key=lambda t: _sort_key(*t))
    return tuple(
        Cell(
            index=index,
            run_id=f"{dataset_id}_{arm}_s{seed}",
            dataset_id=dataset_id,
            arm=arm,
            seed=seed,
            tier=_tier_of(dataset_id, arm),
        )
        for index, (dataset_id, arm, seed) in enumerate(triples)
    )


def render_csv(cells: tuple[Cell, ...]) -> str:
    """Render the cell table as the exact CSV text of `cells.csv` (LF line endings)."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    for cell in cells:
        writer.writerow(cell.as_row())
    return buffer.getvalue()


def _repo_root() -> Path:
    """Return the repository root (this file lives at `<root>/slurm/array/make_cells.py`)."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Write or check `cells.csv`.

    Parameters
    ----------
    argv
        Command-line arguments; `sys.argv[1:]` when omitted.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 when `--check` finds a stale file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate cells.csv")
    group.add_argument("--check", action="store_true", help="fail if cells.csv is stale")
    args = parser.parse_args(argv)

    target = _repo_root() / CELLS_CSV
    text = render_csv(build_cells())

    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target} ({len(text.splitlines()) - 1} rows)")
        return 0

    if not target.is_file():
        print(f"{target} is missing; run make_cells.py --write", file=sys.stderr)
        return 1
    if target.read_text(encoding="utf-8") != text:
        print(f"{target} is stale; run `python slurm/array/make_cells.py --write`", file=sys.stderr)
        return 1
    print(f"{target} is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
