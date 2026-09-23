"""The cell table of the analysis: one row per run, one column per scalar metric.

``05-metrics.md`` §8 works on cells ``(dataset, arm)`` of ``s`` seeds. Every statistic of this
package consumes a column of the table this module builds from the ``metrics/summary.json`` files
the evaluation array writes, so the analysis never re-reads a run directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ihdm.stats.errors import StatsError

__all__ = ["IDENTITY_COLUMNS", "cell_table", "flatten_summary"]

#: The columns that identify a run; they lead every table and are never treated as metrics.
IDENTITY_COLUMNS: tuple[str, ...] = ("run_id", "dataset", "arm", "seed")

#: Keys of a summary whose contents are provenance, not metrics, and are dropped from the table.
_DROPPED_PREFIXES: tuple[str, ...] = (
    "env",
    "sampling",
    "seed_lists",
    "checkpoint_sha256",
    "config_sha256",
    "git_sha",
    "dataset_sha256",
    "created",
)


def flatten_summary(summary: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten one ``summary.json`` into ``{dotted key: scalar}``.

    Nested mappings contribute dotted keys (``final.M``, ``final.inception.kid``); lists and
    strings are dropped, except the identity strings of :data:`IDENTITY_COLUMNS`, because a cell
    table holds one number per run and metric.

    Parameters
    ----------
    summary : Mapping[str, Any]
        The parsed summary of one run.
    prefix : str
        Prefix carried down the recursion.

    Returns
    -------
    dict[str, Any]
        The flat mapping.
    """
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        name = f"{prefix}{key}"
        if not prefix and key in _DROPPED_PREFIXES:
            continue
        if isinstance(value, Mapping):
            flat.update(flatten_summary(value, f"{name}."))
        elif isinstance(value, bool | int | float):
            flat[name] = value
        elif isinstance(value, str) and name in IDENTITY_COLUMNS:
            flat[name] = value
        elif value is None and name.split(".")[-1] in {"t_tau", "kid", "fid"}:
            flat[name] = float("nan")
    return flat


def cell_table(summaries: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Build the analysis table from the summaries of several runs.

    Parameters
    ----------
    summaries : Mapping[str, Mapping[str, Any]]
        ``{run_id: parsed summary.json}``. The identity columns are read from the summary's
        ``run`` block when it has one and from the mapping key otherwise.

    Returns
    -------
    pandas.DataFrame
        One row per run, sorted by ``(dataset, arm, seed)``, with
        ``run_id, dataset, arm, seed`` first and every scalar metric after them.

    Raises
    ------
    StatsError
        If ``summaries`` is empty or a summary carries no run identity.
    """
    if not summaries:
        raise StatsError("no summaries given")

    rows: list[dict[str, Any]] = []
    for run_id, summary in summaries.items():
        flat = flatten_summary(summary)
        identity = dict(summary.get("run", {})) if isinstance(summary.get("run"), Mapping) else {}
        row: dict[str, Any] = {
            "run_id": str(identity.get("run_id", run_id)),
            "dataset": identity.get("dataset"),
            "arm": identity.get("arm"),
            "seed": identity.get("seed"),
        }
        if row["dataset"] is None or row["arm"] is None:
            raise StatsError(f"{run_id}: the summary carries no dataset/arm identity")
        for key, value in flat.items():
            if key.startswith("run."):
                continue
            row.setdefault(key, value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    metrics = [column for column in frame.columns if column not in IDENTITY_COLUMNS]
    frame = frame[[*IDENTITY_COLUMNS, *sorted(metrics)]]
    return frame.sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)
