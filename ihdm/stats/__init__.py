"""Statistics of the experiment (``docs/SPECIFICATIONS/05-metrics.md`` §8 and §8a).

``bootstrap`` holds the percentile bootstrap, the seed-paired arm contrast, the interaction
estimand and the paired plateau gate of D10/D17; ``permutation`` holds the exact permutation test
of a two-cell contrast; ``tables`` turns the runs' ``summary.json`` files into the one table the
analysis works from.
"""

from ihdm.stats.bootstrap import (
    GateResult,
    Interaction,
    Interval,
    PairedDelta,
    bootstrap_ci,
    interaction,
    paired_delta,
    paired_lsd_gate,
)
from ihdm.stats.errors import StatsError
from ihdm.stats.permutation import PermutationResult, permutation_test
from ihdm.stats.tables import cell_table, flatten_summary

__all__ = [
    "GateResult",
    "Interaction",
    "Interval",
    "PairedDelta",
    "PermutationResult",
    "StatsError",
    "bootstrap_ci",
    "cell_table",
    "flatten_summary",
    "interaction",
    "paired_delta",
    "paired_lsd_gate",
    "permutation_test",
]
