"""Metric definitions of the experiment (``docs/SPECIFICATIONS/05-metrics.md``).

``spectral`` holds the fidelity and mechanism endpoints — the fine radial profile, the
log-spectral distance with its octave profile, ``T_tau`` and the inherited band; ``io`` holds the
result-file contract of §9 that every metric module writes through.
"""

from ihdm.metrics.errors import MetricError
from ihdm.metrics.io import read_json, write_json
from ihdm.metrics.spectral import (
    InheritedResult,
    LsdResult,
    inherited_band,
    log_bin_centres,
    log_bin_edges,
    lsd,
    radial_log_profile,
    t_tau,
)

__all__ = [
    "InheritedResult",
    "LsdResult",
    "MetricError",
    "inherited_band",
    "log_bin_centres",
    "log_bin_edges",
    "lsd",
    "radial_log_profile",
    "read_json",
    "t_tau",
    "write_json",
]
