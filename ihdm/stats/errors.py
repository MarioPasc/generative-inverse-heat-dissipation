"""Exception hierarchy of the statistics subpackage."""

from __future__ import annotations

__all__ = ["StatsError"]


class StatsError(Exception):
    """Raised when a statistic is requested outside its domain of validity.

    Covers empty or non-finite samples, paired collections whose keys do not match, resample
    counts that are not positive, and image stacks handed to the paired gate that do not line up
    seed by seed.
    """
