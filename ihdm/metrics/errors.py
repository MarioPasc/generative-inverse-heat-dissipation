"""Exception hierarchy of the metrics subpackage."""

from __future__ import annotations

__all__ = ["MetricError"]


class MetricError(Exception):
    """Raised when a metric is requested outside its domain of validity.

    Covers malformed image stacks (wrong rank, non-square images, mismatched shapes, too few
    images), non-finite input, degenerate spectra whose logarithm is undefined, and result
    objects that cannot be serialised under the JSON contract of ``05-metrics.md`` §9.
    """
