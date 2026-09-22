"""Exception hierarchy of :mod:`ihdm.sampling` (``02-engineering-practices.md`` §2)."""

from __future__ import annotations

__all__ = ["SamplingError"]


class SamplingError(Exception):
    """Raised when a checkpoint, a config, a seed source or a request is unusable."""
