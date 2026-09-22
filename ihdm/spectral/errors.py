"""Exception hierarchy of the spectral subpackage."""

from __future__ import annotations

__all__ = ["SpectralError", "ScheduleError"]


class SpectralError(Exception):
    """Raised when a spectral quantity is requested outside its domain of validity."""


class ScheduleError(SpectralError):
    """Raised when a blur schedule cannot be built, saved or validated."""
