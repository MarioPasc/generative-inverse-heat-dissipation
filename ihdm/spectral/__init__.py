"""Spectral profile of the datasets and the frozen blur schedules of the arms.

``power`` holds the per-mode variance and the radial quantities of
``docs/SPECIFICATIONS/05-metrics.md`` §1; ``schedules`` builds, freezes and validates the
schedule arrays of ``docs/SPECIFICATIONS/04-run-artifacts.md`` §1; ``profile`` assembles the
dataset tables, the known-results checklist and the report's data figure.
"""

from ihdm.spectral.errors import ScheduleError, SpectralError
from ihdm.spectral.power import (
    eigenvalues,
    fit_alpha,
    inherited_share,
    mode_power,
    octave_bins,
    octave_shares,
    power_law,
    radial_profile,
    radial_spectrum,
)
from ihdm.spectral.schedules import (
    ScheduleSpec,
    build_schedule,
    levels_per_octave,
    log_schedule,
    matched_schedule,
    per_level_spread,
    residual_norms,
    save_schedule,
    validate_schedule,
)

__all__ = [
    "ScheduleError",
    "ScheduleSpec",
    "SpectralError",
    "build_schedule",
    "eigenvalues",
    "fit_alpha",
    "inherited_share",
    "levels_per_octave",
    "log_schedule",
    "matched_schedule",
    "mode_power",
    "octave_bins",
    "octave_shares",
    "per_level_spread",
    "power_law",
    "radial_profile",
    "radial_spectrum",
    "residual_norms",
    "save_schedule",
    "validate_schedule",
]
