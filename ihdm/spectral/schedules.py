"""Blur schedules: the log schedule of the paper and the variance-matched schedules.

Frozen contract: ``docs/SPECIFICATIONS/04-run-artifacts.md`` §1. A schedule is a ``float64``
array of shape ``(K + 1,)`` holding ``sigma_{B,k}`` in pixels, with ``s[0] = 0`` (the unblurred
level), ``s[1] = sigma_min = 0.5`` and ``s[K] = sigma_max in {96.0, 24.0}``, strictly
increasing. The heat time of level ``k`` is ``t_k = s[k]^2 / 2``
(``model_code/utils.py: DCTBlur.forward``).

The matched schedule places the ``K`` levels so that each step removes the same between-image
variance of the fitting split: with ``d_{k,i} = exp(-lambda_i t_k)`` the data-dependent part of
the IHDM regression target at level ``k`` is ``R_k = sum_i (d_{k-1,i} - d_{k,i})^2 P_i``
(Rissanen et al., ICLR 2023, Eq. 9 with the skip parameterisation), and the matching makes
``R_2 = … = R_K``. Decision D12: the march is run separately for each terminal blur, with the
spec's own ``sigma_max`` as the endpoint; a schedule is never truncated or rescaled from
another one.

Ported, with attribution, from ``projects/GenAI/analysis/variance_matched_schedule.py``
(``residual_norms``, ``_march``, ``matched_schedule``) and
``projects/GenAI/analysis/delta_star_from_psd.py`` (``blur_schedule``) of the TFM knowledge
base, and from ``knob_evidence.levels_per_octave``. Those modules are never imported at
runtime; ``tests/spectral/test_against_originals.py`` checks the port against them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ihdm.spectral.errors import ScheduleError

__all__ = [
    "ScheduleError",
    "ScheduleSpec",
    "SIGMA_B_OCTAVE_EDGES",
    "SIGMA_B_OCTAVE_LABELS",
    "TERMINAL_BLURS",
    "log_schedule",
    "matched_schedule",
    "build_schedule",
    "residual_norms",
    "per_level_spread",
    "per_level_removed",
    "levels_per_octave",
    "save_schedule",
    "validate_schedule",
    "git_sha",
]

logger = logging.getLogger(__name__)

# levels_per_octave is reported on the sigma_B *pixel* octaves of learning/03 §16.3, closed at
# the largest terminal blur of the design.
SIGMA_B_OCTAVE_EDGES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 96.0)
SIGMA_B_OCTAVE_LABELS: tuple[str, ...] = tuple(
    f"{lo:g}-{hi:g}"
    for lo, hi in zip(SIGMA_B_OCTAVE_EDGES[:-1], SIGMA_B_OCTAVE_EDGES[1:], strict=True)
)
TERMINAL_BLURS: tuple[float, ...] = (96.0, 24.0)

_OUTER_ITERATIONS = 80  # geometric bisections on the per-level target
_INNER_ITERATIONS = 60  # bisections per level, on the heat time


@dataclass(frozen=True)
class ScheduleSpec:
    """Everything that identifies one frozen schedule array.

    The field order differs from the ticket's prose (``kind, sigma_min, sigma_max, K,
    fitted_on``) because a defaulted field cannot precede a required one; every call site uses
    keywords.

    Parameters
    ----------
    kind : {"log", "matched"}
        ``"log"`` is the paper's log-spaced schedule; ``"matched"`` is the constant-arc-length
        schedule of a fitting split.
    sigma_max : float
        Terminal blur length-scale in pixels (96.0 for ``W/2``, 24.0 for ``W/8``).
    sigma_min : float
        Blur of level 1, in pixels.
    K : int
        Number of levels; the array has ``K + 1`` entries.
    fitted_on : str or None
        ``"<dataset_id>/train"`` for a matched schedule, ``None`` for a log schedule.
    """

    kind: Literal["log", "matched"]
    sigma_max: float
    sigma_min: float = 0.5
    K: int = 200
    fitted_on: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("log", "matched"):
            raise ScheduleError(f"unknown schedule kind {self.kind!r}")
        if self.K < 1:
            raise ScheduleError(f"K must be at least 1, got {self.K}")
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ScheduleError(
                f"need 0 < sigma_min < sigma_max, got {self.sigma_min} and {self.sigma_max}"
            )
        if self.kind == "matched" and self.fitted_on is None:
            raise ScheduleError("a matched schedule must record the split it was fitted on")
        if self.kind == "log" and self.fitted_on is not None:
            raise ScheduleError("a log schedule is not fitted on any split")


def _with_endpoints(sigmas: np.ndarray, spec: ScheduleSpec) -> np.ndarray:
    """Prepend level 0 and pin the endpoints exactly, so no drift can break the contract."""
    out = np.empty(spec.K + 1, dtype=np.float64)
    out[0] = 0.0
    out[1:] = sigmas
    out[1] = spec.sigma_min
    out[spec.K] = spec.sigma_max
    if not np.all(np.diff(out) > 0):
        raise ScheduleError("the constructed schedule is not strictly increasing")
    return out


def log_schedule(spec: ScheduleSpec) -> np.ndarray:
    """The paper's log-spaced schedule (App. B.5 of Rissanen et al., ICLR 2023).

    Exactly ``concatenate([[0], exp(linspace(log(sigma_min), log(sigma_max), K))])``, which is
    ``delta_star_from_psd.blur_schedule`` with the terminal blur made explicit.

    Parameters
    ----------
    spec : ScheduleSpec
        Must have ``kind == "log"``.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(K + 1,)``.

    Raises
    ------
    ScheduleError
        If ``spec.kind`` is not ``"log"``.
    """
    if spec.kind != "log":
        raise ScheduleError(f"log_schedule called with kind={spec.kind!r}")
    sigmas = np.exp(np.linspace(np.log(spec.sigma_min), np.log(spec.sigma_max), spec.K))
    return _with_endpoints(sigmas, spec)


def _grouped(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group the modes by their eigenvalue and sum their power (the DC mode is dropped).

    ``lambda`` depends on the mode only through ``i^2 + j^2``, so the marching sums are
    unchanged (up to summation order) while the inner loop shrinks by roughly a factor of
    three on the 192-grid.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(lambda, power)`` as 1-D ``float64`` arrays over the distinct eigenvalues.
    """
    n_pix = power.shape[0]
    idx = np.arange(n_pix)
    squared = (idx[:, None] ** 2 + idx[None, :] ** 2).ravel()
    weights = np.asarray(power, dtype=np.float64).copy()
    weights[0, 0] = 0.0
    unique, inverse = np.unique(squared, return_inverse=True)
    grouped = np.bincount(inverse, weights=weights.ravel(), minlength=unique.size)
    return np.pi**2 * unique / n_pix**2, grouped


def residual_norms(power: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Data-dependent part of the expected squared regression target at each level.

    Ported from ``variance_matched_schedule.residual_norms``.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``. The DC entry is ignored.
    times : numpy.ndarray
        Heat times ``t_k`` of length ``K``, increasing, index 0 being level ``k = 1``.

    Returns
    -------
    numpy.ndarray
        Array of length ``K``; entry ``k`` is ``sum_i (d_{k-1,i} - d_{k,i})^2 P_i`` with
        ``d_0 = 1``.

    Raises
    ------
    ScheduleError
        If ``times`` is not a 1-D array of positive, increasing values.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ScheduleError("times must be a non-empty 1-D array")
    if not np.all(np.diff(times) > 0) or times[0] <= 0:
        raise ScheduleError("times must be positive and strictly increasing")
    lam, weights = _grouped(power)
    decay = np.exp(-lam[None, :] * times[:, None])
    decay_prev = np.concatenate([np.ones((1, lam.size)), decay[:-1]])
    return ((decay_prev - decay) ** 2 * weights).sum(axis=1)


def _march(
    weights: np.ndarray,
    lam: np.ndarray,
    t_first: float,
    t_last: float,
    target: float,
    k_steps: int,
) -> tuple[np.ndarray, bool]:
    """Advance ``k_steps - 1`` levels, each accumulating ``target`` of residual norm.

    Ported from ``variance_matched_schedule._march``; the iteration counts are the original's
    so the two agree to machine precision.

    Parameters
    ----------
    weights, lam : numpy.ndarray
        Grouped power and eigenvalues, as returned by :func:`_grouped`.
    t_first, t_last : float
        Heat times of the first and last level.
    target : float
        Residual norm each step must accumulate.
    k_steps : int
        Number of levels to place.

    Returns
    -------
    tuple[numpy.ndarray, bool]
        The heat times, and whether the march placed every level before reaching ``t_last``.
    """
    buf = np.empty_like(lam)

    def accumulated(decay_prev: np.ndarray, t: float) -> float:
        np.multiply(lam, -t, out=buf)
        np.exp(buf, out=buf)
        np.subtract(decay_prev, buf, out=buf)
        np.square(buf, out=buf)
        return float(np.dot(buf, weights))

    decay_prev = np.exp(-lam * t_first)
    times = [t_first]
    for _ in range(k_steps - 1):
        if accumulated(decay_prev, t_last) < target:
            return np.array(times + [t_last] * (k_steps - len(times))), False
        low, high = times[-1], t_last
        for _ in range(_INNER_ITERATIONS):
            mid = 0.5 * (low + high)
            if accumulated(decay_prev, mid) < target:
                low = mid
            else:
                high = mid
        times.append(0.5 * (low + high))
        decay_prev = np.exp(-lam * times[-1])
    return np.array(times), True


def matched_schedule(power: np.ndarray, spec: ScheduleSpec) -> np.ndarray:
    """Blur length-scales whose per-level regression target is constant.

    The endpoints are ``spec.sigma_min`` and ``spec.sigma_max``; only the spacing between them
    changes. The constant target is found by 80 geometric bisections, each running the march
    of :func:`_march`. Ported from ``variance_matched_schedule.matched_schedule``, whose
    endpoints were fixed at ``0.5`` and ``n_pix / 2``; D12 requires the march to be re-run for
    each terminal blur, which is what the spec's ``sigma_max`` provides.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)`` of the fitting split.
    spec : ScheduleSpec
        Must have ``kind == "matched"``.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(K + 1,)``.

    Raises
    ------
    ScheduleError
        If ``spec.kind`` is not ``"matched"``, or the bisection does not reach the terminal
        blur.
    """
    if spec.kind != "matched":
        raise ScheduleError(f"matched_schedule called with kind={spec.kind!r}")
    lam, weights = _grouped(power)
    if not weights.sum() > 0.0:
        raise ScheduleError("the fitting split has no non-DC variance")
    t_first, t_last = spec.sigma_min**2 / 2, spec.sigma_max**2 / 2

    low = 1e-18
    high = float(((np.exp(-lam * t_first) - np.exp(-lam * t_last)) ** 2 * weights).sum())
    for _ in range(_OUTER_ITERATIONS):
        target = np.sqrt(low * high)
        times, complete = _march(weights, lam, t_first, t_last, target, spec.K)
        if not complete or times[-1] >= t_last * 0.9999:
            high = target
        else:
            low = target
    times, complete = _march(weights, lam, t_first, t_last, high, spec.K)
    if not complete or abs(times[-1] / t_last - 1) > 1e-3:
        raise ScheduleError(
            f"bisection did not converge to the terminal blur sigma_max={spec.sigma_max}"
        )
    return _with_endpoints(np.sqrt(2 * times), spec)


def build_schedule(spec: ScheduleSpec, power: np.ndarray | None = None) -> np.ndarray:
    """Build the schedule a spec describes.

    Parameters
    ----------
    spec : ScheduleSpec
        The schedule to build.
    power : numpy.ndarray or None
        Per-mode variance of the fitting split; required when ``spec.kind == "matched"``.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(K + 1,)``.

    Raises
    ------
    ScheduleError
        If a matched schedule is requested without a spectrum.
    """
    if spec.kind == "log":
        return log_schedule(spec)
    if power is None:
        raise ScheduleError("a matched schedule needs the per-mode variance of its split")
    return matched_schedule(power, spec)


def per_level_removed(power: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Between-image variance removed by each step of a schedule, from level 2 on.

    Level 1's target contains the whole identity-to-``sigma_min`` step, which no schedule
    controls; the analysis scripts drop it (``residual_norms(...)[1:]``) and so does this.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.
    schedule : numpy.ndarray
        Schedule of shape ``(K + 1,)``.

    Returns
    -------
    numpy.ndarray
        Array of length ``K - 1``.
    """
    return residual_norms(power, np.asarray(schedule, dtype=np.float64)[1:] ** 2 / 2)[1:]


def per_level_spread(power: np.ndarray, schedule: np.ndarray) -> float:
    """Max/min of the between-image variance removed per step: the "spread" of the tables.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.
    schedule : numpy.ndarray
        Schedule of shape ``(K + 1,)``.

    Returns
    -------
    float
        ``1.0`` for a schedule matched to this spectrum; larger the worse the mismatch.

    Raises
    ------
    ScheduleError
        If some step removes no variance at all.
    """
    removed = per_level_removed(power, schedule)
    if not removed.min() > 0.0:
        raise ScheduleError("a step of the schedule removes no variance")
    return float(removed.max() / removed.min())


def levels_per_octave(schedule: np.ndarray) -> dict[str, int]:
    """Number of levels whose ``sigma_B`` falls in each pixel octave.

    The bins are those of ``learning/03-terminal-blur-scaffolding.md`` §16.3, extended to the
    ``[64, 96]`` octave of the ``W/2`` terminal blur; the last bin is closed so that the
    counts sum to ``K`` for both terminal blurs. Level 0 (``sigma_B = 0``) is not a level of
    the chain and is not counted. Ported from ``knob_evidence.levels_per_octave``.

    Parameters
    ----------
    schedule : numpy.ndarray
        Schedule of shape ``(K + 1,)``.

    Returns
    -------
    dict[str, int]
        One count per octave label; the values sum to ``K``.

    Raises
    ------
    ScheduleError
        If a level falls outside ``[0.5, 96]``.
    """
    levels = np.asarray(schedule, dtype=np.float64)[1:]
    counts: list[int] = []
    for lo, hi in zip(SIGMA_B_OCTAVE_EDGES[:-1], SIGMA_B_OCTAVE_EDGES[1:], strict=True):
        inside = (levels >= lo) & (levels < hi)
        if hi == SIGMA_B_OCTAVE_EDGES[-1]:
            inside |= levels == hi
        counts.append(int(inside.sum()))
    if sum(counts) != levels.size:
        raise ScheduleError(
            f"levels_per_octave counted {sum(counts)} of {levels.size} levels; some level lies "
            f"outside [{SIGMA_B_OCTAVE_EDGES[0]}, {SIGMA_B_OCTAVE_EDGES[-1]}]"
        )
    return dict(zip(SIGMA_B_OCTAVE_LABELS, counts, strict=True))


def _structural_problems(values: np.ndarray, label: str) -> list[str]:
    """Check an array against ``04-run-artifacts.md`` §1 and return the problems found."""
    problems: list[str] = []
    if values.dtype != np.float64:
        problems.append(f"{label}: dtype must be float64, got {values.dtype}")
    if values.ndim != 1 or values.size < 2:
        problems.append(f"{label}: shape must be (K + 1,), got {values.shape}")
        return problems
    k = values.size - 1
    if values[0] != 0.0:
        problems.append(f"{label}: s[0] must be 0.0, got {values[0]!r}")
    if values[1] != 0.5:
        problems.append(f"{label}: s[1] must be 0.5, got {values[1]!r}")
    if not np.all(np.diff(values) > 0):
        problems.append(f"{label}: the schedule must be strictly increasing")
    if values[k] not in TERMINAL_BLURS:
        problems.append(
            f"{label}: s[K] must be one of {list(TERMINAL_BLURS)}, got {values[k]!r}"
        )
    return problems


def _entry_problems(
    entry: dict[str, Any], values: np.ndarray, digest: str, label: str
) -> list[str]:
    """Check one ``schedules.json`` entry against the array it describes."""
    problems: list[str] = []
    k = values.size - 1
    if entry.get("sha256") != digest:
        problems.append(
            f"{label}: sha256 in schedules.json is {entry.get('sha256')!r}, file is {digest!r}"
        )
    if entry.get("K") != k:
        problems.append(f"{label}: K in schedules.json is {entry.get('K')!r}, array has {k}")
    if entry.get("sigma_max") != float(values[k]):
        problems.append(
            f"{label}: sigma_max in schedules.json is {entry.get('sigma_max')!r}, "
            f"array ends at {float(values[k])!r}"
        )
    if entry.get("sigma_min") != float(values[1]):
        problems.append(
            f"{label}: sigma_min in schedules.json is {entry.get('sigma_min')!r}, "
            f"array starts at {float(values[1])!r}"
        )
    table = entry.get("levels_per_octave")
    if not isinstance(table, dict):
        problems.append(f"{label}: levels_per_octave missing from schedules.json")
    else:
        if sum(table.values()) != k:
            problems.append(f"{label}: levels_per_octave sums to {sum(table.values())}, not {k}")
        if table != levels_per_octave(values):
            problems.append(f"{label}: levels_per_octave does not match the array")
    return problems


def validate_schedule(path: Path) -> list[str]:
    """Check a frozen schedule file against ``04-run-artifacts.md`` §1.

    Checks shape, dtype, monotonicity and endpoints of the array, then its entry in the
    ``schedules.json`` beside it: the SHA-256 of the file bytes, ``K``, the endpoints and the
    ``levels_per_octave`` table.

    Parameters
    ----------
    path : Path
        Path of the ``.npy`` file.

    Returns
    -------
    list[str]
        One message per violation; empty when the file is valid.
    """
    path = Path(path)
    if not path.is_file():
        return [f"{path}: file not found"]
    raw = path.read_bytes()
    try:
        values = np.load(path)
    except (OSError, ValueError) as exc:
        return [f"{path}: cannot be loaded as a .npy array ({exc})"]

    problems = _structural_problems(values, str(path))
    if values.ndim != 1 or values.size < 2:
        return problems

    index = path.parent / "schedules.json"
    if not index.is_file():
        return [*problems, f"{index}: missing; the hash of {path.name} cannot be checked"]
    try:
        table = json.loads(index.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [*problems, f"{index}: cannot be parsed ({exc})"]
    entry = table.get(path.stem)
    if not isinstance(entry, dict):
        return [*problems, f"{index}: no entry for {path.stem!r}"]
    digest = hashlib.sha256(raw).hexdigest()
    return [*problems, *_entry_problems(entry, values, digest, str(path))]


def git_sha(repo_root: Path) -> str:
    """Return the ``HEAD`` commit of ``repo_root``, or ``"unknown"``.

    Parameters
    ----------
    repo_root : Path
        Directory inside the repository.

    Returns
    -------
    str
        The 40-character SHA, or ``"unknown"`` if git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def save_schedule(path: Path, schedule: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    """Write a frozen schedule array and return its ``schedules.json`` entry.

    Parameters
    ----------
    path : Path
        Destination ``.npy`` path; the parent directory is created.
    schedule : numpy.ndarray
        The array, already carrying its exact endpoints.
    meta : dict[str, Any]
        The provenance fields of ``04-run-artifacts.md`` §1 that this function does not
        derive: ``kind``, ``sigma_min``, ``sigma_max``, ``K``, ``fitted_on``, ``n_images``,
        ``images_sha256``, ``spread``, ``git_sha``.

    Returns
    -------
    dict[str, Any]
        ``meta`` plus ``levels_per_octave``, ``sha256`` and ``created``.

    Raises
    ------
    ScheduleError
        If the array violates the structural contract.
    """
    values = np.asarray(schedule, dtype=np.float64)
    problems = _structural_problems(values, str(path))
    if problems:
        raise ScheduleError("; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    entry = dict(meta)
    entry["levels_per_octave"] = levels_per_octave(values)
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["created"] = datetime.now(UTC).isoformat(timespec="seconds")
    logger.info("wrote %s (sha256 %s)", path, entry["sha256"][:12])
    return entry
