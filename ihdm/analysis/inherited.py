"""Reference-side constants and corrected readings of the inherited band (D23, T6.4).

The pre-registered estimator of ``05-metrics.md`` §5 measures the residual of the samples about
the prior state ``d x_s``. With a population mean image ``mu`` that has non-DC content, its model
expectation is ``I - T``, not ``I``, where

    T = sum_{i != 0} (1 - d_i)^2 mu_i^2 / sum_{i != 0} P_i,        d = exp(-lambda sigma_max^2 / 2)

(``docs/RESULTS/inherited_band_audit.md`` §3.2). D23 corrects the reading, not the metric: the
stored ``final.json`` scalars are split exactly (audit §3.3) into

* the within-seed fraction ``G_w = D_pix (W^2 - 1) / sum P_ref``, written as the within-seed share
  ``I_w = 1 - M / (M - 1) G_w``, whose model expectation is ``I`` whatever the mean image;
* the seed-mean bias fraction ``G_b = 1 - inherited_measured - G_w``, whose model expectation is
  ``T + (1 - I) / M``.

Both need ``sum P_ref`` and ``T`` per (dataset, sigma_max). This module computes them once from the
``ref`` split of each dataset (the same split and the same ``P_ref`` as ``run_eval``), together with
the radial curve of the corrected per-mode prediction ``(1 - d^2) + (1 - d)^2 mu^2 / P`` for
figure 5, and writes them to ``docs/RESULTS/inherited_band_constants.json``. The analysis then
reads ``results/`` plus that file, never the raw data. Every entry carries the dataset's
``sha256_images``, and a run is matched to its constants only when its ``dataset_sha256`` agrees.

``python -m ihdm.analysis.inherited --data-root <root> [--out PATH] [--datasets ixi ...]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dctn

from configs.spectral.arms import _ARM_SCHEDULE, _ARM_SIGMA_MAX, EXPERIMENT_CELLS
from ihdm.data.format import DataFormatError, read_dataset
from ihdm.metrics.io import write_json
from ihdm.metrics.spectral import LOW_BAND_SIGMA_PX, N_LOG_BINS, log_bin_centres, log_bin_edges
from ihdm.paths import repo_root
from ihdm.spectral.power import eigenvalues, inherited_share, mode_power, radial_profile

__all__ = [
    "DEFAULT_CONSTANTS_PATH",
    "BandConstants",
    "InheritedConstants",
    "InheritedError",
    "band_constants",
    "bias_expected",
    "bias_fraction",
    "compute_constants",
    "experiment_sigmas",
    "load_constants",
    "main",
    "reference_spectrum",
    "within_seed_fraction",
    "within_seed_share",
]

logger = logging.getLogger(__name__)

#: The committed constants file the tables and figures read by default.
DEFAULT_CONSTANTS_PATH: Path = repo_root() / "docs" / "RESULTS" / "inherited_band_constants.json"
#: Schema version of the constants file.
SCHEMA_VERSION: int = 1
_HASH_CHUNK: int = 1 << 24


class InheritedError(Exception):
    """Raised when the constants cannot be computed or read (bad dataset, bad file)."""


# --------------------------------------------------------------------------------------------
# The corrected readings of the stored scalars (audit §3.3)
# --------------------------------------------------------------------------------------------


def within_seed_fraction(d_pix: float, n_pix: int, sum_power: float) -> float:
    """``G_w = D_pix (W^2 - 1) / sum P_ref``: within-seed variance over population variance.

    Parameters
    ----------
    d_pix : float
        ``final.json: diversity_pix``, the per-non-DC-mode within-seed variance (1/M-normalised).
    n_pix : int
        Image side ``W``; ``W^2 - 1`` is the number of non-DC modes.
    sum_power : float
        ``sum P_ref`` over the non-DC modes of the dataset's ``ref`` split.

    Returns
    -------
    float
        ``G_w``; ``NaN`` when an input is missing or ``sum_power`` is not positive.
    """
    if not (math.isfinite(d_pix) and math.isfinite(sum_power) and sum_power > 0.0):
        return math.nan
    return float(d_pix) * (int(n_pix) ** 2 - 1) / float(sum_power)


def within_seed_share(d_pix: float, n_per_seed: float, n_pix: int, sum_power: float) -> float:
    """``I_w = 1 - M / (M - 1) G_w``, the within-seed share; its model expectation is ``I``.

    Parameters
    ----------
    d_pix : float
        ``final.json: diversity_pix``.
    n_per_seed : float
        ``M``, the samples per seed (``final.json: n_per_seed``).
    n_pix : int
        Image side ``W``.
    sum_power : float
        ``sum P_ref`` over the non-DC modes.

    Returns
    -------
    float
        ``I_w``; ``NaN`` when an input is missing or ``M < 2``.
    """
    if not (math.isfinite(n_per_seed) and n_per_seed >= 2):
        return math.nan
    m = float(n_per_seed)
    return 1.0 - m / (m - 1.0) * within_seed_fraction(d_pix, n_pix, sum_power)


def bias_fraction(inherited_measured: float, d_pix: float, n_pix: int, sum_power: float) -> float:
    """``G_b = 1 - inherited_measured - G_w``: the seed-mean residual about ``d x_s``.

    Parameters
    ----------
    inherited_measured : float
        ``final.json: inherited_measured`` (``1 - sum V / sum P_ref``).
    d_pix : float
        ``final.json: diversity_pix``.
    n_pix : int
        Image side ``W``.
    sum_power : float
        ``sum P_ref`` over the non-DC modes.

    Returns
    -------
    float
        ``G_b``; its model expectation is :func:`bias_expected`.
    """
    if not math.isfinite(inherited_measured):
        return math.nan
    return 1.0 - float(inherited_measured) - within_seed_fraction(d_pix, n_pix, sum_power)


def bias_expected(share_predicted: float, mean_term: float, n_per_seed: float) -> float:
    """``T + (1 - I) / M``, the model expectation of :func:`bias_fraction`.

    Parameters
    ----------
    share_predicted : float
        ``I`` (``inherited_predicted``).
    mean_term : float
        ``T`` of the dataset at the run's ``sigma_max``.
    n_per_seed : float
        ``M``.

    Returns
    -------
    float
        The expectation; ``NaN`` when an input is missing or ``M < 1``.
    """
    if not (math.isfinite(share_predicted) and math.isfinite(mean_term)
            and math.isfinite(n_per_seed) and n_per_seed >= 1):
        return math.nan
    return float(mean_term) + (1.0 - float(share_predicted)) / float(n_per_seed)


# --------------------------------------------------------------------------------------------
# Reference-side constants
# --------------------------------------------------------------------------------------------


def experiment_sigmas() -> dict[str, tuple[float, ...]]:
    """The terminal blurs each dataset is run at, from ``EXPERIMENT_CELLS``.

    Returns
    -------
    dict[str, tuple[float, ...]]
        ``{dataset: sorted sigma_max values}``.
    """
    sigmas: dict[str, set[float]] = {}
    for dataset, arm, _ in EXPERIMENT_CELLS:
        sigmas.setdefault(dataset, set()).add(float(_ARM_SIGMA_MAX[arm]))
    return {dataset: tuple(sorted(values)) for dataset, values in sigmas.items()}


def reference_spectrum(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``P_ref`` and the mean image ``mu`` in DCT coefficients, both with the DC mode zeroed.

    ``P_ref`` is computed exactly as ``run_eval`` does (``mode_power(reference.astype(float32)
    / 255)``), so its sum and ``I`` reproduce the runs' ``inherited_predicted``. The DCT is linear,
    so ``mu`` is the DCT of the mean image; removing each image's DC changes only ``mu[0, 0]``.

    Parameters
    ----------
    reference : numpy.ndarray
        The ``ref`` split, ``(n, W, W)`` ``uint8`` (or float in 0-255).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(P_ref, mu)``, each ``(W, W)`` ``float64``, in ``[0, 1]`` intensity units.
    """
    stack = np.asarray(reference)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2] or stack.shape[0] < 2:
        raise InheritedError(f"reference: expected (n >= 2, W, W), got {stack.shape}")
    power = mode_power(stack.astype(np.float32) / np.float32(255.0))
    power[0, 0] = 0.0
    mean_image = np.zeros(stack.shape[1:], dtype=np.float64)
    for start in range(0, stack.shape[0], 256):
        mean_image += stack[start:start + 256].astype(np.float64).sum(axis=0)
    mean_image /= 255.0 * stack.shape[0]
    mu = dctn(mean_image, norm="ortho")
    mu[0, 0] = 0.0
    return power, mu


def _finite_bins(centres: np.ndarray, *curves: np.ndarray) -> tuple[list[float], ...]:
    """Drop the bins that hold no mode, as ``run_eval._finite_curve`` does for ``final.json``."""
    mask = np.isfinite(centres)
    for curve in curves:
        mask &= np.isfinite(curve)
    return tuple([float(v) for v in np.asarray(a)[mask]] for a in (centres, *curves))


def band_constants(
    power: np.ndarray,
    mu: np.ndarray,
    sigma_max: float,
    low_band_sigma_px: float = LOW_BAND_SIGMA_PX,
    n_bins: int = N_LOG_BINS,
) -> dict[str, Any]:
    """The D23 constants of one spectrum at one terminal blur.

    Parameters
    ----------
    power : numpy.ndarray
        ``P_ref`` of shape ``(W, W)`` (DC ignored).
    mu : numpy.ndarray
        Mean image in DCT coefficients, ``(W, W)`` (DC ignored).
    sigma_max : float
        Terminal blur in pixels.
    low_band_sigma_px : float
        The low band of ``05-metrics.md`` §5: modes with ``lambda <= 2 / sigma_px^2``.
    n_bins : int
        Log bins of the radial curves (the grid of ``final.json: radial``).

    Returns
    -------
    dict
        ``share_predicted`` (``I``), ``mean_term`` (``T``), their low-band versions, and the
        radial curves ``centres``, ``radial_predicted`` (``1 - d^2``) and ``radial_corrected``
        (``(1 - d^2) + (1 - d)^2 mu^2 / P``) on the populated bins.

    Raises
    ------
    InheritedError
        If the shapes disagree, ``sigma_max`` is not positive, or the spectrum is empty.
    """
    p = np.asarray(power, dtype=np.float64).copy()
    m = np.asarray(mu, dtype=np.float64).copy()
    if p.ndim != 2 or p.shape[0] != p.shape[1] or m.shape != p.shape:
        raise InheritedError(f"power {p.shape} and mu {m.shape} must be the same square grid")
    if not (math.isfinite(sigma_max) and sigma_max > 0.0):
        raise InheritedError(f"sigma_max must be positive and finite, got {sigma_max!r}")
    p[0, 0] = 0.0
    m[0, 0] = 0.0
    lam = eigenvalues(p.shape[0])
    d = np.exp(-lam * sigma_max**2 / 2.0)
    shift = (1.0 - d) ** 2 * m**2
    low = lam <= 2.0 / low_band_sigma_px**2
    low[0, 0] = False
    total, total_low = float(p.sum()), float(p[low].sum())
    if not (total > 0.0 and total_low > 0.0):
        raise InheritedError("the spectrum holds no non-DC variance (or none in the low band)")

    safe = np.where(p > 0.0, p, 1.0)
    corrected = np.where(p > 0.0, (1.0 - d**2) + shift / safe, 0.0)
    edges = log_bin_edges(n_bins)
    centres, predicted, corrected_curve = _finite_bins(
        log_bin_centres(n_bins), radial_profile(1.0 - d**2, edges),
        radial_profile(corrected, edges))
    return {
        "sigma_max": float(sigma_max),
        "t_K": float(sigma_max) ** 2 / 2.0,
        "share_predicted": float(inherited_share(p, sigma_max)),
        "mean_term": float(shift.sum()) / total,
        "share_predicted_low_band": float(inherited_share(np.where(low, p, 0.0), sigma_max)),
        "mean_term_low_band": float(shift[low].sum()) / total_low,
        "radial": {"centres": centres, "predicted": predicted, "corrected": corrected_curve},
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_constants(root: Path, sigmas: Sequence[float], verify_sha: bool) -> dict[str, Any]:
    try:
        images, _, splits, meta = read_dataset(root, mmap=True)
    except DataFormatError as error:
        raise InheritedError(f"{root}: not a standard-format dataset ({error})") from error
    sha = str(getattr(meta, "sha256_images", "") or "")
    if verify_sha:
        actual = _file_sha256(root / "images.npy")
        if actual != sha:
            raise InheritedError(f"{root}: images.npy hashes to {actual}, meta.json says {sha}")
    ref_idx = np.sort(np.asarray(splits.get("ref", []), dtype=np.int64))
    if ref_idx.size < 2:
        raise InheritedError(f"{root}: the ref split holds {ref_idx.size} images")
    reference = np.asarray(images[ref_idx])
    power, mu = reference_spectrum(reference)
    n_pix = int(power.shape[0])
    return {
        "sha256": sha,
        "sha256_verified": bool(verify_sha),
        "n_pix": n_pix,
        "n_modes": n_pix * n_pix - 1,
        "n_ref": int(ref_idx.size),
        "sum_power": float(power.sum()),
        "sum_mean_squared": float(np.square(mu).sum()),
        "sigmas": {_sigma_key(s): band_constants(power, mu, s) for s in sigmas},
    }


def _sigma_key(sigma_max: float) -> str:
    return f"{float(sigma_max):g}"


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def compute_constants(
    data_root: str | Path,
    datasets: Iterable[str] | None = None,
    verify_sha: bool = True,
) -> dict[str, Any]:
    """The constants document for every (dataset, sigma_max) of the experiment.

    Parameters
    ----------
    data_root : str or Path
        Folder holding one standard-format dataset per id (``<root>/<dataset>/images.npy`` ...).
    datasets : Iterable[str] or None
        Restrict to these dataset ids; default every dataset of ``EXPERIMENT_CELLS``.
    verify_sha : bool
        Hash ``images.npy`` and require it to equal ``meta.json: sha256_images``.

    Returns
    -------
    dict
        The JSON document :func:`main` writes.

    Raises
    ------
    InheritedError
        If a dataset is unknown to the design, unreadable, or fails its hash check.
    """
    root = Path(data_root)
    design = experiment_sigmas()
    wanted = list(design) if datasets is None else list(datasets)
    unknown = sorted(set(wanted) - set(design))
    if unknown:
        raise InheritedError(f"datasets not in the design: {unknown}")
    arms = sorted({arm for _, arm, _ in EXPERIMENT_CELLS})
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "python -m ihdm.analysis.inherited (T6.4, D23)",
        "created": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "data_root": str(root),
        "definition": {
            "units": "intensities in [0, 1] (uint8 / 255); orthonormal DCT-II; DC mode excluded",
            "P_ref": "mode_power of the ref split, as run_eval (population variance, mean removed)",
            "mu": "DCT of the ref-split mean image",
            "d": "exp(-lambda sigma_max^2 / 2), lambda = pi^2 (i^2 + j^2) / W^2",
            "share_predicted": "I = sum d^2 P / sum P",
            "mean_term": "T = sum (1 - d)^2 mu^2 / sum P",
            "radial.corrected": "unweighted bin mean of (1 - d^2) + (1 - d)^2 mu^2 / P",
            "radial.predicted": "unweighted bin mean of 1 - d^2 (the pre-registered line)",
            "low_band": f"lambda <= 2 / {LOW_BAND_SIGMA_PX:g}^2, i.e. sigma_n >= "
                        f"{LOW_BAND_SIGMA_PX:g} px",
            "source": "docs/RESULTS/inherited_band_audit.md §3.2-3.3",
        },
        "n_log_bins": N_LOG_BINS,
        "low_band_sigma_px": LOW_BAND_SIGMA_PX,
        "arms": {arm: {"schedule": _ARM_SCHEDULE[arm], "sigma_max": float(_ARM_SIGMA_MAX[arm]),
                       "t_K": float(_ARM_SIGMA_MAX[arm]) ** 2 / 2.0} for arm in arms},
        "datasets": {ds: _dataset_constants(root / ds, design[ds], verify_sha) for ds in wanted},
    }


# --------------------------------------------------------------------------------------------
# Reading the constants
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BandConstants:
    """The constants of one dataset at one terminal blur, as the analysis reads them.

    Parameters
    ----------
    dataset, sha256 : str
        Dataset id and its ``sha256_images``.
    n_pix : int
        Image side ``W``.
    sigma_max : float
        Terminal blur in pixels.
    sum_power : float
        ``sum P_ref`` over the non-DC modes.
    share_predicted, mean_term : float
        ``I`` and ``T``.
    centres, radial_predicted, radial_corrected : tuple[float, ...]
        The radial curves of figure 5 on the populated bins.
    """

    dataset: str
    sha256: str
    n_pix: int
    sigma_max: float
    sum_power: float
    share_predicted: float
    mean_term: float
    centres: tuple[float, ...]
    radial_predicted: tuple[float, ...]
    radial_corrected: tuple[float, ...]

    @property
    def expected_measured(self) -> float:
        """``I - T``, the model expectation of the pre-registered ``inherited_measured``."""
        return self.share_predicted - self.mean_term


@dataclass(frozen=True)
class InheritedConstants:
    """A loaded constants file.

    Parameters
    ----------
    path : Path or None
        Where it was read from (``None`` for an in-memory document).
    entries : dict
        ``{(dataset, sigma_key): BandConstants}``.
    """

    path: Path | None
    entries: dict[tuple[str, str], BandConstants]

    @property
    def label(self) -> str:
        """The file, relative to the repository when it lives there; ``"none given"`` if absent."""
        if self.path is None:
            return "none given"
        try:
            return self.path.resolve().relative_to(repo_root()).as_posix()
        except ValueError:
            return str(self.path)

    def lookup(self, dataset: str, sha256: str, sigma_max: float
               ) -> tuple[BandConstants | None, str]:
        """The constants of a run, or ``None`` and the reason they do not apply.

        Parameters
        ----------
        dataset : str
            The run's dataset id.
        sha256 : str
            The run's ``dataset_sha256``.
        sigma_max : float
            The run's ``final.sigma_max``.

        Returns
        -------
        tuple[BandConstants | None, str]
            The entry and ``""``, or ``None`` and a short reason.
        """
        if not (isinstance(sigma_max, int | float) and math.isfinite(sigma_max)):
            return None, f"{dataset}: no sigma_max"
        entry = self.entries.get((dataset, _sigma_key(sigma_max)))
        if entry is None:
            return None, f"{dataset}: no constants at sigma_max {sigma_max:g}"
        if not sha256:
            return None, f"{dataset}: the run records no dataset_sha256"
        if entry.sha256 != sha256:
            return None, (f"{dataset}: dataset_sha256 {str(sha256)[:12]} differs from the "
                          f"constants' {entry.sha256[:12]}")
        return entry, ""


def _parse(document: Mapping[str, Any], path: Path | None) -> InheritedConstants:
    try:
        entries = {}
        for dataset, block in document["datasets"].items():
            for key, band in block["sigmas"].items():
                radial = band["radial"]
                entries[(dataset, key)] = BandConstants(
                    dataset=dataset, sha256=str(block["sha256"]), n_pix=int(block["n_pix"]),
                    sigma_max=float(band["sigma_max"]), sum_power=float(block["sum_power"]),
                    share_predicted=float(band["share_predicted"]),
                    mean_term=float(band["mean_term"]),
                    centres=tuple(float(v) for v in radial["centres"]),
                    radial_predicted=tuple(float(v) for v in radial["predicted"]),
                    radial_corrected=tuple(float(v) for v in radial["corrected"]),
                )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise InheritedError(f"{path or 'constants'}: malformed constants ({error!r})") from error
    return InheritedConstants(path=path, entries=entries)


def load_constants(source: str | Path | Mapping[str, Any] | None = DEFAULT_CONSTANTS_PATH
                   ) -> InheritedConstants:
    """Read a constants file (or an in-memory document); ``None`` gives an empty set.

    Parameters
    ----------
    source : str, Path, Mapping or None
        A path to the JSON written by :func:`main`, the document itself, or ``None``.

    Returns
    -------
    InheritedConstants
        The entries by (dataset, sigma_max).

    Raises
    ------
    InheritedError
        If the file cannot be read or is malformed.
    """
    if source is None:
        return InheritedConstants(path=None, entries={})
    if isinstance(source, Mapping):
        return _parse(source, None)
    path = Path(source)
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise InheritedError(f"{path}: {error}") from error
    return _parse(document, path)


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Compute the constants from the ``ref`` splits and write the JSON; print one line each.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 on success, 1 when a dataset cannot be read or fails its hash check.
    """
    parser = argparse.ArgumentParser(prog="python -m ihdm.analysis.inherited",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="folder of the standard-format datasets (<root>/<dataset>/)")
    parser.add_argument("--out", type=Path, default=DEFAULT_CONSTANTS_PATH,
                        help="constants JSON to write (default: the committed file)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="dataset ids (default: every dataset of the design)")
    parser.add_argument("--no-verify-sha", action="store_true",
                        help="skip hashing images.npy against meta.json")
    args = parser.parse_args(argv)
    try:
        document = compute_constants(args.data_root, args.datasets,
                                     verify_sha=not args.no_verify_sha)
    except InheritedError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    write_json(args.out, document)
    for dataset, block in document["datasets"].items():
        for key, band in block["sigmas"].items():
            print(f"{dataset:<13} sigma {key:>3}: sum P {block['sum_power']:.6g}, "
                  f"I {band['share_predicted']:.5g}, T {band['mean_term']:.5g}, "
                  f"I - T {band['share_predicted'] - band['mean_term']:+.4g}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
