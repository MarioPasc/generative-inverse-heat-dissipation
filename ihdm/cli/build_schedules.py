"""Build and freeze the seven blur-schedule arrays the arms load.

Writes ``schedules/<name>.npy`` (``float64``, shape ``(201,)``) and
``schedules/schedules.json`` per ``docs/SPECIFICATIONS/04-run-artifacts.md`` §1. The five
matched schedules are fitted on the **training** split of their dataset, each with its own
terminal blur (decision D12: the march is re-run for ``W/2`` and for ``W/8``; a schedule is
never truncated or rescaled from another one).

Run as ``python -m ihdm.cli.build_schedules``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ihdm.paths import data_root, repo_root, schedules_dir
from ihdm.spectral.profile import split_power
from ihdm.spectral.schedules import (
    ScheduleError,
    ScheduleSpec,
    build_schedule,
    git_sha,
    per_level_spread,
    save_schedule,
    validate_schedule,
)

logger = logging.getLogger(__name__)

# The frozen names of 04 §1. The first five are loaded by an arm of
# configs/spectral/arms.py; the last two exist for the profile tables only.
SCHEDULES: tuple[tuple[str, ScheduleSpec], ...] = (
    ("log_W2", ScheduleSpec(kind="log", sigma_max=96.0)),
    ("log_W8", ScheduleSpec(kind="log", sigma_max=24.0)),
    ("ixi_W2", ScheduleSpec(kind="matched", sigma_max=96.0, fitted_on="ixi/train")),
    ("ixi_W8", ScheduleSpec(kind="matched", sigma_max=24.0, fitted_on="ixi/train")),
    (
        "lsun_church_W2",
        ScheduleSpec(kind="matched", sigma_max=96.0, fitted_on="lsun_church/train"),
    ),
    ("oasis1_W8", ScheduleSpec(kind="matched", sigma_max=24.0, fitted_on="oasis1/train")),
    (
        "lsun_bedroom_W2",
        ScheduleSpec(kind="matched", sigma_max=96.0, fitted_on="lsun_bedroom/train"),
    ),
)


def _fitting_splits() -> list[str]:
    """The distinct ``<dataset_id>/<split>`` the matched schedules are fitted on."""
    seen = [spec.fitted_on for _, spec in SCHEDULES if spec.fitted_on is not None]
    return sorted(set(seen))


def _spectra(root: Path) -> dict[str, tuple[np.ndarray, int, str]]:
    """Per-mode variance of every fitting split, keyed by ``"<dataset_id>/<split>"``."""
    out: dict[str, tuple[np.ndarray, int, str]] = {}
    for key in _fitting_splits():
        dataset_id, split = key.split("/", 1)
        out[key] = split_power(root, dataset_id, split)
        print(f"  spectrum of {key}: {out[key][1]} images")
    return out


def _entry(name: str, spec: ScheduleSpec, spectra: dict[str, tuple[np.ndarray, int, str]],
           out_dir: Path, sha: str) -> dict[str, Any]:
    """Build one schedule, write its ``.npy`` and return its ``schedules.json`` entry.

    A log schedule is not fitted on any split, so ``fitted_on``, ``n_images``,
    ``images_sha256`` and ``spread`` are ``null`` for it; the spread of the log schedule on
    each of the four datasets is reported in ``docs/RESULTS/data_profile.md`` instead.
    """
    power = spectra[spec.fitted_on][0] if spec.fitted_on else None
    values = build_schedule(spec, power)
    meta: dict[str, Any] = {
        "kind": spec.kind,
        "sigma_min": spec.sigma_min,
        "sigma_max": spec.sigma_max,
        "K": spec.K,
        "fitted_on": spec.fitted_on,
        "n_images": None,
        "images_sha256": None,
        "spread": None,
        "git_sha": sha,
    }
    if spec.fitted_on is not None and power is not None:
        _, n_images, images_sha256 = spectra[spec.fitted_on]
        meta["n_images"] = n_images
        meta["images_sha256"] = images_sha256
        meta["spread"] = round(per_level_spread(power, values), 6)
    return save_schedule(out_dir / f"{name}.npy", values, meta)


def build(root: Path, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Build every schedule of :data:`SCHEDULES` and write ``schedules.json``.

    Parameters
    ----------
    root : Path
        The data root holding the four datasets.
    out_dir : Path
        Destination directory (``schedules/``).

    Returns
    -------
    dict[str, dict[str, Any]]
        The ``schedules.json`` mapping.
    """
    spectra = _spectra(root)
    table: dict[str, dict[str, Any]] = {}
    for name, spec in SCHEDULES:
        table[name] = _entry(name, spec, spectra, out_dir, git_sha(repo_root()))
        print(
            f"  {name:16s} {spec.kind:8s} sigma_max={spec.sigma_max:5.1f} "
            f"fitted_on={str(spec.fitted_on):18s} spread="
            f"{'n/a' if table[name]['spread'] is None else format(table[name]['spread'], '.4f')}"
            f"  sha256={table[name]['sha256'][:12]}"
        )
    (out_dir / "schedules.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return table


def _validate_all(out_dir: Path) -> list[str]:
    """Run ``validate_schedule`` on every produced file and return every problem found."""
    problems: list[str] = []
    for name, _ in SCHEDULES:
        found = validate_schedule(out_dir / f"{name}.npy")
        print(f"  {name:16s} {'OK' if not found else '; '.join(found)}")
        problems.extend(found)
    return problems


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        ``0`` on success, ``1`` if any file fails validation.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--out", type=Path, default=schedules_dir())
    parser.add_argument("--force", action="store_true", help="rebuild even if the files validate")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and (out_dir / "schedules.json").is_file():
        if not _validate_all(out_dir):
            print(f"schedules already frozen and valid in {out_dir}; --force to rebuild")
            return 0
        print("existing schedules do not validate; rebuilding")

    started = datetime.now(UTC)
    try:
        table = build(args.data_root, out_dir)
    except ScheduleError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    problems = _validate_all(out_dir)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    if problems:
        print(f"FAILED: {len(problems)} problem(s) in {out_dir}", file=sys.stderr)
        return 1
    print(f"{len(table)} schedules frozen in {out_dir} in {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
