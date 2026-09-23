"""Turn one raw MRI cohort into the standard dataset format.

Usage
-----
``python -m ihdm.cli.preprocess_mri --cohort ixi|oasis1 [--workers 20] [--limit N]
[--force] [--no-n4] [--sensitivity-copy NAME] [--out $IHDM_DATA_ROOT]``

Stages, in order: list the subjects, register every one of them onto the MNI152 template
(in parallel, cached on disk), apply the quality gate, draw 400 passing subjects with
``np.random.default_rng(2026)``, correct the bias field of those 400 with N4 (in parallel,
cached on disk; decision D15), cut the ten windowed axial slices of each, split by subject,
write the dataset through :func:`ihdm.data.format.write_dataset`, validate it and draw the
quality-control sheets. Every stage except the registration and N4 is cheap, so a rerun
after a cache hit takes a couple of minutes.

The N4 stage sits **before** the foreground-p99 intensity scaling: the percentile is taken
on the corrected volume, so a subject whose coil profile brightened its anterior half is not
also rescaled by that brightening.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ihdm.data.format import DatasetMeta, split_by_subject, validate_dataset, write_dataset
from ihdm.paths import data_root, raw_root, template_dir
from ihdm.preprocess import bias as bias_mod
from ihdm.preprocess import qc
from ihdm.preprocess import registration as registration_mod
from ihdm.preprocess.bias import N4Config
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.mri import (
    IMAGE_SIZE,
    ORIENTATION,
    PIPELINE_VERSION,
    SLICE_Z_INDICES,
    TemplateGeometry,
    assert_grid_matches_affine,
    extract_slices,
    scale_intensity,
    site_from_source,
    slices_to_uint8,
    template_geometry,
)
from ihdm.preprocess.raw_mri import (
    COHORTS,
    RawVolumeRef,
    list_subjects,
    load_sitk,
    sitk_to_ras_array,
)
from ihdm.preprocess.registration import RegistrationConfig

logger = logging.getLogger("ihdm.cli.preprocess_mri")

N_SUBJECTS_TARGET = 400
RNG_SEED = 2026
N_QC_REGISTRATION_TILES = 10
N_QC_ORIENTATION_ROWS = 4
N_QC_N4_ROWS = 6

#: Where the uncorrected datasets are kept once the N4 rebuild replaces them.
SENSITIVITY_DIR = "_sensitivity"

_WORKER: dict[str, object] = {}


@dataclass(frozen=True)
class CohortPaths:
    """Where one cohort reads from and writes to.

    Parameters
    ----------
    cohort : str
        ``"ixi"`` or ``"oasis1"``.
    dataset : Path
        Output dataset directory.
    cache : Path
        Directory of the cached registered volumes.
    cache_n4 : Path
        Directory of the cached N4-corrected volumes.
    raw : Path
        Root of the unprocessed data tree.
    templates : Path
        Directory of the MNI152 template files.
    sensitivity : Path
        Directory the uncorrected dataset is moved to before the N4 rebuild.
    """

    cohort: str
    dataset: Path
    cache: Path
    cache_n4: Path
    raw: Path
    templates: Path
    sensitivity: Path

    def registered(self, subject: str) -> Path:
        """Return the cache path of one subject's registered volume.

        Parameters
        ----------
        subject : str
            Subject identifier.

        Returns
        -------
        Path
            ``<cache>/<subject>.nii.gz``.
        """
        return self.cache / f"{subject}.nii.gz"

    def corrected(self, subject: str) -> Path:
        """Return the cache path of one subject's N4-corrected volume.

        Parameters
        ----------
        subject : str
            Subject identifier.

        Returns
        -------
        Path
            ``<cache_n4>/<subject>.nii.gz``.
        """
        return self.cache_n4 / f"{subject}.nii.gz"


def main(argv: list[str] | None = None) -> int:
    """Run the MRI preprocessing pipeline for one cohort.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        ``0`` on success, ``1`` if the written dataset violates the format contract.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started = time.time()

    cfg = RegistrationConfig()
    n4_cfg = N4Config(shrink_factor=args.n4_shrink_factor)
    paths = _cohort_paths(args)
    geometry = template_geometry(
        paths.templates / f"{registration_mod.TEMPLATE_STEM}_T1w.nii.gz",
        z_indices=SLICE_Z_INDICES,
        size=args.image_size,
    )
    refs = list_subjects(args.cohort, paths.raw, limit=args.limit)
    logger.info("cohort %s: %d subjects to register", args.cohort, len(refs))

    t_register = time.time()
    unreadable = _register_cohort(
        refs, paths, cfg, geometry, workers=args.workers, force=args.force
    )
    register_seconds = time.time() - t_register

    refs = [ref for ref in refs if ref.subject not in set(unreadable)]
    records = _read_records(refs, paths)
    gate = qc.apply_gate(records, exclude=args.exclude)
    selected = _select_subjects(gate.passing, args.n_subjects)
    if len(selected) < args.n_subjects:
        logger.warning(
            "only %d of the requested %d subjects passed the gate in cohort %s; using all of them",
            len(selected), args.n_subjects, args.cohort,
        )

    sites = {
        ref.subject: site_from_source(args.cohort, ref.source)
        for ref in refs
        if ref.subject in set(selected)
    }

    template = registration_mod.load_template(paths.templates, cfg)
    assert_grid_matches_affine(template.image, geometry)

    t_n4 = time.time()
    n4_summary = (
        _correct_cohort(
            selected, refs, sites, paths, cfg, n4_cfg, geometry,
            workers=args.workers, force=args.force,
        )
        if args.n4
        else {"applied": False}
    )
    n4_seconds = time.time() - t_n4
    volume_path: Callable[[str], Path] = paths.corrected if args.n4 else paths.registered

    images, index_rows, diagnostics = _build_images(
        selected, refs, paths, template.foreground, geometry, volume_path
    )
    splits = split_by_subject([row["subject"] for row in index_rows], rng_seed=RNG_SEED)
    index = _build_index(index_rows, splits)

    moved = move_to_sensitivity(paths.dataset, paths.sensitivity) if args.sensitivity_copy else None
    reference_subjects = _sensitivity_subjects(paths.sensitivity)
    _assert_same_subjects(selected, reference_subjects, paths.sensitivity)

    meta = _build_meta(
        args, paths, cfg, n4_cfg, geometry, gate, records, selected, images, diagnostics,
        template, unreadable, sites, n4_summary, reference_subjects,
    )

    write_dataset(paths.dataset, images, index, splits, meta)
    violations = validate_dataset(paths.dataset)

    qc_paths = _write_qc(
        paths, cfg, geometry, template, records, gate, refs, selected, images, index,
        {
            "registration wall time (s)": f"{register_seconds:.0f}",
            "N4 wall time (s)": f"{n4_seconds:.0f}" if args.n4 else "not applied",
            "subjects used": len(selected),
            "exactly-zero pixels (`padding_fraction`)": f"{float((images == 0).mean()):.2%}",
            "outside the acquisition field of view": (
                f"{_fov_padding_fraction(records, selected):.2%}"
            ),
        },
    )
    if args.n4:
        qc_paths.append(_n4_sheet(paths, selected, geometry, template, n4_cfg))

    padding = float((images == 0).mean())
    elapsed = time.time() - started
    print(
        f"{args.cohort}: registered {len(records)}, passed {len(gate.passing)}, "
        f"used {len(selected)} ({images.shape[0]} images), "
        f"metric median {gate.median:.4f} MAD {gate.mad:.4f}, "
        f"padding {padding:.2%}, n4 {'on' if args.n4 else 'off'}, "
        f"wall {elapsed / 60:.1f} min -> {paths.dataset}"
    )
    if moved is not None:
        print(f"{args.cohort}: uncorrected dataset kept at {moved}")
    if violations:
        for violation in violations[:20]:
            logger.error("validate_dataset: %s", violation)
        print(f"{args.cohort}: INVALID, {len(violations)} violation(s)", file=sys.stderr)
        return 1
    logger.info("qc sheets: %s", ", ".join(str(p) for p in qc_paths))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Build and parse the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__, prog="ihdm.cli.preprocess_mri")
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--limit", type=int, default=None, help="register only the first N subjects"
    )
    parser.add_argument("--force", action="store_true", help="re-register cached subjects")
    parser.add_argument(
        "--out", type=Path, default=None, help="dataset root (default IHDM_DATA_ROOT)"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=None, help="registered-volume cache root"
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS_TARGET)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--exclude", nargs="*", default=[], metavar="SUBJECT",
        help="subject ids to exclude after visual inspection of qc/registration_sheet.png",
    )
    parser.add_argument(
        "--n4", action=argparse.BooleanOptionalAction, default=True,
        help="apply N4 bias-field correction before the intensity scaling (decision D15)",
    )
    parser.add_argument(
        "--n4-shrink-factor", type=int, default=N4Config().shrink_factor,
        help="downsampling factor of the N4 field estimate; the field is always applied at "
             "full resolution",
    )
    parser.add_argument(
        "--sensitivity-copy", action="store_true",
        help="before writing, move an existing uncorrected dataset to "
             "<out>/_sensitivity/<cohort>_no_n4 (idempotent: skipped if it is already there)",
    )
    return parser.parse_args(argv)


def _cohort_paths(args: argparse.Namespace) -> CohortPaths:
    """Resolve every directory the run touches."""
    out = Path(args.out) if args.out is not None else data_root()
    cache_root = Path(args.cache_root) if args.cache_root is not None else data_root() / "_cache"
    return CohortPaths(
        cohort=args.cohort,
        dataset=out / args.cohort,
        cache=cache_root / "registered" / args.cohort,
        cache_n4=cache_root / "registered_n4" / args.cohort,
        raw=Path(args.raw_root) if args.raw_root is not None else raw_root(),
        templates=Path(args.template_dir) if args.template_dir is not None else template_dir(),
        sensitivity=out / SENSITIVITY_DIR / f"{args.cohort}_no_n4",
    )


def _init_worker(
    template_path: Path, cfg: RegistrationConfig, geometry: TemplateGeometry
) -> None:
    """Load the template once per worker process and pin SimpleITK to one thread."""
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    for variable in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[variable] = "1"
    _WORKER["template"] = registration_mod.load_template(template_path, cfg)
    _WORKER["cfg"] = cfg
    _WORKER["geometry"] = geometry


def _register_one(task: tuple[RawVolumeRef, Path]) -> tuple[str, str]:
    """Register one subject and write it to the cache; return ``(subject, error)``."""
    ref, destination = task
    template = _WORKER["template"]
    cfg = _WORKER["cfg"]
    geometry = _WORKER["geometry"]
    try:
        moving = load_sitk(ref)
        result = registration_mod.register(moving, template.image, template.dilated_mask, cfg)
        resampled = registration_mod.resample_to_template(
            moving, template.image, result.transform, cfg
        )
        fov = registration_mod.field_of_view_mask(moving, template.image, result.transform)
        fov_window = float(
            extract_slices(
                sitk.GetArrayFromImage(fov).transpose(2, 1, 0).astype(np.float32), geometry
            ).mean()
        )
        registration_mod.write_registered(
            destination, resampled, result, cfg, ref.subject, ref.source,
            extra={"fov_fraction_window": fov_window},
        )
    except (PreprocessError, RuntimeError, MemoryError) as exc:
        return ref.subject, f"{type(exc).__name__}: {exc}"
    return ref.subject, ""


def _register_cohort(
    refs: list[RawVolumeRef],
    paths: CohortPaths,
    cfg: RegistrationConfig,
    geometry: TemplateGeometry,
    workers: int,
    force: bool,
) -> list[str]:
    """Register every subject that is not already cached, in a process pool.

    Returns
    -------
    list[str]
        Subjects whose raw volume could not be registered; they are dropped from the
        cohort and recorded in ``meta.json``.

    Raises
    ------
    PreprocessError
        If more than 5 % (and more than five) of the subjects fail, which is a bug rather
        than a handful of unreadable acquisitions.
    """
    paths.cache.mkdir(parents=True, exist_ok=True)
    tasks = [
        (ref, paths.registered(ref.subject))
        for ref in refs
        if force or not _cached(paths.registered(ref.subject))
    ]
    logger.info("%d of %d subjects need registration", len(tasks), len(refs))
    if not tasks:
        return []

    template_path = paths.templates
    errors: list[tuple[str, str]] = []
    done = 0
    started = time.time()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=max(1, workers),
        initializer=_init_worker,
        initargs=(template_path, cfg, geometry),
    ) as pool:
        for subject, error in pool.imap_unordered(_register_one, tasks, chunksize=1):
            done += 1
            if error:
                errors.append((subject, error))
                logger.error("registration failed for %s: %s", subject, error)
            if done % 10 == 0 or done == len(tasks):
                rate = (time.time() - started) / done
                logger.info(
                    "registered %d/%d (%.1f s/subject, eta %.1f min)",
                    done, len(tasks), rate, rate * (len(tasks) - done) / 60.0,
                )
    if errors:
        head = ", ".join(f"{s} ({e})" for s, e in errors[:5])
        # A single unreadable acquisition must not throw away an hour of work on the other
        # 580; the subject is dropped and recorded. A large fraction failing is a bug.
        if len(errors) > max(5, int(0.05 * len(tasks))):
            raise PreprocessError(f"{len(errors)} subject(s) failed to register: {head}")
        logger.error("%d subject(s) failed to register and are dropped: %s", len(errors), head)
    return [subject for subject, _ in errors]


def _cached(path: Path) -> bool:
    """Return whether a registered volume and its sidecar are both on disk."""
    return path.is_file() and registration_mod.sidecar_path(path).is_file()


def _cached_n4(path: Path) -> bool:
    """Return whether a corrected volume and its sidecar are both on disk."""
    return path.is_file() and bias_mod.sidecar_path(path).is_file()


def _init_n4_worker(
    template_path: Path, cfg: RegistrationConfig, n4_cfg: N4Config, geometry: TemplateGeometry
) -> None:
    """Load the template mask once per worker process and pin SimpleITK to one thread."""
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    for variable in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[variable] = "1"
    _WORKER["template"] = registration_mod.load_template(template_path, cfg)
    _WORKER["n4_cfg"] = n4_cfg
    _WORKER["geometry"] = geometry


def _correct_one(task: tuple[str, str, str, Path, Path]) -> tuple[str, str, dict[str, float]]:
    """Correct one cached registered volume; return ``(subject, error, diagnostics)``."""
    subject, source, site, registered, destination = task
    template = _WORKER["template"]
    n4_cfg = _WORKER["n4_cfg"]
    geometry = _WORKER["geometry"]
    try:
        volume = sitk.ReadImage(str(registered), sitk.sitkFloat32)
        assert_grid_matches_affine(volume, geometry)
        result = bias_mod.n4_correct(volume, template.dilated_mask, n4_cfg)
        bias_mod.write_corrected(
            destination, result, n4_cfg, subject, source,
            extra={"site": site, "registered": str(registered)},
        )
    except (PreprocessError, RuntimeError, MemoryError) as exc:
        return subject, f"{type(exc).__name__}: {exc}", {}
    return subject, "", result.diagnostics


def _correct_cohort(
    selected: list[str],
    refs: list[RawVolumeRef],
    sites: dict[str, str],
    paths: CohortPaths,
    cfg: RegistrationConfig,
    n4_cfg: N4Config,
    geometry: TemplateGeometry,
    workers: int,
    force: bool,
) -> dict[str, object]:
    """Run N4 on every selected subject that is not already cached, in a process pool.

    Only the subjects that enter the dataset are corrected: the quality gate and the seeded
    draw read the *registration* sidecars, so N4 cannot change which subjects are selected,
    and correcting the rest would cost wall time and cache for nothing.

    Parameters
    ----------
    selected : list[str]
        The subjects that enter the dataset.
    refs : list[RawVolumeRef]
        Raw references of the whole cohort, for the ``source`` of each subject.
    sites : dict[str, str]
        Acquisition site per selected subject.
    paths : CohortPaths
        Cache and template locations.
    cfg : RegistrationConfig
        Supplies the mask dilation of the foreground mask.
    n4_cfg : N4Config
        N4 parameters.
    geometry : TemplateGeometry
        Checked against every volume read back from the registered cache.
    workers : int
        Process-pool size.
    force : bool
        Re-correct subjects that are already cached.

    Returns
    -------
    dict[str, object]
        The parameters used and the aggregated log-field diagnostics, for ``meta.json``.

    Raises
    ------
    PreprocessError
        If any selected subject fails: unlike a registration failure, a dropped subject here
        would change the frozen subject list.
    """
    paths.cache_n4.mkdir(parents=True, exist_ok=True)
    by_subject = {ref.subject: ref for ref in refs}
    tasks = [
        (
            subject,
            by_subject[subject].source,
            sites[subject],
            paths.registered(subject),
            paths.corrected(subject),
        )
        for subject in selected
        if force or not _cached_n4(paths.corrected(subject))
    ]
    logger.info("%d of %d selected subjects need N4", len(tasks), len(selected))

    errors: list[tuple[str, str]] = []
    done = 0
    started = time.time()
    if tasks:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=max(1, workers),
            initializer=_init_n4_worker,
            initargs=(paths.templates, cfg, n4_cfg, geometry),
        ) as pool:
            for subject, error, _ in pool.imap_unordered(_correct_one, tasks, chunksize=1):
                done += 1
                if error:
                    errors.append((subject, error))
                    logger.error("N4 failed for %s: %s", subject, error)
                if done % 25 == 0 or done == len(tasks):
                    rate = (time.time() - started) / done
                    logger.info(
                        "corrected %d/%d (%.1f s/subject, eta %.1f min)",
                        done, len(tasks), rate, rate * (len(tasks) - done) / 60.0,
                    )
    if errors:
        head = ", ".join(f"{s} ({e})" for s, e in errors[:5])
        raise PreprocessError(f"{len(errors)} subject(s) failed N4: {head}")

    return _n4_summary(selected, paths, n4_cfg, time.time() - started)


def _n4_summary(
    selected: list[str], paths: CohortPaths, n4_cfg: N4Config, seconds: float
) -> dict[str, object]:
    """Aggregate the per-subject N4 sidecars into the block stored in ``meta.json``."""
    keys = (
        "log_field_min", "log_field_max", "log_field_mean", "log_field_std",
        "field_ratio_max_over_min", "mask_zero_fraction", "seconds",
    )
    values: dict[str, list[float]] = {k: [] for k in keys}
    for subject in selected:
        sidecar = bias_mod.read_n4_sidecar(paths.corrected(subject))
        for key in keys:
            values[key].append(float(sidecar[key]))
    summary: dict[str, object] = {
        "applied": True,
        **n4_cfg.to_json(),
        "cache_root": str(paths.cache_n4),
        "subjects_corrected": len(selected),
        "wall_seconds": round(float(seconds), 1),
    }
    for key in keys:
        array = np.asarray(values[key], dtype=float)
        summary[f"{key}_median"] = float(np.median(array))
        summary[f"{key}_min"] = float(array.min())
        summary[f"{key}_max"] = float(array.max())
    return summary


def move_to_sensitivity(dataset: Path, sensitivity: Path) -> Path | None:
    """Move an existing dataset directory aside so a rebuild can replace it.

    Idempotent in both directions: the move is skipped when the destination already holds a
    dataset (a second run must not overwrite the archived uncorrected copy with the
    corrected one), and when there is nothing to move.

    Parameters
    ----------
    dataset : Path
        The dataset directory that is about to be rewritten.
    sensitivity : Path
        Where the current contents are kept.

    Returns
    -------
    Path | None
        The destination when a move happened, ``None`` when it was skipped.

    Raises
    ------
    PreprocessError
        If the destination exists but is not a directory.
    """
    dataset, sensitivity = Path(dataset), Path(sensitivity)
    if sensitivity.exists() and not sensitivity.is_dir():
        raise PreprocessError(f"{sensitivity} exists and is not a directory")
    if sensitivity.is_dir() and any(sensitivity.iterdir()):
        logger.info("sensitivity copy already present at %s; not moving %s", sensitivity, dataset)
        return None
    if not dataset.is_dir():
        logger.info("no dataset at %s to keep as a sensitivity copy", dataset)
        return None
    sensitivity.parent.mkdir(parents=True, exist_ok=True)
    if sensitivity.is_dir():
        sensitivity.rmdir()
    shutil.move(str(dataset), str(sensitivity))
    logger.info("moved %s to %s", dataset, sensitivity)
    return sensitivity


def _sensitivity_subjects(sensitivity: Path) -> list[str] | None:
    """Return the subject list of the archived uncorrected dataset, if there is one."""
    index_path = Path(sensitivity) / "index.csv"
    if not index_path.is_file():
        return None
    return sorted(pd.read_csv(index_path)["subject"].astype(str).unique().tolist())


def _assert_same_subjects(
    selected: list[str], reference: list[str] | None, sensitivity: Path
) -> None:
    """Fail loudly if the rebuild would not use the archived dataset's subjects.

    Parameters
    ----------
    selected : list[str]
        The subjects this run is about to write.
    reference : list[str] | None
        The archived dataset's subjects, or ``None`` when there is no archive.
    sensitivity : Path
        Where the archive lives, for the error message.

    Raises
    ------
    PreprocessError
        If the two lists differ.
    """
    if reference is None:
        logger.info("no sensitivity copy at %s; subject list not cross-checked", sensitivity)
        return
    if sorted(selected) != reference:
        missing = sorted(set(reference) - set(selected))
        extra = sorted(set(selected) - set(reference))
        raise PreprocessError(
            f"the rebuilt subject list differs from {sensitivity}: "
            f"{len(missing)} missing {missing[:5]}, {len(extra)} new {extra[:5]}"
        )
    logger.info("subject list identical to %s (%d subjects)", sensitivity, len(reference))


def _read_records(refs: list[RawVolumeRef], paths: CohortPaths) -> list[qc.SubjectRecord]:
    """Read every cached sidecar into a :class:`~ihdm.preprocess.qc.SubjectRecord`."""
    records: list[qc.SubjectRecord] = []
    for ref in refs:
        sidecar = registration_mod.read_sidecar(paths.registered(ref.subject))
        records.append(
            qc.SubjectRecord(
                subject=ref.subject,
                source=str(sidecar.get("source", ref.source)),
                final_metric=float(sidecar["final_metric"]),
                iterations=int(sidecar["iterations"]),
                stop_condition=str(sidecar["stop_condition"]),
                hit_max_iterations=bool(sidecar["hit_max_iterations"]),
                fov_fraction=float(sidecar.get("fov_fraction_window", 1.0)),
            )
        )
    return records


def _select_subjects(passing: tuple[str, ...], n_subjects: int) -> list[str]:
    """Draw the first ``n_subjects`` of a seeded permutation of the passing subjects."""
    rng = np.random.default_rng(RNG_SEED)
    permuted = rng.permutation(sorted(passing))
    return sorted(permuted[:n_subjects].tolist())


def _build_images(
    selected: list[str],
    refs: list[RawVolumeRef],
    paths: CohortPaths,
    foreground: np.ndarray,
    geometry: TemplateGeometry,
    volume_path: Callable[[str], Path] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, float]]:
    """Cut, scale and quantise the slices of every selected subject.

    Parameters
    ----------
    selected : list[str]
        Subjects to slice, in the order they enter ``images.npy``.
    refs : list[RawVolumeRef]
        Raw references, for the ``source`` column.
    paths : CohortPaths
        Cache locations.
    foreground : np.ndarray
        Boolean foreground of the intensity rule.
    geometry : TemplateGeometry
        Window and planes.
    volume_path : Callable[[str], Path] | None
        Which cached volume to read per subject; ``paths.corrected`` after N4 and
        ``paths.registered`` without it (the default).

    Returns
    -------
    tuple[np.ndarray, list[dict[str, object]], dict[str, float]]
        The ``uint8`` stack, the index rows and the averaged intensity diagnostics.
    """
    volume_path = volume_path or paths.registered
    by_subject = {ref.subject: ref for ref in refs}
    stacks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    p99_values: list[float] = []
    clip_high: list[float] = []

    for subject in selected:
        volume = _load_registered(volume_path(subject), geometry)
        scaled, info = scale_intensity(volume, foreground)
        stack = slices_to_uint8(extract_slices(scaled, geometry))
        stacks.append(stack)
        p99_values.append(info["p99"])
        clip_high.append(info["clip_high"])
        source = by_subject[subject].source
        rows.extend(
            {
                "subject": subject,
                "slice": position,
                "z_mm": geometry.z_mm[position],
                "source": source,
            }
            for position in range(len(geometry.z_indices))
        )

    images = np.concatenate(stacks, axis=0)
    diagnostics = {
        "p99_median": float(np.median(p99_values)),
        "p99_min": float(np.min(p99_values)),
        "p99_max": float(np.max(p99_values)),
        "clip_high_mean": float(np.mean(clip_high)),
    }
    return images, rows, diagnostics


def _load_registered(path: Path, geometry: TemplateGeometry | None = None) -> np.ndarray:
    """Read a cached registered volume as an ``(i, j, k)`` array on the template grid."""
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if geometry is not None:
        assert_grid_matches_affine(image, geometry)
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0)


def _build_index(rows: list[dict[str, object]], splits: dict[str, object]) -> pd.DataFrame:
    """Assemble ``index.csv`` with the finest split label per row."""
    train = set(splits["train"])
    seed = set(splits["seed"])
    labels = [
        "train" if i in train else ("seed" if i in seed else "ref") for i in range(len(rows))
    ]
    return pd.DataFrame(
        {
            "idx": np.arange(len(rows), dtype=int),
            "subject": [row["subject"] for row in rows],
            "slice": [row["slice"] for row in rows],
            "z_mm": [row["z_mm"] for row in rows],
            "source": [row["source"] for row in rows],
            "split": labels,
        }
    )


def _build_meta(
    args: argparse.Namespace,
    paths: CohortPaths,
    cfg: RegistrationConfig,
    n4_cfg: N4Config,
    geometry: TemplateGeometry,
    gate: qc.GateResult,
    records: list[qc.SubjectRecord],
    selected: list[str],
    images: np.ndarray,
    diagnostics: dict[str, float],
    template: registration_mod.Template,
    unreadable: list[str],
    sites: dict[str, str],
    n4_summary: dict[str, object],
    reference_subjects: list[str] | None,
) -> DatasetMeta:
    """Assemble ``meta.json`` with every pipeline value and every count."""
    n4_prefix = "per registered volume: "
    if args.n4:
        n4_prefix += "N4 bias-field correction, then "
    parameters: dict[str, object] = {
        "orientation": ORIENTATION,
        "registration": cfg.to_json(),
        "registration_template": str(template.path),
        "registration_template_mask": str(template.mask_path),
        "registration_metric": "Mattes mutual information (negated, minimised)",
        "registration_initialiser": "CenteredTransformInitializer (MOMENTS)",
        "registration_transform": "Euler3D (6 dof, rigid)",
        "registration_resamplings": 1,
        "geometry": geometry.to_json(),
        "n4": n4_summary,
        "sites": dict(sorted(sites.items())),
        "site_rule": (
            "IXI: the site token of the raw file name (IXI012-HH-1211-T1.nii.gz -> HH); "
            "OASIS-1: a single scanner, every subject WashU"
        ),
        "intensity_rule": (
            f"{n4_prefix}foreground = template brain mask dilated "
            f"{cfg.mask_dilation_mm:g} mm; p99 of the foreground; clip(v / p99, 0, 1); no low "
            "clip, so the acquisition background noise is kept; round(x * 255) to uint8"
        ),
        "intensity_percentile": 99.0,
        "intensity_diagnostics": diagnostics,
        "quality_gate": gate.to_json(),
        "subject_selection": (
            f"the first {args.n_subjects} of np.random.default_rng({RNG_SEED})."
            "permutation(sorted(passing_subjects))"
        ),
        "split_rule": f"split_by_subject(rng_seed={RNG_SEED}, train_frac=0.8, n_seed=40)",
        "cache_root": str(paths.cache),
        "excluded_by_eye": list(args.exclude),
        "unreadable_subjects": list(unreadable),
        "sensitivity_copy": str(paths.sensitivity) if reference_subjects is not None else None,
        "cli": " ".join(sys.argv),
    }
    # `n4_cfg` is already inside `n4_summary`; keep the reference so a future reader sees the
    # object the run was configured with even when N4 was switched off.
    parameters["n4_config"] = n4_cfg.to_json()
    counts: dict[str, object] = {
        "subjects_match_sensitivity": (
            None if reference_subjects is None else sorted(selected) == reference_subjects
        ),
        "subjects_total": len(records) + len(unreadable),
        "subjects_unreadable": len(unreadable),
        "subjects_passed_registration": len(gate.passing),
        "subjects_used": len(selected),
        "subjects_failed_registration": len(gate.failed),
        "subjects_hit_max_iterations": len(gate.capped),
        "slices_per_subject": len(geometry.z_indices),
        "padding_fraction": float((images == 0).mean()),
        "fov_padding_fraction": _fov_padding_fraction(records, selected),
        "saturated_fraction": float((images == 255).mean()),
        "mean_intensity": float(images.mean()),
    }
    return DatasetMeta(
        dataset_id=args.cohort,
        n_images=int(images.shape[0]),
        image_size=int(images.shape[1]),
        dtype="uint8",
        pipeline="ihdm.preprocess.mri",
        pipeline_version=PIPELINE_VERSION,
        git_sha=_git_sha(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_root=str(paths.raw),
        parameters=parameters,
        counts=counts,
    )


def _fov_padding_fraction(records: list[qc.SubjectRecord], selected: list[str]) -> float:
    """Mean fraction of the stored window that falls outside the acquisition.

    This is the true left-right padding of the sagittal acquisitions, as opposed to
    ``padding_fraction``, which counts every exactly-zero pixel and therefore also counts
    the background noise that the uint8 quantisation rounds down to zero.

    Parameters
    ----------
    records : list[qc.SubjectRecord]
        Registration records of the whole cohort.
    selected : list[str]
        Subjects that entered the dataset.

    Returns
    -------
    float
        Mean of ``1 - fov_fraction`` over the selected subjects.
    """
    chosen = set(selected)
    values = [1.0 - r.fov_fraction for r in records if r.subject in chosen]
    return float(np.mean(values)) if values else 0.0


def _git_sha() -> str:
    """Return the current git commit, or ``"unknown"`` outside a repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_qc(
    paths: CohortPaths,
    cfg: RegistrationConfig,
    geometry: TemplateGeometry,
    template: registration_mod.Template,
    records: list[qc.SubjectRecord],
    gate: qc.GateResult,
    refs: list[RawVolumeRef],
    selected: list[str],
    images: np.ndarray,
    index: pd.DataFrame,
    extra: dict[str, object],
) -> list[Path]:
    """Draw the five quality-control sheets and write the registration report."""
    qc_dir = paths.dataset / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    plane = len(geometry.z_indices) // 2

    template_array = sitk.GetArrayFromImage(template.image).transpose(2, 1, 0)
    template01 = _normalise_for_display(template_array, template.foreground)
    template_slices = extract_slices(template01, geometry)
    mask_array = sitk.GetArrayFromImage(template.brain_mask).transpose(2, 1, 0).astype(float)
    mask_slices = extract_slices(mask_array, geometry)

    tiles = rng.choice(
        np.array(selected), size=min(N_QC_REGISTRATION_TILES, len(selected)), replace=False
    ).tolist()
    by_subject_first_row = {s: i * len(geometry.z_indices) for i, s in enumerate(selected)}
    subject_slices = {s: images[by_subject_first_row[s] + plane] for s in sorted(tiles)}
    first_of_first = by_subject_first_row[selected[0]]

    written = [
        qc.registration_sheet(
            qc_dir / "registration_sheet.png", template_slices[plane], subject_slices,
            mask_slices[plane], geometry.z_mm[plane],
        ),
        qc.metric_distribution(qc_dir / "metric_distribution.png", records, gate),
        qc.registration_report(
            qc_dir / "registration_report.md", paths.cohort, records, gate, extra
        ),
        qc.levels_sheet(
            qc_dir / "levels_sheet.png",
            images[first_of_first: first_of_first + len(geometry.z_indices)],
            geometry.z_mm, selected[0],
        ),
        qc.intensity_sheet(qc_dir / "intensity_sheet.png", images, paths.cohort),
        _orientation_sheet(
            qc_dir, geometry, template_slices, plane, refs, selected, images,
            by_subject_first_row, template01,
        ),
    ]
    (qc_dir / "index_summary.json").write_text(
        json.dumps(
            {
                "split_counts": index["split"].value_counts().to_dict(),
                "subjects": len(selected),
                "slices_per_subject": len(geometry.z_indices),
            },
            indent=2,
        )
    )
    return written


def _normalise_for_display(volume: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Scale a volume to ``[0, 1]`` by its foreground p99, for the QC tiles only.

    Parameters
    ----------
    volume : np.ndarray
        Volume on the template grid.
    foreground : np.ndarray
        Boolean mask selecting the voxels the percentile is taken over.

    Returns
    -------
    np.ndarray
        The volume clipped into ``[0, 1]``.
    """
    cap = max(float(np.percentile(volume[foreground], 99.0)), 1e-6)
    return np.clip(volume / cap, 0.0, 1.0)


def _orientation_sheet(
    qc_dir: Path,
    geometry: TemplateGeometry,
    template_slices: np.ndarray,
    plane: int,
    refs: list[RawVolumeRef],
    selected: list[str],
    images: np.ndarray,
    first_row: dict[str, int],
    template01: np.ndarray,
) -> Path:
    """Draw the orientation sheet: the template plus four subjects, raw and registered."""
    by_subject = {ref.subject: ref for ref in refs}
    rows: list[dict[str, object]] = [
        {
            "label": "TEMPLATE",
            "raw": template01,
            "registered": template_slices[plane],
        }
    ]
    for subject in selected[: N_QC_ORIENTATION_ROWS]:
        raw_array, _ = sitk_to_ras_array(load_sitk(by_subject[subject]))
        cap = max(float(np.percentile(raw_array, 99.5)), 1e-6)
        rows.append(
            {
                "label": subject,
                "raw": np.clip(raw_array / cap, 0.0, 1.0),
                "registered": images[first_row[subject] + plane],
            }
        )
    return qc.orientation_sheet(qc_dir / "orientation_sheet.png", rows, geometry)


def _n4_sheet(
    paths: CohortPaths,
    selected: list[str],
    geometry: TemplateGeometry,
    template: registration_mod.Template,
    n4_cfg: N4Config,
) -> Path:
    """Draw the before/after N4 contact sheet: six subjects x (before, after, log field).

    The two image columns share one display scale per subject (the p99 of the *uncorrected*
    volume's foreground), so a change in the sheet is a change in the data and not in the
    window. The third column is the multiplicative field itself, on a symmetric diverging
    scale centred on 1.

    Parameters
    ----------
    paths : CohortPaths
        Cache and dataset locations.
    selected : list[str]
        The subjects of the dataset; the first :data:`N_QC_N4_ROWS` are drawn.
    geometry : TemplateGeometry
        Window and planes; the middle plane is shown.
    template : registration_mod.Template
        Supplies the foreground used for the display scale.
    n4_cfg : N4Config
        Printed in the sheet's title.

    Returns
    -------
    Path
        The written PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plane = len(geometry.z_indices) // 2
    subjects = selected[:N_QC_N4_ROWS]
    figure, axes = plt.subplots(
        len(subjects), 3, figsize=(7.5, 2.5 * len(subjects)), squeeze=False
    )
    for row, subject in enumerate(subjects):
        before = _load_registered(paths.registered(subject), geometry)
        after = _load_registered(paths.corrected(subject), geometry)
        cap = max(float(np.percentile(before[template.foreground], 99.0)), 1e-6)
        field = np.divide(before, after, out=np.ones_like(before), where=after > 1e-6)
        sidecar = bias_mod.read_n4_sidecar(paths.corrected(subject))
        panels = (
            (np.clip(before / cap, 0.0, 1.0), f"{subject} before", "gray", (0.0, 1.0)),
            (np.clip(after / cap, 0.0, 1.0), f"{subject} after N4", "gray", (0.0, 1.0)),
            (field, "bias field (before / after)", "RdBu_r", (0.8, 1.25)),
        )
        for column, (volume, title, cmap, limits) in enumerate(panels):
            ax = axes[row][column]
            image = extract_slices(volume.astype(np.float32), geometry)[plane]
            handle = ax.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 2:
                figure.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
        axes[row][2].set_xlabel(
            f"log field in mask [{sidecar['log_field_min']:.3f}, {sidecar['log_field_max']:.3f}]",
            fontsize=7,
        )
    figure.suptitle(
        f"{paths.cohort}: N4 bias-field correction, MNI z = {geometry.z_mm[plane]:.0f} mm "
        f"(shrink {n4_cfg.shrink_factor}, iterations {list(n4_cfg.max_iterations)})",
        fontsize=10,
    )
    out = paths.dataset / "qc" / "n4_before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out, dpi=110)
    plt.close(figure)
    logger.info("wrote %s", out)
    return out


__all__ = ["main", "move_to_sensitivity"]

if __name__ == "__main__":
    raise SystemExit(main())
