"""Turn one raw MRI cohort into the standard dataset format.

Usage
-----
``python -m ihdm.cli.preprocess_mri --cohort ixi|oasis1 [--workers 20] [--limit N]
[--force] [--out $IHDM_DATA_ROOT]``

Stages, in order: list the subjects, register every one of them onto the MNI152 template
(in parallel, cached on disk), apply the quality gate, draw 400 passing subjects with
``np.random.default_rng(2026)``, cut the ten windowed axial slices of each, split by
subject, write the dataset through :func:`ihdm.data.format.write_dataset`, validate it and
draw the quality-control sheets. Every stage except the registration is cheap, so a rerun
after a cache hit takes a couple of minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ihdm.data.format import DatasetMeta, split_by_subject, validate_dataset, write_dataset
from ihdm.paths import data_root, raw_root, template_dir
from ihdm.preprocess import qc
from ihdm.preprocess import registration as registration_mod
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
    slices_to_uint8,
    template_geometry,
    to_display_orientation,
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
    raw : Path
        Root of the unprocessed data tree.
    templates : Path
        Directory of the MNI152 template files.
    """

    cohort: str
    dataset: Path
    cache: Path
    raw: Path
    templates: Path

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
    paths = _cohort_paths(args)
    geometry = template_geometry(
        paths.templates / f"{registration_mod.TEMPLATE_STEM}_T1w.nii.gz",
        z_indices=SLICE_Z_INDICES,
        size=args.image_size,
    )
    refs = list_subjects(args.cohort, paths.raw, limit=args.limit)
    logger.info("cohort %s: %d subjects to register", args.cohort, len(refs))

    t_register = time.time()
    _register_cohort(refs, paths, cfg, workers=args.workers, force=args.force)
    register_seconds = time.time() - t_register

    records = _read_records(refs, paths)
    gate = qc.apply_gate(records)
    selected = _select_subjects(gate.passing, args.n_subjects)
    if len(selected) < args.n_subjects:
        logger.warning(
            "only %d of the requested %d subjects passed the gate in cohort %s; using all of them",
            len(selected), args.n_subjects, args.cohort,
        )

    template = registration_mod.load_template(paths.templates, cfg)
    assert_grid_matches_affine(template.image, geometry)
    images, index_rows, diagnostics = _build_images(
        selected, refs, paths, template.foreground, geometry
    )
    splits = split_by_subject([row["subject"] for row in index_rows], rng_seed=RNG_SEED)
    index = _build_index(index_rows, splits)
    meta = _build_meta(
        args, paths, cfg, geometry, gate, records, selected, images, diagnostics, template
    )

    write_dataset(paths.dataset, images, index, splits, meta)
    violations = validate_dataset(paths.dataset)

    qc_paths = _write_qc(
        paths, cfg, geometry, template, records, gate, refs, selected, images, index,
        {"registration wall time (s)": f"{register_seconds:.0f}",
         "subjects used": len(selected)},
    )

    padding = float((images == 0).mean())
    elapsed = time.time() - started
    print(
        f"{args.cohort}: registered {len(records)}, passed {len(gate.passing)}, "
        f"used {len(selected)} ({images.shape[0]} images), "
        f"metric median {gate.median:.4f} MAD {gate.mad:.4f}, "
        f"padding {padding:.2%}, wall {elapsed / 60:.1f} min -> {paths.dataset}"
    )
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
    return parser.parse_args(argv)


def _cohort_paths(args: argparse.Namespace) -> CohortPaths:
    """Resolve every directory the run touches."""
    out = Path(args.out) if args.out is not None else data_root()
    cache_root = Path(args.cache_root) if args.cache_root is not None else data_root() / "_cache"
    return CohortPaths(
        cohort=args.cohort,
        dataset=out / args.cohort,
        cache=cache_root / "registered" / args.cohort,
        raw=Path(args.raw_root) if args.raw_root is not None else raw_root(),
        templates=Path(args.template_dir) if args.template_dir is not None else template_dir(),
    )


def _init_worker(template_path: Path, cfg: RegistrationConfig) -> None:
    """Load the template once per worker process and pin SimpleITK to one thread."""
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    for variable in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[variable] = "1"
    _WORKER["template"] = registration_mod.load_template(template_path, cfg)
    _WORKER["cfg"] = cfg


def _register_one(task: tuple[RawVolumeRef, Path]) -> tuple[str, str]:
    """Register one subject and write it to the cache; return ``(subject, error)``."""
    ref, destination = task
    template = _WORKER["template"]
    cfg = _WORKER["cfg"]
    try:
        moving = load_sitk(ref)
        result = registration_mod.register(moving, template.image, template.dilated_mask, cfg)
        resampled = registration_mod.resample_to_template(
            moving, template.image, result.transform, cfg
        )
        registration_mod.write_registered(
            destination, resampled, result, cfg, ref.subject, ref.source
        )
    except (PreprocessError, RuntimeError, MemoryError) as exc:
        return ref.subject, f"{type(exc).__name__}: {exc}"
    return ref.subject, ""


def _register_cohort(
    refs: list[RawVolumeRef],
    paths: CohortPaths,
    cfg: RegistrationConfig,
    workers: int,
    force: bool,
) -> None:
    """Register every subject that is not already cached, in a process pool.

    Raises
    ------
    PreprocessError
        If any subject fails to register (the message lists the first few).
    """
    paths.cache.mkdir(parents=True, exist_ok=True)
    tasks = [
        (ref, paths.registered(ref.subject))
        for ref in refs
        if force or not _cached(paths.registered(ref.subject))
    ]
    logger.info("%d of %d subjects need registration", len(tasks), len(refs))
    if not tasks:
        return

    template_path = paths.templates
    errors: list[tuple[str, str]] = []
    done = 0
    started = time.time()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=max(1, workers), initializer=_init_worker, initargs=(template_path, cfg)
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
        raise PreprocessError(f"{len(errors)} subject(s) failed to register: {head}")


def _cached(path: Path) -> bool:
    """Return whether a registered volume and its sidecar are both on disk."""
    return path.is_file() and registration_mod.sidecar_path(path).is_file()


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
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, float]]:
    """Cut, scale and quantise the slices of every selected subject.

    Returns
    -------
    tuple[np.ndarray, list[dict[str, object]], dict[str, float]]
        The ``uint8`` stack, the index rows and the averaged intensity diagnostics.
    """
    by_subject = {ref.subject: ref for ref in refs}
    stacks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    p99_values: list[float] = []
    clip_high: list[float] = []

    for subject in selected:
        volume = _load_registered(paths.registered(subject), geometry)
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
    geometry: TemplateGeometry,
    gate: qc.GateResult,
    records: list[qc.SubjectRecord],
    selected: list[str],
    images: np.ndarray,
    diagnostics: dict[str, float],
    template: registration_mod.Template,
) -> DatasetMeta:
    """Assemble ``meta.json`` with every pipeline value and every count."""
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
        "intensity_rule": (
            "per registered volume: foreground = template brain mask dilated "
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
        "cli": " ".join(sys.argv),
    }
    counts: dict[str, object] = {
        "subjects_total": len(records),
        "subjects_passed_registration": len(gate.passing),
        "subjects_used": len(selected),
        "subjects_failed_registration": len(gate.failed),
        "slices_per_subject": len(geometry.z_indices),
        "padding_fraction": float((images == 0).mean()),
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


__all__ = ["main", "to_display_orientation"]

if __name__ == "__main__":
    raise SystemExit(main())
