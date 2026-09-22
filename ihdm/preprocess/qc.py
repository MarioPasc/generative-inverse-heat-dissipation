"""Quality-control sheets and the registration report of the MRI pipeline.

Five PNG sheets and one Markdown report per cohort, written into ``<dataset>/qc/``. They
are the eyeball gates of ``docs/HARNESSES/data.md`` §3: the registration sheet proves the
template outline sits on every subject's brain, the orientation sheet proves anterior is
at the top and the subject's left is on the image left for raw volumes, registered volumes
and the template alike, and the metric distribution plus the report make the quality gate
auditable subject by subject.

Only ``matplotlib`` with the ``Agg`` backend is used, so the sheets render on a headless
machine and inside a worker process.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ihdm.preprocess.errors import PreprocessError  # noqa: E402
from ihdm.preprocess.mri import TemplateGeometry, to_display_orientation  # noqa: E402

__all__ = [
    "GateResult",
    "SubjectRecord",
    "apply_gate",
    "intensity_sheet",
    "levels_sheet",
    "metric_distribution",
    "orientation_sheet",
    "registration_report",
    "registration_sheet",
]

logger = logging.getLogger(__name__)

_DPI = 130


@dataclass(frozen=True)
class SubjectRecord:
    """One subject's registration diagnostics, as read back from the cache sidecar.

    Parameters
    ----------
    subject : str
        Subject identifier.
    source : str
        Raw file path relative to the cohort's raw root.
    final_metric : float
        Metric at convergence (negative Mattes mutual information; lower is better).
    iterations : int
        Optimiser iterations at the finest level.
    stop_condition : str
        The optimiser's stop-condition description.
    hit_max_iterations : bool
        Whether the finest level exhausted the iteration cap.
    """

    subject: str
    source: str
    final_metric: float
    iterations: int
    stop_condition: str
    hit_max_iterations: bool


@dataclass(frozen=True)
class GateResult:
    """Outcome of the registration quality gate over one cohort.

    Parameters
    ----------
    median : float
        Median final metric.
    mad : float
        Raw median absolute deviation of the final metric.
    threshold : float
        ``median + 3 * mad``; a metric above it fails.
    threshold_scaled : float
        ``median + 3 * 1.4826 * mad``, reported for comparison only.
    passing : tuple[str, ...]
        Subjects that passed, sorted.
    failed_metric : tuple[str, ...]
        Subjects failed for a metric above ``threshold``, sorted.
    failed_iterations : tuple[str, ...]
        Subjects failed for exhausting the iteration cap, sorted.
    n_failed_scaled : int
        How many subjects the scaled-MAD threshold would have failed on the metric.
    """

    median: float
    mad: float
    threshold: float
    threshold_scaled: float
    passing: tuple[str, ...]
    failed_metric: tuple[str, ...]
    failed_iterations: tuple[str, ...]
    n_failed_scaled: int

    @property
    def failed(self) -> tuple[str, ...]:
        """Return every failing subject, sorted.

        Returns
        -------
        tuple[str, ...]
            Union of the metric and iteration failures.
        """
        return tuple(sorted(set(self.failed_metric) | set(self.failed_iterations)))

    def to_json(self) -> dict[str, object]:
        """Return the JSON-serialisable summary stored in ``meta.json``.

        Returns
        -------
        dict[str, object]
            Gate statistics and failure lists.
        """
        return {
            "rule": (
                "FAIL if final_metric > median + 3 * MAD (raw median absolute deviation) "
                "or the finest resolution level exhausted max_iterations"
            ),
            "metric_median": self.median,
            "metric_mad": self.mad,
            "metric_threshold": self.threshold,
            "metric_threshold_scaled_mad": self.threshold_scaled,
            "n_failed_scaled_mad": self.n_failed_scaled,
            "failed_metric": list(self.failed_metric),
            "failed_iterations": list(self.failed_iterations),
        }


def apply_gate(records: Sequence[SubjectRecord]) -> GateResult:
    """Apply the registration quality gate to a cohort.

    A subject fails when its final metric is worse (larger, since SimpleITK minimises the
    negative mutual information) than ``median + 3 * MAD``, or when the finest resolution
    level exhausted the iteration cap. ``MAD`` is the raw median absolute deviation; the
    consistency-scaled threshold is computed as well and reported, never applied.

    Parameters
    ----------
    records : Sequence[SubjectRecord]
        One record per registered subject.

    Returns
    -------
    GateResult
        Thresholds, passing subjects and the two failure lists.

    Raises
    ------
    PreprocessError
        If ``records`` is empty.
    """
    if not records:
        raise PreprocessError("the quality gate needs at least one registered subject")

    metrics = np.array([r.final_metric for r in records], dtype=float)
    median = float(np.median(metrics))
    mad = float(np.median(np.abs(metrics - median)))
    threshold = median + 3.0 * mad
    threshold_scaled = median + 3.0 * 1.4826 * mad

    failed_metric = sorted(r.subject for r in records if r.final_metric > threshold)
    failed_iterations = sorted(r.subject for r in records if r.hit_max_iterations)
    failed = set(failed_metric) | set(failed_iterations)
    passing = sorted(r.subject for r in records if r.subject not in failed)
    n_failed_scaled = int((metrics > threshold_scaled).sum())

    logger.info(
        "gate: median %.4f MAD %.4f threshold %.4f -> %d pass, %d fail",
        median, mad, threshold, len(passing), len(failed),
    )
    return GateResult(
        median=median,
        mad=mad,
        threshold=threshold,
        threshold_scaled=threshold_scaled,
        passing=tuple(passing),
        failed_metric=tuple(failed_metric),
        failed_iterations=tuple(failed_iterations),
        n_failed_scaled=n_failed_scaled,
    )


def _imshow(axis: plt.Axes, image: np.ndarray, title: str = "", cmap: str = "gray") -> None:
    """Draw one grayscale tile with no ticks and an optional small title."""
    axis.imshow(image, cmap=cmap, origin="upper", interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    if title:
        axis.set_title(title, fontsize=7)


def _label_axes(axis: plt.Axes, top: str, bottom: str, left: str, right: str) -> None:
    """Write the four anatomical direction labels around a tile."""
    style = {
        "color": "yellow",
        "fontsize": 11,
        "fontweight": "bold",
        "bbox": {"facecolor": "black", "alpha": 0.45, "pad": 1.0, "edgecolor": "none"},
    }
    axis.text(0.5, 0.985, top, ha="center", va="top", transform=axis.transAxes, **style)
    axis.text(0.5, 0.015, bottom, ha="center", va="bottom", transform=axis.transAxes, **style)
    axis.text(0.015, 0.5, left, ha="left", va="center", transform=axis.transAxes, **style)
    axis.text(0.985, 0.5, right, ha="right", va="center", transform=axis.transAxes, **style)


def registration_sheet(
    path: Path,
    template_slice: np.ndarray,
    subject_slices: dict[str, np.ndarray],
    mask_slice: np.ndarray,
    z_mm: float,
) -> Path:
    """Draw the template brain-mask outline over the template and ten registered subjects.

    Parameters
    ----------
    path : Path
        Destination PNG.
    template_slice : np.ndarray
        The template itself through the same slicing code, in the stored orientation.
    subject_slices : dict[str, np.ndarray]
        Subject id to its stored slice at the same plane.
    mask_slice : np.ndarray
        The template brain mask at the same plane, in the stored orientation.
    z_mm : float
        MNI z coordinate of the plane, for the title.

    Returns
    -------
    Path
        ``path``.
    """
    tiles: list[tuple[str, np.ndarray]] = [("TEMPLATE", template_slice)]
    tiles += sorted(subject_slices.items())
    n_cols = 6
    n_rows = int(np.ceil(len(tiles) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.15 * n_rows))
    for axis, (label, tile) in zip(np.ravel(axes), tiles, strict=False):
        _imshow(axis, tile, title=label)
        axis.contour(mask_slice.astype(float), levels=[0.5], colors="lime", linewidths=0.7)
    for axis in np.ravel(axes)[len(tiles):]:
        axis.axis("off")
    figure.suptitle(
        f"Registration gate: template brain-mask outline on the registered subjects "
        f"(axial plane, MNI z = {z_mm:+.0f} mm)",
        fontsize=9,
    )
    return _save(figure, path)


def metric_distribution(path: Path, records: Sequence[SubjectRecord], gate: GateResult) -> Path:
    """Plot the distribution of the final metric and the iteration counts.

    Parameters
    ----------
    path : Path
        Destination PNG.
    records : Sequence[SubjectRecord]
        One record per registered subject.
    gate : GateResult
        The applied gate, for the threshold lines.

    Returns
    -------
    Path
        ``path``.
    """
    metrics = np.array([r.final_metric for r in records], dtype=float)
    iterations = np.array([r.iterations for r in records], dtype=int)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 3.6))
    left.hist(metrics, bins=40, color="0.4")
    left.axvline(gate.median, color="tab:blue", lw=1.2, label=f"median {gate.median:.4f}")
    left.axvline(
        gate.threshold, color="tab:red", lw=1.2,
        label=f"median + 3 MAD = {gate.threshold:.4f}",
    )
    left.axvline(
        gate.threshold_scaled, color="tab:orange", lw=1.0, ls="--",
        label=f"median + 3(1.4826 MAD) = {gate.threshold_scaled:.4f}",
    )
    left.set_xlabel("final metric (negative Mattes MI; lower is better)")
    left.set_ylabel("subjects")
    left.legend(fontsize=6.5)

    right.scatter(iterations, metrics, s=6, color="0.3")
    right.axhline(gate.threshold, color="tab:red", lw=1.0)
    right.set_xlabel("optimiser iterations at the finest level")
    right.set_ylabel("final metric")

    figure.suptitle(
        f"Registration quality: {len(records)} subjects, "
        f"{len(gate.failed)} failed ({len(gate.failed) / len(records):.1%})",
        fontsize=9,
    )
    return _save(figure, path)


def registration_report(
    path: Path,
    cohort: str,
    records: Sequence[SubjectRecord],
    gate: GateResult,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write the per-subject Markdown registration report.

    Parameters
    ----------
    path : Path
        Destination ``.md``.
    cohort : str
        Cohort identifier, for the heading.
    records : Sequence[SubjectRecord]
        One record per registered subject.
    gate : GateResult
        The applied gate.
    extra : dict[str, object] | None
        Extra key/value rows appended to the summary (wall time, selection, ...).

    Returns
    -------
    Path
        ``path``.
    """
    failed = set(gate.failed)
    lines = [
        f"# Registration report — {cohort}",
        "",
        "## Summary",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| subjects registered | {len(records)} |",
        f"| passed the gate | {len(gate.passing)} |",
        f"| failed the gate | {len(failed)} ({len(failed) / len(records):.2%}) |",
        f"| failed on the metric | {len(gate.failed_metric)} |",
        f"| failed on max iterations | {len(gate.failed_iterations)} |",
        f"| metric median | {gate.median:.6f} |",
        f"| metric MAD (raw) | {gate.mad:.6f} |",
        f"| FAIL threshold, median + 3 MAD | {gate.threshold:.6f} |",
        f"| threshold with the scaled MAD (reported, not applied) | {gate.threshold_scaled:.6f} |",
        f"| subjects the scaled-MAD threshold would fail | {gate.n_failed_scaled} |",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "The metric is the negative Mattes mutual information SimpleITK minimises, so a "
        "**lower** value is a better alignment and a subject fails when its value is "
        "**above** the threshold.",
        "",
        "## Failures",
        "",
    ]
    if failed:
        lines += [
            "| subject | metric | iterations | stop condition | reason |",
            "|---|---|---|---|---|",
        ]
        for record in sorted(records, key=lambda r: -r.final_metric):
            if record.subject not in failed:
                continue
            reasons = []
            if record.subject in gate.failed_metric:
                reasons.append("metric")
            if record.subject in gate.failed_iterations:
                reasons.append("max iterations")
            lines.append(
                f"| {record.subject} | {record.final_metric:.6f} | {record.iterations} | "
                f"{record.stop_condition} | {', '.join(reasons)} |"
            )
    else:
        lines.append("None.")

    lines += [
        "", "## All subjects", "",
        "| subject | metric | iterations | gate | stop condition |",
        "|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda r: r.subject):
        verdict = "FAIL" if record.subject in failed else "pass"
        lines.append(
            f"| {record.subject} | {record.final_metric:.6f} | {record.iterations} | "
            f"{verdict} | {record.stop_condition} |"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def orientation_sheet(
    path: Path,
    rows: Sequence[dict[str, object]],
    geometry: TemplateGeometry,
) -> Path:
    """Draw raw mid-planes and the registered slice with anatomical labels.

    One row per entry of ``rows``; each entry carries ``label`` and the RAS-ordered raw
    array ``raw`` (optional) plus the registered stored slice ``registered``.

    Parameters
    ----------
    path : Path
        Destination PNG.
    rows : Sequence[dict[str, object]]
        Rows to draw; the first is expected to be the template.
    geometry : TemplateGeometry
        Supplies the plane's MNI z coordinate for the column title.

    Returns
    -------
    Path
        ``path``.
    """
    n_rows = len(rows)
    figure, axes = plt.subplots(n_rows, 4, figsize=(9.2, 2.45 * n_rows), squeeze=False)
    titles = (
        "raw axial mid-plane",
        "raw coronal mid-plane",
        "raw sagittal mid-plane",
        f"registered, stored slice 5 (z = {geometry.z_mm[5]:+.0f} mm)",
    )
    for row_index, row in enumerate(rows):
        label = str(row["label"])
        raw = row.get("raw")
        panels = _raw_mid_planes(np.asarray(raw)) if raw is not None else [None, None, None]
        panels.append(np.asarray(row["registered"]))
        labels = [("A", "P", "L", "R"), ("S", "I", "L", "R"), ("S", "I", "A", "P"),
                  ("A", "P", "L", "R")]
        for col, (tile, direction) in enumerate(zip(panels, labels, strict=True)):
            axis = axes[row_index][col]
            if tile is None:
                axis.axis("off")
                continue
            _imshow(axis, tile, title=titles[col] if row_index == 0 else "")
            _label_axes(axis, *direction)
            if col == 0:
                axis.set_ylabel(label, fontsize=7)
        axes[row_index][0].set_ylabel(label, fontsize=7)
    figure.suptitle(
        "Orientation gate: anterior at the top, subject-left on the image left "
        "(A/P/L/R, S/I labels drawn on every tile)",
        fontsize=9,
    )
    return _save(figure, path)


def _raw_mid_planes(volume_ras: np.ndarray) -> list[np.ndarray]:
    """Return the axial, coronal and sagittal mid-planes of a RAS-ordered volume.

    Each panel is rendered in the same convention as the stored slices: the axial panel
    has anterior at the top and the subject's left on the image left, the coronal panel
    superior at the top and the subject's left on the image left, and the sagittal panel
    superior at the top and anterior on the image left.
    """
    n_i, n_j, n_k = volume_ras.shape
    axial = to_display_orientation(volume_ras[:, :, n_k // 2])
    # (i, k) -> rows = k descending (S at the top), columns = i (L to R).
    coronal = np.ascontiguousarray(volume_ras[:, n_j // 2, :].T[::-1, :])
    # (j, k) -> rows = k descending (S at the top), columns = j descending (A on the left).
    sagittal = np.ascontiguousarray(volume_ras[n_i // 2, :, :].T[::-1, ::-1])
    return [axial, coronal, sagittal]


def levels_sheet(path: Path, slices: np.ndarray, z_mm: Sequence[float], subject: str) -> Path:
    """Draw one subject's ten stored slices in order.

    Parameters
    ----------
    path : Path
        Destination PNG.
    slices : np.ndarray
        Stack of shape ``(n, size, size)`` in the stored orientation.
    z_mm : Sequence[float]
        MNI z coordinate of each plane.
    subject : str
        Subject identifier, for the title.

    Returns
    -------
    Path
        ``path``.
    """
    n = len(slices)
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.15 * n_rows))
    for index, axis in enumerate(np.ravel(axes)):
        if index >= n:
            axis.axis("off")
            continue
        _imshow(axis, slices[index], title=f"slice {index} · z = {z_mm[index]:+.0f} mm")
    figure.suptitle(f"Axial levels of subject {subject} (stored orientation)", fontsize=9)
    return _save(figure, path)


def intensity_sheet(path: Path, images: np.ndarray, cohort: str) -> Path:
    """Draw the uint8 histogram of a cohort and report the exactly-zero fraction.

    Parameters
    ----------
    path : Path
        Destination PNG.
    images : np.ndarray
        The cohort's ``uint8`` image stack.
    cohort : str
        Cohort identifier, for the title.

    Returns
    -------
    Path
        ``path``.
    """
    values = np.asarray(images).reshape(-1)
    counts = np.bincount(values, minlength=256).astype(float)
    zero_fraction = counts[0] / counts.sum()
    saturated = counts[255] / counts.sum()

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 3.4))
    left.bar(np.arange(256), counts / counts.sum(), width=1.0, color="0.35")
    left.set_yscale("log")
    left.set_xlabel("stored uint8 value")
    left.set_ylabel("fraction of pixels (log)")
    left.set_title("full histogram", fontsize=8)

    per_image_zero = (np.asarray(images) == 0).reshape(len(images), -1).mean(axis=1)
    right.hist(per_image_zero, bins=40, color="0.35")
    right.set_xlabel("fraction of exactly-zero pixels per image")
    right.set_ylabel("images")
    right.set_title("padding / background-at-zero per image", fontsize=8)

    figure.suptitle(
        f"Intensity of {cohort}: exactly zero {zero_fraction:.2%} of all pixels, "
        f"saturated at 255 {saturated:.2%}, mean {values.mean():.1f}",
        fontsize=9,
    )
    return _save(figure, path)


def _save(figure: plt.Figure, path: Path) -> Path:
    """Write a figure to ``path`` and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=_DPI)
    plt.close(figure)
    logger.info("wrote %s", path)
    return path
