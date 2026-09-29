"""House style of the report figures (T6.2): arm palette, report widths, fonts, and saving.

Every figure of :mod:`ihdm.analysis.figures` takes its colours, markers, sizes and save routine
from here, so the same arm has the same colour and marker in every figure. The arm colours are
the first five slots of the ``dataviz`` skill's reference categorical palette, in its fixed order
(validated with ``scripts/validate_palette.js``: worst adjacent CVD ΔE 9.1, normal-vision ΔE 19.6
on the light surface). Three of them sit below 3:1 contrast on white, so every arm also carries
its own marker and every figure has a legend (the skill's relief rule).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = [
    "ARM_ORDER",
    "DATASET_LABELS",
    "DATASET_ORDER",
    "FULL_WIDTH_IN",
    "INK",
    "MUTED",
    "PNG_DPI",
    "SINGLE_WIDTH_IN",
    "ArmStyle",
    "arm_style",
    "figure_style",
    "save_figure",
]

logger = logging.getLogger(__name__)

#: Report template widths in inches (single column, full width).
SINGLE_WIDTH_IN: float = 3.3
FULL_WIDTH_IN: float = 6.75
#: Resolution of the PNG previews (the PDFs are vector).
PNG_DPI: int = 200

#: The arm order of every legend and axis (``configs.spectral.arms.ARMS``, the paper arm first).
ARM_ORDER: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A2p")
#: Panel order: development pair first (MRI, photographs), then the transfer pair.
DATASET_ORDER: tuple[str, ...] = ("ixi", "lsun_church", "oasis1", "lsun_bedroom")
DATASET_LABELS: dict[str, str] = {
    "ixi": "IXI T1",
    "lsun_church": "LSUN Churches",
    "oasis1": "OASIS-1 T1",
    "lsun_bedroom": "LSUN Bedrooms",
}

#: Text and reference-line inks; data marks carry the arm colours.
INK: str = "#0b0b0b"
MUTED: str = "#52514e"
GRID: str = "#e4e3df"
EXTENSION_SHADE: str = "#f0efec"


@dataclass(frozen=True)
class ArmStyle:
    """Colour, marker and legend label of one arm."""

    color: str
    marker: str
    label: str


_ARM_STYLES: dict[str, ArmStyle] = {
    "A0": ArmStyle("#2a78d6", "o", r"A0 ($\sigma_{B,\max}$ 96, log)"),
    "A3": ArmStyle("#eb6834", "s", r"A3 ($\sigma_{B,\max}$ 24, IXI-matched)"),
    "A1": ArmStyle("#1baf7a", "^", r"A1 ($\sigma_{B,\max}$ 24, log)"),
    "A2": ArmStyle("#eda100", "D", r"A2 ($\sigma_{B,\max}$ 96, IXI-matched)"),
    "A2p": ArmStyle("#e87ba4", "v", r"A2$'$ ($\sigma_{B,\max}$ 96, Churches-matched)"),
}


class StyleError(Exception):
    """An arm without a style was requested."""


def arm_style(arm: str) -> ArmStyle:
    """Return the fixed style of ``arm``.

    Parameters
    ----------
    arm : str
        Arm id as written in ``index.csv`` (``A0``, ``A1``, ``A2``, ``A3``, ``A2p``).

    Returns
    -------
    ArmStyle
        Colour, marker and legend label.

    Raises
    ------
    StyleError
        If the arm is not one of the five arms of the experiment.
    """
    try:
        return _ARM_STYLES[arm]
    except KeyError as exc:
        raise StyleError(f"no style for arm {arm!r}") from exc


_RC: dict[str, object] = {
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "text.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "ihdm-t6.2",
    "path.simplify": True,
}


@contextmanager
def figure_style() -> Iterator[None]:
    """Apply the report rcParams for the duration of the block."""
    with plt.rc_context(_RC):
        yield


def save_figure(fig: Figure, out_dir: Path, name: str, pdf: bool = True,
                png_dpi: int = PNG_DPI, pdf_dpi: int = 300) -> list[Path]:
    """Write ``<name>.pdf`` (vector, no creation date) and ``<name>.png``, then close the figure.

    Parameters
    ----------
    fig : Figure
        The figure to write.
    out_dir : Path
        Output directory; created if absent.
    name : str
        File stem.
    pdf : bool
        Write the PDF as well as the PNG preview.
    png_dpi : int
        Resolution of the PNG preview.
    pdf_dpi : int
        Resolution of the rasterised elements inside the PDF (dense scatters, embedded grids).

    Returns
    -------
    list[Path]
        The files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if pdf:
        path = out_dir / f"{name}.pdf"
        # No CreationDate/ModDate and a fixed Producer: byte-stable reruns on the same results.
        fig.savefig(path, format="pdf", dpi=pdf_dpi,
                    metadata={"CreationDate": None, "ModDate": None, "Creator": "ihdm.cli.figures",
                              "Producer": "matplotlib"})
        written.append(path)
    path = out_dir / f"{name}.png"
    fig.savefig(path, format="png", dpi=png_dpi, metadata={"Software": None})
    written.append(path)
    plt.close(fig)
    logger.info("wrote %s", ", ".join(p.name for p in written))
    return written
