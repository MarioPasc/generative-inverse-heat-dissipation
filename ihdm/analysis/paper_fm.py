"""FM of the course paper: the method figure (T8.4), IHDM in the frequency language of the paper.

The ticket is ``docs/SPECIFICATIONS/M8-paper/T8.4-method-figure.md``. Three parts:

* **the top box**: one IXI slice written as :math:`\\mathbf x=\\sum_{i,j}w_{i,j}\\varphi_{i,j}`,
  with :math:`\\varphi_{i,j}` the orthonormal 2-D DCT-II basis images (192 × 192) and
  :math:`w_{i,j}` the DCT coefficients. Two representative modes per octave, framed in the octave
  colour, and an inset of the :math:`(i,j)` index plane in which each octave is a quarter-annulus
  (radius on a log scale, so that all eight octaves are visible);
* **the forward and reverse rows**: noise-free heat states :math:`u(\\sigma_b)` of the same slice,
  one macro-step per octave. The forward arrows carry ``−(octave)`` and
  :math:`q(\\mathbf u_k\\mid\\mathbf u_0)`; the reverse row is the mirror (right to left) and its
  arrows carry ``+(octave)`` and :math:`p_\\theta(\\mathbf u_{k-1}\\mid\\mathbf u_k)`. It shows the
  ideal reverse path, i.e. the forward states read backwards. The two finest octaves (32–96
  c/img) are folded into an ellipsis; both priors are marked;
* **the vertical panel**: IXI's between-image variance per octave (fine at the top, in the order
  the forward row removes them) against the :math:`1/f^2` equal share, the reverse levels each
  schedule gives each octave, and what the schedule does.

The macro-step. ``DCTBlur`` multiplies the DCT coefficient of mode :math:`(i,j)` by
:math:`d=\\exp(-\\lambda_{i,j}\\sigma_B^2/2)` with
:math:`\\lambda_{i,j}=(\\pi i/W)^2+(\\pi j/W)^2`. A mode with radial index
:math:`n=\\sqrt{i^2+j^2}` carries :math:`c=n/2` c/img, so :math:`\\lambda=(2\\pi c/W)^2` and

.. math:: d=\\exp(-\\sigma_B^2/\\sigma_n^2),\\qquad
          \\sigma_n=\\sqrt{2/\\lambda}=W/(\\sqrt2\\,\\pi c)\\approx 43.2/c\\ \\text{px}.

At :math:`\\sigma_b=43.2/c_b` a mode at the octave's lower edge :math:`c_b` keeps
:math:`d=e^{-1}` and every mode at :math:`c\\ge 2c_b` keeps
:math:`d=\\exp(-(c/c_b)^2)\\le e^{-4}`. A blur level :math:`\\sigma_k` therefore acts on the
octave holding :math:`c_k=43.2/\\sigma_k`, and the arrow from :math:`u(\\sigma_b/2)` to
:math:`u(\\sigma_b)` spans exactly the levels that
:func:`ihdm.analysis.paper_f1.levels_per_frequency_octave` counts in that octave.

Every number on the figure is read from a file or computed: the octave shares
(``docs/RESULTS/data_profile/ixi.npz``), the level counts (``schedules/*.npy``, via ``paper_f1``)
and the mode counts (:func:`octave_masks`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Wedge
from matplotlib.text import Text
from scipy.fft import dctn

from ihdm.analysis import paper_f1 as f1
from ihdm.analysis.style import INK, MUTED, arm_style
from ihdm.data.format import read_dataset
from ihdm.spectral.errors import SpectralError
from ihdm.spectral.power import MAX_CYCLES_PER_IMAGE, octave_bins, octave_shares, radial_index
from ihdm.spectral.profile import reference_example

__all__ = [
    "DC_GREY",
    "ELIDED_OCTAVES",
    "FM_NAME",
    "FMData",
    "FMInputs",
    "FMLayoutReport",
    "MacroStepCheck",
    "MethodFigureError",
    "MissingMethodInputError",
    "OCTAVE_COLOURS",
    "caption_text",
    "check_markdown_numbers",
    "dct_basis_1d",
    "dct_mode",
    "dct_weights",
    "draw_fm",
    "fm_layout_report",
    "fm_markdown",
    "fm_style",
    "frequency_level_counts",
    "load_fm_data",
    "macro_step_sigma",
    "mode_cycles",
    "octave_masks",
    "octave_mode_counts",
    "representative_modes",
    "save_fm",
    "synthesise",
    "verify_macro_steps",
]

logger = logging.getLogger(__name__)

#: Output stem.
FM_NAME: str = "fm_method"
#: Height cap of the ticket (FM's own height, ``FM_HEIGHT_IN``, follows from the layout below).
FM_MAX_HEIGHT_IN: float = 3.8
#: Smallest raster resolution of the PDF the ticket allows.
MIN_PDF_DPI: float = 300.0
#: Smallest thumbnails the ticket allows.
MIN_MODE_THUMB_IN: float = 0.22
MIN_STATE_THUMB_IN: float = 0.40

#: Octave colours, coarse (0.5–1 c/img) to fine (64–96 c/img): one violet-purple hue
#: (OKLCH H = 305°, L 0.28 → 0.78, C ≤ 0.17), clear of the arm hues (blue 256°, orange 41°,
#: green 162°, amber 75°, pink 357°). Validated with the dataviz skill's ordinal check
#: (``validate_palette.js --ordinal``): monotone lightness, adjacent ΔL ≥ 0.06, single hue,
#: lightest step 2.05:1 on white.
OCTAVE_COLOURS: tuple[str, ...] = (
    "#3a015f", "#510d7f", "#642996", "#7940ad", "#8e56c5", "#a46cdd", "#ba82f5", "#cc9fff",
)
#: The DC mode (the image mean) belongs to no octave.
DC_GREY: str = "#9a9893"
#: Modes above 96 c/img (the corner of the index plane) belong to no octave either.
CORNER_GREY: str = "#ecebe7"
#: Fill of the vertical panel.
PANEL_FILL: str = "#f5f4f1"

#: Octaves shown in the top equation (coarsest first); the finer ones are "+ ⋯".
EQUATION_OCTAVES: int = 5
#: Octaves folded into the ellipsis of the rows (the finest two).
ELIDED_OCTAVES: int = 2
#: Mathtext font of FM only (the narrowest sans set; F1 and F2 keep the default DejaVu).
_FM_RC: dict[str, object] = {"mathtext.fontset": "stixsans"}

#: Tolerance of the macro-step identities (exact up to float64 round-off).
MACRO_ATOL: float = 1e-12


class MethodFigureError(f1.PaperFigureError):
    """FM cannot be drawn: an input is inconsistent or a verification fails."""


class MissingMethodInputError(MethodFigureError):
    """A required input file does not exist."""


# --------------------------------------------------------------------------------------------
# The DCT basis
# --------------------------------------------------------------------------------------------


def dct_basis_1d(n: int) -> np.ndarray:
    """Orthonormal 1-D DCT-II basis, built from its analytic formula.

    Row :math:`k` is :math:`\\alpha_k\\cos(\\pi(2m+1)k/(2n))`, :math:`m = 0..n-1`, with
    :math:`\\alpha_0=\\sqrt{1/n}` and :math:`\\alpha_k=\\sqrt{2/n}` otherwise; it is independent of
    ``scipy.fft``, against which the tests check it.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    numpy.ndarray
        Matrix ``C`` of shape ``(n, n)``; ``C @ C.T`` is the identity.

    Raises
    ------
    MethodFigureError
        If ``n`` is not positive.
    """
    if n < 1:
        raise MethodFigureError(f"the DCT basis needs n >= 1, got {n}")
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    basis = np.cos(np.pi * (2 * m + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    basis[0] = np.sqrt(1.0 / n)
    return basis


def dct_mode(width: int, i: int, j: int) -> np.ndarray:
    """Basis image :math:`\\varphi_{i,j}` of the orthonormal 2-D DCT-II.

    Parameters
    ----------
    width : int
        Image side in pixels.
    i, j : int
        Vertical (row) and horizontal (column) frequency index.

    Returns
    -------
    numpy.ndarray
        :math:`\\varphi_{i,j}[m,l]=C_{i,m}C_{j,l}`, shape ``(width, width)``.

    Raises
    ------
    MethodFigureError
        If an index lies outside ``[0, width)``.
    """
    if not (0 <= i < width and 0 <= j < width):
        raise MethodFigureError(f"mode ({i}, {j}) is outside a {width}² grid")
    basis = dct_basis_1d(width)
    return np.outer(basis[i], basis[j])


def dct_weights(image: np.ndarray) -> np.ndarray:
    """DCT coefficients :math:`w_{i,j}` of an image (``scipy.fft.dctn``, ``norm="ortho"``).

    Parameters
    ----------
    image : numpy.ndarray
        Square image ``(W, W)``.

    Returns
    -------
    numpy.ndarray
        Coefficients ``(W, W)``, ``float64``.
    """
    return dctn(np.asarray(image, dtype=np.float64), norm="ortho")


def synthesise(weights: np.ndarray) -> np.ndarray:
    """The explicit sum :math:`\\sum_{i,j}w_{i,j}\\varphi_{i,j}`, as :math:`C^\\top W C`.

    Parameters
    ----------
    weights : numpy.ndarray
        Coefficients ``(W, W)``.

    Returns
    -------
    numpy.ndarray
        The image ``(W, W)``.

    Raises
    ------
    MethodFigureError
        If the coefficients are not a square matrix.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise MethodFigureError("synthesis needs a square coefficient matrix")
    basis = dct_basis_1d(weights.shape[0])
    return basis.T @ weights @ basis


# --------------------------------------------------------------------------------------------
# Octaves and the macro-steps
# --------------------------------------------------------------------------------------------


def mode_cycles(width: int) -> np.ndarray:
    """Frequency :math:`c=\\sqrt{i^2+j^2}/2` of every mode, in cycles per image."""
    return radial_index(width) / 2.0


def octave_masks(width: int) -> list[np.ndarray]:
    """Boolean mask of each octave on the ``width`` grid (``05-metrics.md`` §1).

    The octaves are :math:`[c, 2c)` from 0.5 to 64 c/img and the last is :math:`[64, 96]`; the
    DC mode and the corner modes above 96 c/img belong to none.

    Parameters
    ----------
    width : int
        Image side in pixels.

    Returns
    -------
    list[numpy.ndarray]
        Eight masks of shape ``(width, width)``, coarsest first.
    """
    cycles = mode_cycles(width)
    masks = []
    for _, lo, hi in octave_bins():
        inside = (cycles >= lo) & (cycles < hi)
        if hi == MAX_CYCLES_PER_IMAGE:
            inside |= cycles == hi
        masks.append(inside)
    return masks


def octave_mode_counts(width: int) -> tuple[int, ...]:
    """Number of DCT modes in each octave."""
    return tuple(int(m.sum()) for m in octave_masks(width))


def representative_modes(low: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """The two modes shown for the octave starting at ``low`` c/img.

    They are :math:`(0, n_0)` at the octave's lower edge and the diagonal :math:`(n_0, n_0)` at
    its logarithmic centre (:math:`c=\\sqrt2\\,c_0`), with :math:`n_0 = 2c_0`.

    Parameters
    ----------
    low : float
        Lower edge of the octave in c/img; ``2 * low`` must be an integer.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        ``((0, n0), (n0, n0))``.

    Raises
    ------
    MethodFigureError
        If ``2 * low`` is not a positive integer.
    """
    n0 = 2.0 * float(low)
    if n0 < 1 or n0 != round(n0):
        raise MethodFigureError(f"octave edge {low} c/img is not a half-integer index")
    n = int(round(n0))
    return (0, n), (n, n)


def macro_step_sigma(low: float, width: int) -> float:
    """Blur :math:`\\sigma_b = 43.2/c_b` px at which the octave starting at ``low`` keeps e^-1."""
    return float(f1.mode_scale(width) / float(low))


@dataclass(frozen=True)
class MacroStepCheck:
    """The macro-step identities at one octave's state :math:`u(\\sigma_b)`.

    Attributes
    ----------
    label : str
        The octave, ``"lo-hi"`` c/img.
    low : float
        Its lower edge :math:`c_b`.
    sigma : float
        :math:`\\sigma_b = 43.2/c_b` px.
    d_edge : float
        :math:`d` of the modes exactly at :math:`c_b` (all equal).
    d_edge_released : float
        The same, measured by the released ``DCTBlur`` on :math:`\\varphi_{0,2c_b}`.
    n_edge : int
        Number of modes exactly at :math:`c_b`.
    ring_mean, ring_min, ring_max : float
        Mean, min and max of :math:`d` over the radial bin :math:`\\mathrm{round}(n) = 2c_b`.
    n_ring : int
        Modes in that bin.
    beyond_max : float
        Largest :math:`d` among the modes at :math:`c \\ge 2c_b`.
    n_beyond : int
        Number of those modes.
    octave_min : float
        Smallest :math:`d` inside the octave itself (at its upper edge).
    passed : bool
        :math:`d_\\mathrm{edge}=e^{-1}` (computed and released), the bin's range contains
        :math:`e^{-1}`, and :math:`\\max d \\le e^{-4}` beyond :math:`2c_b`.
    """

    label: str
    low: float
    sigma: float
    d_edge: float
    d_edge_released: float
    n_edge: int
    ring_mean: float
    ring_min: float
    ring_max: float
    n_ring: int
    beyond_max: float
    n_beyond: int
    octave_min: float
    passed: bool


def verify_macro_steps(width: int, released: bool = True) -> list[MacroStepCheck]:
    """Check the macro-step identities of the ticket at every octave's :math:`\\sigma_b`.

    Parameters
    ----------
    width : int
        Image side in pixels.
    released : bool
        Also measure :math:`d` at the edge with the released ``DCTBlur`` (imports torch).

    Returns
    -------
    list[MacroStepCheck]
        One record per octave, coarsest first.
    """
    cycles = mode_cycles(width)
    radius = radial_index(width)
    masks = octave_masks(width)
    checks = []
    for (label, low, _), mask in zip(octave_bins(), masks, strict=True):
        sigma = macro_step_sigma(low, width)
        d = f1.heat_multiplier(width, sigma)
        edge = np.isclose(cycles, low, rtol=0.0, atol=1e-12)
        ring = np.rint(radius) == 2.0 * low
        beyond = cycles >= 2.0 * low
        d_edge = float(d[edge].mean())
        edge_ok = bool(np.all(np.abs(d[edge] - np.exp(-1.0)) <= MACRO_ATOL))
        if released:
            i, j = representative_modes(low)[0]
            phi = dct_mode(width, i, j)
            d_rel = float(np.sum(f1.released_dct_blur(phi[None], sigma)[0] * phi))
        else:
            d_rel = d_edge
        beyond_max = float(d[beyond].max()) if beyond.any() else 0.0
        ok = (edge_ok and abs(d_rel - np.exp(-1.0)) <= 1e-9
              and float(d[ring].min()) <= np.exp(-1.0) <= float(d[ring].max())
              and beyond_max <= np.exp(-4.0) + MACRO_ATOL)
        checks.append(MacroStepCheck(
            label=label, low=float(low), sigma=sigma, d_edge=d_edge, d_edge_released=d_rel,
            n_edge=int(edge.sum()), ring_mean=float(d[ring].mean()),
            ring_min=float(d[ring].min()), ring_max=float(d[ring].max()),
            n_ring=int(ring.sum()), beyond_max=beyond_max, n_beyond=int(beyond.sum()),
            octave_min=float(d[mask].min()), passed=bool(ok)))
    return checks


def frequency_level_counts(schedule: np.ndarray, width: int) -> tuple[tuple[int, ...], int]:
    """Reverse levels per frequency octave, by ``paper_f1`` (level 0 dropped).

    Parameters
    ----------
    schedule : numpy.ndarray
        The ``(K + 1,)`` blur levels, level 0 first (0 px).
    width : int
        Image side in pixels.

    Returns
    -------
    tuple[tuple[int, ...], int]
        Eight counts (coarsest first) and the levels folded in from below 0.5 c/img.

    Raises
    ------
    MethodFigureError
        If level 0 is not 0.
    """
    schedule = np.asarray(schedule, dtype=np.float64)
    if schedule.ndim != 1 or schedule.size < 2 or schedule[0] != 0.0:
        raise MethodFigureError("a schedule must be 1-D with level 0 at 0 px")
    return f1.levels_per_frequency_octave(schedule[1:], width)


# --------------------------------------------------------------------------------------------
# Inputs and data
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FMInputs:
    """Every file FM reads.

    Attributes
    ----------
    data_root : Path
        The IHDM data root holding ``<dataset>/images.npy`` and its index.
    repo_root : Path
        Repository root holding ``docs/RESULTS`` and ``schedules``.
    dataset : str
        Dataset of the MRI slice (F1's rule).
    """

    data_root: Path
    repo_root: Path
    dataset: str = f1.DEFAULT_RULES.dataset

    @property
    def profile(self) -> Path:
        """``docs/RESULTS/data_profile/<dataset>.npz``."""
        return self.repo_root / "docs" / "RESULTS" / "data_profile" / f"{self.dataset}.npz"

    def schedule(self, name: str) -> Path:
        """``schedules/<name>.npy``."""
        return self.repo_root / "schedules" / f"{name}.npy"

    @property
    def data_profile_md(self) -> Path:
        """``docs/RESULTS/data_profile.md`` (the level-count table of §5)."""
        return self.repo_root / "docs" / "RESULTS" / "data_profile.md"

    def required(self) -> list[Path]:
        """Every input file, for the missing-input check."""
        folder = self.data_root / self.dataset
        files = [folder / n for n in ("images.npy", "index.csv", "splits.json", "meta.json")]
        files += [self.profile, self.data_profile_md]
        files += [self.schedule(name) for name in f1.SCHEDULES.values()]
        return files


@dataclass
class FMData:
    """Everything :func:`draw_fm` draws, already verified.

    Attributes
    ----------
    width : int
        Image side in pixels.
    example : numpy.ndarray
        The IXI slice, ``float64`` in [0, 1].
    example_index : int
        Its dataset index.
    octave_labels : tuple[str, ...]
        The eight octaves, ``"lo-hi"`` c/img, coarsest first.
    octave_edges : tuple[tuple[float, float], ...]
        Their ``(low, high)`` edges.
    shares : tuple[float, ...]
        IXI ``train`` between-image variance share per octave.
    mode_counts : tuple[int, ...]
        DCT modes per octave.
    states : dict[float, numpy.ndarray]
        Noise-free heat states :math:`u(\\sigma_b)` keyed by the octave's lower edge.
    level_counts : dict[str, tuple[int, ...]]
        Reverse levels per frequency octave, ``{"default": ..., "matched": ...}``.
    folded : dict[str, int]
        Levels folded into the first octave from below 0.5 c/img.
    sigma_counts : dict[str, tuple[int, ...]]
        Levels per sigma_B octave (``data_profile.md`` §5), per schedule name.
    n_levels : dict[str, int]
        K of each schedule.
    sources : dict[str, str]
        Human-readable source of each group of numbers.
    """

    width: int
    example: np.ndarray
    example_index: int
    octave_labels: tuple[str, ...]
    octave_edges: tuple[tuple[float, float], ...]
    shares: tuple[float, ...]
    mode_counts: tuple[int, ...]
    states: dict[float, np.ndarray]
    level_counts: dict[str, tuple[int, ...]]
    folded: dict[str, int]
    sigma_counts: dict[str, tuple[int, ...]]
    n_levels: dict[str, int]
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def lows(self) -> tuple[float, ...]:
        """Lower octave edges, coarsest first."""
        return tuple(lo for lo, _ in self.octave_edges)


def _load_shares(path: Path) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Octave labels and IXI ``train`` shares of the profile, checked against its power array."""
    with np.load(path, allow_pickle=False) as profile:
        missing = {"octave_labels", "octave_shares_train", "power_train"} - set(profile.files)
        if missing:
            raise MethodFigureError(f"{path}: missing {sorted(missing)}")
        labels = tuple(str(s) for s in profile["octave_labels"])
        shares = tuple(float(v) for v in profile["octave_shares_train"])
        power = np.asarray(profile["power_train"], dtype=np.float64)
    expected = tuple(label for label, _, _ in octave_bins())
    if labels != expected:
        raise MethodFigureError(f"{path}: octave labels {labels} != the bins {expected}")
    recomputed = tuple(octave_shares(power).values())
    if not np.allclose(shares, recomputed, rtol=1e-9, atol=1e-12):
        raise MethodFigureError(f"{path}: octave_shares_train disagrees with power_train")
    return labels, shares


def load_fm_data(inputs: FMInputs) -> FMData:
    """Select, verify and compute everything FM draws.

    Parameters
    ----------
    inputs : FMInputs
        The input paths.

    Returns
    -------
    FMData
        The figure's data.

    Raises
    ------
    MissingMethodInputError
        If an input is absent.
    MethodFigureError
        If a source file is inconsistent.
    paper_f1.PaperFigureError
        If a level count differs from ``data_profile.md`` §5 or from the F1 ticket.
    """
    missing = [p for p in inputs.required() if not p.exists()]
    if missing:
        raise MissingMethodInputError("missing input(s): " + ", ".join(str(p) for p in missing))
    images, _, _, _ = read_dataset(inputs.data_root / inputs.dataset)
    width = int(images.shape[-1])
    example, example_index = reference_example(
        inputs.data_root, inputs.dataset, np.random.default_rng(f1.DEFAULT_RULES.example_rng_seed))
    example = np.asarray(example, dtype=np.float64)
    labels, shares = _load_shares(inputs.profile)
    edges = tuple((float(lo), float(hi)) for _, lo, hi in octave_bins())

    schedules = {key: np.asarray(np.load(inputs.schedule(name)), dtype=np.float64)
                 for key, name in f1.SCHEDULES.items()}
    table = f1.parse_level_table(inputs.data_profile_md)
    try:
        sigma_counts = f1.verify_level_counts(
            {f1.SCHEDULES[k]: v for k, v in schedules.items()}, table)
    except SpectralError as exc:  # a level outside [0.5, 96] px: not a valid schedule
        raise MethodFigureError(f"invalid schedule: {exc}") from exc
    counts, folded, n_levels = {}, {}, {}
    for key, schedule in schedules.items():
        counts[key], folded[key] = frequency_level_counts(schedule, width)
        n_levels[key] = int(schedule.size - 1)
        if sum(counts[key]) != n_levels[key]:
            raise MethodFigureError(f"{f1.SCHEDULES[key]}: frequency-octave counts do not sum "
                                    f"to K = {n_levels[key]}")
    states = {lo: f1.heat_blur(example, macro_step_sigma(lo, width)) for lo, _ in edges}
    sources = {
        "octave shares": f"`{_rel(inputs.profile, inputs)}`, `octave_shares_train` (checked "
                         "against `octave_shares(power_train)`)",
        "level counts": "`levels_per_frequency_octave` (paper_f1) of "
                        + " and ".join(f"`schedules/{n}.npy`" for n in f1.SCHEDULES.values())
                        + "; per sigma_B octave checked against `docs/RESULTS/data_profile.md` §5",
        "mode counts": "`octave_masks(192)` (this module)",
        "image": f"`reference_example(root, \"{inputs.dataset}\", "
                 f"np.random.default_rng({f1.DEFAULT_RULES.example_rng_seed}))`",
    }
    return FMData(width=width, example=example, example_index=int(example_index),
                  octave_labels=labels, octave_edges=edges, shares=shares,
                  mode_counts=octave_mode_counts(width), states=states, level_counts=counts,
                  folded=folded, sigma_counts=sigma_counts, n_levels=n_levels, sources=sources)


def _rel(path: Path, inputs: FMInputs) -> str:
    try:
        return str(Path(path).relative_to(inputs.repo_root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------


@contextmanager
def fm_style() -> Iterator[None]:
    """The paper style of F1 plus FM's local mathtext font (``stixsans``)."""
    with f1.paper_style(), plt.rc_context(_FM_RC):
        yield


def _luminance(colour: str) -> float:
    """WCAG relative luminance of a ``#rrggbb`` colour."""
    rgb = np.array([int(colour[k:k + 2], 16) / 255.0 for k in (1, 3, 5)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(np.dot([0.2126, 0.7152, 0.0722], lin))


def _on(colour: str) -> str:
    """Ink or white, whichever contrasts more with ``colour`` (chip text)."""
    lum = _luminance(colour)
    ink = (lum + 0.05) / (_luminance(INK) + 0.05)
    white = 1.05 / (lum + 0.05)
    return INK if ink >= white else "white"


def _num(value: float) -> str:
    """``0.5``, ``1``, ``16``: an octave edge as printed."""
    return f"{value:g}"


def _octave_text(lo: float, hi: float) -> str:
    return f"{_num(lo)}\u2013{_num(hi)}"


class _Sheet:
    """Places axes, text and patches in inches on a fixed-size figure."""

    def __init__(self, width: float, height: float) -> None:
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height))

    def fx(self, x: float) -> float:
        return x / self.w

    def fy(self, y: float) -> float:
        return y / self.h

    def axes(self, x: float, y: float, w: float, h: float, gid: str | None = None,
             **kwargs: Any) -> Axes:
        ax = self.fig.add_axes((x / self.w, y / self.h, w / self.w, h / self.h), **kwargs)
        if gid:
            ax.set_gid(gid)
        return ax

    def text(self, x: float, y: float, s: str, **kwargs: Any) -> Text:
        return self.fig.text(x / self.w, y / self.h, s, **kwargs)

    def image(self, x: float, y: float, size: float, array: np.ndarray, role: str, gid: str,
              vmin: float = 0.0, vmax: float = 1.0, edge: str | None = None,
              lw: float = 1.0) -> Axes:
        ax = self.axes(x, y, size, size, gid=gid, label=f"{role}:{gid}")
        ax.imshow(np.clip(array, vmin, vmax), cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="none", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(edge is not None)
            if edge is not None:
                spine.set_edgecolor(edge)
                spine.set_linewidth(lw)
        return ax

    def arrow(self, start: tuple[float, float], end: tuple[float, float], colour: str = INK,
              lw: float = 0.8, scale: float = 6.0, style: str = "-|>",
              ls: str = "solid") -> FancyArrowPatch:
        patch = FancyArrowPatch((self.fx(start[0]), self.fy(start[1])),
                                (self.fx(end[0]), self.fy(end[1])),
                                transform=self.fig.transFigure, arrowstyle=style,
                                mutation_scale=scale, lw=lw, color=colour, shrinkA=0,
                                shrinkB=0, linestyle=ls)
        self.fig.add_artist(patch)
        return patch

    def line(self, xs: Sequence[float], ys: Sequence[float], **kwargs: Any) -> None:
        self.fig.add_artist(plt.Line2D([self.fx(x) for x in xs], [self.fy(y) for y in ys],
                                       transform=self.fig.transFigure, **kwargs))

    def box(self, x: float, y: float, w: float, h: float, rounding: float, **kwargs: Any) -> None:
        # Drawn in inches (dpi_scale_trans), so the corner rounding is isotropic.
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                               transform=self.fig.dpi_scale_trans, **kwargs)
        self.fig.add_artist(patch)

    def chip(self, x: float, y: float, s: str, colour: str, **kwargs: Any) -> Text:
        return self.text(x, y, s, ha="center", va="center", color=_on(colour), fontsize=7,
                         bbox={"boxstyle": "round,pad=0.22,rounding_size=0.45",
                               "facecolor": colour, "edgecolor": "none"}, **kwargs)


# Layout, in inches (figure 5.5 × FM_HEIGHT_IN).
_MAIN_X0, _MAIN_X1 = 0.02, 4.36
_PANEL_X0, _PANEL_X1 = 4.42, 5.49
_EDGE = 0.02
# Rows: the first state thumbnail's left edge, the thumbnail side, the right reserve for the
# default-prior marker, and the vertical stack (bottom to top).
_ROW_X0 = 0.47
_STATE = 0.42
_PRIOR_RESERVE = 0.12
_Y_PTHETA = 0.10
_Y_REV = 0.18
_Y_PLUS = _Y_REV + _STATE + 0.10
_Y_SIGMA = _Y_PLUS + 0.125
_Y_Q = _Y_SIGMA + 0.125
_Y_FWD = _Y_Q + 0.085
_Y_MINUS = _Y_FWD + _STATE + 0.10
_Y_PRIOR = _Y_MINUS + 0.105  # bottom of the two-line prior labels
# Top box: a header band, two lines of terms (thumbnail, weight, octave label), a bottom pad.
_BOX_Y0 = _Y_PRIOR + 0.27
_HEADER_H = 0.35
_TERM_LINE_H = 0.52
_BOX_H = _HEADER_H + 2 * _TERM_LINE_H + 0.06
_MODE = 0.26
_MRI = 0.62
_INSET = 0.80
#: FM's height, from the stack above.
FM_HEIGHT_IN: float = round(_BOX_Y0 + _BOX_H + _EDGE, 2)


def _top_box(sh: _Sheet, data: FMData) -> None:
    x0, x1 = _MAIN_X0, _MAIN_X1
    y0, y1 = _BOX_Y0, _BOX_Y0 + _BOX_H
    # Below every axes (zorder 0), or its white fill would hide the thumbnails.
    sh.box(x0, y0, x1 - x0, y1 - y0, rounding=0.08, facecolor="white", edgecolor=MUTED,
           lw=0.7, gid="top-box", zorder=-1)
    width = data.width
    # Header: the decomposition and the frequency of a mode.
    sh.text(x0 + 0.08, y1 - 0.05,
            r"$\mathbf{x}=\sum_{i,j}\,w_{i,j}\,\varphi_{i,j}$", ha="left", va="top", fontsize=8)
    # Two texts, not one with a line break: the square-root bar would touch the line above.
    sh.text(x0 + 0.80, y1 - 0.045,
            r"$\varphi_{i,j}$: orthonormal DCT-II basis ($192\times192$); $w_{i,j}$: weights;",
            ha="left", va="top", fontsize=7, color=MUTED)
    sh.text(x0 + 0.80, y1 - 0.24,
            r"mode $(i,j)$ carries $c=\frac{1}{2}\sqrt{i^2+j^2}$ cycles per image (c/img)",
            ha="left", va="center", fontsize=7, color=MUTED)

    # The terms: the DC mode, then two modes per octave, in two lines.
    amp = 2.0 / width  # one grey scale for all modes: the diagonal modes reach |phi| = 2/W
    terms: list[tuple[tuple[int, int], str, int]] = [((0, 0), DC_GREY, -1)]
    for k in range(EQUATION_OCTAVES):
        for mode in representative_modes(data.lows[k]):
            terms.append((mode, OCTAVE_COLOURS[k], k))
    split = 1 + 2 * 3  # DC and three octaves on the first line, two octaves on the second
    lines = (terms[:split], terms[split:])
    step = _MODE + 0.10
    x_mri = x0 + 0.08
    x_terms = x_mri + _MRI + 0.18
    line_tops = (y1 - _HEADER_H, y1 - _HEADER_H - _TERM_LINE_H)
    mri_y = (line_tops[0] + line_tops[1] - _MODE) / 2.0 - _MRI / 2.0
    sh.image(x_mri, mri_y, _MRI, data.example, role="state", gid="x")
    sh.text(x_mri + _MRI / 2, mri_y - 0.02, r"$\mathbf{x}$, one IXI slice", ha="center",
            va="top", fontsize=7)
    sh.text(x_mri + _MRI + 0.09, mri_y + _MRI / 2, "=", ha="center", va="center", fontsize=9)
    for line_no, line in enumerate(lines):
        yb = line_tops[line_no] - _MODE
        xs = [x_terms + 0.10 * line_no + n * step for n in range(len(line))]
        if line_no:
            sh.text(xs[0] - 0.05, yb + _MODE / 2, "+", ha="center", va="center", fontsize=8)
        for n, ((i, j), colour, _) in enumerate(line):
            sh.image(xs[n], yb, _MODE, dct_mode(width, i, j), role="mode", gid=f"phi-{i}-{j}",
                     vmin=-amp, vmax=amp, edge=colour, lw=1.6)
            sh.text(xs[n] + _MODE / 2, yb - 0.015, rf"$w_{{{i}{{,}}{j}}}$", ha="center",
                    va="top", fontsize=7)
            if n + 1 < len(line):
                sh.text(xs[n] + _MODE + (step - _MODE) / 2, yb + _MODE / 2, "+", ha="center",
                        va="center", fontsize=8)
        _octave_labels(sh, data, line, xs, yb - 0.135)
        if line_no == 1:
            xe = xs[-1] + _MODE + 0.04
            sh.text(xe, yb + _MODE / 2, r"$+\,\cdots\,+\,w_{i,j}\,\varphi_{i,j}\,+\,\cdots$",
                    ha="left", va="center", fontsize=7)
            lo, hi = data.octave_edges[EQUATION_OCTAVES][0], data.octave_edges[-1][1]
            sh.text(xe + 0.03, yb - 0.15, f"{_octave_text(lo, hi)} c/img", ha="left",
                    va="top", fontsize=7, color=MUTED)
    _inset(sh, data, x1 - 0.06 - _INSET, y0 + 0.08)


def _octave_labels(sh: _Sheet, data: FMData, line: Sequence[tuple[tuple[int, int], str, int]],
                   xs: Sequence[float], y: float) -> None:
    """Under each pair of modes, a bar in the octave colour and the octave; "mean" under DC."""
    n = 0
    while n < len(line):
        colour, k = line[n][1], line[n][2]
        if k < 0:
            sh.text(xs[n] + _MODE / 2, y, "mean", ha="center", va="top", fontsize=7, color=MUTED)
            n += 1
            continue
        lo, hi = data.octave_edges[k]
        sh.line([xs[n], xs[n + 1] + _MODE], [y] * 2, color=colour, lw=1.6,
                solid_capstyle="butt")
        unit = " c/img" if k == 0 else ""
        sh.text((xs[n] + xs[n + 1] + _MODE) / 2.0, y - 0.015, _octave_text(lo, hi) + unit,
                ha="center", va="top", fontsize=7)
        n += 2


def _log_radius(n: np.ndarray | float) -> np.ndarray | float:
    """Inset radius of radial index ``n``: 1 + log2(n); the DC disc is r < 1."""
    return 1.0 + np.log2(n)


def _inset(sh: _Sheet, data: FMData, x: float, y: float) -> None:
    ax = sh.axes(x, y, _INSET, _INSET, gid="index-plane")
    width = data.width
    r_corner = _log_radius(np.sqrt(2.0) * (width - 1))
    lim = r_corner * 1.02
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    # The corner of the index plane (c > 96 c/img): the image of the square beyond the last ring.
    theta = np.linspace(0.0, np.pi / 2.0, 181)
    n_max = (width - 1) / np.maximum(np.cos(theta), np.sin(theta))
    r_out = _log_radius(n_max)
    r_last = _log_radius(2.0 * data.octave_edges[-1][1])
    outer = np.column_stack([r_out * np.cos(theta), r_out * np.sin(theta)])
    inner = np.column_stack([r_last * np.cos(theta[::-1]), r_last * np.sin(theta[::-1])])
    ax.add_patch(Polygon(np.vstack([outer, inner]), closed=True, facecolor=CORNER_GREY,
                         edgecolor="none", zorder=1))
    for k, (lo, hi) in enumerate(data.octave_edges):
        r_in, r_hi = _log_radius(2.0 * lo), _log_radius(2.0 * hi)
        ax.add_patch(Wedge((0.0, 0.0), r_hi, 0.0, 90.0, width=r_hi - r_in,
                           facecolor=OCTAVE_COLOURS[k], edgecolor="white", lw=0.3, zorder=2))
    ax.add_patch(Wedge((0.0, 0.0), 1.0, 0.0, 90.0, facecolor=DC_GREY, edgecolor="white",
                       lw=0.3, zorder=2))
    # The modes drawn in the equation.
    for k in range(EQUATION_OCTAVES):
        for i, j in representative_modes(data.lows[k]):
            r = _log_radius(np.hypot(i, j))
            angle = np.arctan2(i, j)
            ax.plot([r * np.cos(angle)], [r * np.sin(angle)], marker="o", ms=2.2,
                    mfc="white", mec=INK, mew=0.4, zorder=4, clip_on=False)
    ax.plot([0.0, lim], [0.0, 0.0], color=MUTED, lw=0.5, zorder=3, clip_on=False)
    ax.plot([0.0, 0.0], [0.0, lim], color=MUTED, lw=0.5, zorder=3, clip_on=False)
    sh.text(x + _INSET + 0.01, y, r"$j$", ha="left", va="center", fontsize=7)
    sh.text(x, y + _INSET + 0.01, r"$i$", ha="center", va="bottom", fontsize=7)
    sh.text(x + _INSET / 2 + 0.05, y + _INSET + 0.10, "index plane,\nlog radius",
            ha="center", va="bottom", fontsize=7, color=MUTED, linespacing=1.0)


def _rows(sh: _Sheet, data: FMData) -> None:
    width = data.width
    n_oct = len(data.octave_edges)
    # The states shown, finest first: u(43.2 / c_b) of the octaves n_oct - ELIDED ... 0, i.e.
    # u(1.35 px) down to u(86.4 px); x -> u(1.35 px) is the ellipsis (64-96 and 32-64 removed).
    shown = list(range(n_oct - ELIDED_OCTAVES + 1))[::-1]
    pitch = (_MAIN_X1 - _PRIOR_RESERVE - _ROW_X0 - _STATE) / (len(shown) - 1)
    xs = [_ROW_X0 + n * pitch for n in range(len(shown))]
    gap_centres = [x + _STATE + (pitch - _STATE) / 2.0 for x in xs[:-1]]
    blue, orange = arm_style("A0").color, arm_style("A3").color

    for y, name in ((_Y_FWD, "forward"), (_Y_REV, "reverse")):
        sh.text(0.09, y, name, rotation=90, ha="center", va="bottom", fontsize=7, color=MUTED)
    sh.text(xs[0] - 0.03, _Y_SIGMA, r"$\sigma_B$:", ha="right", va="center", fontsize=7,
            color=MUTED)
    for n, k in enumerate(shown):
        lo = data.lows[k]
        state = data.states[lo]
        sh.image(xs[n], _Y_FWD, _STATE, state, role="state", gid=f"forward-{_num(lo)}")
        sh.image(xs[n], _Y_REV, _STATE, state, role="state", gid=f"reverse-{_num(lo)}")
        sigma = macro_step_sigma(lo, width)
        sh.text(xs[n] + _STATE / 2, _Y_SIGMA, f"{sigma:.3g} px", ha="center", va="center",
                fontsize=7, color=MUTED)
    # One macro-step per gap: the arrow into the state of octave k removes (adds back) octave k.
    q = r"$q(\mathbf{u}_k\mid\mathbf{u}_0)$"
    p = r"$p_\theta(\mathbf{u}_{k-1}\mid\mathbf{u}_k)$"
    for n, xc in enumerate(gap_centres):
        k = shown[n + 1]
        lo, hi = data.octave_edges[k]
        a, b = xs[n] + _STATE + 0.02, xs[n + 1] - 0.02
        sh.arrow((a, _Y_FWD + _STATE / 2), (b, _Y_FWD + _STATE / 2))
        sh.arrow((b, _Y_REV + _STATE / 2), (a, _Y_REV + _STATE / 2))
        sh.chip(xc, _Y_MINUS, f"\u2212({_octave_text(lo, hi)})", OCTAVE_COLOURS[k])
        sh.chip(xc, _Y_PLUS, f"+({_octave_text(lo, hi)})", OCTAVE_COLOURS[k])
        sh.text(xc, _Y_Q, q, ha="center", va="center", fontsize=7)
        sh.text(xc, _Y_PTHETA, p, ha="center", va="center", fontsize=7)
    # The ellipsis: x and the folded finest macro-steps, with what they remove (add back).
    fine_lo, fine_hi = data.octave_edges[n_oct - ELIDED_OCTAVES][0], data.octave_edges[-1][1]
    folded = f"({_octave_text(fine_lo, fine_hi)}\nc/img)"
    x_sym, x_dots = 0.21, 0.35
    for y in (_Y_FWD, _Y_REV):
        sh.text(x_sym, y + _STATE / 2, r"$\mathbf{x}$", ha="center", va="center", fontsize=8)
        sh.text(x_dots, y + _STATE / 2, "\u22ef", ha="center", va="center", fontsize=8)
    sh.text(x_dots - 0.05, _Y_MINUS + 0.015, "\u2212" + folded, ha="center", va="center",
            fontsize=7, color=MUTED, linespacing=1.0)
    sh.text(x_dots - 0.05, _Y_PLUS - 0.035, "+" + folded, ha="center", va="center", fontsize=7,
            color=MUTED, linespacing=1.0)

    # The priors: the default (W/2) beyond the coarsest state, the matched (W/8) just past the
    # state whose sigma_b is the largest below it (21.6 px), i.e. inside the 1-2 c/img step.
    x_default = xs[-1] + _STATE + 0.07
    n_matched = next(n for n, k in enumerate(shown)
                     if macro_step_sigma(data.lows[k], width) < f1.SIGMA_MATCHED
                     <= 2.0 * macro_step_sigma(data.lows[k], width))
    x_matched = xs[n_matched] + _STATE + 0.035
    priors = ((x_default, blue, "default prior (W/2)", f1.SIGMA_DEFAULT),
              (x_matched, orange, "matched prior (W/8)", f1.SIGMA_MATCHED))
    for x, colour, name, sigma in priors:
        for y in (_Y_FWD, _Y_REV):
            sh.line([x, x], [y - 0.02, y + _STATE + 0.02], color=colour, lw=1.1,
                    ls=(0, (2.0, 1.2)))
        sh.text(x + 0.03, _Y_PRIOR, f"{name}\n$\\sigma_B$ = {sigma:g} px", ha="right",
                va="bottom", fontsize=7, color=colour, linespacing=1.05)
    sh.text(0.66, _Y_PRIOR, "dashed: the priors,\nwhere generation starts", ha="left",
            va="bottom", fontsize=7, color=MUTED, linespacing=1.05)


def _panel(sh: _Sheet, data: FMData) -> None:
    x0, x1 = _PANEL_X0, _PANEL_X1
    y0, y1 = _EDGE, sh.h - _EDGE
    sh.box(x0, y0, x1 - x0, y1 - y0, rounding=0.06, facecolor=PANEL_FILL, edgecolor="none",
           gid="side-panel", zorder=-1)
    blue, orange = arm_style("A0").color, arm_style("A3").color
    n_oct = len(data.octave_edges)
    # Columns (inches from the panel's left edge): octave, bar and its value, the two counts.
    x_label, x_bar = x0 + 0.32, x0 + 0.35
    bar_len = 0.155  # the largest share
    x_value = x_bar + bar_len + 0.235  # right edge of the value column, clear of the 1/f² line
    x_def, x_mat = x0 + 0.835, x0 + 0.985
    row_h = 0.19
    y_rows_top = y1 - 0.70
    k_text = "/".join(sorted({f"{v}" for v in data.n_levels.values()}))
    sh.text(x0 + 0.04, y1 - 0.05, "per octave", ha="left", va="top", fontsize=7,
            fontweight="bold")
    sh.text(x0 + 0.04, y1 - 0.17, f"IXI variance\n(%) and levels\nof K = {k_text}",
            ha="left", va="top", fontsize=7, color=MUTED, linespacing=1.05)
    sh.text(x_label, y_rows_top + 0.03, "c/img", ha="right", va="bottom", fontsize=7,
            color=MUTED)
    sh.text(x_def, y_rows_top + 0.03, "default", rotation=90, ha="center", va="bottom",
            fontsize=7, color=blue)
    sh.text(x_mat, y_rows_top + 0.03, "matched", rotation=90, ha="center", va="bottom",
            fontsize=7, color=orange)
    top_share = max(data.shares)
    for r, k in enumerate(range(n_oct - 1, -1, -1)):  # fine at the top, as the forward row
        yc = y_rows_top - (r + 0.5) * row_h
        lo, hi = data.octave_edges[k]
        share = data.shares[k]
        length = bar_len * share / top_share
        sh.text(x_label, yc, _octave_text(lo, hi), ha="right", va="center", fontsize=7)
        sh.box(x_bar, yc - 0.06, max(length, 0.004), 0.12, rounding=0.0,
               facecolor=OCTAVE_COLOURS[k], edgecolor="none", gid=f"share-{lo:g}")
        sh.text(x_value, yc, f"{100.0 * share:.1f}", ha="right", va="center", fontsize=7)
        sh.text(x_def, yc, f"{data.level_counts['default'][k]}", ha="center", va="center",
                fontsize=7, color=blue)
        sh.text(x_mat, yc, f"{data.level_counts['matched'][k]}", ha="center", va="center",
                fontsize=7, color=orange)
    y_rows_bot = y_rows_top - n_oct * row_h
    x_eq = x_bar + bar_len * (1.0 / n_oct) / top_share
    sh.line([x_eq, x_eq], [y_rows_bot + 0.02, y_rows_top - 0.02], color=INK, lw=0.7,
            ls=(0, (1.0, 1.4)))
    sh.text(x_eq - 0.03, y_rows_bot - 0.01, f"{100.0 / n_oct:g}%: 1/f\u00b2", ha="left",
            va="top", fontsize=7, color=MUTED)
    sh.text(x0 + 0.04, y_rows_bot - 0.20,
            "Log spacing: equal\nlevels per octave,\nright for 1/f\u00b2.\n"
            "Matched: equal\nvariance per level,\nmore levels where\nIXI varies most.",
            ha="left", va="top", fontsize=7, linespacing=1.1)


def draw_fm(data: FMData) -> Figure:
    """Draw FM (5.5 in wide) from verified data.

    Parameters
    ----------
    data : FMData
        Output of :func:`load_fm_data`.

    Returns
    -------
    Figure
        The figure; save it with :func:`save_fm`.
    """
    with fm_style():
        sh = _Sheet(f1.PAPER_WIDTH_IN, FM_HEIGHT_IN)
        _top_box(sh, data)
        _rows(sh, data)
        _panel(sh, data)
    return sh.fig


def save_fm(fig: Figure, out_dir: Path) -> list[Path]:
    """Write ``fm_method.{pdf,svg,png}`` with ``paper_f1.save_f1`` under FM's style.

    Parameters
    ----------
    fig : Figure
        Output of :func:`draw_fm`.
    out_dir : Path
        Output folder, created if absent.

    Returns
    -------
    list[Path]
        PDF, SVG and PNG paths.
    """
    with fm_style():
        return f1.save_f1(fig, out_dir, name=FM_NAME)


@dataclass(frozen=True)
class FMLayoutReport:
    """``paper_f1.layout_report`` plus the two thumbnail classes of FM.

    Attributes
    ----------
    base : paper_f1.LayoutReport
        Size, smallest font, text overlaps and clipping.
    min_mode_in, min_state_in : float
        Smallest mode and state thumbnail side.
    min_image_dpi : float
        Lowest resolution of an embedded raster in the PDF and SVG, which embed each array
        unresampled: its pixels over its printed side.
    """

    base: f1.LayoutReport
    min_mode_in: float
    min_state_in: float
    min_image_dpi: float


def fm_layout_report(fig: Figure) -> FMLayoutReport:
    """Measure fonts, both thumbnail classes, text overlaps and clipping of a drawn figure.

    Parameters
    ----------
    fig : Figure
        A figure from :func:`draw_fm`, before saving.

    Returns
    -------
    FMLayoutReport
        The measurements.
    """
    with fm_style():
        base = f1.layout_report(fig)
    sizes: dict[str, list[float]] = {"mode": [], "state": []}
    dpis = []
    for ax in fig.axes:
        role = str(ax.get_label()).split(":", 1)[0]
        if role in sizes and ax.get_images():
            sizes[role].append(min(ax.bbox.width, ax.bbox.height) / fig.dpi)
        for im in ax.get_images():
            pixels = min(im.get_array().shape[:2])
            dpis.append(pixels / (max(ax.bbox.width, ax.bbox.height) / fig.dpi))
    return FMLayoutReport(base, min(sizes["mode"], default=0.0),
                          min(sizes["state"], default=0.0), min(dpis, default=0.0))


# --------------------------------------------------------------------------------------------
# The FM.md record
# --------------------------------------------------------------------------------------------


def _slash(values: Sequence[int]) -> str:
    return "/".join(str(v) for v in values)


def caption_text(data: FMData) -> str:
    """The caption draft (150–200 words), with every number taken from ``data``."""
    labels = data.octave_labels
    first = labels[0].replace("-", "\u2013")
    last = labels[-1].replace("-", "\u2013")
    n_shown = len(labels) - ELIDED_OCTAVES
    fine = f"{_num(data.octave_edges[n_shown][0])}\u2013{_num(data.octave_edges[-1][1])}"
    k_def = data.n_levels["default"]
    equal = 100.0 / len(labels)
    scale = f"{f1.mode_scale(data.width):.1f}"  # 43.2 at W = 192
    kept = f"{f1.mode_scale(data.width) / f1.SIGMA_MATCHED:.1f}"  # c of the matched prior
    floor = _num(data.octave_edges[0][0])
    return (
        "**IHDM, octave by octave.** Top: an IXI slice x is a weighted "
        "sum of orthonormal DCT-II modes φ_{i,j}; mode (i, j) carries c = ½√(i²+j²) cycles per "
        "image (c/img), and the modes fall into octaves (colours; the mean in grey), "
        "quarter-annuli of the (i, j) plane (inset, log radius). Rows: noise-free heat "
        "states of x. The forward process q(u_k | u_0) multiplies a mode at c by "
        f"d = exp(−σ_B²/σ_n²), σ_n ≈ {scale}/c px; each state is drawn at σ_B = {scale}/c_b, "
        "where "
        "the octave starting at c_b keeps d = e⁻¹ and every finer mode d ≤ e⁻⁴. One arrow is "
        f"therefore a macro-step over many of the K = {k_def} levels: default (log) "
        f"{_slash(data.level_counts['default'])} and matched "
        f"{_slash(data.level_counts['matched'])} levels per octave, {first} to {last} c/img "
        f"(right panel; {data.folded['default']} default levels below {floor} c/img in the "
        f"first). The ellipsis folds {fine} c/img. The reverse row is "
        "the ideal path of p_θ(u_{k−1} | u_k): the same states read backwards. Dashed: the "
        "priors, where generation starts; the default (W/2) keeps almost nothing, the matched "
        f"(W/8) the octaves below about {kept} c/img. Right: IXI's between-image variance per "
        f"octave against the {equal:g}% equal share of a 1/f² spectrum."
    )


def _words(text: str) -> int:
    return len(text.replace("*", " ").split())


def fm_markdown(data: FMData, checks: Sequence[MacroStepCheck], report: FMLayoutReport,
                hashes: dict[str, str], stable: dict[str, bool | None], command: str,
                eval_note: str) -> str:
    """The ``FM.md`` record: files, caption, mode counts, checks, sources and command.

    Parameters
    ----------
    data : FMData
        The drawn data.
    checks : Sequence[MacroStepCheck]
        Output of :func:`verify_macro_steps`.
    report : FMLayoutReport
        Output of :func:`fm_layout_report`.
    hashes : dict[str, str]
        SHA-256 per output file name.
    stable : dict[str, bool | None]
        Whether each file equals the previous run's (``None``: no previous file).
    command : str
        The command that wrote the record.
    eval_note : str
        What was done with ``--eval-dir``.

    Returns
    -------
    str
        The markdown text.
    """
    verdict = {None: "new (no previous file)", True: "yes", False: "**no**"}
    roles = {"pdf": "the version for LaTeX (vector, thumbnails embedded unresampled)",
             "svg": "editable master (Inkscape: text kept as text, thumbnails embedded)",
             "png": "preview, 300 dpi"}
    lines = [
        "# FM — the method figure",
        "",
        "Written by `ihdm.cli.paper_fm` (T8.4); do not edit by hand. Ticket: "
        "`docs/SPECIFICATIONS/M8-paper/T8.4-method-figure.md`.",
        "",
        "## Files",
        "",
        "| file | role | sha256 | identical to the previous run |",
        "|---|---|---|---|",
    ]
    for name, digest in hashes.items():
        lines.append(f"| `{name}` | {roles[name.rsplit('.', 1)[1]]} | `{digest[:16]}…` | "
                     f"{verdict[stable[name]]} |")
    base = report.base
    lines += [
        "",
        f"Size {base.size_in[0]:g} × {base.size_in[1]:g} in; smallest font "
        f"{base.min_font_pt:g} pt; smallest mode thumbnail {report.min_mode_in:.2f} in; smallest "
        f"state thumbnail {report.min_state_in:.2f} in; text boxes overlapping: "
        f"{len(base.overlaps)}; text outside the figure: {len(base.outside)}; embedded rasters "
        f"at {report.min_image_dpi:.0f} dpi or more (PDF, SVG). Formulas use "
        "mathtext `stixsans` (FM only; F1 and F2 use the default DejaVu mathtext).",
        "",
    ]
    caption = caption_text(data)
    lines += [f"## Caption draft ({_words(caption)} words)", "", caption, ""]
    lines += [
        "## Octaves: modes, variance and levels",
        "",
        "Modes per octave of the 192 × 192 grid (`octave_masks`): the octaves partition the "
        f"{sum(data.mode_counts):,} non-DC modes with c ≤ 96 c/img; the DC mode and "
        f"{data.width**2 - 1 - sum(data.mode_counts):,} corner modes above 96 c/img are in none.",
        "",
        "| octave (c/img) | modes | IXI variance share | default levels (`"
        f"{f1.SCHEDULES['default']}`) | matched levels (`{f1.SCHEDULES['matched']}`) |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, label in enumerate(data.octave_labels):
        lines.append(f"| {label.replace('-', '–')} | {data.mode_counts[k]:,} | "
                     f"{100.0 * data.shares[k]:.1f}% | {data.level_counts['default'][k]} | "
                     f"{data.level_counts['matched'][k]} |")
    lines.append(f"| total | {sum(data.mode_counts):,} | "
                 f"{100.0 * sum(data.shares):.1f}% | {sum(data.level_counts['default'])} | "
                 f"{sum(data.level_counts['matched'])} |")
    lines += [
        "",
        f"Levels per frequency octave: c_k = 43.2/σ_k binned on the octaves (levels below "
        f"0.5 c/img folded into the first: default {data.folded['default']}, matched "
        f"{data.folded['matched']}); the same function as F1's rugs. Levels per σ_B octave, "
        "checked against `data_profile.md` §5: "
        + " · ".join(f"`{name}` {_slash(c)}" for name, c in data.sigma_counts.items()) + ".",
        "",
        "## Macro-step check",
        "",
        "At σ_b = 43.2/c_b (the released `DCTBlur` multiplier d = exp(−λσ_B²/2), float64): the "
        "modes exactly at c_b keep d = e⁻¹ = 0.367879; the released `DCTBlur` applied to "
        "φ_{0,2c_b} gives the same; the radial bin round(n) = 2c_b brackets e⁻¹; every mode at "
        "c ≥ 2c_b keeps d ≤ e⁻⁴ = 0.018316.",
        "",
        "| octave (c/img) | σ_b (px) | d at c_b (modes) | d by DCTBlur | radial-bin mean d "
        "[min, max] (modes) | max d at c ≥ 2c_b (modes) | min d inside the octave | pass |",
        "|---|---:|---|---:|---|---|---:|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c.label.replace('-', '–')} | {c.sigma:.4g} | {c.d_edge:.6f} ({c.n_edge}) | "
            f"{c.d_edge_released:.6f} | {c.ring_mean:.4f} [{c.ring_min:.4f}, {c.ring_max:.4f}] "
            f"({c.n_ring}) | {c.beyond_max:.6f} ({c.n_beyond:,}) | {c.octave_min:.4g} | "
            f"{'yes' if c.passed else '**no**'} |")
    lines += [
        "",
        "## Selected image and sources",
        "",
        f"- **image:** dataset index {data.example_index}, "
        f"{data.sources.get('image', '')}; the states are noise-free heat states "
        "(`paper_f1.heat_blur`, the released `DCTBlur` formula), the same slice as F1.",
    ]
    for key in ("octave shares", "level counts", "mode counts"):
        lines.append(f"- **{key}:** {data.sources.get(key, '')}")
    lines += [
        f"- **eval tars:** {eval_note}",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    return "\n".join(lines)


def check_markdown_numbers(text: str, data: FMData) -> list[str]:
    """Strings of ``data`` that a written ``FM.md`` must contain (empty list: all present)."""
    wanted = [_slash(data.level_counts["default"]), _slash(data.level_counts["matched"])]
    wanted += [f"{c:,}" for c in data.mode_counts]
    return [w for w in wanted if not re.search(re.escape(w), text)]
