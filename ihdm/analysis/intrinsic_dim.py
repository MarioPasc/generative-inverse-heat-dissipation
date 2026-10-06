"""Intrinsic dimension of the four training splits (T8.0): Levina–Bickel MLE and PCA measures.

The local (nonlinear) dimension is the maximum-likelihood estimator of Levina & Bickel (NeurIPS
2004) with the MacKay–Ghahramani (2005) average of the inverse local estimates,

    m_hat = [ (1/N) sum_i (1/(k-1)) sum_{j=1}^{k-1} log(T_k(x_i) / T_j(x_i)) ]^{-1},

where ``T_j(x)`` is the Euclidean distance from ``x`` to its ``j``-th nearest neighbour, the point
itself excluded. The linear measures are the PCA participation ratio
``(sum lambda)^2 / sum lambda^2`` and the number of components that hold 50%, 90% and 95% of the
variance, both from the eigenvalues of the Gram matrix of the centred data.

Each training split is read as 3,200 images of 192 x 192 grey levels, scaled to [0, 1], with
each image's mean (DC) removed and then centred across images. All distances of one dataset come
from one float64 Gram matrix; a random subset uses the corresponding block of the full distance
matrix, which equals the distances computed on the subset alone because Euclidean distances do
not depend on the centring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ihdm.analysis.tables import AnalysisError
from ihdm.paths import repo_root

__all__ = [
    "DATASETS",
    "MRI_DATASETS",
    "IDConfig",
    "IntrinsicDimError",
    "LinearMeasures",
    "MissingInputError",
    "analyse_dataset",
    "components_for_fraction",
    "distances_from_gram",
    "linear_measures",
    "load_train_matrix",
    "mle_intrinsic_dim",
    "participation_ratio",
    "plot_id_vs_n",
    "reading",
    "run",
]

logger = logging.getLogger(__name__)

#: The four training splits in the order of every table.
DATASETS: tuple[str, ...] = ("ixi", "oasis1", "lsun_church", "lsun_bedroom")
MRI_DATASETS: frozenset[str] = frozenset({"ixi", "oasis1"})
LABELS: dict[str, str] = {
    "ixi": "IXI T1",
    "oasis1": "OASIS-1 T1",
    "lsun_church": "LSUN Churches",
    "lsun_bedroom": "LSUN Bedrooms",
}

SCHEMA: str = "ihdm.intrinsic_dimension/1"
TOOL: str = "python -m ihdm.cli.intrinsic_dim (T8.0)"
FIGURE_NAME: str = "id_vs_n"
FIGURE_WIDTH_IN: float = 5.5
FIGURE_HEIGHT_IN: float = 2.0
MIN_FONT_PT: float = 7.0
SVG_HASHSALT: str = "ihdm-t8.0"
CREATOR: str = "ihdm.cli.intrinsic_dim"
PNG_DPI: int = 300
PDF_DPI: int = 300
#: k of the figure and of the reading.
K_MAIN: int = 10
#: Horizontal offset (multiplicative, log axis) of the one-slice-per-subject markers.
ONE_SLICE_X_SHIFT: float = 0.84


class IntrinsicDimError(AnalysisError):
    """The intrinsic-dimension analysis cannot proceed (bad data, bad configuration, bad figure)."""


class MissingInputError(IntrinsicDimError):
    """A dataset folder or one of its files is absent."""


@dataclass(frozen=True)
class IDConfig:
    """The frozen method of T8.0; the tests shrink it to run on small synthetic splits.

    Parameters
    ----------
    datasets : tuple[str, ...]
        Dataset folders under the data root, in table order.
    k_values : tuple[int, ...]
        Neighbourhood sizes of the MLE (each >= 2).
    n_values : tuple[int, ...]
        Sample sizes; the largest must equal the size of every training split.
    n_subsets : int
        Random subsets averaged for every N below the full split.
    seed : int
        Seed of the ``numpy.random.default_rng`` that draws the subsets, re-created per dataset.
    slice_index : int
        The slice kept by the MRI "one slice per subject" variant.
    fractions : tuple[float, ...]
        Variance fractions of the "components for x% of the variance" measure.
    """

    datasets: tuple[str, ...] = DATASETS
    k_values: tuple[int, ...] = (5, 10, 20)
    n_values: tuple[int, ...] = (320, 1000, 3200)
    n_subsets: int = 10
    seed: int = 2026
    slice_index: int = 5
    fractions: tuple[float, ...] = (0.5, 0.9, 0.95)

    def __post_init__(self) -> None:
        if min(self.k_values) < 2:
            raise IntrinsicDimError(f"every k must be >= 2, got {self.k_values}")
        if min(self.n_values) <= max(self.k_values):
            raise IntrinsicDimError(f"every N must exceed every k: {self.n_values}, "
                                    f"{self.k_values}")
        if self.n_subsets < 2:
            raise IntrinsicDimError("n_subsets must be >= 2 to report a standard deviation")
        if any(not 0.0 < f <= 1.0 for f in self.fractions):
            raise IntrinsicDimError(f"fractions must lie in (0, 1]: {self.fractions}")


@dataclass(frozen=True)
class LinearMeasures:
    """PCA measures of one dataset."""

    participation_ratio: float
    components: dict[str, int]
    total_variance: float
    rank: int
    eigenvalues_head: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Estimators


def distances_from_gram(gram: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance matrix of the rows whose Gram matrix is ``gram``.

    Parameters
    ----------
    gram : np.ndarray
        Symmetric ``(N, N)`` matrix ``X X^T``.

    Returns
    -------
    np.ndarray
        ``(N, N)`` float64 distances ``sqrt(G_ii + G_jj - 2 G_ij)``, clipped at zero before the
        square root (round-off) and with an exact zero diagonal.
    """
    g = np.asarray(gram, dtype=np.float64)
    sq = np.diag(g).copy()
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)
    np.sqrt(d2, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def mle_intrinsic_dim(dist: np.ndarray, k: int) -> float:
    """Levina–Bickel MLE of the intrinsic dimension, MacKay–Ghahramani averaging.

    Parameters
    ----------
    dist : np.ndarray
        ``(N, N)`` Euclidean distance matrix; the diagonal is ignored (the point is excluded).
    k : int
        Number of neighbours, ``2 <= k < N``.

    Returns
    -------
    float
        ``[mean_i (1/(k-1)) sum_{j<k} log(T_k / T_j)]^{-1}``.

    Raises
    ------
    IntrinsicDimError
        If ``k`` is out of range or two points coincide (a zero neighbour distance).
    """
    d = np.array(dist, dtype=np.float64, copy=True)
    n = d.shape[0]
    if d.ndim != 2 or d.shape[1] != n:
        raise IntrinsicDimError(f"distance matrix must be square, got {d.shape}")
    if not 2 <= k < n:
        raise IntrinsicDimError(f"k must satisfy 2 <= k < N, got k={k}, N={n}")
    np.fill_diagonal(d, np.inf)
    nn = np.partition(d, k - 1, axis=1)[:, :k]
    nn.sort(axis=1)
    if not np.all(nn[:, 0] > 0.0):
        raise IntrinsicDimError("two points coincide (zero nearest-neighbour distance)")
    inv_local = np.log(nn[:, -1:] / nn[:, :-1]).sum(axis=1) / (k - 1)
    return float(1.0 / inv_local.mean())


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Return ``(sum lambda)^2 / sum lambda^2`` of non-negative eigenvalues.

    Raises
    ------
    IntrinsicDimError
        If every eigenvalue is zero.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    denom = float(np.sum(lam**2))
    if denom <= 0.0:
        raise IntrinsicDimError("participation ratio of an all-zero spectrum")
    return float(np.sum(lam) ** 2 / denom)


def components_for_fraction(eigenvalues: np.ndarray, fraction: float) -> int:
    """Return the smallest number of leading components holding ``fraction`` of the variance.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Non-negative eigenvalues in any order.
    fraction : float
        Target fraction in (0, 1].

    Returns
    -------
    int
        Smallest ``m`` with ``sum_{i<=m} lambda_(i) >= fraction * sum lambda`` (descending order).
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    cum = np.cumsum(lam) / lam.sum()
    # Round-off can leave the last cumulative value a hair under 1.
    return int(min(np.searchsorted(cum, fraction - 1e-12) + 1, lam.size))


def linear_measures(gram: np.ndarray, fractions: Sequence[float]) -> LinearMeasures:
    """PCA participation ratio and components-for-fraction from the Gram matrix of centred data.

    The nonzero eigenvalues of ``X X^T`` equal those of ``X^T X``, so the ``N x N`` Gram matrix
    gives the PCA spectrum (up to the common factor ``1/N``, which both measures ignore).

    Parameters
    ----------
    gram : np.ndarray
        ``(N, N)`` Gram matrix of the centred rows.
    fractions : Sequence[float]
        Variance fractions.

    Returns
    -------
    LinearMeasures
        The participation ratio, ``{"0.9": m, ...}``, the total variance (trace), the numerical
        rank and the ten leading eigenvalues.
    """
    lam = np.linalg.eigvalsh(np.asarray(gram, dtype=np.float64))[::-1]
    lam = np.clip(lam, 0.0, None)
    tol = lam[0] * lam.size * np.finfo(np.float64).eps if lam.size else 0.0
    return LinearMeasures(
        participation_ratio=participation_ratio(lam),
        components={f"{f:g}": components_for_fraction(lam, f) for f in fractions},
        total_variance=float(lam.sum()),
        rank=int(np.sum(lam > tol)),
        eigenvalues_head=[float(v) for v in lam[:10]],
    )


# --------------------------------------------------------------------------------------------
# Data


def _require(path: Path) -> Path:
    if not path.exists():
        raise MissingInputError(f"{path} not found")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(_require(path).read_text())
    except json.JSONDecodeError as error:
        raise IntrinsicDimError(f"{path}: {error}") from error


def load_train_matrix(ds_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read one training split as the centred ``(N, H*W)`` float64 matrix of the method.

    Parameters
    ----------
    ds_dir : Path
        Dataset folder with ``images.npy``, ``splits.json`` and ``meta.json``.

    Returns
    -------
    x : np.ndarray
        ``(N, H*W)`` float64 rows: grey levels / 255, each row's mean removed, then each column
        centred.
    train : np.ndarray
        Image indices of the rows (``splits.json["train"]``, sorted ascending).
    meta : dict[str, Any]
        The dataset's ``meta.json``.

    Raises
    ------
    MissingInputError
        If the folder or a file is absent.
    IntrinsicDimError
        If the split indices do not fit the image array.
    """
    if not ds_dir.is_dir():
        raise MissingInputError(f"dataset folder {ds_dir} not found")
    meta = _read_json(ds_dir / "meta.json")
    splits = _read_json(ds_dir / "splits.json")
    images = np.load(_require(ds_dir / "images.npy"), mmap_mode="r")
    if "train" not in splits:
        raise IntrinsicDimError(f"{ds_dir / 'splits.json'}: no 'train' split")
    train = np.sort(np.asarray(splits["train"], dtype=np.int64))
    if images.ndim != 3:
        raise IntrinsicDimError(f"{ds_dir / 'images.npy'}: expected (n, H, W), got {images.shape}")
    if train.size == 0 or train.min() < 0 or train.max() >= images.shape[0]:
        raise IntrinsicDimError(f"{ds_dir}: train indices outside images.npy "
                                f"({images.shape[0]} images)")
    if np.unique(train).size != train.size:
        raise IntrinsicDimError(f"{ds_dir}: repeated train indices")
    x = np.asarray(images[train], dtype=np.float64).reshape(train.size, -1)
    x /= 255.0
    x -= x.mean(axis=1, keepdims=True)
    x -= x.mean(axis=0, keepdims=True)
    return x, train, meta


def _slice_rows(ds_dir: Path, train: np.ndarray, slice_index: int) -> np.ndarray:
    """Row positions (into ``train``) of slice ``slice_index`` of every training subject."""
    path = _require(ds_dir / "index.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        slice_of = {int(r["idx"]): int(r["slice"]) for r in rows}
        subject_of = {int(r["idx"]): r["subject"] for r in rows}
    except (KeyError, ValueError) as error:
        raise IntrinsicDimError(f"{path}: needs integer idx and slice columns ({error})") from error
    positions = [p for p, i in enumerate(train.tolist()) if slice_of.get(i) == slice_index]
    subjects = [subject_of[int(train[p])] for p in positions]
    if len(set(subjects)) != len(subjects):
        raise IntrinsicDimError(f"{ds_dir}: a subject has two slices {slice_index}")
    splits = _read_json(ds_dir / "splits.json")
    expected = splits.get("train_subjects")
    if expected is not None and sorted(set(subjects)) != sorted(expected):
        raise IntrinsicDimError(f"{ds_dir}: slice {slice_index} does not cover the "
                                f"{len(expected)} training subjects (found {len(subjects)})")
    return np.asarray(positions, dtype=np.int64)


def _summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "sd": sd, "n_subsets": int(arr.size),
            "values": [float(v) for v in arr]}


def _draw_subsets(n_total: int, cfg: IDConfig) -> dict[int, list[np.ndarray]]:
    """The subsets of every N below the full split, from one fresh generator per dataset."""
    rng = np.random.default_rng(cfg.seed)
    out: dict[int, list[np.ndarray]] = {}
    for n in sorted(cfg.n_values):
        if n < n_total:
            out[n] = [np.sort(rng.choice(n_total, size=n, replace=False))
                      for _ in range(cfg.n_subsets)]
    return out


def analyse_dataset(ds_dir: Path, cfg: IDConfig) -> dict[str, Any]:
    """Every number of the method for one dataset.

    Parameters
    ----------
    ds_dir : Path
        Dataset folder.
    cfg : IDConfig
        The method.

    Returns
    -------
    dict[str, Any]
        ``sha256_images``, sizes, ``mle[k=..][N=..]`` (mean, sd, per-subset values), the MRI
        one-slice variant, the linear measures and the k = 10 change from the smallest to the
        largest N.

    Raises
    ------
    MissingInputError
        If an input is absent.
    IntrinsicDimError
        If the split size differs from the largest N or two images coincide.
    """
    name = ds_dir.name
    x, train, meta = load_train_matrix(ds_dir)
    n_total = x.shape[0]
    if n_total != max(cfg.n_values):
        raise IntrinsicDimError(f"{name}: {n_total} training images, the method expects "
                                f"{max(cfg.n_values)}")
    logger.info("%s: Gram matrix of %d x %d (float64)", name, n_total, x.shape[1])
    gram = x @ x.T
    n_pixels = int(x.shape[1])
    del x
    dist = distances_from_gram(gram)
    logger.info("%s: eigenvalues", name)
    linear = linear_measures(gram, cfg.fractions)
    del gram

    subsets = _draw_subsets(n_total, cfg)
    mle: dict[str, dict[str, Any]] = {}
    for k in cfg.k_values:
        per_n: dict[str, Any] = {}
        for n in sorted(cfg.n_values):
            if n == n_total:
                per_n[f"N={n}"] = _summary([mle_intrinsic_dim(dist, k)])
            else:
                per_n[f"N={n}"] = _summary([mle_intrinsic_dim(dist[np.ix_(s, s)], k)
                                            for s in subsets[n]])
        mle[f"k={k}"] = per_n
        logger.info("%s: k=%d %s", name, k,
                    ", ".join(f"{key} {v['mean']:.2f}" for key, v in per_n.items()))

    result: dict[str, Any] = {
        "label": LABELS.get(name, name),
        "modality": "mri" if name in MRI_DATASETS else "photograph",
        "sha256_images": meta.get("sha256_images"),
        "n_train": int(n_total),
        "n_pixels": n_pixels,
        "mle": mle,
        "linear": {
            "participation_ratio": linear.participation_ratio,
            "components_for_fraction": linear.components,
            "total_variance": linear.total_variance,
            "rank": linear.rank,
            "eigenvalues_head": linear.eigenvalues_head,
        },
    }
    if K_MAIN in cfg.k_values:
        lo, hi = f"N={min(cfg.n_values)}", f"N={max(cfg.n_values)}"
        m_lo, m_hi = mle[f"k={K_MAIN}"][lo]["mean"], mle[f"k={K_MAIN}"][hi]["mean"]
        result[f"relative_change_k{K_MAIN}"] = {"from": lo, "to": hi,
                                                "value": (m_hi - m_lo) / m_lo}
    if name in MRI_DATASETS:
        rows = _slice_rows(ds_dir, train, cfg.slice_index)
        sub = dist[np.ix_(rows, rows)]
        result["one_slice_per_subject"] = {
            "slice": cfg.slice_index,
            "N": int(rows.size),
            "mle": {f"k={k}": mle_intrinsic_dim(sub, k) for k in cfg.k_values},
        }
    return result


# --------------------------------------------------------------------------------------------
# Reading, provenance, run


def reading(datasets: dict[str, dict[str, Any]], cfg: IDConfig) -> dict[str, Any]:
    """Check the three statements of the ticket's reading against the numbers.

    Returns
    -------
    dict[str, Any]
        For each statement, a boolean and the numbers it compares (k = 10).
    """
    mri = [d for d in cfg.datasets if d in MRI_DATASETS and d in datasets]
    photo = [d for d in cfg.datasets if d not in MRI_DATASETS and d in datasets]
    full = f"N={max(cfg.n_values)}"
    key = f"k={K_MAIN}"
    mle_full = {d: datasets[d]["mle"][key][full]["mean"] for d in mri + photo}
    change = {d: datasets[d][f"relative_change_k{K_MAIN}"]["value"] for d in mri + photo}
    pr = {d: datasets[d]["linear"]["participation_ratio"] for d in mri + photo}
    return {
        "k": K_MAIN,
        "mle_lower_for_mri_at_full_n": {
            "holds": max(mle_full[d] for d in mri) < min(mle_full[d] for d in photo),
            "values": mle_full,
        },
        "mri_changes_less_with_n": {
            "holds": max(abs(change[d]) for d in mri) < min(abs(change[d]) for d in photo),
            "relative_change": change,
        },
        "participation_ratio_higher_for_mri": {
            "holds": min(pr[d] for d in mri) > max(pr[d] for d in photo),
            "values": pr,
        },
    }


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run(data_root: Path, out_dir: Path, cfg: IDConfig | None = None) -> dict[str, Any]:
    """Analyse every dataset, one at a time, and write the JSON and the figure.

    Parameters
    ----------
    data_root : Path
        Folder holding one sub-folder per dataset.
    out_dir : Path
        Output folder (created if absent).
    cfg : IDConfig | None
        The method; ``None`` is the frozen T8.0 method.

    Returns
    -------
    dict[str, Any]
        The JSON document, plus ``"_files"``: ``{file name: sha256}`` of what was written.

    Raises
    ------
    MissingInputError
        If the data root or a dataset folder is absent (checked before any computation).
    IntrinsicDimError
        On any analysis or figure failure.
    """
    cfg = cfg or IDConfig()
    if not data_root.is_dir():
        raise MissingInputError(f"data root {data_root} not found")
    for name in cfg.datasets:
        for rel in ("", "images.npy", "splits.json", "meta.json"):
            _require(data_root / name / rel)
        if name in MRI_DATASETS:
            _require(data_root / name / "index.csv")
    datasets = {name: analyse_dataset(data_root / name, cfg) for name in cfg.datasets}
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "provenance": {
            "tool": TOOL,
            "git_sha": _git_sha(),
            "data_root": str(data_root),
            "rng": f"numpy.random.default_rng({cfg.seed}), re-created per dataset; for each N "
                   f"below the full split in ascending order, {cfg.n_subsets} draws of "
                   "rng.choice(n_train, N, replace=False) over positions in the sorted train list",
            "numpy": np.__version__,
        },
        "method": {
            "data": "splits.json['train'], grey levels / 255, each image's mean (DC) removed, "
                    "then centred across images; float64",
            "estimator": "Levina & Bickel (NeurIPS 2004) MLE, MacKay & Ghahramani (2005) "
                         "average of the inverse local estimates",
            "distances": "Euclidean, from the float64 Gram matrix of the full split",
            "k_values": list(cfg.k_values),
            "n_values": list(cfg.n_values),
            "n_subsets": cfg.n_subsets,
            "sd": "sample standard deviation (ddof = 1) over the subsets",
            "seed": cfg.seed,
            "one_slice_per_subject": f"MRI only: slice {cfg.slice_index} of every training "
                                     "subject",
            "linear": "eigenvalues of the Gram matrix; participation ratio "
                      "(sum lambda)^2 / sum lambda^2; components for each variance fraction",
            "fractions": list(cfg.fractions),
        },
        "datasets": datasets,
        "reading": reading(datasets, cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "intrinsic_dimension.json"
    json_path.write_text(json.dumps(doc, indent=2) + "\n")
    figure_paths = plot_id_vs_n(doc, out_dir, cfg)
    doc["_files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [json_path, *figure_paths]}
    return doc


# --------------------------------------------------------------------------------------------
# Figure

#: Grey levels and line styles, MRI solid against photographs dashed (no arm colours).
_LINE_STYLE: dict[str, tuple[str, str, str]] = {
    "ixi": ("#0b0b0b", "-", "o"),
    "oasis1": ("#7a7974", "-", "s"),
    "lsun_church": ("#0b0b0b", "--", "^"),
    "lsun_bedroom": ("#7a7974", "--", "D"),
}


def _figure_rc() -> dict[str, object]:
    return {
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "svg.fonttype": "none",
        "svg.hashsalt": SVG_HASHSALT,
    }


def _audit(fig: Any) -> tuple[float, float, float]:
    from matplotlib.text import Text

    width, height = (float(v) for v in fig.get_size_inches())
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    min_font = min(float(t.get_fontsize()) for t in texts)
    if width > FIGURE_WIDTH_IN + 1e-9 or height > FIGURE_HEIGHT_IN + 1e-9:
        raise IntrinsicDimError(f"figure is {width} x {height} in, limit "
                                f"{FIGURE_WIDTH_IN} x {FIGURE_HEIGHT_IN}")
    if min_font < MIN_FONT_PT:
        raise IntrinsicDimError(f"smallest font {min_font} pt < {MIN_FONT_PT}")
    return width, height, min_font


def plot_id_vs_n(doc: dict[str, Any], out_dir: Path, cfg: IDConfig) -> list[Path]:
    """Draw the k = 10 MLE estimate against N per dataset; write SVG, PDF and PNG byte-stably.

    Parameters
    ----------
    doc : dict[str, Any]
        The JSON document of :func:`run`.
    out_dir : Path
        Output folder.
    cfg : IDConfig
        The method (dataset order, N values).

    Returns
    -------
    list[Path]
        The SVG, PDF and PNG paths.

    Raises
    ------
    IntrinsicDimError
        If the layout breaks the size or font limits.
    """
    from matplotlib import pyplot as plt

    from ihdm.analysis.style import figure_style

    key = f"k={K_MAIN}"
    n_values = sorted(cfg.n_values)
    with figure_style(), plt.rc_context(_figure_rc()):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
        for name in cfg.datasets:
            entry = doc["datasets"][name]
            color, ls, marker = _LINE_STYLE.get(name, ("#0b0b0b", ":", "x"))
            means = [entry["mle"][key][f"N={n}"]["mean"] for n in n_values]
            sds = [entry["mle"][key][f"N={n}"]["sd"] for n in n_values]
            ax.errorbar(n_values, means, yerr=sds, color=color, linestyle=ls, marker=marker,
                        markersize=3.5, linewidth=1.2, capsize=2.0, elinewidth=0.8,
                        label=entry["label"])
            one = entry.get("one_slice_per_subject")
            if one is not None:
                # Drawn left of its N so it does not hide the random-subset mean at the same N.
                ax.plot([one["N"] * ONE_SLICE_X_SHIFT], [one["mle"][key]], linestyle="none",
                        marker=marker, markersize=5.0, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=0.9, zorder=4)
        handles, labels = ax.get_legend_handles_labels()
        if any(name in MRI_DATASETS for name in cfg.datasets):
            (proxy,) = ax.plot([], [], linestyle="none", marker="o", markersize=5.0,
                               markerfacecolor="white", markeredgecolor="#52514e")
            handles.append(proxy)
            labels.append(f"MRI, one slice per subject (N = {min(n_values)})")
        ax.set_xscale("log")
        ax.set_xticks(n_values)
        ax.set_xticklabels([f"{n:,}" for n in n_values])
        ax.minorticks_off()
        ax.set_xlim(n_values[0] / 1.45, n_values[-1] * 1.25)
        ax.set_xlabel("training images N (log scale)")
        ax.set_ylabel(f"MLE intrinsic dim. (k = {K_MAIN})")
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, handlelength=2.6)
        fig.subplots_adjust(left=0.09, right=0.66, bottom=0.21, top=0.96)
        _audit(fig)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [out_dir / f"{FIGURE_NAME}.{ext}" for ext in ("svg", "pdf", "png")]
        fig.savefig(paths[0], format="svg", metadata={"Date": None, "Creator": CREATOR})
        fig.savefig(paths[1], format="pdf", dpi=PDF_DPI,
                    metadata={"CreationDate": None, "ModDate": None, "Creator": CREATOR,
                              "Producer": "matplotlib"})
        fig.savefig(paths[2], format="png", dpi=PNG_DPI, metadata={"Software": None})
        plt.close(fig)
    logger.info("wrote %s", ", ".join(str(p) for p in paths))
    return paths
