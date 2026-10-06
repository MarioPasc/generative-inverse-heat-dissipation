"""Appendix A1 of the course paper (T8.3): the natural-image reference runs.

Figure A1 has two panels: (a) the training loss of the default configuration (A0) on LSUN
Churches and Bedrooms against iteration, with an epoch axis on top, the checkpoints of (b) marked
and the 40k -> 60k extension shaded; (b) one training seed per dataset and its sample at five
checkpoints, drawn with common random numbers (asserted). Table A1 sets the paper's LSUN setup
beside ours; its cells are fixed facts, and the "ours" column is cross-checked against every run's
configuration. The text of ``A1.md`` reads all its numbers from files at build time. A failed
assertion raises :class:`PaperA1Error` (exit code 1); a missing file raises
:class:`MissingInputError` (exit code 2).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ihdm.analysis.tables import AnalysisError
from ihdm.train.validate_run import canonical_records

__all__ = [
    "ARM_LABELS",
    "CHECKPOINTS",
    "ITERS_PER_EPOCH",
    "PAPER_WIDTH_IN",
    "A1Data",
    "A1Paths",
    "A1Report",
    "CRNError",
    "DatasetLoss",
    "LsdSet",
    "MissingInputError",
    "PaperA1Error",
    "ThumbRow",
    "assert_crn",
    "audit_layout",
    "build",
    "draw_a1",
    "epochs_to_iterations",
    "iterations_to_epochs",
    "load_lsd_set",
    "plateau_step",
    "render_table_md",
    "render_table_tex",
    "running_mean",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# Paper constants (defined locally, as in paper_f2.py)
# --------------------------------------------------------------------------------------------

PAPER_WIDTH_IN: float = 5.5
ARM_LABELS: dict[str, str] = {"A0": "default", "A1": "+prior", "A2": "+spacing", "A3": "matched"}
#: The Churches-only spacing control (A2'); not one of the four paper configurations.
CONTROL_LABELS: dict[str, str] = {"A2p": "the A2′ spacing control"}

FIG_HEIGHT_IN: float = 1.70
MAX_HEIGHT_IN: float = 2.0
MIN_FONT_PT: float = 7.0
MIN_THUMB_IN: float = 0.45
PNG_DPI: int = 300
PDF_DPI: int = 300
SVG_HASHSALT: str = "ihdm-t8.3"
CREATOR: str = "ihdm.cli.paper_a1"
FIGURE_NAME: str = "a1_natural_images"

# --------------------------------------------------------------------------------------------
# What the appendix reads
# --------------------------------------------------------------------------------------------

ARM: str = "A0"
#: Panel (a): the default-configuration runs of each photograph dataset (ticket).
LOSS_SEEDS: dict[str, tuple[int, ...]] = {"lsun_church": (1, 2, 3), "lsun_bedroom": (1, 2)}
DATASETS: tuple[str, ...] = ("lsun_church", "lsun_bedroom")
DATASET_NAMES: dict[str, str] = {"lsun_church": "Churches", "lsun_bedroom": "Bedrooms"}
PHOTO_DATASETS: tuple[str, ...] = ("lsun_church", "lsun_bedroom")
MRI_DATASETS: tuple[str, ...] = ("ixi", "oasis1")
#: Panel (b): run seed 1 of A0, its five checkpoints (ticket).
THUMB_RUN_SEED: int = 1
CHECKPOINTS: tuple[int, ...] = (5_000, 15_000, 30_000, 45_000, 60_000)
#: Position of the shown seed in the frozen ``eval_seeds_500.npy`` (its first index).
SEED_POSITION: int = 0
AMP: str = "fp16"
SET_NAME: str = "lsd"
SET_FILES: tuple[str, ...] = ("samples.npy", "seed_idx.npy", "seeds.npy", "request.json")

#: 3,200 training images / batch 16 (frozen contract); asserted against every run's files.
ITERS_PER_EPOCH: int = 200
#: Running mean of the training loss, in iterations (ticket).
LOSS_WINDOW_ITERS: int = 1_000
#: First step of the 40k -> 60k extension (D22): every run resumed at 40,001.
EXTENSION_START: int = 40_001
#: Records before this step are off the y range of panel (a) (warm-up and start-up).
SETTLED_STEP: int = 2_500
#: Plateau rule: the smoothed seed-mean loss stays within this fraction of its final value ...
PLATEAU_TOLERANCE: float = 0.10
#: ... where the final value is the mean over the last this-many iterations ...
FINAL_WINDOW_ITERS: int = 5_000
#: ... and the reported step is rounded up to a multiple of this.
PLATEAU_ROUND: int = 5_000

#: Cited, not measured here: the share of each level's regression target that is the added
#: training noise, N sigma^2 against the deblurring term (projects/GenAI/learning/
#: 03-terminal-blur-scaffolding.md, §6, eq. 6.5: "the per-level task is 98% denoising").
NOISE_SHARE_CITED: float = 0.98
#: The intrinsic-dimension sentence of the text; main replaces it after T8.0 merges.
ID_PLACEHOLDER: str = ("[PLACEHOLDER (Sec. X; T8.0): registered brain slices have a lower "
                       "intrinsic dimension than photographs, which is consistent with MRI being "
                       "learnable at this budget.]")

#: Panel (a) styles per dataset: two greys and two line styles (ticket).
DATASET_STYLE: dict[str, tuple[str, Any]] = {
    "lsun_church": ("#262626", "-"),
    "lsun_bedroom": ("#7d7d7d", (0, (3.0, 1.6))),
}

# --------------------------------------------------------------------------------------------
# Table A1: fixed facts
# --------------------------------------------------------------------------------------------

#: (row label, the paper's LSUN Churches 128² model, ours). The paper's column: Rissanen et al.,
#: ICLR 2023, App. B (arXiv:2206.13397v3): B.1 architecture (channel_mult (1,2,3,4,5),
#: 2 res-blocks), B.2 optimisation (lr 2e-5, batch 32, 1M iterations), B.4 K = 400; ≈ 126k
#: LSUN Churches training images; 160 M parameters measured on 2026-10-03 by instantiating
#: model_code.unet.UNetModel (results_discussion.md §3). Ours: the run configurations,
#: cross-checked by :func:`assert_our_setup`.
TABLE_A1_ROWS: tuple[tuple[str, str, str], ...] = (
    ("training images", "≈ 126k, RGB", "3,200, grayscale"),
    ("framing", "whole scene, 128²", "native-resolution 192² centre crop"),
    ("U-Net", "(1,2,3,4,5), 2 res-blocks: 160 M parameters", "(1,2,2,2), 4 res-blocks: 61 M"),
    ("levels K", "400", "200"),
    ("optimiser", "lr 2e-5, batch 32", "lr 1e-4, batch 16"),
    ("iterations (images seen)", "1M (32M)", "60k (0.96M, 3%)"),
)
TABLE_A1_NOTE: str = "The paper did not train on LSUN Bedrooms."
TABLE_A1_HEAD: tuple[str, str] = (
    "Rissanen et al. (LSUN Churches 128², App. B)",
    "ours (Churches, Bedrooms)",
)
#: The values the "ours" column states, checked against config.json, summary.json and
#: manifest.json of every run in :data:`LOSS_SEEDS`.
OUR_SETUP: dict[str, Any] = {
    "n_train": 3_200,
    "num_channels": 1,
    "image_size": 192,
    "channel_mult": [1, 2, 2, 2],
    "num_res_blocks": 4,
    "n_params_millions": 61,
    "K": 200,
    "lr": 1e-4,
    "batch_size": 16,
    "n_iters": 60_000,
}
#: Images the paper's model saw (1M iterations × batch 32), for the "3%" of Table A1.
PAPER_IMAGES_SEEN: int = 32_000_000


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------


class PaperA1Error(AnalysisError):
    """Appendix A1 cannot be built as the ticket requires (exit code 1)."""


class MissingInputError(PaperA1Error):
    """A file the build needs does not exist (exit code 2)."""


class CRNError(PaperA1Error):
    """The checkpoints of panel (b) do not share their seeds, rng seed or batch size."""


# --------------------------------------------------------------------------------------------
# Epochs and the running mean
# --------------------------------------------------------------------------------------------


def iterations_to_epochs(iterations: Any, iters_per_epoch: int = ITERS_PER_EPOCH) -> Any:
    """Convert iterations to epochs (an epoch is ``iters_per_epoch`` iterations).

    Parameters
    ----------
    iterations : float or np.ndarray
        Iteration counts.
    iters_per_epoch : int
        Iterations per pass over the training set (training images / batch size).

    Returns
    -------
    float or np.ndarray
        Epochs, of the same shape.

    Raises
    ------
    PaperA1Error
        If ``iters_per_epoch`` is not positive.
    """
    if iters_per_epoch <= 0:
        raise PaperA1Error(f"iterations per epoch must be positive, got {iters_per_epoch}")
    return np.asarray(iterations, dtype=float) / iters_per_epoch


def epochs_to_iterations(epochs: Any, iters_per_epoch: int = ITERS_PER_EPOCH) -> Any:
    """Inverse of :func:`iterations_to_epochs`."""
    if iters_per_epoch <= 0:
        raise PaperA1Error(f"iterations per epoch must be positive, got {iters_per_epoch}")
    return np.asarray(epochs, dtype=float) * iters_per_epoch


def running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing running mean; the first ``window - 1`` points average what is available.

    Parameters
    ----------
    values : np.ndarray
        1-D series.
    window : int
        Number of points averaged (>= 1).

    Returns
    -------
    np.ndarray
        Same length as ``values``; point ``i`` is the mean of ``values[max(0, i-window+1):i+1]``.

    Raises
    ------
    PaperA1Error
        If ``window`` is below 1.
    """
    if window < 1:
        raise PaperA1Error(f"running-mean window must be >= 1, got {window}")
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, values.size + 1)
    start = np.maximum(idx - window, 0)
    return (cumsum[idx] - cumsum[start]) / (idx - start)


def plateau_step(steps: np.ndarray, curve: np.ndarray, tolerance: float = PLATEAU_TOLERANCE,
                 final_window: int = FINAL_WINDOW_ITERS) -> tuple[int, float]:
    """Return the first logged step after which ``curve`` stays within ``tolerance`` of its end.

    Parameters
    ----------
    steps : np.ndarray
        Iterations of the curve, increasing.
    curve : np.ndarray
        The smoothed loss.
    tolerance : float
        Allowed relative deviation from the final value.
    final_window : int
        The final value is the mean of the curve over ``steps > steps[-1] - final_window``.

    Returns
    -------
    tuple[int, float]
        The step and the final value.

    Raises
    ------
    PaperA1Error
        If the curve is empty or its final value is not positive.
    """
    steps = np.asarray(steps, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if steps.size == 0 or steps.size != curve.size:
        raise PaperA1Error("plateau of an empty or ragged curve")
    final = float(curve[steps > steps[-1] - final_window].mean())
    if not final > 0:
        raise PaperA1Error(f"final loss {final} is not positive")
    outside = np.flatnonzero(np.abs(curve / final - 1.0) > tolerance)
    if outside.size == 0:
        return int(steps[0]), final
    if outside[-1] + 1 >= steps.size:
        return int(steps[-1]), final
    return int(steps[outside[-1] + 1]), final


# --------------------------------------------------------------------------------------------
# Panel (a): canonical histories
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunHistory:
    """The training loss of one run: steps, raw loss and its running mean."""

    run_id: str
    steps: np.ndarray
    loss: np.ndarray
    smooth: np.ndarray
    window_records: int


@dataclass(frozen=True)
class DatasetLoss:
    """Panel (a) data of one dataset: its runs and their seed mean (of the running means)."""

    dataset: str
    runs: tuple[RunHistory, ...]
    steps: np.ndarray
    mean: np.ndarray


def run_id(dataset: str, arm: str, seed: int) -> str:
    """Return the run id, e.g. ``lsun_church_A0_s1``."""
    return f"{dataset}_{arm}_s{seed}"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise MissingInputError(f"{path} not found")
    return json.loads(path.read_text())


def read_history(results: Path, rid: str) -> RunHistory:
    """Read the training loss of one run from ``runs/<rid>/metrics.canonical.jsonl``.

    The file is read as fig. 7 reads it (one JSON record per line, ``kind == "train"``);
    :func:`ihdm.train.validate_run.canonical_records` is applied again, which leaves a canonical
    history unchanged and drops any abandoned segment otherwise.

    Parameters
    ----------
    results : Path
        The ``_results`` folder.
    rid : str
        Run id.

    Returns
    -------
    RunHistory
        Steps, loss and the running mean over :data:`LOSS_WINDOW_ITERS` iterations.

    Raises
    ------
    MissingInputError
        If the history is absent.
    PaperA1Error
        If it has no training records, or its steps are not increasing.
    """
    path = Path(results) / "runs" / rid / "metrics.canonical.jsonl"
    if not path.is_file():
        raise MissingInputError(f"{path} not found")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = canonical_records(records)
    points = [(int(r["step"]), float(r["loss"])) for r in records
              if r.get("kind") == "train" and r.get("loss") is not None]
    if len(points) < 2:
        raise PaperA1Error(f"{path}: fewer than two training records")
    steps = np.array([p[0] for p in points], dtype=float)
    loss = np.array([p[1] for p in points], dtype=float)
    spacing = np.diff(steps)
    if np.any(spacing <= 0):
        raise PaperA1Error(f"{path}: training steps are not increasing")
    interval = float(np.median(spacing))
    window = max(1, int(round(LOSS_WINDOW_ITERS / interval)))
    return RunHistory(rid, steps, loss, running_mean(loss, window), window)


def load_losses(results: Path) -> tuple[DatasetLoss, ...]:
    """Read panel (a): the A0 runs of :data:`LOSS_SEEDS` and their seed means.

    Raises
    ------
    PaperA1Error
        If the runs of one dataset were not logged at the same steps.
    """
    out = []
    for dataset in DATASETS:
        runs = tuple(read_history(results, run_id(dataset, ARM, s)) for s in LOSS_SEEDS[dataset])
        steps = runs[0].steps
        for run in runs[1:]:
            if not np.array_equal(run.steps, steps):
                raise PaperA1Error(f"{run.run_id}: logged steps differ from {runs[0].run_id}")
        mean = np.mean(np.stack([r.smooth for r in runs]), axis=0)
        out.append(DatasetLoss(dataset, runs, steps, mean))
        logger.info("%s: %d runs, %d records, window %d records", dataset, len(runs), steps.size,
                    runs[0].window_records)
    return tuple(out)


# --------------------------------------------------------------------------------------------
# Our setup, cross-checked against the run files
# --------------------------------------------------------------------------------------------


def read_run_setup(results: Path, rid: str) -> dict[str, Any]:
    """Return the facts of Table A1's "ours" column as one run's files record them.

    Raises
    ------
    MissingInputError
        If ``config.json``, ``summary.json`` or ``manifest.json`` is absent.
    PaperA1Error
        If a key is missing.
    """
    folder = Path(results) / "runs" / rid
    config = _read_json(folder / "config.json")
    summary = _read_json(folder / "summary.json")
    manifest = _read_json(folder / "manifest.json")
    try:
        return {
            "n_train": int(summary["env"]["n_train"]),
            "num_channels": int(config["data"]["num_channels"]),
            "image_size": int(config["data"]["image_size"]),
            "channel_mult": list(config["model"]["channel_mult"]),
            "num_res_blocks": int(config["model"]["num_res_blocks"]),
            "n_params_millions": int(round(float(manifest["n_params"]) / 1e6)),
            "K": int(config["model"]["K"]),
            "lr": float(config["optim"]["lr"]),
            "batch_size": int(config["training"]["batch_size"]),
            "n_iters": int(config["training"]["n_iters"]),
            "eval_seeds_500_sha256": str(summary["seed_lists"]["intermediate_sha256"]),
            "eval_seeds_500_path": str(summary["seed_lists"]["intermediate_path"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperA1Error(f"{folder}: setup key missing or malformed ({exc})") from exc


def assert_our_setup(setups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Assert that every run matches :data:`OUR_SETUP` and :data:`ITERS_PER_EPOCH`.

    Parameters
    ----------
    setups : Mapping[str, Mapping[str, Any]]
        ``{run_id: read_run_setup(...)}``.

    Returns
    -------
    list[str]
        One line per checked fact.

    Raises
    ------
    PaperA1Error
        If any run differs.
    """
    problems, lines = [], []
    for key, expected in OUR_SETUP.items():
        values = {rid: s[key] for rid, s in setups.items()}
        bad = {rid: v for rid, v in values.items() if v != expected}
        if bad:
            problems.append(f"{key}: expected {expected!r}, got {bad}")
        lines.append(f"{key} = {expected!r} in all {len(values)} runs")
    for rid, s in setups.items():
        per_epoch = s["n_train"] / s["batch_size"]
        if per_epoch != ITERS_PER_EPOCH:
            problems.append(f"{rid}: {s['n_train']} / {s['batch_size']} = {per_epoch} iterations "
                            f"per epoch, not {ITERS_PER_EPOCH}")
    lines.append(f"iterations per epoch = n_train / batch = {ITERS_PER_EPOCH} in all "
                 f"{len(setups)} runs")
    if problems:
        raise PaperA1Error("Table A1 'ours' column contradicts the run files: "
                           + "; ".join(problems))
    return lines


# --------------------------------------------------------------------------------------------
# Panel (b): the LSD sample sets of five checkpoints, CRN
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LsdSet:
    """The 500-seed LSD sample set of one checkpoint.

    Attributes
    ----------
    step : int
        The checkpoint.
    seed_idx : np.ndarray
        Dataset index of each seed, ``(n_seeds,)`` int64.
    seeds : np.ndarray
        Seed images, ``(n_seeds, H, W)`` uint8 (memory-mapped).
    samples : np.ndarray
        Samples, seed-major, ``(n_seeds * n_per_seed, H, W)`` uint8 (memory-mapped).
    request : dict[str, Any]
        ``request.json``.
    """

    step: int
    seed_idx: np.ndarray
    seeds: np.ndarray
    samples: np.ndarray
    request: dict[str, Any]

    @property
    def signature(self) -> dict[str, Any]:
        """The sampling signature (rng seed, batch size, ...)."""
        return dict(self.request.get("signature", {}))


def index_sha256(idx: np.ndarray) -> str:
    """Return the sha256 of an index array written as little-endian int64."""
    return hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tar_path(eval_dir: Path, rid: str) -> Path:
    """Return the evaluation tar of a run."""
    return Path(eval_dir) / f"{rid}_amp-{AMP}.tar"


def member_name(rid: str, step: int, name: str) -> str:
    """Return the tar member of one file of one checkpoint's LSD set."""
    return f"{rid}/samples_amp-{AMP}/{step:06d}/{SET_NAME}/{name}"


def extract_checkpoints(eval_dir: Path, rid: str, work: Path,
                        steps: Sequence[int] = CHECKPOINTS) -> Path:
    """Extract only the ``lsd/`` members of ``steps`` from a run's tar into ``work``, once.

    Parameters
    ----------
    eval_dir : Path
        Folder of the evaluation tars.
    rid : str
        Run id.
    work : Path
        Scratch folder; files land in ``work/extract/<rid>/<step>/lsd/``.
    steps : Sequence[int]
        Checkpoints.

    Returns
    -------
    Path
        ``work/extract/<rid>``.

    Raises
    ------
    MissingInputError
        If the tar or a member is absent.
    """
    dest = Path(work) / "extract" / rid
    wanted = {member_name(rid, s, f): dest / f"{s:06d}" / SET_NAME / f
              for s in steps for f in SET_FILES}
    if all(p.is_file() for p in wanted.values()):
        logger.info("%s: LSD sets read from %s", rid, dest)
        return dest
    tar_file = tar_path(eval_dir, rid)
    if not tar_file.is_file():
        raise MissingInputError(f"{tar_file} not found")
    found: set[str] = set()
    with tarfile.open(tar_file) as tar:
        for member in tar:
            target = wanted.get(member.name)
            if target is None or not member.isfile():
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with source, open(tmp, "wb") as out:
                shutil.copyfileobj(source, out, 1 << 22)
            tmp.replace(target)
            found.add(member.name)
            if len(found) == len(wanted):
                break
    missing = sorted(set(wanted) - found)
    if missing:
        raise MissingInputError(f"{tar_file}: members missing: {', '.join(missing)}")
    logger.info("%s: extracted %d members into %s", rid, len(wanted), dest)
    return dest


def load_lsd_set(folder: Path, step: int) -> LsdSet:
    """Read one extracted LSD set (``<run>/<step>/lsd``).

    Raises
    ------
    MissingInputError
        If a member is absent.
    PaperA1Error
        If the shapes or dtypes are inconsistent.
    """
    folder = Path(folder)
    for name in SET_FILES:
        if not (folder / name).is_file():
            raise MissingInputError(f"{folder / name} not found")
    request = json.loads((folder / "request.json").read_text())
    seed_idx = np.load(folder / "seed_idx.npy").astype(np.int64).ravel()
    seeds = np.load(folder / "seeds.npy", mmap_mode="r")
    seeds = seeds.reshape(-1, *seeds.shape[-2:])
    samples = np.load(folder / "samples.npy", mmap_mode="r")
    samples = samples.reshape(-1, *samples.shape[-2:])
    if seeds.dtype != np.uint8 or samples.dtype != np.uint8:
        raise PaperA1Error(f"{folder}: expected uint8 seeds and samples")
    if seeds.shape[0] != seed_idx.size or samples.shape[0] % max(seed_idx.size, 1):
        raise PaperA1Error(f"{folder}: {seed_idx.size} seeds, seeds {seeds.shape}, "
                           f"samples {samples.shape}")
    return LsdSet(int(step), seed_idx, seeds, samples, request)


def assert_crn(sets: Mapping[int, LsdSet], seeds_sha256: str | None = None) -> dict[str, Any]:
    """Assert common random numbers across the checkpoints of one run.

    Every checkpoint must hold the LSD set (one sample per seed) of its own step, with identical
    ``seed_idx`` and seed images, the same ``rng_seed`` and ``batch_size`` in ``request.json``,
    a ``seed_idx_sha256`` in the signature equal to the stored indices, and the same sample
    shape. When ``seeds_sha256`` is given (``summary.json``, the frozen ``eval_seeds_500.npy``),
    the indices must also equal that list, so position 0 is its first index.

    Parameters
    ----------
    sets : Mapping[int, LsdSet]
        ``{step: set}`` of one run.
    seeds_sha256 : str | None
        Expected sha256 of the seed indices.

    Returns
    -------
    dict[str, Any]
        The steps, rng seed, batch size, number of seeds and the sha256 of the shared indices.

    Raises
    ------
    CRNError
        On any violation.
    """
    steps = sorted(sets)
    if len(steps) < 2:
        raise CRNError("common random numbers need at least two checkpoints")
    first = sets[steps[0]]
    sig0 = first.signature
    problems: list[str] = []
    for step in steps:
        current = sets[step]
        sig = current.signature
        where = f"step {step}"
        if current.step != step or int(current.request.get("step", -1)) != step:
            problems.append(f"{where}: request.json step {current.request.get('step')}")
        if current.request.get("set", sig.get("set")) != SET_NAME or sig.get("set") != SET_NAME:
            problems.append(f"{where}: set {sig.get('set')!r} is not {SET_NAME!r}")
        if sig.get("n_per_seed") != 1:
            problems.append(f"{where}: n_per_seed {sig.get('n_per_seed')} != 1")
        for key in ("rng_seed", "batch_size"):
            if sig.get(key) is None or sig.get(key) != sig0.get(key):
                problems.append(f"{where}: {key} {sig.get(key)} != {sig0.get(key)}")
        if not np.array_equal(current.seed_idx, first.seed_idx):
            problems.append(f"{where}: seed_idx differs from step {steps[0]}")
        elif not np.array_equal(current.seeds, first.seeds):
            problems.append(f"{where}: seed images differ from step {steps[0]}")
        if sig.get("seed_idx_sha256") != index_sha256(current.seed_idx):
            problems.append(f"{where}: signature seed_idx_sha256 does not match seed_idx.npy")
        if current.samples.shape != first.samples.shape:
            problems.append(f"{where}: samples {current.samples.shape} != {first.samples.shape}")
    digest = index_sha256(first.seed_idx)
    if seeds_sha256 is not None and digest != seeds_sha256:
        problems.append(f"seed_idx sha256 {digest[:12]} != eval_seeds_500 {seeds_sha256[:12]}")
    if problems:
        raise CRNError("common random numbers violated: " + "; ".join(problems))
    return {
        "steps": steps,
        "rng_seed": sig0.get("rng_seed"),
        "batch_size": sig0.get("batch_size"),
        "n_seeds": int(first.seed_idx.size),
        "seed_idx_sha256": digest,
    }


@dataclass(frozen=True)
class ThumbRow:
    """One row of panel (b): the seed image and its sample at each checkpoint (uint8)."""

    dataset: str
    run_id: str
    image_index: int
    seed: np.ndarray
    samples: dict[int, np.ndarray]
    crn: dict[str, Any]


def thumb_row(dataset: str, rid: str, sets: Mapping[int, LsdSet],
              seeds_sha256: str | None) -> ThumbRow:
    """Assert CRN and return the row of :data:`SEED_POSITION` (one sample per seed)."""
    crn = assert_crn(sets, seeds_sha256)
    first = sets[min(sets)]
    seed = np.array(first.seeds[SEED_POSITION], dtype=np.uint8)
    samples = {s: np.array(sets[s].samples[SEED_POSITION], dtype=np.uint8) for s in sorted(sets)}
    return ThumbRow(dataset, rid, int(first.seed_idx[SEED_POSITION]), seed, samples, crn)


# --------------------------------------------------------------------------------------------
# The numbers of the text
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionCells:
    """Seed-mean Inception precision per (dataset, arm) from ``index.csv``."""

    cells: dict[tuple[str, str], float]
    n_runs: dict[tuple[str, str], int]

    def extreme(self, datasets: Sequence[str], fn: Any) -> tuple[float, tuple[str, str]]:
        """Return ``fn`` (min or max) over the cells of ``datasets`` and its cell."""
        keys = [k for k in self.cells if k[0] in datasets]
        if not keys:
            raise PaperA1Error(f"no precision cells for {list(datasets)}")
        best = fn(keys, key=lambda k: self.cells[k])
        return self.cells[best], best

    def arms(self, dataset: str) -> list[str]:
        """Arms present for ``dataset``, in A0, A1, A2, A3, controls order."""
        order = [*ARM_LABELS, *CONTROL_LABELS]
        present = {a for d, a in self.cells if d == dataset}
        return [a for a in order if a in present] + sorted(present - set(order))


def read_precision(results: Path) -> PrecisionCells:
    """Read the per-run precision of ``<results>/index.csv`` and average it per cell.

    Raises
    ------
    MissingInputError
        If ``index.csv`` is absent.
    PaperA1Error
        If a precision value is missing or not finite.
    """
    path = Path(results) / "index.csv"
    if not path.is_file():
        raise MissingInputError(f"{path} not found")
    values: dict[tuple[str, str], list[float]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["precision"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PaperA1Error(f"{path}: {row.get('run_id')}: precision missing") from exc
            if not math.isfinite(value):
                raise PaperA1Error(f"{path}: {row.get('run_id')}: precision {value}")
            values.setdefault((row["dataset"], row["arm"]), []).append(value)
    cells = {k: float(np.mean(v)) for k, v in sorted(values.items())}
    return PrecisionCells(cells, {k: len(v) for k, v in sorted(values.items())})


@dataclass(frozen=True)
class PhotoChecks:
    """The two one-seed trade-off checks (M7), late-window means."""

    baseline: float
    r128: float
    n32k: float
    n_train_baseline: int
    n_train_n32k: int


def read_photo_checks(path: Path) -> PhotoChecks:
    """Read ``photo_diagnostic.json``: late-window precision and the training-set sizes.

    Raises
    ------
    MissingInputError
        If the file is absent.
    PaperA1Error
        If a key is missing.
    """
    data = _read_json(Path(path))
    try:
        late = data["late_window"]
        tasks = data["tasks"]
        return PhotoChecks(
            baseline=float(late["baseline"]["precision"]),
            r128=float(late["r128"]["precision"]),
            n32k=float(late["n32k"]["precision"]),
            n_train_baseline=int(tasks["baseline"]["env"]["n_train"]),
            n_train_n32k=int(tasks["n32k"]["env"]["n_train"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperA1Error(f"{path}: late_window or tasks key missing ({exc})") from exc


# --------------------------------------------------------------------------------------------
# Table A1
# --------------------------------------------------------------------------------------------

_TEX_SUBS: tuple[tuple[str, str], ...] = (
    ("%", r"\%"),
    ("≈ ", r"$\approx$\,"),
    ("128²", r"$128^2$"),
    ("192²", r"$192^2$"),
    (" K", r" $K$"),
    ("et al. ", r"et al.\ "),
    ("App. B", r"App.~B"),
)


def _tex(text: str) -> str:
    for old, new in _TEX_SUBS:
        text = text.replace(old, new)
    return text


def render_table_tex() -> str:
    """Return Table A1 as a ``booktabs`` tabular that fits the 5.5 in text width."""
    lines = [
        "% Table A1 (T8.3): generated by python -m ihdm.cli.paper_a1; do not edit.",
        "% Fixed facts: Rissanen et al. (ICLR 2023) App. B and results_discussion.md §3; the "
        "'ours' column is cross-checked against every A0 run's config.json, summary.json and "
        "manifest.json.",
        "% Needs \\usepackage{booktabs}. Caption and notes: docs/RESULTS/paper/A1.md.",
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}p{3.1cm}p{5.6cm}p{4.35cm}@{}}",
        "\\toprule",
        # \\raggedright only in the wrapping middle header; the row end sits in the last cell.
        f" & \\raggedright {_tex(TABLE_A1_HEAD[0])} & {_tex(TABLE_A1_HEAD[1])} \\\\",
        "\\midrule",
    ]
    lines += [f"{_tex(a)} & {_tex(b)} & {_tex(c)} \\\\" for a, b, c in TABLE_A1_ROWS]
    lines += [
        "\\bottomrule",
        f"\\multicolumn{{3}}{{@{{}}l}}{{\\footnotesize {_tex(TABLE_A1_NOTE)}}} \\\\",
        "\\end{tabular}",
        "\\endgroup",
        "",
    ]
    return "\n".join(lines)


def render_table_md() -> str:
    """Return Table A1 as a Markdown table."""
    lines = [
        "<!-- Table A1 (T8.3): generated by python -m ihdm.cli.paper_a1; do not edit. -->",
        "",
        "| | " + " | ".join(TABLE_A1_HEAD) + " |",
        "|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in TABLE_A1_ROWS]
    lines += ["", TABLE_A1_NOTE, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Figure A1
# --------------------------------------------------------------------------------------------

_PAPER_RC: dict[str, object] = {
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "svg.fonttype": "none",
    "svg.hashsalt": SVG_HASHSALT,
    "svg.image_inline": True,
    "image.interpolation": "none",
}


@dataclass(frozen=True)
class A1Data:
    """Everything Figure A1 draws; built from files by :func:`build` or synthetically by tests."""

    losses: tuple[DatasetLoss, ...]
    rows: tuple[ThumbRow, ...]
    n_iters: int = 60_000
    checkpoints: tuple[int, ...] = CHECKPOINTS


@dataclass(frozen=True)
class LayoutAudit:
    """Measured legibility of the drawn figure."""

    width_in: float
    height_in: float
    min_font_pt: float
    min_thumb_in: float
    n_texts: int

    def problems(self) -> list[str]:
        """Return every violated format limit."""
        out = []
        if abs(self.width_in - PAPER_WIDTH_IN) > 1e-6:
            out.append(f"width {self.width_in} in != {PAPER_WIDTH_IN}")
        if self.height_in > MAX_HEIGHT_IN + 1e-6:
            out.append(f"height {self.height_in} in > {MAX_HEIGHT_IN}")
        if self.min_font_pt < MIN_FONT_PT:
            out.append(f"smallest font {self.min_font_pt} pt < {MIN_FONT_PT}")
        if self.min_thumb_in < MIN_THUMB_IN:
            out.append(f"smallest thumbnail {self.min_thumb_in:.3f} in < {MIN_THUMB_IN}")
        return out


# Layout in inches (origin bottom-left).
LOSS_X: float = 0.47
LOSS_W: float = 1.66
LOSS_Y: float = 0.31
LOSS_TOP: float = FIG_HEIGHT_IN - 0.33
THUMB_IN: float = 0.47
THUMB_GAP: float = 0.025
#: Vertical gap between the two rows: room for the rotated row labels (wider than a thumbnail).
ROW_GAP: float = 0.08
SEED_GAP: float = 0.07
THUMB_RIGHT: float = PAPER_WIDTH_IN - 0.02
THUMB_TOP: float = FIG_HEIGHT_IN - 0.22
B_LABEL_X: float = 2.27


def _thumb_x0(n_cols: int) -> float:
    return THUMB_RIGHT - (n_cols * THUMB_IN + (n_cols - 2) * THUMB_GAP + SEED_GAP)


def _thumb_x(j: int, n_cols: int) -> float:
    x = _thumb_x0(n_cols) + j * (THUMB_IN + THUMB_GAP)
    if j >= 1:
        x += SEED_GAP - THUMB_GAP
    return x


def _axes(fig: Any, x: float, y: float, w: float, h: float, gid: str, **kw: Any) -> Any:
    width, height = fig.get_size_inches()
    ax = fig.add_axes((x / width, y / height, w / width, h / height), **kw)
    ax.set_gid(gid)
    return ax


def _fig_text(fig: Any, x: float, y: float, text: str, gid: str, **kw: Any) -> Any:
    width, height = fig.get_size_inches()
    artist = fig.text(x / width, y / height, text, **kw)
    artist.set_gid(gid)
    return artist


def _settled_limits(losses: Sequence[DatasetLoss]) -> tuple[float, float] | None:
    values = [r.smooth[r.steps >= SETTLED_STEP] for d in losses for r in d.runs]
    settled = np.concatenate(values) if values else np.empty(0)
    settled = settled[np.isfinite(settled) & (settled > 0)]
    if settled.size == 0:
        return None
    return float(settled.min()) / 1.10, float(settled.max()) * 1.10


def _k_label(step: int) -> str:
    return f"{step // 1000}k" if step % 1000 == 0 else f"{step / 1000:g}k"


def _draw_panel_a(fig: Any, data: A1Data) -> Any:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    from ihdm.analysis.style import EXTENSION_SHADE, MUTED

    _fig_text(fig, 0.02, FIG_HEIGHT_IN - 0.11, "(a)", "a_label", fontsize=8, fontweight="bold",
              ha="left", va="center")
    ax = _axes(fig, LOSS_X, LOSS_Y, LOSS_W, LOSS_TOP - LOSS_Y, "a_loss")
    n_k = data.n_iters / 1000
    ax.axvspan(EXTENSION_START / 1000, n_k, color=EXTENSION_SHADE, zorder=0, lw=0,
               gid="a_extension")
    ax.axvline(EXTENSION_START / 1000, color=MUTED, lw=0.5, ls=":", zorder=1,
               gid="a_extension_start")
    handles = []
    for d in data.losses:
        color, dash = DATASET_STYLE[d.dataset]
        for r in d.runs:
            ax.plot(r.steps / 1000, r.smooth, color=color, lw=0.45, alpha=0.55, ls=dash,
                    zorder=2, gid=f"a_run_{r.run_id}")
        ax.plot(d.steps / 1000, d.mean, color=color, lw=1.3, ls=dash, zorder=3,
                gid=f"a_mean_{d.dataset}")
        at = np.array([np.interp(s, d.steps, d.mean) for s in data.checkpoints])
        ax.plot(np.asarray(data.checkpoints) / 1000, at, ls="none", marker="o", markersize=3.2,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.9, zorder=4,
                gid=f"a_ckpt_{d.dataset}")
        handles.append(Line2D([], [], color=color, lw=1.3, ls=dash,
                              label=f"{DATASET_NAMES[d.dataset]} ({len(d.runs)} runs)"))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if (limits := _settled_limits(data.losses)) is not None:
        ax.set_ylim(*limits)
    ax.set_xlim(0, n_k + 1)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("iteration (thousands)", labelpad=1.5)
    ax.set_ylabel("training loss", labelpad=1.5)
    top = ax.secondary_xaxis(
        "top", functions=(lambda k: iterations_to_epochs(np.asarray(k) * 1000),
                          lambda e: epochs_to_iterations(e) / 1000))
    top.set_gid("a_epoch_axis")
    top.set_xticks([0, 100, 200, 300])
    top.set_xlabel(f"epoch ({ITERS_PER_EPOCH} iterations each)", labelpad=2.0)
    top.tick_params(axis="x", colors=MUTED, length=2.5, width=0.6, pad=1.5)
    top.spines["top"].set_color(MUTED)
    top.spines["top"].set_linewidth(0.6)
    ax.tick_params(axis="both", pad=1.5)
    legend = ax.legend(handles=handles, loc="upper right", handlelength=2.0, borderaxespad=0.3,
                       labelspacing=0.25, frameon=True, facecolor="white", edgecolor="none",
                       framealpha=1.0, borderpad=0.25)
    legend.set_gid("a_legend")
    return ax


def _draw_panel_b(fig: Any, data: A1Data) -> list[Any]:
    from ihdm.analysis.style import MUTED

    _fig_text(fig, B_LABEL_X, FIG_HEIGHT_IN - 0.11, "(b)", "b_label", fontsize=8,
              fontweight="bold", ha="left", va="center")
    n_cols = 1 + len(data.checkpoints)
    headers = ["seed", *(_k_label(s) for s in data.checkpoints)]
    for j, head in enumerate(headers):
        _fig_text(fig, _thumb_x(j, n_cols) + THUMB_IN / 2, THUMB_TOP + 0.04, head,
                  f"b_head_{j}", fontsize=7, ha="center", va="bottom")
    thumbs = []
    for i, row in enumerate(data.rows):
        y = THUMB_TOP - (i + 1) * THUMB_IN - i * ROW_GAP
        _fig_text(fig, _thumb_x(0, n_cols) - 0.05, y + THUMB_IN / 2, DATASET_NAMES[row.dataset],
                  f"b_row_{row.dataset}", fontsize=7, rotation=90, ha="right", va="center")
        images = [row.seed, *(row.samples[s] for s in data.checkpoints)]
        for j, image in enumerate(images):
            ax = _axes(fig, _thumb_x(j, n_cols), y, THUMB_IN, THUMB_IN,
                       f"b_thumb_{row.dataset}_{j}")
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none",
                      gid=f"b_img_{row.dataset}_{j}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_color(MUTED)
            thumbs.append(ax)
    bottom = THUMB_TOP - len(data.rows) * THUMB_IN - (len(data.rows) - 1) * ROW_GAP
    x_mid = (_thumb_x(0, n_cols) + _thumb_x(n_cols - 1, n_cols) + THUMB_IN) / 2
    _fig_text(fig, x_mid, bottom - 0.05, "same seed and sampling noise at every checkpoint",
              "b_note", fontsize=7, color=MUTED, ha="center", va="top")
    return thumbs


def draw_a1(data: A1Data) -> Any:
    """Draw Figure A1 at 5.5 × :data:`FIG_HEIGHT_IN` in (caller saves and closes it)."""
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(PAPER_WIDTH_IN, FIG_HEIGHT_IN))
    _draw_panel_a(fig, data)
    _draw_panel_b(fig, data)
    return fig


def audit_layout(fig: Any) -> LayoutAudit:
    """Measure the figure size, the smallest font and the smallest thumbnail."""
    from matplotlib.text import Text

    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    width, height = (float(v) for v in fig.get_size_inches())
    thumbs = [ax for ax in fig.axes if (ax.get_gid() or "").startswith("b_thumb")]
    sizes = [min(ax.get_position().width * width, ax.get_position().height * height)
             for ax in thumbs]
    return LayoutAudit(width, height, min(float(t.get_fontsize()) for t in texts),
                       min(sizes) if sizes else float("inf"), len(texts))


def paper_style() -> Any:
    """Return a context manager with the house style plus the paper's sizes and SVG settings."""
    from contextlib import ExitStack

    from matplotlib import pyplot as plt

    from ihdm.analysis.style import figure_style

    stack = ExitStack()
    stack.enter_context(figure_style())
    stack.enter_context(plt.rc_context(_PAPER_RC))
    return stack


def save_figure(fig: Any, out_dir: Path, name: str = FIGURE_NAME) -> dict[str, Path]:
    """Write ``<name>.pdf``, ``.png`` (300 dpi) and ``.svg``, byte-stable, and close the figure.

    Must be called inside :func:`paper_style` so that the SVG settings apply.
    """
    from matplotlib import pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {ext: out_dir / f"{name}.{ext}" for ext in ("pdf", "png", "svg")}
    fig.savefig(paths["pdf"], format="pdf", dpi=PDF_DPI,
                metadata={"CreationDate": None, "ModDate": None, "Creator": CREATOR,
                          "Producer": "matplotlib"})
    fig.savefig(paths["png"], format="png", dpi=PNG_DPI, metadata={"Software": None})
    fig.savefig(paths["svg"], format="svg", metadata={"Date": None, "Creator": CREATOR})
    plt.close(fig)
    logger.info("wrote %s", ", ".join(str(p) for p in paths.values()))
    return paths


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class A1Paths:
    """Inputs and outputs of one build (the CLI arguments)."""

    results: Path
    eval_dir: Path
    out: Path
    work: Path
    photo_diagnostic: Path


@dataclass(frozen=True)
class TextNumbers:
    """Every number of the text and caption, with what it was read from."""

    plateau_step: int
    plateau_epoch: float
    plateau_raw: dict[str, int]
    final_loss: dict[str, float]
    loss_at_plateau: dict[str, float]
    photo_max: float
    photo_max_cell: tuple[str, str]
    mri_min: float
    mri_min_cell: tuple[str, str]
    mri_max: float
    mri_max_cell: tuple[str, str]
    n_photo_cells: int
    n_mri_cells: int
    checks: PhotoChecks
    arms: dict[str, list[str]]


@dataclass(frozen=True)
class A1Report:
    """What one build wrote and measured."""

    paths: dict[str, Path]
    audit: LayoutAudit
    rows: tuple[ThumbRow, ...]
    numbers: TextNumbers

    def sha256(self) -> dict[str, str]:
        """sha256 of every written file, by file name."""
        return {p.name: file_sha256(p) for p in sorted(self.paths.values())}


def text_numbers(losses: Sequence[DatasetLoss], precision: PrecisionCells,
                 checks: PhotoChecks) -> TextNumbers:
    """Collect the numbers of the text from the loaded files."""
    raw, final, at = {}, {}, {}
    for d in losses:
        raw[d.dataset], final[d.dataset] = plateau_step(d.steps, d.mean)
    step = int(math.ceil(max(raw.values()) / PLATEAU_ROUND) * PLATEAU_ROUND)
    for d in losses:
        at[d.dataset] = float(np.interp(step, d.steps, d.mean))
    photo_max, photo_cell = precision.extreme(PHOTO_DATASETS, max)
    mri_min, mri_min_cell = precision.extreme(MRI_DATASETS, min)
    mri_max, mri_max_cell = precision.extreme(MRI_DATASETS, max)
    return TextNumbers(
        plateau_step=step, plateau_epoch=float(iterations_to_epochs(step)), plateau_raw=raw,
        final_loss=final, loss_at_plateau=at, photo_max=photo_max, photo_max_cell=photo_cell,
        mri_min=mri_min, mri_min_cell=mri_min_cell, mri_max=mri_max, mri_max_cell=mri_max_cell,
        n_photo_cells=sum(k[0] in PHOTO_DATASETS for k in precision.cells),
        n_mri_cells=sum(k[0] in MRI_DATASETS for k in precision.cells),
        checks=checks, arms={d: precision.arms(d) for d in PHOTO_DATASETS},
    )


def _arm_list(arms: Sequence[str]) -> str:
    names = [ARM_LABELS.get(a) or CONTROL_LABELS.get(a, a) for a in arms]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _arms_phrase(arms: Sequence[str]) -> str:
    """'all four[, plus the A2' spacing control]' when complete, else the names."""
    controls = [CONTROL_LABELS[a].removeprefix("the ") for a in arms if a in CONTROL_LABELS]
    if set(ARM_LABELS) <= set(arms):
        return "all four" + (f", plus the {controls[0]}" if controls else "")
    return _arm_list(arms)


def render_text(n: TextNumbers) -> str:
    """Return the appendix text (about 120 words); the intrinsic-dimension sentence is a
    placeholder that main fills after T8.0 merges."""
    c = n.checks
    ratio = c.n_train_n32k / c.n_train_baseline
    ratio_text = f"{ratio:g}×"
    return (
        "As natural-image references, we trained the configurations of Sec. 3 on LSUN Churches "
        f"({_arms_phrase(n.arms['lsun_church'])}) and Bedrooms "
        f"({_arms_phrase(n.arms['lsun_bedroom'])}); Table A1 lists how this setup differs from "
        "the paper's. The training loss levels off after about "
        f"{n.plateau_step // 1000}k iterations ({n.plateau_epoch:g} epochs; Fig. A1a), yet the "
        "samples stay blurry (Fig. A1b). Inception precision is at most "
        f"{n.photo_max:.2f} in every photograph configuration, against "
        f"{n.mri_min:.2f}–{n.mri_max:.2f} on MRI. Two one-seed checks point to resolution and "
        "framing rather than data size: the paper's whole-scene 128² framing raises precision "
        f"from {c.baseline:.3f} to {c.r128:.3f}, whereas {ratio_text} more training images "
        f"({c.n_train_n32k:,}) give {c.n32k:.3f}. "
        f"{ID_PLACEHOLDER} At this budget the photographs are out of reach, so we do not compare "
        "the two domains."
    )


def render_caption(n: TextNumbers, rows: Sequence[ThumbRow], losses: Sequence[DatasetLoss],
                   ) -> str:
    """Return the caption draft of Figure A1."""
    runs = ", ".join(f"{DATASET_NAMES[d.dataset]} {len(d.runs)}" for d in losses)
    images = ", ".join(f"{DATASET_NAMES[r.dataset]} image {r.image_index}" for r in rows)
    steps = ", ".join(_k_label(s) for s in CHECKPOINTS)
    pct = round(NOISE_SHARE_CITED * 100)
    return (
        "**The natural-image reference runs at our budget (default configuration, A0).** "
        f"(a) Training loss against iteration: running mean over {LOSS_WINDOW_ITERS:,} "
        f"iterations; thin lines are runs ({runs}), thick lines their mean. The top axis gives "
        f"epochs at {ITERS_PER_EPOCH} iterations per epoch "
        f"({OUR_SETUP['n_train']:,} training images, batch {OUR_SETUP['batch_size']}); dots mark "
        "the checkpoints of (b); shading marks the 40k → 60k extension, resumed at iteration "
        f"{EXTENSION_START:,}. About {pct}% of each level's regression target is the added "
        f"training noise and only about {100 - pct}% the deblurring that forms the image, so "
        "the loss is dominated by denoising: its plateau shows that the optimisation settled, "
        "not that the samples converged. "
        f"(b) One training seed per dataset ({images}: the first index of the frozen 500-seed "
        f"evaluation list) and its sample at {steps} iterations, run seed 1, drawn with the same "
        "sampling noise at every checkpoint (common random numbers, asserted), so the columns "
        "differ only by training. The samples stay blurry throughout."
    )


def _fmt_cell(cell: tuple[str, str]) -> str:
    dataset, arm = cell
    names = {**DATASET_NAMES, "ixi": "IXI", "oasis1": "OASIS-1"}
    return f"{names.get(dataset, dataset)} {ARM_LABELS.get(arm) or arm} ({arm})"


def render_notes(paths: A1Paths, audit: LayoutAudit,
                 rows: Sequence[ThumbRow], losses: Sequence[DatasetLoss], n: TextNumbers,
                 setup_lines: Sequence[str], digests: Mapping[str, str]) -> str:
    """Return ``A1.md``: caption, text, table pointer, numbers with sources, checks, files."""
    c = n.checks
    text = render_text(n)
    words = len(text.split())
    words_without = len(text.replace(ID_PLACEHOLDER, "").split())
    lines = [
        "# A1 — the natural-image reference runs (T8.3)",
        "",
        "Generated by `python -m ihdm.cli.paper_a1`; do not edit by hand, rerun the command.",
        f"Inkscape master: `{FIGURE_NAME}.svg` (text kept as text, thumbnails inlined, every "
        f"panel, thumbnail and label a separate element with its own `id`); LaTeX input: "
        f"`{FIGURE_NAME}.pdf`. Table A1: `a1_table.tex` (LaTeX) and `a1_table.md`.",
        "",
        "## Caption draft (Figure A1)",
        "",
        "> " + render_caption(n, rows, losses),
        "",
        "## Text (appendix A1)",
        "",
        "> " + text,
        "",
        f"{words} words with the placeholder, {words_without} without it. The bracketed "
        "intrinsic-dimension sentence is "
        "a **placeholder**: main replaces it with the T8.0 result (Sec. X) after T8.0 merges.",
        "",
        "## Table A1",
        "",
        "See `a1_table.md` / `a1_table.tex`. Its cells are fixed facts (Rissanen et al., ICLR "
        "2023, App. B; `docs/RESULTS/results_discussion.md` §3). The 'ours' column was checked "
        "against the files of every A0 run of panel (a):",
        "",
        *[f"- {line}" for line in setup_lines],
        f"- images seen: {OUR_SETUP['n_iters']:,} × {OUR_SETUP['batch_size']} = "
        f"{OUR_SETUP['n_iters'] * OUR_SETUP['batch_size']:,}, "
        f"{100 * OUR_SETUP['n_iters'] * OUR_SETUP['batch_size'] / PAPER_IMAGES_SEEN:.1f}% of the "
        f"paper's {PAPER_IMAGES_SEEN:,}",
        "",
        "## Numbers and their sources",
        "",
        "| number in the text or caption | value | source |",
        "|---|---|---|",
        f"| plateau (rule: the seed-mean running mean stays within "
        f"{PLATEAU_TOLERANCE:.0%} of its mean over the last {FINAL_WINDOW_ITERS:,} iterations; "
        f"the later of the two datasets, rounded up to {PLATEAU_ROUND:,}) | "
        f"{n.plateau_step:,} iterations = {n.plateau_epoch:g} epochs (raw: "
        + ", ".join(f"{DATASET_NAMES[k]} {v:,}" for k, v in n.plateau_raw.items())
        + ") | `runs/<run>/metrics.canonical.jsonl` |",
        "| seed-mean smoothed loss at the plateau → final | "
        + "; ".join(f"{DATASET_NAMES[k]} {n.loss_at_plateau[k]:.4f} → {n.final_loss[k]:.4f} "
                    f"({100 * (n.final_loss[k] / n.loss_at_plateau[k] - 1):+.1f}%)"
                    for k in n.final_loss)
        + " | `runs/<run>/metrics.canonical.jsonl` |",
        f"| precision, photograph maximum over {n.n_photo_cells} configurations (seed means) | "
        f"{n.photo_max:.4f} ({_fmt_cell(n.photo_max_cell)}) | `<results>/index.csv` `precision` |",
        f"| precision, MRI minimum and maximum over {n.n_mri_cells} configurations | "
        f"{n.mri_min:.4f} ({_fmt_cell(n.mri_min_cell)}) – {n.mri_max:.4f} "
        f"({_fmt_cell(n.mri_max_cell)}) | `<results>/index.csv` `precision` |",
        f"| one-seed checks, late window: baseline → r128 / n32k | {c.baseline:.3f} → "
        f"{c.r128:.3f} / {c.n32k:.3f} | `photo_diagnostic.json` `late_window.*.precision` |",
        f"| training images, baseline / n32k | {c.n_train_baseline:,} / {c.n_train_n32k:,} | "
        "`photo_diagnostic.json` `tasks.*.env.n_train` |",
        f"| iterations per epoch | {ITERS_PER_EPOCH} | `summary.json` `env.n_train` / "
        "`config.json` `training.batch_size`, every run |",
        f"| share of the regression target that is training noise | about "
        f"{round(NOISE_SHARE_CITED * 100)}% | cited: `projects/GenAI/learning/"
        "03-terminal-blur-scaffolding.md` §6, eq. (6.5) |",
        "",
        "Precision is not in `docs/RESULTS/tables/tables.json` (its tables 1a–1c carry KID, FID, "
        "recall and coverage), so the cells are averaged from the per-run `precision` column of "
        "`index.csv`, the same values as `docs/RESULTS/exploratory/exploratory.json` `per_run` and "
        "`results_discussion.md` §4.",
        "",
        "## Checks",
        "",
    ]
    for row in rows:
        crn = row.crn
        lines.append(
            f"- **CRN, {row.run_id}:** steps {', '.join(f'{s:,}' for s in crn['steps'])}: "
            f"identical `seed_idx.npy` and seed images, rng_seed {crn['rng_seed']}, batch_size "
            f"{crn['batch_size']}, one sample per seed, {crn['n_seeds']} seeds; seed_idx sha256 "
            f"`{crn['seed_idx_sha256'][:16]}…` equals `summary.json` "
            f"`seed_lists.intermediate_sha256` (the frozen `eval_seeds_500.npy`); shown: "
            f"position {SEED_POSITION}, image {row.image_index}.")
    lines += [
        "- **Histories:** read through `canonical_records`; every run of a dataset is logged at "
        "the same steps; running mean over "
        + ", ".join(f"{r.run_id} {r.window_records}" for d in losses for r in d.runs[:1])
        + " records (one record per 50 iterations).",
        f"- **Layout (measured on the drawn figure):** {audit.width_in:.2f} × "
        f"{audit.height_in:.2f} in, smallest font {audit.min_font_pt:g} pt over {audit.n_texts} "
        f"text elements, smallest thumbnail {audit.min_thumb_in:.2f} in (limits: "
        f"{PAPER_WIDTH_IN} in wide, ≤ {MAX_HEIGHT_IN} in, ≥ {MIN_FONT_PT:g} pt, ≥ "
        f"{MIN_THUMB_IN} in). Thumbnails: 192 px at {THUMB_IN} in = "
        f"{192 / THUMB_IN:.0f} dpi, embedded unresampled.",
        "",
        "## Files",
        "",
        "| file | sha256 |",
        "|---|---|",
        *[f"| `{name}` | `{digest}` |" for name, digest in digests.items()],
        "",
        "Byte stability: a second run of the command must reproduce these digests.",
        "",
        "## Command",
        "",
        "```bash",
        f"python -m ihdm.cli.paper_a1 --results {paths.results} --eval-dir {paths.eval_dir} "
        "--out docs/RESULTS/paper --work <scratch>",
        "```",
        "",
        "Environment: `OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=\"\" nice -n 19`, "
        "`PYTHONPATH=<repository>`, conda env `ihdm`.",
        "",
    ]
    return "\n".join(lines)


def build(paths: A1Paths) -> A1Report:
    """Build Figure A1, Table A1 and ``A1.md`` from files.

    Parameters
    ----------
    paths : A1Paths
        Inputs and outputs.

    Returns
    -------
    A1Report
        Written files, the layout audit, the thumbnail rows and the numbers of the text.

    Raises
    ------
    MissingInputError
        If an input file is absent.
    PaperA1Error
        If an assertion fails (CRN, setup, layout).
    """
    losses = load_losses(paths.results)
    setups = {r.run_id: read_run_setup(paths.results, r.run_id) for d in losses for r in d.runs}
    setup_lines = assert_our_setup(setups)
    precision = read_precision(paths.results)
    checks = read_photo_checks(paths.photo_diagnostic)
    rows = []
    for dataset in DATASETS:
        rid = run_id(dataset, ARM, THUMB_RUN_SEED)
        folder = extract_checkpoints(paths.eval_dir, rid, paths.work)
        sets = {s: load_lsd_set(folder / f"{s:06d}" / SET_NAME, s) for s in CHECKPOINTS}
        rows.append(thumb_row(dataset, rid, sets, setups[rid]["eval_seeds_500_sha256"]))
        logger.info("%s: CRN holds over %s; image %d", rid, CHECKPOINTS, rows[-1].image_index)
    numbers = text_numbers(losses, precision, checks)
    data = A1Data(tuple(losses), tuple(rows), n_iters=OUR_SETUP["n_iters"])
    out = Path(paths.out)
    with paper_style():
        fig = draw_a1(data)
        audit = audit_layout(fig)
        if problems := audit.problems():
            from matplotlib import pyplot as plt

            plt.close(fig)
            raise PaperA1Error("layout: " + "; ".join(problems))
        written = save_figure(fig, out)
    out.mkdir(parents=True, exist_ok=True)
    table_tex = out / "a1_table.tex"
    table_md = out / "a1_table.md"
    table_tex.write_text(render_table_tex())
    table_md.write_text(render_table_md())
    files = {"pdf": written["pdf"], "png": written["png"], "svg": written["svg"],
             "tex": table_tex, "md": table_md}
    digests = {p.name: file_sha256(p) for p in sorted(files.values())}
    notes = out / "A1.md"
    notes.write_text(render_notes(paths, audit, rows, losses, numbers, setup_lines, digests))
    files["notes"] = notes
    return A1Report(files, audit, tuple(rows), numbers)
