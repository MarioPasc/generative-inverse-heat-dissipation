"""Evaluate one training run from its EMA checkpoints (``05-metrics.md`` §8a and §9).

This is the functional core behind ``python -m ihdm.cli.evaluate_run``: given a run directory it
draws every sample set the metrics need, in process, with the merged sampler, and writes the
result files of §9. The evaluation array of T5.1 runs one of these per training run on an A100.

The sampling budget is D16's, cut from 25 000 chains per run to 8 000:

===================  ===============================  =======================================
set                  chains                           feeds
===================  ===============================  =======================================
``lsd``              1 per seed, 500 seeds, 8 steps    LSD at every evaluated checkpoint, T_tau
``final``            1 per seed, 2 000 seeds           final LSD, KID/FID, recall/coverage, M
``heldout``          50 per seed, 40 seed subjects     diversity, inherited band, PCA
===================  ===============================  =======================================

D17's common random numbers are what make the comparisons paired: the 500 intermediate seeds are
**one frozen list per dataset**, written here on first use, reused at every checkpoint of every
run of that dataset; the sampling RNG seed and the sampling batch are fixed too, so row ``k`` of
the sample set means the same seed and the same noise stream at 35k as at 40k and in A0 as in A3.
Every sampling call passes its indices explicitly through the sampler's ``file`` source, which
refuses an index outside the split it declares.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ihdm.data.errors import DataFormatError
from ihdm.data.format import read_dataset
from ihdm.metrics.diversity import pca_around_seed, within_seed_diversity
from ihdm.metrics.errors import MetricError
from ihdm.metrics.io import read_json, write_json
from ihdm.metrics.memorisation import memorisation_ratio
from ihdm.metrics.spectral import (
    LOW_BAND_SIGMA_PX,
    inherited_band,
    lsd,
    t_tau,
)
from ihdm.paths import repo_root
from ihdm.sampling.chain import SampleRequest, resolve_request, sample_from_seeds
from ihdm.sampling.errors import SamplingError
from ihdm.sampling.loader import (
    checkpoint_sha256,
    load_checkpoint,
    load_ema_model,
    load_run_config,
)
from ihdm.sampling.seeds import assert_in_split, load_seed_images
from ihdm.spectral.power import mode_power
from ihdm.stats.bootstrap import paired_lsd_gate

__all__ = [
    "AMP_MODES",
    "EVALUATED_STEPS",
    "EvalRequest",
    "SeedList",
    "SeedLists",
    "checkpoint_table",
    "ensure_seed_lists",
    "evaluate_run",
    "metrics_dirname",
    "samples_dirname",
    "select_checkpoints",
]

logger = logging.getLogger(__name__)

#: The eight checkpoints D16 evaluates, in iterations.
EVALUATED_STEPS: tuple[int, ...] = (5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)

#: Canonical sizes of the two frozen seed lists (D17).
N_SEEDS_INTERMEDIATE: int = 500
N_SEEDS_FINAL: int = 2000

#: RNG seeds of the two lists and of the samplings, frozen by D17.
SEED_LIST_RNG_INTERMEDIATE: int = 2026
SEED_LIST_RNG_FINAL: int = 0
SAMPLING_RNG_INTERMEDIATE: int = 2026
SAMPLING_RNG_FINAL: int = 0

#: File names of the two lists inside the dataset directory.
SEED_LIST_NAMES: dict[str, str] = {
    "intermediate": "eval_seeds_500.npy",
    "final": "eval_seeds_final_2000.npy",
}

#: Sampling batch, pinned per run because samples reproduce only at a fixed batch (D16).
DEFAULT_SAMPLE_BATCH: int = 32

#: Sampling precisions of ``--amp``. ``off`` is the contract default (D16); the other two run the
#: network under ``torch.autocast`` with that dtype and change the samples bitwise.
AMP_MODES: tuple[str, ...] = ("off", "fp16", "bf16")
_AMP_DTYPE: dict[str, str | None] = {"off": None, "fp16": "float16", "bf16": "bfloat16"}

#: Low-pass length-scale of ``D_lp`` and ``M_lp`` (``05-metrics.md`` §3, §4).
SIGMA_LP: float = 16.0

_STEP_PATTERN = re.compile(r"ema_iter_(\d+)\.pt$")


# --------------------------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRequest:
    """Everything the evaluation of one run needs.

    Parameters
    ----------
    run : Path
        The run directory (the trainer's workdir).
    ckpts : str
        ``"all"``, ``"final"`` or a comma-separated list of steps.
    n_lsd : int
        Seeds of the intermediate LSD set, a prefix of the frozen 500.
    n_final : int
        Seeds of the shared final set, a prefix of the frozen 2 000.
    n_seeds : int
        Held-out seed subjects.
    n_per_seed : int
        Samples per held-out seed.
    sample_batch : int
        Sampling batch; pinned per run, recorded in every ``request.json``.
    fid_batch : int
        Images per Inception forward pass.
    skip_inception : bool
        Skip §7 entirely (no weights, or a quick check).
    device : str or None
        Torch device; ``None`` follows the run's config, falling back to the CPU.
    force : bool
        Recompute samples and result files that already exist.
    a0_final_lsd : float or None
        The threshold of ``T_tau``: the final LSD of the A0 run with the same seed.
    gate : tuple[int, int] or None
        The two checkpoints of the plateau gate, earlier first.
    n_boot_inception, n_boot_gate : int
        Resamples of the two bootstraps.
    k : int
        Neighbour count of recall/coverage.
    amp : str
        Sampling precision, one of :data:`AMP_MODES`. Every mode other than ``"off"`` writes to
        its own ``samples_amp-<mode>/`` and ``metrics_amp-<mode>/`` trees, so samples drawn
        under two precisions can never be mixed in one cache or one result file.
    """

    run: Path
    ckpts: str = "all"
    n_lsd: int = N_SEEDS_INTERMEDIATE
    n_final: int = N_SEEDS_FINAL
    n_seeds: int = 40
    n_per_seed: int = 50
    sample_batch: int = DEFAULT_SAMPLE_BATCH
    fid_batch: int = 64
    skip_inception: bool = False
    device: str | None = None
    force: bool = False
    a0_final_lsd: float | None = None
    gate: tuple[int, int] | None = None
    n_boot_inception: int = 200
    n_boot_gate: int = 1000
    k: int = 5
    amp: str = "off"


# --------------------------------------------------------------------------------------------
# The frozen seed lists (D17)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedList:
    """One frozen evaluation seed list of a dataset.

    Parameters
    ----------
    name : str
        ``"intermediate"`` or ``"final"``.
    path : Path
        The ``.npy`` holding the dataset indices.
    idx : numpy.ndarray
        The indices themselves.
    sha256 : str
        Digest of the array's bytes, recorded in ``summary.json`` so that two runs can be shown
        to have been evaluated on the same seeds.
    rule : str
        How the list was drawn.
    """

    name: str
    path: Path
    idx: np.ndarray
    sha256: str
    rule: str

    @property
    def sidecar(self) -> Path:
        """The JSON written beside the array."""
        return self.path.with_suffix(".json")


@dataclass(frozen=True)
class SeedLists:
    """The two frozen lists of one dataset."""

    intermediate: SeedList
    final: SeedList


def _sha256_array(array: np.ndarray) -> str:
    """Return the SHA-256 of an array's bytes in a canonical dtype and order."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.int64).tobytes()).hexdigest()


def _write_seed_list(
    path: Path, idx: np.ndarray, rule: str, dataset_sha256: str, name: str
) -> SeedList:
    """Write a seed list and its sidecar, and return the record."""
    np.save(path, np.asarray(idx, dtype=np.int64))
    digest = _sha256_array(idx)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "name": name,
                "rule": rule,
                "n": int(np.asarray(idx).size),
                "n_distinct": int(np.unique(idx).size),
                "sha256": digest,
                "dataset_sha256": dataset_sha256,
                "created": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info("wrote the %s evaluation seed list to %s", name, path)
    return SeedList(name=name, path=path, idx=np.asarray(idx, dtype=np.int64), sha256=digest,
                    rule=rule)


def ensure_seed_lists(
    dataset_root: Path,
    n_intermediate: int = N_SEEDS_INTERMEDIATE,
    n_final: int = N_SEEDS_FINAL,
    force: bool = False,
) -> SeedLists:
    """Create, or read back, the two frozen evaluation seed lists of a dataset (D17).

    The lists are written **once per dataset** and reused by every run of every arm, which is
    what makes the LSD contrasts and the plateau gate paired at the seed level. A run evaluated
    with fewer samples than the full budget takes a *prefix* of the list rather than an
    independent smaller draw, so that its noise streams remain a subset of the full run's.

    The intermediate list is drawn without replacement and is therefore capped at the size of the
    training split; the final list is drawn with replacement, as ``05-metrics.md`` §8a requires.
    Both are checked against ``splits["train"]`` on every read.

    Parameters
    ----------
    dataset_root : Path
        A standard-format dataset directory.
    n_intermediate, n_final : int
        Sizes of the two lists.
    force : bool
        Redraw even when the files exist.

    Returns
    -------
    SeedLists
        The two lists with their digests.

    Raises
    ------
    MetricError
        If the directory is not a standard-format dataset, its training split is empty, or a
        stored list is not a 1-D integer array inside the training split.
    """
    root = Path(dataset_root)
    try:
        _, _, splits, meta = read_dataset(root, mmap=True)
    except DataFormatError as error:
        raise MetricError(f"{root}: not a standard-format dataset ({error})") from error

    pool = np.asarray(splits.get("train", []), dtype=np.int64)
    if pool.size == 0:
        raise MetricError(f"{root}: the training split is empty")
    dataset_sha256 = str(getattr(meta, "sha256_images", "") or "")

    lists: dict[str, SeedList] = {}
    for name, (count, rng_seed, replace) in {
        "intermediate": (n_intermediate, SEED_LIST_RNG_INTERMEDIATE, False),
        "final": (n_final, SEED_LIST_RNG_FINAL, True),
    }.items():
        path = root / SEED_LIST_NAMES[name]
        effective = int(count) if replace else int(min(count, pool.size))
        if effective < int(count):
            logger.warning(
                "the training split holds %d images; the %s list is capped at %d distinct seeds",
                pool.size,
                name,
                effective,
            )
        rule = (
            f"{effective} indices of splits['train'] drawn "
            f"{'with' if replace else 'without'} replacement with "
            f"numpy.random.default_rng({rng_seed})"
        )
        if path.exists() and not force:
            try:
                stored = np.asarray(np.load(path))
            except (OSError, ValueError) as error:
                raise MetricError(f"{path}: cannot be read ({error})") from error
            if stored.ndim != 1 or not np.issubdtype(stored.dtype, np.integer):
                raise MetricError(f"{path}: expected a 1-D integer array, got {stored.shape}")
            try:
                assert_in_split(stored, pool, "train")
            except SamplingError as error:
                raise MetricError(f"{path}: {error}") from error
            lists[name] = SeedList(
                name=name,
                path=path,
                idx=np.asarray(stored, dtype=np.int64),
                sha256=_sha256_array(stored),
                rule=rule,
            )
            continue
        rng = np.random.default_rng(rng_seed)
        drawn = np.asarray(rng.choice(pool, size=effective, replace=replace), dtype=np.int64)
        lists[name] = _write_seed_list(path, drawn, rule, dataset_sha256, name)

    return SeedLists(intermediate=lists["intermediate"], final=lists["final"])


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


def checkpoint_table(run: Path) -> dict[int, Path]:
    """Return ``{step: path}`` for every ``ema_iter_*.pt`` of a run.

    Parameters
    ----------
    run : Path
        The run directory.

    Returns
    -------
    dict[int, Path]
        Sorted by step.

    Raises
    ------
    MetricError
        If the run holds no EMA checkpoint.
    """
    directory = Path(run) / "checkpoints"
    table: dict[int, Path] = {}
    for path in sorted(directory.glob("ema_iter_*.pt")):
        match = _STEP_PATTERN.search(path.name)
        if match:
            table[int(match.group(1))] = path.resolve()
    if not table:
        raise MetricError(f"{directory}: no ema_iter_*.pt checkpoint found")
    return dict(sorted(table.items()))


def select_checkpoints(table: dict[int, Path], spec: str) -> tuple[list[int], str]:
    """Resolve a ``--ckpts`` specification against the checkpoints a run actually holds.

    Parameters
    ----------
    table : dict[int, Path]
        The output of :func:`checkpoint_table`.
    spec : str
        ``"all"`` (the eight evaluated steps of D16 present in the run; every available step when
        none of them is, which is what a short pilot run has), ``"final"`` (the largest step), or
        a comma-separated list of steps.

    Returns
    -------
    tuple[list[int], str]
        The selected steps, sorted, and a label naming how they were selected.

    Raises
    ------
    MetricError
        If the specification is empty, not parsable, or names a step the run does not hold.
    """
    text = str(spec).strip()
    if not text:
        raise MetricError("--ckpts is empty")
    available = sorted(table)
    if text == "final":
        return [available[-1]], "final"
    if text == "all":
        chosen = [step for step in EVALUATED_STEPS if step in table]
        if chosen:
            return chosen, "all-evaluated"
        logger.warning(
            "none of the evaluated steps %s is present; falling back to every checkpoint (%s)",
            list(EVALUATED_STEPS),
            available,
        )
        return available, "all-available"
    try:
        wanted = sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError as error:
        raise MetricError(f"--ckpts {text!r} is not 'all', 'final' or a list of steps") from error
    if not wanted:
        raise MetricError(f"--ckpts {text!r} names no step")
    missing = [step for step in wanted if step not in table]
    if missing:
        raise MetricError(f"the run has no checkpoint at steps {missing}; it has {available}")
    return wanted, "explicit"


# --------------------------------------------------------------------------------------------
# The dataset views
# --------------------------------------------------------------------------------------------


@dataclass
class DatasetViews:
    """The image subsets one evaluation needs, read once.

    Parameters
    ----------
    root : Path
        The dataset directory.
    dataset_sha256 : str
        ``meta.json``'s ``sha256_images``.
    reference_idx : numpy.ndarray
        Dataset rows of the ``ref`` split (800 images: the 80 held-out subjects, of which the 40
        seed subjects are a part). This is the reference stack of ``05-metrics.md`` §2.
    train_idx : numpy.ndarray
        Dataset rows of the training split, **sorted**, so a seed index maps to a row of the
        ``train`` array by a plain ``searchsorted``.
    heldout_idx : numpy.ndarray
        ``ref`` minus the seed subjects' rows: the denominator of ``M``.
    train_subjects : list[str]
        Subject of every row of ``train_idx``.
    """

    root: Path
    dataset_sha256: str
    reference_idx: np.ndarray
    train_idx: np.ndarray
    heldout_idx: np.ndarray
    train_subjects: list[str]
    _images: Any = field(repr=False, default=None)
    _cache: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def _take(self, key: str, rows: np.ndarray) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.asarray(self._images[rows], dtype=np.uint8)
        return self._cache[key]

    @property
    def reference(self) -> np.ndarray:
        """The ``ref`` split as a ``uint8`` stack."""
        return self._take("reference", self.reference_idx)

    @property
    def train(self) -> np.ndarray:
        """The training split as a ``uint8`` stack, in sorted dataset order."""
        return self._take("train", self.train_idx)

    @property
    def heldout(self) -> np.ndarray:
        """The held-out real images of ``M``'s denominator."""
        return self._take("heldout", self.heldout_idx)

    def train_rows(self, seed_idx: np.ndarray) -> np.ndarray:
        """Map dataset indices of training seeds to rows of :attr:`train`.

        ``memorisation_ratio`` wants rows of the array it was handed, not dataset indices, and a
        dataset index is usually also a valid row, so getting this wrong produces a wrong
        ``seed_nn_fraction`` rather than an exception. The round trip is asserted.

        Parameters
        ----------
        seed_idx : numpy.ndarray
            Dataset indices, every one of them in the training split.

        Returns
        -------
        numpy.ndarray
            Rows of :attr:`train`.

        Raises
        ------
        MetricError
            If the round trip fails, i.e. an index is not in the training split.
        """
        indices = np.asarray(seed_idx, dtype=np.int64)
        rows = np.searchsorted(self.train_idx, indices)
        if rows.size and (
            rows.max() >= self.train_idx.size
            or not np.array_equal(self.train_idx[rows], indices)
        ):
            raise MetricError(
                "seed indices do not round-trip through the training split; they are not "
                "dataset rows of splits['train']"
            )
        return np.asarray(rows, dtype=np.int64)


def load_views(dataset_root: Path) -> DatasetViews:
    """Read the dataset once and expose the subsets the metrics need.

    Parameters
    ----------
    dataset_root : Path
        A standard-format dataset directory.

    Returns
    -------
    DatasetViews
        The views; the images stay memory-mapped until a subset is asked for.

    Raises
    ------
    MetricError
        If the directory is not a standard-format dataset, or the ``ref`` split is empty or does
        not strictly contain the seed split.
    """
    root = Path(dataset_root)
    try:
        images, index, splits, meta = read_dataset(root, mmap=True)
    except DataFormatError as error:
        raise MetricError(f"{root}: not a standard-format dataset ({error})") from error

    reference = np.asarray(splits.get("ref", []), dtype=np.int64)
    seed_rows = np.asarray(splits.get("seed", []), dtype=np.int64)
    train = np.sort(np.asarray(splits.get("train", []), dtype=np.int64))
    if reference.size < 2:
        raise MetricError(f"{root}: the ref split holds {reference.size} images")
    heldout = np.setdiff1d(reference, seed_rows)
    if heldout.size == 0:
        raise MetricError(
            f"{root}: the ref split is entirely made of seed subjects, so M has no denominator"
        )
    subjects = index.iloc[train]["subject"].astype(str).tolist()
    return DatasetViews(
        root=root,
        dataset_sha256=str(getattr(meta, "sha256_images", "") or ""),
        reference_idx=np.sort(reference),
        train_idx=train,
        heldout_idx=heldout,
        train_subjects=subjects,
        _images=images,
    )


# --------------------------------------------------------------------------------------------
# Sampling with a cache
# --------------------------------------------------------------------------------------------


def _check_amp(amp: str) -> str:
    """Return ``amp`` if it is one of :data:`AMP_MODES`, else raise :class:`MetricError`."""
    if amp not in AMP_MODES:
        raise MetricError(f"amp must be one of {AMP_MODES}, got {amp!r}")
    return amp


def samples_dirname(amp: str = "off") -> str:
    """Return the name of the sample-cache tree of a sampling precision.

    ``"off"`` keeps the historical ``samples`` so that the fp32 caches written before ``--amp``
    existed stay valid; every other mode gets ``samples_amp-<mode>``.

    Parameters
    ----------
    amp : str
        One of :data:`AMP_MODES`.

    Returns
    -------
    str
        The directory name under the run directory.

    Raises
    ------
    MetricError
        If ``amp`` is not a known mode.
    """
    return "samples" if _check_amp(amp) == "off" else f"samples_amp-{amp}"


def metrics_dirname(amp: str = "off") -> str:
    """Return the name of the result-file tree of a precision (see :func:`samples_dirname`).

    Parameters
    ----------
    amp : str
        One of :data:`AMP_MODES`.

    Returns
    -------
    str
        ``"metrics"`` for ``"off"``, ``"metrics_amp-<mode>"`` otherwise.

    Raises
    ------
    MetricError
        If ``amp`` is not a known mode.
    """
    return "metrics" if _check_amp(amp) == "off" else f"metrics_amp-{amp}"


def _amp_signature(amp: str) -> bool | str:
    """Return the ``amp`` entry of a cache signature.

    ``"off"`` maps to ``False``, the value every fp32 ``request.json`` written before ``--amp``
    existed already holds, so those caches are still reused; the other modes record their name.
    """
    return False if _check_amp(amp) == "off" else amp


@dataclass(frozen=True)
class SampleSet:
    """One drawn sample set and where it came from.

    Parameters
    ----------
    name : str
        ``"lsd"``, ``"final"`` or ``"heldout"``.
    samples : numpy.ndarray
        ``(S, M, H, W)`` ``uint8``.
    seed_idx : numpy.ndarray
        Dataset index of every seed row.
    seeds : numpy.ndarray
        The seed images, ``(S, H, W)`` ``uint8``.
    directory : Path
        Where the three files live.
    reused : bool
        Whether the set was read from the cache rather than drawn.
    elapsed_s : float
        Wall-clock seconds spent sampling (zero when reused).
    """

    name: str
    samples: np.ndarray
    seed_idx: np.ndarray
    seeds: np.ndarray
    directory: Path
    reused: bool
    elapsed_s: float

    @property
    def flat(self) -> np.ndarray:
        """The samples as a flat ``(S*M, H, W)`` stack."""
        return self.samples.reshape(-1, *self.samples.shape[2:])


def _signature(
    name: str,
    ckpt_sha: str,
    seed_idx: np.ndarray,
    request: SampleRequest,
    resolved: tuple[float, int, bool],
    device: str,
    amp_signature: bool | str = False,
) -> dict[str, Any]:
    """Return the fingerprint a cached sample set must match to be reused."""
    delta, start_level, prior_noise = resolved
    return {
        "set": name,
        "checkpoint_sha256": ckpt_sha,
        "seed_idx_sha256": _sha256_array(seed_idx),
        "n_seeds": int(np.asarray(seed_idx).size),
        "n_per_seed": int(request.n_per_seed),
        "rng_seed": int(request.rng_seed),
        "batch_size": int(request.batch_size),
        "amp": amp_signature,
        "delta": float(delta),
        "start_level": int(start_level),
        "prior_noise": bool(prior_noise),
        "device_type": str(device).split(":")[0],
    }


def draw_set(
    name: str,
    workdir: Path,
    step: int,
    ckpt_path: Path,
    config: Any,
    dataset_root: Path,
    seed_source: str,
    n_seeds: int,
    n_per_seed: int,
    rng_seed: int,
    batch_size: int,
    device: str,
    model_loader: Any,
    idx_file: Path | None = None,
    split: str = "train",
    force: bool = False,
    amp: str = "off",
) -> SampleSet:
    """Draw one sample set, or reuse the cached one, and write the ``sample_ckpt`` layout.

    Parameters
    ----------
    name : str
        The set's name; also its directory under ``<run>/samples/<step>/``.
    workdir : Path
        The run directory.
    step : int
        Checkpoint step, used for the cache path.
    ckpt_path : Path
        The EMA checkpoint.
    config : ml_collections.ConfigDict
        The run's config.
    dataset_root : Path
        The dataset the seeds come from.
    seed_source : {"file", "seed"}
        ``"file"`` reads the frozen list at ``idx_file``; ``"seed"`` takes the held-out seed
        subjects. The ``"train"`` source is never used here, because D17 forbids drawing seeds
        inside the sampler.
    n_seeds, n_per_seed : int
        Size of the set.
    rng_seed, batch_size : int
        The sampling RNG seed and the pinned batch.
    device : str
        Torch device string.
    model_loader : Callable[[], torch.nn.Module]
        Returns the EMA model on ``device``; called only when the set must actually be drawn.
    idx_file : Path or None
        The frozen seed list, required by ``seed_source="file"``.
    split : str
        The split the frozen list declares.
    force : bool
        Redraw even when a matching cache exists.
    amp : str
        Sampling precision, one of :data:`AMP_MODES`; it selects the cache tree
        (:func:`samples_dirname`) and enters the signature.

    Returns
    -------
    SampleSet
        The samples with their provenance.

    Raises
    ------
    MetricError
        If the seeds cannot be loaded, the sampler refuses the request, or the cache directory
        holds a set drawn under a different precision (it is never overwritten, even with
        ``force``).
    """
    amp_signature = _amp_signature(amp)
    directory = Path(workdir) / samples_dirname(amp) / f"{step:06d}" / name
    try:
        seeds_u8, seed_idx = load_seed_images(
            Path(dataset_root), seed_source, n_seeds, rng_seed, idx_file=idx_file, split=split
        )
    except SamplingError as error:
        raise MetricError(f"{name} set: {error}") from error

    request = SampleRequest(
        n_per_seed=int(n_per_seed),
        batch_size=int(batch_size),
        rng_seed=int(rng_seed),
        amp=amp != "off",
        amp_dtype=_AMP_DTYPE[amp],
    )
    try:
        resolved = resolve_request(request, config)
    except SamplingError as error:
        raise MetricError(f"{name} set: {error}") from error
    ckpt_sha = checkpoint_sha256(Path(ckpt_path))
    signature = _signature(name, ckpt_sha, seed_idx, request, resolved, device, amp_signature)

    samples_path = directory / "samples.npy"
    record_path = directory / "request.json"
    if record_path.exists():
        stored_amp = json.loads(record_path.read_text()).get("signature", {}).get("amp", False)
        if stored_amp != amp_signature:
            raise MetricError(
                f"{directory} holds a {name} set drawn with amp={stored_amp!r}, but this request "
                f"samples with amp={amp_signature!r}; refusing to mix or overwrite precisions"
            )
    if samples_path.exists() and record_path.exists() and not force:
        stored = json.loads(record_path.read_text())
        if stored.get("signature") == signature:
            samples = np.load(samples_path)
            expected = (seeds_u8.shape[0], int(n_per_seed), *seeds_u8.shape[1:])
            if tuple(samples.shape) == expected:
                logger.info("reusing the cached %s set at %s", name, directory)
                return SampleSet(
                    name=name,
                    samples=samples,
                    seed_idx=seed_idx,
                    seeds=seeds_u8,
                    directory=directory,
                    reused=True,
                    elapsed_s=0.0,
                )
        logger.info("the cached %s set at %s does not match the request; redrawing", name,
                    directory)

    model = model_loader()
    start = time.perf_counter()
    try:
        drawn = sample_from_seeds(model, config, seeds_u8, request, device)
    except SamplingError as error:
        raise MetricError(f"{name} set: {error}") from error
    elapsed = time.perf_counter() - start

    # A non-finite value (e.g. a half-precision overflow under --amp) would otherwise pass the
    # clamp as NaN and become an arbitrary byte in the uint8 cast, i.e. a silently wrong sample.
    n_bad = int(np.size(drawn) - np.count_nonzero(np.isfinite(drawn)))
    if n_bad:
        raise MetricError(
            f"{name} set at step {step}: {n_bad} of {np.size(drawn)} sample values are not "
            f"finite (amp={amp}); nothing was written"
        )
    samples = np.rint(drawn * 255.0).clip(0, 255).astype(np.uint8)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(samples_path, samples)
    np.save(directory / "seed_idx.npy", np.asarray(seed_idx, dtype=np.int64))
    np.save(directory / "seeds.npy", seeds_u8)
    n_chains = int(samples.shape[0] * samples.shape[1])
    record_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "set": name,
                "step": int(step),
                "checkpoint": str(ckpt_path),
                "dataset_root": str(dataset_root),
                "seed_source": seed_source,
                "seed_list": None if idx_file is None else str(idx_file),
                "shape": list(samples.shape),
                "device": str(device),
                "amp": amp,
                "elapsed_s": elapsed,
                "s_per_chain": elapsed / n_chains if n_chains else None,
                "created": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info(
        "%s set: %d chains at step %d in %.1f s (%.3f s/chain) -> %s",
        name, n_chains, step, elapsed, elapsed / n_chains if n_chains else float("nan"), directory,
    )
    return SampleSet(
        name=name,
        samples=samples,
        seed_idx=seed_idx,
        seeds=seeds_u8,
        directory=directory,
        reused=False,
        elapsed_s=elapsed,
    )


# --------------------------------------------------------------------------------------------
# The metric blocks
# --------------------------------------------------------------------------------------------


def _lsd_record(samples: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Return the LSD block of ``05-metrics.md`` §9."""
    result = lsd(samples, reference)
    return {
        "lsd": result.lsd,
        "lsd_octaves": dict(result.octaves),
        "variance_ratio": result.variance_ratio,
        "n_samples": result.n_samples,
        "n_reference": result.n_reference,
    }


def _finite_curve(centres: np.ndarray, *curves: np.ndarray) -> dict[str, list[float]]:
    """Drop the radial bins that hold no mode of the grid, which carry ``nan``.

    ``write_json`` refuses non-finite floats by contract, and five of the 48 bins hold no mode at
    ``W = 192`` (T4.1 §6). The bins are dropped rather than written as ``null`` so that the three
    arrays stay index-aligned and a consumer can plot them without filtering.
    """
    mask = np.ones(np.asarray(centres).shape, dtype=bool)
    for curve in curves:
        mask &= np.isfinite(np.asarray(curve, dtype=np.float64))
    return {
        "centres": [float(value) for value in np.asarray(centres)[mask]],
        "values": [[float(value) for value in np.asarray(curve)[mask]] for curve in curves],
        "n_bins": int(mask.sum()),
        "n_bins_dropped": int(mask.size - mask.sum()),
    }


def _inherited_record(
    samples: np.ndarray, seeds: np.ndarray, reference_power: np.ndarray, sigma_max: float
) -> dict[str, Any]:
    """Return the inherited-band block, with the empty radial bins dropped."""
    full = inherited_band(samples, seeds, reference_power, sigma_max)
    low = inherited_band(
        samples, seeds, reference_power, sigma_max, low_band_sigma_px=LOW_BAND_SIGMA_PX
    )
    curve = _finite_curve(full.centres, full.radial_measured, full.radial_predicted)
    return {
        "inherited_measured": full.share_measured,
        "inherited_predicted": full.share_predicted,
        "inherited_measured_low_band": low.share_measured,
        "inherited_predicted_low_band": low.share_predicted,
        "low_band_sigma_px": LOW_BAND_SIGMA_PX,
        "sigma_max": full.sigma_max,
        "radial": {
            "centres": curve["centres"],
            "measured": curve["values"][0],
            "predicted": curve["values"][1],
            "n_bins": curve["n_bins"],
            "n_bins_dropped": curve["n_bins_dropped"],
        },
    }


def _memorisation_record(
    samples: np.ndarray,
    views: DatasetViews,
    seed_idx: np.ndarray,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Return the memorisation block; the per-sample arrays go to ``.npy`` beside the JSON."""
    rows = views.train_rows(seed_idx)
    result = memorisation_ratio(
        samples,
        views.train,
        views.heldout,
        views.train_subjects,
        rows,
        sigma_lp=SIGMA_LP,
        device=device,
    )
    np.save(out_dir / "final_memorisation_per_sample_d.npy", result.per_sample_d)
    np.save(out_dir / "final_memorisation_per_sample_nn.npy", result.per_sample_nn)
    return {
        "M": result.M,
        "M_lp": result.M_lp,
        "d_samples_median": result.d_samples_median,
        "d_heldout_median": result.d_heldout_median,
        "seed_nn_fraction": result.seed_nn_fraction,
        "n_samples": result.n_samples,
        "n_train": result.n_train,
        "n_heldout": result.n_heldout,
        "per_sample_files": [
            "final_memorisation_per_sample_d.npy",
            "final_memorisation_per_sample_nn.npy",
        ],
        "note": (
            "M is two-sided: M << 1 is copying, M >> 1 is a model whose samples are off the "
            "data manifold. It is only readable as 'copying or not' once LSD and KID say the "
            "run fits."
        ),
    }


def _pca_record(
    train: np.ndarray, seeds: np.ndarray, samples: np.ndarray, device: str, out_dir: Path
) -> dict[str, Any]:
    """Return the PCA block; the components and the training scores go to ``.npy``."""
    result = pca_around_seed(train, seeds, samples, n_components=2, device=device)
    np.save(out_dir / "final_pca_components.npy", result.components)
    np.save(out_dir / "final_pca_mean.npy", result.mean)
    np.save(out_dir / "final_pca_train_scores.npy", result.train_scores)
    return {
        "n_components": int(result.components.shape[0]),
        "explained_variance": [float(value) for value in result.explained_variance],
        "seed_scores": [[float(v) for v in row] for row in result.seed_scores],
        "sample_scores": [
            [[float(v) for v in sample] for sample in seed] for seed in result.sample_scores
        ],
        "component_files": [
            "final_pca_components.npy",
            "final_pca_mean.npy",
            "final_pca_train_scores.npy",
        ],
    }


# --------------------------------------------------------------------------------------------
# The evaluation
# --------------------------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the repository HEAD, or ``"unknown"``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _resolve_device(requested: str | None, config: Any) -> str:
    """Return the torch device string, falling back to the CPU when CUDA is absent."""
    import torch

    wanted = str(requested) if requested else str(config.device)
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; running on the CPU")
        return "cpu"
    return wanted


def _identity(config: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Return the run-identity block of ``summary.json``."""
    return {
        "run_id": str(config.run_id),
        "dataset": str(config.data.dataset),
        "arm": str(config.arm),
        "seed": int(config.seed),
        "config_sha256": str(payload.get("config_sha256", "")),
        "schedule_name": str(config.model.blur_schedule_name),
        "blur_sigma_max": float(config.model.blur_sigma_max),
        "K": int(config.model.K),
    }


def evaluate_run(request: EvalRequest) -> dict[str, Any]:
    """Evaluate one run and write the result files of ``05-metrics.md`` §9.

    Parameters
    ----------
    request : EvalRequest
        The evaluation to perform.

    Returns
    -------
    dict[str, Any]
        The summary that was written to ``<run>/metrics/summary.json``.

    Raises
    ------
    MetricError
        If the run, the dataset or the checkpoints cannot be used, or a metric is undefined on
        the data the request produced.
    """
    import torch

    amp = _check_amp(request.amp)
    workdir = Path(request.run).resolve()
    try:
        config = load_run_config(workdir)
    except SamplingError as error:
        raise MetricError(f"{workdir}: {error}") from error

    dataset_root = Path(config.data.root) / str(config.data.dataset)
    views = load_views(dataset_root)
    lists = ensure_seed_lists(dataset_root)
    table = checkpoint_table(workdir)
    steps, selection = select_checkpoints(table, request.ckpts)
    final_step = max(table)
    device = _resolve_device(request.device, config)
    metrics_dir = workdir / metrics_dirname(amp)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "evaluating %s: steps %s (%s), final %d, device %s, amp %s",
        config.run_id, steps, selection, final_step, device, amp,
    )
    reference = views.reference
    reference_power = mode_power(reference.astype(np.float32) / 255.0)

    per_step: dict[int, dict[str, Any]] = {}
    final_record: dict[str, Any] | None = None
    sampling_log: list[dict[str, Any]] = []
    payload_identity: dict[str, Any] = {}

    for step in steps:
        ckpt_path = table[step]
        payload = load_checkpoint(ckpt_path, device)
        payload_identity = {
            "config_sha256": payload.get("config_sha256", ""),
            "run_id": payload.get("run_id", ""),
        }
        model_holder: dict[str, Any] = {}

        def loader(path=ckpt_path, holder=model_holder, pay=payload):
            if "model" not in holder:
                holder["model"] = load_ema_model(path, config, device, payload=pay)
            return holder["model"]

        lsd_set = draw_set(
            "lsd", workdir, step, ckpt_path, config, dataset_root, "file",
            request.n_lsd, 1, SAMPLING_RNG_INTERMEDIATE, request.sample_batch, device, loader,
            idx_file=lists.intermediate.path, force=request.force, amp=amp,
        )
        sampling_log.append(
            {"step": step, "set": "lsd", "reused": lsd_set.reused, "elapsed_s": lsd_set.elapsed_s}
        )

        record_path = metrics_dir / f"ckpt_{step:06d}.json"
        if record_path.exists() and not request.force and lsd_set.reused:
            per_step[step] = read_json(record_path)
        else:
            record = _lsd_record(lsd_set.flat, reference)
            record.update(
                {
                    "step": int(step),
                    "checkpoint": str(ckpt_path),
                    "checkpoint_sha256": checkpoint_sha256(ckpt_path),
                    "seed_list_sha256": lists.intermediate.sha256,
                    "seed_list": str(lists.intermediate.path),
                    "n_seeds": int(lsd_set.seed_idx.size),
                    "sample_rng_seed": SAMPLING_RNG_INTERMEDIATE,
                    "sample_batch": int(request.sample_batch),
                    "amp": amp,
                    "samples_dir": str(lsd_set.directory),
                }
            )
            write_json(record_path, record)
            per_step[step] = record
        logger.info("step %d: LSD %.4f", step, per_step[step]["lsd"])

        if step == final_step:
            final_record = _evaluate_final(
                request, workdir, step, ckpt_path, config, dataset_root, views, lists,
                reference, reference_power, device, loader, metrics_dir, sampling_log,
                per_step[step],
            )

        model_holder.clear()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary = _summary(
        request, config, payload_identity, views, lists, per_step, final_record, steps,
        selection, final_step, device, sampling_log,
    )
    write_json(metrics_dir / "summary.json", summary)

    if request.gate is not None:
        gate_record = run_gate(
            request, workdir, config, dataset_root, lists, reference, table, device
        )
        write_json(metrics_dir / "gate.json", gate_record)
        summary["gate"] = {
            "steps": list(request.gate),
            "extend": gate_record["extend"],
            "difference": gate_record["difference"]["point"],
            "ci_low": gate_record["difference"]["ci_low"],
            "ci_high": gate_record["difference"]["ci_high"],
        }
        write_json(metrics_dir / "summary.json", summary)

    return summary


def _evaluate_final(
    request: EvalRequest,
    workdir: Path,
    step: int,
    ckpt_path: Path,
    config: Any,
    dataset_root: Path,
    views: DatasetViews,
    lists: SeedLists,
    reference: np.ndarray,
    reference_power: np.ndarray,
    device: str,
    loader: Any,
    metrics_dir: Path,
    sampling_log: list[dict[str, Any]],
    ckpt_record: dict[str, Any],
) -> dict[str, Any]:
    """Draw the two final sets and write ``metrics/final.json``."""
    final_path = metrics_dir / "final.json"

    final_set = draw_set(
        "final", workdir, step, ckpt_path, config, dataset_root, "file",
        request.n_final, 1, SAMPLING_RNG_FINAL, request.sample_batch, device, loader,
        idx_file=lists.final.path, force=request.force, amp=request.amp,
    )
    sampling_log.append(
        {"step": step, "set": "final", "reused": final_set.reused,
         "elapsed_s": final_set.elapsed_s}
    )
    heldout_set = draw_set(
        "heldout", workdir, step, ckpt_path, config, dataset_root, "seed",
        request.n_seeds, request.n_per_seed, SAMPLING_RNG_INTERMEDIATE, request.sample_batch,
        device, loader, force=request.force, amp=request.amp,
    )
    sampling_log.append(
        {"step": step, "set": "heldout", "reused": heldout_set.reused,
         "elapsed_s": heldout_set.elapsed_s}
    )

    if (
        final_path.exists()
        and not request.force
        and final_set.reused
        and heldout_set.reused
    ):
        logger.info("reusing %s", final_path)
        return read_json(final_path)

    record: dict[str, Any] = {
        "step": int(step),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": checkpoint_sha256(ckpt_path),
        "intermediate_lsd": ckpt_record["lsd"],
        "seed_list_sha256": lists.final.sha256,
        "seed_list": str(lists.final.path),
        "sample_batch": int(request.sample_batch),
        "amp": request.amp,
    }
    record.update(_lsd_record(final_set.flat, reference))

    diversity = within_seed_diversity(heldout_set.samples, sigma_lp=SIGMA_LP)
    record.update(
        {
            "diversity_pix": diversity.D_pix_mean,
            "diversity_lp": diversity.D_lp_mean,
            "diversity_per_seed_pix": [float(v) for v in diversity.per_seed_pix],
            "diversity_per_seed_lp": [float(v) for v in diversity.per_seed_lp],
            "n_diversity_seeds": diversity.n_seeds,
            "n_per_seed": diversity.n_per_seed,
        }
    )
    record.update(
        _inherited_record(
            heldout_set.samples,
            heldout_set.seeds,
            reference_power,
            float(config.model.blur_sigma_max),
        )
    )
    record.update(
        _memorisation_record(final_set.flat, views, final_set.seed_idx, device, metrics_dir)
    )
    record["pca"] = _pca_record(
        views.train, heldout_set.seeds, heldout_set.samples, device, metrics_dir
    )

    if request.skip_inception:
        record["inception"] = None
        record["inception_note"] = "skipped (--skip-inception)"
    else:
        record["inception"] = _inception_record(request, views, final_set, device)
        record["inception_note"] = "KID is the headline; FID is reported with its n_reference."

    write_json(final_path, record)
    return record


def _inception_record(
    request: EvalRequest, views: DatasetViews, final_set: SampleSet, device: str
) -> dict[str, Any]:
    """Return the §7 block, or raise a :class:`MetricError` naming the weights path."""
    from ihdm.metrics.inception import (
        bootstrap_inception,
        inception_weights_path,
        reference_features,
    )

    logger.info("inception weights: %s", inception_weights_path())
    ref_features = reference_features(
        views.root,
        views.reference,
        views.dataset_sha256,
        device=device,
        batch=request.fid_batch,
        force=request.force,
    )
    from ihdm.metrics.inception import inception_features

    sample_features = inception_features(
        final_set.flat, device=device, batch=request.fid_batch
    )
    result = bootstrap_inception(
        sample_features,
        ref_features,
        k=request.k,
        n_boot=request.n_boot_inception,
        rng_seed=0,
    )
    return result.to_json()


def run_gate(
    request: EvalRequest,
    workdir: Path,
    config: Any,
    dataset_root: Path,
    lists: SeedLists,
    reference: np.ndarray,
    table: dict[int, Path],
    device: str,
) -> dict[str, Any]:
    """Compute the paired plateau gate between the two checkpoints of ``request.gate``.

    Both sets are drawn if they are not cached, so ``--gate`` works on a run whose intermediate
    sets have not been evaluated yet.

    Parameters
    ----------
    request : EvalRequest
        The evaluation request; ``request.gate`` names the two steps, earlier first.
    workdir : Path
        The run directory.
    config : ml_collections.ConfigDict
        The run's config.
    dataset_root : Path
        The dataset.
    lists : SeedLists
        The frozen seed lists.
    reference : numpy.ndarray
        The ``ref`` split.
    table : dict[int, Path]
        The run's checkpoints.
    device : str
        Torch device string.

    Returns
    -------
    dict[str, Any]
        The ``metrics/gate.json`` record.

    Raises
    ------
    MetricError
        If ``request.gate`` is absent, names a missing checkpoint, or names the same step twice.
    """
    if request.gate is None:
        raise MetricError("run_gate needs request.gate")
    early, late = (int(value) for value in request.gate)
    if early == late:
        raise MetricError(f"the gate needs two different steps, got {early} twice")
    missing = [step for step in (early, late) if step not in table]
    if missing:
        raise MetricError(f"the run has no checkpoint at steps {missing}")

    sets = {}
    for step in (early, late):
        payload = load_checkpoint(table[step], device)
        holder: dict[str, Any] = {}

        def loader(path=table[step], holder=holder, pay=payload):
            if "model" not in holder:
                holder["model"] = load_ema_model(path, config, device, payload=pay)
            return holder["model"]

        sets[step] = draw_set(
            "lsd", workdir, step, table[step], config, dataset_root, "file",
            request.n_lsd, 1, SAMPLING_RNG_INTERMEDIATE, request.sample_batch, device, loader,
            idx_file=lists.intermediate.path, force=False, amp=request.amp,
        )
        holder.clear()

    result = paired_lsd_gate(
        sets[early].flat,
        sets[late].flat,
        reference,
        n_boot=request.n_boot_gate,
        rng_seed=0,
    )
    record = result.to_json()
    record.update(
        {
            "step_a": early,
            "step_b": late,
            "seed_list_sha256": lists.intermediate.sha256,
            "sample_rng_seed": SAMPLING_RNG_INTERMEDIATE,
            "sample_batch": int(request.sample_batch),
            "amp": request.amp,
            "rule": (
                "extend the run when the bootstrap 95% CI of LSD(step_a) - LSD(step_b) over "
                "resampled evaluation seeds lies strictly above zero (D10, D17)"
            ),
        }
    )
    logger.info(
        "gate %d vs %d: LSD %.4f -> %.4f, difference %.4f [%.4f, %.4f], extend=%s",
        early, late, result.lsd_a, result.lsd_b, result.difference.point,
        result.difference.low, result.difference.high, result.extend,
    )
    return record


def _summary(
    request: EvalRequest,
    config: Any,
    payload_identity: dict[str, Any],
    views: DatasetViews,
    lists: SeedLists,
    per_step: dict[int, dict[str, Any]],
    final_record: dict[str, Any] | None,
    steps: list[int],
    selection: str,
    final_step: int,
    device: str,
    sampling_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge the per-checkpoint records, the final record and the run identity."""
    lsd_by_step = {int(step): float(record["lsd"]) for step, record in per_step.items()}
    threshold = request.a0_final_lsd
    if threshold is None:
        tau: int | None = None
        note = "not computed: --a0-final-lsd was not given (T_tau needs the A0 run's final LSD)"
    elif not lsd_by_step:
        tau, note = None, "not computed: no checkpoint was evaluated"
    else:
        tau = t_tau(lsd_by_step, float(threshold))
        note = (
            f"first evaluated step with LSD <= {float(threshold):.6g}"
            if tau is not None
            else f"not reached: no evaluated step has LSD <= {float(threshold):.6g}"
        )

    summary: dict[str, Any] = {
        "run": _identity(config, payload_identity),
        "dataset_sha256": views.dataset_sha256,
        "git_sha": _git_sha(),
        "created": datetime.now(UTC).isoformat(),
        "checkpoint_steps": [int(step) for step in steps],
        "checkpoint_selection": selection,
        "final_step": int(final_step),
        "lsd_by_step": lsd_by_step,
        "variance_ratio_by_step": {
            int(step): float(record["variance_ratio"]) for step, record in per_step.items()
        },
        "t_tau": tau,
        "t_tau_note": note,
        "a0_final_lsd": None if threshold is None else float(threshold),
        "seed_lists": {
            "intermediate_sha256": lists.intermediate.sha256,
            "intermediate_path": str(lists.intermediate.path),
            "intermediate_rule": lists.intermediate.rule,
            "final_sha256": lists.final.sha256,
            "final_path": str(lists.final.path),
            "final_rule": lists.final.rule,
        },
        "sampling": {
            "device": device,
            "batch": int(request.sample_batch),
            "amp": request.amp,
            "rng_seed_intermediate": SAMPLING_RNG_INTERMEDIATE,
            "rng_seed_final": SAMPLING_RNG_FINAL,
            "n_lsd": int(request.n_lsd),
            "n_final": int(request.n_final),
            "n_seeds": int(request.n_seeds),
            "n_per_seed": int(request.n_per_seed),
            "log": sampling_log,
        },
        "env": {
            "python": sys.version.split()[0],
            "n_reference": int(views.reference_idx.size),
            "n_train": int(views.train_idx.size),
            "n_heldout": int(views.heldout_idx.size),
        },
    }
    if final_record is not None:
        summary["final"] = {
            key: value
            for key, value in final_record.items()
            if key
            not in {
                "diversity_per_seed_pix",
                "diversity_per_seed_lp",
                "pca",
                "radial",
            }
        }
        summary["final"]["has_inception"] = final_record.get("inception") is not None
    return summary
