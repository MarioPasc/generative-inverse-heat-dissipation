"""The single config factory of the experiment: one arm of one dataset per call.

Frozen contract: ``docs/SPECIFICATIONS/04-run-artifacts.md`` §2. Every value in that table is set
here and nowhere else, so the thirty runs differ only in the two knobs the experiment is about
(``model.blur_sigma_max`` and ``model.blur_schedule``) and in the seed.

Usage with the released trainer::

    python train.py --config configs/spectral/arms.py:ixi,A3 --config.seed=2 --workdir <dir>

The spec string is ``"<dataset_id>,<arm>[,<key>=<value>...]"``; the optional trailing
``key=value`` pairs set any dotted config path (``seed=2``, ``training.batch_size=32``) and exist
so a probe or a test can build a variant without the absl flag machinery.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ml_collections
import numpy as np
import torch

from ihdm.paths import data_root, schedules_dir

__all__ = ["ARMS", "DATASETS", "EXPERIMENT_CELLS", "ArmSpec", "get_config"]

#: The four dataset ids of the experiment (``00-overview.md`` §1).
DATASETS: tuple[str, ...] = ("ixi", "oasis1", "lsun_church", "lsun_bedroom")

#: The photograph datasets; the only ones on which arm A2p is defined.
PHOTOGRAPH_DATASETS: tuple[str, ...] = ("lsun_church", "lsun_bedroom")

#: The five arms (``00-overview.md`` §1); ``A2p`` spells the paper's A2'.
ARMS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A2p")

#: The cells the experiment actually runs: ``(dataset_id, arm, seeds)``. 30 runs in total.
#: ``04-run-artifacts.md`` §2 validates only two arm/dataset rules, so the factory also builds
#: combinations that are legal but not part of the plan (e.g. ``oasis1,A1``); this table is the
#: authoritative list for the submission ticket (T3.3).
EXPERIMENT_CELLS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("ixi", "A0", (1, 2, 3)),
    ("ixi", "A3", (1, 2, 3)),
    ("ixi", "A1", (1, 2)),
    ("ixi", "A2", (1, 2)),
    ("lsun_church", "A0", (1, 2, 3)),
    ("lsun_church", "A3", (1, 2, 3)),
    ("lsun_church", "A1", (1, 2)),
    ("lsun_church", "A2", (1, 2)),
    ("lsun_church", "A2p", (1, 2)),
    ("oasis1", "A0", (1, 2)),
    ("oasis1", "A3", (1, 2)),
    ("lsun_bedroom", "A0", (1, 2)),
    ("lsun_bedroom", "A3", (1, 2)),
)

# arm -> (schedule name, terminal blur). The schedule names are frozen in 04 §2; note that the
# matched schedules of A2/A3 are the IXI-fitted ones on every dataset (D12: fitted once on the
# IXI training split and transferred frozen), while A2p is the churches-fitted control.
_ARM_SCHEDULE: dict[str, str] = {
    "A0": "log_W2",
    "A1": "log_W8",
    "A2": "ixi_W2",
    "A3": "ixi_W8",
    "A2p": "lsun_church_W2",
}
_ARM_SIGMA_MAX: dict[str, float] = {"A0": 96.0, "A1": 24.0, "A2": 96.0, "A3": 24.0, "A2p": 96.0}

#: Arms restricted per dataset; datasets not listed accept every arm the global rules allow.
#: ``04-run-artifacts.md`` §2 names only the Bedrooms rule; the orchestrator confirmed
#: (2026-09-22) that it is symmetric across the transfer pair, so OASIS-1 is restricted too.
_ARMS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "oasis1": ("A0", "A3"),
    "lsun_bedroom": ("A0", "A3"),
}

_K: int = 200
_BLUR_SIGMA_MIN: float = 0.5


@dataclass(frozen=True)
class ArmSpec:
    """One cell of the experiment: a dataset, an arm, a seed and optional overrides.

    Parameters
    ----------
    dataset_id : str
        One of :data:`DATASETS`.
    arm : str
        One of :data:`ARMS`.
    seed : int
        The run seed; 1 by default, overridden by ``--config.seed`` or a ``seed=`` override.
    overrides : tuple[tuple[str, str], ...]
        Dotted config paths and their string values, applied after the config is built.

    Raises
    ------
    ValueError
        If the dataset or the arm is unknown, or the combination is not allowed
        (``04-run-artifacts.md`` §2: A2p only on photographs; Bedrooms only A0 and A3).
    """

    dataset_id: str
    arm: str
    seed: int = 1
    overrides: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.dataset_id not in DATASETS:
            raise ValueError(f"unknown dataset_id {self.dataset_id!r}; expected one of {DATASETS}")
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {ARMS}")
        if self.arm == "A2p" and self.dataset_id not in PHOTOGRAPH_DATASETS:
            raise ValueError(
                f"arm A2p is defined only on the photograph datasets {PHOTOGRAPH_DATASETS}, "
                f"not on {self.dataset_id!r}"
            )
        allowed = _ARMS_BY_DATASET.get(self.dataset_id)
        if allowed is not None and self.arm not in allowed:
            raise ValueError(f"dataset {self.dataset_id!r} accepts only the arms {allowed}")

    @classmethod
    def parse(cls, spec: str) -> ArmSpec:
        """Parse ``"<dataset_id>,<arm>[,<key>=<value>...]"``.

        Parameters
        ----------
        spec : str
            The spec string passed after the colon of ``--config <file>:<spec>``.

        Returns
        -------
        ArmSpec
            The parsed spec.

        Raises
        ------
        ValueError
            If the string has fewer than two comma-separated fields, or a trailing field is
            not a ``key=value`` pair.
        """
        parts = [piece.strip() for piece in spec.split(",") if piece.strip()]
        if len(parts) < 2:
            raise ValueError(
                f"config spec {spec!r} must be '<dataset_id>,<arm>[,<key>=<value>...]'"
            )
        dataset_id, arm = parts[0], parts[1]
        seed = 1
        overrides: list[tuple[str, str]] = []
        for piece in parts[2:]:
            if "=" not in piece:
                raise ValueError(f"override {piece!r} in spec {spec!r} is not '<key>=<value>'")
            key, value = piece.split("=", 1)
            key = key.strip()
            if key == "seed":
                seed = int(value)
            else:
                overrides.append((key, value.strip()))
        return cls(dataset_id=dataset_id, arm=arm, seed=seed, overrides=tuple(overrides))


def _schedule_path(name: str) -> Path:
    """Return the absolute path of ``schedules/<name>.npy``."""
    return (schedules_dir() / f"{name}.npy").resolve()


def _validate_schedule(values: np.ndarray, path: Path, sigma_max: float) -> None:
    """Check a loaded schedule against ``04-run-artifacts.md`` §1.

    A local copy of the checks rather than a call into ``ihdm.spectral.schedules``
    (T1.3's module, not merged yet). It validates the array as loaded, which is the property
    the trainer depends on; T1.3's ``validate_schedule`` additionally checks the hash against
    ``schedules.json``.

    Parameters
    ----------
    values : np.ndarray
        The loaded array.
    path : Path
        Where it came from, for the error messages.
    sigma_max : float
        The terminal blur the arm requires.

    Raises
    ------
    ValueError
        On any violation of the contract.
    """
    if values.dtype != np.float64:
        raise ValueError(f"{path}: dtype must be float64, got {values.dtype}")
    if values.shape != (_K + 1,):
        raise ValueError(f"{path}: shape must be ({_K + 1},), got {values.shape}")
    if values[0] != 0.0:
        raise ValueError(f"{path}: s[0] must be 0.0, got {values[0]!r}")
    if not np.isclose(values[1], _BLUR_SIGMA_MIN):
        raise ValueError(f"{path}: s[1] must be {_BLUR_SIGMA_MIN}, got {values[1]!r}")
    if not np.all(np.diff(values) > 0):
        raise ValueError(f"{path}: the schedule must be strictly increasing")
    if not np.isclose(values[_K], sigma_max):
        raise ValueError(f"{path}: s[K] must be {sigma_max}, got {values[_K]!r}")


def _load_schedule(name: str, sigma_max: float) -> tuple[np.ndarray, Path, str]:
    """Load and validate ``schedules/<name>.npy``.

    Parameters
    ----------
    name : str
        The schedule name (``log_W2``, ``ixi_W8``, ...).
    sigma_max : float
        The terminal blur the arm requires.

    Returns
    -------
    tuple[np.ndarray, Path, str]
        ``(values, path, sha256_of_the_file_bytes)``.

    Raises
    ------
    FileNotFoundError
        If the array has not been produced yet.
    ValueError
        If the array violates ``04-run-artifacts.md`` §1.
    """
    path = _schedule_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"blur schedule {name!r} not found at {path}. The frozen schedule arrays are "
            "produced by ticket T1.3 (docs/SPECIFICATIONS/M1-data/T1.3-profile-and-schedules.md); "
            "set IHDM_SCHEDULES_DIR to point at a directory that holds them."
        )
    raw = path.read_bytes()
    values = np.load(path)
    _validate_schedule(values, path, sigma_max)
    return values, path, hashlib.sha256(raw).hexdigest()


def _coerce(current: Any, text: str) -> Any:
    """Coerce the string ``text`` to the type of the config value it replaces."""
    if isinstance(current, bool):
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    return text


def _apply_override(config: ml_collections.ConfigDict, dotted_key: str, text: str) -> None:
    """Set one dotted config path from its string value, keeping the existing type."""
    node: Any = config
    *parents, leaf = dotted_key.split(".")
    for part in parents:
        node = getattr(node, part)
    setattr(node, leaf, _coerce(getattr(node, leaf), text))


def _build_config(spec: ArmSpec) -> ml_collections.ConfigDict:
    """Build the config of one cell, before overrides are applied."""
    sigma_max = _ARM_SIGMA_MAX[spec.arm]
    schedule_name = _ARM_SCHEDULE[spec.arm]
    schedule, schedule_file, schedule_sha = _load_schedule(schedule_name, sigma_max)

    config = ml_collections.ConfigDict()
    config.dataset_id = spec.dataset_id
    config.arm = spec.arm
    config.run_id = ""  # filled at the end of get_config, refreshed by train.py after --config.seed

    # training
    config.training = training = ml_collections.ConfigDict()
    # D4": MEASURED, not estimated. The Picasso probe (T3.2, job 2405546, A100-SXM4-40GB) peaked
    # at 28.54 GB at batch 16 and OOMed at batch 24 (37.15 GiB allocated, >= 38.0 GiB needed, on a
    # card with 39.52 GiB usable), against D4"'s "raise to 24 only if it peaks <= 30 GB" gate.
    training.batch_size = 16
    training.n_iters = 40000  # D4"": 640k samples at batch 16; the batch-24 27500 is now moot
    training.ckpt_every = 2500
    training.resume_every = 500
    training.log_every = 50
    training.eval_every = 500
    training.grid_every = 2500
    # The same cadences under the released names, which sample.py / evaluate.py still read.
    training.snapshot_freq = training.ckpt_every
    training.snapshot_freq_for_preemption = training.resume_every
    training.log_freq = training.log_every
    training.eval_freq = training.eval_every
    training.sampling_freq = training.grid_every
    # D19 skip policy (ihdm.train.guard): under an enabled GradScaler a non-finite loss is a no-op
    # step, logged as "skip"; abort after 10 consecutive or more than 100 skipped steps.
    training.max_consecutive_skips = 10
    training.max_skips = 100

    # sampling (D3: the paper's prior is N(u_K, delta^2 I))
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.prior_noise = True
    sampling.delta_factor = 1.25

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = training.batch_size
    evaluate.enable_sampling = False
    evaluate.num_samples = 2000
    evaluate.enable_loss = True
    evaluate.calculate_fids = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = spec.dataset_id
    data.root = str(data_root())
    data.split_train = "train"
    data.split_eval = "ref"
    data.image_size = 192
    data.num_channels = 1
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_workers = 4

    # model
    config.model = model = ml_collections.ConfigDict()
    model.K = _K
    model.sigma = 0.01
    model.blur_sigma_min = _BLUR_SIGMA_MIN
    model.blur_sigma_max = sigma_max
    model.blur_schedule_name = schedule_name
    model.blur_schedule_file = str(schedule_file)
    model.blur_schedule_sha256 = schedule_sha
    model.blur_schedule = schedule
    model.train_level_max_inclusive = True  # D3: level K is used at sampling time
    model.model_channels = 128
    model.channel_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attention_levels = (2, 3)
    model.dropout = 0.1
    model.num_heads = 1
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.conv_resample = True
    model.use_fp16 = False
    model.use_scale_shift_norm = False
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.skip_rescale = True
    model.conditional = True
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.ema_rate = 0.999

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    # D19 (recipe v2): D4"'s pre-registered fallback. Array 2408239 ran 2e-4 and 7 of its 11
    # started runs hit a non-finite loss 367-888 steps after the warm-up ended; the released
    # configs use 2e-5 (LSUN Churches 128, FFHQ 256) and 1e-4 (AFHQ 256) at these resolutions.
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.weight_decay = 0.0
    optim.warmup = 1000
    optim.grad_clip = 1.0
    optim.automatic_mp = True

    config.seed = spec.seed
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return config


def get_config(spec: str) -> ml_collections.ConfigDict:
    """Build the config of one ``(dataset, arm)`` cell.

    Parameters
    ----------
    spec : str
        ``"<dataset_id>,<arm>[,<key>=<value>...]"``, e.g. ``"ixi,A3"`` or
        ``"lsun_church,A2p,seed=2"``.

    Returns
    -------
    ml_collections.ConfigDict
        The released config structure (``training``, ``sampling``, ``eval``, ``data``,
        ``model``, ``optim``, ``seed``, ``device``) plus ``dataset_id``, ``arm`` and ``run_id``.

    Raises
    ------
    ValueError
        If the spec is malformed, or the dataset/arm combination is not allowed.
    FileNotFoundError
        If the arm's schedule array has not been produced yet (ticket T1.3).
    """
    arm_spec = ArmSpec.parse(spec)
    config = _build_config(arm_spec)
    for key, value in arm_spec.overrides:
        _apply_override(config, key, value)
    # eval.batch_size follows training.batch_size unless it was overridden explicitly.
    if not any(key == "eval.batch_size" for key, _ in arm_spec.overrides):
        config.eval.batch_size = config.training.batch_size
    config.run_id = f"{config.data.dataset}_{config.arm}_s{int(config.seed)}"
    return config
