"""Diagnose a non-finite training loss from the weights that produced it (T3.4, decision D19).

Input: a run directory saved by the trainer's non-finite guard (the array-1 fixtures of
``docs/RESULTS/nan_diagnosis.md``): ``checkpoints-meta/checkpoint.pth`` holds the live weights
that produced the non-finite loss (the ``GradScaler`` turned that step into a no-op before the
guard saved), ``config.json`` the resolved config, ``metrics.jsonl`` the ``abort`` line.

The training loss is recomputed exactly as ``scripts/losses.py: get_inverse_heat_loss_fn`` does
(blur to level k and k-1, add N(0, sigma^2) noise, predict the difference, sum of squares per
sample), with explicit levels, noise and dropout seeds, under two precisions (fp16 autocast as
in training, and fp32) and two network modes (train: dropout on, as in training; eval: off):

* **parameters**: norms, maxima, finiteness; distance of the live weights from their EMA;
* **survey**: ``--n-batches`` random training batches with training-distributed levels: the
  fraction of batches (and samples) with a non-finite loss per condition;
* **level sweep**: ``--sweep-batches`` batches at every level 1..K;
* **replay**: the exact images of the failing step (the train loader's index sequence is
  replayed from ``config.seed``), at every level, ``--draws`` noise/dropout draws each;
* **hooks**: for one failing input, the first module (in execution order) whose fp16 output is
  non-finite, and every module's max |output| in fp16 and in fp32 against the fp16 limit 65504,
  plus the max |attention logit| of each attention block (computed in fp32 from its input).

The classification (``fp16_forward_overflow`` / ``true_divergence`` / ``other``) follows the
rules of :func:`classify`. Writes one JSON report and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

#: Largest finite float16.
FP16_MAX = 65504.0


class DiagnoseError(Exception):
    """Raised when a fixture cannot be read or is inconsistent."""


@dataclass(frozen=True)
class Condition:
    """One way of evaluating the loss: precision and network mode."""

    amp: bool
    train_mode: bool

    @property
    def name(self) -> str:
        return f"{'fp16' if self.amp else 'fp32'}_{'train' if self.train_mode else 'eval'}"


CONDITIONS: tuple[Condition, ...] = (
    Condition(True, True), Condition(False, True), Condition(True, False), Condition(False, False)
)


@dataclass
class Fixture:
    """Everything loaded from one guard-saved run directory."""

    path: Path
    config: Any
    model: torch.nn.Module
    live_state: dict[str, torch.Tensor]
    ema_params: list[torch.Tensor]
    optimizer_state: dict[str, Any]
    step: int
    abort: dict[str, Any] | None
    last_train: list[dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------


def _strip(key: str) -> str:
    return key[len("module."):] if key.startswith("module.") else key


def _read_metrics(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return the ``abort`` record (or ``None``) and the last five ``train`` records."""
    if not path.is_file():
        return None, []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    aborts = [r for r in records if r.get("kind") == "abort"]
    trains = [r for r in records if r.get("kind") == "train"]
    return (aborts[-1] if aborts else None), trains[-5:]


def build_config(saved: dict[str, Any], device: str, data_root: str | None = None) -> Any:
    """Rebuild the run's ConfigDict from its ``config.json`` alone.

    The saved file is the resolved config the run trained with, so the network, the schedule
    array and the optimiser values are exactly those of the run, whatever the repository's
    defaults are now (recipe v2 changed ``optim.lr``). JSON turned tuples and arrays into
    lists; the keys the network and the blur read as tuples or arrays are converted back.

    Parameters
    ----------
    saved : dict[str, Any]
        The parsed ``config.json``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    data_root : str | None
        Replaces ``data.root`` (the run's data root may not exist where the diagnosis runs).

    Returns
    -------
    ml_collections.ConfigDict
        The config.
    """
    import copy

    import ml_collections

    tree = copy.deepcopy(saved)
    for key in ("channel_mult", "attention_levels"):
        if key in tree["model"]:
            tree["model"][key] = tuple(tree["model"][key])
    tree["model"]["blur_schedule"] = np.asarray(tree["model"]["blur_schedule"], dtype=np.float64)
    if data_root:
        tree["data"]["root"] = data_root
    tree["device"] = torch.device(device)
    return ml_collections.ConfigDict(tree)


def load_fixture(path: Path, device: str, data_root: str | None = None) -> Fixture:
    """Load a guard-saved run directory.

    Parameters
    ----------
    path : Path
        The run directory (``config.json``, ``checkpoints-meta/checkpoint.pth``,
        ``metrics.jsonl``).
    device : str
        Where to put the network.
    data_root : str | None
        Overrides the saved ``data.root``.

    Returns
    -------
    Fixture
        The loaded fixture, with the live weights in ``model``.

    Raises
    ------
    DiagnoseError
        If a required file is missing.
    """
    from model_code.unet import UNetModel

    path = Path(path)
    ckpt_path = path / "checkpoints-meta" / "checkpoint.pth"
    for required in (path / "config.json", ckpt_path):
        if not required.is_file():
            raise DiagnoseError(f"missing {required}")
    config = build_config(json.loads((path / "config.json").read_text()), device, data_root)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    live = {_strip(k): v for k, v in state["model"].items()}
    model = UNetModel(config).to(device)
    model.load_state_dict(live)
    abort, last_train = _read_metrics(path / "metrics.jsonl")
    return Fixture(path, config, model, live, list(state["ema"]["shadow_params"]),
                   state["optimizer"], int(state["step"]), abort, last_train)


@contextmanager
def ema_weights(model: torch.nn.Module, ema_params: list[torch.Tensor]) -> Iterator[None]:
    """Temporarily load the EMA shadow parameters (the order of ``model.parameters()``)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(ema_params):
        raise DiagnoseError(f"EMA holds {len(ema_params)} tensors, the model {len(params)}")
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, e in zip(params, ema_params, strict=True):
            p.copy_(e.to(p.device))
    try:
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, backup, strict=True):
                p.copy_(b)


# ------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------


def param_report(fixture: Fixture, top: int = 10) -> dict[str, Any]:
    """Norms, maxima and finiteness of the live weights, their distance to the EMA, Adam's v.

    Parameters
    ----------
    fixture : Fixture
        The loaded fixture.
    top : int
        How many tensors to list in each ranking.

    Returns
    -------
    dict[str, Any]
        The report section.
    """
    named = [(n, p.detach().float().cpu()) for n, p in fixture.model.named_parameters()]
    rows = []
    for (name, live), ema in zip(named, fixture.ema_params, strict=True):
        ema = ema.float().cpu()
        rows.append({
            "name": name,
            "numel": live.numel(),
            "norm": float(live.norm()),
            "max_abs": float(live.abs().max()),
            "non_finite": int((~torch.isfinite(live)).sum()),
            "rel_dist_to_ema": float((live - ema).norm() / (ema.norm() + 1e-12)),
        })
    exp_avg_sq = [s["exp_avg_sq"].float() for s in fixture.optimizer_state["state"].values()
                  if "exp_avg_sq" in s]
    total = math.sqrt(sum(r["norm"] ** 2 for r in rows))
    return {
        "n_tensors": len(rows),
        "n_params": sum(r["numel"] for r in rows),
        "total_norm": total,
        "non_finite_total": sum(r["non_finite"] for r in rows),
        "max_abs_overall": max(r["max_abs"] for r in rows),
        "optimizer_lr": [g.get("lr") for g in fixture.optimizer_state["param_groups"]],
        "adam_exp_avg_sq_max": max(float(v.max()) for v in exp_avg_sq) if exp_avg_sq else None,
        "top_max_abs": sorted(rows, key=lambda r: -r["max_abs"])[:top],
        "top_rel_dist_to_ema": sorted(rows, key=lambda r: -r["rel_dist_to_ema"])[:top],
    }


# ------------------------------------------------------------------------------------------
# Loss evaluation
# ------------------------------------------------------------------------------------------


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled)


@torch.no_grad()
def per_sample_loss(model, heat, x, levels, noise, condition: Condition, seed: int):
    """The training loss of ``scripts/losses.py`` per sample, for explicit levels and noise.

    Parameters
    ----------
    model : torch.nn.Module
        The U-Net (bare, not ``DataParallel``).
    heat : torch.nn.Module
        The run's ``DCTBlur`` forward process.
    x : torch.Tensor
        Images ``(B, 1, H, W)`` in [0, 1].
    levels : torch.Tensor
        Integer levels ``(B,)`` in ``1..K``.
    noise : torch.Tensor
        The additive noise ``(B, 1, H, W)`` (already multiplied by sigma).
    condition : Condition
        Precision and network mode.
    seed : int
        Seeds torch before the forward, so the dropout masks of a train-mode forward are
        reproducible.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Per-sample losses ``(B,)`` in fp32 and the per-sample max |network output| ``(B,)``.
    """
    model.train(condition.train_mode)
    torch.manual_seed(seed)
    with _autocast(x.device, condition.amp):
        blurred = heat(x, levels).float()
        less_blurred = heat(x, levels - 1).float()
        perturbed = blurred + noise
        diff = model(perturbed, levels)
        prediction = perturbed + diff
        losses = ((less_blurred - prediction) ** 2).reshape(x.shape[0], -1).sum(dim=-1)
    out_max = diff.detach().float().abs().reshape(x.shape[0], -1).amax(dim=-1)
    return losses.float(), out_max


def _batch(dataset: Dataset, indices, device) -> torch.Tensor:
    return torch.stack([dataset[int(i)][0] for i in indices]).to(device).float()


def _levels_noise(k_max: int, batch: int, shape, sigma: float, seed: int, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    levels = torch.randint(1, k_max + 1, (batch,), generator=g)
    noise = torch.randn((batch, *shape), generator=g) * sigma
    return levels.to(device), noise.to(device)


def survey(fixture: Fixture, heat, dataset, n_batches: int, seed: int) -> dict[str, Any]:
    """Loss finiteness on random training batches, levels drawn as in training.

    Every condition sees the same images, levels, noise and dropout seed, for the live and
    for the EMA weights.
    """
    config, model = fixture.config, fixture.model
    rng = np.random.default_rng(seed)
    b, k_max = int(config.training.batch_size), int(config.model.K)
    shape = (1, int(config.data.image_size), int(config.data.image_size))
    out: dict[str, Any] = {}
    for weights in ("live", "ema"):
        ctx = ema_weights(model, fixture.ema_params) if weights == "ema" else nullcontext()
        stats = {c.name: {"batches_non_finite": 0, "samples_non_finite": 0, "losses": [],
                          "failures": []} for c in CONDITIONS}
        with ctx:
            for i in range(n_batches):
                idx = rng.choice(len(dataset), size=b, replace=False)
                x = _batch(dataset, idx, config.device)
                levels, noise = _levels_noise(k_max, b, shape, config.model.sigma, seed + i,
                                              config.device)
                for cond in CONDITIONS:
                    loss, _ = per_sample_loss(model, heat, x, levels, noise, cond, seed + i)
                    bad = ~torch.isfinite(loss)
                    s = stats[cond.name]
                    s["batches_non_finite"] += int(bad.any())
                    s["samples_non_finite"] += int(bad.sum())
                    if not bool(bad.all()):
                        s["losses"].append(float(loss[~bad].mean()))
                    for j in torch.nonzero(bad).flatten().tolist():
                        s["failures"].append({"batch": i, "image": int(idx[j]),
                                              "level": int(levels[j]),
                                              "value": repr(float(loss[j]))})
        for s in stats.values():
            finite = [v for v in s.pop("losses") if math.isfinite(v)]
            s["batch_loss_median"] = float(np.median(finite)) if finite else None
            s["batch_loss_max"] = float(np.max(finite)) if finite else None
            s["failures"] = s["failures"][:50]
        out[weights] = {"n_batches": n_batches, "n_samples": n_batches * b, **stats}
    return out


def level_sweep(fixture: Fixture, heat, x, draws: int, seed: int, condition: Condition,
                levels: list[int] | None = None) -> dict[int, dict[str, Any]]:
    """Evaluate every sample of ``x`` at every level (all samples at the same level at once).

    Returns ``{level: {"non_finite": [sample indices...], "draws_non_finite": [draw...],
    "loss_max": float|None, "out_max": float}}`` over ``draws`` noise/dropout draws per level.
    Draw ``d`` of ``level`` uses the seed :func:`draw_seed` ``(seed, level, d)``.
    """
    config = fixture.config
    k_max, sigma = int(config.model.K), float(config.model.sigma)
    result: dict[int, dict[str, Any]] = {}
    for level in levels or list(range(1, k_max + 1)):
        bad: set[int] = set()
        bad_draws: list[int] = []
        loss_max, out_max = -math.inf, 0.0
        for d in range(draws):
            noise = draw_noise(x, sigma, draw_seed(seed, level, d))
            lv = torch.full((x.shape[0],), level, dtype=torch.long, device=x.device)
            loss, omax = per_sample_loss(fixture.model, heat, x, lv, noise, condition,
                                         draw_seed(seed, level, d))
            finite = torch.isfinite(loss)
            if not bool(finite.all()):
                bad.update(torch.nonzero(~finite).flatten().tolist())
                bad_draws.append(d)
            if finite.any():
                loss_max = max(loss_max, float(loss[finite].max()))
            finite_out = torch.isfinite(omax)
            out_max = max(out_max, float(omax[finite_out].max()) if finite_out.any() else math.inf)
        result[level] = {"non_finite": sorted(bad), "draws_non_finite": bad_draws,
                         "loss_max": loss_max if math.isfinite(loss_max) else None,
                         "out_max": out_max}
    return result


def draw_seed(seed: int, level: int, draw: int) -> int:
    """The seed of draw ``draw`` at ``level`` in :func:`level_sweep` (noise and dropout)."""
    return seed + 1000 * level + draw


def draw_noise(x: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """N(0, sigma^2) noise shaped like ``x``, from a CPU generator seeded with ``seed``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(x.shape, generator=g) * sigma).to(x.device)


# ------------------------------------------------------------------------------------------
# Replay of the failing step
# ------------------------------------------------------------------------------------------


class _Positions(Dataset):
    """A dataset of its own positions: replays a loader's index sequence without the images."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def replay_indices(n_train: int, batch_size: int, seed: int, step: int) -> list[int]:
    """Return the dataset positions of the train batch the trainer drew at loop ``step``.

    ``ihdm.data.dataset.make_loaders`` builds a shuffling, ``drop_last`` train loader over a
    ``torch.Generator`` seeded with ``config.seed``, and ``train.py`` takes one batch per step,
    re-creating the iterator at each epoch end. A loader with the same length, batch size and
    generator yields the same index sequence (each new iterator consumes the generator
    identically, independently of ``num_workers``).

    Parameters
    ----------
    n_train : int
        Length of the train split.
    batch_size : int
        ``training.batch_size``.
    seed : int
        ``config.seed``.
    step : int
        The loop step (0-based: step 0 draws the first batch).

    Returns
    -------
    list[int]
        ``batch_size`` positions in the train split.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(_Positions(n_train), batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0, generator=generator)
    iterator = iter(loader)
    batch = None
    for _ in range(step + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
    return [int(i) for i in batch]


# ------------------------------------------------------------------------------------------
# Hooks
# ------------------------------------------------------------------------------------------


class ActivationRecorder:
    """Forward hooks on every submodule: max |output| and non-finiteness, in execution order.

    Attention blocks also record the max |logit| of ``q.k`` computed in fp32 from the block's
    input, which is what their fp16 ``einsum`` must hold below 65504.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.records: list[dict[str, Any]] = []
        self._handles = []
        for name, module in model.named_modules():
            if name:
                self._handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(module, inputs, output):
            tensors = [t for t in (output if isinstance(output, (tuple, list)) else (output,))
                       if torch.is_tensor(t) and t.is_floating_point()]
            if not tensors:
                return
            with torch.autocast(tensors[0].device.type, enabled=False):
                t = tensors[0].detach().float()
                finite = torch.isfinite(t)
                record = {"module": name, "type": type(module).__name__,
                          "dtype": str(tensors[0].dtype).replace("torch.", ""),
                          "non_finite": int((~finite).sum()),
                          "max_abs": float(t[finite].abs().max()) if finite.any() else None}
                if type(module).__name__ in ("QKVAttention", "QKVAttentionLegacy"):
                    record["logit_max_abs"] = _attention_logit_max(module, inputs[0])
            self.records.append(record)
        return hook

    def first_non_finite(self) -> dict[str, Any] | None:
        return next((r for r in self.records if r["non_finite"]), None)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def _attention_logit_max(module, qkv: torch.Tensor) -> float | None:
    """Max |q.k * scale^2| of an attention module, in fp32 from its (possibly fp16) input."""
    qkv = qkv.detach().float()
    bs, width, length = qkv.shape
    ch = width // (3 * module.n_heads)
    scale = 1 / math.sqrt(math.sqrt(ch))
    if type(module).__name__ == "QKVAttention":
        q, k, _ = qkv.chunk(3, dim=1)
        q = q.reshape(bs * module.n_heads, ch, length)
        k = k.reshape(bs * module.n_heads, ch, length)
    else:
        q, k, _ = qkv.reshape(bs * module.n_heads, ch * 3, length).split(ch, dim=1)
    logits = torch.einsum("bct,bcs->bts", q * scale, k * scale)
    finite = torch.isfinite(logits)
    return float(logits[finite].abs().max()) if finite.any() else None


def hook_profile(fixture: Fixture, heat, x, levels, noise, train_mode: bool,
                 seed: int) -> dict[str, Any]:
    """Run one input in fp16 and fp32 with hooks; compare the per-module maxima."""
    profiles = {}
    for amp in (True, False):
        recorder = ActivationRecorder(fixture.model)
        try:
            loss, _ = per_sample_loss(fixture.model, heat, x, levels, noise,
                                      Condition(amp, train_mode), seed)
        finally:
            recorder.close()
        profiles["fp16" if amp else "fp32"] = (recorder, loss)
    rec16, loss16 = profiles["fp16"]
    rec32, loss32 = profiles["fp32"]
    by_name32 = {r["module"]: r for r in rec32.records}
    rows = []
    for r in rec16.records:
        r32 = by_name32.get(r["module"], {})
        rows.append({**r, "fp32_max_abs": r32.get("max_abs"),
                     "fp32_logit_max_abs": r32.get("logit_max_abs")})
    over = [r for r in rows if (r["fp32_max_abs"] or 0) >= FP16_MAX
            or (r.get("fp32_logit_max_abs") or 0) >= FP16_MAX]
    return {
        "train_mode": train_mode,
        "loss_fp16": [repr(float(v)) for v in loss16],
        "loss_fp32": [float(v) for v in loss32],
        "first_non_finite_fp16": rec16.first_non_finite(),
        "first_non_finite_fp32": rec32.first_non_finite(),
        "modules_at_or_over_fp16_max_in_fp32": over[:20],
        "top_fp32_max_abs": sorted(rows, key=lambda r: -(r["fp32_max_abs"] or 0))[:15],
        "attention": [r for r in rows if "logit_max_abs" in r],
        "n_modules": len(rows),
    }


# ------------------------------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------------------------------


def classify(report: dict[str, Any]) -> tuple[str, list[str]]:
    """Name the failure class from the measurements, with the reasons.

    Rules, in order:

    1. ``true_divergence`` if the weights hold non-finite values, if any fp32 loss (either
       mode) is non-finite, or if the fp32 train-mode median batch loss of the survey exceeds
       10x the median of the last logged train losses (the weights no longer fit the data).
    2. ``fp16_forward_overflow`` if some fp16 loss is non-finite (survey, sweep or replay)
       while every fp32 loss on the same inputs is finite, and the hook profile shows the first
       non-finite fp16 module at an fp32 activation (or attention logit) >= 65504, or none
       non-finite in fp32.
    3. ``other`` otherwise (including "not reproduced": no non-finite fp16 loss found).

    Parameters
    ----------
    report : dict[str, Any]
        The report assembled by :func:`run`.

    Returns
    -------
    tuple[str, list[str]]
        The label and the reasons.
    """
    reasons: list[str] = []
    params = report["parameters"]
    live = report["survey"]["live"]
    if params["non_finite_total"]:
        return "true_divergence", [f"{params['non_finite_total']} non-finite weights"]
    fp32_bad = sum(live[c]["samples_non_finite"] for c in ("fp32_train", "fp32_eval"))
    fp32_bad += report.get("replay", {}).get("fp32_non_finite", 0)
    if fp32_bad:
        return "true_divergence", [f"{fp32_bad} non-finite fp32 losses"]
    logged = [r["loss"] for r in report["fixture"]["last_train"] if r.get("loss") is not None]
    median32 = live["fp32_train"]["batch_loss_median"]
    if logged and median32 is not None and median32 > 10 * float(np.median(logged)):
        return "true_divergence", [f"fp32 median loss {median32:.3g} > 10x logged "
                                   f"{float(np.median(logged)):.3g}"]
    fp16_bad = (live["fp16_train"]["samples_non_finite"] + live["fp16_eval"]["samples_non_finite"]
                + report.get("sweep", {}).get("fp16_non_finite", 0)
                + report.get("replay", {}).get("fp16_non_finite", 0))
    if not fp16_bad:
        return "other", ["no non-finite fp16 loss reproduced at the saved weights"]
    reasons.append(f"{fp16_bad} non-finite fp16 losses, "
                   "0 non-finite fp32 losses on the same inputs")
    hooks = report.get("hooks") or {}
    first = hooks.get("first_non_finite_fp16")
    if first is not None:
        over = {r["module"] for r in hooks.get("modules_at_or_over_fp16_max_in_fp32", [])}
        reasons.append(f"first non-finite fp16 output: {first['module']} ({first['type']})"
                       + ("; its fp32 activation reaches the fp16 limit" if over else ""))
    reasons.append(f"fp32 median batch loss {median32:.4g} vs logged {logged}")
    return "fp16_forward_overflow", reasons


# ------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------


def _sweep_summary(sweep: dict[int, dict[str, Any]]) -> dict[str, Any]:
    bad = {lv: r["non_finite"] for lv, r in sweep.items() if r["non_finite"]}
    return {"levels_with_non_finite": sorted(bad),
            "fp16_non_finite": sum(len(v) for v in bad.values()),
            "non_finite_by_level": bad,
            "draws_non_finite_by_level": {lv: sweep[lv]["draws_non_finite"] for lv in bad},
            "out_max_by_level": {lv: r["out_max"] for lv, r in sweep.items()},
            "loss_max_by_level": {lv: r["loss_max"] for lv, r in sweep.items()}}


def _parse_levels(text: str, k_max: int) -> list[int]:
    """``"all"`` -> 1..K; otherwise a comma list of levels and ``a-b`` ranges."""
    if text == "all":
        return list(range(1, k_max + 1))
    levels: set[int] = set()
    for part in filter(None, (p.strip() for p in text.split(","))):
        low, _, high = part.partition("-")
        levels.update(range(int(low), int(high or low) + 1))
    return sorted(lv for lv in levels if 1 <= lv <= k_max)


def _fp32_check(fixture, heat, x, sweep16, draws, seed, train_mode) -> int:
    """Re-run the failing (level) cells of an fp16 sweep in fp32; count non-finite losses."""
    levels = [lv for lv, r in sweep16.items() if r["non_finite"]]
    if not levels:
        return 0
    sweep32 = level_sweep(fixture, heat, x, draws, seed, Condition(False, train_mode), levels)
    return sum(len(r["non_finite"]) for r in sweep32.values())


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run every measurement and return the report."""
    from ihdm.data.dataset import NpyImageDataset
    from model_code import utils as mutils

    t0 = time.perf_counter()
    fixture = load_fixture(Path(args.fixture), args.device, args.data_root)
    if args.weights == "ema":
        _load_ema_permanently(fixture)
    config = fixture.config
    heat = mutils.create_forward_process_from_sigmas(config, config.model.blur_schedule,
                                                     config.device)
    dataset = NpyImageDataset(Path(config.data.root) / config.data.dataset,
                              config.data.split_train)
    report: dict[str, Any] = {"fixture": {
        "path": str(fixture.path), "run": f"{config.data.dataset}_{config.arm}_s{config.seed}",
        "checkpoint_step": fixture.step, "abort": fixture.abort,
        "last_train": [{k: r.get(k) for k in ("step", "loss", "lr", "grad_norm")}
                       for r in fixture.last_train],
        "lr": float(config.optim.lr), "device": str(config.device), "weights": args.weights,
        "gpu": torch.cuda.get_device_name(0) if config.device.type == "cuda" else None,
        "torch": torch.__version__}}
    report["parameters"] = param_report(fixture)
    logger.info("parameters done (%.0f s)", time.perf_counter() - t0)
    modules = tuple(m for m in args.headroom_modules.split(",") if m)
    if args.headroom_only:
        report["headroom"] = headroom(fixture, heat, dataset, args.n_batches, args.seed, modules)
        report["classification"], report["reasons"] = None, ["--headroom-only"]
        report["seconds"] = time.perf_counter() - t0
        return report
    report["survey"] = survey(fixture, heat, dataset, args.n_batches, args.seed)
    logger.info("survey done (%.0f s)", time.perf_counter() - t0)

    levels = _parse_levels(args.levels, int(config.model.K))
    sweep_x = _batch(dataset, np.random.default_rng(args.seed + 1).choice(
        len(dataset), size=args.sweep_images, replace=False), config.device)
    sweep16 = level_sweep(fixture, heat, sweep_x, 1, args.seed, Condition(True, True), levels)
    report["sweep"] = {"n_images": args.sweep_images, **_sweep_summary(sweep16)}
    report["sweep"]["fp32_non_finite"] = _fp32_check(fixture, heat, sweep_x, sweep16, 1,
                                                     args.seed, True)
    logger.info("sweep done (%.0f s)", time.perf_counter() - t0)

    step = args.replay_step if args.replay_step is not None else (
        int(fixture.abort["step"]) if fixture.abort else None)
    replay_x = None
    if step is not None:
        positions = replay_indices(len(dataset), int(config.training.batch_size),
                                   int(config.seed), step)
        replay_x = _batch(dataset, positions, config.device)
        replay16 = level_sweep(fixture, heat, replay_x, args.draws, args.seed,
                               Condition(True, True), levels)
        report["replay"] = {"step": step, "positions": positions, "draws": args.draws,
                            **_sweep_summary(replay16)}
        report["replay"]["fp32_non_finite"] = _fp32_check(fixture, heat, replay_x, replay16,
                                                          args.draws, args.seed, True)
        logger.info("replay done (%.0f s)", time.perf_counter() - t0)

    report["hooks"] = _hooks_on_worst(fixture, heat, report, sweep_x, replay_x, args.seed)
    report["headroom"] = headroom(fixture, heat, dataset, args.n_batches, args.seed, modules)
    report["classification"], report["reasons"] = classify(report)
    report["seconds"] = time.perf_counter() - t0
    return report


def _load_ema_permanently(fixture: Fixture) -> None:
    """Replace the live weights of the fixture's model by its EMA (``--weights ema``)."""
    params = [p for p in fixture.model.parameters() if p.requires_grad]
    with torch.no_grad():
        for p, e in zip(params, fixture.ema_params, strict=True):
            p.copy_(e.to(p.device))


def headroom(fixture: Fixture, heat, dataset, n_batches: int, seed: int,
             modules: tuple[str, ...]) -> dict[str, Any]:
    """Largest fp32 activation of every module over the survey's samples, against fp16's limit.

    The batches, levels, noise and dropout seeds are exactly those of :func:`survey` (same
    ``seed``), so two weight sets of the same dataset are compared on the same samples. The
    forward runs in fp32 in train mode (dropout on, as in training): the numbers are the true
    magnitudes an fp16 forward would have to store.

    Parameters
    ----------
    fixture : Fixture
        The loaded fixture (live or EMA weights already in the model).
    heat : torch.nn.Module
        The run's forward blur process.
    dataset : torch.utils.data.Dataset
        The train split.
    n_batches : int
        Batches of ``training.batch_size``.
    seed : int
        The survey seed.
    modules : tuple[str, ...]
        Modules reported by name (the ones that overflowed in array 1).

    Returns
    -------
    dict[str, Any]
        ``{"n_samples", "mode", "modules": {name: {"max_abs", "ratio"}}, "module", "max_abs",
        "ratio_to_fp16_max", "per_batch_max": {"median", "p90", "max"}, "top": [...]}``.
    """
    config, model = fixture.config, fixture.model
    rng = np.random.default_rng(seed)
    b, k_max = int(config.training.batch_size), int(config.model.K)
    shape = (1, int(config.data.image_size), int(config.data.image_size))
    best: dict[str, float] = {}
    per_batch: list[float] = []
    leaf_names: set[str] | None = None
    for i in range(n_batches):
        idx = rng.choice(len(dataset), size=b, replace=False)
        x = _batch(dataset, idx, config.device)
        levels, noise = _levels_noise(k_max, b, shape, config.model.sigma, seed + i, config.device)
        recorder = ActivationRecorder(model)
        try:
            per_sample_loss(model, heat, x, levels, noise, Condition(False, True), seed + i)
        finally:
            recorder.close()
        if leaf_names is None:
            names = {r["module"] for r in recorder.records}
            leaf_names = {n for n in names if not any(m.startswith(n + ".") for m in names)}
        batch_max = 0.0
        for r in recorder.records:
            if r["max_abs"] is None or r["module"] not in leaf_names:
                continue
            best[r["module"]] = max(best.get(r["module"], 0.0), r["max_abs"])
            batch_max = max(batch_max, r["max_abs"])
        per_batch.append(batch_max)
    top = sorted(best.items(), key=lambda kv: -kv[1])[:10]
    module, value = top[0] if top else ("", 0.0)
    return {
        "n_samples": n_batches * b,
        "mode": "fp32, train mode (dropout as in training), the survey's samples",
        "modules": {m: {"max_abs": best.get(m), "ratio": (best.get(m) or 0.0) / FP16_MAX}
                    for m in modules},
        "module": module, "max_abs": value, "ratio_to_fp16_max": value / FP16_MAX,
        "per_batch_max": {"median": float(np.median(per_batch)) if per_batch else None,
                          "p90": float(np.quantile(per_batch, 0.9)) if per_batch else None,
                          "max": float(np.max(per_batch)) if per_batch else None},
        "top": [{"module": m, "max_abs": v, "ratio": v / FP16_MAX} for m, v in top],
    }


def _hooks_on_worst(fixture, heat, report, sweep_x, replay_x, seed) -> dict[str, Any]:
    """Pick one failing input (replay first, then sweep) or the level of largest output."""
    config = fixture.config
    draw = 0
    for key, x in (("replay", replay_x), ("sweep", sweep_x)):
        section = report.get(key)
        if section and section["levels_with_non_finite"] and x is not None:
            level = section["levels_with_non_finite"][0]
            draw = section["draws_non_finite_by_level"][level][0]
            source = f"{key} level {level} draw {draw}"
            break
    else:
        section = report["replay"] if replay_x is not None else report["sweep"]
        level = max(section["out_max_by_level"], key=lambda lv: section["out_max_by_level"][lv])
        x = replay_x if replay_x is not None else sweep_x
        source = f"no failure; level of largest fp16 output ({level})"
    seed_ = draw_seed(seed, level, draw)
    noise = draw_noise(x, float(config.model.sigma), seed_)
    levels = torch.full((x.shape[0],), level, dtype=torch.long, device=x.device)
    # Train mode reproduces the training forward (dropout on); its fp16/fp32 pair may not share
    # dropout masks (the fused kernels map random numbers to elements per dtype), so the same
    # input is also profiled in eval mode, where the pair is exactly matched.
    profile = hook_profile(fixture, heat, x, levels, noise, True, seed_)
    profile["source"] = source
    profile["eval_mode"] = hook_profile(fixture, heat, x, levels, noise, False, seed_)
    return profile


def _headroom_line(h: dict[str, Any]) -> str:
    named = ", ".join(f"{m} {v['ratio']:.3f}" for m, v in h["modules"].items())
    return (f"headroom ({h['n_samples']} samples, fp32 train mode): network max {h['module']} "
            f"{h['max_abs']:.1f} = {h['ratio_to_fp16_max']:.3f} of 65504; {named}; per-batch max "
            f"median {h['per_batch_max']['median']:.1f}")


def _summary_lines(report: dict[str, Any]) -> list[str]:
    fx, p = report["fixture"], report["parameters"]
    lines = [f"run {fx['run']} ({fx['weights']} weights) checkpoint step {fx['checkpoint_step']} "
             f"abort {fx['abort']}",
             f"params: total norm {p['total_norm']:.4g}, max |w| {p['max_abs_overall']:.4g}, "
             f"non-finite {p['non_finite_total']}, lr in optimizer {p['optimizer_lr']}"]
    if "survey" not in report:
        return [*lines, _headroom_line(report["headroom"])]
    sv = report["survey"]["live"]
    for cond in CONDITIONS:
        s = sv[cond.name]
        lines.append(f"survey live {cond.name}: {s['batches_non_finite']}/{sv['n_batches']} "
                     f"batches, {s['samples_non_finite']}/{sv['n_samples']} samples non-finite; "
                     f"median batch loss {s['batch_loss_median']}")
    ema = report["survey"]["ema"]
    lines.append(f"survey ema fp16_train: {ema['fp16_train']['batches_non_finite']}/"
                 f"{ema['n_batches']} batches non-finite")
    for key in ("sweep", "replay"):
        if key in report:
            s = report[key]
            lines.append(f"{key}: fp16 non-finite samples {s['fp16_non_finite']} at levels "
                         f"{s['levels_with_non_finite'][:20]}; "
                         f"fp32 non-finite {s['fp32_non_finite']}")
    h = report["hooks"]
    over = [r["module"] for r in h["modules_at_or_over_fp16_max_in_fp32"]][:5]
    lines.append(f"hooks ({h['source']}): first non-finite fp16 module "
                 f"{(h['first_non_finite_fp16'] or {}).get('module')}; "
                 f"modules >= 65504 in fp32: {over}")
    lines.append(_headroom_line(report["headroom"]))
    lines.append(f"CLASSIFICATION {report['classification']}: " + " | ".join(report["reasons"]))
    return lines


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", required=True, help="guard-saved run directory")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--n-batches", type=int, default=64)
    parser.add_argument("--sweep-images", type=int, default=16,
                        help="images of the random batch swept over every level")
    parser.add_argument("--draws", type=int, default=2,
                        help="noise/dropout draws per level in the replay sweep")
    parser.add_argument("--levels", default="all",
                        help='levels of the sweeps: "all" (1..K) or e.g. "1-10,50,200"')
    parser.add_argument("--headroom-modules",
                        default="output_blocks.9.2.conv,output_blocks.4.2.conv",
                        help="modules whose fp32 max |activation| the headroom reports by name")
    parser.add_argument("--headroom-only", action="store_true",
                        help="only the parameters and the fp32 activation headroom")
    parser.add_argument("--weights", choices=("live", "ema"), default="live",
                        help="diagnose the saved live weights or their EMA")
    parser.add_argument("--replay-step", type=int, default=None,
                        help="loop step to replay (default: the abort step in metrics.jsonl)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None, help="overrides the saved data.root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=1))
    for line in _summary_lines(report):
        print(line)
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
