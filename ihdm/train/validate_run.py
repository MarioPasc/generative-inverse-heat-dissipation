"""Validator of a training run directory against ``04-run-artifacts.md`` §3 (with D19).

Used by the loginexa harness (H2-H4 of T3.4, ``slurm/loginexa/check_run.py``) and usable on
any run of the array. Every check returns a list of human-readable problems; an empty list is
a pass. Nothing here writes.

The ``metrics.jsonl`` checks are line by line: strict JSON (no ``NaN``/``Infinity`` tokens), the
exact key set of every ``kind``, value types and ranges, the warm-up learning rate, a pre-clip
``grad_norm`` that varies (array 1 logged the post-clip norm, identically 1.0), ``amp_scale``, and
the cadence of every event kind (H-TRAIN §3). The directory checks cover the manifest, the
config, the EMA checkpoints, ``full_final.pt``, the rolling checkpoint, the grids and ``DONE``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ihdm.train.logging import OCTAVE_BIN_NAMES

__all__ = [
    "EVENT_KEYS",
    "RunExpectation",
    "TRAIN_KEYS",
    "check_metrics",
    "check_run_dir",
    "expectation_from_config",
    "read_metrics_strict",
]

TRAIN_KEYS = frozenset({"step", "kind", "loss", "lr", "it_per_s", "img_per_s", "gpu_mem_peak_gb",
                        "grad_norm", "amp_scale", "wall_s", "loss_per_octave"})
EVENT_KEYS: dict[str, frozenset[str]] = {
    "eval": frozenset({"step", "kind", "loss"}),
    "ckpt": frozenset({"step", "kind", "path"}),
    "grid": frozenset({"step", "kind", "path"}),
    "resume": frozenset({"step", "kind", "from"}),
    "skip": frozenset({"step", "kind", "loss", "n_skipped", "consecutive"}),
    "abort": frozenset({"step", "kind", "reason", "loss", "n_skipped", "consecutive",
                        "resume_saved"}),
    "done": frozenset({"step", "kind", "n_skipped"}),
}
MANIFEST_KEYS = frozenset({"run_id", "dataset_id", "arm", "seed", "git_sha", "git_dirty",
                           "hostname", "gpu", "python", "torch", "cuda", "started",
                           "slurm_job_id", "data", "schedule", "config_sha256", "recipe_sha256",
                           "n_params", "batch_size", "n_iters"})
TOP_LEVEL = frozenset({"manifest.json", "config.json", "metrics.jsonl", "tensorboard",
                       "checkpoints", "checkpoints-meta", "grids"})


@dataclass(frozen=True)
class RunExpectation:
    """What a run directory must contain.

    Parameters
    ----------
    n_iters : int
        The final step.
    log_every, eval_every, ckpt_every, grid_every : int
        The cadences of the last invocation.
    batch_size : int
        ``training.batch_size``.
    lr : float
        ``optim.lr`` (the post-warm-up value).
    warmup : int
        ``optim.warmup``.
    amp : bool
        ``amp_scale`` must be a positive number on every train line (an enabled GradScaler).
    gpu : bool
        ``gpu_mem_peak_gb`` must be positive.
    allow_skips : bool
        ``skip`` events (and ``loss: null`` train lines) are acceptable.
    expect_done : bool
        The run must have finished (``done`` event, ``DONE``, ``full_final.pt``).
    """

    n_iters: int
    log_every: int
    eval_every: int
    ckpt_every: int
    grid_every: int
    batch_size: int
    lr: float
    warmup: int
    amp: bool = True
    gpu: bool = True
    allow_skips: bool = False
    expect_done: bool = True


def expectation_from_config(config: dict[str, Any], **overrides: Any) -> RunExpectation:
    """Build a :class:`RunExpectation` from a run's ``config.json`` (the last invocation's).

    Parameters
    ----------
    config : dict[str, Any]
        The parsed ``config.json``.
    **overrides : Any
        Fields of :class:`RunExpectation` to replace.

    Returns
    -------
    RunExpectation
        The expectation.
    """
    t, o = config["training"], config["optim"]
    values = dict(n_iters=int(t["n_iters"]), log_every=int(t["log_every"]),
                  eval_every=int(t["eval_every"]), ckpt_every=int(t["ckpt_every"]),
                  grid_every=int(t["grid_every"]), batch_size=int(t["batch_size"]),
                  lr=float(o["lr"]), warmup=int(o["warmup"]), amp=bool(o["automatic_mp"]),
                  gpu=str(config.get("device", "")).startswith("cuda"))
    values.update(overrides)
    return RunExpectation(**values)


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-strict JSON token {token}")


def read_metrics_strict(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse ``metrics.jsonl`` refusing ``NaN``/``Infinity`` tokens; return records and problems."""
    records, problems = [], []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            problems.append(f"metrics.jsonl:{number}: empty line")
            continue
        try:
            records.append(json.loads(line, parse_constant=_reject_constant))
        except ValueError as error:
            problems.append(f"metrics.jsonl:{number}: {error}")
    return records, problems


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_train(r: dict[str, Any], exp: RunExpectation) -> list[str]:
    s, out = r["step"], []
    if r["loss"] is None:
        if not exp.allow_skips:
            out.append(f"train {s}: loss is null")
    elif not (_is_num(r["loss"]) and r["loss"] >= 0):
        out.append(f"train {s}: loss {r['loss']!r}")
    want_lr = exp.lr * min(s / exp.warmup, 1.0) if exp.warmup > 0 else exp.lr
    if not (_is_num(r["lr"]) and math.isclose(r["lr"], want_lr, rel_tol=1e-6, abs_tol=1e-12)):
        out.append(f"train {s}: lr {r['lr']!r} != {want_lr!r}")
    if not (_is_num(r["it_per_s"]) and r["it_per_s"] > 0):
        out.append(f"train {s}: it_per_s {r['it_per_s']!r}")
    elif not math.isclose(r["img_per_s"], r["it_per_s"] * exp.batch_size, rel_tol=1e-9):
        out.append(f"train {s}: img_per_s != it_per_s * {exp.batch_size}")
    mem = r["gpu_mem_peak_gb"]
    if not (_is_num(mem) and (mem > 0 if exp.gpu else mem == 0)):
        out.append(f"train {s}: gpu_mem_peak_gb {mem!r}")
    if r["grad_norm"] is not None and not (_is_num(r["grad_norm"]) and r["grad_norm"] > 0):
        out.append(f"train {s}: grad_norm {r['grad_norm']!r}")
    scale = r["amp_scale"]
    if exp.amp and not (_is_num(scale) and scale > 0 and math.log2(scale).is_integer()):
        out.append(f"train {s}: amp_scale {scale!r} is not a positive power of two")
    if not exp.amp and scale is not None:
        out.append(f"train {s}: amp_scale {scale!r} without AMP")
    if not (_is_num(r["wall_s"]) and r["wall_s"] >= 0):
        out.append(f"train {s}: wall_s {r['wall_s']!r}")
    octaves = r["loss_per_octave"]
    if not isinstance(octaves, dict) or list(octaves) != list(OCTAVE_BIN_NAMES):
        out.append(f"train {s}: loss_per_octave keys {list(octaves or {})}")
    elif any(v is not None and not (_is_num(v) and v >= 0) for v in octaves.values()):
        out.append(f"train {s}: loss_per_octave values {octaves}")
    return out


def _check_event(r: dict[str, Any], exp: RunExpectation) -> list[str]:
    kind, s, out = r["kind"], r["step"], []
    if kind == "eval" and not (_is_num(r["loss"]) and r["loss"] > 0):
        out.append(f"eval {s}: loss {r['loss']!r}")
    if kind in ("ckpt", "grid") and not Path(r["path"]).is_file():
        out.append(f"{kind} {s}: {r['path']} does not exist")
    if kind == "skip":
        if not exp.allow_skips:
            out.append(f"skip at step {s}")
        if r["loss"] is not None or r["n_skipped"] < 1 or r["consecutive"] < 1:
            out.append(f"skip {s}: malformed {r}")
    if kind == "abort":
        out.append(f"abort at step {s}: {r.get('reason')}")
    if kind == "done" and not isinstance(r["n_skipped"], int):
        out.append(f"done {s}: n_skipped {r['n_skipped']!r}")
    return out


def _schema(records: list[dict[str, Any]], exp: RunExpectation) -> list[str]:
    problems = []
    for r in records:
        kind = r.get("kind")
        if not isinstance(r.get("step"), int) or isinstance(r.get("step"), bool):
            problems.append(f"record without an integer step: {r}")
            continue
        keys = TRAIN_KEYS if kind == "train" else EVENT_KEYS.get(kind)
        if keys is None:
            problems.append(f"unknown kind {kind!r} at step {r['step']}")
        elif set(r) != keys:
            problems.append(f"{kind} {r['step']}: keys {sorted(set(r) ^ keys)} differ from the "
                            "contract")
        else:
            problems += _check_train(r, exp) if kind == "train" else _check_event(r, exp)
    return problems


def _expected_steps(exp: RunExpectation) -> dict[str, set[int]]:
    n = exp.n_iters
    every = {"train": exp.log_every, "eval": exp.eval_every}
    expected = {k: {s for s in range(0, n + 1) if s % v == 0} for k, v in every.items()}
    for kind, period in (("ckpt", exp.ckpt_every), ("grid", exp.grid_every)):
        expected[kind] = {s for s in range(1, n + 1) if s % period == 0} | {n}
    return expected


def _cadence(records: list[dict[str, Any]], exp: RunExpectation) -> list[str]:
    problems = []
    resumed = any(r.get("kind") == "resume" for r in records)
    for kind, want in _expected_steps(exp).items():
        steps = [r["step"] for r in records if r.get("kind") == kind]
        if set(steps) != want:
            missing, extra = sorted(want - set(steps)), sorted(set(steps) - want)
            problems.append(f"{kind} cadence: missing {missing[:10]} extra {extra[:10]}")
        if len(steps) != len(set(steps)) and not resumed:
            problems.append(f"{kind}: duplicate steps without a resume")
    return problems


def check_metrics(records: list[dict[str, Any]], exp: RunExpectation) -> list[str]:
    """Schema, value and cadence checks of the parsed ``metrics.jsonl`` records.

    Parameters
    ----------
    records : list[dict[str, Any]]
        From :func:`read_metrics_strict`.
    exp : RunExpectation
        What the run must show.

    Returns
    -------
    list[str]
        The problems found (empty on a pass).
    """
    problems = _schema(records, exp) + _cadence(records, exp)
    norms = [r["grad_norm"] for r in records if r.get("kind") == "train"
             and r.get("grad_norm") is not None]
    if len(norms) >= 2 and (len(set(norms)) < 2 or all(v == 1.0 for v in norms)):
        problems.append(f"grad_norm does not vary (post-clip?): {sorted(set(norms))[:5]}")
    done = [r for r in records if r.get("kind") == "done"]
    if exp.expect_done and (not done or done[-1]["step"] != exp.n_iters):
        problems.append(f"no done event at step {exp.n_iters}")
    return problems


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_manifest(workdir: Path, config: dict[str, Any], exp: RunExpectation,
                    data_root: Path | None) -> list[str]:
    manifest = json.loads((workdir / "manifest.json").read_text())
    problems = [f"manifest: missing keys {sorted(MANIFEST_KEYS - set(manifest))}"] if (
        MANIFEST_KEYS - set(manifest)) else []
    sched = manifest.get("schedule", {})
    if sched.get("sha256") != config["model"].get("blur_schedule_sha256"):
        problems.append("manifest schedule sha256 != config model.blur_schedule_sha256")
    if sched.get("values") != config["model"]["blur_schedule"]:
        problems.append("manifest schedule values != config model.blur_schedule")
    file = sched.get("file")
    if file and Path(file).is_file() and _sha256(Path(file)) != sched.get("sha256"):
        problems.append(f"schedule file {file} does not hash to the manifest's sha256")
    registry = Path(file).parent / "schedules.json" if file else None
    if registry is not None and registry.is_file():
        entry = json.loads(registry.read_text()).get(sched.get("name"), {})
        if entry.get("sha256") != sched.get("sha256"):
            problems.append(f"{registry} records {entry.get('sha256')} for {sched.get('name')}, "
                            f"the manifest {sched.get('sha256')}")
    if data_root is not None:
        meta_path = Path(data_root) / config["data"]["dataset"] / "meta.json"
        meta = json.loads(meta_path.read_text())
        if manifest.get("data", {}).get("images_sha256") != meta.get("sha256_images"):
            problems.append(f"manifest data.images_sha256 != {meta_path} sha256_images")
    if manifest.get("batch_size") != exp.batch_size:
        problems.append(f"manifest batch_size {manifest.get('batch_size')} != {exp.batch_size}")
    if not math.isclose(float(config["optim"]["lr"]), exp.lr, rel_tol=1e-12):
        problems.append(f"config optim.lr {config['optim']['lr']} != {exp.lr}")
    if not (isinstance(manifest.get("n_params"), int) and manifest["n_params"] > 0):
        problems.append(f"manifest n_params {manifest.get('n_params')!r}")
    return problems


def _check_checkpoints(workdir: Path, manifest: dict[str, Any], exp: RunExpectation) -> list[str]:
    problems = []
    for step in sorted(_expected_steps(exp)["ckpt"]):
        path = workdir / "checkpoints" / f"ema_iter_{step:06d}.pt"
        if not path.is_file():
            problems.append(f"missing {path.name}")
    last = workdir / "checkpoints" / f"ema_iter_{exp.n_iters:06d}.pt"
    if last.is_file():
        ckpt = torch.load(last, map_location="cpu", weights_only=True)
        if set(ckpt) != {"step", "ema_state_dict", "schedule", "run_id", "config_sha256"}:
            problems.append(f"{last.name}: keys {sorted(ckpt)}")
        elif ckpt["step"] != exp.n_iters or ckpt["config_sha256"] != manifest["config_sha256"]:
            problems.append(f"{last.name}: step/config_sha256 disagree with the manifest")
        elif any(k.startswith("module.") for k in ckpt["ema_state_dict"]):
            problems.append(f"{last.name}: keys carry the DataParallel prefix")
    if not (workdir / "checkpoints-meta" / "checkpoint.pth").is_file():
        problems.append("missing checkpoints-meta/checkpoint.pth")
    final = workdir / "checkpoints" / "full_final.pt"
    if exp.expect_done:
        if not final.is_file():
            problems.append("missing checkpoints/full_final.pt")
        else:
            state = torch.load(final, map_location="cpu", weights_only=True)
            keys_ok = set(state) == {"optimizer", "model", "step", "ema"}
            if not keys_ok or state["step"] != exp.n_iters + 1:
                problems.append(f"full_final.pt: keys {sorted(state)} step {state.get('step')}")
    return problems


def _check_grids(workdir: Path, exp: RunExpectation, image_size: int) -> list[str]:
    from PIL import Image

    problems = []
    seeds = workdir / "grids" / "seeds.npy"
    if not seeds.is_file():
        problems.append("missing grids/seeds.npy")
    else:
        array = np.load(seeds)
        if array.shape != (8, image_size, image_size) or array.dtype != np.uint8:
            problems.append(f"grids/seeds.npy: {array.shape} {array.dtype}")
    for step in sorted(_expected_steps(exp)["grid"]):
        png = workdir / "grids" / f"iter_{step:06d}.png"
        if not png.is_file():
            problems.append(f"missing grids/{png.name}")
            continue
        pixels = np.asarray(Image.open(png))
        if pixels.size == 0 or pixels.min() == pixels.max():
            problems.append(f"grids/{png.name}: empty or constant image {pixels.shape}")
    return problems


def check_run_dir(workdir: Path, exp: RunExpectation | None = None,
                  data_root: Path | None = None) -> list[str]:
    """Check a whole run directory; ``exp`` defaults to the one of its ``config.json``.

    Parameters
    ----------
    workdir : Path
        The run directory.
    exp : RunExpectation | None
        What it must contain.
    data_root : Path | None
        When given, the manifest's data hash is compared with ``<data_root>/<dataset>/meta.json``.

    Returns
    -------
    list[str]
        The problems found (empty on a pass).
    """
    workdir = Path(workdir)
    present = {p.name for p in workdir.iterdir()} if workdir.is_dir() else set()
    need = TOP_LEVEL | ({"DONE"} if exp is None or exp.expect_done else set())
    if need - present:
        return [f"missing top-level entries {sorted(need - present)}"]
    config = json.loads((workdir / "config.json").read_text())
    exp = exp or expectation_from_config(config)
    records, problems = read_metrics_strict(workdir / "metrics.jsonl")
    problems += check_metrics(records, exp)
    problems += _check_manifest(workdir, config, exp, data_root)
    manifest = json.loads((workdir / "manifest.json").read_text())
    problems += _check_checkpoints(workdir, manifest, exp)
    problems += _check_grids(workdir, exp, int(config["data"]["image_size"]))
    if exp.expect_done and (workdir / "DONE").stat().st_size != 0:
        problems.append("DONE is not empty")
    return problems
