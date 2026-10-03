"""Fidelity from held-out seeds (T7.5): Inception features, metrics, contrasts and the reading.

``python -m ihdm.cli.heldout_fidelity --eval-dir <dir of tars> --data-root <IHDM data root>
--work <scratch dir> --out <dir> [--runs <run_id,...>] [--anchors-only]``

Reads only the members it needs from each evaluation tar (the held-out and the final sample sets
of the last checkpoint, their seed lists and requests, and ``final.json``), computes Inception
features on the CPU, caches them per run and set under ``--work`` (so that a crash does not
repeat finished runs), deletes the extracted sample arrays once their features exist, and writes
``heldout_fidelity.json`` and ``heldout_grid.png`` into ``--out``; the markdown tables for the
hand-written README go to ``<work>/tables.md``. The reproduction anchors of the ticket's step 7
always run first; ``--anchors-only`` stops after them.

Exit codes: 0 everything written; 1 analysis failure, including a failed anchor; 2 an input is
missing (a tar, a dataset folder).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ihdm.analysis.heldout_fidelity import (
    MRI_DATASETS,
    N_BOOT_WITHIN,
    HeldoutFidelityError,
    ReferenceSets,
    anchor_features,
    anchor_metrics,
    contrast,
    expected_heldout_seeds,
    index_sha256,
    read_rule,
    select_references,
    set_metrics,
    sign_summary,
)
from ihdm.analysis.tables import DATASET_LABEL, AnalysisError, design_runs
from ihdm.metrics.errors import MetricError
from ihdm.metrics.inception import (
    DEFAULT_K,
    REFERENCE_FEATURES_NAME,
    inception_features,
    inception_weights_path,
)
from ihdm.metrics.io import write_json
from ihdm.paths import repo_root
from ihdm.stats import StatsError

__all__ = ["EXIT_FAIL", "EXIT_MISSING", "EXIT_OK", "build_parser", "main", "selected_runs"]

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_FAIL, EXIT_MISSING = 0, 1, 2

AMP: str = "fp16"
DEVICE: str = "cpu"
MODE: str = "clean"
ARMS: tuple[str, ...] = ("A0", "A3", "A1")
ANCHOR_RUNS: tuple[str, ...] = ("ixi_A0_s1", "lsun_church_A3_s1")
GRID_RUNS: tuple[str, ...] = ("ixi_A0_s1", "ixi_A3_s1", "lsun_church_A0_s1", "lsun_church_A3_s1")
N_GRID_SEEDS: int = 8
N_GRID_SAMPLES: int = 2
N_SEEDS: int = 40
N_PER_SEED: int = 50
N_FINAL: int = 2000
METRICS: tuple[str, ...] = ("precision", "density", "kid", "recall", "coverage")
READING_ROLE: dict[str, str] = {
    "ixi": "primary endpoint",
    "oasis1": "descriptive (transfer, 2 run seeds)",
    "lsun_church": "descriptive only (the photograph models fail); rule applied mechanically",
    "lsun_bedroom": "descriptive only (the photograph models fail); rule applied mechanically",
}
DATASET_FILES: tuple[str, ...] = ("images.npy", "index.csv", "splits.json", "meta.json")


class MissingInput(HeldoutFidelityError):
    """Raised when a tar or a dataset folder the command needs does not exist."""


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


def selected_runs(runs: str | None) -> list[tuple[str, str, str, int]]:
    """Return ``(run_id, dataset, arm, seed)`` of the 24 runs, or of the subset named.

    Parameters
    ----------
    runs : str | None
        Comma-separated run ids, or ``None`` for all 24.

    Returns
    -------
    list[tuple[str, str, str, int]]
        The runs in report order.

    Raises
    ------
    HeldoutFidelityError
        If a named run is not one of the 24.
    """
    every = [r for r in design_runs() if r[2] in ARMS]
    if not runs:
        return every
    wanted = [r.strip() for r in runs.split(",") if r.strip()]
    known = {r[0] for r in every}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise HeldoutFidelityError(f"not one of the 24 runs: {', '.join(unknown)}")
    return [r for r in every if r[0] in set(wanted)]


def tar_path(eval_dir: Path, rid: str) -> Path:
    """Return the evaluation tar of a run."""
    return Path(eval_dir) / f"{rid}_amp-{AMP}.tar"


def check_inputs(eval_dir: Path, data_root: Path, rids: list[str], datasets: list[str]) -> None:
    """Raise :class:`MissingInput` naming every missing tar or dataset file."""
    missing = [str(tar_path(eval_dir, r)) for r in rids if not tar_path(eval_dir, r).is_file()]
    for dataset in datasets:
        root = Path(data_root) / dataset
        missing += [str(root / n) for n in DATASET_FILES if not (root / n).is_file()]
    if missing:
        raise MissingInput("missing input(s): " + ", ".join(missing))


@dataclass
class Dataset:
    """One dataset's images (memory-mapped), splits and reference sets."""

    name: str
    root: Path
    images: np.ndarray
    index: pd.DataFrame
    splits: dict[str, Any]
    refs: ReferenceSets
    heldout_seeds: np.ndarray


def load_dataset(data_root: Path, name: str) -> Dataset:
    """Read a dataset folder and select its references."""
    root = Path(data_root) / name
    index = pd.read_csv(root / "index.csv")
    splits = json.loads((root / "splits.json").read_text())
    refs = select_references(index, splits, name)
    return Dataset(
        name=name,
        root=root,
        images=np.load(root / "images.npy", mmap_mode="r"),
        index=index,
        splits=splits,
        refs=refs,
        heldout_seeds=expected_heldout_seeds(index, splits, name),
    )


# --------------------------------------------------------------------------------------------
# Feature caches
# --------------------------------------------------------------------------------------------


def _features(images: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the CPU Inception features of a ``uint8`` stack and the seconds they took."""
    start = time.perf_counter()
    features = inception_features(np.ascontiguousarray(images), device=DEVICE, mode=MODE)
    return features, time.perf_counter() - start


def _save_npy(path: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``path`` through a temporary file, so a crash leaves no partial file."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def cached_reference(ds: Dataset, kind: str, work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the features of the full ``ref`` (``kind="ref"``) or of R5 (``kind="r5"``)."""
    idx = ds.refs.ref_idx if kind == "ref" else ds.refs.r5_idx
    if idx is None:
        raise HeldoutFidelityError(f"{ds.name} has no {kind} reference")
    path = work / "features" / f"{ds.name}_{kind}.npy"
    sidecar = path.with_suffix(".json")
    digest = index_sha256(idx)
    if path.is_file() and sidecar.is_file():
        record = json.loads(sidecar.read_text())
        if record.get("index_sha256") == digest:
            return np.load(path), record
        logger.warning("%s does not match the index set; recomputing", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("%s %s: %d reference images", ds.name, kind, idx.size)
    features, seconds = _features(np.asarray(ds.images[idx]))
    _save_npy(path, features)
    record = {"index_sha256": digest, "n": int(idx.size), "seconds": seconds,
              "device": DEVICE, "mode": MODE}
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return features, record


def _sha256_file(path: Path) -> str:
    """Return the sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_step(names: list[str], rid: str) -> int:
    """Return the last checkpoint step whose held-out and final sets are both in the tar."""
    pattern = re.compile(rf"^{re.escape(rid)}/samples_amp-{AMP}/(\d{{6}})/heldout/samples\.npy$")
    steps = sorted(int(m.group(1)) for n in names if (m := pattern.match(n)))
    steps = [s for s in steps
             if f"{rid}/samples_amp-{AMP}/{s:06d}/final/samples.npy" in set(names)]
    if not steps:
        raise HeldoutFidelityError(f"{rid}: the tar holds no step with both sample sets")
    return steps[-1]


def _extract(tar: tarfile.TarFile, member: str, destination: Path) -> None:
    """Stream one tar member to ``destination``."""
    source = tar.extractfile(member)
    if source is None:
        raise HeldoutFidelityError(f"{member} is not a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, open(destination, "wb") as out:
        for chunk in iter(lambda: source.read(1 << 22), b""):
            out.write(chunk)


def _as_stack(array: np.ndarray, n: int, what: str) -> np.ndarray:
    """Return ``array`` as an ``(n, H, W)`` ``uint8`` stack."""
    stack = np.asarray(array)
    stack = stack.reshape(-1, *stack.shape[-2:])
    if stack.shape[0] != n or stack.dtype != np.uint8:
        raise HeldoutFidelityError(f"{what}: expected {n} uint8 images, got {stack.shape} "
                                   f"{stack.dtype}")
    return stack


def materialise_run(eval_dir: Path, rid: str, ds: Dataset, work: Path) -> dict[str, Any]:
    """Extract one run's members, compute and cache the H and F features, return the sidecar.

    The sidecar is written last and marks the run as done; a run with a sidecar and both
    feature files is not read again.
    """
    feature_dir = work / "features"
    h_path, f_path = feature_dir / f"{rid}_H.npy", feature_dir / f"{rid}_F.npy"
    sidecar_path = work / "runs" / f"{rid}.json"
    if h_path.is_file() and f_path.is_file() and sidecar_path.is_file():
        logger.info("%s: features read from the cache", rid)
        return json.loads(sidecar_path.read_text())

    tar_file = tar_path(eval_dir, rid)
    scratch = work / "extract" / rid
    start = time.perf_counter()
    with tarfile.open(tar_file) as tar:
        names = tar.getnames()
        step = _find_step(names, rid)
        base = f"{rid}/samples_amp-{AMP}/{step:06d}"
        members = {
            "h_samples": f"{base}/heldout/samples.npy",
            "h_seed_idx": f"{base}/heldout/seed_idx.npy",
            "h_seeds": f"{base}/heldout/seeds.npy",
            "h_request": f"{base}/heldout/request.json",
            "f_samples": f"{base}/final/samples.npy",
            "f_seed_idx": f"{base}/final/seed_idx.npy",
            "f_request": f"{base}/final/request.json",
            "final_json": f"{rid}/metrics_amp-{AMP}/final.json",
        }
        for key, member in members.items():
            if member not in set(names):
                raise HeldoutFidelityError(f"{tar_file}: member {member} is missing")
            _extract(tar, member, scratch / key)
    extract_s = time.perf_counter() - start

    h_request = json.loads((scratch / "h_request").read_text())
    f_request = json.loads((scratch / "f_request").read_text())
    final = json.loads((scratch / "final_json").read_text())
    if list(h_request["shape"][:2]) != [N_SEEDS, N_PER_SEED]:
        raise HeldoutFidelityError(f"{rid}: held-out shape {h_request['shape']}")
    h_seed_idx = np.load(scratch / "h_seed_idx").astype(np.int64)
    if not np.array_equal(np.sort(h_seed_idx), ds.heldout_seeds):
        raise HeldoutFidelityError(f"{rid}: the held-out seeds are not the expected 40 images")
    seeds = np.load(scratch / "h_seeds")
    seeds_match = bool(seeds.dtype == np.uint8
                       and np.array_equal(seeds.reshape(-1, *seeds.shape[-2:]),
                                          np.asarray(ds.images[h_seed_idx])))
    f_seed_idx = np.load(scratch / "f_seed_idx").astype(np.int64)
    if not np.all(np.isin(f_seed_idx, np.asarray(ds.splits["train"], dtype=np.int64))):
        raise HeldoutFidelityError(f"{rid}: a final-set seed is outside the train split")

    h_sha, f_sha = _sha256_file(scratch / "h_samples"), _sha256_file(scratch / "f_samples")
    held = _as_stack(np.load(scratch / "h_samples"), N_SEEDS * N_PER_SEED, f"{rid} heldout")
    fin = _as_stack(np.load(scratch / "f_samples"), N_FINAL, f"{rid} final")

    grid_dir = work / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    per_seed = held.reshape(N_SEEDS, N_PER_SEED, *held.shape[-2:])
    np.savez_compressed(
        grid_dir / f"{rid}.npz",
        seeds=np.asarray(ds.images[h_seed_idx[:N_GRID_SEEDS]]),
        samples=per_seed[:N_GRID_SEEDS, :N_GRID_SAMPLES],
        seed_idx=h_seed_idx[:N_GRID_SEEDS],
    )

    logger.info("%s: Inception features of H (%d) and F (%d)", rid, held.shape[0], fin.shape[0])
    feature_dir.mkdir(parents=True, exist_ok=True)
    h_features, h_s = _features(held)
    _save_npy(h_path, h_features)
    f_features, f_s = _features(fin)
    _save_npy(f_path, f_features)

    sidecar = {
        "run_id": rid,
        "step": step,
        "tar": str(tar_file),
        "tar_bytes": int(tar_file.stat().st_size),
        "heldout_samples_sha256": h_sha,
        "final_samples_sha256": f_sha,
        "heldout_seed_idx": [int(i) for i in h_seed_idx],
        "heldout_seeds_match_dataset": seeds_match,
        "final_seed_idx_sha256": index_sha256(f_seed_idx),
        "heldout_signature": h_request.get("signature", {}),
        "final_signature": f_request.get("signature", {}),
        "stored_inception": final.get("inception", {}),
        "stored_step": final.get("step"),
        "seconds": {"extract": extract_s, "features_h": h_s, "features_f": f_s},
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    for key in ("h_samples", "h_seeds", "f_samples"):
        (scratch / key).unlink(missing_ok=True)
    return sidecar


# --------------------------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------------------------


def run_anchors(
    eval_dir: Path, datasets: dict[str, Dataset], refs: dict[str, np.ndarray], work: Path
) -> dict[str, Any]:
    """Run the reproduction anchors of ticket step 7 and write ``<work>/anchors.json``."""
    ixi = datasets["ixi"]
    cached = np.load(ixi.root / REFERENCE_FEATURES_NAME)
    record: dict[str, Any] = {"a": anchor_features(refs["ixi"], cached), "b": {}}
    record["a"]["cache"] = str(ixi.root / REFERENCE_FEATURES_NAME)
    for rid in ANCHOR_RUNS:
        dataset = rid.rsplit("_", 2)[0]
        sidecar = materialise_run(eval_dir, rid, datasets[dataset], work)
        f_features = np.load(work / "features" / f"{rid}_F.npy")
        local = set_metrics(f_features, refs[dataset], None)
        record["b"][rid] = anchor_metrics(local, sidecar["stored_inception"])
    record["passed"] = all(v["passed"] for v in record["b"].values())
    record["worst_deviation"] = max(v["worst_deviation"] for v in record["b"].values())
    write_json(work / "anchors.json", record)
    return record


# --------------------------------------------------------------------------------------------
# Per-run metrics, contrasts, readings
# --------------------------------------------------------------------------------------------


def run_metrics(rid: str, ds: Dataset, work: Path, ref: np.ndarray,
                r5: np.ndarray | None) -> dict[str, Any]:
    """Return every metric of one run's H and F sets (ticket step 4)."""
    h = np.load(work / "features" / f"{rid}_H.npy")
    f = np.load(work / "features" / f"{rid}_F.npy")
    clusters = np.repeat(np.arange(N_SEEDS), N_PER_SEED)
    r_minus = ref[ds.refs.r_minus_rows]
    out: dict[str, Any] = {
        "H_rminus": set_metrics(h, r_minus, clusters),
        "F_rminus": set_metrics(f, r_minus, None),
        "F_fullref": set_metrics(f, ref, None),
    }
    if r5 is not None:
        out["H_r5"] = set_metrics(h, r5, clusters)
        out["F_r5_descriptive"] = set_metrics(f, r5, None)
    out["gap_F_minus_H_rminus"] = {
        m: float(out["F_rminus"][m] - out["H_rminus"][m]) for m in METRICS
    }
    return out


def dataset_contrasts(dataset: str, per_run: dict[str, dict[str, Any]],
                      runs: list[tuple[str, str, str, int]]) -> dict[str, Any]:
    """Return the A3/A1 − A0 contrasts, the reading and the R5 rows of one dataset."""
    by_arm: dict[str, dict[int, str]] = {}
    for rid, d, arm, seed in runs:
        if d == dataset:
            by_arm.setdefault(arm, {})[seed] = rid
    out: dict[str, Any] = {"role": READING_ROLE[dataset]}
    if "A0" not in by_arm:
        return out
    for arm in ("A3", "A1"):
        if arm not in by_arm:
            continue
        seeds = sorted(set(by_arm[arm]) & set(by_arm["A0"]))
        if not seeds:
            continue

        def values(a: str, block: str, metric: str, seeds: list[int] = seeds) -> dict[int, float]:
            return {s: float(per_run[by_arm[a][s]][block][metric]) for s in seeds}

        blocks = {"H_rminus": "H", "F_rminus": "F"}
        if dataset in MRI_DATASETS:
            blocks.update({"H_r5": "H_r5", "F_r5_descriptive": "F_r5_descriptive"})
        arm_record: dict[str, Any] = {"seeds": seeds}
        for block, label in blocks.items():
            arm_record[label] = {m: contrast(values(arm, block, m), values("A0", block, m))
                                 for m in METRICS}
        if arm == "A3":
            arm_record["reading"] = read_rule(
                arm_record["F"]["precision"]["deltas"], arm_record["H"]["precision"]["deltas"],
                arm_record["F"]["kid"]["deltas"], arm_record["H"]["kid"]["deltas"],
            )
            if dataset in MRI_DATASETS:
                arm_record["r5_sensitivity_H"] = {
                    m: sign_summary(arm_record["H_r5"][m]["deltas"]) for m in ("precision", "kid")
                }
        out[arm] = arm_record
    return out


# --------------------------------------------------------------------------------------------
# Grid and markdown
# --------------------------------------------------------------------------------------------


def draw_grid(work: Path, out: Path) -> None:
    """Draw the held-out seeds and two samples each for IXI and Churches, A0 and A3, seed 1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for rid in GRID_RUNS:
        path = work / "grid" / f"{rid}.npz"
        if path.is_file():
            panels.append((rid, np.load(path)))
    if not panels:
        logger.warning("no grid excerpt available; heldout_grid.png not drawn")
        return
    rows = 1 + N_GRID_SAMPLES
    dpi, cell, label_w = 100, 1.5, 2.2
    fig, axes = plt.subplots(rows * len(panels), N_GRID_SEEDS,
                             figsize=(label_w + N_GRID_SEEDS * cell,
                                      rows * len(panels) * cell + 0.6),
                             dpi=dpi, squeeze=False)
    for p, (rid, panel) in enumerate(panels):
        stacks = [panel["seeds"]] + [panel["samples"][:, j] for j in range(N_GRID_SAMPLES)]
        names = ["held-out seed"] + [f"sample {j + 1}" for j in range(N_GRID_SAMPLES)]
        for r, (stack, name) in enumerate(zip(stacks, names, strict=True)):
            row = rows * p + r
            for col in range(N_GRID_SEEDS):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(stack[col], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                if row == 0:
                    ax.set_title(f"idx {int(panel['seed_idx'][col])}", fontsize=8)
            axes[row, 0].set_ylabel(f"{rid}\n{name}", rotation=0, ha="right", va="center",
                                    fontsize=9)
    fig.suptitle("Held-out seeds (never seen in training) and two samples each, EMA 60k, "
                 "fp16, rng 2026", fontsize=11)
    fig.subplots_adjust(left=label_w / (label_w + N_GRID_SEEDS * cell), right=0.995, top=0.96,
                        bottom=0.005, wspace=0.03, hspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _ci(record: dict[str, Any], name: str) -> str:
    low, high = record[f"{name}_ci"]
    return f"{record[name]:.3f} [{low:.3f}, {high:.3f}]"


def render_tables(report: dict[str, Any]) -> str:
    """Return the markdown tables the README quotes."""
    lines = ["## Per run", "",
             "| run | set | ref | precision [95% CI] | density [95% CI] | KID | recall "
             "| coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for rid, metrics in report["runs"].items():
        for block, (set_name, ref_name) in {
            "H_rminus": ("H", "R⁻"), "F_rminus": ("F", "R⁻"), "H_r5": ("H", "R5"),
            "F_r5_descriptive": ("F", "R5 (desc.)"), "F_fullref": ("F", "full ref"),
        }.items():
            if block in metrics["metrics"]:
                m = metrics["metrics"][block]
                lines.append(f"| {rid} | {set_name} | {ref_name} | {_ci(m, 'precision')} | "
                             f"{_ci(m, 'density')} | {m['kid']:.4f} | {m['recall']:.3f} | "
                             f"{m['coverage']:.3f} |")
    lines += ["", "## Generalisation gap m_F − m_H (R⁻)", "",
              "| run | precision | density | KID |", "|---|---|---|---|"]
    for rid, metrics in report["runs"].items():
        g = metrics["metrics"]["gap_F_minus_H_rminus"]
        lines.append(f"| {rid} | {g['precision']:+.3f} | {g['density']:+.3f} | "
                     f"{g['kid']:+.4f} |")
    lines += ["", "## Contrasts arm − A0", "",
              "| dataset | arm | set | metric | per-seed Δ | mean [95% CI] | p (p_min) |",
              "|---|---|---|---|---|---|---|"]
    for dataset, record in report["contrasts"].items():
        for arm in ("A3", "A1"):
            if arm not in record:
                continue
            for label in ("H", "F", "H_r5", "F_r5_descriptive"):
                for metric in ("precision", "density", "kid"):
                    if label not in record[arm]:
                        continue
                    c = record[arm][label][metric]
                    deltas = ", ".join(f"{d:+.4f}" for d in c["deltas"])
                    lines.append(
                        f"| {DATASET_LABEL[dataset]} | {arm} | {label} | {metric} | {deltas} | "
                        f"{c['mean']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                        f"{c['p_value']:.3f} ({c['p_min']:.3f}) |")
    lines += ["", "## Readings (A3 vs A0)", ""]
    for dataset, record in report["contrasts"].items():
        if "A3" in record:
            r = record["A3"]["reading"]
            p = r["precision"]
            lines.append(f"- {DATASET_LABEL[dataset]} ({record['role']}): **{r['text']}** "
                         f"[{r['label']}]; precision G_F = {p['G_F']:+.4f}, G_H = "
                         f"{p['G_H']:+.4f}, 0.5·G_F = {p['threshold']:+.4f}, gains_H = "
                         f"{', '.join(f'{g:+.4f}' for g in p['gains_h'])}; "
                         f"kid_comparator_gain_positive = {r['kid_comparator_gain_positive']}")
            if "kid" in r:
                k = r["kid"]
                lines.append(f"  - KID fallback: G_F = {k['G_F']:+.5f}, G_H = {k['G_H']:+.5f}, "
                             f"outcome {k['text']}")
            if "r5_sensitivity_H" in record["A3"]:
                for metric, s in record["A3"]["r5_sensitivity_H"].items():
                    lines.append(f"  - R5 sensitivity, H {metric}: Δ_H = "
                                 f"{', '.join(f'{d:+.4f}' for d in s['deltas'])}, mean "
                                 f"{s['mean']:+.4f}, signs agree {s['signs_agree']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="folder of the <run_id>_amp-fp16.tar evaluation tars")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="IHDM data root holding the <dataset>/ folders")
    parser.add_argument("--work", type=Path, required=True,
                        help="scratch folder for extracted members and feature caches")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--runs", default=None, help="comma-separated subset of the 24 run ids")
    parser.add_argument("--anchors-only", action="store_true",
                        help="run the reproduction anchors (ticket step 7) and stop")
    return parser


def _run(args: argparse.Namespace) -> int:
    """Body of :func:`main`; raises on failure."""
    started = time.perf_counter()
    runs = selected_runs(args.runs)
    names = sorted({r[1] for r in runs} | {"ixi", "lsun_church"})
    check_inputs(args.eval_dir, args.data_root,
                 sorted({r[0] for r in runs} | set(ANCHOR_RUNS)), names)
    work: Path = args.work
    work.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_dataset(args.data_root, name) for name in names}
    refs: dict[str, np.ndarray] = {}
    r5s: dict[str, np.ndarray | None] = {}
    ref_records: dict[str, Any] = {}
    for name, ds in datasets.items():
        refs[name], ref_record = cached_reference(ds, "ref", work)
        r5s[name], r5_record = (cached_reference(ds, "r5", work) if ds.refs.r5_idx is not None
                                else (None, None))
        ref_records[name] = {
            "n_ref": int(ds.refs.ref_idx.size), "ref_idx_sha256": index_sha256(ds.refs.ref_idx),
            "n_r_minus": int(ds.refs.r_minus_idx.size),
            "r_minus_idx_sha256": index_sha256(ds.refs.r_minus_idx),
            "n_r5": None if ds.refs.r5_idx is None else int(ds.refs.r5_idx.size),
            "r5_idx_sha256": None if ds.refs.r5_idx is None else index_sha256(ds.refs.r5_idx),
            "n_seed_subjects": len(ds.refs.seed_subjects),
            "heldout_seed_idx_sha256": index_sha256(ds.heldout_seeds),
            "seconds_ref_features": ref_record.get("seconds"),
            "seconds_r5_features": None if r5_record is None else r5_record.get("seconds"),
        }
        logger.info("%s: ref %d, R⁻ %d, R5 %s", name, ds.refs.ref_idx.size,
                    ds.refs.r_minus_idx.size, ref_records[name]["n_r5"])

    anchors = run_anchors(args.eval_dir, datasets, refs, work)
    a = anchors["a"]
    print(f"anchor (a): max |Δ| {a['max_abs_diff']:.3e}, min cosine {a['min_cosine']:.8f}")
    for rid, record in anchors["b"].items():
        status = "PASS" if record["passed"] else "FAIL"
        kid = record["metrics"]["kid"]
        print(f"anchor (b) {rid}: {status}, worst |Δ| {record['worst_deviation']:.4f}, "
              f"KID {kid['local']:.5f} in {kid['stored_ci']}")
    if not anchors["passed"]:
        raise HeldoutFidelityError("a reproduction anchor failed; see anchors.json")
    if args.anchors_only:
        return EXIT_OK

    per_run: dict[str, dict[str, Any]] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    for rid, dataset, _arm, _seed in runs:
        sidecars[rid] = materialise_run(args.eval_dir, rid, datasets[dataset], work)
        per_run[rid] = run_metrics(rid, datasets[dataset], work, refs[dataset], r5s[dataset])
        logger.info("%s: metrics done", rid)

    contrasts = {d: dataset_contrasts(d, per_run, runs)
                 for d in dict.fromkeys(r[1] for r in runs)}
    report: dict[str, Any] = {
        "ticket": "T7.5",
        "created": datetime.now(UTC).isoformat(),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain", "--", "ihdm"])),
        "extractor": {"device": DEVICE, "mode": MODE, "weights": str(inception_weights_path()),
                      "k": DEFAULT_K},
        "bootstrap_within_run": {"n_boot": N_BOOT_WITHIN, "rng_seed": 0, "alpha": 0.05,
                                 "H_unit": "seed (40 clusters of 50)", "F_unit": "sample"},
        "contrast_statistics": "ihdm.stats.paired_delta (10,000 seed resamples, rng 0) and "
                               "ihdm.stats.permutation_test (exact) with their defaults",
        "amendment": "2026-10-03, by main, before any contrast was computed: when rule 1 fires "
                     "on precision the whole rule, rule 1 included, is applied to KID (gain A0 "
                     "- A3); if the KID comparator also fails the outcome is 'not evaluable on "
                     "precision or KID'. R5 sensitivity is H only (sign agreement of Δ_H); F "
                     "against R5 is descriptive because training seeds include R5 images.",
        "anchors": anchors,
        "references": ref_records,
        "runs": {rid: {"provenance": {k: v for k, v in sidecars[rid].items()
                                      if k != "heldout_seed_idx"},
                       "heldout_seed_idx": sidecars[rid]["heldout_seed_idx"],
                       "metrics": per_run[rid]} for rid in per_run},
        "contrasts": contrasts,
    }
    report["seconds_total_this_invocation"] = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "heldout_fidelity.json", report)
    draw_grid(work, args.out / "heldout_grid.png")
    (work / "tables.md").write_text(render_tables(report))
    for dataset, record in contrasts.items():
        if "A3" in record:
            print(f"{dataset}: {record['A3']['reading']['text']} ({record['role']})")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run the command and return its exit code.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 everything written, 1 analysis failure (including a failed anchor), 2 missing input.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except MissingInput as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_MISSING
    except (AnalysisError, MetricError, StatsError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
