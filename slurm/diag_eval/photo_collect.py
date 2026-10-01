"""Collect the photograph-failure diagnostic of T7.3: per-task JSON, merged reading, sample grid.

``task`` runs at the end of every array task of ``photo_eval.sbatch``: it reads the
``metrics_amp-<mode>/`` tree of the shadow run (``ckpt_<step>.json`` with the per-step
``inception`` and ``memorisation`` blocks of ``evaluate_run --inception-steps``, ``final.json``,
``summary.json``) and writes ``<run_id>_photo.json``. ``merge`` runs locally on the three task
JSONs, checks the protocol, computes the late-window means and applies the pre-registered reading
of ``docs/SPECIFICATIONS/M7-diagnostics/README.md`` literally. ``grid`` draws the 16 first rows of
each run's 60k LSD set (training seeds and their samples) side by side in one PNG.

The reading (fixed 2026-10-01, before any diagnostic run existed): a factor *lifts the failure*
if its late-window precision is >= 0.10 **and** >= 10 x the baseline's, **and** its late-window
KID is <= 0.5 x the baseline's. The late-window value of a quantity is its mean over the
checkpoints 45k, 50k, 55k and 60k.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ihdm.metrics.run_eval import metrics_dirname, samples_dirname

#: The four checkpoints of the late window.
LATE_STEPS: tuple[int, ...] = (45000, 50000, 55000, 60000)

#: The pre-registered protocol every merged task must satisfy.
PROTOCOL: dict[str, Any] = {
    "n_samples": 500, "sample_batch": 32, "sample_rng_seed": 2026, "amp": "fp16",
}

#: The pre-registered thresholds of the reading.
PRECISION_FLOOR: float = 0.10
PRECISION_FACTOR: float = 10.0
KID_FACTOR: float = 0.5

#: The 2 x 2 reading table of the README, keyed by (r128 lifts, n32k lifts).
READINGS: dict[tuple[bool, bool], str] = {
    (True, False): "resolution and framing",
    (False, True): "data size",
    (True, True): "either change suffices",
    (False, False): (
        "the budget or recipe (lr, length, model width), which the diagnostic does not test"
    ),
}

#: The roles of the three runs and their run ids.
ROLES: dict[str, str] = {
    "baseline": "lsun_church_A0_s1",
    "r128": "lsun_church_r128_A0_s1",
    "n32k": "lsun_church_n32k_A0_s1",
}

#: The baseline's stored LSD at the late checkpoints (``~/execs/ihdm/eval/
#: lsun_church_A0_s1_amp-fp16_summary.json``, production evaluation, same protocol).
BASELINE_LSD_ANCHORS: dict[int, float] = {
    45000: 1.42166, 50000: 1.36797, 55000: 1.36529, 60000: 1.44537,
}
ANCHOR_TOLERANCE: float = 0.003

_INCEPTION_KEYS: tuple[str, ...] = (
    "kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low", "fid_ci_high", "precision",
    "recall", "density", "coverage", "k", "n_samples", "n_reference", "n_boot",
)
_MEMORISATION_KEYS: tuple[str, ...] = (
    "M", "M_lp", "seed_nn_fraction", "d_samples_median", "d_heldout_median", "n_samples",
)
_FINAL_KEYS: tuple[str, ...] = (
    "diversity_pix", "diversity_lp", "n_diversity_seeds", "n_per_seed", "inherited_measured",
    "inherited_predicted", "sigma_max", "final_set", "step",
)

#: Quantities averaged over the late window: (name, where in the step row).
LATE_QUANTITIES: tuple[tuple[str, str, str], ...] = (
    ("precision", "inception", "precision"),
    ("recall", "inception", "recall"),
    ("kid", "inception", "kid"),
    ("fid", "inception", "fid"),
    ("density", "inception", "density"),
    ("coverage", "inception", "coverage"),
    ("variance_ratio", "", "variance_ratio"),
    ("lsd", "", "lsd"),
    ("M", "memorisation", "M"),
    ("seed_nn_fraction", "memorisation", "seed_nn_fraction"),
)


class CollectError(Exception):
    """A result file of the diagnostic is missing, inconsistent or off-protocol."""


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CollectError(f"{path} is missing")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------
# task
# --------------------------------------------------------------------------------------------


def step_row(ckpt: dict[str, Any]) -> dict[str, Any]:
    """Return the flat record of one checkpoint from its ``ckpt_<step>.json``.

    Parameters
    ----------
    ckpt : dict[str, Any]
        The record written by ``evaluate_run --inception-steps``.

    Returns
    -------
    dict[str, Any]
        LSD block, Inception and memorisation blocks, provenance.

    Raises
    ------
    CollectError
        If the record has no ``inception`` or ``memorisation`` block, or carries a ``delta``
        (the diagnostic samples at the default noise level).
    """
    step = int(ckpt.get("step", -1))
    if not isinstance(ckpt.get("inception"), dict) or not isinstance(
        ckpt.get("memorisation"), dict
    ):
        raise CollectError(f"step {step}: no per-step inception/memorisation block")
    if "delta" in ckpt:
        raise CollectError(f"step {step}: sampled at delta={ckpt['delta']!r}, not the default")
    return {
        "step": step,
        "lsd": ckpt["lsd"],
        "variance_ratio": ckpt["variance_ratio"],
        "lsd_octaves": ckpt["lsd_octaves"],
        "n_samples": ckpt["n_samples"],
        "inception": {key: ckpt["inception"].get(key) for key in _INCEPTION_KEYS},
        "memorisation": {key: ckpt["memorisation"].get(key) for key in _MEMORISATION_KEYS},
        "checkpoint_sha256": ckpt["checkpoint_sha256"],
        "seed_list_sha256": ckpt["seed_list_sha256"],
        "sample_rng_seed": ckpt["sample_rng_seed"],
        "sample_batch": ckpt["sample_batch"],
        "amp": ckpt["amp"],
    }


def collect_task(
    shadow: Path, run_id: str, amp: str, steps: list[int], meta: dict[str, Any]
) -> dict[str, Any]:
    """Return the task record of one run: one row per late checkpoint plus provenance.

    Parameters
    ----------
    shadow : Path
        The (shadow) run directory.
    run_id : str
        The run's id; must equal ``summary.json["run"]["run_id"]``.
    amp : str
        The sampling precision of the evaluation.
    steps : list[int]
        The checkpoints to collect.
    meta : dict[str, Any]
        Job provenance (job id, node, git SHA).

    Returns
    -------
    dict[str, Any]
        The task record.

    Raises
    ------
    CollectError
        If a file is missing or names another run.
    """
    metrics = Path(shadow) / metrics_dirname(amp)
    summary = _read(metrics / "summary.json")
    if summary["run"]["run_id"] != run_id:
        raise CollectError(f"{metrics}: summary is of {summary['run']['run_id']}, not {run_id}")
    rows = [step_row(_read(metrics / f"ckpt_{int(step):06d}.json")) for step in steps]
    final_path = metrics / "final.json"
    final = _read(final_path) if final_path.is_file() else {}
    return {
        "run_id": run_id,
        "amp": amp,
        "steps": [int(step) for step in steps],
        "rows": rows,
        "final": {key: final.get(key) for key in _FINAL_KEYS} if final else None,
        "identity": summary["run"],
        "dataset_sha256": summary["dataset_sha256"],
        "eval_git_sha": summary["git_sha"],
        "seed_lists": summary["seed_lists"],
        "sampling": summary["sampling"],
        "env": summary["env"],
        "meta": meta,
        "created": datetime.now(UTC).isoformat(),
    }


# --------------------------------------------------------------------------------------------
# merge: late window and the reading
# --------------------------------------------------------------------------------------------


def check_protocol(task: dict[str, Any], steps: tuple[int, ...] = LATE_STEPS) -> list[str]:
    """Return the protocol violations of a task record (empty when it follows the protocol)."""
    problems = []
    got_steps = tuple(int(row["step"]) for row in task["rows"])
    if got_steps != tuple(steps):
        problems.append(f"steps {got_steps} != {tuple(steps)}")
    for row in task["rows"]:
        for key, want in PROTOCOL.items():
            if row.get(key) != want:
                problems.append(f"step {row['step']}: {key}={row.get(key)!r} != {want!r}")
        if row["inception"].get("n_samples") != PROTOCOL["n_samples"]:
            problems.append(f"step {row['step']}: inception n_samples "
                            f"{row['inception'].get('n_samples')}")
        if row["seed_list_sha256"] != task["seed_lists"]["intermediate_sha256"]:
            problems.append(f"step {row['step']}: not the frozen 500-seed list")
    return problems


def late_window(task: dict[str, Any]) -> dict[str, float]:
    """Return the mean of every :data:`LATE_QUANTITIES` entry over the task's rows."""
    out: dict[str, float] = {}
    for name, block, key in LATE_QUANTITIES:
        values = [float((row[block] if block else row)[key]) for row in task["rows"]]
        out[name] = float(np.mean(values))
    out["n_checkpoints"] = len(task["rows"])
    return out


def lifts(factor: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    """Apply the pre-registered rule to one factor's late-window means.

    Parameters
    ----------
    factor, baseline : dict[str, float]
        Late-window means (:func:`late_window`) of the factor's run and of the baseline.

    Returns
    -------
    dict[str, Any]
        The three conditions with their numbers and ``lifts`` (all three hold).
    """
    precision, kid = float(factor["precision"]), float(factor["kid"])
    p_base, kid_base = float(baseline["precision"]), float(baseline["kid"])
    conditions = {
        "precision_ge_0.10": precision >= PRECISION_FLOOR,
        "precision_ge_10x_baseline": precision >= PRECISION_FACTOR * p_base,
        "kid_le_0.5x_baseline": kid <= KID_FACTOR * kid_base,
    }
    return {
        "precision": precision,
        "kid": kid,
        "baseline_precision": p_base,
        "baseline_kid": kid_base,
        "precision_threshold": max(PRECISION_FLOOR, PRECISION_FACTOR * p_base),
        "kid_threshold": KID_FACTOR * kid_base,
        "precision_ratio": precision / p_base if p_base > 0 else None,
        "kid_ratio": kid / kid_base if kid_base > 0 else None,
        "conditions": conditions,
        "lifts": all(conditions.values()),
    }


def reading(r128_lifts: bool, n32k_lifts: bool) -> str:
    """Return the README's reading of one cell of the 2 x 2 table."""
    return READINGS[(bool(r128_lifts), bool(n32k_lifts))]


def anchor_check(baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare the baseline's LSD at the late steps with the production evaluation."""
    out: dict[str, Any] = {"tolerance": ANCHOR_TOLERANCE, "steps": {}}
    passed = True
    for row in baseline["rows"]:
        step = int(row["step"])
        want = BASELINE_LSD_ANCHORS.get(step)
        if want is None:
            out["steps"][step] = {"status": "no anchor"}
            passed = False
            continue
        diff = float(row["lsd"]) - want
        ok = abs(diff) <= ANCHOR_TOLERANCE
        passed &= ok
        out["steps"][step] = {"stored": want, "reproduced": float(row["lsd"]),
                              "difference": diff, "ok": ok}
    out["passed"] = passed
    return out


def merge(task_files: list[Path], extra: dict[str, Any], allow_off_protocol: bool = False
          ) -> dict[str, Any]:
    """Merge the three task records and apply the reading.

    Raises
    ------
    CollectError
        If a role is missing or duplicated, or a task is off-protocol (unless allowed).
    """
    tasks = [json.loads(Path(path).read_text()) for path in task_files]
    by_role: dict[str, dict[str, Any]] = {}
    for role, run_id in ROLES.items():
        matches = [task for task in tasks if task["run_id"] == run_id]
        if len(matches) != 1:
            raise CollectError(f"{len(matches)} task records for {run_id} ({role})")
        by_role[role] = matches[0]
    violations = {role: check_protocol(task) for role, task in by_role.items()}
    if any(violations.values()) and not allow_off_protocol:
        raise CollectError(f"off-protocol tasks: {violations}")
    late = {role: late_window(task) for role, task in by_role.items()}
    rule = {role: lifts(late[role], late["baseline"]) for role in ("r128", "n32k")}
    return {
        "ticket": "T7.3",
        "description": (
            "Churches A0 seed 1: baseline, r128 (128², whole scene) and n32k (32,000 training "
            "images), each evaluated at 45k/50k/55k/60k (EMA) on the frozen 500 training seeds "
            "(rng 2026, batch 32, fp16, default delta) against its own 800-image ref split; "
            "late-window value = mean over the four checkpoints; reading pre-registered in "
            "docs/SPECIFICATIONS/M7-diagnostics/README.md."
        ),
        "rule": {
            "precision_floor": PRECISION_FLOOR,
            "precision_factor": PRECISION_FACTOR,
            "kid_factor": KID_FACTOR,
            "text": (
                "a factor lifts the failure if its late-window precision is >= 0.10 and >= 10x "
                "the baseline's, and its late-window KID is <= 0.5x the baseline's"
            ),
        },
        "protocol_violations": violations,
        "anchor": anchor_check(by_role["baseline"]),
        "late_window": late,
        "lifts": rule,
        "reading": {
            "r128_lifts": rule["r128"]["lifts"],
            "n32k_lifts": rule["n32k"]["lifts"],
            "text": reading(rule["r128"]["lifts"], rule["n32k"]["lifts"]),
        },
        "tasks": by_role,
        **extra,
    }


# --------------------------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------------------------


def load_grid_rows(shadow: Path, amp: str, step: int, n: int) -> dict[str, np.ndarray]:
    """Return the first ``n`` seeds, samples and seed indices of a run's LSD set at ``step``."""
    directory = Path(shadow) / samples_dirname(amp) / f"{int(step):06d}" / "lsd"
    for name in ("samples.npy", "seeds.npy", "seed_idx.npy"):
        if not (directory / name).is_file():
            raise CollectError(f"{directory / name} is missing")
    samples = np.load(directory / "samples.npy")
    return {
        "samples": np.asarray(samples[:n, 0]),
        "seeds": np.asarray(np.load(directory / "seeds.npy")[:n]),
        "seed_idx": np.asarray(np.load(directory / "seed_idx.npy")[:n], dtype=np.int64),
    }


def draw_grid(panels: list[tuple[str, dict[str, np.ndarray]]], out: Path, cell_px: int = 192
              ) -> None:
    """Draw one PNG: per run a row of seeds and a row of samples, every cell ``cell_px`` wide.

    Every image is shown in a cell of the same size with nearest-neighbour interpolation, so a
    128² run is upscaled by 1.5 for display (its row label says so).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(panel["samples"].shape[0] for _, panel in panels)
    dpi = 100
    cell = cell_px / dpi
    label_w = 2.4
    fig, axes = plt.subplots(
        2 * len(panels), n, figsize=(label_w + n * cell, 2 * len(panels) * cell + 0.6),
        dpi=dpi, squeeze=False,
    )
    for p, (label, panel) in enumerate(panels):
        side = panel["samples"].shape[-1]
        for kind_i, kind in enumerate(("seeds", "samples")):
            row = 2 * p + kind_i
            for col in range(n):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                if col < panel[kind].shape[0]:
                    ax.imshow(panel[kind][col], cmap="gray", vmin=0, vmax=255,
                              interpolation="nearest")
                else:
                    ax.axis("off")
            scale = "" if side == cell_px else f"\n{side}² shown ×{cell_px / side:g} (nearest)"
            axes[row, 0].set_ylabel(f"{label}\n{kind} ({side}²){scale}", rotation=0,
                                    ha="right", va="center", fontsize=9)
    fig.suptitle("Churches A0 s1, EMA 60k: the first 16 seeds of the frozen 500 list and their "
                 "samples (fp16, rng 2026, batch 32)", fontsize=11)
    fig.subplots_adjust(left=label_w / (label_w + n * cell), right=0.995, top=0.95,
                        bottom=0.005, wspace=0.03, hspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the ``task``, ``merge`` or ``grid`` subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    task = sub.add_parser("task")
    task.add_argument("--shadow", type=Path, required=True)
    task.add_argument("--run-id", required=True)
    task.add_argument("--amp", default="fp16")
    task.add_argument("--steps", default=":".join(str(s) for s in LATE_STEPS),
                      help="colon-separated checkpoints")
    task.add_argument("--meta", default="{}", help="JSON object of provenance")
    task.add_argument("--out", type=Path, required=True)
    merged = sub.add_parser("merge")
    merged.add_argument("task_files", type=Path, nargs="+")
    merged.add_argument("--extra", default="{}", help="JSON object merged at the top level")
    merged.add_argument("--allow-off-protocol", action="store_true", help="smoke only")
    merged.add_argument("--out", type=Path, required=True)
    grid = sub.add_parser("grid")
    grid.add_argument("--panel", action="append", required=True,
                      help="LABEL=SHADOW_DIR, in display order")
    grid.add_argument("--amp", default="fp16")
    grid.add_argument("--step", type=int, default=60000)
    grid.add_argument("--n", type=int, default=16)
    grid.add_argument("--out", type=Path, required=True)
    grid.add_argument("--index-out", type=Path, default=None,
                      help="JSON of the seed indices shown per panel")
    args = parser.parse_args(argv)
    try:
        if args.command == "grid":
            panels = []
            for item in args.panel:
                label, _, path = item.partition("=")
                panels.append((label, load_grid_rows(Path(path), args.amp, args.step, args.n)))
            draw_grid(panels, args.out)
            if args.index_out is not None:
                args.index_out.write_text(json.dumps(
                    {label: [int(v) for v in panel["seed_idx"]] for label, panel in panels},
                    indent=1) + "\n")
            print(f"wrote {args.out}")
            return 0
        if args.command == "task":
            steps = [int(item) for item in args.steps.split(":") if item.strip()]
            record = collect_task(args.shadow, args.run_id, args.amp, steps,
                                  json.loads(args.meta))
        else:
            record = merge(args.task_files, json.loads(args.extra), args.allow_off_protocol)
    except CollectError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {args.out}")
    if args.command == "merge":
        print(f"anchor passed: {record['anchor']['passed']}; reading: {record['reading']['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
