"""Collect the delta-sweep results of T7.2: one JSON per task, then one JSON for the sweep.

``task`` runs at the end of every array task of ``delta_sweep.sbatch``: it reads the
``metrics<amp>_delta-<d>/`` trees of the shadow run and writes ``<run_id>_delta_sweep.json`` with
one row per noise level (LSD, octaves, variance ratio, the Inception block, ``M``, diversity and the
inherited band, with their provenance). ``merge`` runs locally on the copied-back task JSONs and
writes ``docs/RESULTS/delta_sweep/delta_sweep.json`` with the reproduction anchor checked.

``inherited_measured`` is the pre-registered share, biased under a non-zero mean image (D23).
Every row also carries the D23 within-seed share ``I_w = 1 - M/(M-1) D_pix (W^2-1) / sum P_ref``
(``ihdm.analysis.inherited.within_seed_share``, constants from
``docs/RESULTS/inherited_band_constants.json`` matched by dataset, sha256 and sigma_max) and
``rho = (1 - I_w) / (1 - I)``, the within-seed variance relative to what the blur removed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ihdm.analysis.inherited import DEFAULT_CONSTANTS_PATH, load_constants, within_seed_share
from ihdm.metrics.run_eval import metrics_dirname

#: The stored ``lsd_060000`` of each run at delta = 1.25 sigma (docs/RESULTS/tables/t7_t_tau.md).
ANCHORS: dict[str, float] = {
    "ixi_A0_s1": 0.2497,
    "ixi_A3_s1": 0.08400,
    "lsun_church_A0_s1": 1.445,
    "lsun_church_A3_s1": 0.6416,
}
ANCHOR_TOLERANCE: float = 0.003
ANCHOR_DELTA: float = 0.0125

_FINAL_KEYS: tuple[str, ...] = (
    "M", "M_lp", "seed_nn_fraction", "diversity_pix", "diversity_lp", "n_diversity_seeds",
    "n_per_seed", "inherited_measured", "inherited_predicted", "inherited_measured_low_band",
    "inherited_predicted_low_band", "low_band_sigma_px", "sigma_max", "final_set",
    "seed_list_sha256",
)
_INCEPTION_KEYS: tuple[str, ...] = (
    "kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low", "fid_ci_high", "precision",
    "recall", "density", "coverage", "n_samples", "n_reference",
)


class CollectError(Exception):
    """A result file of the sweep is missing or inconsistent."""


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CollectError(f"{path} is missing")
    return json.loads(path.read_text())


def task_row(shadow: Path, amp: str, delta: float, step: int) -> dict[str, Any]:
    """Return the record of one (run, delta) condition from its metrics tree.

    Parameters
    ----------
    shadow : Path
        The (shadow) run directory.
    amp : str
        The sampling precision of the sweep.
    delta : float
        The noise level of the condition.
    step : int
        The evaluated checkpoint.

    Returns
    -------
    dict[str, Any]
        One flat row plus the octave profile, the Inception block and the provenance.

    Raises
    ------
    CollectError
        If a result file is missing or records another noise level or checkpoint.
    """
    metrics = Path(shadow) / metrics_dirname(amp, delta)
    summary = _read(metrics / "summary.json")
    ckpt = _read(metrics / f"ckpt_{step:06d}.json")
    final = _read(metrics / "final.json")
    for name, record in (("ckpt", ckpt), ("final", final)):
        if record.get("delta") != float(delta) or int(record.get("step", -1)) != int(step):
            raise CollectError(
                f"{metrics}: {name} record has delta={record.get('delta')!r}, "
                f"step={record.get('step')!r}; expected {delta!r}, {step}"
            )
    inception = final.get("inception") or {}
    return {
        "delta": float(delta),
        "step": int(step),
        "lsd": ckpt["lsd"],
        "variance_ratio": ckpt["variance_ratio"],
        "lsd_octaves": ckpt["lsd_octaves"],
        "n_lsd_samples": ckpt["n_samples"],
        **{key: final.get(key) for key in _FINAL_KEYS},
        "inception": {key: inception.get(key) for key in _INCEPTION_KEYS} if inception else None,
        "checkpoint_sha256": ckpt["checkpoint_sha256"],
        "seed_list_sha256_lsd": ckpt["seed_list_sha256"],
        "sample_rng_seed": ckpt["sample_rng_seed"],
        "sample_batch": ckpt["sample_batch"],
        "amp": ckpt["amp"],
        "sampling_log": summary["sampling"]["log"],
        "config_sha256": summary["run"]["config_sha256"],
        "dataset": summary["run"]["dataset"],
        "dataset_sha256": summary["dataset_sha256"],
    }


def d23_shares(row: dict[str, Any], constants_path: Path | None = DEFAULT_CONSTANTS_PATH
               ) -> dict[str, Any]:
    """Return the D23 within-seed share ``I_w`` and ``rho`` of one row.

    Parameters
    ----------
    row : dict[str, Any]
        A row of :func:`task_row` (needs ``dataset``, ``dataset_sha256``, ``sigma_max``,
        ``diversity_pix`` and ``n_per_seed``).
    constants_path : Path or None
        The D23 constants file.

    Returns
    -------
    dict[str, Any]
        ``I_w_d23``, ``rho_d23``, the constants used and their source; ``None`` values with the
        reason when the constants do not apply.
    """
    constants = load_constants(constants_path)
    entry, reason = constants.lookup(
        str(row["dataset"]), str(row["dataset_sha256"]), float(row["sigma_max"])
    )
    if entry is None:
        return {"I_w_d23": None, "rho_d23": None, "d23_note": reason}
    share_w = within_seed_share(
        float(row["diversity_pix"]), float(row["n_per_seed"]), entry.n_pix, entry.sum_power
    )
    removed = 1.0 - entry.share_predicted
    return {
        "I_w_d23": share_w,
        "rho_d23": (1.0 - share_w) / removed if removed > 0.0 else None,
        "d23_share_predicted": entry.share_predicted,
        "d23_sum_power_ref": entry.sum_power,
        "d23_n_pix": entry.n_pix,
        "d23_constants": constants.label,
    }


def collect_task(
    shadow: Path, run_id: str, amp: str, deltas: list[float], step: int, sigma: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Return the task record: every condition of one run."""
    rows = []
    for delta in deltas:
        row = task_row(shadow, amp, delta, step)
        row["sigma"] = float(sigma)
        row["delta_over_sigma"] = float(delta) / float(sigma)
        row.update(d23_shares(row))
        rows.append(row)
    return {
        "run_id": run_id,
        "amp": amp,
        "step": int(step),
        "deltas": [float(d) for d in deltas],
        "rows": rows,
        "meta": meta,
        "created": datetime.now(UTC).isoformat(),
    }


def anchor_check(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the delta = 1.25 sigma LSD of every run with its stored ``lsd_060000``."""
    out: dict[str, Any] = {"tolerance": ANCHOR_TOLERANCE, "delta": ANCHOR_DELTA, "runs": {}}
    passed = True
    for task in tasks:
        run_id = task["run_id"]
        rows = [row for row in task["rows"] if math.isclose(row["delta"], ANCHOR_DELTA)]
        if run_id not in ANCHORS or not rows:
            out["runs"][run_id] = {"status": "no anchor"}
            passed = False
            continue
        got, want = float(rows[0]["lsd"]), ANCHORS[run_id]
        ok = abs(got - want) <= ANCHOR_TOLERANCE
        passed &= ok
        out["runs"][run_id] = {"stored": want, "reproduced": got, "difference": got - want,
                               "ok": ok}
    out["passed"] = passed
    return out


def merge(task_files: list[Path], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge the task records into the sweep record."""
    tasks = [json.loads(Path(path).read_text()) for path in task_files]
    return {
        "ticket": "T7.2",
        "description": (
            "delta sweep at checkpoint 60,000 (EMA), fp16, batch 32: the frozen 500-seed LSD set "
            "(rng 2026, same noise stream across delta) feeds LSD, octaves, variance ratio and, "
            "via --final-from-lsd, KID/FID/precision/recall/density/coverage and M; held-out "
            "diversity 40 seeds x 5 feeds D_pix, D_lp and the inherited band."
        ),
        "anchor": anchor_check(tasks),
        "tasks": tasks,
        **extra,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``task`` or the ``merge`` subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    task = sub.add_parser("task")
    task.add_argument("--shadow", type=Path, required=True)
    task.add_argument("--run-id", required=True)
    task.add_argument("--amp", default="fp16")
    task.add_argument("--deltas", required=True, help="colon-separated noise levels")
    task.add_argument("--step", type=int, default=60000)
    task.add_argument("--sigma", type=float, default=0.01)
    task.add_argument("--meta", default="{}", help="JSON object of provenance")
    task.add_argument("--out", type=Path, required=True)
    merged = sub.add_parser("merge")
    merged.add_argument("task_files", type=Path, nargs="+")
    merged.add_argument("--extra", default="{}", help="JSON object merged at the top level")
    merged.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "task":
            deltas = [float(item) for item in args.deltas.split(":") if item.strip()]
            record = collect_task(
                args.shadow, args.run_id, args.amp, deltas, args.step, args.sigma,
                json.loads(args.meta),
            )
        else:
            record = merge(args.task_files, json.loads(args.extra))
    except CollectError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    if args.command == "merge":
        print(f"anchor passed: {record['anchor']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
