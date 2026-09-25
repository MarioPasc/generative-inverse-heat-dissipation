"""Check a run directory with ``ihdm.train.validate_run`` and print a PASS/FAIL verdict.

    python slurm/loginexa/check_run.py <run_dir> [--data-root R] [--lr 1e-4] [--allow-skips]
        [--no-done] [--n-iters N] [--label H3]

Prints the H-TRAIN §2 artefact listing, the H-TRAIN §3 cadence summary, every problem found and
one ``<label> check_run PASS|FAIL <run_dir>`` line; exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ihdm.train.validate_run import (  # noqa: E402
    check_run_dir,
    expectation_from_config,
    read_metrics_strict,
)


def _listing(run: Path) -> None:
    print("-- H-TRAIN §2 listing")
    print("top:", sorted(p.name for p in run.iterdir()))
    for sub in ("checkpoints", "checkpoints-meta", "grids"):
        if (run / sub).is_dir():
            print(f"{sub}:", sorted(f"{p.name} ({p.stat().st_size / 2**20:.1f} MiB)"
                                    for p in (run / sub).iterdir()))
    lines = (run / "metrics.jsonl").read_text().splitlines()
    for line in lines[-3:]:
        print("tail:", line[:300])
    manifest = json.loads((run / "manifest.json").read_text())
    print("manifest:", {k: manifest.get(k) for k in (
        "run_id", "git_sha", "gpu", "torch", "cuda", "n_params", "batch_size", "n_iters",
        "config_sha256", "recipe_sha256")})
    print("manifest schedule:", manifest["schedule"]["name"], manifest["schedule"]["sha256"])
    print("manifest data.images_sha256:", manifest["data"]["images_sha256"])


def _cadence(run: Path) -> None:
    records, _ = read_metrics_strict(run / "metrics.jsonl")
    kinds: dict[str, list[int]] = {}
    for r in records:
        kinds.setdefault(r.get("kind", "?"), []).append(r.get("step"))
    print("-- H-TRAIN §3 cadence:", {k: (len(v), v[:3], v[-1]) for k, v in kinds.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--allow-skips", action="store_true")
    parser.add_argument("--no-done", action="store_true")
    parser.add_argument("--label", default="check")
    args = parser.parse_args(argv)

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    overrides = {"allow_skips": args.allow_skips, "expect_done": not args.no_done}
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.n_iters is not None:
        overrides["n_iters"] = args.n_iters
    exp = expectation_from_config(config, **overrides)
    print(f"-- expectation {exp}")
    _listing(run)
    _cadence(run)
    problems = check_run_dir(run, exp, Path(args.data_root) if args.data_root else None)
    for problem in problems:
        print("PROBLEM", problem)
    verdict = "PASS" if not problems else "FAIL"
    print(f"{args.label} check_run {verdict} {run} ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
