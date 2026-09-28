"""Synthetic evaluation outputs of the 30-run campaign, built from the real schemas (T5.2).

The templates of ``summary.json``, ``ckpt_<step>.json``, ``final.json`` and the ``.npy`` sidecars
come from ONE real :func:`ihdm.metrics.run_eval.evaluate_run` call on the tiny T4.3 run
(``tests/metrics/test_run_eval.py``), cached per process; the Inception block and the gate
records come from the real result classes. The run directories are T3.5's synthetic 60k runs
(``tests/cli/test_check_array.py::make_run``), extended from 40k by a resume. Every JSON is written
with :func:`ihdm.metrics.io.write_json`, and every tar is the shadow run exactly as
``slurm/eval/eval_array.sbatch`` returns it: ``<run_id>/metrics<amp>/…``, ``<run_id>/samples<amp>/…``
and links to the real run's files.
"""

from __future__ import annotations

import copy
import csv
import functools
import json
import math
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ihdm.metrics.inception import InceptionResult
from ihdm.metrics.io import write_json
from ihdm.metrics.run_eval import (
    EvalRequest,
    evaluate_run,
    evaluated_steps,
    metrics_dirname,
    samples_dirname,
)
from ihdm.stats.bootstrap import GateResult, Interval
from tests.cli.test_check_array import make_run
from tests.metrics.test_run_eval import _build_dataset, _config, _write_run

REPO = Path(__file__).resolve().parents[2]
REPO_CELLS = REPO / "slurm" / "array" / "cells.csv"
EXPECTED_SEEDS = REPO / "slurm" / "eval" / "expected_seed_lists.csv"
N_ITERS = 60000
AMP = "fp16"
SUFFIX = "_amp-fp16"
#: Run 11 of the campaign: an abort at 30,136 abandoned by a resume from 30,001 (D21).
RUN_11 = "lsun_church_A3_s3"
GATE_RUNS = ("ixi_A0_s1", "lsun_church_A0_s1")
SIDECARS = (
    "final_memorisation_per_sample_d.npy",
    "final_memorisation_per_sample_nn.npy",
    "final_pca_components.npy",
    "final_pca_mean.npy",
    "final_pca_train_scores.npy",
)


@functools.cache
def real_templates() -> dict[str, Any]:
    """Run the real ``evaluate_run`` once and return its result files (JSON and ``.npy`` bytes)."""
    with tempfile.TemporaryDirectory(prefix="t52_template_") as tmp:
        base = Path(tmp)
        dataset = _build_dataset(base / "data")
        workdir = _write_run(base / "runs" / "template", _config(dataset))
        evaluate_run(EvalRequest(run=workdir, ckpts="250,750", n_lsd=8, n_final=8, n_seeds=2,
                                 n_per_seed=3, sample_batch=8, skip_inception=True,
                                 device="cpu"))
        metrics = workdir / metrics_dirname("off")
        return {
            "summary": json.loads((metrics / "summary.json").read_text()),
            "ckpt": json.loads((metrics / "ckpt_000750.json").read_text()),
            "final": json.loads((metrics / "final.json").read_text()),
            "sidecars": {name: (metrics / name).read_bytes() for name in SIDECARS},
        }


def read_expected() -> dict[str, tuple[str, str]]:
    with EXPECTED_SEEDS.open(newline="") as handle:
        return {r["dataset"]: (r["intermediate_sha256"], r["final_sha256"])
                for r in csv.DictReader(handle)}


def repo_cells() -> list[dict[str, str]]:
    with REPO_CELLS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _inception(offset: float) -> dict[str, Any]:
    return InceptionResult(
        kid=0.02 + offset, kid_ci_low=0.015 + offset, kid_ci_high=0.025 + offset,
        fid=40.0 + 100 * offset, fid_ci_low=38.0 + 100 * offset, fid_ci_high=42.0 + 100 * offset,
        precision=0.6, recall=0.5 - offset, density=0.9, coverage=0.7 - offset, k=5,
        n_samples=2000, n_reference=800, n_boot=200, feature_dim=2048,
        weights_path="/tmp/inception-2015-12-05.pt",
        notes={"kid": "headline", "fid": "biased upward at n_reference = 800"},
    ).to_json()


def _lsd(step: int, offset: float) -> float:
    return 0.25 + offset + 0.2 * math.exp(-step / 15000)


def gate_record(pair: tuple[int, int], seed_sha: str, amp: str = AMP) -> dict[str, Any]:
    """A gate record as ``run_eval.run_gate`` writes it (``GateResult.to_json`` + context)."""
    lsd_a, lsd_b = _lsd(pair[0], 0.0), _lsd(pair[1], 0.0)
    result = GateResult(
        lsd_a=lsd_a, lsd_b=lsd_b,
        difference=Interval(point=lsd_a - lsd_b, low=0.8 * (lsd_a - lsd_b),
                            high=1.2 * (lsd_a - lsd_b), n=500, n_boot=1000, alpha=0.05),
        extend=True, relative_change=(lsd_a - lsd_b) / lsd_a, n_seeds=500, n_reference=800,
        n_bins=43,
    )
    record = result.to_json()
    record.update({"step_a": pair[0], "step_b": pair[1], "seed_list_sha256": seed_sha,
                   "sample_rng_seed": 2026, "sample_batch": 32, "amp": amp,
                   "rule": "extend the run when the bootstrap 95% CI ... (D10, D17)"})
    return record


@dataclass
class Campaign:
    """The synthetic inputs of one collection, relocatable under ``root``."""

    root: Path

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def gate_dir(self) -> Path:
        return self.root / "eval" / "gate"

    @property
    def run_root(self) -> Path:
        return self.root / "runs"

    @property
    def cells(self) -> Path:
        return self.root / "cells.csv"

    @property
    def shadow_root(self) -> Path:
        return self.root / "shadow"

    def metrics_dir(self, run_id: str) -> Path:
        return self.shadow_root / run_id / metrics_dirname(AMP)

    def summary_path(self, run_id: str) -> Path:
        return self.eval_dir / f"{run_id}{SUFFIX}_summary.json"

    def tar_path(self, run_id: str) -> Path:
        return self.eval_dir / f"{run_id}{SUFFIX}.tar"

    def repack(self, run_id: str, plain_summary: bool = True) -> None:
        """Re-tar the shadow run and, by default, refresh the plain summary from it."""
        with tarfile.open(self.tar_path(run_id), "w") as tar:
            tar.add(self.shadow_root / run_id, arcname=run_id)
        if plain_summary:
            shutil.copyfile(self.metrics_dir(run_id) / "summary.json", self.summary_path(run_id))

    def edit_json(self, run_id: str, name: str, edit: Callable[[dict], None],
                  plain_summary: bool = True) -> None:
        """Edit a result file of the shadow run through ``write_json`` and repack."""
        path = self.metrics_dir(run_id) / name
        record = json.loads(path.read_text())
        edit(record)
        write_json(path, record)
        self.repack(run_id, plain_summary)

    def config(self, out: Path, **overrides: Any):
        from ihdm.analysis.collect import CollectConfig

        values = dict(eval_dir=self.eval_dir, gate_dir=self.gate_dir, run_root=self.run_root,
                      cells=self.cells, out=out, n_iters=N_ITERS, amp=AMP,
                      expected_seed_lists=EXPECTED_SEEDS)
        values.update(overrides)
        return CollectConfig(**values)

    def cli_args(self, out: Path, *extra: str) -> list[str]:
        return ["--eval-dir", str(self.eval_dir), "--run-root", str(self.run_root), "--cells",
                str(self.cells), "--out", str(out), "--expected-seed-lists", str(EXPECTED_SEEDS),
                *extra]


def _splice_run_11(workdir: Path) -> None:
    """Give a run run 11's history: 6 kept skips, then an abort abandoned by a resume at 30,001."""
    path = workdir / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    kept_skips = [{"step": s, "kind": "skip", "loss": None, "n_skipped": i + 1, "consecutive": 1}
                  for i, s in enumerate((27883, 27892, 27893, 27896, 27975, 28091))]
    abandoned = [
        {"step": 30111, "kind": "skip", "loss": None, "n_skipped": 7, "consecutive": 1},
        {"step": 30136, "kind": "abort", "reason": "10 consecutive non-finite training losses",
         "loss": "nan", "n_skipped": 29, "consecutive": 10,
         "abort_state": "checkpoints-meta/abort_step_030136.pth"},
    ]
    resume = {"step": 30001, "kind": "resume", "from": "checkpoints-meta/checkpoint.pth"}
    before = [r for r in records if r["step"] < 30001]
    after = [r for r in records if r["step"] >= 30001]
    spliced = [*before, *kept_skips, *after[:3], *abandoned, resume, *after]
    for record in spliced:
        if record["kind"] == "done":
            record["n_skipped"] = 6
    path.write_text("".join(json.dumps(r) + "\n" for r in spliced))


def _write_results(campaign: Campaign, row: dict[str, str], position: int,
                   expected: dict[str, tuple[str, str]]) -> None:
    templates = real_templates()
    run_id, dataset = row["run_id"], row["dataset_id"]
    offset = 0.001 * position
    intermediate, final_sha = expected[dataset]
    shadow = campaign.shadow_root / run_id
    metrics = campaign.metrics_dir(run_id)
    metrics.mkdir(parents=True)
    run_dir = campaign.run_root / run_id
    for item in ("config.json", "manifest.json", "checkpoints", "grids", "metrics.jsonl"):
        (shadow / item).symlink_to(run_dir / item)
    request = shadow / samples_dirname(AMP) / f"{N_ITERS:06d}" / "lsd" / "request.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps({"shape": [500, 1, 192, 192]}))

    steps = evaluated_steps(N_ITERS)
    for step in steps:
        ckpt = copy.deepcopy(templates["ckpt"])
        ckpt.update({"step": step, "lsd": _lsd(step, offset), "amp": AMP,
                     "seed_list_sha256": intermediate,
                     "checkpoint": str(run_dir / "checkpoints" / f"ema_iter_{step:06d}.pt")})
        write_json(metrics / f"ckpt_{step:06d}.json", ckpt)
    final = copy.deepcopy(templates["final"])
    final.update({"step": N_ITERS, "amp": AMP, "seed_list_sha256": final_sha,
                  "lsd": _lsd(N_ITERS, offset) + 0.001,
                  "intermediate_lsd": _lsd(N_ITERS, offset), "M": 1.0 + offset,
                  "inception": _inception(offset),
                  "inception_note": "KID is the headline; FID is reported with its n_reference."})
    write_json(metrics / "final.json", final)
    for name, data in templates["sidecars"].items():
        (metrics / name).write_bytes(data)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = copy.deepcopy(templates["summary"])
    summary["run"].update({"run_id": run_id, "dataset": dataset, "arm": row["arm"],
                           "seed": int(row["seed"]),
                           "config_sha256": manifest["config_sha256"]})
    summary.update({
        "checkpoint_steps": list(steps), "checkpoint_selection": "all", "final_step": N_ITERS,
        "lsd_by_step": {step: _lsd(step, offset) for step in steps},
        "variance_ratio_by_step": {step: 1.0 for step in steps},
    })
    summary["seed_lists"].update({"intermediate_sha256": intermediate, "final_sha256": final_sha})
    summary["sampling"].update({"amp": AMP, "device": "cuda", "batch": 32, "n_lsd": 500,
                                "n_final": 2000, "n_seeds": 40, "n_per_seed": 50})
    summary["final"] = {k: v for k, v in final.items()
                        if k not in {"diversity_per_seed_pix", "diversity_per_seed_lp", "pca",
                                     "radial"}}
    summary["final"]["has_inception"] = True
    write_json(metrics / "summary.json", summary)
    campaign.repack(run_id)


def build_campaign(root: Path) -> Campaign:
    """Write the healthy 30-run campaign of the repository's ``cells.csv`` under ``root``."""
    campaign = Campaign(Path(root))
    campaign.gate_dir.mkdir(parents=True)
    shutil.copyfile(REPO_CELLS, campaign.cells)
    expected = read_expected()
    for position, row in enumerate(repo_cells()):
        workdir = make_run(campaign.run_root, row["dataset_id"], row["arm"], int(row["seed"]),
                           n_iters=N_ITERS, extended_from=40000)
        if row["run_id"] == RUN_11:
            _splice_run_11(workdir)
        _write_results(campaign, row, position, expected)
    for run_id in GATE_RUNS:
        dataset = run_id.rsplit("_", 2)[0]
        seed_sha = expected[dataset][0]
        write_json(campaign.gate_dir / f"{run_id}{SUFFIX}_gate.json",
                   gate_record((35000, 40000), seed_sha))
        write_json(campaign.gate_dir / f"{run_id}{SUFFIX}_gate_055000_060000.json",
                   gate_record((55000, 60000), seed_sha))
    return campaign


def copy_campaign(source: Campaign, root: Path) -> Campaign:
    """Copy a campaign (links kept as links) so one test can damage it."""
    shutil.copytree(source.root, root, symlinks=True)
    return Campaign(Path(root))
