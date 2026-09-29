"""Collect the outputs of the evaluated runs into one checked ``results/`` folder (T5.2).

The evaluation array (T5.1, T3.5) leaves its outputs in three places: per run a plain
``<run_id><amp>_summary.json`` and a tar of the shadow run in the eval directory, the plateau-gate
JSONs in ``<eval dir>/gate/`` under two naming schemes, and the training run directories. This
module reads them, runs the integrity checks of ``docs/SPECIFICATIONS/M5-evaluation/T5.2`` and,
only when every check passes, publishes one folder that T6.1/T6.2 read and nothing else::

    results/
    ├── collection.json   provenance, input sha256, check verdicts
    ├── index.csv         one row per run
    ├── gates/            <run_id>_gate_<early:06d>_<late:06d>.json
    └── runs/<run_id>/    summary.json, ckpt_<step>.json, final.json, the .npy sidecars,
                          manifest.json, config.json, metrics.canonical.jsonl, grid_final.png

The folder is staged in a sibling temporary directory and renamed into place at the end, so a
reader sees either no folder or a complete one. ``allow_missing`` (dry runs only) tolerates absent
inputs and publishes a folder marked ``"complete": false``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configs.spectral.arms import EXPERIMENT_CELLS
from ihdm.cli.check_array import Cell, CheckArrayError, _recipe_hash, read_cells
from ihdm.metrics.io import write_json
from ihdm.metrics.run_eval import AMP_MODES, EVAL_STRIDE, evaluated_steps, metrics_dirname
from ihdm.paths import repo_root
from ihdm.train.validate_run import canonical_records, read_metrics_strict

__all__ = [
    "CHECKS",
    "CollectConfig",
    "CollectError",
    "CollectionReport",
    "CheckResult",
    "GateFile",
    "amp_suffix",
    "collect",
    "discover_gates",
    "format_report",
    "read_expected_seed_lists",
    "required_gate_pairs",
    "strict_json",
]

logger = logging.getLogger(__name__)

#: The integrity checks. C1-C7 are the ticket's list; C8 and C9 cover the archive and the gates.
CHECKS: dict[str, str] = {
    "C1": "cells.csv == EXPERIMENT_CELLS; all cells present; summary/manifest identity match",
    "C2": "checkpoint_steps == evaluated_steps(n_iters); every ckpt_<step>.json and final.json",
    "C3": "sampling.amp == --amp in every summary and result file",
    "C4": "seed-list digests == slurm/eval/expected_seed_lists.csv",
    "C5": "config_sha256 == manifest; manifest recipe_sha256 == hash of config.json",
    "C6": "strict JSON (no NaN/Infinity) and every 05-metrics.md §9 key present",
    "C7": "canonical metrics.jsonl ends with done at n_iters and holds no abort",
    "C8": "tar summary == plain summary; .npy sidecars and final grid present",
    "C9": "gates: both naming schemes, strict, pair/amp/seed list consistent, required pairs",
}

#: ``05-metrics.md`` §9 keys of ``ckpt_<step>.json`` (plus the context the checks read).
CKPT_KEYS: tuple[str, ...] = ("lsd", "lsd_octaves", "variance_ratio", "n_samples", "step",
                              "seed_list_sha256", "amp")
#: ``05-metrics.md`` §9 keys of ``final.json`` outside the Inception block.
FINAL_KEYS: tuple[str, ...] = ("lsd", "intermediate_lsd", "diversity_pix", "diversity_lp", "M",
                               "M_lp", "seed_nn_fraction", "inherited_measured",
                               "inherited_predicted", "inception", "step", "seed_list_sha256",
                               "amp", "per_sample_files", "pca")
#: The Inception block (§7): FID and KID with their CIs, recall, coverage, ``n_reference``.
INCEPTION_KEYS: tuple[str, ...] = ("kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low",
                                   "fid_ci_high", "recall", "coverage", "precision", "density",
                                   "n_reference")
SUMMARY_KEYS: tuple[str, ...] = ("run", "checkpoint_steps", "final_step", "lsd_by_step",
                                 "seed_lists", "sampling", "final")
SUMMARY_RUN_KEYS: tuple[str, ...] = ("run_id", "dataset", "arm", "seed", "config_sha256")
SUMMARY_SEED_KEYS: tuple[str, ...] = ("intermediate_sha256", "final_sha256")
GATE_KEYS: tuple[str, ...] = ("step_a", "step_b", "lsd_a", "lsd_b", "difference", "extend",
                              "n_seeds", "amp", "seed_list_sha256")
GATE_DIFFERENCE_KEYS: tuple[str, ...] = ("point", "ci_low", "ci_high", "alpha", "n", "n_boot",
                                         "excludes_zero")
#: The per-sample and PCA arrays ``evaluate_run`` writes beside ``final.json`` (small; no samples).
SIDECARS: tuple[str, ...] = (
    "final_memorisation_per_sample_d.npy",
    "final_memorisation_per_sample_nn.npy",
    "final_pca_components.npy",
    "final_pca_mean.npy",
    "final_pca_train_scores.npy",
)
#: The pair a legacy ``<run_id><amp>_gate.json`` holds (gate job 2432703, before T3.5).
LEGACY_GATE_PAIR: tuple[int, int] = (35000, 40000)
#: Cells the gate job evaluates (``slurm/eval/gate.sbatch`` ``GATE_CELLS=0:3``).
DEFAULT_GATE_CELLS: tuple[int, ...] = (0, 3)

_GATE_NAME = re.compile(r"^(?P<stem>.+)_gate(?:_(?P<early>\d{6})_(?P<late>\d{6}))?\.json$")
_AMP_TAIL = re.compile(r"_amp-(?:" + "|".join(m for m in AMP_MODES if m != "off") + r")$")

Json = dict[str, Any]


class CollectError(Exception):
    """Raised when the collection cannot start (unusable arguments or cell table)."""


# --------------------------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectConfig:
    """Where the inputs are and what the collection must find.

    Parameters
    ----------
    eval_dir : Path
        Holds ``<run_id><amp>_summary.json`` and ``<run_id><amp>.tar`` per run.
    gate_dir : Path
        Holds the gate JSONs (both naming schemes).
    run_root : Path
        Holds one training run directory per ``run_id``.
    cells : Path
        ``slurm/array/cells.csv``.
    out : Path
        The ``results/`` folder to publish; it must not exist yet.
    n_iters : int
        The run length; the evaluated steps are ``evaluated_steps(n_iters)``.
    amp : str
        The sampling precision every run must have used (D20: ``fp16``).
    allow_missing : bool
        Tolerate absent inputs and publish a partial folder (dry runs only).
    expected_seed_lists : Path
        ``slurm/eval/expected_seed_lists.csv``.
    gate_cells : tuple[int, ...]
        Indices of the cells whose gates are required.
    """

    eval_dir: Path
    gate_dir: Path
    run_root: Path
    cells: Path
    out: Path
    n_iters: int = 60000
    amp: str = "fp16"
    allow_missing: bool = False
    expected_seed_lists: Path = field(
        default_factory=lambda: repo_root() / "slurm" / "eval" / "expected_seed_lists.csv"
    )
    gate_cells: tuple[int, ...] = DEFAULT_GATE_CELLS


@dataclass
class CheckResult:
    """The problems and the absent inputs one check found.

    Parameters
    ----------
    id : str
        ``C1`` … ``C9``.
    title : str
        What the check asserts.
    problems : list[str]
        Contradictions found in inputs that exist; always a failure.
    missing : list[str]
        Absent inputs; a failure unless ``allow_missing``.
    """

    id: str
    title: str
    problems: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """``FAIL`` (a problem), ``MISSING`` (only absent inputs) or ``PASS``."""
        if self.problems:
            return "FAIL"
        return "MISSING" if self.missing else "PASS"

    def to_json(self) -> Json:
        """Return the record written into ``collection.json``."""
        return {"id": self.id, "title": self.title, "verdict": self.verdict,
                "problems": list(self.problems), "missing": list(self.missing)}


@dataclass(frozen=True)
class GateFile:
    """One gate JSON found in the gate directory.

    Parameters
    ----------
    path : Path
        The file.
    run_id : str
        The run it belongs to (the amp suffix removed).
    pair : tuple[int, int]
        ``(early, late)``, from the name; the legacy name means :data:`LEGACY_GATE_PAIR`.
    legacy : bool
        The name carries no step pair.
    """

    path: Path
    run_id: str
    pair: tuple[int, int]
    legacy: bool

    @property
    def dest_name(self) -> str:
        """The name in ``results/gates/``."""
        return f"{self.run_id}_gate_{self.pair[0]:06d}_{self.pair[1]:06d}.json"


@dataclass
class RunRecord:
    """What the collection found for one cell."""

    cell: Cell
    tier: str
    has_run_dir: bool = False
    has_summary: bool = False
    has_tar: bool = False
    gates: list[str] = field(default_factory=list)
    row: dict[str, Any] = field(default_factory=dict)
    inputs: Json = field(default_factory=dict)


@dataclass
class CollectionReport:
    """The outcome of :func:`collect`.

    Parameters
    ----------
    config : CollectConfig
        The request.
    checks : dict[str, CheckResult]
        One per entry of :data:`CHECKS`.
    runs : list[RunRecord]
        One per cell, in table order.
    gates : list[Json]
        Every gate collected, with its values.
    ignored : list[str]
        Files of the gate directory that belong to another precision or are not gates.
    written : Path or None
        The published folder, or ``None`` when nothing was published.
    """

    config: CollectConfig
    checks: dict[str, CheckResult]
    runs: list[RunRecord] = field(default_factory=list)
    gates: list[Json] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    written: Path | None = None

    @property
    def n_problems(self) -> int:
        """Problems over all checks."""
        return sum(len(c.problems) for c in self.checks.values())

    @property
    def n_missing(self) -> int:
        """Absent inputs over all checks."""
        return sum(len(c.missing) for c in self.checks.values())

    @property
    def verdict(self) -> str:
        """``COMPLETE`` (all pass), ``INCOMPLETE`` (only absent inputs) or ``FAIL``."""
        if self.n_problems:
            return "FAIL"
        return "INCOMPLETE" if self.n_missing else "COMPLETE"

    def fail(self, check: str, who: str, message: str) -> None:
        """Record a problem under ``check``."""
        self.checks[check].problems.append(f"{who}: {message}")

    def miss(self, check: str, who: str, message: str) -> None:
        """Record an absent input under ``check``."""
        self.checks[check].missing.append(f"{who}: {message}")


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def amp_suffix(amp: str) -> str:
    """Return the file-name suffix of a precision: ``""`` for ``off``, ``_amp-<mode>`` otherwise.

    Parameters
    ----------
    amp : str
        One of :data:`ihdm.metrics.run_eval.AMP_MODES`.

    Returns
    -------
    str
        The suffix ``slurm/eval/common.sh`` ``ihdm_amp_suffix`` prints.

    Raises
    ------
    CollectError
        If ``amp`` is not a known mode.
    """
    if amp not in AMP_MODES:
        raise CollectError(f"amp must be one of {AMP_MODES}, got {amp!r}")
    return "" if amp == "off" else f"_amp-{amp}"


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-strict JSON token {token}")


def strict_json(data: bytes | str) -> Any:
    """Parse JSON refusing the ``NaN``/``Infinity`` tokens that ``json.loads`` accepts.

    Parameters
    ----------
    data : bytes or str
        The document.

    Returns
    -------
    Any
        The parsed object.

    Raises
    ------
    ValueError
        If the text is not strict JSON.
    """
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    return json.loads(text, parse_constant=_reject_constant)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_record(path: Path) -> Json:
    return {"path": str(path), "sha256": _sha256_file(path), "size": Path(path).stat().st_size}


def _missing_keys(obj: Any, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(obj, dict):
        return list(keys)
    return [key for key in keys if key not in obj]


def _git_sha() -> str:
    """Return the HEAD of the collecting code, or ``"unknown"`` (e.g. an rsynced tree)."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True,
                                text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def required_gate_pairs(n_iters: int) -> tuple[tuple[int, int], ...]:
    """Return the gate pairs every gate cell must hold at run length ``n_iters``.

    Parameters
    ----------
    n_iters : int
        The run length.

    Returns
    -------
    tuple[tuple[int, int], ...]
        The legacy 35k/40k pair (job 2432703) when the run reaches 40k, and
        ``(n_iters - 5000, n_iters)`` (T3.5, D22), sorted and without duplicates.
    """
    pairs = {(n_iters - EVAL_STRIDE, n_iters)}
    if n_iters >= LEGACY_GATE_PAIR[1]:
        pairs.add(LEGACY_GATE_PAIR)
    return tuple(sorted(pairs))


def read_expected_seed_lists(path: Path) -> dict[str, tuple[str, str]]:
    """Read ``expected_seed_lists.csv`` into ``{dataset: (intermediate_sha256, final_sha256)}``.

    Parameters
    ----------
    path : Path
        The table written by T5.1 (``evaluation_plan.md`` §5).

    Returns
    -------
    dict[str, tuple[str, str]]
        The two digests per dataset.

    Raises
    ------
    CollectError
        If the file is missing or lacks a column.
    """
    if not Path(path).is_file():
        raise CollectError(f"no expected seed-list table at {path}")
    with Path(path).open(newline="") as handle:
        try:
            return {row["dataset"]: (row["intermediate_sha256"], row["final_sha256"])
                    for row in csv.DictReader(handle)}
        except KeyError as error:
            raise CollectError(f"{path}: missing column {error}") from error


def _check_cell_table(report: CollectionReport, cells: list[Cell]) -> None:
    """C1: the table holds each cell of ``EXPERIMENT_CELLS`` exactly once (30 at D16)."""
    who = Path(report.config.cells).name
    for label, values in (("run_id", [c.run_id for c in cells]),
                          ("index", [c.index for c in cells])):
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            report.fail("C1", who, f"repeated {label} {repeated}")
    want = {(d, a, s) for d, a, seeds in EXPERIMENT_CELLS for s in seeds}
    have = {(c.dataset_id, c.arm, c.seed) for c in cells}
    if have != want:
        report.fail("C1", who, f"is not EXPERIMENT_CELLS ({len(want)} cells): absent "
                    f"{sorted(want - have)}, extra {sorted(have - want)}")


def _read_tiers(path: Path) -> dict[str, str]:
    with Path(path).open(newline="") as handle:
        return {row["run_id"]: row.get("tier", "") or "" for row in csv.DictReader(handle)}


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------


def discover_gates(gate_dir: Path, amp: str) -> tuple[list[GateFile], list[str]]:
    """Find the gate JSONs of one precision under both naming schemes.

    ``<run_id><amp>_gate_<early:06d>_<late:06d>.json`` is the pair-named form (T3.5);
    ``<run_id><amp>_gate.json`` is the legacy form of gate job 2432703 and means 35k/40k.

    Parameters
    ----------
    gate_dir : Path
        The gate directory (absent is not an error: no gate is found).
    amp : str
        The precision to collect; gates of another precision are ignored.

    Returns
    -------
    tuple[list[GateFile], list[str]]
        The gates of ``amp`` sorted by run and pair, and the names of the ``.json`` files ignored.
    """
    suffix = amp_suffix(amp)
    found: list[GateFile] = []
    ignored: list[str] = []
    if not Path(gate_dir).is_dir():
        return found, ignored
    for path in sorted(Path(gate_dir).glob("*.json")):
        match = _GATE_NAME.match(path.name)
        stem = match.group("stem") if match else ""
        if suffix:
            ours = stem.endswith(suffix)
            run_id = stem[: -len(suffix)] if ours else ""
        else:
            ours = bool(match) and not _AMP_TAIL.search(stem)
            run_id = stem
        if not match or not ours or not run_id:
            ignored.append(path.name)
            continue
        legacy = match.group("early") is None
        pair = LEGACY_GATE_PAIR if legacy else (int(match.group("early")),
                                                int(match.group("late")))
        found.append(GateFile(path=path, run_id=run_id, pair=pair, legacy=legacy))
    found.sort(key=lambda g: (g.run_id, g.pair, g.legacy))
    return found, ignored


def _check_gate(report: CollectionReport, gate: GateFile, dataset: str | None,
                expected: dict[str, tuple[str, str]]) -> tuple[bytes, Json] | None:
    """Parse and check one gate file; return its bytes and record, or ``None`` if unusable."""
    who = gate.path.name
    data = gate.path.read_bytes()
    try:
        record = strict_json(data)
    except ValueError as error:
        report.fail("C6", who, f"not strict JSON ({error})")
        return None
    if not isinstance(record, dict):
        report.fail("C9", who, "is not a JSON object")
        return None
    absent = _missing_keys(record, GATE_KEYS)
    absent += [f"difference.{k}" for k in _missing_keys(record.get("difference"),
                                                          GATE_DIFFERENCE_KEYS)]
    if absent:
        report.fail("C9", who, f"missing keys {absent}")
        return None
    if (record["step_a"], record["step_b"]) != gate.pair:
        report.fail("C9", who, f"holds steps ({record['step_a']}, {record['step_b']}), its name "
                    f"says {gate.pair}")
    if record["amp"] != report.config.amp:
        report.fail("C9", who, f"amp {record['amp']!r} != --amp {report.config.amp!r}")
    if dataset is None:
        report.fail("C9", who, f"run {gate.run_id} is not a cell of cells.csv")
    elif dataset in expected and record["seed_list_sha256"] != expected[dataset][0]:
        report.fail("C9", who, "seed_list_sha256 != the expected intermediate list of "
                    f"{dataset}")
    return data, record


def _collect_gates(report: CollectionReport, stage: Path, cells: list[Cell],
                   expected: dict[str, tuple[str, str]], records: dict[str, RunRecord]) -> None:
    config = report.config
    gates, report.ignored = discover_gates(config.gate_dir, config.amp)
    dataset_of = {c.run_id: c.dataset_id for c in cells}
    seen: dict[tuple[str, tuple[int, int]], str] = {}
    for gate in gates:
        key = (gate.run_id, gate.pair)
        if key in seen:
            report.fail("C9", gate.path.name, f"same pair as {seen[key]}")
            continue
        seen[key] = gate.path.name
        checked = _check_gate(report, gate, dataset_of.get(gate.run_id), expected)
        if checked is None:
            continue
        data, record = checked
        (stage / "gates").mkdir(parents=True, exist_ok=True)
        (stage / "gates" / gate.dest_name).write_bytes(data)
        difference = record["difference"]
        report.gates.append({
            "run_id": gate.run_id, "early": gate.pair[0], "late": gate.pair[1],
            "source": str(gate.path), "sha256": _sha256_bytes(data), "legacy_name": gate.legacy,
            "dest": f"gates/{gate.dest_name}", "lsd_a": record["lsd_a"],
            "lsd_b": record["lsd_b"], "difference": difference["point"],
            "ci_low": difference["ci_low"], "ci_high": difference["ci_high"],
            "extend": record["extend"],
        })
        if gate.run_id in records:
            records[gate.run_id].gates.append(f"{gate.pair[0]:06d}_{gate.pair[1]:06d}")
    by_index = {c.index: c for c in cells}
    for index in config.gate_cells:
        cell = by_index.get(index)
        if cell is None:
            report.fail("C9", f"gate cell {index}", "not a row of cells.csv")
            continue
        for pair in required_gate_pairs(config.n_iters):
            if (cell.run_id, pair) not in seen:
                report.miss("C9", cell.run_id, f"no gate {pair[0]:06d}_{pair[1]:06d} in "
                            f"{config.gate_dir}")


# --------------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------------


def _load_json_file(report: CollectionReport, path: Path, who: str) -> tuple[bytes, Any] | None:
    data = path.read_bytes()
    try:
        obj = strict_json(data)
    except ValueError as error:
        report.fail("C6", who, f"{path.name} is not strict JSON ({error})")
        return None
    if not isinstance(obj, dict):
        report.fail("C6", who, f"{path.name} is not a JSON object")
        return None
    return data, obj


def _run_directory(report: CollectionReport, record: RunRecord, dest: Path) -> Json | None:
    """Check and stage the training run's own files; return its manifest."""
    cell, config = record.cell, report.config
    who = cell.run_id
    workdir = Path(config.run_root) / cell.run_id
    record.has_run_dir = workdir.is_dir()
    if not record.has_run_dir:
        report.miss("C1", who, f"no run directory {workdir}")
        return None
    manifest: Json | None = None
    run_config: Json | None = None
    for name in ("manifest.json", "config.json"):
        path = workdir / name
        if not path.is_file():
            report.fail("C1", who, f"no {name} in {workdir}")
            continue
        loaded = _load_json_file(report, path, who)
        record.inputs[name] = _file_record(path)
        if loaded is None:
            continue
        (dest / name).write_bytes(loaded[0])
        if name == "manifest.json":
            manifest = loaded[1]
        else:
            run_config = loaded[1]
    if manifest is not None:
        want = {"run_id": cell.run_id, "dataset_id": cell.dataset_id, "arm": cell.arm,
                "seed": cell.seed}
        for key, value in want.items():
            if manifest.get(key) != value:
                report.fail("C1", who, f"manifest {key} {manifest.get(key)!r} != cells.csv "
                            f"{value!r}")
        if manifest.get("n_iters") != config.n_iters:
            report.fail("C2", who, f"manifest n_iters {manifest.get('n_iters')!r} != "
                        f"{config.n_iters}")
        if run_config is not None and manifest.get("recipe_sha256") != _recipe_hash(run_config):
            report.fail("C5", who, "manifest recipe_sha256 != the hash of config.json's recipe")
    _metrics_history(report, record, workdir, dest)
    grid = workdir / "grids" / f"iter_{config.n_iters:06d}.png"
    if grid.is_file():
        shutil.copyfile(grid, dest / "grid_final.png")
        record.inputs["grid_final"] = _file_record(grid)
    else:
        report.fail("C8", who, f"no final grid {grid}")
    return manifest


def _metrics_history(report: CollectionReport, record: RunRecord, workdir: Path,
                     dest: Path) -> None:
    """C7 on ``metrics.jsonl``; stage the canonical history line for line."""
    who, n_iters = record.cell.run_id, report.config.n_iters
    path = workdir / "metrics.jsonl"
    if not path.is_file():
        report.fail("C7", who, f"no metrics.jsonl in {workdir}")
        return
    record.inputs["metrics_jsonl"] = _file_record(path)
    records, problems = read_metrics_strict(path)
    for problem in problems:
        report.fail("C6", who, problem)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    kept = canonical_records(records)
    if not problems and len(lines) == len(records):
        # canonical_records keeps the parsed objects themselves, so the kept lines are copied
        # byte for byte rather than re-serialised.
        text_of = {id(obj): line for obj, line in zip(records, lines, strict=True)}
        body = [text_of[id(obj)] for obj in kept]
    else:
        body = [json.dumps(obj) for obj in kept]
    (dest / "metrics.canonical.jsonl").write_text("".join(line + "\n" for line in body))
    aborts = [r.get("step") for r in kept if r.get("kind") == "abort"]
    if aborts:
        report.fail("C7", who, f"abort in the canonical history at steps {aborts}")
    last = kept[-1] if kept else {}
    if last.get("kind") != "done" or last.get("step") != n_iters:
        report.fail("C7", who, f"canonical history ends with {last.get('kind')!r} at "
                    f"{last.get('step')!r}, not done at {n_iters}")
    record.row["n_skipped"] = sum(r.get("kind") == "skip" for r in kept)
    record.row["n_resumes"] = sum(r.get("kind") == "resume" for r in kept)


def _summary_checks(report: CollectionReport, record: RunRecord, summary: Json,
                    manifest: Json | None, expected: dict[str, tuple[str, str]]) -> bool:
    """C1-C6 on the plain summary; return whether its structure is usable."""
    cell, config = record.cell, report.config
    who = cell.run_id
    absent = _missing_keys(summary, SUMMARY_KEYS)
    absent += [f"run.{k}" for k in _missing_keys(summary.get("run"), SUMMARY_RUN_KEYS)]
    absent += [f"seed_lists.{k}" for k in _missing_keys(summary.get("seed_lists"),
                                                         SUMMARY_SEED_KEYS)]
    absent += [f"sampling.{k}" for k in _missing_keys(summary.get("sampling"), ("amp",))]
    if absent:
        report.fail("C6", who, f"summary.json misses keys {absent}")
        return False
    run = summary["run"]
    want = {"run_id": cell.run_id, "dataset": cell.dataset_id, "arm": cell.arm,
            "seed": cell.seed}
    for key, value in want.items():
        if run[key] != value:
            report.fail("C1", who, f"summary run.{key} {run[key]!r} != cells.csv {value!r}")
    steps = list(evaluated_steps(config.n_iters))
    if summary["checkpoint_steps"] != steps:
        report.fail("C2", who, f"checkpoint_steps {summary['checkpoint_steps']} != "
                    f"evaluated_steps({config.n_iters}) ({len(steps)} steps)")
    if summary["final_step"] != config.n_iters:
        report.fail("C2", who, f"final_step {summary['final_step']} != {config.n_iters}")
    if sorted(int(k) for k in summary["lsd_by_step"]) != steps:
        report.fail("C2", who, "lsd_by_step does not hold exactly the evaluated steps")
    if summary["sampling"]["amp"] != config.amp:
        report.fail("C3", who, f"sampling.amp {summary['sampling']['amp']!r} != --amp "
                    f"{config.amp!r}")
    digests = (summary["seed_lists"]["intermediate_sha256"], summary["seed_lists"]["final_sha256"])
    if cell.dataset_id not in expected:
        report.fail("C4", who, f"dataset {cell.dataset_id} has no row in the expected table")
    elif digests != expected[cell.dataset_id]:
        report.fail("C4", who, f"seed-list digests {tuple(d[:12] for d in digests)} != expected "
                    f"{tuple(d[:12] for d in expected[cell.dataset_id])}")
    if manifest is not None and run["config_sha256"] != manifest.get("config_sha256"):
        report.fail("C5", who, "summary run.config_sha256 != manifest config_sha256")
    if not isinstance(summary["final"], dict):
        report.fail("C6", who, "summary.json has no final block")
    return True


def _tar_members(report: CollectionReport, who: str, path: Path,
                 names: list[str]) -> dict[str, bytes] | None:
    """Return the regular files ``names`` (paths inside the tar) that the tar holds."""
    try:
        with tarfile.open(path, "r:") as tar:
            wanted = set(names)
            found: dict[str, bytes] = {}
            for member in tar.getmembers():
                name = member.name.removeprefix("./")
                if name in wanted and member.isfile():
                    handle = tar.extractfile(member)
                    if handle is not None:
                        found[name] = handle.read()
            return found
    except (tarfile.TarError, OSError) as error:
        report.fail("C8", who, f"cannot read {path}: {error}")
        return None


def _result_file_checks(report: CollectionReport, record: RunRecord, members: dict[str, bytes],
                        prefix: str, dest: Path,
                        expected: dict[str, tuple[str, str]]) -> Json | None:
    """C2-C6 on the ckpt and final records of the tar; stage them; return ``final.json``."""
    cell, config = record.cell, report.config
    who = cell.run_id
    digests = expected.get(cell.dataset_id, ("", ""))
    for step in evaluated_steps(config.n_iters):
        name = f"ckpt_{step:06d}.json"
        data = members.get(prefix + name)
        if data is None:
            report.fail("C2", who, f"the tar holds no {name}")
            continue
        try:
            ckpt = strict_json(data)
        except ValueError as error:
            report.fail("C6", who, f"{name} is not strict JSON ({error})")
            continue
        (dest / name).write_bytes(data)
        absent = _missing_keys(ckpt, CKPT_KEYS)
        if absent:
            report.fail("C6", who, f"{name} misses keys {absent}")
            continue
        if ckpt["step"] != step:
            report.fail("C2", who, f"{name} holds step {ckpt['step']}")
        if ckpt["amp"] != config.amp:
            report.fail("C3", who, f"{name} amp {ckpt['amp']!r} != {config.amp!r}")
        if cell.dataset_id in expected and ckpt["seed_list_sha256"] != digests[0]:
            report.fail("C4", who, f"{name} seed_list_sha256 != the expected intermediate list")
    data = members.get(prefix + "final.json")
    if data is None:
        report.fail("C2", who, "the tar holds no final.json")
        return None
    try:
        final = strict_json(data)
    except ValueError as error:
        report.fail("C6", who, f"final.json is not strict JSON ({error})")
        return None
    (dest / "final.json").write_bytes(data)
    absent = _missing_keys(final, FINAL_KEYS)
    if absent:
        report.fail("C6", who, f"final.json misses keys {absent}")
        return None
    if final["inception"] is None:
        report.fail("C6", who, "final.json has no Inception block (evaluated with "
                    "--skip-inception?)")
    else:
        absent = _missing_keys(final["inception"], INCEPTION_KEYS)
        if absent:
            report.fail("C6", who, f"final.json inception misses keys {absent}")
    if final["step"] != config.n_iters:
        report.fail("C2", who, f"final.json holds step {final['step']}, not {config.n_iters}")
    if final["amp"] != config.amp:
        report.fail("C3", who, f"final.json amp {final['amp']!r} != {config.amp!r}")
    if cell.dataset_id in expected and final["seed_list_sha256"] != digests[1]:
        report.fail("C4", who, "final.json seed_list_sha256 != the expected final list")
    return final


def _archive(report: CollectionReport, record: RunRecord, summary_bytes: bytes | None,
             dest: Path, expected: dict[str, tuple[str, str]]) -> Json | None:
    """C8 and the result-file checks on the run's tar; return ``final.json``."""
    cell, config = record.cell, report.config
    who = cell.run_id
    tar_path = Path(config.eval_dir) / f"{cell.run_id}{amp_suffix(config.amp)}.tar"
    record.has_tar = tar_path.is_file()
    if not record.has_tar:
        report.miss("C8", who, f"no tar {tar_path}")
        return None
    record.inputs["tar"] = _file_record(tar_path)
    prefix = f"{cell.run_id}/{metrics_dirname(config.amp)}/"
    names = [prefix + n for n in ("summary.json", "final.json", *SIDECARS)]
    names += [prefix + f"ckpt_{s:06d}.json" for s in evaluated_steps(config.n_iters)]
    members = _tar_members(report, who, tar_path, names)
    if members is None:
        return None
    in_tar = members.get(prefix + "summary.json")
    if in_tar is None:
        report.fail("C8", who, "the tar holds no summary.json")
    elif summary_bytes is not None and in_tar != summary_bytes:
        report.fail("C8", who, "the tar's summary.json differs from the plain summary")
    for name in SIDECARS:
        data = members.get(prefix + name)
        if data is None:
            report.fail("C8", who, f"the tar holds no {name}")
        else:
            (dest / name).write_bytes(data)
    return _result_file_checks(report, record, members, prefix, dest, expected)


def _index_row(record: RunRecord, summary: Json | None, final: Json | None,
               n_iters: int, amp: str) -> None:
    cell = record.cell
    steps = evaluated_steps(n_iters)
    row = record.row
    row.update({"index": cell.index, "run_id": cell.run_id, "dataset": cell.dataset_id,
                "arm": cell.arm, "seed": cell.seed, "tier": record.tier, "n_iters": n_iters,
                "amp": amp})
    if summary is not None:
        row["n_checkpoints"] = len(summary.get("checkpoint_steps") or [])
        row["checkpoint_steps"] = ";".join(str(s) for s in summary.get("checkpoint_steps") or [])
        row["final_step"] = summary.get("final_step")
        by_step = summary.get("lsd_by_step") or {}
        for step in steps:
            row[f"lsd_{step:06d}"] = by_step.get(str(step))
    if final is not None:
        row.update({"lsd_final": final.get("lsd"), "lsd_final_ckpt": final.get("intermediate_lsd"),
                    "M": final.get("M"), "M_lp": final.get("M_lp"),
                    "seed_nn_fraction": final.get("seed_nn_fraction"),
                    "D_pix": final.get("diversity_pix"), "D_lp": final.get("diversity_lp"),
                    "inherited_measured": final.get("inherited_measured"),
                    "inherited_predicted": final.get("inherited_predicted")})
        inception = final.get("inception") or {}
        for key in ("kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low", "fid_ci_high",
                    "recall", "coverage", "precision", "density"):
            row[key] = inception.get(key)
        row["fid_n_reference"] = inception.get("n_reference")


def _collect_run(report: CollectionReport, record: RunRecord, stage: Path,
                 expected: dict[str, tuple[str, str]]) -> None:
    cell, config = record.cell, report.config
    dest = stage / "runs" / cell.run_id
    dest.mkdir(parents=True, exist_ok=True)
    manifest = _run_directory(report, record, dest)

    summary_path = Path(config.eval_dir) / f"{cell.run_id}{amp_suffix(config.amp)}_summary.json"
    record.has_summary = summary_path.is_file()
    summary: Json | None = None
    summary_bytes: bytes | None = None
    if not record.has_summary:
        report.miss("C1", cell.run_id, f"no summary {summary_path}")
    else:
        record.inputs["summary"] = _file_record(summary_path)
        loaded = _load_json_file(report, summary_path, cell.run_id)
        if loaded is not None:
            summary_bytes, summary = loaded
            if _summary_checks(report, record, summary, manifest, expected):
                (dest / "summary.json").write_bytes(summary_bytes)
            else:
                summary = None
    final = _archive(report, record, summary_bytes, dest, expected)
    _index_row(record, summary, final, config.n_iters, config.amp)


# --------------------------------------------------------------------------------------------
# The collection
# --------------------------------------------------------------------------------------------


def _index_columns(n_iters: int, gate_pairs: list[tuple[int, int]]) -> list[str]:
    columns = ["index", "run_id", "dataset", "arm", "seed", "tier", "n_iters", "amp",
               "n_checkpoints", "checkpoint_steps", "final_step", "lsd_final", "lsd_final_ckpt",
               "kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low", "fid_ci_high",
               "fid_n_reference", "recall", "coverage", "precision", "density", "M", "M_lp",
               "seed_nn_fraction", "D_pix", "D_lp", "inherited_measured", "inherited_predicted",
               "n_skipped", "n_resumes"]
    columns += [f"lsd_{step:06d}" for step in evaluated_steps(n_iters)]
    for early, late in gate_pairs:
        stem = f"gate_{early:06d}_{late:06d}"
        columns += [f"{stem}_diff", f"{stem}_ci_low", f"{stem}_ci_high", f"{stem}_extend"]
    return columns


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_index(report: CollectionReport, stage: Path) -> list[str]:
    pairs = sorted({(g["early"], g["late"]) for g in report.gates})
    columns = _index_columns(report.config.n_iters, pairs)
    for gate in report.gates:
        record = next((r for r in report.runs if r.cell.run_id == gate["run_id"]), None)
        if record is None:
            continue
        stem = f"gate_{gate['early']:06d}_{gate['late']:06d}"
        record.row.update({f"{stem}_diff": gate["difference"], f"{stem}_ci_low": gate["ci_low"],
                           f"{stem}_ci_high": gate["ci_high"], f"{stem}_extend": gate["extend"]})
    with (stage / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for record in report.runs:
            writer.writerow([_cell_text(record.row.get(c)) for c in columns])
    return columns


def _collection_json(report: CollectionReport, columns: list[str]) -> Json:
    config = report.config
    return {
        "created": datetime.now(UTC).isoformat(),
        "tool": "python -m ihdm.cli.collect_results (T5.2)",
        "git_sha": _git_sha(),
        "complete": report.verdict == "COMPLETE",
        "verdict": report.verdict,
        "allow_missing": config.allow_missing,
        "n_iters": config.n_iters,
        "amp": config.amp,
        "evaluated_steps": list(evaluated_steps(config.n_iters)),
        "required_gate_pairs": [list(p) for p in required_gate_pairs(config.n_iters)],
        "gate_cells": list(config.gate_cells),
        "inputs": {
            "eval_dir": str(config.eval_dir),
            "gate_dir": str(config.gate_dir),
            "run_root": str(config.run_root),
            "cells": _file_record(config.cells),
            "expected_seed_lists": _file_record(config.expected_seed_lists),
            "runs": {r.cell.run_id: r.inputs for r in report.runs},
            "gates_ignored": list(report.ignored),
        },
        "gates": report.gates,
        "index_columns": columns,
        "checks": [c.to_json() for c in report.checks.values()],
        "notes": [
            "metrics.canonical.jsonl is metrics.jsonl after validate_run.canonical_records; the "
            "raw file stays in the archives and its sha256 is under inputs.runs.<run_id>."
            "metrics_jsonl.",
            "Samples are not copied; the tars under inputs.runs.<run_id>.tar are the archive.",
        ],
    }


def _stage_dir(out: Path) -> Path:
    stage = out.parent / f".{out.name}.partial-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    return stage


def collect(config: CollectConfig) -> CollectionReport:
    """Check every input and publish ``config.out`` if the collection may be published.

    The folder is published when every check passes, or, with ``allow_missing``, in any case
    (marked ``"complete": false`` unless every check passes). Otherwise nothing is left on disk.

    Parameters
    ----------
    config : CollectConfig
        The request.

    Returns
    -------
    CollectionReport
        The checks, the per-run records and where the folder was published.

    Raises
    ------
    CollectError
        If ``out`` exists, ``n_iters`` is not a positive multiple of 5,000, the precision is
        unknown, or the cell or seed-list table is unusable.
    """
    out = Path(config.out).resolve()
    if out.exists():
        raise CollectError(f"{out} exists; move it away first (results are never overwritten)")
    if config.n_iters < EVAL_STRIDE or config.n_iters % EVAL_STRIDE:
        raise CollectError(f"n_iters {config.n_iters} is not a positive multiple of "
                           f"{EVAL_STRIDE}")
    amp_suffix(config.amp)
    try:
        cells = read_cells(config.cells)
    except CheckArrayError as error:
        raise CollectError(str(error)) from error
    expected = read_expected_seed_lists(config.expected_seed_lists)
    tiers = _read_tiers(config.cells)

    report = CollectionReport(config=config,
                              checks={k: CheckResult(k, t) for k, t in CHECKS.items()})
    report.runs = [RunRecord(cell=c, tier=tiers.get(c.run_id, "")) for c in cells]
    records = {r.cell.run_id: r for r in report.runs}
    _check_cell_table(report, cells)
    _unknown_summaries(report, records)

    stage = _stage_dir(out)
    try:
        _collect_gates(report, stage, cells, expected, records)
        for record in report.runs:
            _collect_run(report, record, stage, expected)
        columns = _write_index(report, stage)
        write_json(stage / "collection.json", _collection_json(report, columns))
        if report.verdict == "COMPLETE" or config.allow_missing:
            os.rename(stage, out)
            report.written = out
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def _unknown_summaries(report: CollectionReport, records: dict[str, RunRecord]) -> None:
    """A summary of the collected precision for a run that is not a cell is a C1 problem."""
    tail = f"{amp_suffix(report.config.amp)}_summary.json"
    eval_dir = Path(report.config.eval_dir)
    if not eval_dir.is_dir():
        return
    for path in sorted(eval_dir.glob(f"*{tail}")):
        run_id = path.name[: -len(tail)]
        if report.config.amp == "off" and _AMP_TAIL.search(run_id):
            continue
        if run_id not in records:
            report.fail("C1", path.name, "a summary for a run that is not a cell of cells.csv")


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def _shown(items: list[str], limit: int) -> list[str]:
    lines = items[:limit]
    if len(items) > limit:
        lines.append(f"... (+{len(items) - limit} more)")
    return lines


def format_report(report: CollectionReport, limit: int = 6) -> str:
    """Render the per-run table, the gates, the checks and the verdict line.

    Parameters
    ----------
    report : CollectionReport
        From :func:`collect`.
    limit : int
        Problems and missing items printed per check (the full lists are in
        ``collection.json``).

    Returns
    -------
    str
        The text printed by the CLI.
    """
    config = report.config
    steps = evaluated_steps(config.n_iters)
    lines = [
        f"collect_results: {len(report.runs)} cells of {config.cells}; n_iters {config.n_iters} "
        f"({len(steps)} steps {steps[0]}..{steps[-1]}); amp {config.amp}; "
        f"allow_missing={config.allow_missing}",
        f"eval dir {config.eval_dir}; gate dir {config.gate_dir}; run root {config.run_root}",
        "",
        f"{'idx':>3} {'run_id':<20} {'run':<4} {'summary':<7} {'tar':<4} {'skip':>4} gates",
    ]
    for r in report.runs:
        flag = {True: "ok", False: "-"}
        lines.append(f"{r.cell.index:>3} {r.cell.run_id:<20} {flag[r.has_run_dir]:<4} "
                     f"{flag[r.has_summary]:<7} {flag[r.has_tar]:<4} "
                     f"{_cell_text(r.row.get('n_skipped')) or '-':>4} "
                     f"{','.join(r.gates) or '-'}")
    lines.append("")
    lines.append(f"gates collected: {len(report.gates)}"
                 + (f" (ignored: {', '.join(report.ignored)})" if report.ignored else ""))
    for g in report.gates:
        lines.append(f"  {g['dest']:<48} <- {Path(g['source']).name}: LSD {g['lsd_a']:.4f} -> "
                     f"{g['lsd_b']:.4f}, diff {g['difference']:+.5f} "
                     f"[{g['ci_low']:+.5f}, {g['ci_high']:+.5f}], extend={g['extend']}")
    lines.append("")
    lines.append("checks:")
    for check in report.checks.values():
        lines.append(f"  {check.id} {check.verdict:<7} {check.title} "
                     f"({len(check.problems)} problems, {len(check.missing)} missing)")
        lines += [f"       problem: {p}" for p in _shown(check.problems, limit)]
        lines += [f"       missing: {m}" for m in _shown(check.missing, limit)]
    lines.append("")
    if report.written is not None:
        state = "complete" if report.verdict == "COMPLETE" else "PARTIAL (--allow-missing)"
        where = f"written ({state}): {report.written}"
    else:
        where = "nothing written"
    lines.append(f"VERDICT: {report.verdict} ({report.n_problems} problems, {report.n_missing} "
                 f"missing); {where}")
    return "\n".join(lines)
