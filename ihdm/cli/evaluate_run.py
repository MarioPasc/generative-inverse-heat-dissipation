"""Evaluate one training run: draw the sample sets and write the result files of `05` §9.

Imperative shell of :mod:`ihdm.metrics.run_eval` and the command the evaluation array of T5.1
runs once per training run:

``python -m ihdm.cli.evaluate_run --run <workdir> [--ckpts all|final|<list>] [--gate 35000,40000]``

``--ckpts all`` evaluates the eight steps of D16 (5 000, 10 000, …, 40 000) that the run holds;
``--gate a,b`` writes the paired plateau statistic of D10/D17 to ``metrics/gate.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ihdm.metrics.errors import MetricError
from ihdm.metrics.run_eval import (
    DEFAULT_SAMPLE_BATCH,
    N_SEEDS_FINAL,
    N_SEEDS_INTERMEDIATE,
    EvalRequest,
    evaluate_run,
)

__all__ = ["build_parser", "main", "request_from_args"]

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command.

    Returns
    -------
    argparse.ArgumentParser
        The parser of ``M4-metrics/T4.3`` §1 with the D16/D17 defaults.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="run directory (the workdir)")
    parser.add_argument(
        "--ckpts",
        default="all",
        help="'all' (the eight evaluated steps), 'final', or a comma-separated list of steps",
    )
    parser.add_argument(
        "--n-lsd", type=int, default=N_SEEDS_INTERMEDIATE,
        help="seeds of the per-checkpoint LSD set; a prefix of the frozen 500 list",
    )
    parser.add_argument(
        "--n-final", type=int, default=N_SEEDS_FINAL,
        help="seeds of the shared final set; a prefix of the frozen 2000 list",
    )
    parser.add_argument("--n-seeds", type=int, default=40, help="held-out seed subjects")
    parser.add_argument("--n-per-seed", type=int, default=50, help="samples per held-out seed")
    parser.add_argument(
        "--sample-batch", type=int, default=DEFAULT_SAMPLE_BATCH,
        help="sampling batch; pinned per run because samples reproduce only at a fixed batch",
    )
    parser.add_argument("--fid-batch", type=int, default=64, help="Inception forward batch")
    parser.add_argument(
        "--fid-boot", type=int, default=200, help="resamples of the KID/FID bootstrap"
    )
    parser.add_argument(
        "--gate-boot", type=int, default=1000, help="resamples of the plateau-gate bootstrap"
    )
    parser.add_argument("--k", type=int, default=5, help="neighbours of recall/coverage")
    parser.add_argument(
        "--skip-inception", action="store_true", help="do not compute FID, KID, recall, coverage"
    )
    parser.add_argument(
        "--a0-final-lsd", type=float, default=None,
        help="final LSD of the A0 run with the same seed; the threshold of T_tau",
    )
    parser.add_argument(
        "--gate", default=None,
        help="two checkpoint steps 'a,b' (earlier first) for the paired plateau gate",
    )
    parser.add_argument("--device", default=None, help="torch device (default: the run's config)")
    parser.add_argument(
        "--force", action="store_true", help="redraw samples and rewrite result files"
    )
    return parser


def _parse_gate(value: str | None) -> tuple[int, int] | None:
    """Return the two steps of ``--gate``, or ``None``."""
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise MetricError(f"--gate wants two steps 'a,b', got {value!r}")
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError as error:
        raise MetricError(f"--gate wants two integer steps, got {value!r}") from error


def request_from_args(args: argparse.Namespace) -> EvalRequest:
    """Build the :class:`EvalRequest` of a parsed command line.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    EvalRequest
        The request.

    Raises
    ------
    MetricError
        If ``--gate`` is malformed.
    """
    return EvalRequest(
        run=Path(args.run),
        ckpts=str(args.ckpts),
        n_lsd=int(args.n_lsd),
        n_final=int(args.n_final),
        n_seeds=int(args.n_seeds),
        n_per_seed=int(args.n_per_seed),
        sample_batch=int(args.sample_batch),
        fid_batch=int(args.fid_batch),
        skip_inception=bool(args.skip_inception),
        device=args.device,
        force=bool(args.force),
        a0_final_lsd=None if args.a0_final_lsd is None else float(args.a0_final_lsd),
        gate=_parse_gate(args.gate),
        n_boot_inception=int(args.fid_boot),
        n_boot_gate=int(args.gate_boot),
        k=int(args.k),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and print a one-line summary.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` on success.

    Raises
    ------
    MetricError
        On any unusable run, dataset, checkpoint or metric.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    request = request_from_args(args)

    if not request.skip_inception:
        from ihdm.metrics.inception import inception_weights_path

        print(f"inception weights path: {inception_weights_path()}")

    summary = evaluate_run(request)
    identity = summary["run"]
    steps = summary["checkpoint_steps"]
    final = summary.get("final", {})
    inception = final.get("inception") or {}
    print(
        f"OK {identity['run_id']} dataset={identity['dataset']} arm={identity['arm']} "
        f"seed={identity['seed']} steps={steps} "
        f"lsd={ {k: round(v, 4) for k, v in summary['lsd_by_step'].items()} } "
        f"t_tau={summary['t_tau']} "
        f"final_lsd={final.get('lsd')} M={final.get('M')} "
        f"D_pix={final.get('diversity_pix')} "
        f"kid={inception.get('kid')} fid={inception.get('fid')} "
        f"n_ref={inception.get('n_reference')} "
        f"-> {Path(request.run) / 'metrics'}"
    )
    if summary.get("gate"):
        gate = summary["gate"]
        print(
            f"GATE {gate['steps']} difference={gate['difference']:.5f} "
            f"CI=[{gate['ci_low']:.5f}, {gate['ci_high']:.5f}] extend={gate['extend']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MetricError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(2) from error
