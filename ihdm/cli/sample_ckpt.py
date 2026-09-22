"""Draw samples from an EMA checkpoint, starting from given seed images.

Imperative shell of :mod:`ihdm.sampling` and the only sampling entry point of the experiment
(``04-run-artifacts.md`` §4): resolve the run's config and checkpoint, take the seeds from the
training split, from the 40 held-out seed subjects or from an array on disk, run the reverse
chain in batches, and write ``samples.npy``, ``seeds.npy``, ``seed_idx.npy``, ``request.json``
and ``preview.png`` into the output directory.

Run as ``python -m ihdm.cli.sample_ckpt --run <workdir> --ckpt ema_iter_020000.pt
--source seed --n-seeds 40 --n-per-seed 50 --out <dir>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ihdm.paths import repo_root
from ihdm.sampling.chain import SampleRequest, resolve_request, sample_from_seeds
from ihdm.sampling.errors import SamplingError
from ihdm.sampling.loader import (
    checkpoint_sha256,
    load_checkpoint,
    load_ema_model,
    load_run_config,
    resolve_checkpoint_path,
)
from ihdm.sampling.seeds import load_seed_images

__all__ = ["build_parser", "main"]

logger = logging.getLogger(__name__)

#: Preview layout: at most this many seed rows and this many sample columns beside the seed.
PREVIEW_ROWS: int = 8
PREVIEW_COLS: int = 4


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command.

    Returns
    -------
    argparse.ArgumentParser
        The parser of the CLI frozen in ``04-run-artifacts.md`` §4, plus an additive
        ``--device`` override.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="run directory (the workdir)")
    parser.add_argument(
        "--ckpt", required=True, help="checkpoint file name under <run>/checkpoints, or a path"
    )
    parser.add_argument(
        "--source", required=True, choices=("train", "seed", "file"), help="the seed source"
    )
    parser.add_argument("--seeds-file", type=Path, default=None, help="uint8 .npy for source file")
    parser.add_argument("--n-seeds", required=True, type=int, help="how many seed images")
    parser.add_argument("--n-per-seed", required=True, type=int, help="samples drawn per seed")
    parser.add_argument("--start-level", type=int, default=None, help="start level, default K")
    parser.add_argument("--delta", type=float, default=None, help="noise sd, default 1.25 sigma")
    parser.add_argument("--batch", type=int, default=64, help="images per network batch")
    parser.add_argument("--rng-seed", type=int, default=0, help="seed of the sampling RNG")
    parser.add_argument("--amp", action="store_true", help="run the network under autocast")
    parser.add_argument(
        "--device", default=None, help="sampling device (default: the run config's device)"
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    return parser


def _git_sha(repo: Path) -> str:
    """Return the repository's HEAD sha, or ``"unknown"`` when git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _resolve_device(args: argparse.Namespace, config: Any) -> torch.device:
    """Return the sampling device, falling back to CPU when the config asks for an absent GPU."""
    wanted = torch.device(args.device) if args.device else torch.device(str(config.device))
    if wanted.type == "cuda" and not torch.cuda.is_available():
        logger.warning("config asks for %s but CUDA is unavailable; sampling on the CPU", wanted)
        return torch.device("cpu")
    return wanted


def _load_seeds(args: argparse.Namespace, config: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(seeds uint8 (N, H, W), idx int64 (N,))`` for the requested source."""
    if args.source == "file":
        if args.seeds_file is None:
            raise SamplingError("--source file requires --seeds-file")
        seeds = np.asarray(np.load(args.seeds_file))
        if seeds.ndim != 3:
            raise SamplingError(f"{args.seeds_file}: expected (N, H, W), got {seeds.shape}")
        seeds = seeds[: args.n_seeds].astype(np.uint8)
        return seeds, np.arange(seeds.shape[0], dtype=np.int64)
    dataset_root = Path(config.data.root) / str(config.data.dataset)
    return load_seed_images(dataset_root, args.source, args.n_seeds, args.rng_seed)


def _write_preview(path: Path, seeds_u8: np.ndarray, samples: np.ndarray) -> Path:
    """Write the preview PNG: one row per seed, the seed first and then its samples."""
    from torchvision.utils import make_grid, save_image

    n_rows = min(PREVIEW_ROWS, seeds_u8.shape[0])
    n_cols = min(PREVIEW_COLS, samples.shape[1])
    panel = []
    for row in range(n_rows):
        panel.append(seeds_u8[row].astype(np.float32) / 255.0)
        panel.extend(samples[row, col] for col in range(n_cols))
    tensor = torch.from_numpy(np.stack(panel).astype(np.float32))[:, None]
    save_image(make_grid(tensor, nrow=n_cols + 1, padding=2), path)
    return path


def _seed_multiplicity(seed_idx: np.ndarray) -> dict[str, Any]:
    """Describe how often each dataset index was used as a seed.

    The memorisation endpoint of ``05-metrics.md`` §4 draws 5 000 training seeds from a 3 200
    image split, so indices repeat by design; its ``seed_nn_fraction`` needs to know which
    positions share a seed.

    Parameters
    ----------
    seed_idx : np.ndarray
        The dataset indices of the seeds, one per row of ``seeds.npy``.

    Returns
    -------
    dict[str, Any]
        ``n_seeds``, ``n_unique``, ``max_count``, ``drawn_with_replacement`` and the counts of
        the repeated indices.
    """
    values, counts = np.unique(np.asarray(seed_idx, dtype=np.int64), return_counts=True)
    repeated = {int(v): int(c) for v, c in zip(values, counts, strict=True) if c > 1}
    return {
        "n_seeds": int(seed_idx.shape[0]),
        "n_unique": int(values.size),
        "max_count": int(counts.max()) if counts.size else 0,
        "drawn_with_replacement": bool(values.size < seed_idx.shape[0]),
        "repeated_idx_counts": repeated,
    }


def _request_record(
    args: argparse.Namespace,
    config: Any,
    request: SampleRequest,
    resolved: tuple[float, int, bool],
    payload: dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
    seeds_u8: np.ndarray,
    seed_idx: np.ndarray,
    samples: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    """Build the ``request.json`` record: every argument plus the provenance of the run."""
    delta, start_level, prior_noise = resolved
    n_samples = int(samples.shape[0] * samples.shape[1])
    return {
        "args": {
            "run": str(Path(args.run).resolve()),
            "ckpt": args.ckpt,
            "source": args.source,
            "seeds_file": None if args.seeds_file is None else str(args.seeds_file),
            "n_seeds": int(args.n_seeds),
            "n_per_seed": int(args.n_per_seed),
            "start_level": args.start_level,
            "delta": args.delta,
            "batch": int(args.batch),
            "rng_seed": int(args.rng_seed),
            "amp": bool(args.amp),
            "device": args.device,
            "out": str(Path(args.out).resolve()),
        },
        "request": {
            "n_per_seed": request.n_per_seed,
            "delta": delta,
            "start_level": start_level,
            "prior_noise": prior_noise,
            "batch_size": request.batch_size,
            "rng_seed": request.rng_seed,
            "amp": request.amp,
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": checkpoint_sha256(ckpt_path),
            "step": payload.get("step"),
            "run_id": payload.get("run_id"),
            "config_sha256": payload.get("config_sha256"),
            "schedule_name": (payload.get("schedule") or {}).get("name"),
            "schedule_sha256": (payload.get("schedule") or {}).get("sha256"),
        },
        "shapes": {
            "seeds": list(seeds_u8.shape),
            "samples": list(samples.shape),
            "n_samples": n_samples,
        },
        "seeds": _seed_multiplicity(seed_idx),
        "env": {
            "git_sha": _git_sha(repo_root()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "created": datetime.now(UTC).isoformat(),
        },
        "timing": {
            "wall_s": elapsed,
            "samples_per_s": (n_samples / elapsed) if elapsed > 0 else None,
            "s_per_chain": (elapsed / n_samples) if n_samples else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the sampler and write the five artefacts.

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
    SamplingError
        On any unusable run, checkpoint, seed source or request.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)

    config = load_run_config(args.run)
    ckpt_path = resolve_checkpoint_path(args.run, args.ckpt)
    device = _resolve_device(args, config)

    request = SampleRequest(
        n_per_seed=args.n_per_seed,
        delta=args.delta,
        start_level=args.start_level,
        batch_size=args.batch,
        rng_seed=args.rng_seed,
        amp=args.amp,
    )
    resolved = resolve_request(request, config)
    seeds_u8, seed_idx = _load_seeds(args, config)
    payload = load_checkpoint(ckpt_path, device)
    model = load_ema_model(ckpt_path, config, device, payload=payload)
    # The weights now live in the model; drop the second copy before the chain allocates.
    payload = {key: value for key, value in payload.items() if key != "ema_state_dict"}

    start = time.perf_counter()
    samples = sample_from_seeds(model, config, seeds_u8, request, device)
    elapsed = time.perf_counter() - start

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "samples.npy", np.rint(samples * 255.0).clip(0, 255).astype(np.uint8))
    np.save(out_dir / "seeds.npy", seeds_u8.astype(np.uint8))
    np.save(out_dir / "seed_idx.npy", np.asarray(seed_idx, dtype=np.int64))
    record = _request_record(
        args, config, request, resolved, payload, ckpt_path, device, seeds_u8, seed_idx,
        samples, elapsed,
    )
    (out_dir / "request.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    _write_preview(out_dir / "preview.png", seeds_u8, samples)

    n_samples = record["shapes"]["n_samples"]
    print(
        f"OK {record['checkpoint']['run_id']} step={record['checkpoint']['step']} "
        f"source={args.source} seeds={seeds_u8.shape[0]} n_per_seed={request.n_per_seed} "
        f"start_level={resolved[1]} delta={resolved[0]:.4g} prior_noise={resolved[2]} "
        f"device={device} {n_samples} samples in {elapsed:.1f} s "
        f"({n_samples / elapsed if elapsed else float('nan'):.2f} samples/s, "
        f"{elapsed / n_samples if n_samples else float('nan'):.3f} s/chain) -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SamplingError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(2) from error
