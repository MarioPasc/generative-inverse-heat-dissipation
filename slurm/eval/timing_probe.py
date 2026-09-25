"""A100 timing probe of T5.1: s/chain per AMP mode and batch, LSD per mode, Inception time.

Run by ``slurm/eval/timing.sbatch`` on a *shadow* run directory (links to a real run's
checkpoints, a copied ``config.json`` whose ``data.root`` points at a shadow dataset on local
disk), so it writes nothing outside ``$LOCALSCRATCH``. The phases, in priority order so that a
TIMEOUT loses the least important numbers:

A. the D16 LSD set (500 frozen training seeds, ``rng_seed`` 2026, batch 32) at one checkpoint
   under each AMP mode, drawn through ``run_eval.draw_set`` so the plateau-gate command that
   follows reuses it; s/chain, LSD, the paired AMP-minus-off LSD difference with its bootstrap
   CI, and pixel differences to the fp32 set;
B. Inception feature time for 2 000 samples and for the 800-image reference;
C. s/chain at batch 64 on a prefix of the same seeds, per mode.

``--phases`` selects a subset, so the worker can run the plateau gate between B and C. A
deadline guard skips a phase whose projected cost no longer fits. The JSON record is
rewritten after every phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ihdm.metrics.run_eval import (
    SAMPLING_RNG_INTERMEDIATE,
    checkpoint_table,
    draw_set,
    ensure_seed_lists,
    load_views,
)
from ihdm.metrics.spectral import lsd
from ihdm.sampling.chain import SampleRequest, sample_from_seeds
from ihdm.sampling.loader import load_checkpoint, load_ema_model, load_run_config
from ihdm.sampling.seeds import load_seed_images
from ihdm.stats.bootstrap import paired_lsd_gate

logger = logging.getLogger("timing_probe")

_AMP_DTYPE: dict[str, str | None] = {"off": None, "fp16": "float16", "bf16": "bfloat16"}
#: Pessimistic s/chain used to project a phase before the mode has been measured (T3.2: 3.21).
_PESSIMISTIC_S_PER_CHAIN: float = 3.6


class Probe:
    """Holds the shared state of the probe and writes the record after every phase."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.record: dict[str, Any] = {"phases": {}, "skipped": []}
        self.run = Path(args.run).resolve()
        self.config = load_run_config(self.run)
        self.dataset_root = Path(self.config.data.root) / str(self.config.data.dataset)
        self.lists = ensure_seed_lists(self.dataset_root)
        self._check_lists(args.expected)
        self.views = load_views(self.dataset_root)
        table = checkpoint_table(self.run)
        self.step = int(args.step)
        self.ckpt = table[self.step]
        self.device = "cuda"
        start = time.perf_counter()
        payload = load_checkpoint(self.ckpt, self.device)
        self.model = load_ema_model(self.ckpt, self.config, self.device, payload=payload)
        load_s = time.perf_counter() - start
        self.sets: dict[str, Any] = {}
        self.s_per_chain_b32: dict[str, float] = {}
        self.record["setup"] = {
            "run": str(self.run),
            "step": self.step,
            "checkpoint": str(self.ckpt),
            "dataset_root": str(self.dataset_root),
            "intermediate_sha256": self.lists.intermediate.sha256,
            "final_sha256": self.lists.final.sha256,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "model_load_s": round(load_s, 3),
        }
        self.save()

    def _check_lists(self, expected_csv: Path | None) -> None:
        """Refuse to time anything on seed lists other than the ones the real array will use."""
        if expected_csv is None:
            return
        with Path(expected_csv).open() as handle:
            rows = {row["dataset"]: row for row in csv.DictReader(handle)}
        want = rows[str(self.config.data.dataset)]
        got = (self.lists.intermediate.sha256, self.lists.final.sha256)
        if got != (want["intermediate_sha256"], want["final_sha256"]):
            raise SystemExit(f"FAIL seed lists {got} differ from {expected_csv}: {want}")
        logger.info("seed lists match %s: %s", expected_csv, got)

    def save(self) -> None:
        """Rewrite the JSON record."""
        Path(self.args.out).write_text(json.dumps(self.record, indent=2, sort_keys=True) + "\n")

    def fits(self, label: str, projected_s: float) -> bool:
        """Return whether ``projected_s`` fits before the deadline; record a skip otherwise."""
        remaining = float(self.args.deadline) - time.time()
        if projected_s <= remaining - float(self.args.reserve):
            return True
        logger.warning("skipping %s: projected %.0f s, %.0f s left", label, projected_s, remaining)
        self.record["skipped"].append(
            {"phase": label, "projected_s": round(projected_s, 1), "remaining_s": round(remaining)}
        )
        self.save()
        return False

    def warm_up(self, modes: list[str]) -> None:
        """Run one short chain per mode so kernel selection is not billed to phase A."""
        seeds = self.views.reference[:4]
        for mode in modes:
            request = SampleRequest(
                n_per_seed=1, batch_size=4, rng_seed=1, amp=mode != "off",
                amp_dtype=_AMP_DTYPE[mode],
            )
            start = time.perf_counter()
            sample_from_seeds(self.model, self.config, seeds, request, self.device)
            torch.cuda.synchronize()
            logger.info("warm-up %s: %.1f s", mode, time.perf_counter() - start)

    def phase_a(self, mode: str) -> None:
        """Draw the 500-seed LSD set under ``mode`` at batch 32 and score it."""
        n = int(self.args.n)
        guess = self.s_per_chain_b32.get("off", _PESSIMISTIC_S_PER_CHAIN)
        if not self.fits(f"A:{mode}", n * guess * 1.05):
            return
        torch.cuda.reset_peak_memory_stats()
        drawn = draw_set(
            "lsd", self.run, self.step, self.ckpt, self.config, self.dataset_root, "file",
            n, 1, SAMPLING_RNG_INTERMEDIATE, int(self.args.batch), self.device,
            lambda: self.model, idx_file=self.lists.intermediate.path, amp=mode,
        )
        s_chain = drawn.elapsed_s / n
        self.s_per_chain_b32[mode] = s_chain
        self.sets[mode] = drawn
        result = lsd(drawn.flat, self.views.reference)
        entry: dict[str, Any] = {
            "mode": mode,
            "batch": int(self.args.batch),
            "n_chains": n,
            "elapsed_s": round(drawn.elapsed_s, 2),
            "s_per_chain": round(s_chain, 4),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "lsd": result.lsd,
            "lsd_octaves": dict(result.octaves),
            "variance_ratio": result.variance_ratio,
            "samples_dir": str(drawn.directory),
        }
        if mode != "off" and "off" in self.sets:
            entry["vs_off"] = self._versus_off(mode)
        self.record["phases"][f"A_{mode}"] = entry
        logger.info("A %s: %.3f s/chain, LSD %.4f", mode, s_chain, result.lsd)
        self.save()

    def _versus_off(self, mode: str) -> dict[str, Any]:
        """Return the paired difference between ``mode``'s set and the fp32 set."""
        off, amp = self.sets["off"], self.sets[mode]
        gate = paired_lsd_gate(
            off.flat, amp.flat, self.views.reference, n_boot=1000, rng_seed=0
        ).to_json()
        pixel = np.abs(off.flat.astype(np.int16) - amp.flat.astype(np.int16))
        return {
            "lsd_off": gate["lsd_a"],
            "lsd_amp": gate["lsd_b"],
            "delta_lsd_off_minus_amp": gate["difference"],
            "abs_delta_lsd": abs(float(gate["lsd_a"]) - float(gate["lsd_b"])),
            "pixel_mean_abs_u8": float(pixel.mean()),
            "pixel_max_abs_u8": int(pixel.max()),
            "pixel_fraction_changed": float((pixel > 0).mean()),
        }

    def phase_b(self) -> None:
        """Time Inception on 2 000 samples and on the reference split."""
        from ihdm.metrics.inception import build_extractor, inception_features

        if not self.fits("B:inception", 600.0):
            return
        source = next((self.sets[m] for m in ("off", "fp16", "bf16") if m in self.sets), None)
        stack = self.views.reference if source is None else source.flat
        reps = int(np.ceil(int(self.args.n_inception) / stack.shape[0]))
        images = np.concatenate([stack] * reps)[: int(self.args.n_inception)]
        start = time.perf_counter()
        build_extractor(self.device)
        build_s = time.perf_counter() - start
        inception_features(images[:64], device=self.device, batch=int(self.args.fid_batch))
        start = time.perf_counter()
        features = inception_features(images, device=self.device, batch=int(self.args.fid_batch))
        samples_s = time.perf_counter() - start
        start = time.perf_counter()
        inception_features(self.views.reference, device=self.device, batch=int(self.args.fid_batch))
        reference_s = time.perf_counter() - start
        self.record["phases"]["B_inception"] = {
            "n_samples": int(images.shape[0]),
            "fid_batch": int(self.args.fid_batch),
            "extractor_build_s": round(build_s, 2),
            "samples_s": round(samples_s, 2),
            "reference_n": int(self.views.reference.shape[0]),
            "reference_s": round(reference_s, 2),
            "finite": bool(np.isfinite(features).all()),
        }
        logger.info("B inception: %d samples in %.1f s", images.shape[0], samples_s)
        self.save()

    def phase_c(self, mode: str) -> None:
        """Time ``n_b64`` chains at batch 64 under ``mode`` (not cached, not scored)."""
        n = int(self.args.n_b64)
        guess = self.s_per_chain_b32.get(mode, _PESSIMISTIC_S_PER_CHAIN)
        if not self.fits(f"C:{mode}", n * guess * 1.1):
            return
        seeds, _ = load_seed_images(
            self.dataset_root, "file", n, SAMPLING_RNG_INTERMEDIATE,
            idx_file=self.lists.intermediate.path,
        )
        request = SampleRequest(
            n_per_seed=1, batch_size=64, rng_seed=SAMPLING_RNG_INTERMEDIATE,
            amp=mode != "off", amp_dtype=_AMP_DTYPE[mode],
        )
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        sample_from_seeds(self.model, self.config, seeds, request, self.device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.record["phases"][f"C_{mode}"] = {
            "mode": mode,
            "batch": 64,
            "n_chains": n,
            "elapsed_s": round(elapsed, 2),
            "s_per_chain": round(elapsed / n, 4),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        logger.info("C %s: %.3f s/chain at batch 64", mode, elapsed / n)
        self.save()


def main(argv: list[str] | None = None) -> int:
    """Run the three phases and return 0 (skipped phases are recorded, not failures)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--step", type=int, default=7500)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n-b64", type=int, default=128)
    parser.add_argument("--n-inception", type=int, default=2000)
    parser.add_argument("--fid-batch", type=int, default=64)
    parser.add_argument("--modes", default="off,fp16,bf16")
    parser.add_argument("--deadline", type=float, required=True, help="epoch seconds")
    parser.add_argument("--reserve", type=float, default=300.0, help="seconds kept free")
    parser.add_argument(
        "--expected", type=Path, default=None, help="expected_seed_lists.csv to check against"
    )
    parser.add_argument("--phases", default="ABC", help="subset of A, B, C to run, in order")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    probe = Probe(args)
    probe.warm_up(modes)
    if "A" in args.phases:
        for mode in modes:
            probe.phase_a(mode)
    if "B" in args.phases:
        probe.phase_b()
    if "C" in args.phases:
        for mode in modes:
            probe.phase_c(mode)
    print(json.dumps({key: value for key, value in probe.record.items() if key != "setup"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
