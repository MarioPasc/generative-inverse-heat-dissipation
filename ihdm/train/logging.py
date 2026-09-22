"""Sanity logging: ``metrics.jsonl`` and tensorboard.

Frozen contract: ``docs/SPECIFICATIONS/04-run-artifacts.md`` §3.2. One JSON object per line, one
line per event, flushed immediately so a killed run keeps everything it had written.

The per-octave decomposition answers the question the experiment is about: the two knobs change
*where in the spectrum* the model spends its capacity, so a per-step scalar loss is not enough.
Each sample of a batch was trained at one blur level ``k``; its blur scale is
``sigma_B = blur_schedule[k]`` pixels, and the octave bins of ``05-metrics.md`` §1
(``0.5-1, 1-2, ..., 64-96``) are reused here, read as sigma_B in pixels. A level whose sigma_B
falls in ``[a, b)`` is counted in bin ``"a-b"``; the last bin is closed at 96. Bins with no sample
in the window are ``null``, never 0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "OCTAVE_BIN_EDGES",
    "OCTAVE_BIN_NAMES",
    "MetricsLogger",
    "global_grad_norm",
    "octave_bin",
]

# Octave bin edges in sigma_B pixels (05-metrics.md §1, reused for the loss decomposition).
OCTAVE_BIN_EDGES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 96.0)
OCTAVE_BIN_NAMES: tuple[str, ...] = (
    "0.5-1",
    "1-2",
    "2-4",
    "4-8",
    "8-16",
    "16-32",
    "32-64",
    "64-96",
)


def octave_bin(sigma_b: float) -> str | None:
    """Return the name of the octave bin holding ``sigma_b``, or ``None`` if outside.

    Parameters
    ----------
    sigma_b : float
        A blur scale in pixels.

    Returns
    -------
    str | None
        One of :data:`OCTAVE_BIN_NAMES`, or ``None`` when ``sigma_b`` is below 0.5 or
        above 96 (levels outside the bins of ``05-metrics.md`` §1).
    """
    if sigma_b < OCTAVE_BIN_EDGES[0] or sigma_b > OCTAVE_BIN_EDGES[-1]:
        return None
    # The last bin is closed at 96; every other bin is [a, b).
    index = int(np.searchsorted(OCTAVE_BIN_EDGES, sigma_b, side="right")) - 1
    index = min(index, len(OCTAVE_BIN_NAMES) - 1)
    return OCTAVE_BIN_NAMES[index]


def global_grad_norm(parameters: Any) -> float:
    """Return the 2-norm of the concatenated gradients of ``parameters``.

    Called from ``train.py`` after the released ``optimize_fn`` has run, because
    ``scripts/losses.py: optimization_manager`` discards the pre-clip norm returned by
    ``torch.nn.utils.clip_grad_norm_`` and that file may only be edited at the frozen hook
    line. The value reported therefore saturates at ``config.optim.grad_clip``: a value equal
    to ``grad_clip`` means the step was clipped.

    Parameters
    ----------
    parameters : Iterable[torch.nn.Parameter]
        The model parameters, immediately after the optimiser step.

    Returns
    -------
    float
        The gradient norm, or 0.0 when no parameter carries a gradient.
    """
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


class MetricsLogger:
    """Writes ``metrics.jsonl`` and the tensorboard scalars of one run.

    :meth:`log_train` is called on **every** training step and emits a line only every
    ``config.training.log_every`` steps: the per-octave loss is an average over the window
    since the last line, so every step's per-sample losses have to reach the logger.

    Parameters
    ----------
    workdir : Path
        The run directory. ``metrics.jsonl`` is appended to; ``tensorboard/`` is created.
    config : ml_collections.ConfigDict
        Must provide ``training.log_every``, ``training.batch_size``, ``model.blur_schedule``
        and ``device``.
    """

    def __init__(self, workdir: Path, config: Any) -> None:
        from torch.utils import tensorboard  # heavy import, only needed for a real run

        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._path = self.workdir / "metrics.jsonl"
        self._writer = tensorboard.SummaryWriter(str(self.workdir / "tensorboard"))

        self._log_every = int(config.training.log_every)
        self._batch_size = int(config.training.batch_size)
        self._schedule = np.asarray(config.model.blur_schedule, dtype=np.float64)
        self._cuda = torch.cuda.is_available() and not str(config.device).startswith("cpu")

        self._t_start = time.perf_counter()
        self._reset_window()

    def _reset_window(self) -> None:
        """Drop the accumulated window statistics and restart the peak-memory counter."""
        self._window_losses: list[np.ndarray] = []
        self._window_levels: list[np.ndarray] = []
        self._window_dt = 0.0
        self._window_steps = 0
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()

    def _write(self, record: dict[str, Any]) -> None:
        """Append one JSON object to ``metrics.jsonl`` and flush it to disk."""
        with self._path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def _loss_per_octave(self) -> dict[str, float | None]:
        """Average the window's per-sample losses into the octave bins of the blur schedule."""
        totals: dict[str, list[float]] = {name: [] for name in OCTAVE_BIN_NAMES}
        if self._window_losses:
            losses = np.concatenate(self._window_losses)
            levels = np.concatenate(self._window_levels)
            sigmas = self._schedule[np.clip(levels, 0, self._schedule.size - 1)]
            for loss, sigma in zip(losses, sigmas, strict=True):
                name = octave_bin(float(sigma))
                if name is not None:
                    totals[name].append(float(loss))
        return {
            name: (float(np.mean(values)) if values else None) for name, values in totals.items()
        }

    def log_train(
        self,
        step: int,
        loss: torch.Tensor | float,
        losses_batch: torch.Tensor,
        fwd_steps_batch: torch.Tensor,
        lr: float,
        dt: float,
        grad_norm: float | None = None,
    ) -> bool:
        """Accumulate one training step and emit a ``train`` line every ``log_every`` steps.

        Parameters
        ----------
        step : int
            The loop index of the step that has just finished.
        loss : torch.Tensor | float
            The batch mean loss (unused for the line, which reports the window mean; kept so
            the caller does not have to reduce ``losses_batch`` itself).
        losses_batch : torch.Tensor
            Per-sample losses, shape ``(B,)``.
        fwd_steps_batch : torch.Tensor
            Per-sample blur levels, shape ``(B,)``, integer.
        lr : float
            The learning rate the optimiser used for this step (after warm-up scaling).
        dt : float
            Seconds this step took, measured by the caller.
        grad_norm : float | None
            The gradient norm, measured by the caller at log steps only.

        Returns
        -------
        bool
            ``True`` if a line was written.
        """
        del loss  # the emitted value is the window mean, which equals the mean of losses_batch
        self._window_losses.append(losses_batch.detach().float().cpu().numpy().ravel())
        self._window_levels.append(fwd_steps_batch.detach().cpu().numpy().ravel().astype(np.int64))
        self._window_dt += float(dt)
        self._window_steps += 1

        if step % self._log_every != 0:
            return False

        window_loss = float(np.mean(np.concatenate(self._window_losses)))
        it_per_s = self._window_steps / self._window_dt if self._window_dt > 0 else 0.0
        record = {
            "step": int(step),
            "kind": "train",
            "loss": window_loss,
            "lr": float(lr),
            "it_per_s": it_per_s,
            "img_per_s": it_per_s * self._batch_size,
            "gpu_mem_peak_gb": self._peak_memory_gb(),
            "grad_norm": float(grad_norm) if grad_norm is not None else None,
            "wall_s": time.perf_counter() - self._t_start,
            "loss_per_octave": self._loss_per_octave(),
        }
        self._write(record)
        self._writer.add_scalar("train/loss", window_loss, step)
        self._writer.add_scalar("train/lr", record["lr"], step)
        self._writer.add_scalar("train/it_per_s", record["it_per_s"], step)
        for name, value in record["loss_per_octave"].items():
            if value is not None:
                self._writer.add_scalar(f"train/loss_octave_{name}", value, step)
        self._reset_window()
        return True

    def _peak_memory_gb(self) -> float:
        """Return the peak CUDA allocation of the current window in GiB (0.0 on CPU)."""
        if not self._cuda:
            return 0.0
        return float(torch.cuda.max_memory_allocated()) / float(2**30)

    def log_eval(self, step: int, loss: float) -> None:
        """Write an ``eval`` line (EMA weights, 25 batches of the ``ref`` split).

        Parameters
        ----------
        step : int
            The loop index at which the evaluation ran.
        loss : float
            The mean loss over the evaluation batches.
        """
        self._write({"step": int(step), "kind": "eval", "loss": float(loss)})
        self._writer.add_scalar("eval/loss", float(loss), step)

    def log_event(self, kind: str, step: int, **fields: Any) -> None:
        """Write one non-scalar event line (``ckpt``, ``grid``, ``resume``, ``abort``, ``done``).

        Parameters
        ----------
        kind : str
            The event name written to the ``kind`` field.
        step : int
            The loop index at which the event happened.
        **fields : Any
            Extra JSON-serialisable fields merged into the record.
        """
        record: dict[str, Any] = {"step": int(step), "kind": kind}
        record.update(fields)
        self._write(record)

    def close(self) -> None:
        """Flush and close the tensorboard writer."""
        self._writer.flush()
        self._writer.close()
