"""The non-finite-loss guard of the trainer (decision D19, ``04-run-artifacts.md`` §3.2 (b)).

Under fp16 AMP with an **enabled** ``GradScaler``, a non-finite loss produces non-finite
gradients, ``scaler.unscale_`` records them, ``scaler.step`` skips ``optimizer.step()`` and
``scaler.update()`` halves the scale: the step is a no-op on the weights (the released code's
behaviour). The trainer then logs a ``skip`` event and carries on, and aborts only when the
events stop being isolated: ``consecutive >= max_consecutive_skips`` or
``n_skipped > max_skips``.

With a **disabled** scaler (``optim.automatic_mp=False``, or a CPU run, where
``torch.cuda.amp.GradScaler`` disables itself) ``scaler.step`` calls ``optimizer.step()``
unconditionally, so the non-finite gradients are applied and the weights are lost: the first
non-finite loss aborts.

The decision logic is a pure state machine (:class:`NonFiniteGuard`), and the count of skipped
steps that survives a resume is recomputed from ``metrics.jsonl`` by a pure function
(:func:`committed_skips`), so both are unit-tested without a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ihdm.train.errors import TrainError

__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_SKIPS",
    "DEFAULT_MAX_SKIPS",
    "GuardDecision",
    "NonFiniteGuard",
    "SkipPolicy",
    "committed_skips",
    "committed_skips_from_file",
    "policy_from_config",
]

#: D19: abort after this many consecutive skipped steps.
DEFAULT_MAX_CONSECUTIVE_SKIPS = 10
#: D19: abort when more than this many steps have been skipped over the whole run.
DEFAULT_MAX_SKIPS = 100


@dataclass(frozen=True)
class SkipPolicy:
    """Limits of the guard and whether the optimiser step is a no-op on a non-finite loss.

    Parameters
    ----------
    max_consecutive : int
        Abort when this many consecutive steps have been skipped (``>=``).
    max_total : int
        Abort when the number of skipped steps of the run exceeds this value (``>``).
    scaler_enabled : bool
        ``True`` when an enabled ``GradScaler`` drives the optimiser step, i.e. a non-finite
        loss is skipped rather than applied.

    Raises
    ------
    TrainError
        If ``max_consecutive < 1`` or ``max_total < 0``.
    """

    max_consecutive: int = DEFAULT_MAX_CONSECUTIVE_SKIPS
    max_total: int = DEFAULT_MAX_SKIPS
    scaler_enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_consecutive < 1:
            raise TrainError(f"max_consecutive_skips must be >= 1, got {self.max_consecutive}")
        if self.max_total < 0:
            raise TrainError(f"max_skips must be >= 0, got {self.max_total}")


@dataclass(frozen=True)
class GuardDecision:
    """What the trainer must do after one step.

    Parameters
    ----------
    skipped : bool
        The step was a no-op on the weights and must be logged as a ``skip`` event.
    abort : bool
        The run must stop (exit code 3).
    n_skipped : int
        Skipped steps of the whole run so far, this one included.
    consecutive : int
        Length of the current run of consecutive skipped steps, this one included.
    reason : str | None
        Why the run aborts; ``None`` unless ``abort``.
    """

    skipped: bool
    abort: bool
    n_skipped: int
    consecutive: int
    reason: str | None = None


class NonFiniteGuard:
    """State machine that turns "was this step's loss finite?" into skip/abort decisions.

    Parameters
    ----------
    policy : SkipPolicy
        The limits and the scaler state.
    n_skipped : int
        Skipped steps carried over from before a resume (see :func:`committed_skips`); the
        consecutive count always restarts at 0, as does the ``GradScaler`` scale.
    """

    def __init__(self, policy: SkipPolicy, n_skipped: int = 0) -> None:
        if n_skipped < 0:
            raise TrainError(f"n_skipped must be >= 0, got {n_skipped}")
        self.policy = policy
        self.n_skipped = int(n_skipped)
        self.consecutive = 0

    def observe(self, loss_is_finite: bool) -> GuardDecision:
        """Record one step and return the decision for it.

        Parameters
        ----------
        loss_is_finite : bool
            Whether the step's training loss was finite.

        Returns
        -------
        GuardDecision
            ``skipped=False, abort=False`` for a finite loss; otherwise a skip, an abort, or
            both (a skip that crosses a limit).
        """
        if loss_is_finite:
            self.consecutive = 0
            return GuardDecision(False, False, self.n_skipped, 0)
        if not self.policy.scaler_enabled:
            return GuardDecision(
                False,
                True,
                self.n_skipped,
                self.consecutive,
                "non-finite training loss with the GradScaler disabled: the optimiser step "
                "applied the non-finite gradients",
            )
        self.n_skipped += 1
        self.consecutive += 1
        reason = None
        if self.consecutive >= self.policy.max_consecutive:
            reason = (
                f"{self.consecutive} consecutive non-finite training losses "
                f"(max_consecutive_skips={self.policy.max_consecutive})"
            )
        elif self.n_skipped > self.policy.max_total:
            reason = (
                f"{self.n_skipped} non-finite training losses in this run "
                f"(max_skips={self.policy.max_total})"
            )
        return GuardDecision(True, reason is not None, self.n_skipped, self.consecutive, reason)


def policy_from_config(config: Any, scaler_enabled: bool) -> SkipPolicy:
    """Build the :class:`SkipPolicy` of a run, defaulting the limits a config does not set.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        The run config; ``training.max_consecutive_skips`` and ``training.max_skips`` are
        optional (the released and smoke configs lack them).
    scaler_enabled : bool
        Whether an enabled ``GradScaler`` drives the optimiser step.

    Returns
    -------
    SkipPolicy
        The policy.
    """
    training = config.training
    return SkipPolicy(
        max_consecutive=int(training.get("max_consecutive_skips", DEFAULT_MAX_CONSECUTIVE_SKIPS)),
        max_total=int(training.get("max_skips", DEFAULT_MAX_SKIPS)),
        scaler_enabled=bool(scaler_enabled),
    )


def committed_skips(records: Iterable[dict[str, Any]], initial_step: int) -> int:
    """Count the ``skip`` events that belong to the training state being resumed.

    ``metrics.jsonl`` is append-only across restarts, so it can hold ``skip`` lines of steps
    that were later undone: a job killed after its last rolling checkpoint loses every step
    after it, and the next job replays them (its ``resume`` line carries the step it restarts
    from). Replaying the log in order and discarding, at each ``resume`` line, the skips at or
    after its step leaves exactly the skips of the surviving history; the final filter against
    ``initial_step`` does the same for the restart being set up now.

    Parameters
    ----------
    records : Iterable[dict[str, Any]]
        The parsed lines of ``metrics.jsonl``, in file order.
    initial_step : int
        The step the trainer is about to start from (``state["step"]`` after the restore).

    Returns
    -------
    int
        The number of skipped steps before ``initial_step`` in the surviving history.
    """
    steps: list[int] = []
    for record in records:
        kind = record.get("kind")
        if kind == "resume":
            restart = int(record["step"])
            steps = [s for s in steps if s < restart]
        elif kind == "skip":
            steps.append(int(record["step"]))
    return sum(1 for s in steps if s < initial_step)


def committed_skips_from_file(path: Path, initial_step: int) -> int:
    """Apply :func:`committed_skips` to a ``metrics.jsonl`` file (0 if it does not exist).

    Parameters
    ----------
    path : Path
        The run's ``metrics.jsonl``.
    initial_step : int
        The step the trainer is about to start from.

    Returns
    -------
    int
        The carried-over skip count.

    Raises
    ------
    TrainError
        If a line is not valid JSON (a truncated last line is tolerated: a job killed
        mid-write can leave one, and it cannot be a ``skip`` line that the checkpoint covers).
    """
    path = Path(path)
    if initial_step <= 0 or not path.is_file():
        return 0
    lines = path.read_text().splitlines()
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == len(lines):
                break
            raise TrainError(f"{path}:{number}: not valid JSON ({error})") from error
    return committed_skips(records, initial_step)
