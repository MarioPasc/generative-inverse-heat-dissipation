"""Unit tests of ``ihdm.train.logging``: octave binning, window averaging, strict JSON.

The strict-JSON case is a regression test: the GPU demonstration of this ticket wrote
``"grad_norm": NaN`` on the first line of ``metrics.jsonl``, because AMP's ``GradScaler`` skips
the first optimiser steps and leaves non-finite gradients behind. Python reads that token back,
but ``jq``, ``pandas.read_json`` and every strict parser reject it, and T4.x reads these files.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configs.spectral import smoke
from ihdm.train.logging import OCTAVE_BIN_NAMES, MetricsLogger, octave_bin


@pytest.mark.parametrize(
    ("sigma_b", "expected"),
    [
        (0.49, None),
        (0.5, "0.5-1"),
        (0.999, "0.5-1"),
        (1.0, "1-2"),
        (3.9, "2-4"),
        (4.0, "4-8"),
        (63.9, "32-64"),
        (64.0, "64-96"),
        (96.0, "64-96"),  # the last bin is closed at 96
        (96.1, None),
        (135.0, None),
    ],
)
def test_octave_bin_edges(sigma_b, expected):
    assert octave_bin(sigma_b) == expected


def _logger(tmp_path, log_every: int = 2) -> MetricsLogger:
    config = smoke.get_config()
    config.training.log_every = log_every
    return MetricsLogger(tmp_path, config)


def _records(tmp_path) -> list[dict]:
    with (tmp_path / "metrics.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_a_line_is_written_only_every_log_every_steps(tmp_path):
    logger = _logger(tmp_path, log_every=3)
    written = [
        logger.log_train(
            step, 1.0, torch.ones(4), torch.full((4,), 2, dtype=torch.long), 1e-4, 0.1, 0.5
        )
        for step in range(7)
    ]
    logger.close()
    assert written == [True, False, False, True, False, False, True]
    assert [record["step"] for record in _records(tmp_path)] == [0, 3, 6]


def test_the_loss_is_the_window_mean_and_the_octaves_follow_the_levels(tmp_path):
    schedule = np.asarray(smoke.get_config().model.blur_schedule)
    low_bin, high_bin = octave_bin(float(schedule[1])), octave_bin(float(schedule[8]))
    assert low_bin == "0.5-1"  # level 1 is sigma_B = 0.5 by construction
    assert high_bin is not None and low_bin != high_bin

    logger = _logger(tmp_path, log_every=2)
    # Steps 1 and 2 form one window: step 0 would emit a line of its own (0 % log_every == 0),
    # which is the released cadence convention and is covered by the cadence test above.
    logger.log_train(1, 1.0, torch.tensor([2.0, 4.0]), torch.tensor([1, 1]), 1e-4, 0.1, 0.5)
    logger.log_train(2, 1.0, torch.tensor([10.0, 20.0]), torch.tensor([8, 8]), 1e-4, 0.1, 0.5)
    logger.close()

    record = _records(tmp_path)[0]
    assert record["step"] == 2
    assert record["loss"] == pytest.approx((2 + 4 + 10 + 20) / 4)
    assert record["loss_per_octave"][low_bin] == pytest.approx(3.0)
    assert record["loss_per_octave"][high_bin] == pytest.approx(15.0)
    assert all(
        record["loss_per_octave"][name] is None
        for name in OCTAVE_BIN_NAMES
        if name not in (low_bin, high_bin)
    )
    assert record["it_per_s"] == pytest.approx(2 / 0.2)
    assert record["img_per_s"] == pytest.approx(record["it_per_s"] * 4)
    assert record["gpu_mem_peak_gb"] == 0.0


def test_the_window_resets_after_a_line(tmp_path):
    logger = _logger(tmp_path, log_every=1)
    logger.log_train(0, 1.0, torch.tensor([100.0]), torch.tensor([1]), 1e-4, 0.1, 0.5)
    logger.log_train(1, 1.0, torch.tensor([1.0]), torch.tensor([1]), 1e-4, 0.1, 0.5)
    logger.close()
    assert [record["loss"] for record in _records(tmp_path)] == [100.0, 1.0]


def test_non_finite_fields_are_written_as_null_not_nan(tmp_path):
    logger = _logger(tmp_path, log_every=1)
    logger.log_train(
        0, 1.0, torch.tensor([1.0]), torch.tensor([1]), 1e-4, 0.1, float("nan")
    )
    logger.log_train(
        1, 1.0, torch.tensor([1.0]), torch.tensor([1]), 1e-4, 0.1, float("inf")
    )
    logger.close()

    text = (tmp_path / "metrics.jsonl").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert all(record["grad_norm"] is None for record in _records(tmp_path))


def test_events_carry_their_kind_and_extra_fields(tmp_path):
    logger = _logger(tmp_path)
    logger.log_eval(4, 1.25)
    logger.log_event("ckpt", 4, path="/somewhere/ema_iter_000004.pt")
    logger.log_event("resume", 7, **{"from": "/somewhere/checkpoint.pth"})
    logger.close()

    records = _records(tmp_path)
    assert records[0] == {"step": 4, "kind": "eval", "loss": 1.25}
    assert records[1] == {"step": 4, "kind": "ckpt", "path": "/somewhere/ema_iter_000004.pt"}
    assert records[2] == {"step": 7, "kind": "resume", "from": "/somewhere/checkpoint.pth"}


def test_grad_norm_and_amp_scale_are_written_as_given(tmp_path):
    logger = _logger(tmp_path, log_every=1)
    logger.log_train(0, 1.0, torch.tensor([1.0]), torch.tensor([1]), 1e-4, 0.1, 3.5,
                     amp_scale=32768.0)
    logger.log_train(1, 1.0, torch.tensor([1.0]), torch.tensor([1]), 1e-4, 0.1, None)
    logger.close()
    first, second = _records(tmp_path)
    assert (first["grad_norm"], first["amp_scale"]) == (3.5, 32768.0)
    assert (second["grad_norm"], second["amp_scale"]) == (None, None)


def test_a_skipped_step_adds_no_loss_but_keeps_the_cadence(tmp_path):
    logger = _logger(tmp_path, log_every=2)
    nan = torch.tensor([float("nan")])
    logger.log_train(1, 1.0, torch.tensor([2.0]), torch.tensor([1]), 1e-4, 0.1, 1.0)
    logger.log_train(2, 1.0, nan, torch.tensor([1]), 1e-4, 0.1, float("nan"), skipped=True)
    logger.log_train(3, 1.0, nan, torch.tensor([1]), 1e-4, 0.1, None, skipped=True)
    logger.log_train(4, 1.0, nan, torch.tensor([1]), 1e-4, 0.1, None, skipped=True)
    logger.close()
    first, second = _records(tmp_path)
    assert first["step"] == 2 and first["loss"] == pytest.approx(2.0)  # only the finite step
    assert first["it_per_s"] == pytest.approx(2 / 0.2)  # the skipped step still took time
    assert second["step"] == 4 and second["loss"] is None  # a window of skipped steps only
    assert all(value is None for value in second["loss_per_octave"].values())
    assert "NaN" not in (tmp_path / "metrics.jsonl").read_text()
