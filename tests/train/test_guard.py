"""Unit tests of ``ihdm.train.guard``: the D19 skip policy and the skip count across resumes."""

from __future__ import annotations

import json

import pytest

from configs.spectral import smoke
from ihdm.train.errors import TrainError
from ihdm.train.guard import (
    DEFAULT_MAX_CONSECUTIVE_SKIPS,
    DEFAULT_MAX_SKIPS,
    NonFiniteGuard,
    SkipPolicy,
    committed_skips,
    committed_skips_from_file,
    policy_from_config,
)


def _run(guard: NonFiniteGuard, pattern: str):
    """Feed ``pattern`` ("." finite, "x" non-finite) and return the decisions."""
    return [guard.observe(char == ".") for char in pattern]


def test_finite_losses_never_skip_or_abort():
    decisions = _run(NonFiniteGuard(SkipPolicy()), "." * 50)
    assert not any(d.skipped or d.abort for d in decisions)
    assert decisions[-1].n_skipped == 0 and decisions[-1].consecutive == 0


def test_an_isolated_non_finite_loss_is_a_skip_and_resets_the_streak():
    guard = NonFiniteGuard(SkipPolicy())
    decisions = _run(guard, "..x..x.")
    skips = [i for i, d in enumerate(decisions) if d.skipped]
    assert skips == [2, 5]
    assert not any(d.abort for d in decisions)
    assert [decisions[i].consecutive for i in skips] == [1, 1]
    assert guard.n_skipped == 2 and guard.consecutive == 0


@pytest.mark.parametrize("limit", [1, 2, 10])
def test_the_limit_th_consecutive_skip_aborts(limit):
    guard = NonFiniteGuard(SkipPolicy(max_consecutive=limit, max_total=1000))
    decisions = _run(guard, "x" * limit)
    assert all(d.skipped for d in decisions)
    assert [d.abort for d in decisions] == [False] * (limit - 1) + [True]
    assert decisions[-1].consecutive == limit
    assert "consecutive" in decisions[-1].reason


def test_nine_consecutive_skips_then_a_finite_loss_do_not_abort():
    guard = NonFiniteGuard(SkipPolicy())
    decisions = _run(guard, "x" * 9 + "." + "x" * 9)
    assert not any(d.abort for d in decisions)
    assert guard.n_skipped == 18


@pytest.mark.parametrize("max_total", [0, 1, 5, 100])
def test_more_than_max_skips_in_total_aborts(max_total):
    guard = NonFiniteGuard(SkipPolicy(max_consecutive=10, max_total=max_total))
    decisions = _run(guard, "x." * (max_total + 1))
    skips = [d for d in decisions if d.skipped]
    assert len(skips) == max_total + 1
    assert [d.abort for d in skips] == [False] * max_total + [True]
    assert skips[-1].n_skipped == max_total + 1
    assert "max_skips" in skips[-1].reason


def test_the_total_survives_a_resume_but_the_streak_does_not():
    guard = NonFiniteGuard(SkipPolicy(max_consecutive=3, max_total=5), n_skipped=5)
    first = guard.observe(False)
    assert first.skipped and first.abort and first.n_skipped == 6 and first.consecutive == 1


def test_a_disabled_scaler_aborts_at_the_first_non_finite_loss():
    guard = NonFiniteGuard(SkipPolicy(scaler_enabled=False))
    decisions = _run(guard, "...x")
    assert not any(d.abort for d in decisions[:3])
    assert decisions[3].abort and not decisions[3].skipped
    assert decisions[3].n_skipped == 0
    assert "disabled" in decisions[3].reason


@pytest.mark.parametrize(("kwargs"), [{"max_consecutive": 0}, {"max_total": -1}])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(TrainError):
        SkipPolicy(**kwargs)


def test_negative_carried_count_is_rejected():
    with pytest.raises(TrainError):
        NonFiniteGuard(SkipPolicy(), n_skipped=-1)


def test_policy_defaults_when_the_config_lacks_the_keys():
    config = smoke.get_config()
    assert "max_skips" not in config.training
    policy = policy_from_config(config, scaler_enabled=True)
    assert policy == SkipPolicy(DEFAULT_MAX_CONSECUTIVE_SKIPS, DEFAULT_MAX_SKIPS, True)
    assert (DEFAULT_MAX_CONSECUTIVE_SKIPS, DEFAULT_MAX_SKIPS) == (10, 100)


def test_policy_reads_the_config_keys():
    config = smoke.get_config()
    config.training.max_consecutive_skips = 3
    config.training.max_skips = 7
    assert policy_from_config(config, scaler_enabled=False) == SkipPolicy(3, 7, False)


def _skip(step):
    return {"step": step, "kind": "skip", "loss": None}


def _resume(step):
    return {"step": step, "kind": "resume", "from": "x"}


@pytest.mark.parametrize(
    ("records", "initial_step", "expected"),
    [
        ([], 0, 0),
        ([], 501, 0),
        ([_skip(10), _skip(20)], 0, 0),  # a fresh start never carries anything over
        ([_skip(10), _skip(20)], 501, 2),
        # skips after the last checkpoint (501) were undone by the kill
        ([_skip(10), _skip(600), _skip(700)], 501, 1),
        # an earlier killed segment: 600 was undone by the resume at 501, 650 survives
        ([_skip(10), _skip(600), _resume(501), _skip(650), _skip(1100)], 1001, 2),
        # a skip exactly at the resume step was undone (the checkpoint precedes it)
        ([_skip(501), _resume(501)], 1001, 0),
        # the guard's abort saved the state after the skipped step: it is kept
        ([_skip(1382)], 1383, 1),
        ([{"step": 0, "kind": "train"}, {"step": 0, "kind": "eval"}, _skip(3)], 4, 1),
    ],
)
def test_committed_skips_replays_the_log(records, initial_step, expected):
    assert committed_skips(records, initial_step) == expected


def test_committed_skips_from_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    assert committed_skips_from_file(path, 100) == 0  # no file yet
    lines = [_skip(3), _skip(4), _resume(4), _skip(7)]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines) + '{"step": 9, "ki')
    assert committed_skips_from_file(path, 8) == 2  # 3 and 7; 4 undone; truncated tail ignored


def test_committed_skips_from_file_rejects_a_corrupt_middle_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(_skip(3)) + "\nnot json\n" + json.dumps(_skip(4)) + "\n")
    with pytest.raises(TrainError):
        committed_skips_from_file(path, 10)
