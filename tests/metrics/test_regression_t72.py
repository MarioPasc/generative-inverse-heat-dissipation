"""Regression of T7.2: the default evaluation is byte-identical to the base commit ``48363e2``.

The goldens in ``tests/metrics/data/golden_t72.json`` were written by
``python -m tests.metrics.golden_capture --write`` with the base-commit code, before any change of
T7.2 (``--delta``, the W rule of ``ihdm.metrics.spectral``, ``--final-from-lsd``). These tests
recompute the same quantities with the current code:

* the metric layer at ``W = 192`` and ``W = 96`` must match exactly (``float.hex`` strings);
* the default ``evaluate_run`` at ``W = 96`` and ``W = 192`` must write the same file tree with
  the same JSON keys and the same normalised content digests. The ``samples.npy`` digests are
  compared first: if they differ the platform's sampler differs from the capture machine's and the
  comparison of the metric files is meaningless, so the test is skipped rather than failed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.metrics.golden_capture import (
    DEFAULT_PATH_CKPTS,
    GOLDEN_PATH,
    default_path,
    metric_layer,
)


@pytest.fixture(scope="module")
def golden() -> dict:
    """The goldens captured at the base commit."""
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("width", [192, 96])
def test_the_metric_layer_is_bitwise_unchanged(golden, width):
    assert metric_layer(width) == golden["metric_layer"][str(width)]


@pytest.mark.parametrize("width", sorted(DEFAULT_PATH_CKPTS))
def test_the_default_evaluation_is_byte_identical_to_the_base_commit(golden, tmp_path, width):
    expected = golden["default_path"][str(width)]
    got = default_path(Path(tmp_path).resolve(), width)
    assert sorted(got) == sorted(expected), "the default path writes a different file tree"

    samples = [name for name in expected if name.endswith("samples.npy")]
    assert samples, "the golden holds no samples.npy"
    if any(got[name]["sha256"] != expected[name]["sha256"] for name in samples):
        pytest.skip("the sampler is not bitwise reproducible against the capture platform")

    for name, record in expected.items():
        if "keys" in record:
            assert got[name]["keys"] == record["keys"], f"{name}: the JSON keys changed"
        assert got[name]["sha256"] == record["sha256"], f"{name}: the content changed"
