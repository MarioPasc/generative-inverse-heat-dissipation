"""EMA checkpoint contract (``docs/SPECIFICATIONS/04-run-artifacts.md`` §3.3).

Every metric of the experiment is computed offline from these files, by T2.2's
``ihdm.sampling.load_ema_model``, which instantiates a bare ``model_code.unet.UNetModel`` and
loads ``ema_state_dict`` into it. The keys must therefore carry no ``DataParallel`` prefix, and
each file must identify on its own which schedule and which config produced it.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

EXPECTED_KEYS = {"step", "ema_state_dict", "schedule", "run_id", "config_sha256"}


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_ema_checkpoint_has_exactly_the_contract_keys(smoke_run):
    payload = _load(smoke_run / "checkpoints" / "ema_iter_000006.pt")
    assert set(payload) == EXPECTED_KEYS
    assert payload["step"] == 6
    assert payload["run_id"] == "synthetic_smoke_s1"
    assert len(payload["config_sha256"]) == 64


def test_state_dict_keys_have_no_dataparallel_prefix(smoke_run):
    payload = _load(smoke_run / "checkpoints" / "ema_iter_000006.pt")
    state = payload["ema_state_dict"]
    assert state, "the EMA state dict is empty"
    assert not any(key.startswith("module.") for key in state)
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert all(value.device.type == "cpu" for value in state.values())


def test_schedule_record_matches_the_config(smoke_run):
    payload = _load(smoke_run / "checkpoints" / "ema_iter_000006.pt")
    config = json.loads((smoke_run / "config.json").read_text())
    record = payload["schedule"]

    assert set(record) == {"name", "sha256", "values"}
    assert record["name"] == config["model"]["blur_schedule_name"]
    assert record["sha256"] == config["model"]["blur_schedule_sha256"]
    np.testing.assert_allclose(record["values"], config["model"]["blur_schedule"])

    recomputed = hashlib.sha256(
        np.ascontiguousarray(np.asarray(record["values"], dtype=np.float64)).tobytes()
    ).hexdigest()
    assert recomputed == record["sha256"]


def test_manifest_and_checkpoint_agree(smoke_run):
    payload = _load(smoke_run / "checkpoints" / "ema_iter_000006.pt")
    manifest = json.loads((smoke_run / "manifest.json").read_text())
    assert payload["run_id"] == manifest["run_id"]
    assert payload["config_sha256"] == manifest["config_sha256"]
    assert payload["schedule"]["sha256"] == manifest["schedule"]["sha256"]
    np.testing.assert_allclose(payload["schedule"]["values"], manifest["schedule"]["values"])


def test_ema_weights_differ_from_the_final_live_weights(smoke_run):
    ema = _load(smoke_run / "checkpoints" / "ema_iter_000006.pt")["ema_state_dict"]
    full = _load(smoke_run / "checkpoints" / "full_final.pt")
    live = {key.removeprefix("module."): value for key, value in full["model"].items()}

    assert set(ema) == set(live)
    trainable = [key for key in ema if ema[key].is_floating_point() and ema[key].numel() > 1]
    assert any(not torch.allclose(ema[key], live[key]) for key in trainable), (
        "the EMA snapshot is identical to the live weights: the EMA was not applied"
    )


def test_full_final_carries_the_released_four_keys(smoke_run):
    full = _load(smoke_run / "checkpoints" / "full_final.pt")
    assert set(full) == {"optimizer", "model", "step", "ema"}
    assert full["step"] >= 7
    assert set(full["ema"]) == {"decay", "num_updates", "shadow_params"}


def test_intermediate_checkpoint_is_a_full_snapshot(smoke_run):
    payload = _load(smoke_run / "checkpoints" / "ema_iter_000003.pt")
    assert payload["step"] == 3
    assert set(payload) == EXPECTED_KEYS
