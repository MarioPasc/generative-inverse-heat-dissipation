"""Config and checkpoint loading (``04-run-artifacts.md`` §3.3, §4, §6)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ihdm.sampling.errors import SamplingError
from ihdm.sampling.loader import (
    build_heat_module,
    checkpoint_sha256,
    load_ema_model,
    load_run_config,
    resolve_checkpoint_path,
)


def test_load_run_config_restores_types(tiny_run, tiny_config):
    """``config.json`` round-trips to the types the released code expects."""
    workdir, _ = tiny_run
    config = load_run_config(workdir)

    assert isinstance(config.model.blur_schedule, np.ndarray)
    assert config.model.blur_schedule.dtype == np.float64
    np.testing.assert_allclose(config.model.blur_schedule, tiny_config.model.blur_schedule)
    assert isinstance(config.model.channel_mult, tuple)
    assert isinstance(config.model.attention_levels, tuple)
    assert isinstance(config.device, torch.device)
    assert int(config.model.K) == int(tiny_config.model.K)
    assert config.model.blur_schedule_sha256 == tiny_config.model.blur_schedule_sha256


def test_load_run_config_falls_back_to_manifest(tmp_path, schedules_dir):
    """Without ``config.json`` the config is rebuilt from the manifest's dataset/arm/seed."""
    workdir = tmp_path / "runs" / "lsun_church_A0_s2"
    workdir.mkdir(parents=True)
    (workdir / "manifest.json").write_text(
        json.dumps({"dataset_id": "lsun_church", "arm": "A0", "seed": 2})
    )

    config = load_run_config(workdir)

    assert config.data.dataset == "lsun_church"
    assert config.arm == "A0"
    assert int(config.seed) == 2
    assert config.model.blur_schedule.shape == (201,)


def test_load_run_config_without_any_metadata_raises(tmp_path):
    """A directory with neither file is not a run."""
    with pytest.raises(SamplingError, match="neither config.json nor manifest.json"):
        load_run_config(tmp_path)


def test_resolve_checkpoint_path_accepts_bare_name(tiny_run):
    """A bare file name is looked up under ``<run>/checkpoints``."""
    workdir, ckpt = tiny_run
    assert resolve_checkpoint_path(workdir, ckpt.name) == ckpt.resolve()
    assert resolve_checkpoint_path(workdir, str(ckpt)) == ckpt.resolve()
    with pytest.raises(SamplingError, match="not found"):
        resolve_checkpoint_path(workdir, "ema_iter_999999.pt")


def test_load_ema_model_round_trips_the_weights(tiny_run, tiny_config):
    """Every tensor of ``ema_state_dict`` reaches the bare ``UNetModel`` unchanged."""
    _, ckpt = tiny_run
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)

    model = load_ema_model(ckpt, tiny_config, torch.device("cpu"))

    loaded = model.state_dict()
    assert set(loaded) == set(payload["ema_state_dict"])
    for key, value in payload["ema_state_dict"].items():
        torch.testing.assert_close(loaded[key], value)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert not isinstance(model, torch.nn.DataParallel)


def test_schedule_hash_mismatch_raises(tiny_run, tiny_config):
    """A checkpoint trained under another schedule is refused before any weight is loaded."""
    _, ckpt = tiny_run
    tiny_config.model.blur_schedule_sha256 = "0" * 64

    with pytest.raises(SamplingError, match="blur-schedule hash mismatch"):
        load_ema_model(ckpt, tiny_config, torch.device("cpu"))


def test_missing_schedule_hash_raises(tiny_run, tiny_config):
    """An empty hash on either side is an error, not a licence to sample."""
    _, ckpt = tiny_run
    tiny_config.model.blur_schedule_sha256 = ""

    with pytest.raises(SamplingError, match="hash missing"):
        load_ema_model(ckpt, tiny_config, torch.device("cpu"))


def test_payload_without_ema_state_dict_raises(tmp_path):
    """A file that is not an EMA checkpoint of §3.3 is reported as such."""
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"step": 1}, path)

    with pytest.raises(SamplingError, match="no 'ema_state_dict'"):
        load_ema_model(path, None, torch.device("cpu"))


def test_checkpoint_sha256_matches_the_file_bytes(tiny_run):
    """The recorded checkpoint hash is the hash of the file."""
    import hashlib

    _, ckpt = tiny_run
    assert checkpoint_sha256(ckpt) == hashlib.sha256(ckpt.read_bytes()).hexdigest()


def test_build_heat_module_matches_the_schedule(tiny_config):
    """The blur module carries the run's schedule and leaves level 0 untouched."""
    heat = build_heat_module(tiny_config, "cpu")
    x = torch.rand(2, 1, 32, 32)

    at_zero = heat(x, torch.zeros(2, dtype=torch.long))

    torch.testing.assert_close(at_zero.float(), x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        heat.blur_sigmas.cpu().numpy(), np.asarray(tiny_config.model.blur_schedule)
    )
