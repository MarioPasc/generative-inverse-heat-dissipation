"""Tests of ``ihdm.cli.diagnose_nan`` on a CPU fixture built from the smoke config.

The real diagnosis runs on a V100 in fp16 autocast. Here CPU fp16 autocast plays that role: it
also stores conv outputs in float16, so a conv whose output exceeds 65504 becomes ``inf`` and the
``GroupNorm32`` after it (computed in fp32 from that input) turns it into NaN, exactly the
mechanism the diagnosis looks for, while the fp32 forward of the same input stays finite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from configs.spectral import smoke
from ihdm.cli import diagnose_nan as dn
from ihdm.data.dataset import NpyImageDataset
from ihdm.train.manifest import write_config_json
from model_code.ema import ExponentialMovingAverage
from model_code.unet import UNetModel


def _make_fixture(tmp_path: Path, dataset_root: Path, scale_first_conv: float = 1.0,
                  logged_loss: float = 1e9, attention: bool = False) -> Path:
    """A guard-saved run directory: config.json, checkpoint.pth (DataParallel keys), metrics."""
    config = smoke.get_config()
    config.data.root = str(dataset_root.parent)
    if attention:  # exercise the attention hooks the real U-Net has at 48^2 and 24^2
        config.model.attention_levels = (1,)
    torch.manual_seed(0)
    model = UNetModel(config)
    # The EMA keeps the healthy weights, as in array 1 (EMA loss 0.39 where the live one was 1.55).
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    with torch.no_grad():
        model.input_blocks[0][0].weight.mul_(scale_first_conv)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    run = tmp_path / "fixture"
    (run / "checkpoints-meta").mkdir(parents=True)
    write_config_json(run, config)
    torch.save({"optimizer": optimizer.state_dict(),
                "model": {f"module.{k}": v for k, v in model.state_dict().items()},
                "step": 6, "ema": ema.state_dict()},
               run / "checkpoints-meta" / "checkpoint.pth")
    lines = [{"step": 4, "kind": "train", "loss": logged_loss, "lr": 2e-4, "grad_norm": 1.0},
             {"step": 5, "kind": "abort", "reason": "non-finite training loss", "loss": "nan"}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return run


def _args(run: Path, out: Path, **overrides):
    argv = ["--fixture", str(run), "--out", str(out), "--n-batches", "2", "--sweep-images", "4",
            "--draws", "1", "--device", "cpu"]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        argv += [flag] if value is True else [flag, str(value)]
    return dn.parse_args(argv)


def test_replay_indices_match_the_training_loader(synthetic_dataset):
    config = smoke.get_config()
    dataset = NpyImageDataset(synthetic_dataset, "train")
    generator = torch.Generator()
    generator.manual_seed(int(config.seed))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0,
                        generator=generator)
    batches_per_epoch = len(dataset) // 4
    iterator, seen = iter(loader), []
    for _ in range(2 * batches_per_epoch + 3):  # crosses two epoch boundaries, as train.py does
        try:
            seen.append(next(iterator)[0])
        except StopIteration:
            iterator = iter(loader)
            seen.append(next(iterator)[0])
    for step in (0, 1, batches_per_epoch, 2 * batches_per_epoch + 2):
        positions = dn.replay_indices(len(dataset), 4, int(config.seed), step)
        replayed = torch.stack([dataset[p][0] for p in positions])
        assert torch.equal(replayed, seen[step]), f"step {step}"


def test_the_recorder_finds_the_first_fp16_overflow():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.GroupNorm(8, 32),
                              torch.nn.Conv2d(32, 1, 3, padding=1))
    with torch.no_grad():
        net[0].weight.mul_(1e5)
    x = torch.rand(2, 1, 8, 8)
    recorder = dn.ActivationRecorder(net)
    with torch.autocast("cpu", dtype=torch.float16):
        out16 = net(x)
    recorder.close()
    assert not torch.isfinite(out16).all()
    assert recorder.first_non_finite()["module"] == "0"
    recorder32 = dn.ActivationRecorder(net)
    net(x)
    recorder32.close()
    assert recorder32.first_non_finite() is None
    assert recorder32.records[0]["max_abs"] >= dn.FP16_MAX


@pytest.mark.parametrize("legacy", [False, True])
def test_attention_logits_are_recomputed_in_fp32(legacy):
    from model_code.unet import QKVAttention, QKVAttentionLegacy

    module = (QKVAttentionLegacy if legacy else QKVAttention)(n_heads=1)
    torch.manual_seed(0)
    qkv = torch.randn(2, 3 * 8, 5) * 100
    q, k, _ = qkv.chunk(3, dim=1) if not legacy else qkv.split(8, dim=1)
    expected = torch.einsum("bct,bcs->bts", q, k).abs().max() / np.sqrt(8)
    assert dn._attention_logit_max(module, qkv) == pytest.approx(float(expected), rel=1e-5)
    wrapper = torch.nn.Sequential(module)
    recorder = dn.ActivationRecorder(wrapper)
    wrapper(qkv)
    recorder.close()
    assert recorder.records[0]["logit_max_abs"] == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"parameters": {"non_finite_total": 3}}, "true_divergence"),
        ({"fp32_bad": 2}, "true_divergence"),
        ({"fp32_median": 50.0}, "true_divergence"),
        ({"fp16_bad": 4}, "fp16_forward_overflow"),
        ({}, "other"),
    ],
)
def test_classify_rules(patch, expected):
    def cond(bad: int, median: float = 0.4) -> dict:
        return {"samples_non_finite": bad, "batch_loss_median": median}

    report = {
        "parameters": {"non_finite_total": 0, **patch.get("parameters", {})},
        "survey": {"live": {"fp16_train": cond(patch.get("fp16_bad", 0)), "fp16_eval": cond(0),
                            "fp32_train": cond(patch.get("fp32_bad", 0),
                                               patch.get("fp32_median", 0.4)),
                            "fp32_eval": cond(0)}},
        "fixture": {"last_train": [{"loss": 0.4}, {"loss": 0.45}]},
        "hooks": {"first_non_finite_fp16": {"module": "input_blocks.0.0", "type": "Conv2d"},
                  "modules_at_or_over_fp16_max_in_fp32": [{"module": "input_blocks.0.0"}]},
    }
    label, reasons = dn.classify(report)
    assert label == expected
    assert reasons


def test_a_healthy_fixture_is_not_reproduced(tmp_path, synthetic_dataset):
    run = _make_fixture(tmp_path, synthetic_dataset, attention=True)
    report = dn.run(_args(run, tmp_path / "r.json", replay_step=5))
    attention = report["hooks"]["attention"]
    assert attention and all(r["logit_max_abs"] is not None for r in attention)
    assert report["hooks"]["eval_mode"]["train_mode"] is False
    assert report["classification"] == "other"
    assert report["parameters"]["non_finite_total"] == 0
    assert report["survey"]["live"]["n_batches"] == 2
    assert set(report["survey"]["live"]) >= {c.name for c in dn.CONDITIONS}
    assert report["replay"]["step"] == 5 and len(report["replay"]["positions"]) == 4
    assert report["sweep"]["fp16_non_finite"] == 0
    assert report["hooks"]["n_modules"] > 10


def test_an_fp16_overflow_is_found_and_classified(tmp_path, synthetic_dataset):
    # The first conv's output exceeds 65504 in fp16 (inf -> NaN after the GroupNorm), while in
    # fp32 every GroupNorm rescales it and the loss stays finite.
    run = _make_fixture(tmp_path, synthetic_dataset, scale_first_conv=1e6)
    out = tmp_path / "report.json"
    assert dn.main(["--fixture", str(run), "--out", str(out), "--n-batches", "2",
                    "--sweep-images", "4", "--draws", "1", "--device", "cpu"]) == 0
    report = json.loads(out.read_text())
    live = report["survey"]["live"]
    assert live["fp16_train"]["batches_non_finite"] == 2
    assert live["fp32_train"]["samples_non_finite"] == 0
    assert report["replay"]["step"] == 5  # read from the abort line
    assert report["replay"]["fp32_non_finite"] == 0
    first = report["hooks"]["first_non_finite_fp16"]
    assert first["module"] == "input_blocks.0.0"
    assert any(r["module"] == "input_blocks.0.0"
               for r in report["hooks"]["modules_at_or_over_fp16_max_in_fp32"])
    assert report["classification"] == "fp16_forward_overflow"
    assert np.isfinite(report["survey"]["live"]["fp32_train"]["batch_loss_median"])
    assert report["survey"]["ema"]["fp16_train"]["batches_non_finite"] == 0
    assert report["headroom"]["ratio_to_fp16_max"] >= 1.0
    assert report["headroom"]["module"] == "input_blocks.0.0"


def test_headroom_only_compares_weight_sets_on_the_same_samples(tmp_path, synthetic_dataset):
    run = _make_fixture(tmp_path, synthetic_dataset, scale_first_conv=1e6)
    live = dn.run(_args(run, tmp_path / "a.json", headroom_only=True,
                        headroom_modules="input_blocks.0.0,out.2"))
    ema = dn.run(_args(run, tmp_path / "b.json", headroom_only=True, weights="ema",
                       headroom_modules="input_blocks.0.0,out.2"))
    assert "survey" not in live and live["classification"] is None
    assert live["headroom"]["n_samples"] == ema["headroom"]["n_samples"] == 8
    first_live = live["headroom"]["modules"]["input_blocks.0.0"]["ratio"]
    first_ema = ema["headroom"]["modules"]["input_blocks.0.0"]["ratio"]
    assert first_live > 1e4 * first_ema  # the same inputs through a conv scaled by 1e6
    assert live["headroom"]["per_batch_max"]["max"] == pytest.approx(live["headroom"]["max_abs"])


def test_the_ema_weights_can_be_diagnosed_instead(tmp_path, synthetic_dataset):
    run = _make_fixture(tmp_path, synthetic_dataset, scale_first_conv=1e6)
    report = dn.run(_args(run, tmp_path / "r.json", weights="ema"))
    assert report["fixture"]["weights"] == "ema"
    assert report["survey"]["live"]["fp16_train"]["batches_non_finite"] == 0
    assert report["classification"] == "other"
    assert report["headroom"]["ratio_to_fp16_max"] < 1.0
