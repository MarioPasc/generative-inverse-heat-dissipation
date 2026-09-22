"""The prior-noise fix (decision D3, ``04-run-artifacts.md`` §5).

The paper's prior is $p(u_K) = \\mathcal N(u_K, \\delta^2 I)$; the released
``scripts.sampling.get_initial_sample`` returned the noiseless blur of a training batch. With
``config.sampling.prior_noise`` the draw becomes stochastic.

Runs in-process on the CPU: no model is built, so ``torch.nn.DataParallel`` (the reason the smoke
test needs a subprocess) is not involved.
"""

from __future__ import annotations

import torch

from configs.spectral import smoke
from model_code import utils as mutils
from scripts import sampling

DELTA = 0.0125  # config.model.sigma * config.sampling.delta_factor


def _config(dataset_root, prior_noise: bool):
    config = smoke.get_config()
    config.data.root = str(dataset_root.parent)
    config.sampling.prior_noise = prior_noise
    return config


def _draw(config, seed: int) -> torch.Tensor:
    heat = mutils.create_forward_process_from_sigmas(
        config, config.model.blur_schedule, config.device)
    torch.manual_seed(seed)
    prior, _ = sampling.get_initial_sample(config, heat, DELTA)
    return prior


def test_without_the_flag_the_prior_is_deterministic(synthetic_dataset):
    config = _config(synthetic_dataset, prior_noise=False)
    torch.testing.assert_close(_draw(config, seed=0), _draw(config, seed=1))


def test_with_the_flag_two_draws_differ(synthetic_dataset):
    config = _config(synthetic_dataset, prior_noise=True)
    first, second = _draw(config, seed=0), _draw(config, seed=1)
    assert not torch.allclose(first, second)


def test_the_flag_only_adds_noise_of_the_right_scale(synthetic_dataset):
    noiseless = _draw(_config(synthetic_dataset, prior_noise=False), seed=0)
    noisy = _draw(_config(synthetic_dataset, prior_noise=True), seed=0)
    assert not torch.allclose(noiseless, noisy)
    residual = (noisy - noiseless).std().item()
    # 4000 pixels per draw: the sample standard deviation of the added noise is DELTA to ~5%.
    assert abs(residual - DELTA) / DELTA < 0.15


def test_original_images_are_returned_unblurred(synthetic_dataset):
    config = _config(synthetic_dataset, prior_noise=True)
    heat = mutils.create_forward_process_from_sigmas(
        config, config.model.blur_schedule, config.device)
    torch.manual_seed(0)
    prior, originals = sampling.get_initial_sample(config, heat, DELTA)
    assert originals.shape == prior.shape
    assert originals.min() >= 0.0 and originals.max() <= 1.0
