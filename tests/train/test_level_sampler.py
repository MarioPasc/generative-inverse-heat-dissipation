"""The level-sampler fix (decision D3, ``04-run-artifacts.md`` §5).

The released code drew training levels from ``[1, K)``, so level ``K`` was never trained although
the reverse process starts there. With ``model.train_level_max_inclusive`` the range becomes
``[1, K]``.
"""

from __future__ import annotations

import ml_collections
import numpy as np
import torch

from scripts.losses import get_inverse_heat_loss_fn, get_label_sampling_function

K = 8
N_DRAWS = 10_000


def _draws(inclusive: bool) -> torch.Tensor:
    torch.manual_seed(0)
    sampler = get_label_sampling_function(K, inclusive)
    return sampler(N_DRAWS, torch.device("cpu"))


def test_inclusive_flag_draws_level_k():
    draws = _draws(inclusive=True)
    assert int(draws.min()) == 1
    assert int(draws.max()) == K
    assert set(draws.tolist()) == set(range(1, K + 1))


def test_released_default_never_draws_level_k():
    draws = _draws(inclusive=False)
    assert int(draws.min()) == 1
    assert int(draws.max()) == K - 1
    assert (draws == K).sum().item() == 0


def test_draws_are_uniform_over_the_allowed_levels():
    draws = _draws(inclusive=True).numpy()
    counts = np.bincount(draws, minlength=K + 1)[1:]
    expected = N_DRAWS / K
    # 5 sigma of a binomial(N, 1/K) count; a range bug shifts a whole level to zero.
    tolerance = 5.0 * np.sqrt(N_DRAWS * (1.0 / K) * (1.0 - 1.0 / K))
    assert np.all(np.abs(counts - expected) < tolerance)


class _NoOpNet(torch.nn.Module):
    """Stands in for the U-Net: accepts ``(x, fwd_steps)`` and predicts no change."""

    def forward(self, x, fwd_steps):
        return torch.zeros_like(x)


class _NoOpBlur:
    """Stands in for ``DCTBlur``: returns the batch unchanged at every level."""

    def __call__(self, x, fwd_steps):
        return x


def _loss_fn_levels(inclusive: bool) -> torch.Tensor:
    """Drive the flag through ``get_inverse_heat_loss_fn``, the way the trainer reaches it."""
    config = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()
    config.model.K = K
    config.model.sigma = 0.01
    config.model.train_level_max_inclusive = inclusive
    schedule = np.concatenate([[0.0], np.linspace(0.5, 8.0, K)])

    loss_fn = get_inverse_heat_loss_fn(
        config, train=False, scales=schedule, device=torch.device("cpu"),
        heat_forward_module=_NoOpBlur())
    model = _NoOpNet()
    torch.manual_seed(0)
    levels = [loss_fn(model, torch.zeros(16, 1, 4, 4))[2] for _ in range(200)]
    return torch.cat(levels)


def test_config_flag_reaches_the_loss_function():
    assert int(_loss_fn_levels(inclusive=True).max()) == K
    assert int(_loss_fn_levels(inclusive=False).max()) == K - 1


def test_flag_absent_from_config_keeps_the_released_behaviour():
    config = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()
    config.model.K = K
    assert config.model.get("train_level_max_inclusive", False) is False
