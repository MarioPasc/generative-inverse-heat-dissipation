"""The parametrised reverse chain and the batched ``sample_from_seeds`` (``04`` §4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ihdm.sampling.chain import (
    AMP_DTYPES,
    SampleRequest,
    prior_state,
    resolve_request,
    reverse_chain,
    sample_from_seeds,
)
from ihdm.sampling.errors import SamplingError
from ihdm.sampling.loader import build_heat_module


class ZeroModel(torch.nn.Module):
    """A network whose residual is always zero, so ``u_mean == u`` at every level."""

    def forward(self, x, levels):  # noqa: D102 - trivial test double
        return torch.zeros_like(x)


class CountingModel(torch.nn.Module):
    """Records the levels it is called with and returns a level-dependent constant."""

    def __init__(self) -> None:
        super().__init__()
        self.levels: list[int] = []

    def forward(self, x, levels):  # noqa: D102 - trivial test double
        self.levels.append(int(levels[0]))
        return torch.full_like(x, 0.01)


def test_resolve_request_defaults_come_from_the_config(tiny_config):
    """``None`` fields fall back to the config's delta, K and prior-noise switch."""
    delta, start_level, prior_noise = resolve_request(SampleRequest(n_per_seed=1), tiny_config)

    assert delta == pytest.approx(
        float(tiny_config.sampling.delta_factor) * float(tiny_config.model.sigma)
    )
    assert start_level == int(tiny_config.model.K)
    assert prior_noise is bool(tiny_config.sampling.prior_noise)


def test_resolve_request_overrides_win(tiny_config):
    """Explicit fields override the config."""
    request = SampleRequest(n_per_seed=1, delta=0.5, start_level=2, prior_noise=True)
    assert resolve_request(request, tiny_config) == (0.5, 2, True)


@pytest.mark.parametrize(
    "request_kwargs, match",
    [
        ({"n_per_seed": 0}, "n_per_seed must be positive"),
        ({"n_per_seed": 1, "batch_size": 0}, "batch_size must be positive"),
        ({"n_per_seed": 1, "delta": -1.0}, "delta must be non-negative"),
        ({"n_per_seed": 1, "start_level": 9}, r"start_level must lie in \[0, 8\]"),
        ({"n_per_seed": 1, "start_level": -1}, r"start_level must lie in \[0, 8\]"),
    ],
)
def test_resolve_request_validates(tiny_config, request_kwargs, match):
    """Every out-of-contract field is rejected before any tensor is allocated."""
    with pytest.raises(SamplingError, match=match):
        resolve_request(SampleRequest(**request_kwargs), tiny_config)


def test_prior_state_blurs_to_the_requested_level(tiny_config):
    """Without prior noise the prior state is exactly the seed blurred to ``level``."""
    heat = build_heat_module(tiny_config, "cpu")
    x = torch.rand(3, 1, 32, 32)

    for level in (0, 3, int(tiny_config.model.K)):
        expected = heat(x, torch.full((3,), level, dtype=torch.long)).float()
        torch.testing.assert_close(
            prior_state(heat, x, level, delta=0.0125, prior_noise=False), expected
        )


def test_prior_state_noise_is_generator_driven(tiny_config):
    """The prior noise comes from the explicit generator, so it is reproducible."""
    heat = build_heat_module(tiny_config, "cpu")
    x = torch.rand(2, 1, 32, 32)
    clean = prior_state(heat, x, 8, delta=0.5, prior_noise=False)

    first = prior_state(heat, x, 8, 0.5, True, torch.Generator().manual_seed(11))
    second = prior_state(heat, x, 8, 0.5, True, torch.Generator().manual_seed(11))
    other = prior_state(heat, x, 8, 0.5, True, torch.Generator().manual_seed(12))

    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, other)
    assert (first - clean).std().item() == pytest.approx(0.5, rel=0.15)


def test_reverse_chain_with_zero_model_and_no_noise_returns_the_prior(tiny_config):
    """``delta = 0`` and a zero residual leave the prior state untouched."""
    heat = build_heat_module(tiny_config, "cpu")
    x = torch.rand(2, 1, 32, 32)
    u_start = prior_state(heat, x, 8, delta=0.0, prior_noise=False)

    out = reverse_chain(ZeroModel(), u_start, start_level=8, delta=0.0)

    torch.testing.assert_close(out, u_start)


def test_reverse_chain_visits_every_level_downwards():
    """The loop runs ``start_level, ..., 1``, exactly like the released sampler."""
    model = CountingModel()
    reverse_chain(model, torch.zeros(1, 1, 8, 8), start_level=5, delta=0.0)
    assert model.levels == [5, 4, 3, 2, 1]


def test_reverse_chain_at_level_zero_is_the_identity():
    """A start level of 0 takes no step and returns its input."""
    model = CountingModel()
    u = torch.rand(2, 1, 8, 8)
    out = reverse_chain(model, u, start_level=0, delta=0.3)
    assert model.levels == []
    torch.testing.assert_close(out, u)


def test_reverse_chain_returns_the_mean_not_the_noised_state():
    """The released loop returns ``u_mean``; with a constant residual that is exact."""
    u = torch.zeros(1, 1, 8, 8)
    out = reverse_chain(CountingModel(), u, start_level=3, delta=0.0)
    torch.testing.assert_close(out, torch.full_like(u, 0.03))


def _model(tiny_run, tiny_config):
    from ihdm.sampling.loader import load_ema_model

    _, ckpt = tiny_run
    return load_ema_model(ckpt, tiny_config, torch.device("cpu"))


def test_sample_from_seeds_shape_and_range(tiny_run, tiny_config, tiny_dataset):
    """The stack is ``(N, n_per_seed, H, W)`` float32 in [0, 1]."""
    from ihdm.sampling.seeds import load_seed_images

    seeds, _ = load_seed_images(tiny_dataset, "seed", n=2, rng_seed=0)
    model = _model(tiny_run, tiny_config)

    samples = sample_from_seeds(
        model, tiny_config, seeds, SampleRequest(n_per_seed=3, batch_size=4), "cpu"
    )

    assert samples.shape == (2, 3, 32, 32)
    assert samples.dtype == np.float32
    assert samples.min() >= 0.0 and samples.max() <= 1.0


def test_sample_from_seeds_is_reproducible(tiny_run, tiny_config, tiny_dataset):
    """The same ``rng_seed`` reproduces the stack bitwise; a different one does not."""
    from ihdm.sampling.seeds import load_seed_images

    seeds, _ = load_seed_images(tiny_dataset, "seed", n=2, rng_seed=0)
    model = _model(tiny_run, tiny_config)
    tiny_config.sampling.prior_noise = True
    request = SampleRequest(n_per_seed=2, batch_size=3, rng_seed=5, delta=0.05)

    first = sample_from_seeds(model, tiny_config, seeds, request, "cpu")
    second = sample_from_seeds(model, tiny_config, seeds, request, "cpu")
    elsewhere = SampleRequest(n_per_seed=2, batch_size=3, rng_seed=6, delta=0.05)
    other = sample_from_seeds(model, tiny_config, seeds, elsewhere, "cpu")

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_deterministic_request_gives_identical_replicates(tiny_run, tiny_config, tiny_dataset):
    """``prior_noise=False`` and ``delta=0`` make the chain deterministic in its seed."""
    from ihdm.sampling.seeds import load_seed_images

    seeds, _ = load_seed_images(tiny_dataset, "seed", n=2, rng_seed=0)
    model = _model(tiny_run, tiny_config)

    samples = sample_from_seeds(
        model,
        tiny_config,
        seeds,
        SampleRequest(n_per_seed=3, delta=0.0, prior_noise=False, batch_size=4),
        "cpu",
    )

    for seed_row in range(samples.shape[0]):
        for replicate in range(1, samples.shape[1]):
            np.testing.assert_array_equal(samples[seed_row, 0], samples[seed_row, replicate])
    assert not np.array_equal(samples[0, 0], samples[1, 0])


def test_start_level_below_k_starts_from_the_blurred_seed(tiny_run, tiny_config, tiny_dataset):
    """``start_level=0`` with no noise returns the seed itself: the chain took no step."""
    from ihdm.sampling.seeds import load_seed_images

    seeds, _ = load_seed_images(tiny_dataset, "seed", n=2, rng_seed=0)
    model = _model(tiny_run, tiny_config)

    at_zero = sample_from_seeds(
        model,
        tiny_config,
        seeds,
        SampleRequest(n_per_seed=1, start_level=0, delta=0.0, prior_noise=False),
        "cpu",
    )
    at_two = sample_from_seeds(
        model,
        tiny_config,
        seeds,
        SampleRequest(n_per_seed=1, start_level=2, delta=0.0, prior_noise=False),
        "cpu",
    )

    np.testing.assert_allclose(
        at_zero[:, 0], seeds.astype(np.float32) / 255.0, rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(at_two[:, 0], at_zero[:, 0])


def test_batching_does_not_change_a_deterministic_result(tiny_run, tiny_config, tiny_dataset):
    """With no randomness the batch size only changes how the work is cut up."""
    from ihdm.sampling.seeds import load_seed_images

    seeds, _ = load_seed_images(tiny_dataset, "seed", n=2, rng_seed=0)
    model = _model(tiny_run, tiny_config)
    kwargs = {"n_per_seed": 2, "delta": 0.0, "prior_noise": False}

    small = sample_from_seeds(
        model, tiny_config, seeds, SampleRequest(batch_size=1, **kwargs), "cpu"
    )
    large = sample_from_seeds(
        model, tiny_config, seeds, SampleRequest(batch_size=16, **kwargs), "cpu"
    )

    np.testing.assert_allclose(small, large, rtol=1e-5, atol=1e-6)


def test_sample_from_seeds_rejects_a_bad_stack(tiny_run, tiny_config):
    """Seeds must be a non-empty ``(N, H, W)`` stack."""
    model = _model(tiny_run, tiny_config)
    flat = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(SamplingError, match=r"\(N, H, W\) stack"):
        sample_from_seeds(model, tiny_config, flat, SampleRequest(n_per_seed=1), "cpu")


# --------------------------------------------------------------------------------------------
# The autocast dtype (T5.1, ``--amp bf16``)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [None, "float16", "bfloat16"])
def test_resolve_request_accepts_the_known_autocast_dtypes(tiny_config, dtype):
    """``None`` (the device default) and the two half-precision dtypes pass validation."""
    request = SampleRequest(n_per_seed=1, amp=True, amp_dtype=dtype)
    resolve_request(request, tiny_config)


@pytest.mark.parametrize("dtype", ["float32", "fp16", "half", ""])
def test_resolve_request_rejects_an_unknown_autocast_dtype(tiny_config, dtype):
    """A dtype outside ``AMP_DTYPES`` is refused before any tensor is allocated."""
    with pytest.raises(SamplingError, match="amp_dtype must be one of"):
        resolve_request(SampleRequest(n_per_seed=1, amp=True, amp_dtype=dtype), tiny_config)


def test_the_default_request_keeps_autocast_off():
    """The defaults are today's behaviour: no autocast, no dtype."""
    request = SampleRequest(n_per_seed=1)
    assert request.amp is False and request.amp_dtype is None
    assert AMP_DTYPES == ("float16", "bfloat16")


def test_bf16_autocast_samples_are_valid(tiny_run, tiny_config, tiny_dataset):
    """``bfloat16`` runs the real network end to end and stays finite in [0, 1]."""
    from ihdm.sampling.seeds import load_seed_images

    seeds, _ = load_seed_images(tiny_dataset, "seed", n=2, rng_seed=0)
    model = _model(tiny_run, tiny_config)
    kwargs = {"n_per_seed": 2, "batch_size": 4, "rng_seed": 3, "delta": 0.05}

    fp32 = sample_from_seeds(model, tiny_config, seeds, SampleRequest(**kwargs), "cpu")
    bf16 = sample_from_seeds(
        model, tiny_config, seeds, SampleRequest(amp=True, amp_dtype="bfloat16", **kwargs), "cpu"
    )

    assert bf16.shape == fp32.shape and bf16.dtype == np.float32
    assert np.isfinite(bf16).all()
    assert bf16.min() >= 0.0 and bf16.max() <= 1.0


class AutocastProbe(torch.nn.Module):
    """Records whether autocast is on, and with which dtype, at every network call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[bool, torch.dtype]] = []

    def forward(self, x, levels):  # noqa: D102 - trivial test double
        self.seen.append((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")))
        return torch.zeros_like(x)


@pytest.mark.parametrize(
    "amp, dtype, expected",
    [
        (False, None, False),
        (True, "bfloat16", torch.bfloat16),
        (True, "float16", torch.float16),
    ],
)
def test_the_requested_autocast_dtype_reaches_the_network(tiny_config, amp, dtype, expected):
    """The network, and only the network, runs under autocast with the requested dtype."""
    probe = AutocastProbe()
    seeds = np.full((1, 32, 32), 128, dtype=np.uint8)
    request = SampleRequest(n_per_seed=1, batch_size=1, amp=amp, amp_dtype=dtype, start_level=2)
    sample_from_seeds(probe, tiny_config, seeds, request, "cpu")

    assert len(probe.seen) == 2
    for enabled, seen_dtype in probe.seen:
        assert enabled is bool(amp)
        if amp:
            assert seen_dtype == expected
