"""The reverse chain of the inverse heat dissipation model, parametrised by the start level.

This is a copy of the released ``scripts.sampling.get_sampling_fn_inverse_heat`` (Rissanen,
Heinonen and Solin, *Generative modelling with inverse heat dissipation*, ICLR 2023;
https://github.com/AaltoML/generative-inverse-heat-dissipation), kept step for step:

.. code-block:: text

    for i in range(K, 0, -1):
        u_mean = model(u, i) + u
        u      = u_mean + delta * noise
    return u_mean

Three things differ, and they are the reason the released function cannot be reused as is
(``04-run-artifacts.md`` §4):

* the chain starts at an arbitrary ``start_level`` instead of always at ``K``, which the
  start-level sweep needs;
* the prior state is built from *given* seed images instead of a batch the released
  ``get_initial_sample`` draws from the training loader itself;
* every noise draw comes from an explicit :class:`torch.Generator`, so one ``rng_seed``
  reproduces a sample stack bitwise instead of depending on the process-global RNG.

The released file is left untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ihdm.sampling.errors import SamplingError
from ihdm.sampling.loader import build_heat_module

__all__ = ["SampleRequest", "prior_state", "resolve_request", "reverse_chain", "sample_from_seeds"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleRequest:
    """One sampling job (``04-run-artifacts.md`` §4).

    Parameters
    ----------
    n_per_seed : int
        Samples drawn from each seed: 1 for the fidelity/LSD and memorisation endpoints, 50 for
        the diversity and inherited-band endpoints.
    delta : float | None
        Standard deviation of the sampling noise. ``None`` uses
        ``config.sampling.delta_factor * config.model.sigma``.
    start_level : int | None
        Level the chain starts from. ``None`` uses ``config.model.K``; a smaller value blurs the
        seed only to that level and runs the correspondingly shorter chain.
    prior_noise : bool | None
        Whether the prior draw adds ``N(0, delta^2 I)`` (D3). ``None`` uses
        ``config.sampling.prior_noise``.
    batch_size : int
        How many ``(seed, replicate)`` pairs are run through the network at once.
    rng_seed : int
        Seed of the ``torch.Generator`` that draws the prior and sampling noise.
    amp : bool
        Run the network under ``torch.autocast`` on the sampling device.
    """

    n_per_seed: int
    delta: float | None = None
    start_level: int | None = None
    prior_noise: bool | None = None
    batch_size: int = 64
    rng_seed: int = 0
    amp: bool = False


def resolve_request(request: SampleRequest, config: Any) -> tuple[float, int, bool]:
    """Fill the ``None`` fields of a request from the run's config.

    Parameters
    ----------
    request : SampleRequest
        The request as given by the caller.
    config : ml_collections.ConfigDict
        The run's config.

    Returns
    -------
    tuple[float, int, bool]
        ``(delta, start_level, prior_noise)``.

    Raises
    ------
    SamplingError
        If ``n_per_seed`` or ``batch_size`` is not positive, ``delta`` is negative, or
        ``start_level`` falls outside ``[0, K]``.
    """
    if request.n_per_seed <= 0:
        raise SamplingError(f"n_per_seed must be positive, got {request.n_per_seed}")
    if request.batch_size <= 0:
        raise SamplingError(f"batch_size must be positive, got {request.batch_size}")

    k = int(config.model.K)
    delta = (
        float(config.sampling.delta_factor) * float(config.model.sigma)
        if request.delta is None
        else float(request.delta)
    )
    if delta < 0.0:
        raise SamplingError(f"delta must be non-negative, got {delta}")
    start_level = k if request.start_level is None else int(request.start_level)
    if not 0 <= start_level <= k:
        raise SamplingError(f"start_level must lie in [0, {k}], got {start_level}")
    prior_noise = (
        bool(config.sampling.get("prior_noise", False))
        if request.prior_noise is None
        else bool(request.prior_noise)
    )
    return delta, start_level, prior_noise


def _noise_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Draw standard normal noise shaped like ``x`` from ``generator``."""
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def prior_state(
    heat_module: torch.nn.Module,
    x: torch.Tensor,
    level: int,
    delta: float,
    prior_noise: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return the prior draw of the chain: ``x`` blurred to ``level``, plus optional noise.

    This is the seed-driven equivalent of ``scripts.sampling.get_initial_sample`` with the D3
    hook enabled, except that the images are supplied by the caller.

    Parameters
    ----------
    heat_module : torch.nn.Module
        The run's ``model_code.utils.DCTBlur``.
    x : torch.Tensor
        ``(B, C, H, W)`` float tensor in ``[0, 1]`` on the sampling device.
    level : int
        The blur level to start from; ``0`` leaves ``x`` unblurred.
    delta : float
        Standard deviation of the prior noise.
    prior_noise : bool
        Whether to add ``N(0, delta^2 I)`` to the blurred images (D3).
    generator : torch.Generator | None
        Source of the prior noise.

    Returns
    -------
    torch.Tensor
        ``(B, C, H, W)`` float32 prior state.
    """
    levels = torch.full((x.shape[0],), int(level), dtype=torch.long, device=x.device)
    state = heat_module(x, levels).float()
    if prior_noise:
        state = state + delta * _noise_like(state, generator)
    return state


def reverse_chain(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    u_start: torch.Tensor,
    start_level: int,
    delta: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the reverse chain from ``start_level`` down to 1 and return the last mean.

    Parameters
    ----------
    model_fn : Callable
        ``(u, levels) -> residual``; the network's prediction is added to its input, as in the
        released loop.
    u_start : torch.Tensor
        ``(B, C, H, W)`` prior state.
    start_level : int
        The level ``u_start`` lives at. ``0`` means no reverse step is taken and ``u_start`` is
        returned unchanged.
    delta : float
        Standard deviation of the sampling noise added after each step.
    generator : torch.Generator | None
        Source of the sampling noise.

    Returns
    -------
    torch.Tensor
        ``(B, C, H, W)`` the mean of the last step, matching the released sampler, which returns
        ``u_mean`` rather than the noised ``u``.
    """
    u = u_start.float()
    u_mean = u
    for level in range(int(start_level), 0, -1):
        levels = torch.full((u.shape[0],), level, dtype=torch.long, device=u.device)
        u_mean = model_fn(u, levels) + u
        u = u_mean + delta * _noise_like(u_mean, generator)
    return u_mean


def _seed_batch(
    seeds: torch.Tensor, flat_start: int, flat_stop: int, n_per_seed: int
) -> torch.Tensor:
    """Return the seed images of the flat ``(seed, replicate)`` slice."""
    positions = torch.arange(flat_start, flat_stop, device=seeds.device) // n_per_seed
    return seeds.index_select(0, positions)


def sample_from_seeds(
    model: torch.nn.Module,
    config: Any,
    seeds_u8: np.ndarray,
    request: SampleRequest,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Draw ``request.n_per_seed`` samples from each seed image.

    The ``(seed, replicate)`` pairs are enumerated in row-major order and cut into batches of
    ``request.batch_size``, so the same ``rng_seed`` and the same batch size reproduce a stack
    bitwise on the same device.

    Parameters
    ----------
    model : torch.nn.Module
        The EMA network, already on ``device`` and in eval mode.
    config : ml_collections.ConfigDict
        The run's config; the blur schedule, ``model.K``, ``model.sigma`` and ``sampling.*`` are
        read from it.
    seeds_u8 : np.ndarray
        ``(N, H, W)`` ``uint8`` seed images.
    request : SampleRequest
        The sampling job.
    device : torch.device | str
        Where the chain runs.

    Returns
    -------
    np.ndarray
        ``(N, n_per_seed, H, W)`` ``float32`` in ``[0, 1]`` (clipped).

    Raises
    ------
    SamplingError
        If the seed array is not a ``(N, H, W)`` stack or the request is inconsistent with the
        config (see :func:`resolve_request`).
    """
    from model_code.utils import get_model_fn

    seeds_u8 = np.asarray(seeds_u8)
    if seeds_u8.ndim != 3 or seeds_u8.shape[0] == 0:
        raise SamplingError(f"seeds must be a non-empty (N, H, W) stack, got {seeds_u8.shape}")
    delta, start_level, prior_noise = resolve_request(request, config)

    device = torch.device(device)
    n_seeds, height, width = seeds_u8.shape
    n_per_seed = int(request.n_per_seed)
    total = n_seeds * n_per_seed

    heat_module = build_heat_module(config, device)
    model_fn = get_model_fn(model, train=False)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(request.rng_seed))

    seeds = torch.from_numpy(seeds_u8.astype(np.float32) / 255.0)[:, None].to(device)
    out = np.empty((total, height, width), dtype=np.float32)

    with torch.inference_mode():
        for flat_start in range(0, total, int(request.batch_size)):
            flat_stop = min(flat_start + int(request.batch_size), total)
            batch = _seed_batch(seeds, flat_start, flat_stop, n_per_seed)
            # The prior draw stays in float32: the DCT blur at sigma_B = 96 px spans many orders
            # of magnitude in the mode weights, which half precision cannot carry. Only the
            # network evaluations run under autocast.
            state = prior_state(heat_module, batch, start_level, delta, prior_noise, generator)
            with torch.autocast(device_type=device.type, enabled=bool(request.amp)):
                samples = reverse_chain(model_fn, state, start_level, delta, generator)
            out[flat_start:flat_stop] = (
                samples.float().clamp(0.0, 1.0).squeeze(1).detach().cpu().numpy()
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return out.reshape(n_seeds, n_per_seed, height, width)
