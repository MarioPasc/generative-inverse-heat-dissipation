"""Offline sampling from an EMA checkpoint with given seed images (``04-run-artifacts.md`` §4).

This is the only sampling path the metrics of ``05-metrics.md`` use: the spectral endpoints start
from training seeds, the diversity, memorisation-mechanism and inherited-band endpoints from the
40 held-out seed subjects.
"""

from ihdm.sampling.chain import (
    SampleRequest,
    prior_state,
    resolve_request,
    reverse_chain,
    sample_from_seeds,
)
from ihdm.sampling.errors import SamplingError
from ihdm.sampling.loader import (
    build_heat_module,
    checkpoint_sha256,
    load_checkpoint,
    load_ema_model,
    load_run_config,
    resolve_checkpoint_path,
)
from ihdm.sampling.seeds import SEED_SLICE, load_seed_images

__all__ = [
    "SEED_SLICE",
    "SampleRequest",
    "SamplingError",
    "build_heat_module",
    "checkpoint_sha256",
    "load_checkpoint",
    "load_ema_model",
    "load_run_config",
    "load_seed_images",
    "prior_state",
    "resolve_checkpoint_path",
    "resolve_request",
    "reverse_chain",
    "sample_from_seeds",
]
