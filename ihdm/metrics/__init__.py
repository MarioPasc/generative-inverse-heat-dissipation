"""Metric definitions of the experiment (``docs/SPECIFICATIONS/05-metrics.md``).

``spectral`` holds the fidelity and mechanism endpoints — the fine radial profile, the
log-spectral distance with its octave profile, ``T_tau`` and the inherited band; ``lowpass`` the
released heat kernel as a pure numpy low-pass; ``memorisation`` the ratio ``M`` and the seed
nearest-neighbour fraction; ``diversity`` the within-seed diversity and the PCA around a seed;
``inception`` the KID/FID/recall/coverage of §7, computed in memory; ``run_eval`` the evaluation
of one whole run; ``io`` holds the result-file contract of §9 that every metric module writes
through.

``run_eval`` is **not** re-exported here: it is the evaluation of a whole run, it imports the
sampler and the released ``model_code`` package, and nothing that only wants a metric should pay
for that. Import it as ``ihdm.metrics.run_eval``.
"""

from ihdm.metrics.diversity import DivResult, PcaResult, pca_around_seed, within_seed_diversity
from ihdm.metrics.errors import MetricError
from ihdm.metrics.inception import (
    InceptionResult,
    bootstrap_inception,
    fid_from_features,
    inception_features,
    inception_weights_path,
    kid_from_features,
    recall_coverage,
    reference_features,
)
from ihdm.metrics.io import read_json, write_json
from ihdm.metrics.lowpass import dct_lowpass, remove_dc, to_float_stack
from ihdm.metrics.memorisation import MemResult, memorisation_ratio, nn_distances
from ihdm.metrics.spectral import (
    InheritedResult,
    LsdResult,
    inherited_band,
    log_bin_centres,
    log_bin_edges,
    lsd,
    radial_log_profile,
    t_tau,
)

__all__ = [
    "DivResult",
    "InceptionResult",
    "InheritedResult",
    "LsdResult",
    "MemResult",
    "MetricError",
    "PcaResult",
    "bootstrap_inception",
    "dct_lowpass",
    "fid_from_features",
    "inception_features",
    "inception_weights_path",
    "inherited_band",
    "kid_from_features",
    "log_bin_centres",
    "log_bin_edges",
    "lsd",
    "memorisation_ratio",
    "nn_distances",
    "pca_around_seed",
    "radial_log_profile",
    "read_json",
    "recall_coverage",
    "reference_features",
    "remove_dc",
    "t_tau",
    "to_float_stack",
    "within_seed_diversity",
    "write_json",
]
