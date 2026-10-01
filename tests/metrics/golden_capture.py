"""Capture (or recompute) the regression goldens of T7.2: run at the base commit to freeze them.

``python -m tests.metrics.golden_capture --write`` writes ``tests/metrics/data/golden_t72.json``;
the tests in ``test_regression_t72.py`` recompute the same digests with :func:`compute` and
compare. Two families:

* ``metric_layer``: the spectral, diversity and memorisation blocks on fixed numpy stacks at
  ``W = 192`` and ``W = 96``, as exact ``float.hex`` strings (pure numpy/scipy, so bitwise
  reproducible on one build);
* ``default_path``: the file tree, the JSON key structure and the normalised sha256 of every
  file ``evaluate_run`` writes with the default request (no ``--delta``) on the fixture of
  ``test_run_eval`` at ``W = 96`` (both checkpoints; the legacy-grid branch) and at ``W = 192``
  (the final checkpoint only, to keep the test under a minute), together with the digest of every
  ``samples.npy`` so the comparison can tell "the sampler differs on this platform" (skip) from
  "the metric layer changed" (fail).

The fixture's side is switched by setting ``tests.metrics.test_run_eval.IMAGE_SIZE`` for the
duration of the call, because the base-commit helpers read that module global and take no size
argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_t72.json"

#: Keys whose values change on every call (timestamps, timings, provenance of the checkout), and
#: the two digests that depend on the temporary base directory: the fixture's config holds
#: ``data.root``, its sha256 is pickled into every checkpoint, so both ``config_sha256`` and
#: ``checkpoint_sha256`` change with the directory the fixture is written to.
VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "created", "git_sha", "elapsed_s", "s_per_chain", "python",
        "config_sha256", "checkpoint_sha256",
    }
)


def _hex(value: Any) -> Any:
    """Return floats as exact hex strings, recursively."""
    if isinstance(value, float):
        return float(value).hex()
    if isinstance(value, dict):
        return {str(k): _hex(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hex(v) for v in value]
    if isinstance(value, np.ndarray):
        return _hex(value.tolist())
    if isinstance(value, np.floating):
        return float(value).hex()
    if isinstance(value, np.integer):
        return int(value)
    return value


def _stacks(width: int) -> dict[str, np.ndarray]:
    """Deterministic uint8 stacks with a power-law-ish spectrum at side ``width``."""
    rng = np.random.default_rng(1234 + width)
    from scipy.fft import idctn

    idx = np.arange(width)
    radius = np.sqrt(idx[:, None] ** 2 + idx[None, :] ** 2)
    weight = 1.0 / np.maximum(radius, 1.0)

    def draw(n: int, gain: float) -> np.ndarray:
        coeffs = rng.normal(size=(n, width, width)) * weight * gain
        images = idctn(coeffs, axes=(1, 2), norm="ortho") + 0.5
        return np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8)

    return {
        "reference": draw(12, 8.0),
        "samples": draw(10, 6.0),
        "train": draw(14, 8.0),
        "heldout": draw(6, 8.0),
        "seeds": draw(3, 8.0),
        "per_seed": draw(12, 6.0).reshape(3, 4, width, width),
    }


def metric_layer(width: int) -> dict[str, Any]:
    """The metric blocks of ``run_eval`` on fixed stacks at side ``width``."""
    from ihdm.metrics.diversity import within_seed_diversity
    from ihdm.metrics.memorisation import memorisation_ratio
    from ihdm.metrics.run_eval import _inherited_record, _lsd_record
    from ihdm.spectral.power import mode_power

    s = _stacks(width)
    ref_power = mode_power(s["reference"].astype(np.float32) / 255.0)
    out: dict[str, Any] = {
        "lsd": _lsd_record(s["samples"], s["reference"]),
        "inherited": _inherited_record(s["per_seed"], s["seeds"], ref_power, width / 2.0),
    }
    div = within_seed_diversity(s["per_seed"], sigma_lp=16.0)
    out["diversity"] = {
        "D_pix": div.D_pix_mean,
        "D_lp": div.D_lp_mean,
        "per_seed_pix": list(div.per_seed_pix),
        "per_seed_lp": list(div.per_seed_lp),
    }
    subjects = [f"S{i // 2}" for i in range(s["train"].shape[0])]
    mem = memorisation_ratio(
        s["samples"], s["train"], s["heldout"], subjects, np.arange(10) % 14,
        sigma_lp=16.0, device="cpu",
    )
    out["memorisation"] = {"M": mem.M, "M_lp": mem.M_lp, "seed_nn_fraction": mem.seed_nn_fraction}
    return _hex(out)


def _normalise(value: Any, base: str) -> Any:
    """Drop volatile keys and replace the temporary base directory in strings."""
    if isinstance(value, dict):
        return {k: _normalise(v, base) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_normalise(v, base) for v in value]
    if isinstance(value, str):
        return value.replace(base, "<BASE>")
    return value


def _key_tree(value: Any) -> Any:
    """The nested key structure of a JSON value (lists reduced to their first element)."""
    if isinstance(value, dict):
        return {k: _key_tree(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_key_tree(value[0])] if value and isinstance(value[0], (dict, list)) else "list"
    return type(value).__name__


def digest_tree(workdir: Path, base: Path) -> dict[str, Any]:
    """Digest every file ``evaluate_run`` wrote under ``workdir`` (samples*/ and metrics*/)."""
    files: dict[str, Any] = {}
    for path in sorted(workdir.rglob("*")):
        rel = path.relative_to(workdir).as_posix()
        if not path.is_file() or not rel.startswith(("samples", "metrics")):
            continue
        if path.suffix == ".json":
            payload = _normalise(json.loads(path.read_text()), str(base))
            text = json.dumps(payload, sort_keys=True)
            files[rel] = {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "keys": _key_tree(json.loads(path.read_text())),
            }
        else:
            files[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return files


#: Checkpoints evaluated by the default-path golden at each fixture side.
DEFAULT_PATH_CKPTS: dict[int, str] = {96: "250,750", 192: "750"}


def default_path(base: Path, image_size: int = 96) -> dict[str, Any]:
    """Run the default ``evaluate_run`` on the fixture of side ``image_size`` and digest it."""
    from ihdm.metrics.run_eval import EvalRequest, evaluate_run
    from tests.metrics import test_run_eval as fixture

    previous = fixture.IMAGE_SIZE
    fixture.IMAGE_SIZE = int(image_size)
    try:
        dataset = fixture._build_dataset(base / "data")
        config = fixture._config(dataset)
        workdir = fixture._write_run(base / "runs" / config.run_id, config)
        evaluate_run(
            EvalRequest(
                run=workdir, ckpts=DEFAULT_PATH_CKPTS[int(image_size)], n_lsd=8, n_final=8,
                n_seeds=2, n_per_seed=3, sample_batch=8, skip_inception=True, device="cpu",
            )
        )
    finally:
        fixture.IMAGE_SIZE = previous
    return digest_tree(workdir, base)


def compute(include_default_path: bool = True) -> dict[str, Any]:
    """Return every golden quantity."""
    out: dict[str, Any] = {
        "metric_layer": {str(w): metric_layer(w) for w in (192, 96)},
    }
    if include_default_path:
        out["default_path"] = {}
        for size in DEFAULT_PATH_CKPTS:
            with tempfile.TemporaryDirectory() as tmp:
                out["default_path"][str(size)] = default_path(Path(tmp).resolve(), size)
    return out


def main() -> None:
    """Print or write the goldens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {GOLDEN_PATH}")
    args = parser.parse_args()
    golden = compute()
    if args.write:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN_PATH}")
    else:
        print(json.dumps(golden, indent=1, sort_keys=True)[:2000])


if __name__ == "__main__":
    main()
