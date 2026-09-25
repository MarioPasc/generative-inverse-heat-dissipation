"""The recipe check on resume (decision D19, ``04-run-artifacts.md`` §3.2 (c)).

A resubmission resumes whatever run directory it is pointed at. Array 1 showed why that needs a
guard: its run directories were left behind, and a v2 submission (lr 1e-4) on top of them would
have continued the v1 runs (lr 2e-4) silently, producing a hybrid that no table could describe.

The *recipe* of a run is its resolved config minus the keys that may legitimately change between
submissions of the same run: the iteration count (the plateau-gate extension, D10) and the
cadences (a harness may shorten ``resume_every``), under both their new and their released names.
Everything else (``optim.*``, ``training.batch_size``, ``model.*`` with the schedule and its hash,
``data.*``, ``sampling.*``, ``eval.*``, ``seed``, ``device``, the arm) must be identical, or the
trainer aborts before writing anything.

The reference is the ``recipe_sha256`` the run's first start stored in ``manifest.json``; the
per-key difference that makes the error message useful is computed against the run's
``config.json``, which holds the same resolved config. A manifest without ``recipe_sha256``
(written before D19) is checked key by key against ``config.json`` alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ihdm.train.errors import TrainError
from ihdm.train.manifest import config_to_json

__all__ = [
    "RECIPE_EXCLUDED_KEYS",
    "RECIPE_EXIT_CODE",
    "RecipeMismatchError",
    "check_resume_recipe",
    "flatten",
    "recipe_diff",
    "recipe_of",
    "recipe_sha256",
]

#: Exit code of a resume refused by the recipe check (3 is the non-finite-loss guard).
RECIPE_EXIT_CODE = 4

#: Dotted keys excluded from the recipe: the iteration count, the cadences under their new and
#: released names, and ``run_id`` (derived from ``data.dataset``, ``arm`` and ``seed``, which are
#: all compared).
RECIPE_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "training.n_iters",
        "training.ckpt_every",
        "training.resume_every",
        "training.log_every",
        "training.eval_every",
        "training.grid_every",
        "training.snapshot_freq",
        "training.snapshot_freq_for_preemption",
        "training.log_freq",
        "training.eval_freq",
        "training.sampling_freq",
        "run_id",
    }
)


class RecipeMismatchError(TrainError):
    """Raised when a resume would continue a run with a different recipe."""


def flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested JSON dict into ``{"a.b.c": leaf}``; lists stay leaves.

    Parameters
    ----------
    tree : dict[str, Any]
        A JSON-like nested dict (``config.json``, or :func:`ihdm.train.manifest.config_to_json`).
    prefix : str
        The dotted path of ``tree`` itself.

    Returns
    -------
    dict[str, Any]
        One entry per leaf.
    """
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def recipe_of(config_json: dict[str, Any]) -> dict[str, Any]:
    """Return the recipe (flattened, excluded keys dropped) of a JSON-form config.

    Parameters
    ----------
    config_json : dict[str, Any]
        The config as :func:`ihdm.train.manifest.config_to_json` returns it, or as read back
        from ``config.json`` (the two are equal: the JSON round trip of floats is exact).

    Returns
    -------
    dict[str, Any]
        ``{dotted_key: value}`` for every key of the recipe.
    """
    return {k: v for k, v in flatten(config_json).items() if k not in RECIPE_EXCLUDED_KEYS}


def recipe_sha256(config: Any) -> str:
    """Return the SHA-256 of the canonical JSON form of the recipe of ``config``.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        The resolved config.

    Returns
    -------
    str
        Hex digest; stored in ``manifest.json`` as ``recipe_sha256``.
    """
    payload = json.dumps(recipe_of(config_to_json(config)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _short(value: Any, limit: int = 80) -> str:
    """Render a value for an error line, truncating long lists (the schedule has 201 floats)."""
    text = json.dumps(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def recipe_diff(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """List the keys whose value differs between two recipes, one readable line per key.

    Parameters
    ----------
    saved : dict[str, Any]
        The recipe the run was started with.
    current : dict[str, Any]
        The recipe of the invocation that tries to resume it.

    Returns
    -------
    list[str]
        ``"<key>: <saved> -> <current>"`` lines, sorted by key; empty when the recipes agree.
    """
    missing = object()
    lines = []
    for key in sorted(set(saved) | set(current)):
        before, after = saved.get(key, missing), current.get(key, missing)
        if before != after:
            left = "<absent>" if before is missing else _short(before)
            right = "<absent>" if after is missing else _short(after)
            lines.append(f"{key}: {left} -> {right}")
    return lines


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON object at ``path``, or ``None`` if the file does not exist."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RecipeMismatchError(f"{path} is not valid JSON ({error})") from error


def check_resume_recipe(workdir: Path, config: Any) -> None:
    """Refuse to resume ``workdir`` with a config whose recipe differs from the run's.

    Reads only; writes nothing. Called by ``train.py`` before the manifest, ``config.json``,
    the tensorboard writer or the ``resume`` line are touched.

    Parameters
    ----------
    workdir : Path
        The run directory being resumed (it holds a rolling checkpoint).
    config : ml_collections.ConfigDict
        The resolved config of the current invocation.

    Raises
    ------
    RecipeMismatchError
        If the manifest's ``recipe_sha256`` differs from the current one, if the recipe stored
        in ``config.json`` differs key by key, or if neither file exists (the recipe of the
        checkpoint cannot be established).
    """
    workdir = Path(workdir)
    manifest = _read_json(workdir / "manifest.json")
    saved_config = _read_json(workdir / "config.json")
    current = recipe_of(config_to_json(config))
    diff = recipe_diff(recipe_of(saved_config), current) if saved_config is not None else []

    stored_hash = (manifest or {}).get("recipe_sha256")
    if stored_hash is not None:
        if stored_hash != recipe_sha256(config):
            detail = "; ".join(diff) if diff else "config.json does not show which key changed"
            raise RecipeMismatchError(
                f"recipe of {workdir} (manifest recipe_sha256 {stored_hash[:12]}) differs from "
                f"this invocation's ({recipe_sha256(config)[:12]}): {detail}"
            )
        return
    if saved_config is None:
        raise RecipeMismatchError(
            f"{workdir} holds a rolling checkpoint but neither a manifest with recipe_sha256 nor a "
            "config.json: the recipe it was trained with cannot be established"
        )
    if diff:
        raise RecipeMismatchError(
            f"recipe of {workdir} (config.json) differs from this invocation's: " + "; ".join(diff)
        )
