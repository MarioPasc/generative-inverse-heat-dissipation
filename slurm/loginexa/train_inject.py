"""TEST-ONLY driver: run the real trainer with non-finite losses injected at chosen steps.

Exercises the D19 skip and abort paths of ``train.py`` (``ihdm.train.guard``) end to end. The
injection lives here, never in ``train.py`` or ``scripts/``: it monkeypatches
``scripts.losses.get_step_fn`` and ``scripts.losses.get_inverse_heat_loss_fn`` inside this
process only, so the production entry point (``python train.py --config ...``, whose config is
locked by ``config_flags``) cannot trigger it. Used by ``tests/train/test_guard_train.py`` (CPU)
and by item H5 of ``slurm/loginexa/harness.sh`` (V100, real CUDA ``GradScaler``).

The injected loss is the true loss times NaN, inside the same autocast region, so the backward
pass produces NaN gradients exactly as an fp16 overflow in the network would: with an enabled
``GradScaler`` the step is skipped and the scale halved; with a disabled one ``optimizer.step()``
applies the NaN gradients.

``--cpu-grad-scaler`` replaces ``torch.cuda.amp.GradScaler`` (which disables itself without CUDA)
by ``torch.amp.GradScaler("cpu")``, which implements the same unscale/skip/update semantics on
the CPU; it lets the CPU test suite reach the skip branch. It is meaningless on a GPU.

Exit code: that of ``train.train`` (0, 3 for the guard, 4 for the recipe check).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from scripts import losses  # noqa: E402

_STATE = {"inject": False}


def _parse_steps(text: str) -> frozenset[int]:
    """Parse ``"3,4,10-19"`` into a set of loop steps."""
    steps: set[int] = set()
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            steps.update(range(low, high + 1))
        else:
            steps.add(int(part))
    return frozenset(steps)


def install_injection(steps: frozenset[int]) -> None:
    """Patch ``scripts.losses`` so the training loss is NaN at the loop steps in ``steps``.

    Parameters
    ----------
    steps : frozenset[int]
        Loop steps (``state["step"]`` at the start of the step) whose loss is made non-finite.
    """
    original_loss_factory = losses.get_inverse_heat_loss_fn
    original_step_factory = losses.get_step_fn

    def loss_factory(config, train, scales, device, heat_forward_module):
        inner = original_loss_factory(config, train, scales, device,
                                      heat_forward_module=heat_forward_module)
        if not train:
            return inner

        def loss_fn(model, batch):
            loss, per_sample, levels = inner(model, batch)
            if _STATE["inject"]:
                loss, per_sample = loss * float("nan"), per_sample * float("nan")
            return loss, per_sample, levels

        return loss_fn

    def step_factory(train, *args: Any, **kwargs: Any):
        step_fn = original_step_factory(train, *args, **kwargs)
        if not train:
            return step_fn

        def wrapped(state, batch):
            _STATE["inject"] = int(state["step"]) in steps
            return step_fn(state, batch)

        wrapped.scaler = step_fn.scaler
        return wrapped

    losses.get_inverse_heat_loss_fn = loss_factory
    losses.get_step_fn = step_factory


def _coerce(text: str) -> Any:
    """Turn a ``--set`` value into bool/int/float when it looks like one."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _build_config(args: argparse.Namespace):
    """The smoke config (with ``--smoke-data-root``) or an arm config (``--arm-spec``)."""
    if args.smoke_data_root:
        from configs.spectral import smoke

        config = smoke.get_config()
        config.data.root = args.smoke_data_root
    else:
        from configs.spectral import arms

        config = arms.get_config(args.arm_spec)
    for item in args.set:
        key, value = item.split("=", 1)
        node = config
        *parents, leaf = key.split(".")
        for part in parents:
            node = getattr(node, part)
        setattr(node, leaf, _coerce(value))
    return config


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, install the injection and run ``train.train``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke-data-root", help="data root holding the synthetic dataset")
    source.add_argument("--arm-spec", help='arm config spec, e.g. "ixi,A0"')
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--inject", default="", help='loop steps to poison, e.g. "3,5-14"')
    parser.add_argument("--set", action="append", default=[], help="dotted.key=value override")
    parser.add_argument("--cpu-grad-scaler", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(logging.INFO)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.cpu_grad_scaler:
        torch.cuda.amp.GradScaler = lambda *a, **k: torch.amp.GradScaler("cpu", *a, **k)
    install_injection(_parse_steps(args.inject))
    config = _build_config(args)
    print(f"[train_inject] inject={sorted(_parse_steps(args.inject))} "
          f"cpu_grad_scaler={args.cpu_grad_scaler} device={config.device}", flush=True)

    import train as trainer

    trainer.train(config, args.workdir)


if __name__ == "__main__":
    main()
