"""H1 of the loginexa harness: the V100 runs the production training step of the real U-Net.

Checks, printing one ``H1 <check> PASS|FAIL <evidence>`` line each:

1. the device is a V100 (compute capability 7.0) and the overlay's torch carries sm_70 kernels;
2. five production train steps of ``ixi,A0`` at batch 16 (``configs/spectral/arms.py``, fp16
   autocast forward, scaled backward, unscale, pre-clip norm, clip, ``GradScaler`` step) through
   the released ``scripts.losses.get_step_fn`` and ``create_model``, on real batches of the
   ``train`` split: finite losses, the D19 ``grad_norm`` hook, the scale, the peak memory;
3. one EMA evaluation step (the ``eval`` path of ``get_step_fn``).

Run under ``timeout 25m`` with ``CUDA_VISIBLE_DEVICES`` pinned by ``slurm/loginexa/harness.sh``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def report(name: str, ok: bool, evidence: str) -> bool:
    print(f"H1 {name} {'PASS' if ok else 'FAIL'} {evidence}", flush=True)
    return ok


def main() -> int:
    from configs.spectral import arms
    from model_code import utils as mutils
    from model_code.ema import ExponentialMovingAverage
    from scripts import datasets, losses

    ok = True
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    arch = torch.cuda.get_arch_list()
    ok &= report("device", cap == (7, 0) and "sm_70" in arch,
                 f"{name} capability={cap} torch={torch.__version__} cuda={torch.version.cuda} "
                 f"arch={arch} cudnn={torch.backends.cudnn.version()}")

    config = arms.get_config("ixi,A0")
    torch.manual_seed(1)
    model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    scales = config.model.blur_schedule
    heat = mutils.create_forward_process_from_sigmas(config, scales, config.device)
    optimize_fn = losses.optimization_manager(config)
    train_step = losses.get_step_fn(train=True, scales=scales, config=config,
                                    optimize_fn=optimize_fn, heat_forward_module=heat)
    eval_step = losses.get_step_fn(train=False, scales=scales, config=config,
                                   optimize_fn=optimize_fn, heat_forward_module=heat)
    trainloader, testloader = datasets.get_dataset(config, uniform_dequantization=False)
    batches = iter(trainloader)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for _ in range(5):
        batch = next(batches)[0].to(config.device).float()
        t0 = time.perf_counter()
        loss, _, _ = train_step(state, batch)
        torch.cuda.synchronize()
        rows.append((int(state["step"]), float(loss), float(state["grad_norm"]),
                     float(train_step.scaler.get_scale()), time.perf_counter() - t0))
    for row in rows:
        print("H1 step=%d loss=%.4f grad_norm_preclip=%.4g amp_scale=%.0f dt=%.2fs" % row)
    peak = torch.cuda.max_memory_allocated() / 2**30
    ok &= report("train_step_fp16_b16",
                 all(torch.isfinite(torch.tensor(r[1])) for r in rows)
                 and batch.shape == (16, 1, 192, 192),
                 f"batch={tuple(batch.shape)} dtype_autocast=fp16 losses={[round(r[1], 4) for r in rows]} "
                 f"peak_alloc_gib={peak:.2f} reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.2f}")
    ok &= report("grad_norm_hook", all(r[2] > 0 for r in rows[-3:]) and len({r[2] for r in rows}) > 1,
                 f"pre-clip norms={[f'{r[2]:.4g}' for r in rows]} (inf/nan on the first AMP steps is "
                 "the GradScaler finding its scale)")
    ok &= report("amp_scale", train_step.scaler.is_enabled(), f"scales={[r[3] for r in rows]}")
    eval_batch = next(iter(testloader))[0].to(config.device).float()
    eval_loss, _, _ = eval_step(state, eval_batch)
    ok &= report("eval_step", bool(torch.isfinite(eval_loss)), f"eval_loss={float(eval_loss):.4f}")
    print(f"H1 overall {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
