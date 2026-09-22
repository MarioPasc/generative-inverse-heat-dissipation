"""Runs the released, unmodified train.py loop for 5 CPU iterations over the synthetic dataset.

This is the acceptance test of T0.1 item 4: the released trainer, untouched, must be able to
train over the standard-format backend end to end.

Run in a subprocess with ``CUDA_VISIBLE_DEVICES=""``: ``model_code/utils.py: create_model``
wraps the model in ``torch.nn.DataParallel(model, device_ids=None)``, which auto-detects CUDA
devices independently of ``config.device`` -- on a host with a GPU this places the scattered
input on ``cuda:0`` while the model itself stays on the CPU device given by ``config.device``,
raising a device-mismatch error. Masking CUDA at the process level (rather than patching
``torch.cuda.is_available()`` in-process, which is unreliable because ``torch.cuda.device_count``
is cached for the life of the process) keeps this test strictly on the CPU, matching the ticket's
constraint that the GPU is used only for the one-line CUDA availability check.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_released_train_runs_five_cpu_iterations(tmp_path, synthetic_dataset):
    workdir = tmp_path / "run"
    script = textwrap.dedent(
        f"""
        from configs.spectral import smoke
        import train as released_train

        config = smoke.get_config()
        config.data.root = {str(synthetic_dataset.parent)!r}
        config.training.n_iters = 5
        released_train.train(config, {str(workdir)!r})
        """
    )

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=58,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workdir / "checkpoints-meta" / "checkpoint.pth").is_file()
