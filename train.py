"""Training entry point of the spectral-allocation experiment.

The released loop of ``AaltoML/generative-inverse-heat-dissipation`` (model, optimiser, EMA,
released state dict, one-step train/eval functions, the ``range(initial_step, n_iters + 1)``
loop) is kept as it was; only the hook points of ``docs/SPECIFICATIONS/04-run-artifacts.md`` §5
are changed:

* seeding at the top of :func:`train` from ``config.seed``;
* the manifest and ``config.json`` written before the first step;
* ``ihdm.train.logging.MetricsLogger`` in place of the single tensorboard scalar;
* the checkpoint cadences ``ckpt_every`` / ``resume_every`` and the EMA checkpoint format of
  ``04`` §3.3 in place of the released ``checkpoint_<n>.pth`` numbering;
* ``ihdm.train.grids.save_grid`` in place of the released gif/video writers;
* a NaN/Inf guard, and ``full_final.pt`` plus ``DONE`` at the end.

Decision D19 (after array 2408239) adds, through ``ihdm.train.guard`` and ``ihdm.train.recipe``:

* the skip policy: with an enabled ``GradScaler`` a non-finite loss is a no-op step, logged as
  ``skip``; the run aborts (exit 3) after ``training.max_consecutive_skips`` consecutive or more
  than ``training.max_skips`` skipped steps, and at the first non-finite loss when the scaler is
  disabled (the optimiser step then applies the non-finite gradients);
* ``grad_norm`` = the pre-clip norm of the unscaled gradients and ``amp_scale`` on every train
  line, read from the ``scripts/losses.py`` hook at log steps only (no new GPU synchronisation);
* the recipe check on resume: a resume whose recipe differs from the run's exits with code 4
  before writing anything.

Resuming and extending (decision D10) are native: ``initial_step`` comes from the released
``restore_checkpoint`` on ``checkpoints-meta/checkpoint.pth``, so resubmitting the same workdir
with a larger ``--config.training.n_iters`` continues the run.
"""

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags

from ihdm.train import checkpoints as ckpt_utils
from ihdm.train import grids, manifest, recipe
from ihdm.train import guard as guard_utils
from ihdm.train.logging import MetricsLogger
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from scripts import datasets, losses, utils

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

#: Batches of the ``ref`` split used for every evaluation point (released choice, kept).
N_EVAL_BATCHES = 25

#: Seed images of the sample grids (``04-run-artifacts.md`` §3).
N_GRID_SEEDS = 8

#: Exit code of the NaN/Inf guard.
ABORT_EXIT_CODE = 3

#: Exit code of a resume refused by the recipe check (D19).
RECIPE_EXIT_CODE = recipe.RECIPE_EXIT_CODE


def main(argv):
    del argv
    logging.getLogger().setLevel(logging.INFO)
    train(FLAGS.config, FLAGS.workdir)


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy and torch generators from ``config.seed``.

    The data loaders carry their own ``torch.Generator`` seeded from ``config.seed``
    (``ihdm.data.dataset.make_loaders``), so a fresh run is reproducible end to end. A resumed
    run replays the data order from the beginning of the loader, because the released state
    dict does not checkpoint the sampler.

    Parameters
    ----------
    seed : int
        The run seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_eval(eval_step_fn, state, testloader, eval_iter, config):
    """Average the loss over :data:`N_EVAL_BATCHES` batches of the ``ref`` split (EMA weights).

    Returns
    -------
    tuple[float, Iterator]
        The mean loss and the (possibly restarted) evaluation iterator.
    """
    total = 0.0
    for _ in range(N_EVAL_BATCHES):
        try:
            eval_batch = next(eval_iter)[0].to(config.device).float()
        except StopIteration:  # Start new epoch
            eval_iter = iter(testloader)
            eval_batch = next(eval_iter)[0].to(config.device).float()
        eval_loss, _, _ = eval_step_fn(state, eval_batch)
        total += float(eval_loss.detach().item())
    return total / N_EVAL_BATCHES, eval_iter


def _save_grid(workdir, step, model, ema, model_evaluation_fn, config, heat_forward_module, seeds):
    """Write one sample grid with the EMA weights, restoring the live weights afterwards."""
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    try:
        return grids.save_grid(
            workdir, step, model_evaluation_fn, config, heat_forward_module, seeds)
    finally:
        ema.restore(model.parameters())


def _finalise(workdir, state, model, ema, config, config_hash, metrics, final_step, n_skipped):
    """Write the final EMA checkpoint (if missing), ``full_final.pt``, ``done`` and ``DONE``.

    Idempotent, so it is safe on a fresh run, on a resumed run and on a resubmission of a run
    that has already reached ``n_iters``. ``DONE`` is written last, only once both checkpoints
    are on disk.
    """
    final_ema = ckpt_utils.ema_checkpoint_path(workdir, final_step)
    if not final_ema.is_file():
        ckpt_utils.save_ema(
            workdir, final_step, model, ema, config, config.run_id, config_hash)
        metrics.log_event("ckpt", final_step, path=str(final_ema))
    ckpt_utils.save_resume(workdir, state)
    ckpt_utils.save_full_final(workdir, state)
    metrics.log_event("done", final_step, n_skipped=int(n_skipped))
    (Path(workdir) / "DONE").touch()
    logging.info("Run %s finished at step %d", config.run_id, final_step)


def _check_recipe_or_exit(workdir, config):
    """D19: refuse to resume a run whose recipe differs from this invocation's.

    Runs before the manifest, ``config.json``, the tensorboard writer or the ``resume`` line are
    touched, so a refused resume leaves the run directory byte-identical.
    """
    try:
        recipe.check_resume_recipe(workdir, config)
    except recipe.RecipeMismatchError as error:
        logging.error("Refusing to resume: %s", error)
        sys.exit(RECIPE_EXIT_CODE)


def _step_scalars(state, scaler, scaler_enabled):
    """Read the pre-clip gradient norm and the AMP scale of the last step (log steps only).

    Both reads synchronise with the GPU, which the loop already does at log steps.
    """
    grad_norm = state.get('grad_norm')
    grad_norm = float(grad_norm) if grad_norm is not None else None
    amp_scale = float(scaler.get_scale()) if scaler_enabled else None
    return grad_norm, amp_scale


def _abort(workdir, state, metrics, step, loss, decision, save_state):
    """Log the ``abort`` event, keep the resume checkpoint when it is still valid, exit 3.

    With an enabled scaler every skipped step was a no-op, so the weights are those that produced
    the non-finite losses and are saved for diagnosis (array 1's fixtures came from exactly this).
    With a disabled scaler the optimiser has just applied non-finite gradients, so the last
    rolling checkpoint is the only good state left and is not overwritten.
    """
    metrics.log_event("abort", step, reason=decision.reason, loss=repr(float(loss.item())),
                      n_skipped=decision.n_skipped, consecutive=decision.consecutive,
                      resume_saved=bool(save_state))
    if save_state:
        ckpt_utils.save_resume(workdir, state)
    metrics.close()
    logging.error("Aborting at step %d: %s", step, decision.reason)
    sys.exit(ABORT_EXIT_CODE)


def train(config, workdir):
    """Runs the training pipeline.
    Based on code from https://github.com/yang-song/score_sde_pytorch

    Args:
            config: Configuration to use.
            workdir: Working directory for checkpoints and TF summaries. If this
                    contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Hook: seeding (04-run-artifacts.md §5). config.seed is unused by the released code.
    seed_everything(int(config.seed))

    if config.device == torch.device('cpu'):
        logging.info("RUNNING ON CPU")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_meta_dir = ckpt_utils.resume_path(workdir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_meta_dir.parent.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    model_evaluation_fn = mutils.get_model_fn(model, train=False)

    # Resume training when intermediate checkpoints are detected
    state = utils.restore_checkpoint(str(checkpoint_meta_dir), state, config.device)
    initial_step = int(state['step'])

    # Hook: D19 recipe check. A resume must continue the run's own recipe.
    if initial_step > 0:
        _check_recipe_or_exit(workdir, config)

    # Build data iterators
    trainloader, testloader = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(trainloader)
    eval_iter = iter(testloader)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)

    # Get the forward process definition
    scales = config.model.blur_schedule
    heat_forward_module = mutils.create_forward_process_from_sigmas(
        config, scales, config.device)

    # Get the loss function
    train_step_fn = losses.get_step_fn(train=True, scales=scales, config=config,
                                       optimize_fn=optimize_fn,
                                       heat_forward_module=heat_forward_module)
    eval_step_fn = losses.get_step_fn(train=False, scales=scales, config=config,
                                      optimize_fn=optimize_fn,
                                      heat_forward_module=heat_forward_module)

    # Hook: D19 skip policy. The scaler only protects the step when AMP actually drives it: with
    # optim.automatic_mp=False the released code calls optimizer.step() directly, and on a host
    # without CUDA torch.cuda.amp.GradScaler disables itself and steps unconditionally.
    scaler = train_step_fn.scaler
    scaler_enabled = bool(config.optim.automatic_mp) and scaler.is_enabled()
    guard = guard_utils.NonFiniteGuard(
        guard_utils.policy_from_config(config, scaler_enabled),
        n_skipped=guard_utils.committed_skips_from_file(workdir / "metrics.jsonl", initial_step))

    # Hook: run identity, manifest and metrics (04-run-artifacts.md §5). run_id is recomputed
    # here because --config.seed is applied after the config factory has run.
    config.run_id = manifest.run_id_from_config(config)
    config_hash = manifest.config_sha256(config)
    manifest.write_manifest(workdir, config, model, manifest.load_data_meta(config))
    manifest.write_config_json(workdir, config)
    seeds = grids.prepare_seeds(
        workdir, Path(config.data.root) / config.data.dataset, int(config.seed), n=N_GRID_SEEDS)
    metrics = MetricsLogger(workdir, config)

    num_train_steps = int(config.training.n_iters)
    on_cuda = str(config.device).startswith("cuda")
    if initial_step > 0:
        metrics.log_event("resume", initial_step, **{"from": str(checkpoint_meta_dir)})
    logging.info("Starting training loop at step %d.", initial_step)
    logging.info("Running on %s", config.device)
    logging.info("Skip policy: %s; %d skipped steps carried over.", guard.policy, guard.n_skipped)

    # D10: a resubmission with the same or a smaller n_iters must exit cleanly, not retrain.
    if initial_step > num_train_steps:
        logging.info("Saved step %d is past n_iters %d; nothing to train.",
                     initial_step, num_train_steps)
        _finalise(workdir, state, model, ema, config, config_hash, metrics, num_train_steps,
                  guard.n_skipped)
        metrics.close()
        return

    for step in range(initial_step, num_train_steps + 1):
        # Train step
        t_step = time.perf_counter()
        try:
            batch = next(train_iter)[0].to(config.device).float()
        except StopIteration:  # Start new epoch if run out of data
            train_iter = iter(trainloader)
            batch = next(train_iter)[0].to(config.device).float()
        loss, losses_batch, fwd_steps_batch = train_step_fn(state, batch)

        # Hook: per-step timing; synchronise only at log steps so the window total is exact.
        is_log_step = step % config.training.log_every == 0
        if is_log_step and on_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_step

        # Hook: NaN/Inf guard with the D19 skip policy (ihdm.train.guard).
        decision = guard.observe(bool(torch.isfinite(loss)))
        if decision.skipped:
            metrics.log_event("skip", step, loss=None, n_skipped=decision.n_skipped,
                              consecutive=decision.consecutive)
            logging.warning("Non-finite loss at step %d: step skipped by the GradScaler "
                            "(%d in this run, %d consecutive).",
                            step, decision.n_skipped, decision.consecutive)
        if decision.abort:
            _abort(workdir, state, metrics, step, loss, decision, save_state=scaler_enabled)

        grad_norm, amp_scale = (_step_scalars(state, scaler, scaler_enabled) if is_log_step
                                else (None, None))
        metrics.log_train(step, loss, losses_batch, fwd_steps_batch,
                          optimizer.param_groups[0]['lr'], dt, grad_norm,
                          amp_scale=amp_scale, skipped=decision.skipped)
        if is_log_step:
            logging.info("step: %d, training_loss: %.5e", step, loss.item())

        # Save a temporary checkpoint to resume training if training is stopped
        if step != 0 and step % config.training.resume_every == 0:
            logging.info("Saving temporary checkpoint")
            ckpt_utils.save_resume(workdir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_every == 0:
            logging.info("Starting evaluation")
            eval_loss, eval_iter = _run_eval(
                eval_step_fn, state, testloader, eval_iter, config)
            metrics.log_eval(step, eval_loss)
            logging.info("step: %d, eval_loss: %.5e", step, eval_loss)

        # Save a checkpoint periodically
        if (step != 0 and step % config.training.ckpt_every == 0) or step == num_train_steps:
            logging.info("Saving a checkpoint")
            path = ckpt_utils.save_ema(
                workdir, step, model, ema, config, config.run_id, config_hash)
            metrics.log_event("ckpt", step, path=str(path))

        # Generate samples periodically
        if (step != 0 and step % config.training.grid_every == 0) or step == num_train_steps:
            logging.info("Sampling...")
            path = _save_grid(workdir, step, model, ema, model_evaluation_fn, config,
                              heat_forward_module, seeds)
            metrics.log_event("grid", step, path=str(path))

    _finalise(workdir, state, model, ema, config, config_hash, metrics, num_train_steps,
              guard.n_skipped)
    metrics.close()


if __name__ == "__main__":
    app.run(main)
