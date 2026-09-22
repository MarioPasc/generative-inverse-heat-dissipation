"""CPU smoke-test config: tiny model, tiny images, 6 iterations, on a synthetic dataset.

Frozen contract: ``docs/SPECIFICATIONS/04-run-artifacts.md`` §6. Used by
``tests/train/test_smoke_train.py`` and ``tests/train/test_checkpoint_format.py`` (T2.1,
which run ``train.train`` end to end over the synthetic fixture) and by the Picasso
import-check job.

Sets every key the released ``train.py``/``scripts/losses.py``/``scripts/sampling.py``
read (the old cadence names: ``snapshot_freq``, ``snapshot_freq_for_preemption``,
``log_freq``, ``eval_freq``, ``sampling_freq``) as well as the new-style cadence
names (``ckpt_every``, ``resume_every``, ``log_every``, ``eval_every``,
``grid_every``) that a later ticket's hook-based trainer will read, with matching
values, per ``04-run-artifacts.md`` §2's "kept under the old names for
compatibility" pattern. ``data.dataset = "synthetic"`` names the folder a test
fixture writes under ``data.root`` (set by the test, not fixed here).
"""

import hashlib

import ml_collections
import numpy as np
import torch


def get_config() -> ml_collections.ConfigDict:
    """Return the CPU smoke-test config.

    Returns
    -------
    ml_collections.ConfigDict
        A config with the released structure (``training``, ``sampling``, ``eval``,
        ``data``, ``model``, ``optim``, ``seed``, ``device``), sized for a fast,
        deterministic CPU run over the synthetic fixture dataset.
    """
    config = ml_collections.ConfigDict()

    # run identity, mirroring configs/spectral/arms.py so train.py reads the same keys
    # on both configs. "smoke" is not one of the five arms; the trainer only echoes it.
    config.dataset_id = "synthetic"
    config.arm = "smoke"
    config.run_id = "synthetic_smoke_s1"

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 4
    training.n_iters = 6
    training.snapshot_freq = 3
    training.log_freq = 1
    training.eval_freq = 3
    training.sampling_freq = 3
    training.snapshot_freq_for_preemption = 2
    # new-style names (D5-era trainer, T2.1), kept in sync with the old ones above
    training.ckpt_every = 3
    training.resume_every = 2
    training.log_every = 1
    training.eval_every = 3
    training.grid_every = 3

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.prior_noise = False
    sampling.delta_factor = 1.25

    # eval
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 4
    evaluate.enable_sampling = False
    evaluate.num_samples = 8
    evaluate.enable_loss = True
    evaluate.calculate_fids = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "synthetic"
    data.root = ""  # filled in by the test with the tmp_path holding the fixture
    data.split_train = "train"
    data.split_eval = "ref"
    data.image_size = 32
    data.num_channels = 1
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_workers = 0

    # model
    config.model = model = ml_collections.ConfigDict()
    model.K = 8
    model.sigma = 0.01
    model.dropout = 0.1
    # model_code/nn.py:normalization() hardcodes GroupNorm32(32, channels), so every
    # channel count in the network must be a multiple of 32; 16 (04-run-artifacts.md
    # §6's literal value) is not. 32 is the smallest value that satisfies the released,
    # read-only model_code/ and keeps the network tiny.
    model.model_channels = 32
    model.channel_mult = (1, 2)
    model.conv_resample = True
    model.num_heads = 1
    model.conditional = True
    model.attention_levels = ()
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.num_res_blocks = 1
    model.use_fp16 = False
    model.use_scale_shift_norm = False
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.skip_rescale = True
    model.blur_sigma_min = 0.5
    model.blur_sigma_max = 8.0
    model.blur_schedule = np.array(
        [0.0]
        + list(
            np.exp(
                np.linspace(np.log(model.blur_sigma_min), np.log(model.blur_sigma_max), model.K)
            )
        )
    )
    # Provenance keys of 04-run-artifacts.md §3.1/§3.3; the smoke schedule is generated here
    # rather than loaded from schedules/, so it has no file and its hash is of its own bytes.
    model.blur_schedule_name = "smoke_log_K8"
    model.blur_schedule_file = ""
    model.blur_schedule_sha256 = hashlib.sha256(
        np.ascontiguousarray(model.blur_schedule).tobytes()
    ).hexdigest()
    # D3, as in the arm configs: level K is used at sampling time, so it is trained.
    model.train_level_max_inclusive = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 2
    optim.grad_clip = 1.0
    optim.automatic_mp = False

    config.seed = 1
    config.device = torch.device("cpu")

    return config
