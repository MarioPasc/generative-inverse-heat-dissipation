"""Training-side helpers used by ``train.py`` at the hook points of ``04-run-artifacts.md`` §5.

The modules are deliberately independent of each other and of the released code, so that
``train.py`` stays a thin shell around the released loop:

``manifest``
    ``manifest.json``, ``config.json``, the run id and the config hash.
``logging``
    ``metrics.jsonl`` and tensorboard, including the per-octave loss decomposition.
``checkpoints``
    EMA checkpoints, ``full_final.pt`` and the rolling resume checkpoint path.
``grids``
    the fixed training seeds and the two-row sample grid.
"""

from ihdm.train.errors import TrainError

__all__ = ["TrainError"]
