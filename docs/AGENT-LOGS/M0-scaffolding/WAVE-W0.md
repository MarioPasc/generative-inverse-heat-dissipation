# Wave W0 — T0.1 scaffolding (2026-09-22)

| field | value |
|---|---|
| Base commit | `c23fbff` (docs only on top of the released fork `3f735ca`) |
| Agent | `T0.1`, sonnet5-xhigh, manual worktree `wt/T0.1`, branch `ticket/T0.1-scaffolding-and-data-format` |
| Head | `35a9932` (10 commits) |
| Verdict | **ACCEPT** |
| Merged | `1c7e2c5` (`merge(T0.1)`), pushed to `origin/main` |
| Log | `T0.1-scaffolding-and-data-format.md` |

## Verification by the orchestrator

- `git status --porcelain` empty; `git diff --name-only c23fbff..35a9932` = 18 files (17 code/config/test + the log), matching the log's table.
- Re-ran `pytest -q` in the worktree and again on `main` after the merge: 18 passed in 3.4 s. `ruff check ihdm tests configs/spectral`: clean. `import ihdm, scripts.datasets, configs.spectral.smoke` OK without `mpi4py`; torch 2.14.0+cu130, CUDA visible on the RTX 3060.

## Interventions

1. Approved (by message) adding `blobfile` and `opencv-python-headless` to the env: the released `scripts/datasets.py` and `scripts/utils.py` import them unconditionally.
2. Accepted two contract deviations the agent found by running the code, and folded them into the specs (`04-run-artifacts.md` §6): smoke config `model_channels=32` (released `GroupNorm32` needs multiples of 32); CPU smoke runs of the released trainer go in a subprocess with `CUDA_VISIBLE_DEVICES=""` because `create_model`'s `DataParallel(device_ids=None)` scatters to `cuda:0` whenever CUDA is visible. T2.2 is told to instantiate `UNetModel` directly.

## What the decomposition got wrong

Nothing structural. Two things the spec did not know: the released code's hidden dependencies (`blobfile`, `cv2`) and the GroupNorm-32 constraint; both are now in the specs. The synthetic fixture only exercises `seed == ref` (2 ref subjects, 2 seed subjects); real data will exercise `seed ⊊ ref` first in W1, so the orchestrator will check `validate_dataset` on the real outputs with that in mind.
