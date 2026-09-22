# 02 — Engineering practices (implementation level)

These rules apply to every ticket. They are the "level 2" of the plan: how the code is written,
organised and verified. They favour correctness and auditability over generality; this is a course
project with a four-week horizon, so **no abstraction is added without two concrete users**.

## 1. Layout

```
generative-inverse-heat-dissipation/
├── train.py, sample.py, evaluate.py     # the released entry points (edited only at hook points)
├── model_code/, scripts/                # the released code (read-only except the listed hooks)
├── configs/
│   ├── <released configs>               # untouched
│   └── spectral/                        # ours: arms.py (config factory), smoke.py
├── ihdm/                                # ours: one package, importable as `ihdm`
│   ├── paths.py                         # data root, run root, template paths (env-overridable)
│   ├── data/                            # standard format: format.py (write/read/validate), dataset.py (torch Dataset)
│   ├── preprocess/                      # mri.py, registration.py, photos.py, fetch_hf.py, qc.py
│   ├── spectral/                        # mode_power, eigenvalues, schedules (log, matched), profile
│   ├── train/                           # manifest.py, logging.py, checkpoints.py (helpers used by train.py)
│   ├── sampling/                        # sample_from_checkpoint and seed handling
│   ├── metrics/                         # spectral.py, memorisation.py, diversity.py
│   ├── stats/                           # bootstrap, permutation, tables
│   ├── analysis/                        # tables.py, figures.py
│   └── cli/                             # one argparse module per command, run as `python -m ihdm.cli.<name>`
├── schedules/                           # frozen .npy arrays + schedules.json (committed)
├── slurm/                               # Picasso launcher/worker pairs (picasso-sbatch conventions)
├── tests/                               # pytest, mirrors ihdm/ (tests/data, tests/preprocess, …)
├── docs/                                # SPECIFICATIONS, HARNESSES, AGENT-LOGS, RESULTS
├── environment.yml, pyproject.toml, .gitignore, README.md
└── runs/ (git-ignored)                  # local training outputs
```

The released code keeps its import style (`from scripts import losses`); new code uses
`from ihdm.<sub> import <name>`. New CLIs are modules under `ihdm/cli/` so they are testable and
importable; none lives at the repository root.

## 2. Python conventions (from the global CLAUDE.md, restated for agents that will not read it)

- Python 3.10+: `X | None`, `list[int]`, `tuple[str, ...]`. Type hints on every signature.
- `@dataclass(frozen=True)` for every configuration object that crosses a function boundary
  (`LoaderConfig`, `RegistrationConfig`, `ScheduleSpec`, `SampleRequest`, …). Never raw dicts for
  parameters. The released trainer keeps `ml_collections.ConfigDict`; our config factory returns
  one, built from our dataclasses.
- NumPy-style docstrings with `Parameters`, `Returns`, `Raises`. No usage examples in docstrings.
- One exception hierarchy per subpackage: `class DataFormatError(Exception)`,
  `class PreprocessError(Exception)`, `class ScheduleError(Exception)`, `class MetricError(Exception)`.
  Validation failures raise these, never `assert`.
- Functions ≤ ~50 lines, nesting ≤ 2 levels; extract helpers. One class per file above ~100 lines.
- `logging.getLogger(__name__)` in library code; `print` only in CLIs for the final report line.
- Explicit device and memory management in torch code: `torch.no_grad()` / `torch.inference_mode()`
  for sampling and metrics, `.detach().cpu()` before `numpy`, `torch.cuda.empty_cache()` between
  large sampling batches.
- Determinism: every stochastic function takes an explicit `seed: int` or `rng: np.random.Generator`
  / `torch.Generator`; no module-level RNG state.
- Paths: `pathlib.Path` everywhere; absolute paths only in `ihdm/paths.py`, read from the
  environment (`IHDM_DATA_ROOT`, `IHDM_RUN_ROOT`) with the documented defaults.
- Comments explain *why*. No commented-out code committed.
- Tooling: `ruff` (line length 100, rules `E,F,I,UP,B`), `pytest`. Configured in `pyproject.toml`.

## 3. Patterns that apply here (and what each buys)

| pattern | where | what it buys |
|---|---|---|
| **Contract first, validators at the boundary** | `ihdm/data/format.py` (`validate_dataset`), schedule files (`ihdm/spectral/schedules.py: validate_schedule`), run artefacts (`ihdm/train/manifest.py`) | two agents can build producer and consumer in parallel against a frozen spec; every artefact is checked when written and when read |
| **Functional core, imperative shell** | preprocessing stages are pure functions `ndarray -> ndarray` (registration, windowing, intensity); the CLI does I/O, parallelism and logging | unit tests on synthetic arrays; the same functions serve the QC sheets and the profile |
| **Strategy** | schedule builders (`log_schedule`, `matched_schedule`) behind one `build_schedule(spec)`; dataset backends behind `scripts/datasets.get_dataset` | arms differ by a spec value, never by code paths |
| **Adapter** | `ihdm/data/dataset.py: NpyImageDataset` presents the standard format to the released trainer with the exact tensor contract the trainer already expects | the trainer's loop is untouched |
| **Manifest + idempotent CLI** | every CLI writes `<out>/manifest.json` (inputs, parameters, git SHA, versions, timestamp, content hash) and skips work whose manifest already exists unless `--force` | reruns are cheap; provenance is always recoverable |
| **Fail fast** | validators raise the subpackage's exception with the offending path/key; CLIs exit non-zero | a bad dataset never reaches the GPU queue |
| **Testing pyramid** | unit (synthetic arrays) → contract (validators on written artefacts) → smoke (tiny end-to-end on CPU) → harness (real data, GPU; see `docs/HARNESSES/`) | each layer catches what the previous cannot |

Patterns deliberately **not** used: plugin registries, abstract base classes with one
implementation, dependency-injection frameworks, Hydra/OmegaConf (the fork uses ml_collections),
notebooks in the repository, a custom experiment tracker (tensorboard + `metrics.jsonl` suffice).

## 4. Testing rules

- `pytest` only; fixtures in `tests/conftest.py` (a synthetic dataset of 64 images at $32^2$ written
  in the standard format into `tmp_path`; a tiny config; a random-field generator with a known
  $1/f^\alpha$ spectrum).
- Parametrise over edge cases: empty split, single subject, all-zero image, NaN input, odd
  image size, `K=1`.
- Scientific assertions with tolerances: `np.testing.assert_allclose(rtol=…)`,
  `torch.testing.assert_close`. Identities that must hold: LSD$(X,X)=0$; the log schedule
  reproduces `np.exp(np.linspace(...))`; a matched schedule on a white spectrum equals the
  log schedule's per-level variance profile within tolerance; `M=1` when samples *are* held-out
  images; diversity $=0$ for identical samples.
- Real-data checks live in the harnesses, not in `pytest` (they need the data disk or a GPU);
  mark any test that needs them `@pytest.mark.integration` and skip when the path is absent.
- Every ticket's log records the exact test command and its verbatim summary line.

## 5. Git and branching

- Work happens in a git worktree per ticket at `projects/GenAI/code/wt/<ticket-id>/`, on branch
  `ticket/<ticket-id>-<slug>`, cut from the base commit named in the ticket prompt.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), imperative subject ≤ 72 chars,
  a body explaining *why* for non-trivial changes. Commit only your own files, by explicit path.
- Agents never push, rebase or merge. The orchestrator merges `--no-ff` into `main`, runs the
  tests after each merge, and pushes to `origin` (Mario's fork).
- Never commit data, checkpoints, `runs/`, `wt/`, or files > 5 MB other than the schedule arrays
  and the result figures.

## 6. Ticket protocol

1. Read `00-overview.md`, this file, the contracts your ticket names, then your ticket.
2. Create your log from `docs/AGENT-LOGS/TEMPLATE.md` (or invoke the `ticket-log` skill) **before
   editing code**; paste the prompt verbatim; write the plan; keep it updated as you go.
3. Implement in small verified steps; run the fast tests often; run the full tests before the
   final commit and record the output.
4. Fill the acceptance table at the end of your log honestly; a "no" with a reason is better than
   a "yes" without evidence.
5. Final message to the orchestrator: status, branch, worktree, head SHA, log path, test summary,
   three bullets on what was built, anything unfinished.
