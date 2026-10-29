# Wave W22 — tutorial notebook for the course deliverable (T9.1), 2026-10-29

**Orchestrator:** main (Claude Opus 5.5), TFM session `2026-10-28_genai-methods-section`, continued.

**Mandate.** Mario (2026-10-29): a code expert learns the code through the expert-team skill and the notebook is finished from its knowledge. The notebook sits at the repository root and is cited in the README. It has brain and natural images in `assets/`, the power spectra, the prior at W/2 against W/8, and the paper's schedule against the MRI-matched one. Results are uploaded and plotted, with no training or inference. Formulas follow the course notation, and the text is scoped and clear. Main reviews the result.

**Design.** An expert-team run (`.claude/notes/2026-10-29-course-notebook/`, untracked) uses one read-only expert, `expert-code` (`domain-expert`, Opus 5.5 high). Its domain brief covers the spectral, schedule, prior and figure code and the results docs, and it answered round 1 (Q1–Q7). The build phase is this wave: one builder, consulting the expert through main and reviewed by the expert claim by claim. The concurrency rule (at most 2 agents including reviewers) holds.

## Setup

| ticket | agent | base | worktree / branch | status |
|---|---|---|---|---|
| T9.1 tutorial notebook | opus55-high | `b428313` | `wt/T9.1` / `ticket/T9.1-tutorial-notebook` | ACCEPT after 1 return |

## Expert findings that changed the ticket (verified by main)

- On the discrete 192² grid, a c⁻² spectrum does not hold equal variance per octave: 23.7/15.2/12.2/11.2/10.7/10.5/10.4/6.1%. Under `log_W2` its E_k spread is 2.0×, against 3.8× for Churches and 8.9× for IXI. Log spacing equalises E_k only in the continuum limit.
- IXI row 3705, the Figure 1 slice, is a held-out seed and not a training slice.
- Figure 2(b)-bottom has no machine-readable source; the generated tables in `F2.md` are the only copy.
- Env `ihdm` has no notebook tooling.

## Review

- **First hand-back (`536d6c4`).** Main checked the diff against the ownership in ticket §3: 18 files, all inside it, plus `assets/images/manifest.json`, which was accepted. Main re-ran the 6 new tests (pass), ruff (clean) and the jupytext round trip (the source equals the `.ipynb`). Main also read all 11 figures and reviewed the prose.
- **The builder's open failure** was `test_regression_t72` (2 failures). It is an environment artefact: with the repository's canonical fast-suite command (`OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" nice -n 19 env PYTHONPATH=$PWD … -m "not integration"`), main ran its 4 tests and all passed (48 s). The notebook tooling installed into `ihdm` changed no numeric package: main listed today's new dist-info folders.
- **Expert claim review** (`expert-code`): 16 claims, 10 ✓, 6 ≈, none ✗, and one licence claim not in its sources. Two omissions: the DC/noise convention of E_k + W²σ², and R⁻ (400 images) versus R (800 images) between panels (b) and (c).
- **Return 1 (17 items)**, from main and the expert:
  - the continuum c⁻² reference per octave bin is 13.2% per full octave and 7.7% for the 0.58 octave 64–96 c/img (12.5% was wrong);
  - the octave counting rule;
  - "median matched sample";
  - 1.8 c/img;
  - the synthetic E_k shape, from computed values;
  - absolute GitHub links;
  - licences quoted from their sources (IXI CC BY-SA 3.0; LSUN "to verify");
  - assets CLI: everything is staged until all checks pass, with a test and a sabotage check;
  - `TutorialAssetError` for an unknown dataset;
  - the README env-update command.
- **After return 1 (`058ab99`).** The 8 new tests pass and ruff is clean. The builder's canonical fast suite gave 1469 passed, 1 skipped and 30 deselected (927.5 s). Main confirmed by grep that every requested text change is present and none of the old wording remains. The notebook runs in 3.9 s on CPU; the `.ipynb` is 1.8 MB, the HTML 2.2 MB and `assets/` 132 KB.

## Verdict

**T9.1: ACCEPT.** It is merged `--no-ff` into `integration/W22` (`7dc9218`, worktree `wt/integration-W22`), and the 8 new tests pass there. Merging into `main` and pushing to GitHub wait for Mario. Before a public push, the licence of the two LSUN Churches images must be settled: neither the source dataset card nor the LSUN README states one.

## Expert-team record

`docs/AGENT-LOGS/M9-notebook/expert-team/` holds `expert-code-brief.md`, the expert's domain brief (round-1 answers not included; they are summarised above and in the ticket), and `plan.md` (decisions 1–12).

## Report-level findings handed to Mario

- On the 192² grid, log spacing gives a 2.0× spread of E_k even for c⁻² data. Report §4.2 could cite this as the grid floor (Churches 3.8×, IXI 8.9×).
- "every level removes the same variance" should read "every level's target carries the same between-image variance".
