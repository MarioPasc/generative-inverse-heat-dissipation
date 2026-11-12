# Plan — course tutorial notebook ("Desarrollos realizados")

Orchestrator: main. Expert: expert-code (domain-expert, Opus 5.5 high). Builder: one agent via parallel-agents (worktree, base 960d6e7), ticket `docs/SPECIFICATIONS/M9-notebook/T9.1-tutorial-notebook.md`.

## Requirement (course, verbatim)
"Desarrollos realizados: Enlace a repositorio GitHub. Aparte del código debe haber un Notebook, al estilo de la documentación de Pytorch (por ejemplo, https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) en el que se ilustre la funcionalidad implementada. Se puede generar con Sphinx."

## Mario's constraints (2026-10-29)
- Notebook at repo root, cited in README.md.
- A few brain / natural images in `assets/` for visuals.
- Show: power spectra brain vs natural; the prior with our change (W/2 vs W/8); paper schedule vs MRI-matched schedule; uploaded results, plotted and explained.
- No training, no sampling.
- Formulas in the course notation (same as the report's Methodology / Appendix C).
- Scoped, clear, explanatory text; no AI phrasing.
- Orchestrator reviews the result.

## Draft structure (to confirm against the expert's brief)
0. Title, one-paragraph purpose, links (paper, report), how to run (env; no data needed: everything in `assets/`).
1. Setup and assets.
2. Two kinds of images: brain MRI slices (IXI) and natural photographs (LSUN Churches).
3. The forward process: heat equation in the DCT basis, F(t)=V e^{-Λt} V^T, σ_B=√(2t), λ_i t=(c_i σ_B/ω)^2; blur one slice at several σ_B; contrast with the VP/DDPM marginal (√ᾱ_t I → F(t)).
4. Where the between-image variance lives: P_i, radial profile, octave shares, IXI vs Churches, 1/f^2 reference.
5. Change 1, the prior: noise-free prior state F(t_K)x at W/2 vs W/8; inherited share I(σ_B,K) curve, markers at 24 and 96 px.
6. Change 2, the level spacing: σ_B,k of log vs matched; levels per octave; E_k per level under both on IXI; spread; synthetic 1/f^2 check that log spacing gives flat E_k.
7. Results: Figure 2 (precision/recall F vs H, distance to seed, octave errors) from uploaded numbers/figure, with a reading guide.
8. Map from notebook to code (module table) and summary.

## Open points for the expert (round 1)
- Data access and cost of each computation; which precomputed arrays exist.
- Licences for example images (IXI CC BY-SA 3.0; LSUN; OASIS-1).
- Reusable functions for every plot; numbers each plot must reproduce.

## Decisions
| # | Decision | Evidence | By |
|---|---|---|---|
| 1 | One expert (code + results); builder separate, read-only expert reviews | expert-team invariant 1; memory genai-orchestrator-does-not-code | main |
| 2 | No training/sampling; H-D §D2 items 5–6 superseded | Mario 2026-10-29 | Mario |
| 3 | Load data_profile/*.npz and schedules/*.npy; never refit (matched_schedule > 3 min) | expert brief; round 1 | main |
| 4 | Show all four schedules in σ_B pixel octaves (as the report's table) | report tab:levels | main |
| 5 | Inherited share on the training split (8.5%, Churches 29.1%) | data_profile.md §3; report §4.2 | main |
| 6 | Synthetic c⁻² on the grid: show non-equal shares and 2.0× spread; never claim flat | expert Q4, re-run by main | main |
| 7 | Images: IXI 3705 (held-out seed, Fig. 1 slice), 3311 (train); Churches 1180 (train), 107 (held-out seed) | expert Q2 | main |
| 8 | Fig. 2(b)-bottom parsed from generated F2.md tables (no machine-readable source) | expert Q1 | main |
| 9 | Install jupytext/nbclient/nbconvert/nbformat/ipykernel into env ihdm; add to dev extras | expert Q5 | main |
| 10 | README: new top section, upstream README kept below | harness H-D §D1 | main |
| 11 | No Sphinx site; HTML export is the rendered copy (course says Sphinx is optional) | course requirement | main |
| 12 | LSUN images: licence not stated by source; raise with Mario before any public push | expert brief | main → Mario |
