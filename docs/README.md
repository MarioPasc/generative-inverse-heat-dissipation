# docs/ — specifications, harnesses and agent logs of the spectral-allocation IHDM experiment

Symlinked from `projects/GenAI/code/docs` in the TFM knowledge base; versioned in this fork.
Start with `SPECIFICATIONS/00-overview.md`.

```
docs/
├── README.md                                   this tree
├── SPECIFICATIONS/                             level 1 (what) and level 2 (how)
│   ├── 00-overview.md                          goal, arms, datasets, constraints, decisions D1–D23, acceptance
│   ├── 01-milestones.md                        M0–M6 with exit criteria; ticket index with ownership; waves
│   ├── 02-engineering-practices.md             layout, Python conventions, patterns used and avoided, tests, git, ticket protocol
│   ├── 03-data-format.md                       frozen contract: the standard dataset format and its module
│   ├── 04-run-artifacts.md                     frozen contracts: schedule files, config factory, run directory, checkpoints, sampler API, hook points
│   ├── 05-metrics.md                           frozen definitions: LSD, T_tau, diversity, M, inherited band, PCA, FID/KID/prdc, statistics
│   ├── M0-scaffolding/T0.1-scaffolding-and-data-format.md
│   ├── M1-data/T1.1-mri-pipeline.md · T1.2-photograph-pipeline.md · T1.3-profile-and-schedules.md
│   ├── M2-training/T2.1-trainer-hooks-and-arm-configs.md · T2.2-offline-sampler.md · T2.3-local-pilot-3060.md
│   ├── M3-picasso/T3.1-picasso-setup.md · T3.2-picasso-probe.md · T3.3-full-array-submission.md · T3.4-training-recovery-loginexa.md
│   ├── M4-metrics/T4.1-spectral-metrics.md · T4.2-memorisation-and-diversity.md · T4.3-evaluate-run-and-statistics.md
│   ├── M5-evaluation/README.md · T5.1-evaluation-array.md · T5.2-collect-results.md
│   ├── M6-analysis/README.md · T6.1-tables-and-statistics.md · T6.2-figures.md
│   └── M7-diagnostics/README.md · T7.1–T7.5    post-hoc diagnostics (W14: photo failure, δ sweep, exploratory hardening; W15: held-out-seed fidelity)
├── HARNESSES/
│   ├── README.md                               what a harness is; the four harnesses
│   ├── data.md                                 H-DATA: validation, counts, eyeball gates, known results, schedules
│   ├── training.md                             H-TRAIN: smoke, artefacts, cadence, resume, seeding, pilot numbers
│   ├── metrics.md                              H-METRICS: identities, pilot bracket, FID licence
│   └── picasso.md                              H-PICASSO: quota, env, import check, probe, array, plateau gate
├── AGENT-LOGS/
│   ├── README.md · TEMPLATE.md                 the log protocol and template (also the `ticket-log` skill)
│   ├── ORCHESTRATOR-SESSION.md                 the orchestrator's own log with the prompts it issued
│   └── M<k>-<slug>/T<k>.<n>-<slug>.md · WAVE-*.md
└── RESULTS/                                    outputs that tickets commit (profiles, figures, pilot and probe reports, submissions)
    ├── results_discussion.md                   START HERE for results: metrics, hypotheses verdicts, discussion, limitations
    ├── tables/                                 pre-registered tables 1a–9 (`ihdm.cli.analyse`) + README (method, caveats)
    ├── figures/                                figures 1–7 (`ihdm.cli.figures`) + README (caption drafts)
    ├── exploratory/                            post-hoc readings X1–X3 + fig. X1 (`ihdm.analysis.exploratory`), not pre-registered
    ├── delta_sweep/                            M7/T7.2: sampling-noise sweep on four 60k checkpoints (README + JSON)
    ├── photo_diagnostic/                       M7/T7.3: one-factor Churches diagnostic (128² framing, 32k images), late window
    ├── heldout_fidelity/                       M7/T7.5: fidelity of samples from held-out seeds vs training seeds (README + JSON + grid)
    ├── diagnostic_training.md                  M7/T7.1: submission record of the two diagnostic training runs
    ├── data_profile.md · metrics_bracket.md    spectral profile of the four datasets; metric noise floors
    ├── inherited_band_audit.md                 D23: the inherited-band estimator correction
    └── submissions.md · collection.md · …      the Picasso campaign record and the collection procedure
```

Reading order for a new agent: `00` → `01` → `02` → the contracts your ticket names → your
ticket → the harness it names → `AGENT-LOGS/TEMPLATE.md`.
