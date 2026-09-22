# docs/ — specifications, harnesses and agent logs of the spectral-allocation IHDM experiment

Symlinked from `projects/GenAI/code/docs` in the TFM knowledge base; versioned in this fork.
Start with `SPECIFICATIONS/00-overview.md`.

```
docs/
├── README.md                                   this tree
├── SPECIFICATIONS/                             level 1 (what) and level 2 (how)
│   ├── 00-overview.md                          goal, arms, datasets, constraints, decisions D1–D13, acceptance
│   ├── 01-milestones.md                        M0–M6 with exit criteria; ticket index with ownership; waves
│   ├── 02-engineering-practices.md             layout, Python conventions, patterns used and avoided, tests, git, ticket protocol
│   ├── 03-data-format.md                       frozen contract: the standard dataset format and its module
│   ├── 04-run-artifacts.md                     frozen contracts: schedule files, config factory, run directory, checkpoints, sampler API, hook points
│   ├── 05-metrics.md                           frozen definitions: LSD, T_tau, diversity, M, inherited band, PCA, FID/KID/prdc, statistics
│   ├── M0-scaffolding/T0.1-scaffolding-and-data-format.md
│   ├── M1-data/T1.1-mri-pipeline.md · T1.2-photograph-pipeline.md · T1.3-profile-and-schedules.md
│   ├── M2-training/T2.1-trainer-hooks-and-arm-configs.md · T2.2-offline-sampler.md · T2.3-local-pilot-3060.md
│   ├── M3-picasso/T3.1-picasso-setup.md · T3.2-picasso-probe.md · T3.3-full-array-submission.md
│   ├── M4-metrics/T4.1-spectral-metrics.md · T4.2-memorisation-and-diversity.md · T4.3-evaluate-run-and-statistics.md
│   ├── M5-evaluation/README.md                 sketch (T5.1 evaluation array, T5.2 collect)
│   └── M6-analysis/README.md                   sketch (T6.1 tables and statistics, T6.2 figures)
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
```

Reading order for a new agent: `00` → `01` → `02` → the contracts your ticket names → your
ticket → the harness it names → `AGENT-LOGS/TEMPLATE.md`.
