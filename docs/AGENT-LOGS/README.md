# Agent logs

One Markdown file per ticket, in a folder per milestone, written by the agent that executed the
ticket from `TEMPLATE.md` (or through the `ticket-log` skill in `.claude/skills/ticket-log/`).
The orchestrator's own log is `ORCHESTRATOR-SESSION.md`; wave summaries (what was merged, which
verdict each ticket got, what the decomposition got wrong) are `WAVE-<id>.md` in the milestone
folder.

```
AGENT-LOGS/
├── README.md, TEMPLATE.md, ORCHESTRATOR-SESSION.md
├── M0-scaffolding/T0.1-scaffolding-and-data-format.md
├── M1-data/T1.1-mri-pipeline.md, T1.2-photograph-pipeline.md, T1.3-profile-and-schedules.md, WAVE-*.md
├── M2-training/…
├── M3-picasso/…
├── M4-metrics/…
├── M5-evaluation/…
└── M6-analysis/…
```

Rules: the prompt is pasted verbatim; decisions name the rejected alternative; the file list
matches the diff; test output is verbatim; results carry numbers and paths; a "no" in the
acceptance table with a reason beats a "yes" without evidence.
