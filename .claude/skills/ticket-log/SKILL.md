---
name: ticket-log
description: Create and maintain the mandatory ticket log of an implementation agent working on a ticket of the spectral-allocation IHDM project (docs/SPECIFICATIONS). Use at the start of any ticket ("start ticket T1.1", "write my ticket log", "log this ticket"), whenever a decision is taken mid-ticket, and before the final commit. Produces docs/AGENT-LOGS/M<k>-<slug>/T<k>.<n>-<slug>.md from docs/AGENT-LOGS/TEMPLATE.md with the prompt pasted verbatim, the decisions, the files touched, the test output and the results.
---

# ticket-log

You are an implementation agent on a ticket of `docs/SPECIFICATIONS/`. Your log is the only
record the orchestrator trusts, and it verifies every claim in it against `git diff` and by
re-running your commands.

## When to run

1. **Before editing any code**: create the file and fill Identity, §1 (prompt verbatim) and §2
   (plan).
2. **Whenever you decide something** with a real alternative, or assume something you could not
   verify: append it to §2 and message `main` with the assumption.
3. **Before your final commit**: fill §3–§8, run the tests one last time and paste the summary,
   then commit the log with `docs(agent-log): T<k>.<n> <slug>`.

## Procedure

```bash
# 1. locate the milestone folder and the file name from your ticket id
MILESTONE_DIR=docs/AGENT-LOGS/M<k>-<milestone-slug>        # e.g. M1-data
LOG=$MILESTONE_DIR/T<k>.<n>-<ticket-slug>.md               # e.g. T1.1-mri-pipeline.md
mkdir -p "$MILESTONE_DIR" && cp docs/AGENT-LOGS/TEMPLATE.md "$LOG"
```

Then edit `$LOG` section by section. The milestone slugs are `M0-scaffolding`, `M1-data`,
`M2-training`, `M3-picasso`, `M4-metrics`, `M5-evaluation`, `M6-analysis`; the ticket slug is the
one in the ticket file name under `docs/SPECIFICATIONS/`.

## Rules for the content

- §1 holds the delegation prompt **verbatim and complete**, inside a fenced block. Never
  paraphrase it.
- §2 decisions: each names the alternative rejected and the reason. Each assumption states what
  breaks if it is wrong.
- §3 file table must equal `git diff --name-only <base>..HEAD`. Check it with that command.
- §4 test output is pasted verbatim (the summary line and every failure). Real-data or GPU
  verification is a table with numbers, paths and timings.
- §5 results: the numbers, tables and artefact paths the ticket produced — what the report will
  quote.
- §8 acceptance table repeats the ticket's acceptance list; a "no" with a reason is expected
  when something could not be done.
- English; no em-dashes needed; concise.

## Final message to the orchestrator

At most 15 lines: STATUS, BRANCH, WORKTREE, HEAD, LOG path, TESTS summary, three bullets on what
was built, anything the orchestrator must know, anything unfinished.
