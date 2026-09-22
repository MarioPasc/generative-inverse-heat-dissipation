# Ticket log — T<k>.<n> <title>

<!-- Location: docs/AGENT-LOGS/M<k>-<milestone-slug>/T<k>.<n>-<ticket-slug>.md
     Create this file BEFORE editing code; keep it updated as you work; commit it as your final
     commit with message `docs(agent-log): T<k>.<n> <ticket-slug>`.
     The orchestrator verifies every claim against `git diff` and by re-running your commands. -->

## Identity

| field | value |
|---|---|
| Ticket | T<k>.<n> — <title> |
| Agent name / model / effort | <name> / <model> / <effort> |
| Branch / worktree | `ticket/T<k>.<n>-<slug>` / `<absolute path>` |
| Base commit → head commit | `<sha>` → `<sha>` |
| Started / finished | <ISO> / <ISO> |
| Status | complete \| partial \| blocked |

## 1. Prompt as received (verbatim, complete)

```
<paste the whole delegation prompt here; do not summarise>
```

## 2. Plan and decisions

**Restatement:** <two or three sentences in your own words>

**Plan:**
1. <step>
2. <step>

**Decisions taken during the work** (each with the alternative rejected and why):
- <decision> — <alternative> rejected because <reason>

**Assumptions proceeded on** (each also messaged to `main` when made):
- <assumption> — what breaks if wrong

## 3. Files created / modified / removed

| path | created / modified / removed | what and why |
|---|---|---|
| `<path>` | | |

<!-- must match `git diff --name-only <base>..HEAD` exactly -->

**Commits:** `<sha> <subject>` per line.

## 4. Tests and verification

**Command:** `<exact command>` → **Result:** <N passed, M failed, K skipped, time>

```
<summary line and the full text of every failure>
```

**Beyond unit tests** (real data, GPU, cluster; numbers, paths, timings):

| what | how | evidence | outcome |
|---|---|---|---|
| | | | |

## 5. Results of the ticket

<The numbers, tables, figure paths and artefact paths the ticket produced. This is the section
the orchestrator and the report will quote.>

## 6. Open questions, risks, follow-ups

- <question for `main`> — why it matters; what was done meanwhile
- <risk> — severity — suggested owner

## 7. Deliberately not done

- <item> — out of scope / deferred / blocked by X

## 8. Acceptance self-assessment

| # | criterion (from the ticket) | met | evidence |
|---|---|---|---|
| 1 | | yes \| partial \| no | |

**Overall:** <what you are confident about, what you are not, what to scrutinise first>
