# Harnesses — how each layer of the project is verified

A harness is a fixed, repeatable check that an artefact is what its contract says. The `pytest`
suite covers what synthetic data can prove; the harnesses below cover what needs real data, a
GPU, or the cluster. Each ticket names the harness it must pass; the orchestrator re-runs the
relevant harness before merging.

| harness | file | verifies | runs where |
|---|---|---|---|
| H-DATA | `data.md` | the standard-format datasets, their QC sheets, the profile's known results, the schedule files | workstation, CPU |
| H-TRAIN | `training.md` | the trainer's artefacts, cadence, resume, seeding, the two fixes, throughput | workstation CPU (smoke) and RTX 3060 (pilot) |
| H-METRICS | `metrics.md` | identities and monotonicities every metric must satisfy; the pilot numbers | workstation |
| H-PICASSO | `picasso.md` | environment, data integrity, probe artefacts, array submission, monitoring | Picasso |

Conventions: every harness command is copy-pasteable from the repository root inside the `ihdm`
env; every check states its expected outcome; failures are recorded in the ticket log with the
verbatim output.
