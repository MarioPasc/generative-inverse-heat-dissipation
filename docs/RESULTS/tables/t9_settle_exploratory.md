## Table 9 — Settling step of the LSD curve (exploratory, defined after seeing the data (2026-09-29), not pre-registered)

**Status: complete (30/30 runs)**

| dataset | contrast | arm per seed | A0 per seed | arm mean | A0 mean | mean Δ | status |
|---|---|---|---|---:|---:|---:|---|
| IXI | A3−A0 | 5k / 5k / 5k | 55k / 55k / 60k | 5k | 56.67k | -51.67k | ok |
| IXI | A1−A0 | 5k / 5k | 55k / 55k | 5k | 55k | -50k | ok |
| IXI | A2−A0 | 30k / 5k | 55k / 55k | 17.50k | 55k | -37.50k | ok |
| Churches | A3−A0 | 5k / 25k / 25k | 45k / 45k / 55k | 18.33k | 48.33k | -30k | ok |
| Churches | A1−A0 | 5k / 30k | 45k / 45k | 17.50k | 45k | -27.50k | ok |
| Churches | A2−A0 | 20k / not reached | 45k / 45k | — | 45k | — | Δ undefined: a curve ends above its threshold |
| Churches | A2′−A0 | 30k / 35k | 45k / 45k | 32.50k | 45k | -12.50k | ok |
| OASIS-1 | A3−A0 | 5k / 5k | 60k / 60k | 5k | 60k | -55k | ok |
| Bedrooms | A3−A0 | 5k / 20k | 50k / 45k | 12.50k | 47.50k | -35k | ok |
| interaction | (A3−A0) IXI − (A3−A0) Churches | — | — | — | — | -21.67k | ok |

- Exploratory, defined after seeing the data (2026-09-29), not pre-registered. Settling step: the first checkpoint after which the run's LSD curve (500 frozen seeds) stays at or below the primary T_τ threshold (the same-seed A0 run's LSD at the last checkpoint) through the last step.
- Descriptive only: per-seed values, means and differences, no CI and no p-value, so nothing here is confirmatory.
- Why it exists: the pre-registered T_τ (first crossing, tables 2, 3 and 7) fires at the first checkpoint in most runs because the LSD curve is low at 5k, higher over 10k-30k and lower again by the end.

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results (collection COMPLETE, created 2026-09-29T11:57:08.292774+00:00, git cff6585925); tables: git 4c8c81b19a, 2026-09-29T11:57:51.909550+00:00</sub>
