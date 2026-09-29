## Table 7 — T_τ: first checkpoint at which a run's LSD reaches its A0 run's final LSD

**PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated): not for quoting.**

**Status: incomplete: 24/30 runs** (missing: oasis1_A3_s1, oasis1_A3_s2, lsun_bedroom_A0_s1, lsun_bedroom_A0_s2, lsun_bedroom_A3_s1, lsun_bedroom_A3_s2)

| dataset | arm | seed | LSD 5k | LSD 60k | threshold (A0, 500) | T_τ | threshold (A0, 2k) | T_τ, 2k threshold (sensitivity) | settling step (exploratory) | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| IXI | A0 | 1 | 0.2273 | 0.2497 | 0.2497 | 5k | 0.2527 | 5k | 55k | ok |
| IXI | A0 | 2 | 0.2444 | 0.2404 | 0.2404 | 35k | 0.2449 | 5k | 55k | ok |
| IXI | A0 | 3 | 0.2060 | 0.2271 | 0.2271 | 5k | 0.2289 | 5k | 60k | ok |
| IXI | A3 | 1 | 0.1773 | 0.08400 | 0.2497 | 5k | 0.2527 | 5k | 5k | ok |
| IXI | A3 | 2 | 0.1612 | 0.08047 | 0.2404 | 5k | 0.2449 | 5k | 5k | ok |
| IXI | A3 | 3 | 0.1491 | 0.07686 | 0.2271 | 5k | 0.2289 | 5k | 5k | ok |
| IXI | A1 | 1 | 0.2129 | 0.1180 | 0.2497 | 5k | 0.2527 | 5k | 5k | ok |
| IXI | A1 | 2 | 0.1970 | 0.07907 | 0.2404 | 5k | 0.2449 | 5k | 5k | ok |
| IXI | A2 | 1 | 0.1667 | 0.2284 | 0.2497 | 5k | 0.2527 | 5k | 30k | ok |
| IXI | A2 | 2 | 0.2123 | 0.2042 | 0.2404 | 5k | 0.2449 | 5k | 5k | ok |
| Churches | A0 | 1 | 0.7529 | 1.445 | 1.445 | 5k | 1.452 | 5k | 45k | ok |
| Churches | A0 | 2 | 0.4034 | 1.129 | 1.129 | 5k | 1.127 | 5k | 45k | ok |
| Churches | A0 | 3 | 0.7760 | 1.023 | 1.023 | 5k | 1.024 | 5k | 55k | ok |
| Churches | A3 | 1 | 0.6040 | 0.6416 | 1.445 | 5k | 1.452 | 5k | 5k | ok |
| Churches | A3 | 2 | 0.9013 | 1.092 | 1.129 | 5k | 1.127 | 5k | 25k | ok |
| Churches | A3 | 3 | 0.7242 | 0.7796 | 1.023 | 5k | 1.024 | 5k | 25k | ok |
| Churches | A1 | 1 | 0.7696 | 0.6656 | 1.445 | 5k | 1.452 | 5k | 5k | ok |
| Churches | A1 | 2 | 0.9379 | 1.015 | 1.129 | 5k | 1.127 | 5k | 30k | ok |
| Churches | A2 | 1 | 0.8862 | 1.101 | 1.445 | 5k | 1.452 | 5k | 20k | ok |
| Churches | A2 | 2 | 1.136 | 1.347 | 1.129 | not reached | 1.127 | not reached | not reached | ok |
| Churches | A2′ | 1 | 0.4945 | 0.9014 | 1.445 | 5k | 1.452 | 5k | 30k | ok |
| Churches | A2′ | 2 | 0.7143 | 0.8581 | 1.129 | 5k | 1.127 | 5k | 35k | ok |
| OASIS-1 | A0 | 1 | 0.2477 | 0.2472 | 0.2472 | 60k | 0.2434 | not reached | 60k | ok |
| OASIS-1 | A0 | 2 | 0.2320 | 0.2209 | 0.2209 | 50k | 0.2183 | 50k | 60k | ok |
| OASIS-1 | A3 | 1 | — | — | — | — | — | — | — | missing: oasis1_A3_s1 |
| OASIS-1 | A3 | 2 | — | — | — | — | — | — | — | missing: oasis1_A3_s2 |
| Bedrooms | A0 | 1 | — | — | — | — | — | — | — | missing: lsun_bedroom_A0_s1 |
| Bedrooms | A0 | 2 | — | — | — | — | — | — | — | missing: lsun_bedroom_A0_s2 |
| Bedrooms | A3 | 1 | — | — | — | — | — | — | — | missing: lsun_bedroom_A3_s1, lsun_bedroom_A0_s1 |
| Bedrooms | A3 | 2 | — | — | — | — | — | — | — | missing: lsun_bedroom_A3_s2, lsun_bedroom_A0_s2 |

- 05 §2: 'T_τ(arm) is the smallest checkpoint step s at which LSD_arm(s) ≤ LSD^A0_final', LSD^A0_final being the A0 run with the same seed at its last checkpoint; +∞ ('not reached') if never.
- Primary threshold: the A0 run's LSD at 60k on the curve's own estimator (the 500 frozen seeds and noise stream of every curve point, D17), so curve and threshold share one sample set. For A0 itself T_τ ≤ the last step by construction, and it is earlier when the A0 curve dipped below its final value before the end.
- Sensitivity: the A0 run's 2,000-sample final LSD. It differs from the primary threshold by the set difference (0.003 on IXI A0 s1, as large as the gate's resolution), which can move T_τ by itself; read the primary column.
- Resolution 5k steps (the evaluated checkpoints, D16/D22).
- The settling step is exploratory, defined after seeing the data (2026-09-29), not pre-registered: the first checkpoint after which the curve stays at or below the primary threshold to the end (table 9).

<sub>source: /media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project/_results_partial_24 (collection INCOMPLETE, created 2026-09-29T07:38:12.518998+00:00, git 9fb5c84635); tables: git 66ec64ef68, 2026-09-29T08:04:59.317403+00:00</sub>
