# Diagnostic training runs (M7, T7.1) — submission record

Post-hoc diagnostic of the photograph failure, `docs/SPECIFICATIONS/M7-diagnostics/README.md`
(the reading rule is pre-registered there). Two runs, arm A0, seed 1, the 60k recipe of the 30
production runs; each changes one factor of `lsun_church_A0_s1`. Launcher and cells:
`slurm/diag_train/`; log: `docs/AGENT-LOGS/M7-diagnostics/T7.1-photo-diagnostic.md`.

| run_id | dataset | N (train / ref / seed) | side | σ_B,max | schedule |
|---|---|---|---|---|---|
| `lsun_church_r128_A0_s1` | `lsun_church_r128` | 4,000 (3,200 / 800 / 40) | 128 | 64 | `log_W2_128` |
| `lsun_church_n32k_A0_s1` | `lsun_church_n32k` | 32,800 (32,000 / 800 / 40) | 192 | 96 | `log_W2` |

(filled as the work proceeds)
