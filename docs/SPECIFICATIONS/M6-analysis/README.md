# M6 — Analysis, tables and figures for the report (sketch; completed when M5 closes)

## T6.1 — Tables and statistics

`ihdm/analysis/tables.py` + `python -m ihdm.cli.analyse --results results/ --out docs/RESULTS/tables/`:
per dataset × arm × seed cell table; the paired deltas against A0; the interaction
$\Delta_{\text{MRI}} - \Delta_{\text{photo}}$ for each endpoint (diversity, $M$, final LSD,
$T_\tau$, FID) with bootstrap CIs and permutation $p$; the share carried by A1 and A2; the A2′
control; the transfer sign table (IXI vs OASIS-1, Churches vs Bedrooms). Markdown and LaTeX
(`booktabs`) output.

## T6.2 — Figures

`ihdm/analysis/figures.py`: LSD versus iteration per dataset (arms as lines, seeds as thin lines,
the A0 final LSD as the $T_\tau$ threshold); the per-octave LSD profile at the final checkpoint;
diversity and $M$ per dataset and arm (points per seed); the PCA-around-seed panels (two seeds,
A0 vs A3, IXI vs Churches); the inherited-band figure (measured versus predicted radial curve,
both terminal blurs); sample grids (seed row + samples) for A0 and A3 on each dataset; the
training sanity panel (loss, it/s, per-octave loss). Vector PDF, matplotlib, widths matching the
report template (3.3 in single column, 6.75 in full width).
