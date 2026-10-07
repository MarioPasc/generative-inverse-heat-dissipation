# Paper figures and tables (M8)

Every file here is generated. **Never edit by hand; rerun the command.** The design is in
`docs/SPECIFICATIONS/M8-paper/` (`00-framing.md`, `figure-plan-part1.md`).

**Formats.**

- The **SVG** is the editable master for Inkscape: text stays as text, images are embedded, ids
  are deterministic.
- The **PDF** is the file LaTeX includes.
- The **PNG** (300 dpi) is for quick viewing.

All outputs are byte-stable.

| item | files | notes | command |
|---|---|---|---|
| **F1**, the visual abstract | `f1_visual_abstract.{svg,pdf,png}` | [`F1.md`](F1.md): caption, sources, selected indices, checks | `python -m ihdm.cli.paper_f1 --data-root $IHDM_DATA_ROOT --eval-dir <SanDisk>/evaluation/eval_2488269 --out docs/RESULTS/paper` |
| **F2**, the four configurations on MRI | `f2_arms.{svg,pdf,png}` | [`F2.md`](F2.md) | `python -m ihdm.cli.paper_f2 --data-root $IHDM_DATA_ROOT --results $IHDM_DATA_ROOT/_results --eval-dir <SanDisk>/evaluation/eval_2488269 --heldout docs/RESULTS/heldout_fidelity/heldout_fidelity.json docs/RESULTS/paper/f2_a2_heldout/heldout_fidelity.json --out docs/RESULTS/paper` |
| **T1**, the main table | `t1_main.{tex,md}` | [`T1.md`](T1.md) | produced by the F2 command |
| +spacing held-out extension | `f2_a2_heldout/` | the T7.5 analysis applied to `ixi_A2_s1` and `ixi_A2_s2`; the A0 values equal T7.5 exactly | `python -m ihdm.cli.paper_f2 a2-heldout …` (see `F2.md`) |

**Configuration names** in every figure and table:

- default (A0: W/2, log);
- +prior (A1: W/8, log);
- +spacing (A2: W/2, IXI-matched);
- matched (A3: W/8, IXI-matched).

Width 5.5 in (the NIPS 2015 `\textwidth`); smallest font 7 pt.

**Wave W17 (2026-10-06):**

| item | files | notes | command |
|---|---|---|---|
| **A1**, appendix: the natural-image reference runs | `a1_natural_images.{svg,pdf,png}`, `a1_table.{tex,md}` | [`A1.md`](A1.md): caption, about 155 words of text (the intrinsic-dimension sentence filled by main), sources | `python -m ihdm.cli.paper_a1 --results $IHDM_DATA_ROOT/_results --eval-dir <SanDisk>/evaluation/eval_2488269 --out docs/RESULTS/paper` |
| intrinsic dimension (supports A1 and `00-framing.md` §2) | `../intrinsic_dimension/{intrinsic_dimension.json, README.md, id_vs_n.*}` | MLE k = 10, N = 3,200: IXI 17.2, OASIS-1 22.1, Churches 30.1, Bedrooms 30.8; the PCA participation ratio is higher for MRI | `python -m ihdm.cli.intrinsic_dim --data-root $IHDM_DATA_ROOT --out docs/RESULTS/intrinsic_dimension` |

**Wave W18 (2026-10-07):**

| item | files | notes | command |
|---|---|---|---|
| **FM**, the method figure | `fm_method.{svg,pdf,png}` | [`FM.md`](FM.md): caption, modes, variance and levels per octave, the macro-step check, sources. 6 explicit macro-steps plus an ellipsis for 32–96 c/img; `stixsans` mathtext (FM only) | `python -m ihdm.cli.paper_fm --data-root $IHDM_DATA_ROOT --out docs/RESULTS/paper` |

**Wave W19 (2026-10-07):**

| item | files | notes | command |
|---|---|---|---|
| **F3**, fidelity against diversity | `f3_fidelity_diversity.{svg,pdf,png}` | [`F3.md`](F3.md): caption, numbers per run with sources, the R5 sensitivity, the denominator check, the corrections to the ticket | `python -m ihdm.cli.paper_f3 --results $IHDM_DATA_ROOT/_results --heldout docs/RESULTS/heldout_fidelity/heldout_fidelity.json docs/RESULTS/paper/f2_a2_heldout/heldout_fidelity.json --tables docs/RESULTS/tables/tables.json --exploratory docs/RESULTS/exploratory/exploratory.json --constants docs/RESULTS/inherited_band_constants.json --out docs/RESULTS/paper` |

**2026-10-07, later (Mario):** F3 leaves the paper. Its panel (a) moves into a reworked F2, which also gains a distance-to-seed plot and band-pass images on the spectrum (ticket T8.8, wave W21). The F3 files stay here as a record. The paper's main text uses F1, FM, F2 and T1.
