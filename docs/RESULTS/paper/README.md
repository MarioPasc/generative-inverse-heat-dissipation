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

**Pending:**

- appendix A1 (natural-image reference runs) and the intrinsic-dimension result: T8.3 and T8.0,
  wave W17;
- F3, fidelity against diversity: part 2, not yet designed.
