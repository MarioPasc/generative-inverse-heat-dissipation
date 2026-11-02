# Methodology section: plan (2026-10-28)

Written by Claude (Opus 5.5) for the team before drafting `proyecto/sections/methodology.tex`.
Sources: `M8-paper/00-framing.md` (claims ledger), `RESULTS/results_discussion.md` §2–§2a,
`RESULTS/heldout_fidelity/README.md`, `projects/GenAI/learning/01-ihdm-method.md`, the course-notation
sheets `projects/GenAI/learning/course-notation-{sde-score,diffusion-eval}.md`, and the code
(`ihdm/spectral/schedules.py`, `ihdm/sampling/chain.py`, `ihdm/metrics/{memorisation,diversity}.py`).

## Narrative line (one sentence per step)

1. IHDM is the course's score-based diffusion with the heat equation moved from the density to the image.
2. Two of its settings, the terminal blur of the prior and the spacing of the levels, encode natural-image
   assumptions; everything else is kept.
3. We set both from the training data's between-image variance spectrum $P_i$: the prior keeps the coarse
   anatomy that registered subjects share; the spacing equalises the per-level regression target, of which
   the paper's log spacing is the $1/f^2$ special case.
4. Keeping more of a real image in the prior should raise fidelity and lower diversity and novelty, so a single
   population score cannot judge it; the evaluation measures fidelity, diversity and copying separately, and
   seeds the sampler from unseen subjects to separate memorisation from inheritance.

## Subsections, contents and length (target ≈ 1,150 words + 9 numbered equations + 1 code figure)

| subsection | role (user's three parts) | contents | words |
|---|---|---|---|
| intro paragraph | — | what is kept, what changes, what is measured; pointer to the experimental design | 60 |
| 3.1 From score-based models to inverse heat dissipation | **kept from the paper** | Fokker–Planck with $A=0$ → heat equation on the density; NCSN's $p_\sigma$; IHDM moves the PDE to the image; DCT solution (per-mode linear ODE); forward kernel; reverse Gaussian; ELBO → MSE (DDPM parallel); $K\to\infty$ link to the score and the Langevin step ($\delta=\sqrt2\sigma$, paper uses $1.25\sigma$); the KDE prior as the course's $p_\sigma$ formula on blurred training images; defaults kept | 420 |
| 3.2 Matching the prior and the level spacing to brain MRI | **what we change** | $P_i$; the inherited share $I(\sigma_{B,K})$; W/2 → W/8 and why (shared anatomy, the paper's own memorisation warning); $R_k$ and the matched rule; log spacing equalises $R_k$ exactly for $1/f^2$ in the continuum (derivation); fitted once on IXI, per terminal blur, frozen for OASIS-1; the 2×2 | 340 |
| 3.3 Measuring fidelity and diversity separately | **our evaluation** | why one population score is not enough; four families (per-sample fidelity, across-image diversity, population distance, within-seed diversity) + spectral + copying; training-seeded vs held-out-seeded sets and $R^-$; the confounds | 380 |
| 3.4 Experimental design | later | data and registration, configurations and seeds, training recipe, development protocol, statistics | (next task) |

## Formula + code per contribution (user request 2026-10-28)

| contribution | formula | code / pseudocode |
|---|---|---|
| prior (W/8) | KDE prior; inherited share $I$ | Python excerpt of `ihdm/sampling/chain.py` (`prior_state` + `reverse_chain`), the line that the change touches marked |
| matched spacing | $R_k$; $R_2=\dots=R_K$ | Algorithm: bisection on the common target, inner bisection per level (`schedules.py: matched_schedule`, `_march`) |
| evaluation | precision, recall, KID, LSD, $D_{pix}$, seed-NN | Python excerpt of `memorisation._seed_fraction` and `diversity._seed_variance` |

One figure with three panels holds the code so the three boxes cost one float.

## Notation (course first; IHDM where the course has nothing)

- Data $\mathbf{x}\sim p_0(\mathbf{x})$ (SBM:3); state $\mathbf{u}_k$, $\mathbf{u}_0=\mathbf{x}$; Gaussian $\mathcal N(\cdot;\text{mean},\text{covariance})$ as in SBM:15.
- Course symbols kept: $p_\sigma$ (SBM:15), $\pi(\mathbf{x})$ for the prior (SBM:5), $\alpha$ for the Langevin step (SBM:8), $A,B$ for Fokker–Planck (SDE:26).
- IHDM symbols: $F(t)$, $\mathbf V$, $\lambda_i$ (eigenvalues; the course's loss weights $\lambda_i$ are not used), $\sigma$, $\delta$, $K$, $\sigma_{B,k}$, $t_k=\sigma_{B,k}^2/2$.
- Ours: $P_i$, $I$, $R_k$, $D_{pix}$, seed-NN, $R^-$.

## Must not claim (from 00-framing.md §4)

No results in Methods. No "the schedule must follow the spectrum in general"; no "significant"; W/8 always
stated with its copying cost; "continuum" for the $1/f^2$ equality.
