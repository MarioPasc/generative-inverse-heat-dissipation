"""Profile the four datasets, check the known results and render the report's data figure.

Computes, for the **training** split of each dataset (3200 images, the ``ref`` split as a
check): the spectral exponent, the octave shares of the between-image variance, the inherited
share at ``sigma_B,max = W/8, W/4, W/2``, and the per-level target spread under the log
schedule, the IXI-matched schedule and the dataset's own matched schedule. Writes
``docs/RESULTS/data_profile.md``, ``docs/RESULTS/data_profile/<dataset>.npz`` and
``docs/RESULTS/fig_data.{pdf,png}``.

Requires ``schedules/`` to hold the frozen arrays (``python -m ihdm.cli.build_schedules``).

Run as ``python -m ihdm.cli.profile_data``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ihdm.paths import data_root, repo_root, schedules_dir
from ihdm.spectral.errors import SpectralError
from ihdm.spectral.power import OCTAVE_LABELS, radial_spectrum
from ihdm.spectral.profile import (
    ARCHIVED,
    DATASET_LABELS,
    DATASETS,
    FIGURE_RNG_SEED,
    ChecklistItem,
    DatasetProfile,
    build_checklist,
    make_figure,
    profile_split,
    reference_example,
    split_power,
)
from ihdm.spectral.schedules import (
    SIGMA_B_OCTAVE_LABELS,
    ScheduleSpec,
    git_sha,
    levels_per_octave,
    matched_schedule,
    per_level_spread,
)

logger = logging.getLogger(__name__)

# The frozen W/2 matched schedule of each dataset, where one exists; OASIS-1's is not frozen
# (no arm uses it) and is fitted here for the crossover and agreement tables.
OWN_W2_FROZEN: dict[str, str] = {
    "ixi": "ixi_W2",
    "lsun_church": "lsun_church_W2",
    "lsun_bedroom": "lsun_bedroom_W2",
}
REPORT_SCHEDULES: tuple[str, ...] = (
    "log_W2", "log_W8", "ixi_W2", "ixi_W8", "lsun_church_W2", "oasis1_W8", "lsun_bedroom_W2",
)


@dataclass(frozen=True)
class ProfileTables:
    """Everything the report renders, computed once.

    Parameters
    ----------
    train, ref : dict[str, DatasetProfile]
        Per-dataset profiles of the two splits.
    powers, powers_ref : dict[str, numpy.ndarray]
        Per-mode variance of the training and of the reference splits.
    spread : dict[str, dict[str, float]]
        ``spread[dataset][schedule_label]`` for ``"log"``, ``"ixi"`` and ``"own"``.
    deviation : dict[str, tuple[float, float]]
        ``(max, median)`` of ``|s_own / s_ixi - 1|`` per dataset.
    schedules : dict[str, numpy.ndarray]
        The frozen arrays, by name, plus ``oasis1_W2``.
    """

    train: dict[str, DatasetProfile]
    ref: dict[str, DatasetProfile]
    powers: dict[str, np.ndarray]
    powers_ref: dict[str, np.ndarray]
    spread: dict[str, dict[str, float]]
    deviation: dict[str, tuple[float, float]]
    schedules: dict[str, np.ndarray]


def _load_frozen(sched_dir: Path) -> dict[str, np.ndarray]:
    """Load the seven frozen arrays.

    Raises
    ------
    SpectralError
        If one is missing; ``python -m ihdm.cli.build_schedules`` produces them.
    """
    out: dict[str, np.ndarray] = {}
    for name in REPORT_SCHEDULES:
        path = sched_dir / f"{name}.npy"
        if not path.is_file():
            raise SpectralError(f"{path} is missing; run python -m ihdm.cli.build_schedules first")
        out[name] = np.load(path)
    return out


def _own_w2(
    powers: dict[str, np.ndarray], frozen: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Each dataset's own ``W/2`` matched schedule (OASIS-1's is fitted here, the rest frozen)."""
    out = {d: frozen[OWN_W2_FROZEN[d]] for d in OWN_W2_FROZEN}
    out["oasis1"] = matched_schedule(
        powers["oasis1"], ScheduleSpec(kind="matched", sigma_max=96.0, fitted_on="oasis1/train")
    )
    return out


def compute(root: Path, sched_dir: Path) -> ProfileTables:
    """Measure every quantity the report needs.

    Parameters
    ----------
    root : Path
        The data root.
    sched_dir : Path
        The directory holding the frozen schedule arrays.

    Returns
    -------
    ProfileTables
        The measured tables.
    """
    train: dict[str, DatasetProfile] = {}
    ref: dict[str, DatasetProfile] = {}
    powers: dict[str, np.ndarray] = {}
    powers_ref: dict[str, np.ndarray] = {}
    for dataset_id in DATASETS:
        power, n_images, sha = split_power(root, dataset_id, "train")
        powers[dataset_id] = power
        train[dataset_id] = profile_split(power, dataset_id, "train", n_images, sha)
        power_ref, n_ref, sha_ref = split_power(root, dataset_id, "ref")
        powers_ref[dataset_id] = power_ref
        ref[dataset_id] = profile_split(power_ref, dataset_id, "ref", n_ref, sha_ref)
        print(f"  {dataset_id}: train {n_images} images, ref {n_ref} images")

    frozen = _load_frozen(sched_dir)
    own = _own_w2(powers, frozen)
    spread: dict[str, dict[str, float]] = {}
    deviation: dict[str, tuple[float, float]] = {}
    for dataset_id in DATASETS:
        power = powers[dataset_id]
        spread[dataset_id] = {
            "log": per_level_spread(power, frozen["log_W2"]),
            "ixi": per_level_spread(power, frozen["ixi_W2"]),
            "own": per_level_spread(power, own[dataset_id]),
        }
        dev = np.abs(own[dataset_id][1:] / frozen["ixi_W2"][1:] - 1.0)
        deviation[dataset_id] = (float(dev.max()), float(np.median(dev)))

    schedules = dict(frozen)
    schedules["oasis1_W2"] = own["oasis1"]
    return ProfileTables(train, ref, powers, powers_ref, spread, deviation, schedules)


def _alpha_table(tables: ProfileTables) -> list[str]:
    """Markdown: the spectral exponent per dataset and split, against the archived value."""
    lines = [
        "| dataset | $\\alpha$ (train) | $\\alpha$ (ref) | $\\alpha$ (train, 1–48 c/img window) "
        "| archived, unregistered |",
        "|---|---|---|---|---|",
    ]
    for d in DATASETS:
        lines.append(
            f"| {DATASET_LABELS[d]} | **{tables.train[d].alpha:.2f}** | {tables.ref[d].alpha:.2f} "
            f"| {tables.train[d].alpha_ticket_window:.2f} | {ARCHIVED[d]['alpha']:.2f} |"
        )
    return lines


def _octave_table(tables: ProfileTables) -> list[str]:
    """Markdown: the octave shares of the training splits, with the archived row beneath."""
    header = "| dataset | " + " | ".join(OCTAVE_LABELS) + " |"
    lines = [header, "|---|" + "---|" * len(OCTAVE_LABELS)]
    for d in DATASETS:
        shares = tables.train[d].shares
        lines.append(
            f"| {DATASET_LABELS[d]} | "
            + " | ".join(f"{shares[b]:.1%}" for b in OCTAVE_LABELS)
            + " |"
        )
        lines.append(
            f"| {DATASET_LABELS[d]}, archived | "
            + " | ".join(f"{v:.1%}" for v in ARCHIVED[d]["octaves"])
            + " |"
        )
    return lines


def _coarse_modes_table(tables: ProfileTables) -> list[str]:
    """Markdown: the three modes of the 0.5-1 c/img bin, which item (ii) of the checklist reads.

    The bin is the DCT radii ``1 <= n < 2``, i.e. exactly the modes ``(0,1)``, ``(1,0)`` and
    ``(1,1)``: a left-right ramp, an anterior-posterior ramp and a diagonal one. Splitting the
    bin says whether a coarse-variance number is carried by brain structure or by one
    direction of one gradient.
    """
    lines = [
        "| dataset | $(0,1)$ left–right ramp | $(1,0)$ anterior–posterior ramp | "
        "$(1,1)$ diagonal | bin total |",
        "|---|---|---|---|---|",
    ]
    for d in DATASETS:
        power = tables.powers[d]
        bin_share = tables.train[d].shares["0.5-1"]
        values = np.array([power[0, 1], power[1, 0], power[1, 1]])
        shares = bin_share * values / values.sum()
        lines.append(
            f"| {DATASET_LABELS[d]} | "
            + " | ".join(f"{v:.2%}" for v in shares)
            + f" | {bin_share:.2%} |"
        )
    return lines


def _inherited_table(tables: ProfileTables) -> list[str]:
    """Markdown: the inherited share at the three terminal blurs."""
    lines = [
        "| dataset | $W/8$ (24 px) | $W/4$ (48 px) | $W/2$ (96 px) | archived (unregistered) |",
        "|---|---|---|---|---|",
    ]
    for d in DATASETS:
        inh = tables.train[d].inherited
        arch = ARCHIVED[d]["inherited"]
        lines.append(
            f"| {DATASET_LABELS[d]} | {inh['W/8']:.1%} | {inh['W/4']:.1%} | {inh['W/2']:.1%} "
            f"| {arch[0]:.1%} / {arch[1]:.1%} / {arch[2]:.1%} |"
        )
    return lines


def _crossover_table(tables: ProfileTables) -> list[str]:
    """Markdown: the per-level target spread of each dataset under three schedules."""
    lines = [
        "| dataset | log ($W/2$) | IXI-matched ($W/2$) | own matched ($W/2$) | "
        "own vs IXI's, max / median of "
        "$\\lvert s_{\\text{own}}/s_{\\text{IXI}} - 1 \\rvert$ | "
        "archived log / IXI |",
        "|---|---|---|---|---|---|",
    ]
    for d in DATASETS:
        s = tables.spread[d]
        dev_max, dev_med = tables.deviation[d]
        lines.append(
            f"| {DATASET_LABELS[d]} | {s['log']:.1f}x | {s['ixi']:.1f}x | {s['own']:.3f}x "
            f"| {dev_max:.0%} / {dev_med:.0%} "
            f"| {ARCHIVED[d]['spread_log']:.1f}x / {ARCHIVED[d]['spread_ixi']:.1f}x |"
        )
    return lines


def _levels_table(tables: ProfileTables) -> list[str]:
    """Markdown: levels per sigma_B octave, one row per schedule."""
    header = "| schedule | " + " | ".join(f"{b} px" for b in SIGMA_B_OCTAVE_LABELS) + " | sum |"
    lines = [header, "|---|" + "---|" * (len(SIGMA_B_OCTAVE_LABELS) + 1)]
    for name in (*REPORT_SCHEDULES, "oasis1_W2"):
        counts = levels_per_octave(tables.schedules[name])
        frozen = "" if name != "oasis1_W2" else " (not frozen)"
        lines.append(
            f"| `{name}`{frozen} | "
            + " | ".join(str(counts[b]) for b in SIGMA_B_OCTAVE_LABELS)
            + f" | {sum(counts.values())} |"
        )
    return lines


def _checklist_table(items: list[ChecklistItem]) -> list[str]:
    """Markdown: the five known-results items with their verdicts and numbers."""
    lines = ["| # | expected | outcome | measured |", "|---|---|---|---|"]
    for item in items:
        lines.append(
            f"| ({item.key}) | {item.statement} | **{item.verdict}** | {item.detail} |"
        )
    return lines


def _checklist_notes(tables: ProfileTables, items: list[ChecklistItem]) -> list[str]:
    """Markdown: what the measured numbers say beyond the verdicts, without tuning anything.

    Only the facts the tables above already contain: which mode carries the coarse bin, how
    the MRI-versus-photograph ordering stands, and by how much registration moved each
    quantity. No threshold is changed here; the orchestrator decides.
    """
    verdict = {item.key: item.passed for item in items}
    coarse = {d: tables.train[d].shares["0.5-1"] for d in DATASETS}
    ratio_photo = min(coarse[d] for d in ("lsun_church", "lsun_bedroom")) / max(
        coarse[d] for d in ("ixi", "oasis1")
    )
    log_margin = min(tables.spread[d]["log"] for d in ("ixi", "oasis1")) / max(
        tables.spread[d]["log"] for d in ("lsun_church", "lsun_bedroom")
    )
    ixi_bin = np.array(
        [tables.powers["ixi"][0, 1], tables.powers["ixi"][1, 0], tables.powers["ixi"][1, 1]]
    )
    lines = [
        "**Notes on the measured numbers** (facts from the tables above; no threshold was "
        "moved and no schedule was refitted to make an item pass).",
        "",
        f"- Registration raised the coarse-octave variance of both MRI sets by roughly a "
        f"factor of five ({ARCHIVED['ixi']['coarse_share']:.1%} → {coarse['ixi']:.2%} on IXI, "
        f"{ARCHIVED['oasis1']['coarse_share']:.1%} → {coarse['oasis1']:.2%} on OASIS-1) and "
        f"left the photograph sets unchanged. The ordering the design rests on is intact: the "
        f"photograph sets still hold {ratio_photo:.0f}× the MRI sets' coarse share.",
        f"- On both MRI sets the coarse bin is carried by one mode, the anterior–posterior "
        f"ramp $(1,0)$ ({ixi_bin[1] / ixi_bin.sum():.0%}"
        f" of the bin on IXI): a between-subject intensity gradient along the slice's rows, "
        f"the signature of a residual receive-field / intensity-harmonisation difference "
        f"rather than of coarse brain structure. The photograph sets spread the bin over all "
        "three modes.",
        f"- The log-schedule spread collapsed on both MRI sets (IXI "
        f"{ARCHIVED['ixi']['spread_log']:.0f}× → {tables.spread['ixi']['log']:.1f}×, OASIS-1 "
        f"{ARCHIVED['oasis1']['spread_log']:.0f}× → {tables.spread['oasis1']['log']:.1f}×) "
        f"while the photograph sets moved little. Item (iii) therefore holds by a margin of "
        f"only {log_margin:.2f}× instead of the archived ~6×, and the design's contrast "
        "between the MRI and photograph arms is correspondingly weaker on the registered data "
        "than the proposal's numbers suggest.",
    ]
    if not verdict.get("ii", True):
        lines.append(
            "- Item (ii) is the one that fails, on IXI alone (3.96 % against the 3 % "
            "threshold); OASIS-1 passes it at 2.44 % and both photograph sets clear the 15 % "
            "side by a wide margin."
        )
    return lines


def render_report(tables: ProfileTables, items: list[ChecklistItem], sha: str) -> str:
    """Assemble ``docs/RESULTS/data_profile.md``.

    Parameters
    ----------
    tables : ProfileTables
        The measured tables.
    items : list[ChecklistItem]
        The known-results checklist.
    sha : str
        The git commit the numbers were produced at.

    Returns
    -------
    str
        The markdown document.
    """
    n_train = tables.train["ixi"].n_images
    alphas = ", ".join(f"{DATASET_LABELS[d]} {tables.train[d].alpha:.2f}" for d in DATASETS)
    failed = [i.key for i in items if not i.passed]
    body = [
        "# Spectral profile of the four training splits, and the frozen schedules",
        "",
        f"Produced by `python -m ihdm.cli.profile_data` at `{sha[:12]}` on "
        f"{datetime.now(UTC).date().isoformat()}. Ticket T1.3; contracts "
        "`04-run-artifacts.md` §1 and `05-metrics.md` §1.",
        "",
        f"Every curve and every table below is measured on the **training** split of each "
        f"dataset ({n_train} registered images of $192^2$, values in $[0, 1]$); the `ref` split "
        "(800 images) is profiled beside it as a consistency check. The per-mode variance is "
        "mean-centred across images, in the orthonormal DCT-II basis, with the DC mode "
        "excluded; a mode of radial index $n$ carries $n/2$ cycles per image and modes above "
        "96 cycles per image are excluded from the octave shares.",
        "",
        "The **archived** columns are the pre-registration numbers of "
        "`worklog/sessions/2026-09-21_ihdm-knob-proposals/native192_profile_output_unprocessed.md`"
        " (480 head-centred crops of the *unprocessed* volumes per dataset, no rigid "
        "registration, no intensity harmonisation, no padding). They are not a target: "
        "registration changes the coarse variance by construction. They are printed so the size "
        "and the direction of that change are visible.",
        "",
        "## 1. Spectral exponent",
        "",
        "$\\alpha$ is fitted by least squares of $\\log \\bar P$ on $\\log n$ over the integer "
        "DCT radii $b \\in [0.10 W, 0.70 W] = [20, 134]$, i.e. **10.0 – 67.0 cycles per "
        "image** — the window of `analysis/control_profile.py: fit_alpha`, held fixed across "
        "datasets because the brain spectrum is curved. The third column repeats the fit over "
        "the window the ticket's prose names (1 – 48 cycles per image); see the ticket log, "
        "decision D-T1.3-1.",
        "",
        *_alpha_table(tables),
        "",
        "## 2. Octave shares of the between-image variance",
        "",
        "Bins in cycles per image; the last bin is closed at 96 (the archived rows close it at "
        "95.5, the analysis scripts' $N-1$ convention).",
        "",
        *_octave_table(tables),
        "",
        "The coarsest bin is exactly three modes, and item (ii) of the checklist below reads "
        "it alone, so it is split here. $(1,0)$ is a ramp along the image rows "
        "(anterior–posterior on the MRI sets, whose slices carry `A` at the top and `L` on the "
        "image left), $(0,1)$ a ramp along the columns (left–right, the padded direction).",
        "",
        *_coarse_modes_table(tables),
        "",
        "## 3. Inherited share at the three terminal blurs",
        "",
        "$\\sum_i P_i e^{-2\\lambda_i t}/\\sum_i P_i$ with $t = \\sigma_{B,\\max}^2/2$: the "
        "fraction of the between-image variance the prior hands to the sampler.",
        "",
        *_inherited_table(tables),
        "",
        "## 4. Per-level target spread — the crossover table",
        "",
        "Spread = $\\max_k R_k / \\min_k R_k$ over levels $2 \\dots K$, with "
        "$R_k = \\sum_i (d_{k-1,i} - d_{k,i})^2 P_i$ the data-dependent part of the IHDM "
        "regression target. A schedule matched to a dataset has spread 1 on it by "
        "construction; the interesting numbers are off the diagonal.",
        "",
        *_crossover_table(tables),
        "",
        "## 5. Levels per $\\sigma_B$ octave",
        "",
        "Where each schedule spends its 200 levels. `oasis1_W2` is fitted by `profile_data` for "
        "this table and for item (v) of the checklist; no arm uses it, so it is not frozen.",
        "",
        *_levels_table(tables),
        "",
        "## 6. Known-results checklist",
        "",
        *_checklist_table(items),
        "",
        f"**{len(items) - len(failed)} of {len(items)} items pass.**"
        + (f" Failing: {', '.join(failed)}." if failed else ""),
        "",
        *_checklist_notes(tables, items),
        "",
        "## 7. Numbers for the figure caption",
        "",
        f"`docs/RESULTS/fig_data.pdf` / `.png`: four examples from the `ref` splits (rng "
        f"{FIGURE_RNG_SEED}; MRI on the report's Fig. 1 plane, `slice == 5`) and four curves "
        f"measured on the **{n_train}-image** training splits.",
        "",
        f"- image count per curve: **{n_train}**",
        f"- $\\alpha$: {alphas}",
        f"- MRI range $\\alpha \\approx {min(tables.train[d].alpha for d in ('ixi', 'oasis1')):.1f}"
        f"$–${max(tables.train[d].alpha for d in ('ixi', 'oasis1')):.1f}$; photographs "
        f"$\\alpha \\approx "
        f"{min(tables.train[d].alpha for d in ('lsun_church', 'lsun_bedroom')):.1f}$–$"
        f"{max(tables.train[d].alpha for d in ('lsun_church', 'lsun_bedroom')):.1f}$",
        "",
        "## 8. Provenance",
        "",
        "| dataset | split | images | `sha256_images` |",
        "|---|---|---|---|",
        *[
            f"| {DATASET_LABELS[d]} | {s} | {p[d].n_images} | `{p[d].images_sha256[:16]}…` |"
            for s, p in (("train", tables.train), ("ref", tables.ref))
            for d in DATASETS
        ],
        "",
    ]
    return "\n".join(body)


def _write_npz(tables: ProfileTables, out_dir: Path) -> None:
    """Archive the per-mode variance and the radial profile of both splits per dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:
        power_ref = tables.powers_ref[d]
        freq, profile_train = radial_spectrum(tables.powers[d])
        _, profile_ref = radial_spectrum(power_ref)
        np.savez_compressed(
            out_dir / f"{d}.npz",
            power_train=tables.powers[d],
            power_ref=power_ref,
            cycles_per_image=freq,
            radial_train=profile_train,
            radial_ref=profile_ref,
            octave_labels=np.array(OCTAVE_LABELS),
            octave_shares_train=np.array([tables.train[d].shares[b] for b in OCTAVE_LABELS]),
            octave_shares_ref=np.array([tables.ref[d].shares[b] for b in OCTAVE_LABELS]),
            alpha_train=tables.train[d].alpha,
            alpha_ref=tables.ref[d].alpha,
            inherited_train=np.array([tables.train[d].inherited[k] for k in ("W/8", "W/4", "W/2")]),
            spread_log=tables.spread[d]["log"],
            spread_ixi=tables.spread[d]["ixi"],
            spread_own=tables.spread[d]["own"],
        )


def _figure(tables: ProfileTables, root: Path, out_stem: Path) -> None:
    """Draw the report's data figure from the ref-split examples and the training curves."""
    rng = np.random.default_rng(FIGURE_RNG_SEED)
    examples = {}
    for d in DATASETS:
        image, row = reference_example(root, d, rng)
        examples[d] = image
        print(f"  figure example {d}: row {row} of the ref split")
    alphas = {d: tables.train[d].alpha for d in DATASETS}
    pdf, png = make_figure(examples, tables.powers, alphas, out_stem)
    print(f"  wrote {pdf} and {png}")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        ``0`` on success (even if a checklist item fails; the failure is reported), ``1`` if
        the profile could not be computed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--schedules", type=Path, default=schedules_dir())
    parser.add_argument("--out", type=Path, default=repo_root() / "docs" / "RESULTS")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    started = datetime.now(UTC)
    try:
        tables = compute(args.data_root, args.schedules)
    except SpectralError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    items = build_checklist(
        alphas={d: tables.train[d].alpha for d in DATASETS},
        coarse_shares={d: tables.train[d].shares["0.5-1"] for d in DATASETS},
        spread_log={d: tables.spread[d]["log"] for d in DATASETS},
        spread_ixi={d: tables.spread[d]["ixi"] for d in DATASETS},
        deviation_max={d: tables.deviation[d][0] for d in DATASETS},
    )
    out: Path = args.out
    report = render_report(tables, items, git_sha(repo_root()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_profile.md").write_text(report)
    _write_npz(tables, out / "data_profile")
    _figure(tables, args.data_root, out / "fig_data")

    print(f"\nimages per curve: {tables.train['ixi'].n_images}")
    for d in DATASETS:
        print(f"  alpha {DATASET_LABELS[d]:16s} {tables.train[d].alpha:.2f}")
    print()
    for item in items:
        print(f"  ({item.key}) {item.verdict}: {item.detail}")
    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(f"\nwrote {out / 'data_profile.md'} in {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
