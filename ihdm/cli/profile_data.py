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
    MRI_DATASETS,
    PHOTO_DATASETS,
    SENSITIVITY_IDS,
    ChecklistItem,
    DatasetProfile,
    SiteProfile,
    build_checklist,
    coarse_bin_breakdown,
    make_figure,
    profile_split,
    reference_example,
    sensitivity_root,
    site_profiles,
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

#: Placeholder printed where a "no N4" number does not exist (photographs, or no archive).
NO_VALUE: str = "&mdash;"


@dataclass(frozen=True)
class SensitivityTables:
    """The same quantities, measured on the archived uncorrected MRI datasets.

    The photograph entries repeat the corrected run's numbers: N4 is applied to the MRI side
    only, so their data is bit-identical in both branches and the checklist items that
    compare MRI against photographs need all four values in one place.

    Parameters
    ----------
    train : dict[str, DatasetProfile]
        Per-dataset profile of the uncorrected training splits.
    powers : dict[str, numpy.ndarray]
        Per-mode variance of those splits.
    spread : dict[str, dict[str, float]]
        As :class:`ProfileTables`, but under the schedules refitted on the uncorrected data.
    deviation : dict[str, tuple[float, float]]
        ``(max, median)`` of ``|s_own / s_ixi - 1|`` under those schedules.
    schedules : dict[str, numpy.ndarray]
        ``ixi_W2``, ``ixi_W8``, ``oasis1_W2``, ``oasis1_W8`` refitted on the uncorrected
        training splits.
    sites : dict[str, list[SiteProfile]]
        Per-site coarse-bin numbers of the uncorrected MRI datasets.
    """

    train: dict[str, DatasetProfile]
    powers: dict[str, np.ndarray]
    spread: dict[str, dict[str, float]]
    deviation: dict[str, tuple[float, float]]
    schedules: dict[str, np.ndarray]
    sites: dict[str, list[SiteProfile]]


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
    sites : dict[str, list[SiteProfile]]
        Per-site coarse-bin numbers of the two MRI datasets.
    sensitivity : SensitivityTables | None
        The uncorrected counterpart, or ``None`` when the archive is not on this machine.
    """

    train: dict[str, DatasetProfile]
    ref: dict[str, DatasetProfile]
    powers: dict[str, np.ndarray]
    powers_ref: dict[str, np.ndarray]
    spread: dict[str, dict[str, float]]
    deviation: dict[str, tuple[float, float]]
    schedules: dict[str, np.ndarray]
    sites: dict[str, list[SiteProfile]]
    sensitivity: SensitivityTables | None = None


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


def _spread_and_deviation(
    powers: dict[str, np.ndarray], log_w2: np.ndarray, ixi_w2: np.ndarray,
    own: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, tuple[float, float]]]:
    """Per-level spreads under the log / IXI-matched / own schedules, and the own-vs-IXI gap."""
    spread: dict[str, dict[str, float]] = {}
    deviation: dict[str, tuple[float, float]] = {}
    for dataset_id in DATASETS:
        power = powers[dataset_id]
        spread[dataset_id] = {
            "log": per_level_spread(power, log_w2),
            "ixi": per_level_spread(power, ixi_w2),
            "own": per_level_spread(power, own[dataset_id]),
        }
        dev = np.abs(own[dataset_id][1:] / ixi_w2[1:] - 1.0)
        deviation[dataset_id] = (float(dev.max()), float(np.median(dev)))
    return spread, deviation


def _sensitivity_tables(
    root: Path,
    powers: dict[str, np.ndarray],
    train: dict[str, DatasetProfile],
    frozen: dict[str, np.ndarray],
) -> SensitivityTables | None:
    """Measure the same quantities on the archived uncorrected MRI datasets.

    The matched schedules are **refitted on the uncorrected training splits**: items (iii) to
    (v) of the checklist compare a dataset against its own matched schedule, so scoring the
    uncorrected data against the N4-fitted arrays would measure the correction rather than
    the uncorrected pipeline. The photograph schedules are reused from ``frozen``: their data
    is untouched by N4, so refitting them would reproduce the same arrays.

    Parameters
    ----------
    root : Path
        The data root; the archive lives under ``<root>/_sensitivity``.
    powers : dict[str, numpy.ndarray]
        The corrected run's per-mode variances, for the photograph entries.
    train : dict[str, DatasetProfile]
        The corrected run's profiles, for the photograph entries.
    frozen : dict[str, numpy.ndarray]
        The frozen arrays, for ``log_W2`` and the photograph schedules.

    Returns
    -------
    SensitivityTables | None
        The measured tables, or ``None`` when the archive is not present.
    """
    sroot = sensitivity_root(root)
    missing = [sid for sid in SENSITIVITY_IDS.values() if not (sroot / sid).is_dir()]
    if missing:
        logger.warning("no sensitivity copies under %s (missing %s)", sroot, missing)
        return None

    alt_powers: dict[str, np.ndarray] = dict(powers)
    alt_train: dict[str, DatasetProfile] = dict(train)
    for dataset_id in MRI_DATASETS:
        sid = SENSITIVITY_IDS[dataset_id]
        power, n_images, sha = split_power(sroot, sid, "train")
        alt_powers[dataset_id] = power
        alt_train[dataset_id] = profile_split(power, sid, "train", n_images, sha)
        print(f"  {sid}: train {n_images} images (sensitivity)")

    alt_schedules = {
        f"{dataset_id}_{name}": matched_schedule(
            alt_powers[dataset_id],
            ScheduleSpec(
                kind="matched", sigma_max=sigma_max,
                fitted_on=f"{SENSITIVITY_IDS[dataset_id]}/train",
            ),
        )
        for dataset_id in MRI_DATASETS
        for name, sigma_max in (("W2", 96.0), ("W8", 24.0))
    }
    own = {
        "ixi": alt_schedules["ixi_W2"],
        "oasis1": alt_schedules["oasis1_W2"],
        "lsun_church": frozen["lsun_church_W2"],
        "lsun_bedroom": frozen["lsun_bedroom_W2"],
    }
    spread, deviation = _spread_and_deviation(
        alt_powers, frozen["log_W2"], alt_schedules["ixi_W2"], own
    )
    sites = {d: site_profiles(sroot, SENSITIVITY_IDS[d], "train") for d in MRI_DATASETS}
    return SensitivityTables(alt_train, alt_powers, spread, deviation, alt_schedules, sites)


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
    spread, deviation = _spread_and_deviation(powers, frozen["log_W2"], frozen["ixi_W2"], own)

    schedules = dict(frozen)
    schedules["oasis1_W2"] = own["oasis1"]
    sites = {d: site_profiles(root, d, "train") for d in MRI_DATASETS}
    sensitivity = _sensitivity_tables(root, powers, train, frozen)
    return ProfileTables(
        train, ref, powers, powers_ref, spread, deviation, schedules, sites, sensitivity
    )


def _no_n4(tables: ProfileTables, dataset_id: str) -> DatasetProfile | None:
    """Return the uncorrected profile of an MRI dataset, or ``None`` where there is none."""
    if tables.sensitivity is None or dataset_id not in MRI_DATASETS:
        return None
    return tables.sensitivity.train[dataset_id]


def _alpha_table(tables: ProfileTables) -> list[str]:
    """Markdown: the spectral exponent per dataset and split, against the archived value."""
    lines = [
        "| dataset | $\\alpha$ (train) | $\\alpha$ (ref) | $\\alpha$ (train, 1–48 c/img window) "
        "| **no N4** (train) | archived, unregistered |",
        "|---|---|---|---|---|---|",
    ]
    for d in DATASETS:
        alt = _no_n4(tables, d)
        lines.append(
            f"| {DATASET_LABELS[d]} | **{tables.train[d].alpha:.2f}** | {tables.ref[d].alpha:.2f} "
            f"| {tables.train[d].alpha_ticket_window:.2f} "
            f"| {NO_VALUE if alt is None else format(alt.alpha, '.2f')} "
            f"| {ARCHIVED[d]['alpha']:.2f} |"
        )
    return lines


def _octave_table(tables: ProfileTables) -> list[str]:
    """Markdown: the octave shares of the training splits, with the sensitivity and archived rows.

    The columns of this table are the octaves themselves, so the "no N4" sensitivity figures
    are an extra **row** per MRI dataset rather than an extra column, exactly as the archived
    pre-registration figures already are.
    """
    header = "| dataset | " + " | ".join(OCTAVE_LABELS) + " |"
    lines = [header, "|---|" + "---|" * len(OCTAVE_LABELS)]
    for d in DATASETS:
        shares = tables.train[d].shares
        lines.append(
            f"| {DATASET_LABELS[d]} | "
            + " | ".join(f"{shares[b]:.1%}" for b in OCTAVE_LABELS)
            + " |"
        )
        alt = _no_n4(tables, d)
        if alt is not None:
            lines.append(
                f"| {DATASET_LABELS[d]}, **no N4** | "
                + " | ".join(f"{alt.shares[b]:.1%}" for b in OCTAVE_LABELS)
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
    direction of one gradient, which is the whole reason N4 was added (decision D15).
    """
    lines = [
        "| dataset | $(0,1)$ left–right ramp | $(1,0)$ anterior–posterior ramp | "
        "$(1,1)$ diagonal | bin total | $(1,0)$ share of the bin |",
        "|---|---|---|---|---|---|",
    ]

    def row(label: str, power: np.ndarray, bin_share: float) -> str:
        values = np.array([power[0, 1], power[1, 0], power[1, 1]], dtype=float)
        shares = bin_share * values / values.sum()
        return (
            f"| {label} | " + " | ".join(f"{v:.2%}" for v in shares)
            + f" | {bin_share:.2%} | {values[1] / values.sum():.0%} |"
        )

    for d in DATASETS:
        lines.append(row(DATASET_LABELS[d], tables.powers[d], tables.train[d].shares["0.5-1"]))
        alt = _no_n4(tables, d)
        if alt is not None and tables.sensitivity is not None:
            lines.append(
                row(
                    f"{DATASET_LABELS[d]}, **no N4**",
                    tables.sensitivity.powers[d],
                    alt.shares["0.5-1"],
                )
            )
    return lines


def _site_table(tables: ProfileTables) -> list[str]:
    """Markdown: the coarse bin and its ``(1,0)`` share per acquisition site, before and after N4.

    IXI mixes three sites and two field strengths (Guys 1.5 T, HH 3 T, IOP 1.5 T) and
    OASIS-1 is one scanner, so this is the table that says whether the coarse-bin ramp T1.3
    found is a multi-site artefact and how much of it N4 removes.
    """
    lines = [
        "| dataset | site | subjects | images | coarse share, N4 | coarse share, no N4 | "
        "$(1,0)$ share of the bin, N4 | no N4 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset_id in MRI_DATASETS:
        alt_by_site = {
            entry.site: entry
            for entry in (
                tables.sensitivity.sites.get(dataset_id, [])
                if tables.sensitivity is not None
                else []
            )
        }
        for entry in tables.sites.get(dataset_id, []):
            alt = alt_by_site.get(entry.site)
            lines.append(
                f"| {DATASET_LABELS[dataset_id]} | {entry.site} | {entry.n_subjects} "
                f"| {entry.n_images} | **{entry.coarse_share:.2%}** "
                f"| {NO_VALUE if alt is None else format(alt.coarse_share, '.2%')} "
                f"| **{entry.mode_10_share:.0%}** "
                f"| {NO_VALUE if alt is None else format(alt.mode_10_share, '.0%')} |"
            )
    return lines


def _inherited_table(tables: ProfileTables) -> list[str]:
    """Markdown: the inherited share at the three terminal blurs."""
    lines = [
        "| dataset | $W/8$ (24 px) | $W/4$ (48 px) | $W/2$ (96 px) | "
        "**no N4** $W/8$ / $W/4$ / $W/2$ | archived (unregistered) |",
        "|---|---|---|---|---|---|",
    ]
    for d in DATASETS:
        inh = tables.train[d].inherited
        arch = ARCHIVED[d]["inherited"]
        alt = _no_n4(tables, d)
        alt_cell = (
            NO_VALUE
            if alt is None
            else f"{alt.inherited['W/8']:.1%} / {alt.inherited['W/4']:.1%} "
                 f"/ {alt.inherited['W/2']:.1%}"
        )
        lines.append(
            f"| {DATASET_LABELS[d]} | {inh['W/8']:.1%} | {inh['W/4']:.1%} | {inh['W/2']:.1%} "
            f"| {alt_cell} | {arch[0]:.1%} / {arch[1]:.1%} / {arch[2]:.1%} |"
        )
    return lines


def _crossover_table(tables: ProfileTables) -> list[str]:
    """Markdown: the per-level target spread of each dataset under three schedules.

    The "no N4" column repeats the three spreads on the uncorrected data **under schedules
    refitted on the uncorrected training splits**: the statement each item makes is about a
    dataset together with its own matched schedule, so mixing an uncorrected dataset with an
    N4-fitted schedule would measure the correction instead of the pipeline.
    """
    lines = [
        "| dataset | log ($W/2$) | IXI-matched ($W/2$) | own matched ($W/2$) | "
        "own vs IXI's, max / median of "
        "$\\lvert s_{\\text{own}}/s_{\\text{IXI}} - 1 \\rvert$ | "
        "**no N4**: log / IXI-matched / max dev | archived log / IXI |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in DATASETS:
        s = tables.spread[d]
        dev_max, dev_med = tables.deviation[d]
        alt_cell = NO_VALUE
        if tables.sensitivity is not None:
            alt_s = tables.sensitivity.spread[d]
            alt_dev = tables.sensitivity.deviation[d][0]
            alt_cell = f"{alt_s['log']:.1f}x / {alt_s['ixi']:.1f}x / {alt_dev:.0%}"
        lines.append(
            f"| {DATASET_LABELS[d]} | {s['log']:.1f}x | {s['ixi']:.1f}x | {s['own']:.3f}x "
            f"| {dev_max:.0%} / {dev_med:.0%} | {alt_cell} "
            f"| {ARCHIVED[d]['spread_log']:.1f}x / {ARCHIVED[d]['spread_ixi']:.1f}x |"
        )
    return lines


def _levels_table(tables: ProfileTables) -> list[str]:
    """Markdown: levels per sigma_B octave, one row per schedule.

    The columns are the blur octaves, so the "no N4" counterparts of the four MRI schedules
    are extra rows: they are the arrays T1.3 froze, i.e. what the refit replaced.
    """
    header = "| schedule | " + " | ".join(f"{b} px" for b in SIGMA_B_OCTAVE_LABELS) + " | sum |"
    lines = [header, "|---|" + "---|" * (len(SIGMA_B_OCTAVE_LABELS) + 1)]
    rows: list[tuple[str, np.ndarray]] = [
        (f"`{name}`" + ("" if name != "oasis1_W2" else " (not frozen)"), tables.schedules[name])
        for name in (*REPORT_SCHEDULES, "oasis1_W2")
    ]
    if tables.sensitivity is not None:
        rows.extend(
            (f"`{name}` **no N4**", array)
            for name, array in sorted(tables.sensitivity.schedules.items())
        )
    for label, array in rows:
        counts = levels_per_octave(array)
        lines.append(
            f"| {label} | "
            + " | ".join(str(counts[b]) for b in SIGMA_B_OCTAVE_LABELS)
            + f" | {sum(counts.values())} |"
        )
    return lines


def _checklist_table(
    items: list[ChecklistItem], items_no_n4: list[ChecklistItem] | None
) -> list[str]:
    """Markdown: the five known-results items on the N4 data, with the uncorrected outcome beside.

    Parameters
    ----------
    items : list[ChecklistItem]
        The checklist evaluated on the corrected data; this is the claim.
    items_no_n4 : list[ChecklistItem] | None
        The same checklist on the archived uncorrected data, or ``None``.

    Returns
    -------
    list[str]
        The markdown table.
    """
    alt = {item.key: item for item in (items_no_n4 or [])}
    lines = [
        "| # | expected | outcome (N4) | outcome (no N4) | measured on the N4 data | "
        "measured without N4 |",
        "|---|---|---|---|---|---|",
    ]
    for item in items:
        other = alt.get(item.key)
        lines.append(
            f"| ({item.key}) | {item.statement} | **{item.verdict}** "
            f"| {NO_VALUE if other is None else other.verdict} | {item.detail} "
            f"| {NO_VALUE if other is None else other.detail} |"
        )
    return lines


def _checklist_for(
    profiles: dict[str, DatasetProfile],
    spread: dict[str, dict[str, float]],
    deviation: dict[str, tuple[float, float]],
) -> list[ChecklistItem]:
    """Build the five-item checklist from one branch's measured tables."""
    return build_checklist(
        alphas={d: profiles[d].alpha for d in DATASETS},
        coarse_shares={d: profiles[d].shares["0.5-1"] for d in DATASETS},
        spread_log={d: spread[d]["log"] for d in DATASETS},
        spread_ixi={d: spread[d]["ixi"] for d in DATASETS},
        deviation_max={d: deviation[d][0] for d in DATASETS},
    )


def _checklist_notes(tables: ProfileTables, items: list[ChecklistItem]) -> list[str]:
    """Markdown: what the measured numbers say beyond the verdicts, without tuning anything.

    Only the facts the tables above already contain: which mode carries the coarse bin, what
    N4 did to it, how the MRI-versus-photograph ordering stands, and by how much the
    processing moved each quantity. No threshold is changed here; the orchestrator decides.
    """
    coarse = {d: tables.train[d].shares["0.5-1"] for d in DATASETS}
    ratio_photo = min(coarse[d] for d in PHOTO_DATASETS) / max(coarse[d] for d in MRI_DATASETS)
    log_margin = min(tables.spread[d]["log"] for d in MRI_DATASETS) / max(
        tables.spread[d]["log"] for d in PHOTO_DATASETS
    )
    _, ixi_mode10 = coarse_bin_breakdown(tables.powers["ixi"])
    lines = [
        "**Notes on the measured numbers** (facts from the tables above; no threshold was "
        "moved, no N4 parameter was chosen by looking at a spectral number, and no schedule "
        "was refitted to make an item pass).",
        "",
    ]
    if tables.sensitivity is not None:
        alt = {d: tables.sensitivity.train[d].shares["0.5-1"] for d in MRI_DATASETS}
        _, alt_mode10 = coarse_bin_breakdown(tables.sensitivity.powers["ixi"])
        lines.append(
            f"- **N4 moved the coarse bin of the MRI sets** from {alt['ixi']:.2%} to "
            f"{coarse['ixi']:.2%} on IXI and from {alt['oasis1']:.2%} to "
            f"{coarse['oasis1']:.2%} on OASIS-1, and the share of that bin carried by the "
            f"single anterior–posterior ramp mode $(1,0)$ from {alt_mode10:.0%} to "
            f"{ixi_mode10:.0%} on IXI. The photograph sets are byte-identical in both "
            "branches: N4 corrects an MRI acquisition artefact and nothing was applied to "
            "them."
        )
    else:
        lines.append(
            f"- The coarse bin of IXI is {coarse['ixi']:.2%}, of which {ixi_mode10:.0%} sits "
            "in the single anterior–posterior ramp mode $(1,0)$. The uncorrected sensitivity "
            "copies were not found on this machine, so the `no N4` columns are empty."
        )
    lines.extend(
        [
            f"- Registration raised the coarse-octave variance of both MRI sets well above "
            f"the archived pre-registration values ({ARCHIVED['ixi']['coarse_share']:.1%} → "
            f"{coarse['ixi']:.2%} on IXI, {ARCHIVED['oasis1']['coarse_share']:.1%} → "
            f"{coarse['oasis1']:.2%} on OASIS-1) and left the photograph sets unchanged. The "
            f"ordering the design rests on is intact: the photograph sets hold "
            f"{ratio_photo:.0f}× the MRI sets' coarse share.",
            f"- The log-schedule spread is far below the archived values on both MRI sets "
            f"(IXI {ARCHIVED['ixi']['spread_log']:.0f}× → {tables.spread['ixi']['log']:.1f}×, "
            f"OASIS-1 {ARCHIVED['oasis1']['spread_log']:.0f}× → "
            f"{tables.spread['oasis1']['log']:.1f}×) while the photograph sets moved little. "
            f"Item (iii) therefore holds by a margin of {log_margin:.2f}× instead of the "
            "archived ~6×, and the design's contrast between the MRI and photograph arms is "
            "correspondingly weaker on the processed data than the proposal's numbers "
            "suggest.",
        ]
    )
    failing = [item for item in items if not item.passed]
    if failing:
        lines.append(
            "- Failing item(s): "
            + "; ".join(f"({item.key}) {item.detail}" for item in failing)
            + "."
        )
    return lines


def render_report(
    tables: ProfileTables,
    items: list[ChecklistItem],
    items_no_n4: list[ChecklistItem] | None,
    sha: str,
) -> str:
    """Assemble ``docs/RESULTS/data_profile.md``.

    Parameters
    ----------
    tables : ProfileTables
        The measured tables.
    items : list[ChecklistItem]
        The known-results checklist on the N4-corrected data; this is the claim.
    items_no_n4 : list[ChecklistItem] | None
        The same checklist on the archived uncorrected data, printed beside it.
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
        f"{datetime.now(UTC).date().isoformat()}. Tickets T1.3 and T1.4; contracts "
        "`04-run-artifacts.md` §1 and `05-metrics.md` §1.",
        "",
        f"Every curve and every table below is measured on the **training** split of each "
        f"dataset ({n_train} images of $192^2$, values in $[0, 1]$); the `ref` split "
        "(800 images) is profiled beside it as a consistency check. The per-mode variance is "
        "mean-centred across images, in the orthonormal DCT-II basis, with the DC mode "
        "excluded; a mode of radial index $n$ carries $n/2$ cycles per image and modes above "
        "96 cycles per image are excluded from the octave shares.",
        "",
        "The MRI datasets are **rigidly registered and N4 bias-field corrected** (decision "
        "D15, ticket T1.4): N4 runs on each registered volume, with SimpleITK's default "
        "parameters and the dilated template brain mask, before the foreground-p99 intensity "
        "scaling. The **no N4** columns and rows are the same quantities measured on the "
        "archived uncorrected datasets under `$IHDM_DATA_ROOT/_sensitivity/`, which hold the "
        "identical 400 subjects, splits and slices. They are the sensitivity row the reviewer "
        "asked for: the claim is stated on the corrected data, the uncorrected numbers are "
        "printed beside it, and the crossover table is recomputed rather than assumed. N4 is "
        "applied to the MRI side only, so the photograph numbers are identical in both "
        "branches and their `no N4` cells are left empty.",
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
        "### 2.1 The coarse bin per acquisition site",
        "",
        "IXI mixes three sites and two field strengths (Guys 1.5 T, HH 3 T, IOP 1.5 T); "
        "OASIS-1 is a single 1.5 T scanner. Each row is measured on that site's training "
        "images alone, so `coarse share` is the site's own between-image variance in the "
        "0.5–1 c/img bin, not its contribution to the pooled one. This is the table decision "
        "D15 asks for: if the coarse bin were a multi-site artefact it would be small within "
        "a site and large across sites, and N4 should shrink the $(1,0)$ share everywhere.",
        "",
        *_site_table(tables),
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
        "this table and for item (v) of the checklist; no arm uses it, so it is not frozen. The "
        "`no N4` rows are the same fits on the uncorrected training splits, i.e. the arrays "
        "T1.3 froze and this refit replaced.",
        "",
        *_levels_table(tables),
        "",
        "## 6. Known-results checklist",
        "",
        "The claim is the `N4` column. The `no N4` column is the same five items evaluated on "
        "the archived uncorrected datasets, each scored against schedules refitted on those "
        "same uncorrected training splits.",
        "",
        *_checklist_table(items, items_no_n4),
        "",
        f"**{len(items) - len(failed)} of {len(items)} items pass on the N4 data.**"
        + (f" Failing: {', '.join(failed)}." if failed else "")
        + (
            ""
            if items_no_n4 is None
            else f" Without N4: {sum(i.passed for i in items_no_n4)} of {len(items_no_n4)}"
                 + (
                     ""
                     if all(i.passed for i in items_no_n4)
                     else ", failing "
                          + ", ".join(i.key for i in items_no_n4 if not i.passed)
                 )
                 + "."
        ),
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
        *(
            []
            if tables.sensitivity is None
            else [
                f"| {DATASET_LABELS[SENSITIVITY_IDS[d]]} | train "
                f"| {tables.sensitivity.train[d].n_images} "
                f"| `{tables.sensitivity.train[d].images_sha256[:16]}…` |"
                for d in MRI_DATASETS
            ]
        ),
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
    if tables.sensitivity is None:
        return
    for d in MRI_DATASETS:
        power = tables.sensitivity.powers[d]
        freq, profile_train = radial_spectrum(power)
        np.savez_compressed(
            out_dir / f"{SENSITIVITY_IDS[d]}.npz",
            power_train=power,
            cycles_per_image=freq,
            radial_train=profile_train,
            octave_labels=np.array(OCTAVE_LABELS),
            octave_shares_train=np.array(
                [tables.sensitivity.train[d].shares[b] for b in OCTAVE_LABELS]
            ),
            alpha_train=tables.sensitivity.train[d].alpha,
            inherited_train=np.array(
                [tables.sensitivity.train[d].inherited[k] for k in ("W/8", "W/4", "W/2")]
            ),
            spread_log=tables.sensitivity.spread[d]["log"],
            spread_ixi=tables.sensitivity.spread[d]["ixi"],
            spread_own=tables.sensitivity.spread[d]["own"],
            schedule_W2=tables.sensitivity.schedules[f"{d}_W2"],
            schedule_W8=tables.sensitivity.schedules[f"{d}_W8"],
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
    items = _checklist_for(tables.train, tables.spread, tables.deviation)
    items_no_n4 = (
        None
        if tables.sensitivity is None
        else _checklist_for(
            tables.sensitivity.train, tables.sensitivity.spread, tables.sensitivity.deviation
        )
    )
    out: Path = args.out
    report = render_report(tables, items, items_no_n4, git_sha(repo_root()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_profile.md").write_text(report)
    _write_npz(tables, out / "data_profile")
    _figure(tables, args.data_root, out / "fig_data")

    print(f"\nimages per curve: {tables.train['ixi'].n_images}")
    for d in DATASETS:
        alt = "" if tables.sensitivity is None or d not in MRI_DATASETS else (
            f"   (no N4 {tables.sensitivity.train[d].alpha:.2f})"
        )
        print(f"  alpha {DATASET_LABELS[d]:16s} {tables.train[d].alpha:.2f}{alt}")
    print()
    for dataset_id in MRI_DATASETS:
        for entry in tables.sites.get(dataset_id, []):
            print(
                f"  site {dataset_id:7s} {entry.site:5s} n={entry.n_subjects:3d} "
                f"coarse {entry.coarse_share:.2%} (1,0) {entry.mode_10_share:.0%}"
            )
    print()
    for item in items:
        print(f"  ({item.key}) {item.verdict}: {item.detail}")
    if items_no_n4 is not None:
        print("\n  no N4: " + ", ".join(f"({i.key}) {i.verdict}" for i in items_no_n4))
    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(f"\nwrote {out / 'data_profile.md'} in {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
