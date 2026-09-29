"""Tests of ``ihdm.analysis.collect`` (T5.2) on a synthetic 30-run campaign of the real schemas.

One test per failure mode of every check: each damages one input of a healthy campaign and
asserts that the check names it and that nothing is published.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ihdm.analysis.collect import (
    SIDECARS,
    CollectError,
    collect,
    discover_gates,
    format_report,
    required_gate_pairs,
    strict_json,
)
from ihdm.metrics.run_eval import evaluated_steps
from tests.analysis.synthetic_eval import (
    GATE_RUNS,
    RUN_11,
    SUFFIX,
    gate_record,
    read_expected,
    repo_cells,
)

RUN = "ixi_A3_s2"


def _collect(campaign, tmp_path, **overrides):
    out = tmp_path / "results"
    return collect(campaign.config(out, **overrides)), out


def _assert_failed(report, out, check, fragment):
    assert report.verdict == "FAIL", format_report(report)
    assert report.checks[check].verdict == "FAIL"
    assert any(fragment in p for p in report.checks[check].problems), report.checks[check].problems
    assert report.written is None and not out.exists()
    assert not list(out.parent.glob(f".{out.name}.partial-*"))


# --------------------------------------------------------------------------------------------
# The healthy campaign
# --------------------------------------------------------------------------------------------


def test_a_healthy_campaign_is_published_complete(campaign, tmp_path):
    report, out = _collect(campaign, tmp_path)
    assert report.verdict == "COMPLETE", format_report(report)
    assert all(c.verdict == "PASS" for c in report.checks.values())
    assert report.written == out.resolve()
    assert sorted(p.name for p in out.iterdir()) == ["collection.json", "gates", "index.csv",
                                                     "runs"]
    assert len(list((out / "runs").iterdir())) == 30
    assert not list(tmp_path.glob(".results.partial-*"))
    info = json.loads((out / "collection.json").read_text())
    assert info["complete"] is True and info["verdict"] == "COMPLETE"
    assert info["n_iters"] == 60000 and info["amp"] == "fp16"
    assert info["evaluated_steps"] == list(evaluated_steps(60000))
    assert [c["verdict"] for c in info["checks"]] == ["PASS"] * 9


def test_each_run_folder_holds_exactly_the_contract_files(campaign, tmp_path):
    _, out = _collect(campaign, tmp_path)
    steps = evaluated_steps(60000)
    want = {"summary.json", "final.json", "manifest.json", "config.json",
            "metrics.canonical.jsonl", "grid_final.png", *SIDECARS,
            *(f"ckpt_{s:06d}.json" for s in steps)}
    for folder in (out / "runs").iterdir():
        assert {p.name for p in folder.iterdir()} == want, folder.name
    assert len(steps) == 12
    # No samples and no raw metrics.jsonl anywhere in the results.
    names = {p.name for p in out.rglob("*")}
    assert "samples.npy" not in names and "metrics.jsonl" not in names


def test_result_files_are_copied_byte_for_byte(campaign, tmp_path):
    _, out = _collect(campaign, tmp_path)
    run = out / "runs" / RUN
    assert (run / "summary.json").read_bytes() == campaign.summary_path(RUN).read_bytes()
    assert (run / "final.json").read_bytes() == (campaign.metrics_dir(RUN) /
                                                 "final.json").read_bytes()
    assert (run / "manifest.json").read_bytes() == (campaign.run_root / RUN /
                                                    "manifest.json").read_bytes()
    assert (run / "grid_final.png").read_bytes() == (campaign.run_root / RUN / "grids" /
                                                     "iter_060000.png").read_bytes()


def test_the_gates_of_both_naming_schemes_are_renamed(campaign, tmp_path):
    report, out = _collect(campaign, tmp_path)
    pairs = ((35000, 40000), (55000, 60000))
    assert sorted(p.name for p in (out / "gates").iterdir()) == sorted(
        f"{r}_gate_{early:06d}_{late:06d}.json" for r in GATE_RUNS for early, late in pairs)
    legacy = out / "gates" / "ixi_A0_s1_gate_035000_040000.json"
    assert legacy.read_bytes() == (campaign.gate_dir / "ixi_A0_s1_amp-fp16_gate.json").read_bytes()
    assert {g["legacy_name"] for g in report.gates} == {True, False}


def test_the_index_has_one_row_per_cell_in_table_order(campaign, tmp_path):
    _, out = _collect(campaign, tmp_path)
    with (out / "index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["run_id"] for r in rows] == [c["run_id"] for c in repo_cells()]
    columns = list(rows[0])
    for column in ("index", "run_id", "dataset", "arm", "seed", "tier", "n_iters", "amp",
                   "checkpoint_steps", "lsd_final", "kid", "kid_ci_low", "kid_ci_high", "fid",
                   "fid_ci_low", "fid_ci_high", "fid_n_reference", "M", "M_lp", "D_pix", "D_lp",
                   "recall", "coverage", "n_skipped", "lsd_005000", "lsd_060000",
                   "gate_035000_040000_extend", "gate_055000_060000_diff"):
        assert column in columns, column
    first = rows[0]
    assert first["checkpoint_steps"].split(";") == [str(s) for s in evaluated_steps(60000)]
    assert first["fid_n_reference"] == "800" and first["amp"] == "fp16"
    assert first["gate_055000_060000_extend"] == "true"
    assert rows[1]["gate_055000_060000_extend"] == ""
    final = json.loads((out / "runs" / first["run_id"] / "final.json").read_text())
    assert float(first["kid"]) == final["inception"]["kid"]
    assert float(first["D_lp"]) == final["diversity_lp"]


def test_run_11_keeps_its_canonical_history_and_six_skips(campaign, tmp_path):
    report, out = _collect(campaign, tmp_path)
    raw_path = campaign.run_root / RUN_11 / "metrics.jsonl"
    raw = raw_path.read_text().splitlines()
    canonical = (out / "runs" / RUN_11 / "metrics.canonical.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["kind"] for line in canonical]
    assert "abort" in raw_path.read_text() and "abort" not in kinds
    assert kinds.count("skip") == 6 and kinds[-1] == "done"
    assert set(canonical) < set(raw)  # lines are copied, not re-serialised
    rows = {r.cell.run_id: r.row for r in report.runs}
    assert rows[RUN_11]["n_skipped"] == 6
    assert all(row["n_skipped"] == 0 for run, row in rows.items() if run != RUN_11)
    info = json.loads((out / "collection.json").read_text())
    recorded = info["inputs"]["runs"][RUN_11]["metrics_jsonl"]
    assert recorded["sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()


def test_the_output_is_never_overwritten(campaign, tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(CollectError, match="exists"):
        _collect(campaign, tmp_path)


# --------------------------------------------------------------------------------------------
# C1 identity and presence
# --------------------------------------------------------------------------------------------


def test_c1_a_missing_run_fails_and_publishes_nothing(campaign, tmp_path):
    campaign.summary_path(RUN).unlink()
    campaign.tar_path(RUN).unlink()
    report, out = _collect(campaign, tmp_path)
    assert report.verdict == "INCOMPLETE"
    assert report.checks["C1"].verdict == "MISSING" and report.checks["C8"].verdict == "MISSING"
    assert report.written is None and not out.exists()


def test_c1_allow_missing_publishes_a_partial_folder(campaign, tmp_path):
    campaign.summary_path(RUN).unlink()
    campaign.tar_path(RUN).unlink()
    report, out = _collect(campaign, tmp_path, allow_missing=True)
    assert report.verdict == "INCOMPLETE" and report.written == out.resolve()
    info = json.loads((out / "collection.json").read_text())
    assert info["complete"] is False and info["verdict"] == "INCOMPLETE"
    assert any(RUN in m for m in info["checks"][0]["missing"])
    assert not (out / "runs" / RUN / "summary.json").exists()
    assert (out / "runs" / RUN / "metrics.canonical.jsonl").exists()


def test_c1_a_missing_run_directory_is_missing(campaign, tmp_path):
    shutil.rmtree(campaign.run_root / RUN)
    report, _ = _collect(campaign, tmp_path)
    assert any("no run directory" in m for m in report.checks["C1"].missing)


def test_c1_a_summary_of_another_cell_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "summary.json", lambda s: s["run"].update(seed=3))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C1", "summary run.seed 3 != cells.csv 2")


def test_c1_a_manifest_of_another_cell_fails(campaign, tmp_path):
    path = campaign.run_root / RUN / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["arm"] = "A0"
    path.write_text(json.dumps(manifest))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C1", "manifest arm 'A0' != cells.csv 'A3'")


def test_c1_a_summary_for_a_run_that_is_not_a_cell_fails(campaign, tmp_path):
    shutil.copyfile(campaign.summary_path(RUN),
                    campaign.eval_dir / f"ixi_A9_s1{SUFFIX}_summary.json")
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C1", "not a cell")


# --------------------------------------------------------------------------------------------
# C2 steps
# --------------------------------------------------------------------------------------------


def test_c2_a_40k_summary_fails_at_60k(campaign, tmp_path):
    steps = list(evaluated_steps(40000))

    def forty(summary):
        summary.update(checkpoint_steps=steps, final_step=40000,
                       lsd_by_step={str(s): 0.3 for s in steps})

    campaign.edit_json(RUN, "summary.json", forty)
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C2", "!= evaluated_steps(60000) (12 steps)")
    assert any("final_step 40000" in p for p in report.checks["C2"].problems)


def test_c2_a_missing_checkpoint_record_fails(campaign, tmp_path):
    (campaign.metrics_dir(RUN) / "ckpt_045000.json").unlink()
    campaign.repack(RUN)
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C2", "the tar holds no ckpt_045000.json")


def test_c2_a_checkpoint_record_of_the_wrong_step_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "ckpt_010000.json", lambda r: r.update(step=15000))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C2", "ckpt_010000.json holds step 15000")


def test_c2_a_manifest_of_another_length_fails(campaign, tmp_path):
    path = campaign.run_root / RUN / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["n_iters"] = 40000
    path.write_text(json.dumps(manifest))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C2", "manifest n_iters 40000 != 60000")


# --------------------------------------------------------------------------------------------
# C3 precision
# --------------------------------------------------------------------------------------------


def test_c3_a_run_sampled_in_another_precision_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "summary.json", lambda s: s["sampling"].update(amp="bf16"))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C3", "sampling.amp 'bf16' != --amp 'fp16'")


def test_c3_a_result_file_of_another_precision_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "final.json", lambda r: r.update(amp="off"))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C3", "final.json amp 'off'")


def test_c3_collecting_another_precision_finds_no_run(campaign, tmp_path):
    report, _ = _collect(campaign, tmp_path, amp="off")
    assert report.verdict == "INCOMPLETE"
    assert len(report.checks["C1"].missing) == 30 and not report.gates


# --------------------------------------------------------------------------------------------
# C4 seed lists
# --------------------------------------------------------------------------------------------


def test_c4_a_foreign_seed_list_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "summary.json",
                       lambda s: s["seed_lists"].update(intermediate_sha256="0" * 64))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C4", "seed-list digests")


def test_c4_a_checkpoint_on_another_list_fails(campaign, tmp_path):
    other = read_expected()["oasis1"][0]
    campaign.edit_json(RUN, "ckpt_020000.json", lambda r: r.update(seed_list_sha256=other))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C4", "ckpt_020000.json seed_list_sha256")


# --------------------------------------------------------------------------------------------
# C5 hashes
# --------------------------------------------------------------------------------------------


def test_c5_a_config_hash_that_disagrees_with_the_manifest_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "summary.json", lambda s: s["run"].update(config_sha256="d" * 64))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C5", "config_sha256 != manifest")


def test_c5_a_recipe_hash_that_disagrees_with_the_config_fails(campaign, tmp_path):
    path = campaign.run_root / RUN / "config.json"
    config = json.loads(path.read_text())
    config["optim"]["grad_clip"] = 0.5
    path.write_text(json.dumps(config))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C5", "recipe_sha256")


# --------------------------------------------------------------------------------------------
# C6 strict JSON and keys
# --------------------------------------------------------------------------------------------


def test_c6_a_nan_token_fails(campaign, tmp_path):
    path = campaign.metrics_dir(RUN) / "final.json"
    path.write_text(path.read_text().replace('"M": ', '"M": NaN, "M_was": ', 1))
    campaign.repack(RUN)
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C6", "final.json is not strict JSON")


def test_c6_a_nan_token_in_the_summary_fails(campaign, tmp_path):
    path = campaign.metrics_dir(RUN) / "summary.json"
    path.write_text(path.read_text().replace('"t_tau": null', '"t_tau": Infinity', 1))
    campaign.repack(RUN)
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C6", "summary.json is not strict JSON")


@pytest.mark.parametrize("name, key, fragment", [
    ("final.json", "M_lp", "final.json misses keys ['M_lp']"),
    ("final.json", "diversity_pix", "final.json misses keys ['diversity_pix']"),
    ("ckpt_060000.json", "lsd_octaves", "ckpt_060000.json misses keys ['lsd_octaves']"),
    ("summary.json", "lsd_by_step", "summary.json misses keys ['lsd_by_step']"),
])
def test_c6_a_missing_section_9_key_fails(campaign, tmp_path, name, key, fragment):
    campaign.edit_json(RUN, name, lambda r: r.pop(key))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C6", fragment)


def test_c6_a_final_without_inception_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "final.json", lambda r: r.update(inception=None))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C6", "no Inception block")


def test_c6_an_inception_block_without_its_ci_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "final.json", lambda r: r["inception"].pop("kid_ci_high"))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C6", "inception misses keys ['kid_ci_high']")


# --------------------------------------------------------------------------------------------
# C7 the training history
# --------------------------------------------------------------------------------------------


def _edit_history(campaign, run_id, edit):
    path = campaign.run_root / run_id / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(r) + "\n" for r in edit(records)))


def test_c7_an_abort_not_abandoned_by_a_resume_fails(campaign, tmp_path):
    abort = {"step": 50000, "kind": "abort", "reason": "x", "loss": "nan", "n_skipped": 10,
             "consecutive": 10, "abort_state": None}
    _edit_history(campaign, RUN, lambda rs: [*rs[:-1], abort, rs[-1]])
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C7", "abort in the canonical history at steps [50000]")


def test_c7_a_history_that_stops_at_40k_fails(campaign, tmp_path):
    _edit_history(campaign, RUN, lambda rs: [r for r in rs if r["step"] <= 40000])
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C7", "not done at 60000")


def test_c7_a_nan_token_in_the_history_fails(campaign, tmp_path):
    path = campaign.run_root / RUN / "metrics.jsonl"
    path.write_text(path.read_text().replace('"loss": 0.3,', '"loss": NaN,', 1))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C6", "non-strict JSON token NaN")


# --------------------------------------------------------------------------------------------
# C8 archive
# --------------------------------------------------------------------------------------------


def test_c8_a_plain_summary_that_differs_from_the_tar_fails(campaign, tmp_path):
    campaign.edit_json(RUN, "summary.json", lambda s: s.update(git_sha="stale"),
                       plain_summary=False)
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C8", "differs from the plain summary")


def test_c8_a_missing_sidecar_fails(campaign, tmp_path):
    (campaign.metrics_dir(RUN) / "final_pca_components.npy").unlink()
    campaign.repack(RUN)
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C8", "the tar holds no final_pca_components.npy")


def test_c8_an_unreadable_tar_fails(campaign, tmp_path):
    campaign.tar_path(RUN).write_bytes(b"not a tar")
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C8", "cannot read")


def test_c8_a_missing_final_grid_fails(campaign, tmp_path):
    (campaign.run_root / RUN / "grids" / "iter_060000.png").unlink()
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C8", "no final grid")


# --------------------------------------------------------------------------------------------
# C9 gates
# --------------------------------------------------------------------------------------------


def test_c9_a_missing_required_gate_is_missing(campaign, tmp_path):
    (campaign.gate_dir / f"lsun_church_A0_s1{SUFFIX}_gate_055000_060000.json").unlink()
    report, _ = _collect(campaign, tmp_path)
    assert report.verdict == "INCOMPLETE"
    assert report.checks["C9"].missing == ["lsun_church_A0_s1: no gate 055000_060000 in "
                                           f"{campaign.gate_dir}"]


def test_c9_a_gate_whose_content_contradicts_its_name_fails(campaign, tmp_path):
    path = campaign.gate_dir / f"ixi_A0_s1{SUFFIX}_gate_055000_060000.json"
    path.write_text(json.dumps(gate_record((50000, 55000), read_expected()["ixi"][0])))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C9", "holds steps (50000, 55000), its name says (55000, 60000)")


def test_c9_a_gate_of_another_precision_inside_ours_fails(campaign, tmp_path):
    path = campaign.gate_dir / f"ixi_A0_s1{SUFFIX}_gate.json"
    path.write_text(json.dumps(gate_record((35000, 40000), read_expected()["ixi"][0], "off")))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C9", "amp 'off' != --amp 'fp16'")


def test_c9_a_legacy_and_a_pair_named_gate_of_the_same_pair_fail(campaign, tmp_path):
    shutil.copyfile(campaign.gate_dir / f"ixi_A0_s1{SUFFIX}_gate.json",
                    campaign.gate_dir / f"ixi_A0_s1{SUFFIX}_gate_035000_040000.json")
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C9", "same pair as")


def test_c9_a_gate_on_a_foreign_seed_list_fails(campaign, tmp_path):
    path = campaign.gate_dir / f"lsun_church_A0_s1{SUFFIX}_gate.json"
    path.write_text(json.dumps(gate_record((35000, 40000), read_expected()["ixi"][0])))
    report, out = _collect(campaign, tmp_path)
    _assert_failed(report, out, "C9", "seed_list_sha256")


def test_c9_gates_of_another_precision_are_ignored(campaign, tmp_path):
    other = gate_record((55000, 60000), read_expected()["ixi"][0], "bf16")
    (campaign.gate_dir / "ixi_A0_s1_amp-bf16_gate_055000_060000.json").write_text(
        json.dumps(other))
    (campaign.gate_dir / "notes.json").write_text("{}")
    report, _ = _collect(campaign, tmp_path)
    assert report.verdict == "COMPLETE", format_report(report)
    assert report.ignored == ["ixi_A0_s1_amp-bf16_gate_055000_060000.json", "notes.json"]


# --------------------------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------------------------


def test_strict_json_refuses_the_tokens_json_loads_accepts():
    assert json.loads('{"a": NaN}')["a"] != 0  # the stdlib accepts it
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="non-strict"):
            strict_json(f'{{"a": {token}}}')
    assert strict_json(b'{"a": "nan"}') == {"a": "nan"}


def test_required_gate_pairs_follow_the_run_length():
    assert required_gate_pairs(60000) == ((35000, 40000), (55000, 60000))
    assert required_gate_pairs(40000) == ((35000, 40000),)


def test_discover_gates_reads_both_names(tmp_path):
    for name in ("ixi_A0_s1_amp-fp16_gate.json", "ixi_A0_s1_amp-fp16_gate_055000_060000.json",
                 "ixi_A0_s1_gate_035000_040000.json", "ixi_A0_s1_amp-fp16_gate.tar"):
        (tmp_path / name).write_text("{}")
    gates, ignored = discover_gates(tmp_path, "fp16")
    assert [(g.run_id, g.pair, g.legacy) for g in gates] == [
        ("ixi_A0_s1", (35000, 40000), True), ("ixi_A0_s1", (55000, 60000), False)]
    assert gates[0].dest_name == "ixi_A0_s1_gate_035000_040000.json"
    assert ignored == ["ixi_A0_s1_gate_035000_040000.json"]
    gates, _ = discover_gates(tmp_path, "off")
    assert [(g.run_id, g.pair) for g in gates] == [("ixi_A0_s1", (35000, 40000))]
    assert discover_gates(tmp_path / "absent", "fp16") == ([], [])


def test_the_report_names_every_check_and_the_verdict(campaign, tmp_path):
    report, _ = _collect(campaign, tmp_path)
    text = format_report(report)
    for check in ("C1", "C5", "C9"):
        assert f"  {check} PASS" in text
    assert "VERDICT: COMPLETE (0 problems, 0 missing)" in text
    assert "ixi_A0_s1_gate_055000_060000.json" in text


def test_the_healthy_fixture_is_the_repository_table(pristine_campaign):
    assert Path(pristine_campaign.cells).read_text() == (
        Path(__file__).resolve().parents[2] / "slurm" / "array" / "cells.csv").read_text()
