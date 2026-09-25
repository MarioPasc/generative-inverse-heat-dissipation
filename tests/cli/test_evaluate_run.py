"""Tests of ``ihdm.cli.evaluate_run``: argument parsing and the end-to-end run of ``main``.

The run fixture is built with the helpers of ``tests/metrics/test_run_eval.py`` (imported as a
namespace-package module, which ``pythonpath = ["."]`` makes importable) so that the two suites
agree on what a tiny run looks like; see that file for why it is 96² and 14 subjects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ihdm.cli.evaluate_run import build_parser, main, request_from_args
from ihdm.metrics.errors import MetricError
from ihdm.metrics.io import read_json
from ihdm.metrics.run_eval import DEFAULT_SAMPLE_BATCH, N_SEEDS_FINAL, N_SEEDS_INTERMEDIATE
from tests.metrics.test_run_eval import _build_dataset, _config, _write_run


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory) -> Path:
    """A run directory with two EMA checkpoints, its dataset beside it."""
    base = tmp_path_factory.mktemp("t43cli")
    dataset = _build_dataset(base / "data")
    config = _config(dataset)
    return _write_run(base / "runs" / config.run_id, config)


def _parse(argv: list[str]):
    """Parse ``argv`` and return the evaluation request it describes."""
    return request_from_args(build_parser().parse_args(argv))


# --------------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------------


def test_the_defaults_are_the_d16_budget():
    """500 intermediate seeds, 2 000 final, 40 x 50 held out, sampling batch 32."""
    request = _parse(["--run", "/tmp/run"])
    assert request.run == Path("/tmp/run")
    assert request.ckpts == "all"
    assert request.n_lsd == N_SEEDS_INTERMEDIATE
    assert request.n_final == N_SEEDS_FINAL
    assert request.n_seeds == 40
    assert request.n_per_seed == 50
    assert request.sample_batch == DEFAULT_SAMPLE_BATCH
    assert request.n_boot_inception == 200
    assert request.n_boot_gate == 1000
    assert request.k == 5
    assert request.gate is None
    assert request.a0_final_lsd is None
    assert request.skip_inception is False
    assert request.force is False
    assert request.device is None


def test_run_is_required():
    """``--run`` has no default; the parser exits rather than guessing one."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize("spec", ["all", "final", "5000", "35000,40000", "250,500,750"])
def test_ckpts_forms_are_carried_through_verbatim(spec):
    """``--ckpts`` is resolved against the run, not by the parser, so it arrives unchanged."""
    assert _parse(["--run", "/tmp/run", "--ckpts", spec]).ckpts == spec


def test_the_counts_and_the_flags_reach_the_request():
    """Every knob of the ticket's command line maps onto a field of the request."""
    request = _parse(
        [
            "--run", "/tmp/run", "--ckpts", "250,750",
            "--n-lsd", "32", "--n-final", "64", "--n-seeds", "4", "--n-per-seed", "10",
            "--sample-batch", "20", "--fid-batch", "32", "--fid-boot", "50",
            "--gate-boot", "300", "--k", "3", "--a0-final-lsd", "0.25",
            "--skip-inception", "--force", "--device", "cuda",
        ]
    )
    assert (request.n_lsd, request.n_final) == (32, 64)
    assert (request.n_seeds, request.n_per_seed) == (4, 10)
    assert (request.sample_batch, request.fid_batch) == (20, 32)
    assert (request.n_boot_inception, request.n_boot_gate, request.k) == (50, 300, 3)
    assert request.a0_final_lsd == 0.25
    assert request.skip_inception is True
    assert request.force is True
    assert request.device == "cuda"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("35000,40000", (35000, 40000)), (" 250 , 750 ", (250, 750)), ("750,250", (750, 250))],
)
def test_gate_is_parsed_as_an_ordered_pair(value, expected):
    """``--gate`` keeps the order given: the earlier checkpoint is ``a``."""
    assert _parse(["--run", "/tmp/run", "--gate", value]).gate == expected


@pytest.mark.parametrize("value", ["35000", "1,2,3", "a,b"])
def test_gate_rejects_anything_that_is_not_two_integers(value):
    """A malformed gate is a :class:`MetricError`, not a traceback from inside the evaluation."""
    with pytest.raises(MetricError, match="--gate"):
        _parse(["--run", "/tmp/run", "--gate", value])


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def test_main_runs_end_to_end_and_prints_the_summary_line(cli_run, capsys):
    """``main`` evaluates the fixture run and writes the result files of ``05`` §9."""
    code = main(
        [
            "--run", str(cli_run), "--ckpts", "250,750",
            "--n-lsd", "8", "--n-final", "8", "--n-seeds", "2", "--n-per-seed", "3",
            "--sample-batch", "8", "--skip-inception", "--device", "cpu",
            "--gate", "250,750", "--gate-boot", "100",
        ]
    )
    assert code == 0

    printed = capsys.readouterr().out
    assert printed.startswith("OK ")
    assert "GATE [250, 750]" in printed
    assert "extend=" in printed
    assert "inception weights path" not in printed

    summary = read_json(cli_run / "metrics" / "summary.json")
    assert sorted(int(step) for step in summary["lsd_by_step"]) == [250, 750]
    assert (cli_run / "metrics" / "gate.json").exists()
    assert (cli_run / "metrics" / "final.json").exists()


def test_main_prints_the_inception_weights_path_before_any_work(monkeypatch, capsys):
    """T5.1 pre-seeds the weights on the compute nodes and needs the path the CLI will read."""
    import ihdm.cli.evaluate_run as module

    monkeypatch.setattr(
        module,
        "evaluate_run",
        lambda request: {
            "run": {"run_id": "r", "dataset": "d", "arm": "A0", "seed": 1},
            "checkpoint_steps": [750],
            "lsd_by_step": {750: 0.5},
            "t_tau": None,
            "final": {"lsd": 0.5, "M": 1.0, "diversity_pix": 1e-3, "inception": None},
        },
    )
    assert main(["--run", "/tmp/run"]) == 0
    printed = capsys.readouterr().out
    assert "inception weights path:" in printed
    assert "inception-2015-12-05.pt" in printed


def test_main_stays_quiet_about_the_weights_when_inception_is_skipped(monkeypatch, capsys):
    """``--skip-inception`` must not touch clean-fid at all."""
    import ihdm.cli.evaluate_run as module

    monkeypatch.setattr(
        module,
        "evaluate_run",
        lambda request: {
            "run": {"run_id": "r", "dataset": "d", "arm": "A0", "seed": 1},
            "checkpoint_steps": [750],
            "lsd_by_step": {750: 0.5},
            "t_tau": None,
            "final": {},
        },
    )
    assert main(["--run", "/tmp/run", "--skip-inception"]) == 0
    assert "inception weights path:" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# --amp (T5.1)
# --------------------------------------------------------------------------------------------


def test_amp_defaults_to_off():
    """The D16 contract samples in full precision unless ``--amp`` says otherwise."""
    assert _parse(["--run", "/tmp/run"]).amp == "off"


@pytest.mark.parametrize("mode", ["off", "fp16", "bf16"])
def test_amp_modes_reach_the_request(mode):
    """Each of the three modes is carried to the request verbatim."""
    assert _parse(["--run", "/tmp/run", "--amp", mode]).amp == mode


@pytest.mark.parametrize("mode", ["on", "fp32", "true", ""])
def test_amp_rejects_an_unknown_mode(mode, capsys):
    """argparse refuses anything outside the three modes."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--run", "/tmp/run", "--amp", mode])
    assert "invalid choice" in capsys.readouterr().err
