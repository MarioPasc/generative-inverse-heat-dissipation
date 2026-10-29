"""Integration test: the tutorial notebook executes end to end on CPU from tracked files only.

Invariants: the committed ``.ipynb`` is the jupytext rendering of ``ihdm_mri_tutorial.py``; it runs
without error in under 60 s; its source never reads the data drive; and the printed numbers that
the prose quotes reproduce ``docs/RESULTS/data_profile.md`` and the ticket's spreads.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ihdm.paths import repo_root

nbformat = pytest.importorskip("nbformat")
jupytext = pytest.importorskip("jupytext")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK = repo_root() / "ihdm_mri_tutorial.ipynb"
SOURCE = repo_root() / "ihdm_mri_tutorial.py"
TIME_LIMIT_S = 60.0

# Strings that a correct run must print (data_profile.md §2, §3, §5; spreads of T9.1 §7.3).
EXPECTED_OUTPUT = (
    "IXI                4.1%   11.3%   11.0%   16.8%   29.8%   18.3%    7.8%    1.0%",
    "Churches          23.8%   21.3%   16.1%   12.2%    9.6%    7.5%    6.1%    3.4%",
    "synthetic c^-2    23.7%   15.2%   12.2%   11.2%   10.7%   10.5%   10.4%    6.1%",
    "IXI       I(24 px) =  8.5%",
    "I(96 px) =  0.3%",
    "Churches  I(24 px) = 29.1%",
    "I(96 px) =  1.8%",
    "log_W2                       27     26     26     26     27     26     26     16    200",
    "ixi_W2                       19     30     38     35     26     23     20      9    200",
    "log_W8                       36     36     35     36     36     21      0      0    200",
    "ixi_W8                       24     37     46     43     32     18      0      0    200",
    "IXI, log_W2: spread max E_k / min E_k = 8.9x",
    "IXI, ixi_W2: spread max E_k / min E_k = 1.0x",
    "IXI, log_W8: spread max E_k / min E_k = 8.5x",
    "IXI, ixi_W8: spread max E_k / min E_k = 1.0x",
    "Churches        log_W2 spread = 3.8x",
    "synthetic c^-2  log_W2 spread = 2.0x",
)


def _stdout(notebook: object) -> str:
    """Concatenate every stream output of an executed notebook."""
    return "".join(
        output.get("text", "")
        for cell in notebook.cells  # type: ignore[attr-defined]
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_source_never_reads_the_data_drive() -> None:
    text = SOURCE.read_text()
    assert "/media/" not in text
    assert "data_root" not in text


def test_committed_notebook_matches_its_source() -> None:
    committed = nbformat.read(NOTEBOOK, as_version=4)
    paired = jupytext.read(SOURCE)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in paired.cells
    ]


@pytest.mark.integration
def test_notebook_executes_on_cpu_and_reproduces_the_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    notebook = jupytext.read(SOURCE)
    started = time.perf_counter()
    nbclient.NotebookClient(
        notebook, timeout=120, kernel_name="python3",
        resources={"metadata": {"path": str(repo_root())}},
    ).execute()
    elapsed = time.perf_counter() - started
    assert elapsed < TIME_LIMIT_S, f"notebook took {elapsed:.1f} s"

    stdout = _stdout(notebook)
    for expected in EXPECTED_OUTPUT:
        assert expected in stdout, expected
    for line in stdout.splitlines():
        if "max |heat - DCTBlur|" in line:
            assert float(line.rsplit("=", 1)[1]) < 1e-5, line
    images = sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    )
    assert images == 11  # ten figures and the full Figure 2
