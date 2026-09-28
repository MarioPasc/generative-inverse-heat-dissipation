"""Fixtures of the collection tests: one healthy synthetic campaign, copied per test."""

from __future__ import annotations

import pytest

from tests.analysis.synthetic_eval import Campaign, build_campaign, copy_campaign


@pytest.fixture(scope="session")
def pristine_campaign(tmp_path_factory) -> Campaign:
    """The healthy 30-run campaign, built once per session; never modified."""
    return build_campaign(tmp_path_factory.mktemp("t52") / "campaign")


@pytest.fixture
def campaign(pristine_campaign, tmp_path) -> Campaign:
    """A private copy of the healthy campaign that the test may damage."""
    return copy_campaign(pristine_campaign, tmp_path / "campaign")
