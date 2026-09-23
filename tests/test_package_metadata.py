"""Tests for package metadata exposed at runtime."""

from __future__ import annotations

from importlib.metadata import version

import data_consistency_checker


def test_runtime_version_matches_installed_metadata() -> None:
    """The public version comes from the installed distribution metadata."""
    assert data_consistency_checker.__version__ == version("data-consistency-checker")
