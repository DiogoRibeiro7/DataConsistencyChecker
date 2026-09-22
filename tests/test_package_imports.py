"""Tests for the installed package import surface."""

from __future__ import annotations

from check_data_consistency import DataConsistencyChecker as LegacyChecker
from data_consistency_checker import DataConsistencyChecker


def test_primary_package_import_exposes_checker() -> None:
    """The package root exposes the main checker class."""
    assert DataConsistencyChecker.__name__ == "DataConsistencyChecker"
    assert DataConsistencyChecker.__module__ == "data_consistency_checker.checker"


def test_legacy_import_reexports_same_checker() -> None:
    """The historical import path remains compatible after installation."""
    assert LegacyChecker is DataConsistencyChecker
