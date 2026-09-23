"""Tests for the structured report API."""

from __future__ import annotations

import json

import pandas as pd

from data_consistency_checker import DataConsistencyChecker, DataConsistencyReport


def _checker() -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(
        pd.DataFrame(
            {
                "value": list(range(12)),
                "group": ["a", "b"] * 6,
            }
        )
    )
    return checker


def test_get_report_returns_public_report_type() -> None:
    """A completed analysis is exposed through the package report model."""
    checker = _checker()
    checker.check_data_quality(execute_list=["MISSING_VALUES"])

    report = checker.get_report()

    assert isinstance(report, DataConsistencyReport)
    assert report.n_rows == 12
    assert report.n_columns == 2
    assert report.executed_tests == ("MISSING_VALUES",)
    assert len(report.patterns) == 2
    assert report.exceptions == ()
    assert len(report.row_scores) == 12


def test_report_is_strict_json_serializable() -> None:
    """The report dictionary can be emitted without a custom JSON encoder."""
    checker = _checker()
    checker.check_data_quality(execute_list=["MISSING_VALUES"])

    payload = checker.get_report().to_dict()

    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["n_rows"] == 12
    assert decoded["executed_tests"] == ["MISSING_VALUES"]


def test_report_snapshot_does_not_expose_checker_state() -> None:
    """Mutating serialized output does not alter the checker."""
    checker = _checker()
    checker.check_data_quality(execute_list=["MISSING_VALUES"])

    payload = checker.get_report().to_dict()
    payload["row_scores"][0]["FINAL SCORE"] = 999

    assert checker.get_outlier_score_summary().iloc[0]["FINAL SCORE"] == 0


def test_report_before_analysis_has_defined_empty_state() -> None:
    """A checker with initialized data but no run still returns a valid report."""
    checker = _checker()

    report = checker.get_report()

    assert report.n_rows == 12
    assert report.n_columns == 2
    assert report.executed_tests == ()
    assert report.patterns == ()
    assert report.exceptions == ()
    assert len(report.row_scores) == 12
