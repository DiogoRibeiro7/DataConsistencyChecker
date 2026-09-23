"""Tests for immutable analysis configuration and the one-shot API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from data_consistency_checker import (
    DataConsistencyConfig,
    DataConsistencyReport,
    analyze,
)


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": list(range(12)),
            "group": ["a", "b"] * 6,
        }
    )


def test_config_is_immutable() -> None:
    """Analysis configuration cannot drift after construction."""
    config = DataConsistencyConfig(execute_tests=("MISSING_VALUES",))

    with pytest.raises(FrozenInstanceError):
        config.fast_only = True  # type: ignore[misc]


def test_config_rejects_conflicting_test_filters() -> None:
    """Include and exclude filters cannot be active simultaneously."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        DataConsistencyConfig(
            execute_tests=("MISSING_VALUES",),
            exclude_tests=("VERY_LARGE",),
        )


def test_analyze_returns_structured_report() -> None:
    """The one-shot API executes the configured checks and returns a report."""
    report = analyze(
        _dataframe(),
        DataConsistencyConfig(
            execute_tests=("MISSING_VALUES",),
            verbose=-1,
        ),
    )

    assert isinstance(report, DataConsistencyReport)
    assert report.executed_tests == ("MISSING_VALUES",)
    assert report.n_rows == 12
    assert report.n_columns == 2


def test_analyze_is_reproducible_for_same_input_and_config() -> None:
    """Equivalent inputs and immutable config produce equivalent report payloads."""
    config = DataConsistencyConfig(
        execute_tests=("MISSING_VALUES",),
        verbose=-1,
    )

    first = analyze(_dataframe(), config).to_dict()
    second = analyze(_dataframe(), config).to_dict()

    assert first == second


def test_analyze_default_config_runs_implemented_tests() -> None:
    """Omitting a config uses the documented defaults."""
    report = analyze(_dataframe())

    assert report.n_rows == 12
    assert len(report.executed_tests) > 0


def test_known_date_columns_are_forwarded() -> None:
    """Explicit date-column configuration is accepted by the one-shot API."""
    frame = _dataframe()
    frame["date"] = pd.date_range("2026-01-01", periods=12, freq="D").astype(str)

    report = analyze(
        frame,
        DataConsistencyConfig(
            execute_tests=("MISSING_VALUES",),
            known_date_cols=("date",),
            verbose=-1,
        ),
    )

    assert report.n_columns == 3
