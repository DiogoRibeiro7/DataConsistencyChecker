"""Tests for analysis planning helpers."""

from __future__ import annotations

import pytest

from data_consistency_checker import DataConsistencyChecker


def test_limit_subset_sizes_reduces_search_space(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Large combination spaces should be reduced to a workable subset size."""
    checker = DataConsistencyChecker(max_combinations=40, verbose=1)
    columns = ["target"]
    similar = {"target": ["a", "b", "c", "d", "e", "f"]}

    limited, max_subset_size, can_process = checker.get_limit_subset_sizes(
        columns,
        similar,
        calc_size=100,
    )

    assert limited is True
    assert max_subset_size == 3
    assert can_process is True
    assert "limiting test to subsets of size 3" in capsys.readouterr().out


def test_limit_subset_sizes_rejects_unworkable_search_space(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tests should be skipped when even pairs exceed the combination limit."""
    checker = DataConsistencyChecker(max_combinations=10, verbose=1)
    columns = ["target"]
    similar = {"target": ["a", "b", "c", "d", "e", "f"]}

    limited, max_subset_size, can_process = checker.get_limit_subset_sizes(
        columns,
        similar,
        calc_size=100,
    )

    assert limited is False
    assert max_subset_size == 2
    assert can_process is False
    output = capsys.readouterr().out
    assert "Skipping test" in output
    assert "15 combinations" in output
