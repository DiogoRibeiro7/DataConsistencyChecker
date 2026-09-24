"""Coverage tests for result-state management."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker

_TEST_A = "TEST_A"
_TEST_B = "TEST_B"
_COL_A = "a"
_COL_B = "b"


def _checker_with_results() -> DataConsistencyChecker:
    """Create a checker with deterministic pattern and exception state."""
    checker = DataConsistencyChecker(verbose=-1)

    result_a = checker.get_results_col_name(_TEST_A, _COL_A)
    result_b = checker.get_results_col_name(_TEST_B, _COL_B)

    checker.patterns_arr = [
        (_TEST_A, _COL_A, "pattern-a"),
        (_TEST_B, _COL_B, "pattern-b"),
    ]
    checker.patterns_df = pd.DataFrame(
        [
            {"Test ID": _TEST_A, "Column(s)": _COL_A, "Pattern ID": 101},
            {"Test ID": _TEST_B, "Column(s)": _COL_B, "Pattern ID": 102},
        ]
    )

    checker.results_summary_arr = [
        (_TEST_A, _COL_A, "issue-a"),
        (_TEST_B, _COL_B, "issue-b"),
    ]
    checker.exceptions_summary_df = pd.DataFrame(
        [
            {"Test ID": _TEST_A, "Column(s)": _COL_A, "Issue ID": 201},
            {"Test ID": _TEST_B, "Column(s)": _COL_B, "Issue ID": 202},
        ]
    )

    checker.test_results_df = pd.DataFrame(
        {
            result_a: [1, 0],
            result_b: [0, 1],
            "FINAL SCORE": [1, 1],
        }
    )
    checker.results_dict = {
        result_a: [True, False],
        result_b: [False, True],
    }
    checker.col_to_original_cols_dict = {
        _COL_A: [_COL_A],
        _COL_B: [_COL_B],
        result_a: [_COL_A],
        result_b: [_COL_B],
    }

    return checker


def _assert_only_test_b_remains(checker: DataConsistencyChecker) -> None:
    """Assert that state associated with TEST_A has been removed."""
    result_a = checker.get_results_col_name(_TEST_A, _COL_A)
    result_b = checker.get_results_col_name(_TEST_B, _COL_B)

    assert [row[0] for row in checker.patterns_arr] == [_TEST_B]
    assert checker.patterns_df["Test ID"].tolist() == [_TEST_B]
    assert [row[0] for row in checker.results_summary_arr] == [_TEST_B]
    assert checker.exceptions_summary_df["Test ID"].tolist() == [_TEST_B]
    assert result_a not in checker.test_results_df.columns
    assert result_b in checker.test_results_df.columns
    assert result_a not in checker.results_dict
    assert result_b in checker.results_dict


def test_add_result_column_preserves_existing_results() -> None:
    """Adding a result column should retain the existing result frame."""
    checker = DataConsistencyChecker(verbose=-1)
    checker.test_results_df = pd.DataFrame({"existing": [1, 0]})

    checker._add_result_column("new", [0, 1])

    assert checker.test_results_df.to_dict(orient="list") == {
        "existing": [1, 0],
        "new": [0, 1],
    }


def test_clear_results_requires_exactly_one_selector(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Calls without one unambiguous selector should leave state unchanged."""
    checker = _checker_with_results()

    checker.clear_results()
    assert "No method of removing results specified" in capsys.readouterr().out
    assert len(checker.patterns_arr) == 2

    checker.clear_results(
        test_id_list=[_TEST_A],
        clear_all_patterns=True,
    )
    assert "Only one method or removing results may be specified" in capsys.readouterr().out
    assert len(checker.patterns_arr) == 2


def test_clear_results_rejects_uninitialized_result_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Clearing should stop cleanly when result tables are unavailable."""
    checker = _checker_with_results()
    checker.exceptions_summary_df = None

    checker.clear_results(test_id_list=[_TEST_A])

    assert "There are no results to clear" in capsys.readouterr().out


def test_clear_results_by_test_id_updates_all_result_views() -> None:
    """Removing a test should remove its patterns, exceptions, and result column."""
    checker = _checker_with_results()

    checker.clear_results(test_id_list=[_TEST_A])

    _assert_only_test_b_remains(checker)


def test_clear_results_by_code_test_uses_code_test_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The code-test shortcut should delegate to the same test-ID removal path."""
    checker = _checker_with_results()
    monkeypatch.setattr(checker, "get_tests_for_codes", lambda: [_TEST_A])

    checker.clear_results(clear_code_tests=True)

    _assert_only_test_b_remains(checker)


def test_clear_results_by_original_column_updates_all_result_views() -> None:
    """Removing a source column should remove result state derived from it."""
    checker = _checker_with_results()

    checker.clear_results(col_name_list=[_COL_A])

    _assert_only_test_b_remains(checker)


def test_clear_results_by_pattern_id_only_removes_patterns() -> None:
    """Pattern-ID clearing should not alter exception state."""
    checker = _checker_with_results()

    checker.clear_results(pattern_id_list=[101])

    assert [row[0] for row in checker.patterns_arr] == [_TEST_B]
    assert checker.patterns_df["Pattern ID"].tolist() == [102]
    assert len(checker.results_summary_arr) == 2
    assert len(checker.exceptions_summary_df) == 2


def test_clear_results_by_issue_id_only_removes_selected_exception() -> None:
    """Issue-ID clearing should remove the matching exception result column."""
    checker = _checker_with_results()
    result_a = checker.get_results_col_name(_TEST_A, _COL_A)
    result_b = checker.get_results_col_name(_TEST_B, _COL_B)

    checker.clear_results(issue_id_list=[201])

    assert len(checker.patterns_arr) == 2
    assert [row[0] for row in checker.results_summary_arr] == [_TEST_B]
    assert checker.exceptions_summary_df["Issue ID"].tolist() == [202]
    assert result_a not in checker.test_results_df.columns
    assert result_b in checker.test_results_df.columns
    assert result_a not in checker.results_dict


def test_clear_all_patterns_leaves_exceptions_untouched() -> None:
    """Clearing all patterns should preserve exception result state."""
    checker = _checker_with_results()

    checker.clear_results(clear_all_patterns=True)

    assert checker.patterns_arr == []
    assert checker.patterns_df.empty
    assert len(checker.results_summary_arr) == 2
    assert len(checker.exceptions_summary_df) == 2


def test_clear_all_exceptions_clears_exception_state() -> None:
    """Clearing all exceptions should empty all exception-specific containers."""
    checker = _checker_with_results()

    checker.clear_results(clear_all_exceptions=True)

    assert checker.results_summary_arr == []
    assert checker.exceptions_summary_df.empty
    assert checker.test_results_df.empty
    assert checker.results_dict == {}
    assert len(checker.patterns_arr) == 2


def test_restore_results_restores_safe_snapshot_with_copies() -> None:
    """Restoring should rebuild all mutable result state from the safe snapshot."""
    checker = _checker_with_results()

    checker.safe_patterns_arr = list(checker.patterns_arr)
    checker.safe_patterns_df = checker.patterns_df.copy()
    checker.safe_results_summary_arr = list(checker.results_summary_arr)
    checker.safe_exceptions_summary_df = checker.exceptions_summary_df.copy()
    checker.safe_test_results_df = checker.test_results_df.copy()
    checker.safe_results_dict = dict(checker.results_dict)
    checker.safe_test_results_by_column_np = np.array([[1.0, 0.0], [0.0, 1.0]])

    checker.clear_results(clear_all_patterns=True)
    checker.results_summary_arr = []
    checker.exceptions_summary_df = checker.exceptions_summary_df[0:0]
    checker.test_results_df = checker.test_results_df[0:0]
    checker.results_dict = {}
    checker.test_results_by_column_np = np.zeros((2, 2))

    checker.restore_results()

    assert len(checker.patterns_arr) == 2
    assert checker.patterns_df["Pattern ID"].tolist() == [101, 102]
    assert len(checker.results_summary_arr) == 2
    assert checker.exceptions_summary_df["Issue ID"].tolist() == [201, 202]
    assert len(checker.test_results_df) == 2
    assert len(checker.results_dict) == 2
    np.testing.assert_array_equal(
        checker.test_results_by_column_np,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    checker.patterns_arr.clear()
    assert len(checker.safe_patterns_arr) == 2


def test_output_stats_handles_no_executed_tests(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Statistics output should handle an empty execution cleanly."""
    checker = DataConsistencyChecker(verbose=-1)
    checker.n_tests_executed = 0

    checker._output_stats()

    assert capsys.readouterr().out.strip() == "No tests executed."
