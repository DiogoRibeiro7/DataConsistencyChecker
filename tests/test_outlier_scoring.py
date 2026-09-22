"""Tests for raw and normalized outlier scoring."""

from __future__ import annotations

import pandas as pd

from check_data_consistency import DataConsistencyChecker


def _checker_with_results() -> DataConsistencyChecker:
    """Return a checker with deterministic synthetic exception-result columns."""
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame({"x": [1, 2, 3, 4]})
    checker.test_results_df = pd.DataFrame(
        {
            "TEST A -- x RESULT": [1, 0, 1, 0],
            "TEST B -- x RESULT": [1, 0, 0, 0],
            "TEST C -- x RESULT": [0, 0, 1, 0],
        }
    )
    checker._calculate_final_scores()
    return checker


def test_raw_outlier_scores_remain_backward_compatible() -> None:
    """The default API continues to return raw integer flag counts."""
    checker = _checker_with_results()

    assert checker.get_outlier_scores() == [2, 0, 2, 0]


def test_normalized_outlier_scores_are_bounded() -> None:
    """Normalized scores divide raw counts by active result columns."""
    checker = _checker_with_results()

    assert checker.get_outlier_scores(normalized=True) == [
        2 / 3,
        0.0,
        2 / 3,
        0.0,
    ]


def test_score_recalculation_does_not_count_derived_columns() -> None:
    """Repeated score calculation must not inflate scores with prior outputs."""
    checker = _checker_with_results()

    checker._calculate_final_scores()
    checker._calculate_final_scores()

    assert checker.get_outlier_scores() == [2, 0, 2, 0]
    assert checker.get_outlier_scores(normalized=True) == [
        2 / 3,
        0.0,
        2 / 3,
        0.0,
    ]


def test_score_summary_is_defensive_copy() -> None:
    """Editing the summary must not mutate checker state."""
    checker = _checker_with_results()

    summary = checker.get_outlier_score_summary()
    summary.loc[0, "FINAL SCORE"] = 999

    assert checker.get_outlier_score_summary().loc[0, "FINAL SCORE"] == 2


def test_empty_result_set_returns_zero_scores() -> None:
    """No exception-bearing patterns yields well-defined zero scores."""
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame({"x": [1, 2, 3]})
    checker.test_results_df = pd.DataFrame(index=range(3))
    checker._calculate_final_scores()

    assert checker.get_outlier_scores() == [0, 0, 0]
    assert checker.get_outlier_scores(normalized=True) == [0.0, 0.0, 0.0]

    summary = checker.get_outlier_score_summary()
    assert summary["FINAL SCORE"].tolist() == [0, 0, 0]
    assert summary["NORMALIZED SCORE"].tolist() == [0.0, 0.0, 0.0]
