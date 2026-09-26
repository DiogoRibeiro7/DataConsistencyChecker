"""clear_results() keeps the scores and the other derived results consistent with the remaining findings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker

ROW = 42  # the corrupted row of the order table


def _orders() -> pd.DataFrame:
    rows = np.arange(500)
    df = pd.DataFrame(
        {
            "price": (rows % 50) * 2.0 + 10,
            "quantity": rows % 7 + 1,
            "region": np.where(rows % 3 == 0, "north", "south"),
        }
    )
    df["total"] = df["price"] * df["quantity"]
    df.loc[ROW, "total"] = 1.0
    return df


@pytest.fixture
def checker() -> DataConsistencyChecker:
    dc = DataConsistencyChecker(verbose=-1)
    dc.init_data(_orders())
    dc.check_data_quality()
    return dc


def _issue_id(dc: DataConsistencyChecker, test_id: str) -> int:
    exceptions = dc.get_exceptions_list()
    return int(exceptions.loc[exceptions["Test ID"] == test_id, "Issue ID"].iloc[0])


def test_clearing_a_check_updates_the_scores(checker) -> None:
    assert checker.get_outlier_scores()[ROW] == 5

    checker.clear_results(test_id_list=["SIMILAR_TO_PRODUCT"])

    assert checker.get_outlier_scores()[ROW] == 4
    assert checker.get_outlier_scores(normalized=True)[ROW] == pytest.approx(1.0)
    assert ("SIMILAR_TO_PRODUCT", '"price" AND "quantity" AND "total"') not in checker.get_results_by_row_id(ROW)


def test_clearing_an_issue_updates_the_cell_scores(checker) -> None:
    before = checker.get_exceptions_by_column().loc[ROW].sum()

    checker.clear_results(issue_id_list=[_issue_id(checker, "UNUSUAL_ORDER_MAGNITUDE")])

    assert checker.get_outlier_scores()[ROW] == 4
    assert checker.get_exceptions_by_column().loc[ROW].sum() == pytest.approx(before - 1.0)


def test_clearing_all_exceptions_keeps_every_row_with_a_zero_score(checker) -> None:
    patterns_before = len(checker.get_patterns_list(show_short_list_only=False))

    checker.clear_results(clear_all_exceptions=True)

    assert checker.get_outlier_scores() == [0] * 500
    assert checker.get_exceptions_list() is None  # no exceptions left
    assert checker.get_exceptions_by_column().to_numpy().sum() == 0
    assert len(checker.get_patterns_list(show_short_list_only=False)) == patterns_before


def test_clearing_a_column_removes_its_patterns_and_exceptions(checker) -> None:
    checker.clear_results(col_name_list=["total"])

    patterns = checker.get_patterns_list(show_short_list_only=False)
    assert len(patterns) > 0
    assert not any("total" in columns for columns in patterns["Column(s)"])
    assert checker.get_exceptions_list() is None  # all five exceptions involve total
    assert checker.get_outlier_scores()[ROW] == 0


def test_cleared_findings_stay_cleared_when_results_are_appended(checker) -> None:
    checker.clear_results(test_id_list=["SIMILAR_TO_PRODUCT"])
    checker.clear_results(issue_id_list=[_issue_id(checker, "LARGER_DIFF_RANGE")])
    checker.clear_results(col_name_list=["region"])
    patterns = checker.get_patterns_list(show_short_list_only=False)
    checker.clear_results(pattern_id_list=[int(patterns["Pattern ID"].iloc[-1])])
    expected_exceptions = sorted(checker.get_exceptions_list()["Test ID"])
    expected_patterns = sorted(checker.get_patterns_list(show_short_list_only=False)["Column(s)"])

    checker.check_data_quality(execute_list=["NEGATIVE"], append_results=True)  # finds nothing new

    assert sorted(checker.get_exceptions_list()["Test ID"]) == expected_exceptions
    assert sorted(checker.get_patterns_list(show_short_list_only=False)["Column(s)"]) == expected_patterns


def test_restore_results_restores_the_scores(checker) -> None:
    checker.clear_results(clear_all_exceptions=True)

    checker.restore_results()

    assert checker.get_outlier_scores()[ROW] == 5
    assert len(checker.get_exceptions_list()) == 5
