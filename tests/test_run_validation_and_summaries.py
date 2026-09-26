"""Input validation of check_data_quality() and the per-test / per-column summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_consistency_checker import DataConsistencyChecker

CHECKMARK = "✔"


def _frame() -> pd.DataFrame:
    rows = np.arange(60)
    df = pd.DataFrame({"neg_all": -(rows + 1.0), "neg_most": -(rows + 3.0), "other": rows % 7 + 1.0})
    df.loc[11, "neg_most"] = 5.0
    return df


def _checker() -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(_frame())
    return checker


def test_check_data_quality_requires_initialised_data(capsys) -> None:
    checker = DataConsistencyChecker(verbose=-1)

    checker.check_data_quality()

    assert "Valid dataframe not specified" in capsys.readouterr().out
    assert checker.patterns_df is None


def test_contamination_count_larger_than_the_dataset_is_rejected(capsys) -> None:
    checker = _checker()

    checker.check_data_quality(execute_list=["NEGATIVE"], freq_contamination_level=1000)

    assert "more than the number of rows" in capsys.readouterr().out
    assert checker.patterns_df is None


def test_contamination_fraction_allowing_no_rows_is_rejected(capsys) -> None:
    checker = _checker()

    checker.check_data_quality(execute_list=["NEGATIVE"], freq_contamination_level=0.001)

    assert "not allowing even 1 row" in capsys.readouterr().out
    assert checker.patterns_df is None


def test_patterns_summary_marks_the_columns_of_each_test() -> None:
    checker = _checker()
    checker.check_data_quality(execute_list=["NEGATIVE"])

    summary = checker.summarize_patterns_by_test_and_feature()

    assert list(summary.index) == ["NEGATIVE"]
    assert list(summary.columns) == ["neg_all", "neg_most", "other"]
    assert summary.loc["NEGATIVE"].tolist() == [CHECKMARK, "", ""]


def test_exceptions_summary_counts_exceptions_per_column() -> None:
    checker = _checker()
    checker.check_data_quality(execute_list=["NEGATIVE"])

    summary = checker.summarize_exceptions_by_test_and_feature()

    assert list(summary.index) == ["NEGATIVE"]
    assert summary.loc["NEGATIVE"].tolist() == ["", 1, ""]
