"""A numeric column may hold a few values that are not numbers, such as "?". INVALID_NUMBERS flags them.

The checks below failed on such values. They now treat them as missing values, which neither support nor break a
pattern.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker

ROWS = [3, 500]

CHECKS = [
    "COLUMN_TENDS_ASC",
    "COLUMN_TENDS_DESC",
    "MUCH_LARGER",
    "CORRELATED_NUMERIC",
    "SUM_OF_COLUMNS",
    "MEAN_OF_COLUMNS",
    "MIN_OF_COLUMNS",
    "MAX_OF_COLUMNS",
    "MATCHED_SET_POS_NEG",
    "GROUPED_STRINGS_BY_NUMERIC",
    "CORRELATED_GIVEN_VALUE",
    "BINARY_MATCHES_VALUES",
    "SMALL_AVG_RANK_PER_ROW",
    "LARGE_AVG_RANK_PER_ROW",
]


def _run(test_id: str, placeholder: object, as_text: bool = False) -> DataConsistencyChecker:
    """Run the check on its synthetic data, with the placeholder in some rows of each numeric column.

    With as_text, the numbers are text, as read_csv() leaves a column holding values such as "?".
    """
    np.random.seed(0)
    random.seed(0)
    df = DataConsistencyChecker(verbose=-1).generate_synth_data(all_cols=False, execute_list=[test_id])
    for col_name in df.columns:
        if pd.api.types.is_numeric_dtype(df[col_name]) and df[col_name].nunique() > 2:
            df[col_name] = df[col_name].astype(str if as_text else object)
            df.loc[ROWS, col_name] = placeholder
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    checker.check_data_quality(execute_list=[test_id], raise_on_error=True)
    return checker


def _findings(checker: DataConsistencyChecker) -> tuple:
    patterns = checker.get_patterns_list(show_short_list_only=False)
    exceptions = checker.get_exceptions_list()
    return (
        None if patterns is None else patterns["Column(s)"].tolist(),
        None if exceptions is None else exceptions[["Column(s)", "Number of Exceptions"]].values.tolist(),
    )


@pytest.mark.parametrize("as_text", [False, True], ids=["numbers", "text"])
@pytest.mark.parametrize("test_id", CHECKS)
def test_a_value_that_is_not_a_number_counts_as_missing(test_id: str, as_text: bool) -> None:
    with_placeholders = _run(test_id, "?", as_text)
    with_missing_values = _run(test_id, np.nan, as_text)

    assert with_placeholders.get_execution_failures() == []
    assert _findings(with_placeholders) == _findings(with_missing_values)
    np.testing.assert_array_equal(with_placeholders.test_results_by_column_np,
                                  with_missing_values.test_results_by_column_np)


def test_invalid_numbers_still_flags_the_values() -> None:
    df = pd.DataFrame({"amount": np.arange(1.0, 1001.0)}).astype(object)
    df.loc[ROWS, "amount"] = "?"
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)

    checker.check_data_quality(execute_list=["INVALID_NUMBERS"])

    assert sorted(np.flatnonzero(checker.get_outlier_scores())) == ROWS


def test_the_sample_leaves_out_rows_with_values_that_are_not_numbers() -> None:
    df = pd.DataFrame({"amount": np.arange(1000.0), "other": np.arange(1000.0) * 2})
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    rows = list(checker.sample_df.index[:4])  # Rows the sample includes when all values are numbers
    df = df.astype({"amount": object})
    df.loc[rows, "amount"] = "?"

    checker.init_data(df)

    assert len(checker.sample_df) == 50
    assert set(checker.sample_df.index).isdisjoint(rows)


def test_last_char_small_set_handles_many_blank_values() -> None:
    rng = random.Random(0)
    codes = ["".join(rng.choice("wxyz") for _ in range(4)) + rng.choice("abc") for _ in range(980)]
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(pd.DataFrame({"code": codes + [" "] * 20}))

    checker.check_data_quality(execute_list=["LAST_CHAR_SMALL_SET"], raise_on_error=True)

    assert checker.get_execution_failures() == []


def test_new_data_replaces_the_numeric_values_of_the_previous_data() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)}))
    checker.init_data(pd.DataFrame({"a": [f"id{x}" for x in range(100)], "c": np.arange(100.0)}))

    assert set(checker.numeric_vals_nan) == {"c"}
    assert list(checker.get_numeric_vals_nan_df().columns) == ["c"]
    assert not checker.get_no_value_mask("a").any()
