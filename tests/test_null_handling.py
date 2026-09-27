"""Every check runs on its synthetic data, and missing values neither support nor break a pattern.

See "Missing values" in docs/guide/how-it-works.md: a row with a missing value in a column a check examines is not
counted against the pattern and is not flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker

ALL_TEST_IDS = DataConsistencyChecker(verbose=-1).get_test_list()

# For these checks, which values are missing is the data itself.
ABOUT_MISSING_VALUES = {
    "MISSING_VALUES",
    "MISSING_VALUES_PER_ROW",
    "MATCHED_MISSING",
    "OPPOSITE_MISSING",
    "MATCHED_ZERO_MISSING",
    "PREDICT_NULL_DT",
}
# CORRELATED_GIVEN_VALUE conditions on the values of a column, and a missing value is one of them.
MISSING_AS_A_VALUE = {"CORRELATED_GIVEN_VALUE"}
# Row-wise checks judge a row on the values it has: only a row with no values at all cannot be judged.
PER_ROW = {test_id for test_id in ALL_TEST_IDS if test_id.endswith("_PER_ROW")}


def _rows_flagged_for_missing_values(checker: DataConsistencyChecker, per_row: bool) -> list[str]:
    """Return the results flagging a row that has missing values in the columns examined."""
    flagged_for_missing = []
    for results_col_name, flagged in checker.results_dict.items():
        values = checker.orig_df[list(checker.col_to_original_cols_dict[results_col_name])]
        missing = values.isna().all(axis=1) if per_row else values.isna().any(axis=1)
        if (np.asarray(flagged, dtype=bool) & missing.to_numpy()).any():
            flagged_for_missing.append(results_col_name)
    return flagged_for_missing


@pytest.mark.parametrize("add_nones", ["none", "one-row", "random"])
@pytest.mark.parametrize("test_id", ALL_TEST_IDS)
def test_check_runs_on_its_synthetic_data(test_id: str, add_nones: str) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(checker.generate_synth_data(execute_list=[test_id], add_nones=add_nones))

    checker.check_data_quality(execute_list=[test_id], raise_on_error=True)

    assert checker.get_execution_failures() == []
    if test_id not in ABOUT_MISSING_VALUES | MISSING_AS_A_VALUE:
        assert _rows_flagged_for_missing_values(checker, per_row=test_id in PER_ROW) == []


def test_columns_never_present_together_are_not_related() -> None:
    # "a" is missing in even rows and "b" in odd rows: no row shows how they relate, so no finding may involve both.
    rng = np.random.default_rng(0)
    a = rng.integers(1, 100, 200).astype(float)
    b = rng.integers(1, 100, 200).astype(float)
    df = pd.DataFrame({"a": a, "b": b, "sum": a + b, "mean": (a + b) / 2, "diff": a - b})
    df.loc[::2, "a"] = np.nan
    df.loc[1::2, "b"] = np.nan
    checks = ["CONSTANT_DIFF", "CONSTANT_PRODUCT", "CONSTANT_RATIO", "SUM_OF_COLUMNS", "MEAN_OF_COLUMNS",
              "SIMILAR_TO_DIFF", "ZERO_VALUES_PER_ROW", "UNIQUE_VALUES_PER_ROW"]
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)

    checker.check_data_quality(execute_list=checks, raise_on_error=True)

    findings = pd.concat([checker.patterns_df, checker.exceptions_summary_df])
    related = [(test_id, cols) for test_id, cols in zip(findings["Test ID"], findings["Column(s)"])
               if {"a", "b"} <= set(checker.col_to_original_cols_dict[cols])]
    assert related == []
