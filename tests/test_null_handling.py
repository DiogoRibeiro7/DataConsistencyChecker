"""Every check runs on its synthetic data, and missing values neither support nor break a pattern.

See "Missing values" in docs/guide/how-it-works.md: a row with a missing value in a column a check examines is not
counted against the pattern and is not flagged.
"""

from __future__ import annotations

import numpy as np
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
