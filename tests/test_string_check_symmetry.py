"""Regression tests for string checks that handled the same input differently from their mirrors.

Each test builds a small dataset on which a check and its mirror (or its siblings) must agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_consistency_checker import DataConsistencyChecker

N_ROWS = 1_000


def _run(df: pd.DataFrame, execute_list: list[str]) -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    checker.check_data_quality(execute_list=execute_list)
    return checker


def _patterns(checker: DataConsistencyChecker) -> list[str]:
    return checker.patterns_df["Column(s)"].tolist()


def _exceptions(checker: DataConsistencyChecker) -> list[tuple[str, int]]:
    exceptions = checker.exceptions_summary_df
    return list(zip(exceptions["Column(s)"], exceptions["Number of Exceptions"]))


# ----------------------------------------------------------------------------------------------------------------------
# CORRELATED_GIVEN_VALUE
# ----------------------------------------------------------------------------------------------------------------------


def test_correlated_given_value_skips_pairs_correlated_without_conditioning() -> None:
    # Pairs correlated overall are skipped. That correlation took the first numeric column from one sample and the
    # others from another, so pairs with the first numeric column compared unrelated rows and were never skipped.
    rng = np.random.default_rng(0)
    group = rng.choice(["A", "B", "C"], N_ROWS)
    x = rng.permutation(N_ROWS).astype(float)
    u = rng.permutation(N_ROWS).astype(float)
    df = pd.DataFrame({
        "x": x,
        "y": x * 2 + 5,  # correlated overall
        "group": group,
        "u": u,
        "v": np.where(group == "B", N_ROWS - u, u),  # correlated within each group, but not overall
    })
    df.loc[[3, 7], "y"] = np.nan

    checker = _run(df, ["CORRELATED_GIVEN_VALUE"])

    assert _patterns(checker) == ['"group" AND "u" AND "v"']
    assert _exceptions(checker) == []
