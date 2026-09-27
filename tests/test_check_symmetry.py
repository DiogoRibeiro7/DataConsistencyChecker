"""Regression tests for checks whose mirrored branches, or mirrored sibling checks, disagreed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_consistency_checker import DataConsistencyChecker


def _run(df: pd.DataFrame, execute_list: list[str], **kwargs) -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    checker.check_data_quality(execute_list=execute_list, **kwargs)
    return checker


def _patterns(checker: DataConsistencyChecker) -> list[str]:
    return checker.patterns_df["Column(s)"].tolist()


def _near_constant(num_rows: int) -> np.ndarray:
    """A numeric column with one value in all but 3 rows: fewer than the default contamination level of 0.5%."""
    values = np.zeros(num_rows)
    values[:3] = [1.0, 2.0, 3.0]
    return values


# ----------------------------------------------------------------------------------------------------------------------
# BINARY_MATCHES_VALUES and BINARY_MATCHES_SUM
# ----------------------------------------------------------------------------------------------------------------------


def test_binary_matches_values_skips_only_the_near_constant_column() -> None:
    # A near-constant numeric column ended the check, so numeric columns after it were never examined.
    num = np.random.default_rng(0).integers(0, 1000, 1000).astype(float)
    df = pd.DataFrame({"near_constant": _near_constant(1000), "num": num, "label": (num > 500).astype(int)})

    checker = _run(df, ["BINARY_MATCHES_VALUES"])

    assert _patterns(checker) == ['"num" AND "label"']


def test_binary_matches_sum_skips_only_the_pairs_with_a_near_constant_column() -> None:
    # A near-constant numeric column ended the check, so pairs of other numeric columns were never examined.
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, 1000).astype(float)
    b = rng.integers(0, 100, 1000).astype(float)
    df = pd.DataFrame({"near_constant": _near_constant(1000), "a": a, "b": b, "label": (a + b > 100).astype(int)})

    checker = _run(df, ["BINARY_MATCHES_SUM"])

    assert _patterns(checker) == ['"a" AND "b" AND "label"']
