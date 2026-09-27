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


def _flagged_rows(checker: DataConsistencyChecker) -> dict[tuple[str, str], list[int]]:
    """Map each (test ID, column set) with exceptions to the rows it flagged."""
    exceptions = checker.exceptions_summary_df
    return {
        (test_id, columns): np.flatnonzero(checker.test_results_df[checker.get_results_col_name(test_id, columns)]).tolist()
        for test_id, columns in zip(exceptions["Test ID"], exceptions["Column(s)"])
    }


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


# ----------------------------------------------------------------------------------------------------------------------
# TWO_PAIRS
# ----------------------------------------------------------------------------------------------------------------------


def test_two_pairs_uses_the_sample_matches_of_a_cached_inner_pair() -> None:
    # "c" and "d" are first tested as an inner pair for the outer pair ("a", "x"). For the outer pair ("a", "b"),
    # the cached pair ("c", "d") was then compared using the sample matches of the previous inner pair, ("x", "d").
    rng = np.random.default_rng(0)
    a, x, b, c = (rng.integers(0, 5, 1000) for _ in range(4))
    d = np.where(a == b, c, c + 1)
    df = pd.DataFrame({"a": a, "x": x, "b": b, "c": c, "d": d})

    checker = _run(df, ["TWO_PAIRS"])

    assert _patterns(checker) == ['"a" AND "b" AND "c" AND "d"']


# ----------------------------------------------------------------------------------------------------------------------
# SIMILAR_TO_NEGATIVE
# ----------------------------------------------------------------------------------------------------------------------


def test_similar_to_negative_treats_zero_as_the_negative_of_zero() -> None:
    # The sample test accepted rows where both values are 0, but the full test flagged them.
    y = np.random.default_rng(0).integers(1, 1000, 1000).astype(float)
    y[[10, 20, 30]] = 0.0
    df = pd.DataFrame({"x": -y, "y": y})

    checker = _run(df, ["SIMILAR_TO_NEGATIVE"])

    assert _patterns(checker) == ['"x" AND "y"']
    assert checker.exceptions_summary_df.empty


# ----------------------------------------------------------------------------------------------------------------------
# LARGER_THAN_ABS_DIFF
# ----------------------------------------------------------------------------------------------------------------------


def test_larger_than_abs_diff_skips_a_triple_with_any_two_columns_usually_equal() -> None:
    # The check is not enabled, so its method is called directly. Of the three pairs of columns that are skipped
    # when usually equal, one was tested twice and another never. Here "x" is usually equal to "z", so
    # "x" > abs("y" - "z") holds trivially, and was reported.
    rng = np.random.default_rng(0)
    x = rng.integers(100, 200, 1000).astype(float)
    z = x.copy()
    z[:50] += 1  # Equal in 95% of the rows
    df = pd.DataFrame({"x": x, "y": rng.integers(50, 150, 1000).astype(float), "z": z})
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    checker.freq_contamination_level = 5  # As check_data_quality() sets it by default: 0.5% of the rows

    checker._check_larger_than_abs_diff("LARGER_THAN_ABS_DIFF")

    assert checker.patterns_arr == []
    assert checker.results_summary_arr == []


# ----------------------------------------------------------------------------------------------------------------------
# EARLY_DATES and LATE_DATES
# ----------------------------------------------------------------------------------------------------------------------


def test_early_and_late_dates_flag_mirror_image_dates_alike() -> None:
    # EARLY_DATES used midpoint quartiles and LATE_DATES linear ones. The last date here is past the upper limit the
    # midpoint quartiles give (2505 days), but not the one the linear quartiles give (2525 days), so it was flagged
    # in the mirrored column as early, but not in this one as late.
    offsets = pd.to_timedelta(np.append(np.arange(101) * 10, 2515), unit="D")
    start = pd.Timestamp("2000-01-01")
    df = pd.DataFrame({"later": start + offsets, "earlier": start - offsets})

    checker = _run(df, ["EARLY_DATES", "LATE_DATES"])

    exceptions = checker.exceptions_summary_df
    assert list(zip(exceptions["Test ID"], exceptions["Column(s)"], exceptions["Number of Exceptions"])) == [
        ("EARLY_DATES", "earlier", 1),
        ("LATE_DATES", "later", 1),
    ]


# ----------------------------------------------------------------------------------------------------------------------
# LARGE_GIVEN_DATE and SMALL_GIVEN_DATE
# ----------------------------------------------------------------------------------------------------------------------


def test_large_and_small_given_date_flag_mirror_image_values_alike() -> None:
    # "neg_v" mirrors "v", so each check should flag in one column what the other flags in the other. LARGE_GIVEN_DATE
    # also flagged a value equal to its threshold, which SMALL_GIVEN_DATE does not. And SMALL_GIVEN_DATE examined any
    # bin whose first quartile is above the column's first decile, where LARGE_GIVEN_DATE only examines bins whose
    # median and third quartile are below the column's.
    v = np.arange(1000.0)  # Increases with the date, so each of the 10 bins of dates has 100 rows
    v[150] = 440.375  # In bin 1, exactly LARGE_GIVEN_DATE's threshold: Q3 175.25 + 1.5 * 3.5 * IQR 50.5
    v[250] = 700.0  # In bin 2, large for its bin
    v[450] = 100.0  # In bin 4, small for its bin, but the bin's values are not larger than the column's
    df = pd.DataFrame({"when": pd.date_range("2020-01-01", periods=1000, freq="D"), "v": v, "neg_v": -v})

    checker = _run(df, ["LARGE_GIVEN_DATE", "SMALL_GIVEN_DATE"])

    assert _flagged_rows(checker) == {
        ("LARGE_GIVEN_DATE", '"when" AND "v"'): [250],
        ("SMALL_GIVEN_DATE", '"when" AND "neg_v"'): [250],
    }
