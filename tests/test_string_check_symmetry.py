"""Regression tests for string checks that handled the same input differently from their mirrors.

Each test builds a small dataset on which a check and its mirror (or its siblings) must agree.
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
import pytest

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


def _findings(checker: DataConsistencyChecker) -> list[tuple]:
    """The patterns and exceptions found, with their descriptions."""
    patterns = checker.patterns_df
    exceptions = checker.exceptions_summary_df
    return sorted(
        list(zip(patterns["Test ID"], patterns["Column(s)"], patterns["Description of Pattern"]))
        + list(zip(exceptions["Test ID"], exceptions["Column(s)"], exceptions["Description of Pattern"],
                   exceptions["Number of Exceptions"])))


def _flagged_rows(checker: DataConsistencyChecker, test_id: str, column_set: str) -> list[int]:
    return np.flatnonzero(checker.results_dict[checker.get_results_col_name(test_id, column_set)]).tolist()


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


# ----------------------------------------------------------------------------------------------------------------------
# FIRST_CHAR_* and LAST_CHAR_*
# ----------------------------------------------------------------------------------------------------------------------


def _codes(chars, rng: np.random.Generator, at_end: bool = False) -> list[str]:
    """Distinct values starting (or, with at_end, ending) with characters drawn from chars."""
    picks = rng.choice(list(chars), N_ROWS)
    return [f"q{i:04d}{c}" if at_end else f"{c}q{i:04d}" for i, c in enumerate(picks)]


@pytest.mark.parametrize(
    ("test_id", "chars", "odd_value", "odd_breaks_pattern"),
    [
        ("FIRST_CHAR_UPPERCASE", "ABC", "Àgua", False),
        ("FIRST_CHAR_UPPERCASE", "ABC", "Ωmega", False),
        ("FIRST_CHAR_UPPERCASE", "ABC", "×2", True),
        ("FIRST_CHAR_LOWERCASE", "abc", "école", False),
        ("FIRST_CHAR_LOWERCASE", "abc", "ñandú", False),
        ("FIRST_CHAR_LOWERCASE", "abc", "×2", True),
    ],
)
def test_first_char_case_checks_use_unicode_case(test_id, chars, odd_value, odd_breaks_pattern) -> None:
    # FIRST_CHAR_LOWERCASE accepted only a-z, and FIRST_CHAR_UPPERCASE A-Z plus code points 193-221, which leave out
    # "À" and take in "×".
    values = _codes(chars, np.random.default_rng(0))
    values[10] = values[20] = odd_value

    checker = _run(pd.DataFrame({"code": values}), [test_id])

    if odd_breaks_pattern:
        assert _patterns(checker) == []
        assert _flagged_rows(checker, test_id, "code") == [10, 20]
    else:
        assert _patterns(checker) == ["code"]
        assert _exceptions(checker) == []


@pytest.mark.parametrize(
    ("test_id", "chars"),
    [
        ("FIRST_CHAR_ALPHA", string.ascii_letters),
        ("FIRST_CHAR_NUMERIC", string.digits),
        ("FIRST_CHAR_SMALL_SET", "ABC"),
        ("FIRST_CHAR_UPPERCASE", string.ascii_uppercase),
        ("FIRST_CHAR_LOWERCASE", string.ascii_lowercase),
        ("LAST_CHAR_SMALL_SET", "abc"),
    ],
)
def test_first_and_last_char_checks_look_past_whitespace(test_id, chars) -> None:
    # Leading and trailing whitespace have their own checks. Only FIRST_CHAR_ALPHA and LAST_CHAR_SMALL_SET skipped it
    # when testing the first or last character, and none of the FIRST_CHAR_* checks when deciding whether all values
    # start with the same character.
    at_end = test_id.startswith("LAST_CHAR")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"varied": _codes(chars, rng, at_end), "same": _codes(chars[0], rng, at_end)})
    padded = df.copy()
    for col_name in padded.columns:
        padded.loc[[5, 17], col_name] = [f"{x}  " if at_end else f"  {x}" for x in padded.loc[[5, 17], col_name]]

    expected = _findings(_run(df, [test_id]))

    assert (test_id, "varied") in [finding[:2] for finding in expected]
    assert _findings(_run(padded, [test_id])) == expected


# ----------------------------------------------------------------------------------------------------------------------
# A_PREFIX_OF_B and A_SUFFIX_OF_B
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("test_id", ["A_PREFIX_OF_B", "A_SUFFIX_OF_B"])
def test_prefix_and_suffix_checks_look_past_whitespace(test_id) -> None:
    # A_PREFIX_OF_B stripped the values when testing the full columns but not the sample, and A_SUFFIX_OF_B never did.
    rng = np.random.default_rng(0)
    a = ["".join(rng.choice(list(string.ascii_lowercase), 8)) for _ in range(N_ROWS)]
    b = [f"{x}-tail" if test_id == "A_PREFIX_OF_B" else f"head-{x}" for x in a]
    padded_a = [f"  {x}  " if i % 5 == 0 else x for i, x in enumerate(a)]

    checker = _run(pd.DataFrame({"a": padded_a, "b": b}), [test_id])

    assert _patterns(checker) == ['"a" AND "b"']
    assert _exceptions(checker) == []


@pytest.mark.parametrize("test_id", ["A_PREFIX_OF_B", "A_SUFFIX_OF_B"])
def test_prefix_and_suffix_checks_skip_columns_that_are_the_same(test_id) -> None:
    # Only A_PREFIX_OF_B skipped pairs of columns that are the same wherever both have values. Equal values are not a
    # strict prefix or suffix of each other, so such pairs can only support a pattern through their missing values.
    # Here the columns never both have values, and A_SUFFIX_OF_B reported that every row supported one.
    values = [f"v{i:04d}" for i in range(N_ROWS)]
    half = N_ROWS // 2
    df = pd.DataFrame({"a": values[:half] + [None] * half, "b": [None] * half + values[half:]})

    checker = _run(df, [test_id])

    assert _patterns(checker) == []
    assert _exceptions(checker) == []


# ----------------------------------------------------------------------------------------------------------------------
# NUMBER_*_CHARS
# ----------------------------------------------------------------------------------------------------------------------


def _letters(i: int, n: int) -> str:
    """A distinct string of n letters for each i."""
    return "".join(string.ascii_lowercase[(i // 26 ** k) % 26] for k in range(n))


@pytest.mark.parametrize(
    ("test_id", "make_value"),
    [
        ("NUMBER_ALPHA_CHARS", lambda i, n: _letters(i, n)),
        ("NUMBER_NUMERIC_CHARS", lambda i, n: "x" + f"{i:010d}"[-n:]),
        ("NUMBER_ALPHANUMERIC_CHARS", lambda i, n: "$" + f"{i:010d}"[-n:]),
        ("NUMBER_NON-ALPHANUMERIC_CHARS", lambda i, n: "@" * n + _letters(i, 3)),
        ("NUMBER_CHARS", lambda i, n: _letters(i, n)),
    ],
)
def test_number_of_chars_checks_do_not_flag_missing_values(test_id, make_value) -> None:
    # NUMBER_ALPHANUMERIC_CHARS gave missing values the median count. With values of 4 and 6 characters, that is 5,
    # which no value has, so the rows with missing values were flagged as rare.
    counts = [4] * 496 + [6] * 500
    df = pd.DataFrame({"code": [make_value(i, n) for i, n in enumerate(counts)] + [None] * 4})

    checker = _run(df, [test_id])

    assert _exceptions(checker) == []


# ----------------------------------------------------------------------------------------------------------------------
# LARGE_GIVEN_* and SMALL_GIVEN_*
# ----------------------------------------------------------------------------------------------------------------------


def _given_data(test_id: str, distinct_numbers: bool) -> pd.DataFrame:
    """Numbers that are small (for LARGE_*) or large (for SMALL_*) given label "A", with one outlier there, row 50."""
    rng = np.random.default_rng(0)
    label_idx = np.repeat([0, 1, 2], [200, 400, 400])
    low, high = rng.uniform(0, 3, N_ROWS), rng.uniform(10, 30, N_ROWS)
    if not distinct_numbers:
        low, high = low.round(), high.round()
    numbers = np.where(label_idx == 0, low, high)
    numbers[50] = 20.0
    if test_id.startswith("SMALL"):
        numbers = 33.0 - numbers
    return pd.DataFrame({"label": ["ABC"[i] for i in label_idx], "number": numbers})


@pytest.mark.parametrize("distinct_numbers", [True, False], ids=["many_values", "few_values"])
@pytest.mark.parametrize("test_id", ["LARGE_GIVEN_VALUE", "SMALL_GIVEN_VALUE"])
def test_given_value_checks_skip_numeric_columns_with_few_values(test_id, distinct_numbers) -> None:
    # SMALL_GIVEN_VALUE skipped numeric columns with fewer than sqrt(N) distinct values; LARGE_GIVEN_VALUE did not.
    checker = _run(_given_data(test_id, distinct_numbers), [test_id])

    if distinct_numbers:
        assert _flagged_rows(checker, test_id, '"label" AND "number"') == [50]
    else:
        assert _exceptions(checker) == []


@pytest.mark.parametrize("distinct_values", [True, False], ids=["many_values", "one_value"])
@pytest.mark.parametrize("test_id", ["LARGE_GIVEN_PREFIX", "SMALL_GIVEN_PREFIX"])
def test_given_prefix_checks_skip_columns_whose_prefix_is_the_value(test_id, distinct_values) -> None:
    # LARGE_GIVEN_PREFIX skipped columns with almost as many distinct first words as distinct values, where the first
    # word says nothing the value does not; SMALL_GIVEN_PREFIX did not.
    df = _given_data(test_id, distinct_numbers=True)
    suffixes = np.random.default_rng(1).choice(list(string.ascii_letters), N_ROWS) if distinct_values else "x" * N_ROWS
    df["label"] = [f"{label}-{suffix}" for label, suffix in zip(df["label"], suffixes)]

    checker = _run(df, [test_id])

    if distinct_values:
        assert _flagged_rows(checker, test_id, '"label" AND "number"') == [50]
    else:
        assert _exceptions(checker) == []


@pytest.mark.parametrize("test_id", ["LARGE_GIVEN_PREFIX", "SMALL_GIVEN_PREFIX"])
def test_given_prefix_checks_use_the_same_limits_for_dates(test_id) -> None:
    # LARGE_GIVEN_PREFIX sets the limit for dates at 1.5 times the subset's IQR, as dates vary less than numbers.
    # SMALL_GIVEN_PREFIX used iqr_limit (3.5 by default), so missed this date 2.5 IQRs before the first quartile.
    rng = np.random.default_rng(0)
    label_idx = np.repeat([0, 1, 2], [200, 400, 400])
    days = rng.integers(0, 101, N_ROWS)
    days[50] = 200
    large = test_id.startswith("LARGE")
    starts = np.where(label_idx == 0, np.datetime64("2000-06-01" if large else "2020-06-01"),
                      np.datetime64("2020-06-01" if large else "2000-06-01"))
    suffixes = rng.choice(list(string.ascii_letters), N_ROWS)
    df = pd.DataFrame({
        "label": [f"{'ABC'[i]}-{suffix}" for i, suffix in zip(label_idx, suffixes)],
        "when": pd.to_datetime(starts) + pd.to_timedelta(days if large else -days, unit="D"),
    })

    checker = _run(df, [test_id])

    assert _flagged_rows(checker, test_id, '"label" AND "when"') == [50]
