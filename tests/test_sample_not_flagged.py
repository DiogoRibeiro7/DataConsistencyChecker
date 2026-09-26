"""Behavioural tests for selecting example rows that were not flagged."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker

N_EXAMPLES = 10


def _checker(df: pd.DataFrame) -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    return checker


def _pattern_sample(checker: DataConsistencyChecker, test_id: str, column_set: str, **kwargs) -> pd.DataFrame:
    return checker._get_sample_not_flagged(test_id, column_set, is_patterns=True, **kwargs)


@pytest.fixture(scope="module")
def mixed_checker() -> DataConsistencyChecker:
    rows = np.arange(40)
    df = pd.DataFrame(
        {
            "bin_a": np.where(rows % 2 == 0, "yes", "no"),
            "bin_b": np.where((rows // 2) % 2 == 0, 0, 1),
            "bin_c": np.where((rows // 4) % 2 == 0, "x", "y"),
            "num": (rows % 7) * 3 + 1.0,
            "zero_num": np.where(rows % 10 < 2, 0.0, rows + 1.0),
            "pos": np.where(rows % 5 == 0, 0.0, rows + 1.0),
            "neg": np.where(rows % 5 == 0, 0.0, -(rows + 1.0)),
            "posneg": np.where(rows % 2 == 0, rows + 1.0, -(rows + 1.0)),
            "same1": rows % 6 + 1,
            "same2": np.where(rows % 3 == 0, 99, rows % 6 + 1),
            "miss": [None if i % 4 == 0 else f"v{i}" for i in rows],
            "other": [f"o{i}" for i in rows],
        }
    )
    return _checker(df)


def test_binary_pair_sample_covers_every_value_combination(mixed_checker) -> None:
    sample = _pattern_sample(mixed_checker, "BINARY_SAME", '"bin_a" AND "bin_b"')

    assert list(sample.columns) == ["bin_a", "bin_b"]
    assert set(zip(sample["bin_a"], sample["bin_b"])) == {("no", 0), ("no", 1), ("yes", 0), ("yes", 1)}


def test_binary_triple_sample_draws_from_balanced_combinations(mixed_checker) -> None:
    cols = ["bin_a", "bin_b", "bin_c"]
    sample = _pattern_sample(mixed_checker, "BINARY_AND", '"bin_a" AND "bin_b" AND "bin_c"')
    first_rows_per_combination = mixed_checker.orig_df.groupby(cols).head(N_EXAMPLES // 4)

    assert len(sample) == N_EXAMPLES
    assert set(sample.index) <= set(first_rows_per_combination.index)
    # At most two rows are taken per combination, so ten rows span at least five combinations.
    assert len(set(zip(*(sample[c] for c in cols)))) >= 5


def test_binary_target_sample_balances_the_binary_column(mixed_checker) -> None:
    sample = _pattern_sample(mixed_checker, "BINARY_MATCHES_SUM", '"num" AND "same1" AND "bin_a"')

    assert sample["bin_a"].value_counts().to_dict() == {"no": N_EXAMPLES // 2, "yes": N_EXAMPLES // 2}


def test_binary_test_on_non_binary_columns_uses_generic_sample(mixed_checker) -> None:
    sample = _pattern_sample(mixed_checker, "BINARY_SAME", '"num" AND "same1"')

    assert list(sample.columns) == ["num", "same1"]
    assert 0 < len(sample) <= N_EXAMPLES
    assert sample.index.is_unique


_SIDES = {
    "zero": lambda s: s == 0,
    "non_zero": lambda s: s != 0,
    "positive": lambda s: s > 0,
    "negative": lambda s: s < 0,
    "missing": lambda s: s.isna(),
    "present": lambda s: s.notna(),
}


@pytest.mark.parametrize(
    ("test_id", "column_set", "column", "first_side", "second_side"),
    [
        ("MATCHED_ZERO", '"zero_num" AND "num"', "zero_num", "zero", "non_zero"),
        ("POSITIVE", "pos", "pos", "zero", "positive"),
        ("NEGATIVE", "neg", "neg", "zero", "negative"),
        ("MATCHED_SET_POS_NEG", '"posneg" AND "num"', "posneg", "positive", "negative"),
        ("MATCHED_MISSING", '"miss" AND "other"', "miss", "missing", "present"),
        ("PREDICT_NULL_DT", '"other" AND "miss"', "miss", "missing", "present"),
    ],
)
def test_two_sided_pattern_sample_shows_both_sides(
    mixed_checker, test_id, column_set, column, first_side, second_side
) -> None:
    sample = _pattern_sample(mixed_checker, test_id, column_set)

    assert len(sample) == N_EXAMPLES
    assert _SIDES[first_side](sample[column]).sum() == N_EXAMPLES // 2
    assert _SIDES[second_side](sample[column]).sum() == N_EXAMPLES // 2


def test_same_or_constant_sample_shows_matching_and_differing_rows(mixed_checker) -> None:
    sample = _pattern_sample(mixed_checker, "SAME_OR_CONSTANT", '"same1" AND "same2"')
    same = sample["same1"] == sample["same2"]

    assert same.sum() == N_EXAMPLES // 2
    assert (~same).sum() == N_EXAMPLES // 2


def test_matched_zero_sample_keeps_every_zero_when_zeros_are_scarce() -> None:
    rows = np.arange(30)
    df = pd.DataFrame({"x": np.where(np.isin(rows, [4, 17]), 0.0, rows + 1.0), "y": rows % 4})

    sample = _pattern_sample(_checker(df), "MATCHED_ZERO", '"x" AND "y"')

    assert len(sample) == N_EXAMPLES
    assert set(sample.index[sample["x"] == 0]) == {4, 17}


def test_c_is_a_or_b_sample_covers_each_source_column() -> None:
    rows = np.arange(30)
    df = pd.DataFrame(
        {
            "a": rows % 5,
            "b": rows % 3,
            "c": [np.nan if i % 7 == 0 else float(i) for i in rows],
        }
    )
    same_column = np.array(["BOTH" if i % 3 == 0 else ("a" if i % 3 == 1 else "b") for i in rows])

    sample = _pattern_sample(
        _checker(df),
        "C_IS_A_OR_B",
        '"a" AND "b" AND "c"',
        display_info={"Same Column": same_column},
    )

    assert list(sample.columns) == ["a", "b", "c"]
    assert set(same_column[sample.index]) == {"BOTH", "a", "b"}
    assert sample["c"].isna().any()
    assert sample["c"].notna().any()


def test_two_pairs_sample_shows_matching_and_non_matching_first_pair() -> None:
    rows = np.arange(30)
    df = pd.DataFrame({"a": rows % 5, "b": rows % 3, "c": rows % 4, "d": rows % 6})
    first_pair_matches = rows % 4 == 0

    sample = _pattern_sample(
        _checker(df),
        "TWO_PAIRS",
        '"a" AND "b" AND "c" AND "d"',
        display_info={"match_1_2_arr": first_pair_matches},
    )

    assert list(sample.columns) == ["a", "b", "c", "d"]
    assert first_pair_matches[sample.index].sum() == N_EXAMPLES // 2
    assert (~first_pair_matches[sample.index]).sum() == N_EXAMPLES // 2


def test_generic_sample_includes_each_rare_value_of_last_column() -> None:
    rows = np.arange(30)
    df = pd.DataFrame({"key": rows % 5, "rare": [np.nan] * 27 + [1.5, 2.5, 3.5]})

    sample = _pattern_sample(_checker(df), "RARE_VALUES", '"key" AND "rare"')

    assert {27, 28, 29} <= set(sample.index)
    assert sample["rare"].isna().any()
    assert sample.index.is_unique
    assert len(sample) <= N_EXAMPLES


def test_generic_sample_prefers_rows_with_populated_values() -> None:
    rows = np.arange(30)
    df = pd.DataFrame({"partial": [float(i + 1) if i < 6 else np.nan for i in rows], "key": rows % 5})

    sample = _pattern_sample(_checker(df), "RARE_VALUES", '"partial" AND "key"')

    assert set(range(6)) <= set(sample.index)
    assert sample.index.is_unique
    assert len(sample) <= N_EXAMPLES


def test_generic_sample_spreads_rows_across_distinct_values() -> None:
    rows = np.arange(30)
    df = pd.DataFrame({"key": rows % 5, "label": [f"m{i % 12}" for i in rows]})

    sample = _pattern_sample(_checker(df), "RARE_VALUES", '"key" AND "label"')

    assert len(sample) == N_EXAMPLES
    assert sample["label"].nunique() == N_EXAMPLES


def test_unique_values_sample_returns_requested_number_of_rows() -> None:
    rows = np.arange(30)
    df = pd.DataFrame({"key": rows % 5, "label": [f"id{i}" for i in rows]})

    sample = _pattern_sample(_checker(df), "UNIQUE_VALUES", "label")

    assert list(sample.columns) == ["label"]
    assert len(sample) == N_EXAMPLES
    assert sample.index.is_unique


@pytest.fixture
def ordered_checker() -> DataConsistencyChecker:
    rows = np.arange(30)
    # (i * 7) % 30 is a permutation of 0..29, so sorting it yields consecutive integers.
    return _checker(pd.DataFrame({"seq": (rows * 7) % 30, "key": rows % 4}))


def test_consecutive_sample_returns_adjacent_rows(ordered_checker) -> None:
    sample = _pattern_sample(ordered_checker, "COLUMN_ORDERED_ASC", '"seq" AND "key"', show_consecutive=True)

    assert len(sample) == N_EXAMPLES
    assert (np.diff(sample.index.to_numpy()) == 1).all()


def test_consecutive_sample_follows_sort_column(ordered_checker) -> None:
    sample = _pattern_sample(
        ordered_checker,
        "GROUPED_STRINGS_BY_NUMERIC",
        '"seq" AND "key"',
        show_consecutive=True,
        sort_col="seq",
    )

    assert len(sample) == N_EXAMPLES
    assert (np.diff(sample["seq"].to_numpy()) == 1).all()


def test_exception_sample_excludes_flagged_rows() -> None:
    df = pd.DataFrame({"v": np.arange(12) + 1.0, "w": np.arange(12) % 3})
    checker = _checker(df)
    result_col = checker.get_results_col_name("RARE_VALUES", "v")
    checker.test_results_df = pd.DataFrame({result_col: [i % 2 == 0 for i in range(12)]})
    checker.col_to_original_cols_dict[result_col] = ["v"]

    sample = checker._get_sample_not_flagged("RARE_VALUES", "v", is_patterns=False)

    assert list(sample.columns) == ["v"]
    assert len(sample) > 0
    assert all(i % 2 == 1 for i in sample.index)


@pytest.fixture
def positive_run() -> tuple[DataConsistencyChecker, object]:
    rows = np.arange(200)
    df = pd.DataFrame({"amount": rows + 1.0, "other": rows % 7})
    df.loc[3, "amount"] = -5.0
    checker = _checker(df)
    checker.check_data_quality(execute_list=["POSITIVE"])
    summary = checker.exceptions_summary_df
    assert len(summary) == 1
    return checker, summary.iloc[0]["Display Information"]


def test_exception_sample_from_real_run_omits_flagged_row(positive_run) -> None:
    checker, display_info = positive_run

    sample = checker._get_sample_not_flagged("POSITIVE", "amount", is_patterns=False, display_info=display_info)

    assert len(sample) == N_EXAMPLES
    assert 3 not in sample.index
    assert (sample["amount"] > 0).all()


def test_display_examples_not_flagged_writes_sample_table(positive_run) -> None:
    checker, display_info = positive_run
    output = StringIO()

    checker._display_examples_not_flagged("POSITIVE", ["amount"], "amount", False, display_info, output)

    html = output.getvalue()
    assert "Examples of values NOT flagged" in html
    assert "<table" in html
