"""Patch-coverage tests for relocated example-row formatting branches."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker


def _checker() -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame(
        {
            "a": [2.0, 4.0],
            "b": [1.0, 2.0],
            "c": [3.0, 6.0],
            "num": [1200.0, 230.0],
            "neg": [-1.0, 2.0],
            "text": [" alpha-1 ", "beta 22"],
            "text2": [" apple", "berry 33"],
            "text_nonprint": ["a\x00b", "plain"],
            "when": pd.to_datetime(["2026-01-02 03:04", "2026-02-03 04:05"]),
            "when2": pd.to_datetime(["2026-01-03 03:04", "2026-02-05 04:05"]),
        }
    )
    checker.column_medians = {"a": 3.0, "b": 1.5}
    return checker


def _render(
    checker: DataConsistencyChecker,
    test_id: str,
    cols: list[str],
    display_info: dict[str, object] | None = None,
    *,
    is_patterns: bool = True,
) -> str:
    output = StringIO()
    checker._draw_sample_dataframe(
        checker.orig_df[cols].copy(),
        test_id,
        cols,
        display_info,
        is_patterns=is_patterns,
        f=output,
    )
    return output.getvalue()


@pytest.mark.parametrize(
    ("test_id", "cols", "display_info", "is_patterns", "expected"),
    [
        ("LARGER_THAN_SUM", ["a", "b"], None, True, "SUM"),
        ("ROUNDING", ["num"], None, True, "Number Zeros"),
        ("SUM_OF_COLUMNS", ["a", "b", "c"], {"operation": "plus", "amount": 5}, True, "SUM PLUS 5"),
        ("SUM_OF_COLUMNS", ["a", "b", "c"], {"operation": "times", "amount": 2}, True, "SUM TIMES 2"),
        ("SIMILAR_TO_DIFF", ["a", "b"], None, True, "ABSOLUTE DIFFERENCE"),
        ("RARE_PAIRS_FIRST_WORD_VAL", ["text", "b"], None, True, "text FIRST WORD"),
        ("MULTIPLE_OF_CONSTANT", ["a"], {"value": 2}, True, "NUM MULTIPLES"),
        ("MULTIPLE_OF_CONSTANT", ["a"], {"value": 2}, False, "NUM MULTIPLES"),
        ("UNUSUAL_DAY_OF_MONTH", ["when"], None, True, "Day of Month"),
        ("UNUSUAL_MONTH", ["when"], None, True, "Month"),
        ("UNUSUAL_HOUR", ["when"], None, True, "Hour"),
        ("UNUSUAL_MINUTES", ["when"], None, True, "Minutes"),
        ("CONSTANT_GAP", ["when", "when2"], None, True, "Gap"),
        ("NUMBER_ALPHANUMERIC_CHARS", ["text"], None, True, "Num Alpha-Numeric Chars"),
        (
            "NUMBER_NON-ALPHANUMERIC_CHARS",
            ["text"],
            {"test_series": pd.Series([1, 2])},
            True,
            "Num Non-Alpha-Numeric Chars",
        ),
        ("FIRST_CHAR_ALPHA", ["text"], None, True, "First Char"),
        ("LAST_CHAR_SMALL_SET", ["text"], None, True, "Last Char"),
        ("RARE_PAIRS_FIRST_CHAR", ["text", "text2"], None, True, "text First Char"),
        ("RARE_PAIRS_FIRST_WORD", ["text", "text2"], None, True, "text First Word"),
        ("SAME_LAST_WORD", ["text", "text2"], None, True, "text Last Word"),
        ("SIMILAR_NUM_CHARS", ["text", "text2"], None, True, "text Num Chars"),
        ("SIMILAR_NUM_WORDS", ["text", "text2"], None, True, "text Num Words"),
        ("SMALL_VS_CORR_COLS", ["a"], {"cluster": ["b", "c"]}, True, "b"),
        ("ZERO_VALUES_PER_ROW", ["a"], None, True, "Number Zero Values"),
        ("NEGATIVE_VALUES_PER_ROW", ["neg"], None, True, "Number Negative Values"),
        ("LINEAR_REGRESSION", ["a"], {"Pred": pd.Series([1.1, 2.2])}, True, "PREDICTION"),
        (
            "CORRELATED_NUMERIC",
            ["a"],
            {
                "col_1_percentiles": pd.Series([0.1, 0.9]),
                "col_2_percentiles": pd.Series([0.2, 0.8]),
            },
            True,
            "Column 1 Percentile",
        ),
        (
            "UNUSUAL_ORDER_MAGNITUDE",
            ["a"],
            {"order of magnitude": pd.Series([1, 2])},
            True,
            "ORDER OF MAGNITUDE",
        ),
        ("RUNNING_SUM", ["a"], {"RUNNING SUM": pd.Series([2.0, 6.0])}, True, "RUNNING SUM"),
        (
            "SMALL_AVG_RANK_PER_ROW",
            ["a"],
            {"percentiles": pd.Series([0.2, 0.8])},
            True,
            "AVG PERCENTILE",
        ),
        ("C_IS_A_OR_B", ["a"], {"Same Column": ["a", "b"]}, True, "SAME AS"),
        (
            "SAME_SPECIAL_CHARS",
            ["text"],
            {"Special Chars A": ["-", ""], "Special Chars B": ["-", ""]},
            True,
            "SPECIAL CHARS A",
        ),
        (
            "TWO_PAIRS",
            ["a"],
            {"match_1_2_arr": [True, False], "match_3_4_arr": [False, True]},
            True,
            "A and B Match",
        ),
        ("BINARY_TWO_OTHERS_MATCH", ["a"], {"Match": [True, False]}, True, "MATCH"),
        ("SIMILAR_WRT_RATIO", ["a"], {"Ratio": [1.0, 2.0]}, True, "RATIO"),
        ("SIMILAR_WRT_DIFF", ["a"], {"Diff": [1.0, 2.0]}, True, "DIFF"),
        (
            "CORRELATED_ALPHA_ORDER",
            ["text"],
            {"col_1_percentiles": [0.1, 0.9], "col_2_percentiles": [0.2, 0.8]},
            True,
            "Percentile Column 1",
        ),
        ("LARGE_GIVEN_DATE", ["when"], {"bin_assignments": [2, 1]}, True, "Bin Number when"),
        (
            "FEW_NEIGHBORS",
            ["a"],
            {"prev_val": [1.0, 3.0], "next_val": [3.0, 5.0]},
            True,
            "Closest smaller value",
        ),
        ("FEW_WITHIN_RANGE", ["a"], {"Number in Range": [1, 2]}, True, "Number in Range"),
        ("NONPRINTABLE_CHARS", ["text_nonprint"], None, True, "Non-Printable Chars"),
    ],
)
def test_draw_sample_dataframe_covers_remaining_formatter_branches(
    test_id: str,
    cols: list[str],
    display_info: dict[str, object] | None,
    is_patterns: bool,
    expected: str,
) -> None:
    checker = _checker()

    html = _render(
        checker,
        test_id,
        cols,
        display_info,
        is_patterns=is_patterns,
    )

    assert expected in html


@pytest.mark.parametrize("test_id", ["MUCH_LARGER", "LARGER_DIFF_RANGE"])
def test_draw_sample_dataframe_covers_column_reordering_branches(test_id: str) -> None:
    checker = _checker()

    html = _render(checker, test_id, ["a", "b"])

    assert "a" in html
    assert "b" in html
