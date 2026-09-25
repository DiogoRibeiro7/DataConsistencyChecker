"""Coverage tests for example-row display helpers."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

import data_consistency_checker.display_mixin as display_module
from data_consistency_checker import DataConsistencyChecker


@pytest.mark.parametrize(
    ("test_id", "cols", "display_info", "expected_column"),
    [
        ("RARE_VALUES", ["a"], {"counts": {"x": 2, "y": 1}}, "Count of Value"),
        ("NUMBER_DECIMALS", ["num"], None, "Number decimals"),
        ("SUM_OF_COLUMNS", ["a", "b", "c"], None, "SUM"),
        ("SIMILAR_TO_PRODUCT", ["a", "b"], None, "PRODUCT"),
        ("SIMILAR_TO_RATIO", ["a", "b"], None, "DIVISION RESULTS"),
        ("MEAN_OF_COLUMNS", ["a", "b", "c"], None, "MEAN"),
        ("MIN_OF_COLUMNS", ["a", "b", "c"], None, "MIN"),
        ("MAX_OF_COLUMNS", ["a", "b", "c"], None, "MAX"),
        ("LEADING_WHITESPACE", ["text"], None, "NUM LEADING SPACES"),
        ("TRAILING_WHITESPACE", ["text"], None, "NUM TRAILING SPACES"),
        ("NUMBER_ALPHA_CHARS", ["text"], None, "Num Alpha Chars"),
        ("NUMBER_NUMERIC_CHARS", ["text"], None, "Num Numeric Chars"),
        ("NUMBER_CHARS", ["text"], None, "Num Chars"),
        ("FIRST_WORD_SMALL_SET", ["text"], None, "First Word"),
        ("LAST_WORD_SMALL_SET", ["text"], None, "Last Word"),
        ("NUMBER_WORDS", ["text"], None, "Num Words"),
        ("LONGEST_WORDS", ["text"], None, "Longest Word Len"),
        ("MISSING_VALUES_PER_ROW", ["a"], None, "Number Missing Values"),
        ("UNIQUE_VALUES_PER_ROW", ["a"], None, "Number Unique Values"),
    ],
)
def test_draw_sample_dataframe_adds_expected_explanatory_column(
    monkeypatch: pytest.MonkeyPatch,
    test_id: str,
    cols: list[str],
    display_info: dict[str, object] | None,
    expected_column: str,
) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [4, 6],
            "num": [1.25, 2.0],
            "text": ["  alpha 1", "beta 22  "],
        }
    )

    if test_id == "RARE_VALUES":
        df = pd.DataFrame({"a": ["x", "y"]})
    else:
        df = checker.orig_df[cols].copy()

    captured: list[pd.DataFrame] = []
    monkeypatch.setattr(display_module, "is_notebook", lambda: True)
    monkeypatch.setattr(
        display_module,
        "display",
        lambda value: captured.append(value.copy()),
    )

    checker._draw_sample_dataframe(
        df,
        test_id,
        cols,
        display_info,
        is_patterns=True,
        f=None,
    )

    assert expected_column in captured[0].columns


def test_draw_sample_dataframe_writes_html_output() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    df = pd.DataFrame({"a": [1, 2]})
    output = StringIO()

    checker._draw_sample_dataframe(
        df,
        "NUMBER_CHARS",
        ["a"],
        None,
        is_patterns=True,
        f=output,
    )

    html = output.getvalue()
    assert "<table" in html
    assert "Num Chars" in html


def test_draw_sample_dataframe_handles_date_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2026-01-02 03:04", "2026-02-03 04:05"]),
        }
    )
    captured: list[pd.DataFrame] = []
    monkeypatch.setattr(display_module, "is_notebook", lambda: True)
    monkeypatch.setattr(
        display_module,
        "display",
        lambda value: captured.append(value.copy()),
    )

    checker._draw_sample_dataframe(
        df,
        "UNUSUAL_DAY_OF_WEEK",
        ["when"],
        None,
        is_patterns=True,
        f=None,
    )

    assert "Day of Week" in captured[0].columns


def test_draw_rows_around_flagged_row_uses_neighborhood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame(
        {
            "a": list(range(20)),
            "b": list(range(20, 40)),
        }
    )
    checker.num_rows = len(checker.orig_df)

    captured: list[pd.DataFrame] = []
    monkeypatch.setattr(
        checker,
        "_draw_sample_dataframe",
        lambda df, *args, **kwargs: captured.append(df.copy()),
    )

    flagged = checker.orig_df.loc[[10], ["a", "b"]]
    checker._draw_rows_around_flagged_row(
        flagged,
        "TEST",
        ["a", "b"],
        display_info=None,
        f=None,
    )

    assert captured[0].index.min() == 5
    assert captured[0].index.max() == 14
