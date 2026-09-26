"""Coverage tests for display orchestration helpers."""

from __future__ import annotations

import pandas as pd

from data_consistency_checker import DataConsistencyChecker


def test_display_examples_not_flagged_skips_unsupported_pattern(capsys) -> None:
    checker = DataConsistencyChecker(verbose=-1)

    checker._display_examples_not_flagged(
        "UNIQUE_VALUES",
        ["a"],
        "a",
        True,
        None,
        None,
    )

    assert "Examples are not shown" in capsys.readouterr().out


def test_display_examples_not_flagged_uses_consecutive_sampling(monkeypatch) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    sampled = pd.DataFrame({"group": ["b", "a"], "value": [1, 2]})
    calls: dict[str, object] = {}

    def fake_sample(test_id, column_set, **kwargs):
        calls["test_id"] = test_id
        calls["column_set"] = column_set
        calls.update(kwargs)
        return sampled

    def fake_draw(df, test_id, cols, display_info, is_patterns, f):  # noqa: ARG001 - mirrors _draw_sample_dataframe
        calls["draw_df"] = df
        calls["draw_test_id"] = test_id

    monkeypatch.setattr(checker, "_get_sample_not_flagged", fake_sample)
    monkeypatch.setattr(checker, "_draw_sample_dataframe", fake_draw)

    checker._display_examples_not_flagged(
        "GROUPED_STRINGS_BY_NUMERIC",
        ["group", "value"],
        '"group" AND "value"',
        False,
        None,
        None,
    )

    assert calls["show_consecutive"] is True
    assert calls["sort_col"] == "group"
    assert calls["is_patterns"] is False
    assert calls["draw_test_id"] == "GROUPED_STRINGS_BY_NUMERIC"


def test_display_rows_with_tests_stops_at_zero_score(capsys) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame({"a": [1]})
    checker.exceptions_summary_df = pd.DataFrame(columns=["Column(s)"])
    checker.test_results_df = pd.DataFrame({"FINAL SCORE": [0]})

    checker._display_rows_with_tests(
        checker.test_results_df,
        n_rows=1,
        check_score=True,
    )

    assert "remaining rows have no flagged issues" in capsys.readouterr().out


def test_display_rows_with_tests_renders_flagged_row(capsys, monkeypatch) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame({"a": [10], "b": [20]})
    checker.exceptions_summary_df = pd.DataFrame({"Column(s)": ["a"]})
    result_col = checker.get_results_col_name("TEST_A", "a")
    checker.test_results_df = pd.DataFrame(
        {
            result_col: [1],
            "FINAL SCORE": [1],
        }
    )
    checker.col_to_original_cols_dict = {result_col: ["a"]}

    monkeypatch.setattr(checker, "get_test_list", lambda: ["TEST_A"])

    checker._display_rows_with_tests(
        checker.test_results_df,
        n_rows=1,
        check_score=True,
    )

    output = capsys.readouterr().out
    assert "Row: 0 Final Score: 1" in output
    assert "TEST_A" in output
    assert "✔" in output
