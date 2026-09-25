"""Coverage tests for score-relationship plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker


def _base_checker() -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame(
        {
            "num": [float(x) for x in range(12)],
            "cat": ["a", "b", "c"] * 4,
        }
    )
    checker.num_rows = len(checker.orig_df)
    checker.test_results_df = pd.DataFrame(
        {"FINAL SCORE": [0, 1, 0, 2, 1, 0, 3, 0, 1, 2, 0, 1]}
    )
    checker.exceptions_summary_df = pd.DataFrame(
        [{"Test ID": "TEST", "Number of Exceptions": 1}]
    )
    checker.numeric_cols = ["num"]
    checker.date_cols = []
    checker.binary_cols = []
    checker.string_cols = ["cat"]
    return checker


def test_plot_columns_vs_final_scores_handles_no_exceptions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = _base_checker()
    checker.exceptions_summary_df = pd.DataFrame()

    checker.plot_columns_vs_final_scores()

    assert "No exceptions found." in capsys.readouterr().out


def test_plot_columns_vs_final_scores_covers_numeric_and_categorical_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _base_checker()
    shown = {"count": 0}
    monkeypatch.setattr(
        plt,
        "show",
        lambda: shown.__setitem__("count", shown["count"] + 1),
    )

    checker.plot_columns_vs_final_scores()

    assert shown["count"] == 2


def test_plot_columns_vs_final_scores_handles_multirow_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    data = {f"num_{idx}": [float(x + idx) for x in range(12)] for idx in range(5)}
    data.update({f"cat_{idx}": ["a", "b", "c"] * 4 for idx in range(5)})
    checker.orig_df = pd.DataFrame(data)
    checker.num_rows = len(checker.orig_df)
    checker.test_results_df = pd.DataFrame(
        {"FINAL SCORE": [0, 1, 0, 2, 1, 0, 3, 0, 1, 2, 0, 1]}
    )
    checker.exceptions_summary_df = pd.DataFrame(
        [{"Test ID": "TEST", "Number of Exceptions": 1}]
    )
    checker.numeric_cols = [f"num_{idx}" for idx in range(5)]
    checker.date_cols = []
    checker.binary_cols = []
    checker.string_cols = [f"cat_{idx}" for idx in range(5)]

    shown = {"count": 0}
    monkeypatch.setattr(
        plt,
        "show",
        lambda: shown.__setitem__("count", shown["count"] + 1),
    )

    checker.plot_columns_vs_final_scores()

    assert shown["count"] == 2


def test_plot_columns_vs_final_scores_handles_high_cardinality_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    categories = ["common"] * 14 + [f"rare_{idx}" for idx in range(10)]
    checker.orig_df = pd.DataFrame(
        {
            "cat": categories,
        }
    )
    checker.num_rows = len(checker.orig_df)
    checker.test_results_df = pd.DataFrame(
        {"FINAL SCORE": [idx % 4 for idx in range(checker.num_rows)]}
    )
    checker.exceptions_summary_df = pd.DataFrame(
        [{"Test ID": "TEST", "Number of Exceptions": 1}]
    )
    checker.numeric_cols = []
    checker.date_cols = []
    checker.binary_cols = []
    checker.string_cols = ["cat"]

    shown = {"count": 0}
    monkeypatch.setattr(
        plt,
        "show",
        lambda: shown.__setitem__("count", shown["count"] + 1),
    )

    checker.plot_columns_vs_final_scores()

    assert shown["count"] == 1
