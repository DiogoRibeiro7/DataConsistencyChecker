"""Coverage tests for reporting and plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker


def _checker() -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(
        pd.DataFrame(
            {
                "a": list(range(12)),
                "b": list(range(12)),
                "c": [x * 2 for x in range(12)],
            }
        )
    )
    return checker


def test_plot_final_scores_by_row_returns_without_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    checker.test_results_df = None
    called = {"show": False}
    monkeypatch.setattr(plt, "show", lambda: called.__setitem__("show", True))

    checker.plot_final_scores_distribution_by_row()

    assert called["show"] is False


def test_plot_final_scores_by_test_handles_no_exceptions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = _checker()
    checker.exceptions_summary_df = pd.DataFrame(
        columns=["Test ID", "Number of Exceptions"]
    )

    checker.plot_final_scores_distribution_by_test()

    assert "No exceptions found" in capsys.readouterr().out


def test_feature_pair_quality_runs_on_numeric_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    shown = {"count": 0}
    monkeypatch.setattr(plt, "show", lambda: shown.__setitem__("count", shown["count"] + 1))

    checker.check_data_quality_by_feature_pairs(max_features_shown=2)

    assert shown["count"] > 0


def test_quick_report_calls_summary_and_plot_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()

    frame = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(checker, "get_patterns_list", lambda: frame)
    monkeypatch.setattr(checker, "summarize_patterns_by_test_and_feature", lambda: frame)
    monkeypatch.setattr(checker, "get_exceptions_list", lambda: frame)
    monkeypatch.setattr(checker, "summarize_exceptions_by_test_and_feature", lambda: frame)
    monkeypatch.setattr(checker, "summarize_exceptions_by_test", lambda: frame)
    monkeypatch.setattr(checker, "summarize_patterns_and_exceptions", lambda: frame)

    calls: list[str] = []
    monkeypatch.setattr(checker, "plot_final_scores_distribution_by_row", lambda: calls.append("row"))
    monkeypatch.setattr(checker, "plot_final_scores_distribution_by_feature", lambda: calls.append("feature"))
    monkeypatch.setattr(checker, "plot_final_scores_distribution_by_test", lambda: calls.append("test"))

    checker.quick_report()

    assert calls == ["row", "feature", "test"]
