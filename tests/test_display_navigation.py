"""Coverage tests for display navigation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

import data_consistency_checker.display_mixin as display_module

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker.test_registry import TestDefinition


def _checker() -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame({"a": [10, 20, 30]})
    checker.test_results_df = pd.DataFrame({"FINAL SCORE": [0, 1, 2]})
    checker.test_results_by_column_np = pd.DataFrame([[0], [1], [1]]).to_numpy()
    return checker


def _definition() -> TestDefinition:
    return TestDefinition(
        short_description="Example",
        description="Example description",
        test_func=lambda **_: None,
        gen_func=None,
        shortlist=False,
        implemented=True,
        fast=True,
        code=False,
    )


def test_display_next_handles_exhausted_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = _checker()
    checker.found_tests = []
    checker.current_display_test = 0

    checker.display_next()

    assert "No further test results" in capsys.readouterr().out


def test_display_next_advances_to_next_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    checker.found_tests = ["TEST"]
    checker.current_display_test = 0
    checker.test_dict = {"TEST": _definition()}

    calls: list[list[str]] = []
    monkeypatch.setattr(
        checker,
        "display_detailed_results",
        lambda test_id_list, max_shown: calls.append(test_id_list),
    )

    checker.display_next()

    assert calls == [["TEST"]]
    assert checker.current_display_test == 1


def test_display_least_flagged_rows_without_result_breakdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = _checker()

    checker.display_least_flagged_rows(with_results=False, n_rows=2)

    output = capsys.readouterr().out
    assert "FINAL SCORE" in output
    assert "10" in output


def test_display_most_flagged_rows_returns_without_results() -> None:
    checker = _checker()
    checker.test_results_df = None

    assert checker.display_most_flagged_rows() is None


def test_display_next_notebook_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    checker.found_tests = ["TEST"]
    checker.current_display_test = 0
    checker.test_dict = {"TEST": _definition()}

    displayed: list[object] = []
    monkeypatch.setattr(display_module, "is_notebook", lambda: True)
    monkeypatch.setattr(display_module, "display", displayed.append)
    monkeypatch.setattr(
        checker,
        "display_detailed_results",
        lambda test_id_list, max_shown: None,
    )

    checker.display_next()

    assert len(displayed) == 1
    assert checker.current_display_test == 1


def test_display_least_flagged_rows_with_result_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    calls: list[tuple[list[int], int]] = []

    monkeypatch.setattr(
        checker,
        "_display_rows_with_tests",
        lambda sorted_df, n_rows: calls.append(
            (sorted_df["FINAL SCORE"].tolist(), n_rows)
        ),
    )

    checker.display_least_flagged_rows(with_results=True, n_rows=2)

    assert calls == [([0, 1, 2], 2)]


def test_display_least_flagged_rows_notebook_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    displayed: list[object] = []
    monkeypatch.setattr(display_module, "is_notebook", lambda: True)
    monkeypatch.setattr(display_module, "display", displayed.append)

    checker.display_least_flagged_rows(with_results=False, n_rows=2)

    assert len(displayed) == 1


def test_display_most_flagged_rows_with_result_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    calls: list[tuple[list[int], int, bool]] = []

    monkeypatch.setattr(
        checker,
        "_display_rows_with_tests",
        lambda sorted_df, n_rows, check_score=False: calls.append(
            (sorted_df["FINAL SCORE"].tolist(), n_rows, check_score)
        ),
    )

    checker.display_most_flagged_rows(with_results=True, n_rows=2)

    assert calls == [([2, 1, 0], 2, True)]


def test_display_most_flagged_rows_without_result_breakdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = _checker()

    checker.display_most_flagged_rows(with_results=False, n_rows=2)

    output = capsys.readouterr().out
    assert "FINAL SCORE" in output
    assert "2" in output


def test_display_most_flagged_rows_notebook_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker()
    displayed: list[object] = []
    monkeypatch.setattr(display_module, "is_notebook", lambda: True)
    monkeypatch.setattr(display_module, "display", displayed.append)

    checker.display_most_flagged_rows(with_results=False, n_rows=2)

    assert len(displayed) == 1
