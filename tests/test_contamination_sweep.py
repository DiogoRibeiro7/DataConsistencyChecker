"""Changing the contamination level between runs, and the test_contamination_level() sweep."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker

CHECKS = ["SIMILAR_CHARACTERS", "MEAN_OF_COLUMNS", "NEGATIVE"]


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    return DataConsistencyChecker(verbose=-1).generate_synth_data(execute_list=CHECKS)


def _checker(df: pd.DataFrame) -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    return checker


def _findings(checker: DataConsistencyChecker) -> list[str | None]:
    return [None if df is None else df.drop(columns=["Display Information"]).to_csv()
            for df in (checker.patterns_df, checker.exceptions_summary_df)]


def _counts(checker: DataConsistencyChecker) -> tuple[int, int, int]:
    scores = checker.test_results_by_column_np
    issues = 0 if checker.exceptions_summary_df is None else len(checker.exceptions_summary_df)
    return issues, int((scores.sum(axis=1) > 0).sum()), int((scores.sum(axis=0) > 0).sum())


def test_changing_the_level_between_runs_gives_the_same_results_as_a_new_checker(synth) -> None:
    # Cached analyses computed at the first level used to be reused at the second one.
    reused = _checker(synth)
    reused.check_data_quality(execute_list=CHECKS, freq_contamination_level=1)
    reused.check_data_quality(execute_list=CHECKS, freq_contamination_level=0.05)
    fresh = _checker(synth)
    fresh.check_data_quality(execute_list=CHECKS, freq_contamination_level=0.05)

    assert _findings(reused) == _findings(fresh)


def test_sweep_counts_what_a_run_at_each_level_finds(synth) -> None:
    levels = [0.0001, 2, 0.01, 0.05]  # 0.0001 of 1,000 rows is less than a row, so it counts as one row

    counts = _checker(synth).test_contamination_level(levels, execute_list=CHECKS, plot=False)

    expected = []
    for level in [1, 2, 0.01, 0.05]:
        checker = _checker(synth)
        checker.check_data_quality(execute_list=CHECKS, freq_contamination_level=level)
        expected.append(_counts(checker))
    assert [tuple(values) for values in zip(*counts)] == expected
    assert counts[0][-1] > 0


def test_sweep_keeps_the_checkers_own_results(synth) -> None:
    checker = _checker(synth)
    checker.check_data_quality(execute_list=CHECKS)
    before = _findings(checker), checker.freq_contamination_level

    checker.test_contamination_level([1, 0.05], execute_list=CHECKS, plot=False)

    assert (_findings(checker), checker.freq_contamination_level) == before


def test_sweep_plots_the_three_counts(synth, monkeypatch) -> None:
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(len(plt.gcf().axes)))

    _checker(synth).test_contamination_level([1, 0.05], execute_list=["NEGATIVE"])
    plt.close("all")

    assert shown == [3]


@pytest.mark.parametrize("level", [0, 0.0, 1.5, 5.0, -2, 1_001])
def test_sweep_rejects_invalid_levels(synth, level) -> None:
    with pytest.raises(ValueError, match="Invalid contamination level"):
        _checker(synth).test_contamination_level([level], plot=False)


def test_sweep_needs_data() -> None:
    with pytest.raises(ValueError, match="init_data"):
        DataConsistencyChecker(verbose=-1).test_contamination_level(plot=False)
