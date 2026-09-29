"""Every check with a dedicated plot can display its findings, with plots, on its synthetic data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from data_consistency_checker import DataConsistencyChecker

PLOTTED_CHECKS = [
    "BINARY_RARE_COMBINATION",
    "CORRELATED_GIVEN_VALUE",
    "LARGE_GIVEN_DATE",
    "LARGE_GIVEN_PAIR",
    "LARGE_GIVEN_PREFIX",
    "LARGE_GIVEN_VALUE",
    "LARGER_DIFF_RANGE",
    "LARGER_SAME_RANGE",
    "MUCH_LARGER",
    "RARE_PAIRS_FIRST_CHAR",
    "RARE_PAIRS_FIRST_WORD",
    "SMALL_GIVEN_DATE",
    "SMALL_GIVEN_PAIR",
    "SMALL_GIVEN_PREFIX",
    "SMALL_GIVEN_VALUE",
]
IMPLEMENTED = set(DataConsistencyChecker(verbose=-1).get_test_list())


@pytest.mark.parametrize(
    "test_id",
    [t for t in PLOTTED_CHECKS if t in IMPLEMENTED],
)
def test_findings_are_displayed_with_plots(test_id, monkeypatch) -> None:
    shown: list[int] = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(len(plt.get_fignums())))
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(checker.generate_synth_data(all_cols=False, execute_list=[test_id]))
    checker.check_data_quality(execute_list=[test_id])
    assert len(checker.patterns_df) + len(checker.exceptions_summary_df) > 0

    try:
        checker.display_detailed_results(test_id_list=[test_id], show_short_list_only=False, include_examples=False)
    finally:
        plt.close("all")

    assert shown, "no plot was drawn"
