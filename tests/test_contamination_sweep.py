"""Changing the contamination level between runs."""

from __future__ import annotations

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


def test_changing_the_level_between_runs_gives_the_same_results_as_a_new_checker(synth) -> None:
    # Cached analyses computed at the first level used to be reused at the second one.
    reused = _checker(synth)
    reused.check_data_quality(execute_list=CHECKS, freq_contamination_level=1)
    reused.check_data_quality(execute_list=CHECKS, freq_contamination_level=0.05)
    fresh = _checker(synth)
    fresh.check_data_quality(execute_list=CHECKS, freq_contamination_level=0.05)

    assert _findings(reused) == _findings(fresh)
