"""The checker leaves the caller's global state alone: warning filters, pandas options and random generators."""

from __future__ import annotations

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker.checker_utils import library_call

PANDAS_OPTIONS = ["display.width", "display.max_columns", "display.max_colwidth", "display.max_rows",
                  "display.float_format"]
# These checks seed the global generators so their models give reproducible results.
RESEEDING_CHECKS = ["PREV_VALUES_DT", "DECISION_TREE_CLASSIFIER", "DECISION_TREE_REGRESSOR", "LINEAR_REGRESSION"]


def _global_state() -> tuple:
    np_state = np.random.get_state()
    return (
        list(warnings.filters),
        {name: pd.get_option(name) for name in PANDAS_OPTIONS},
        random.getstate(),
        (np_state[0], np_state[1].tolist(), *np_state[2:]),
    )


def test_checker_leaves_the_callers_global_state_alone(monkeypatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    random.seed(123)
    np.random.seed(123)
    before = _global_state()

    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(checker.generate_synth_data(execute_list=RESEEDING_CHECKS + ["LARGE_GIVEN_VALUE"]))
    checker.check_data_quality(execute_list=RESEEDING_CHECKS + ["LARGE_GIVEN_VALUE"])
    assert _global_state() == before

    # Displaying results draws random example rows, so it may advance the random generators, like any sampling.
    checker.display_detailed_results(show_short_list_only=False)
    after = _global_state()
    assert after[0] == before[0], "warning filters changed"
    assert after[1] == before[1], "pandas options changed"


def test_library_call_hides_routine_warnings_only_during_the_call() -> None:
    @library_call
    def noisy() -> str:
        warnings.warn("a routine deprecation", FutureWarning, stacklevel=1)
        return "done"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert noisy() == "done"
        assert caught == []

        warnings.warn("the caller's own deprecation", FutureWarning, stacklevel=1)
        assert [str(w.message) for w in caught] == ["the caller's own deprecation"]


def test_library_call_restores_the_callers_state_when_the_call_fails() -> None:
    @library_call
    def failing() -> None:
        raise ValueError("boom")

    filters = list(warnings.filters)
    with pytest.raises(ValueError, match="boom"):
        failing()
    assert list(warnings.filters) == filters
