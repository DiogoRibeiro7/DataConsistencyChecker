"""Coverage tests for relocated helper methods."""

from __future__ import annotations

import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker.test_registry import TestDefinition


def _definition(short: str, long: str) -> TestDefinition:
    return TestDefinition(
        short_description=short,
        description=long,
        test_func=lambda **_: None,
        gen_func=None,
        shortlist=False,
        implemented=True,
        fast=True,
        code=False,
    )


def test_decision_tree_rules_are_rendered_as_categories() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.orig_df = pd.DataFrame({"color": ["red", "blue"]})

    rules = "color_red <= 0.50\ncolor_blue > 0.50"
    rendered = checker.get_decision_tree_rules_as_categories(rules, ["color"])

    assert "color is not red" in rendered
    assert "color is blue" in rendered


def test_output_current_test_supports_quiet_and_verbose_modes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = DataConsistencyChecker(verbose=0)
    checker.test_dict = {
        "TEST": _definition("", "A longer test description."),
    }

    checker._output_current_test(1, "TEST")
    assert capsys.readouterr().out == ""

    checker.verbose = 1
    checker._output_current_test(2, "TEST")
    assert "Executing test   2: TEST" in capsys.readouterr().out

    checker.verbose = 2
    checker._output_current_test(3, "TEST")
    output = capsys.readouterr().out
    assert "Executing test   3: TEST" in output
    assert "A longer test description" in output


def test_add_synthetic_column_preserves_existing_columns() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.synth_df = pd.DataFrame({"existing": [1, 2]})

    checker._add_synthetic_column("new", [3, 4])

    assert checker.synth_df.to_dict(orient="list") == {
        "existing": [1, 2],
        "new": [3, 4],
    }
