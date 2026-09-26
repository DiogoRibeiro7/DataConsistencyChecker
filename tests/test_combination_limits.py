"""Checks over pairs or triples of columns skip themselves when there are too many combinations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_consistency_checker import DataConsistencyChecker


def _mixed_frame() -> pd.DataFrame:
    rows = np.arange(60)
    return pd.DataFrame(
        {
            **{f"num{i}": (rows * (i + 2)) % 17 + i + 1.0 for i in range(4)},
            **{f"txt{i}": [f"w{(r * (i + 3)) % 11} v{r % (i + 2)}" for r in rows] for i in range(4)},
            **{f"bin{i}": np.where((rows // (i + 1)) % 2 == 0, "y", "n") for i in range(4)},
            **{f"day{i}": pd.date_range("2021-01-01", periods=60, freq="D") + pd.Timedelta(days=i * 3) for i in range(3)},
        }
    )


def test_combination_checks_report_skipping_when_max_combinations_is_exceeded(capsys) -> None:
    checker = DataConsistencyChecker(verbose=1, max_combinations=1)
    checker.init_data(_mixed_frame())

    checker.check_data_quality()

    skips = [line for line in capsys.readouterr().out.splitlines() if "Skipping" in line]
    assert checker.get_execution_failures() == []
    assert len(skips) >= 50
    for kind in ("numeric", "string", "binary"):
        assert any(f"pairs of {kind} columns" in line for line in skips), kind
    assert any("triples of numeric columns" in line for line in skips)
    # Checks on single columns still run
    assert len(checker.patterns_df) > 0
