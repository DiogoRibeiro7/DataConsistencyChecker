"""Edge-case coverage for dataset initialization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker


def test_init_data_rejects_too_small_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = DataConsistencyChecker(verbose=-1)

    checker.init_data(pd.DataFrame({"a": [1, 2, 3]}))

    assert "too few rows" in capsys.readouterr().out
    assert checker.num_rows == -1


def test_init_data_renames_duplicate_columns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    df = pd.DataFrame(
        [[i, i + 1] for i in range(12)],
        columns=["value", "value"],
    )

    checker.init_data(df)

    assert "Duplicate column names encountered" in capsys.readouterr().out
    assert checker.orig_df.columns.tolist() == ["value_0", "value_1"]


def test_init_data_removes_constant_columns() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    df = pd.DataFrame(
        {
            "constant": [1] * 12,
            "varying": list(range(12)),
        }
    )

    checker.init_data(df)

    assert "constant" not in checker.orig_df.columns
    assert "varying" in checker.orig_df.columns


def test_init_data_respects_known_date_columns() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    df = pd.DataFrame(
        {
            "when": [f"2026-09-{day:02d}" for day in range(1, 13)],
            "value": list(range(12)),
        }
    )

    checker.init_data(df, known_date_cols=["when"])

    assert "when" in checker.date_cols
    assert "when" not in checker.string_cols


def test_a_mostly_missing_column_of_text_is_a_string_column() -> None:
    # Its few values are all text, so few non-numeric values does not make it numeric.
    values = pd.Categorical([None] * 895 + ["Y"] * 3, categories=["Y"])
    df = pd.DataFrame({"flag": values, "x": np.arange(898.0), "y": np.arange(898.0) % 7})
    checker = DataConsistencyChecker(verbose=-1)

    checker.init_data(df)

    assert "flag" in checker.string_cols


def test_a_categorical_numeric_column_with_a_few_other_values_loads() -> None:
    values = pd.Categorical([str(i % 50) for i in range(999)] + ["x"])
    checker = DataConsistencyChecker(verbose=-1)

    checker.init_data(pd.DataFrame({"code": values, "y": np.arange(1000.0)}))

    assert "code" in checker.numeric_cols
    assert checker.orig_df["code"].tolist()[-1] == "x"
