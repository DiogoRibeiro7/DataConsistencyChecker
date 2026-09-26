"""The package must behave the same on pandas 2 and pandas 3.

pandas 3 changed defaults the checks were written for: text is stored in a string dtype that keeps missing values
missing, datetimes default to microseconds, assignments no longer upcast, and logical operations with plain lists
raise. These tests pin the pandas 2 behaviour the package now provides explicitly, so they pass on both versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker.checker_utils import as_str, column_from_values, map_elements, normalise_dtypes

ALL_TEST_IDS = DataConsistencyChecker(verbose=-1).get_test_list()


def test_as_str_converts_missing_values_to_strings() -> None:
    values = pd.Series(["a", None, np.nan, 1.5], index=[3, 5, 7, 9], name="col", dtype=object)

    result = as_str(values)

    assert result.tolist() == ["a", "None", "nan", "1.5"]
    assert result.dtype == object
    assert result.index.tolist() == [3, 5, 7, 9]
    assert result.name == "col"


def test_as_str_formats_dates_as_astype_did() -> None:
    dates = pd.Series(pd.to_datetime(["2020-01-02", None]))

    assert as_str(dates).tolist() == ["2020-01-02", "NaT"]


def test_map_elements_applies_the_function_to_each_cell() -> None:
    df = pd.DataFrame({"a": [0, 1], "b": [-2, 0]})

    assert map_elements(df, lambda x: x == 0).to_numpy().tolist() == [[True, False], [False, True]]


def test_normalise_dtypes_uses_object_text_and_nanosecond_dates() -> None:
    df = pd.DataFrame({
        "text": pd.array(["x", None], dtype="string"),
        "when": pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"])).astype("datetime64[s]"),
        "num": [1.5, 2.5],
    })

    result = normalise_dtypes(df)

    assert result["text"].dtype == object
    assert result["text"].iloc[0] == "x"
    assert isinstance(result["text"].iloc[1], float) and np.isnan(result["text"].iloc[1])
    assert result["when"].dtype == "datetime64[ns]"
    assert result["when"].tolist() == df["when"].tolist()
    assert result["num"].dtype == "float64"
    assert df["text"].dtype == "string"  # the input is left unchanged


def test_normalise_dtypes_handles_duplicate_column_names() -> None:
    df = pd.DataFrame([["x", "y"], [None, "z"]], columns=["a", "a"]).astype("string")

    assert list(normalise_dtypes(df).dtypes) == [object, object]


def test_column_from_values_keeps_none_in_text_columns() -> None:
    index = pd.RangeIndex(3)

    text = column_from_values(["a", None, "b"], index)

    assert text.dtype == object
    assert text.iloc[1] is None
    assert column_from_values([1, 2, 3], index).dtype == "int64"
    assert column_from_values([1.5, None, 2.0], index).dtype == "float64"


def test_init_data_accepts_string_dtype_and_second_resolution_dates() -> None:
    num_rows = 50
    df = pd.DataFrame({
        "code": pd.array([f"id-{i % 7}" if i % 10 else None for i in range(num_rows)], dtype="string"),
        "when": pd.Series(pd.date_range("2020-01-01", periods=num_rows, freq="D")).astype("datetime64[s]"),
    })

    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)

    assert checker.string_cols == ["code"]
    assert checker.date_cols == ["when"]
    assert checker.orig_df["code"].dtype == object
    assert checker.orig_df["when"].dtype == "datetime64[ns]"


@pytest.mark.parametrize("add_nones", ["one-row", "in-sync", "random", "80-percent"])
def test_synthetic_nulls_can_be_added_to_boolean_columns(add_nones: str) -> None:
    synth = DataConsistencyChecker(verbose=-1).generate_synth_data(
        execute_list=["BINARY_MATCHES_VALUES"], add_nones=add_nones
    )

    column = synth["bin_match_val all"]
    assert column.dtype == object
    assert column.isna().any()
    assert set(column.dropna()) <= {True, False}


@pytest.mark.parametrize("test_id", ALL_TEST_IDS)
def test_check_runs_on_synthetic_data_with_missing_values(test_id: str) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    synth = checker.generate_synth_data(execute_list=[test_id], add_nones="random")
    checker.init_data(synth)

    checker.check_data_quality(execute_list=[test_id], raise_on_error=True)

    assert checker.get_execution_failures() == []
