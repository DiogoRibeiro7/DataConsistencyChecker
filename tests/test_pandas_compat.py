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


# ----------------------------------------------------------------------------------------------------------------------
# Branches that only run on particular data. Each contains code changed for pandas 3.
# ----------------------------------------------------------------------------------------------------------------------


def _findings(df: pd.DataFrame, test_id: str) -> list[str]:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    checker.check_data_quality(execute_list=[test_id], raise_on_error=True)
    found = pd.concat([checker.patterns_df, checker.exceptions_summary_df])
    return sorted(found.loc[found["Test ID"] == test_id, "Column(s)"])


@pytest.mark.parametrize("overlap", [False, True], ids=["separated", "overlapping"])
def test_binary_matches_values_when_the_first_value_goes_with_larger_numbers(overlap: bool) -> None:
    num = np.random.default_rng(0).permutation(200).astype(float)
    label = np.where(num >= 100, 0, 1)
    if overlap:
        label[num == 5] = 0  # the groups now overlap, and only their percentiles are separated

    assert _findings(pd.DataFrame({"num": num, "label": label}), "BINARY_MATCHES_VALUES") == ['"num" AND "label"']


def test_binary_same_checks_a_larger_sample_of_large_datasets() -> None:
    labels = np.random.default_rng(0).integers(0, 2, 10_050)

    assert _findings(pd.DataFrame({"a": labels, "b": labels}), "BINARY_SAME") == ['"a" AND "b"']


def test_position_non_alphanumeric_counts_positions_from_the_end() -> None:
    values = [f"{'x' * (3 + i % 6)}-{i % 90 + 10}" for i in range(200)]

    assert _findings(pd.DataFrame({"code": values}), "POSITION_NON-ALPHANUMERIC") == ["code"]


@pytest.mark.parametrize("smaller_sums_value", [
    0,
    pytest.param(1, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="Known defect: when the second value goes with the smaller sums, the check tests the opposite direction",
    )),
])
def test_binary_matches_sum_finds_which_value_goes_with_smaller_sums(smaller_sums_value: int) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.integers(0, 100, 200).astype(float), "b": rng.integers(0, 100, 200).astype(float)})
    df["label"] = np.where(df["a"] + df["b"] < 101, smaller_sums_value, 1 - smaller_sums_value)

    assert _findings(df, "BINARY_MATCHES_SUM") == ['"a" AND "b" AND "label"']


def test_init_data_converts_categorical_text_to_strings() -> None:
    values = pd.Categorical([["a", "b", None][i % 3] for i in range(30)])
    checker = DataConsistencyChecker(verbose=-1)

    checker.init_data(pd.DataFrame({"cat": values, "num": np.arange(30.0)}))

    assert checker.orig_df["cat"].dtype == object
    assert checker.orig_df["cat"].tolist()[:3] == ["a", "b", "nan"]
