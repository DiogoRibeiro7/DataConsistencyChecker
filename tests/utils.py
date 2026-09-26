"""Shared helpers for the per-check regression modules (``tests/test_<TEST_ID>.py``).

Each per-check module sets a module-level ``test_id`` (``conftest.py`` uses it to
skip checks that are not implemented) and describes what that check is expected
to report as patterns and as exceptions:

* ``test_real`` runs the check on the OpenML datasets in ``list_real_files.py``;
* ``test_synthetic_*`` run it on the synthetic data generated for that check,
  without nulls and with nulls injected in several ways;
* ``test_synthetic_all_cols_*`` run it on the synthetic data generated for all
  checks at once.

An expected result is either a list of column sets (serialised as
``'"a" AND "b"'`` and matched regardless of column order) or an ``int``: ``n >= 0``
means exactly ``n`` rows, and ``-n`` means at least ``n`` rows.

The flags below select which families of tests run. Tests of a disabled family
are reported as skipped, and the skip reason names the flag to change.
"""

from __future__ import annotations

import os
import pickle

import dill
import pandas as pd
import pytest
from list_real_files import real_files
from sklearn.datasets import fetch_openml

from data_consistency_checker import DataConsistencyChecker

# Settings. Adjust these before running the tests to run a reduced set of tests.
TEST_REAL = False
TEST_SYNTHETIC = True
TEST_SYNTHETIC_NONES = True
TEST_SYNTHETIC_ALL_COLUMNS = False

PRINT_OUTPUT = True

cache_folder = "dc_cache"


def _requires(description: str, **flags: bool) -> pytest.MarkDecorator:
    """Return a marker that skips a test unless all the given setting flags are enabled."""
    disabled = [name for name, enabled in flags.items() if not enabled]
    settings = " and ".join(f"{name} = False" for name in disabled)
    return pytest.mark.skipif(bool(disabled), reason=f"{description} are disabled ({settings} in tests/utils.py)")


requires_real_data = _requires("real-data tests", TEST_REAL=TEST_REAL)
requires_synthetic = _requires("synthetic-data tests", TEST_SYNTHETIC=TEST_SYNTHETIC)
requires_synthetic_nones = _requires(
    "synthetic tests with nulls",
    TEST_SYNTHETIC=TEST_SYNTHETIC,
    TEST_SYNTHETIC_NONES=TEST_SYNTHETIC_NONES,
)
requires_synthetic_all_columns = _requires(
    "all-columns synthetic tests",
    TEST_SYNTHETIC=TEST_SYNTHETIC,
    TEST_SYNTHETIC_ALL_COLUMNS=TEST_SYNTHETIC_ALL_COLUMNS,
)


def _column_signature(value):
    """Canonicalize a serialized column set for order-independent comparison."""
    parts = [part.strip().strip('"') for part in str(value).split(" AND ")]
    return tuple(sorted(parts))


def _contains_column_set(values, expected):
    """Return whether expected matches one serialized column set in values."""
    expected_signature = _column_signature(expected)
    return any(_column_signature(value) == expected_signature for value in values)


def _assert_count(found_df: pd.DataFrame, expected: int, kind: str) -> None:
    """Assert the number of rows found: exactly ``expected``, or at least ``-expected`` if negative."""
    if expected < 0:
        assert len(found_df) >= abs(expected), f"expected at least {abs(expected)} {kind}, found {len(found_df)}"
    else:
        assert len(found_df) == expected, f"expected {expected} {kind}, found {len(found_df)}"


def _assert_column_sets_found(found_df: pd.DataFrame, expected: list[str], kind: str) -> None:
    """Assert that every expected column set appears in ``found_df``."""
    for col in expected:
        found = _contains_column_set(found_df["Column(s)"].values, col)
        if PRINT_OUTPUT and not found:
            print("Missing column: ", col)
        assert found, f"{kind}: no result for {col}"


def _assert_expected(
    found_df: pd.DataFrame | None, expected: int | list[str], kind: str, length_check: str | None
) -> None:
    """Compare the patterns or exceptions found by a synthetic run with the expected result.

    For a list, ``length_check`` is ``"exact"`` (no other rows found), ``"at_least"``
    (other rows allowed) or ``None`` (number of rows not checked). ``found_df`` is None
    when nothing was found.
    """
    if found_df is None:
        found_df = pd.DataFrame(columns=["Test ID", "Column(s)"])
    if PRINT_OUTPUT:
        print(f"len({kind}_df)", len(found_df))
    if isinstance(expected, int):
        if PRINT_OUTPUT:
            print(f"expected_{kind}_cols", expected)
        _assert_count(found_df, expected, kind)
        return

    if PRINT_OUTPUT:
        print(f"len(expected_{kind}_cols)", len(expected))
    if length_check == "exact":
        assert len(found_df) == len(expected), f"expected {len(expected)} {kind}, found {len(found_df)}"
    elif length_check == "at_least":
        assert len(found_df) >= len(expected), f"expected at least {len(expected)} {kind}, found {len(found_df)}"
    _assert_column_sets_found(found_df, expected, kind)


def build_default_results():
    """Return the expected real-data results with no patterns and no exceptions for every dataset."""
    return {dataset: ([], []) for dataset in real_files}


def load_openml_file(dataset_name):
    try:
        data = fetch_openml(dataset_name, version=1)
    except Exception:
        # Most datasets have a version 1, but if not, just load any version. This may result in a warning.
        data = fetch_openml(dataset_name)
    return pd.DataFrame(data.data, columns=data.feature_names)


def real_test(test_id, expected_results_dict):
    """Run ``test_id`` on each real dataset and compare with ``expected_results_dict[dataset]``.

    Each value of ``expected_results_dict`` is ``(expected_patterns, expected_exceptions)``.
    """
    # If not present, create the folder for the cache
    os.makedirs(cache_folder, exist_ok=True)

    for dataset_name in real_files:
        print(f"Testing {dataset_name}")

        expected_patterns_cols = expected_results_dict[dataset_name][0]
        expected_exceptions_cols = expected_results_dict[dataset_name][1]

        file_name = os.path.join(cache_folder, dataset_name + "_dc.pkl")
        if os.path.exists(file_name):
            with open(file_name, "rb") as file_handle:
                dc = pickle.load(file_handle)
        else:
            data_df = load_openml_file(dataset_name)
            dc = DataConsistencyChecker(verbose=-1)
            dc.init_data(data_df)

        fast_only = dataset_name in ["Satellite"]
        dc.check_data_quality(execute_list=[test_id], fast_only=fast_only)
        assert dc.num_exceptions == 0

        # Check the returned patterns without exceptions are correct
        patterns_df = dc.get_patterns_list(show_short_list_only=False)
        if isinstance(expected_patterns_cols, int):
            _assert_count(patterns_df, expected_patterns_cols, "patterns")
        else:
            assert ((patterns_df is None) and (len(expected_patterns_cols) == 0)) or (
                len(patterns_df) == len(expected_patterns_cols)
            )
            _assert_column_sets_found(patterns_df, expected_patterns_cols, "patterns")

        # Check the returned patterns with exceptions are correct
        exceptions_df = dc.get_exceptions_list()
        if isinstance(expected_exceptions_cols, int):
            _assert_count(exceptions_df, expected_exceptions_cols, "exceptions")
        else:
            assert ((exceptions_df is None) and (len(expected_patterns_cols) == 0)) or (
                len(exceptions_df) == len(expected_exceptions_cols)
            )
            _assert_column_sets_found(exceptions_df, expected_exceptions_cols, "exceptions")


def synth_test(test_id, add_nones, expected_patterns_cols, expected_exceptions_cols, allow_more=False):
    """Run ``test_id`` on the synthetic data generated for it and compare with the expected results.

    ``add_nones`` selects how nulls are injected (see ``generate_synth_data``). With
    ``allow_more``, results beyond the listed column sets are tolerated.
    """
    execute_list = [test_id]
    dc = DataConsistencyChecker(verbose=-1)
    synth_df = dc.generate_synth_data(all_cols=False, execute_list=execute_list, add_nones=add_nones)
    dc.init_data(synth_df)
    dc.check_data_quality(execute_list=execute_list)

    if PRINT_OUTPUT:
        print()
        print("add_nones:", add_nones)
    length_check = "at_least" if allow_more else "exact"
    patterns_df = dc.get_patterns_list(show_short_list_only=False)
    _assert_expected(patterns_df, expected_patterns_cols, "patterns", length_check)
    exceptions_df = dc.get_exceptions_list()
    _assert_expected(exceptions_df, expected_exceptions_cols, "exceptions", length_check)


def synth_test_all_cols(test_id, add_nones, expected_patterns_cols, expected_exceptions_cols):
    """Run ``test_id`` on the synthetic data generated for all checks.

    Only the listed column sets are checked; other results are tolerated.
    """
    # If not present, create the folder for the cache
    os.makedirs(cache_folder, exist_ok=True)

    # Note: the cached data does not depend on add_nones, so the first variant run creates it for all.
    file_name = os.path.join(cache_folder, "synth_all_cols_dc.pkl")
    if os.path.exists(file_name):
        with open(file_name, "rb") as file_handle:
            dc = pickle.load(file_handle)
    else:
        dc = DataConsistencyChecker(verbose=-1)
        synth_df = dc.generate_synth_data(all_cols=True, add_nones=add_nones)
        dc.init_data(synth_df)
        with open(file_name, "wb") as filehandler:
            dill.dump(dc, filehandler)

    dc.check_data_quality(execute_list=[test_id])

    if PRINT_OUTPUT:
        print()
        print("add_nones:", add_nones)
    patterns_df = dc.get_patterns_list(show_short_list_only=False)
    _assert_expected(patterns_df, expected_patterns_cols, "patterns", length_check=None)
    exceptions_df = dc.get_exceptions_list()
    _assert_expected(exceptions_df, expected_exceptions_cols, "exceptions", length_check=None)
