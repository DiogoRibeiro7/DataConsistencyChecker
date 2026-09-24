from __future__ import annotations
from typing import Any, Callable

from dataexcept import OutlierDetectionError, exception_to_dict
from dataexcept import wrap as wrap_dataexcept

import pandas as pd
import numpy as np
import numbers
import os
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import math
import statistics
import datetime
import time
from dateutil.relativedelta import relativedelta
import calendar
import pandas.api.types as pandas_types
import random
import string
import scipy
from itertools import combinations
from IPython import get_ipython
from IPython.display import display, Markdown, HTML
from textwrap import wrap
try:
    from termcolor import colored
except:
    colored = None
import concurrent
from decimal import Decimal, ROUND_HALF_UP
from multiprocessing import Process, Queue

# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
import scipy.stats as scipy_stats

# Mixins
from .display_mixin import DisplayMixin
from .plots_mixin import PlotsMixin
from .synth_data_mixin import SynthDataMixin
from .results_mixin import ResultsMixin
from .analysis_cache_mixin import AnalysisCacheMixin

# Test implementation mixins
from .test_implementations import (
    BaseTestsMixin,
    NumericTestsMixin,
    DateTestsMixin,
    StringTestsMixin,
    BinaryTestsMixin,
    MultiColumnTestsMixin,
)

# Utility functions split out from the main module
from .checker_utils import (
    safe_div,
    is_number,
    convert_to_numeric,
    get_num_decimal_digits,
    get_non_alphanumeric,
    styling_orig_row,
    styling_flagged_rows,
    is_notebook,
    print_line,
    print_text,
    is_missing,
    array_to_str,
    replace_special_with_space,
    truncate_description,
    is_uppercase,
    call_test,
    clean_x_tick_labels,
    set_warnings_levels,
)

# Test registry and definitions
from .test_registry import TestDefinition
from .tests_definitions import get_all_test_definitions

# Uncomment to debug any warnings.
# warnings.filterwarnings("error")

# Constants used to generate synthetic data
letters = string.ascii_letters
digits = string.digits
alphanumeric = letters + digits


class DataConsistencyChecker(BaseTestsMixin, NumericTestsMixin, DateTestsMixin, StringTestsMixin, BinaryTestsMixin, MultiColumnTestsMixin, DisplayMixin, PlotsMixin, SynthDataMixin, ResultsMixin, AnalysisCacheMixin):
    """
    Automated data quality checker performing 164 tests to identify patterns and anomalies.

    This class examines tabular datasets for consistency patterns across single columns,
    pairs of columns, and larger column sets. It identifies both patterns and exceptions
    to those patterns, useful for EDA and interpretable outlier detection.
    """

    def __init__(
        self,
        iqr_limit: float = 3.5,
        idr_limit: float = 1.0,
        max_combinations: int = 100_000,
        verbose: int = 1
    ) -> None:
        """
        Initialize a DataConsistencyChecker object.

        Args:
            iqr_limit: Inter-quartile range multiplier used to identify outliers.
                Higher values reduce false positives. Default: 3.5
            idr_limit: Inter-decile range multiplier for outlier detection in
                strictly positive data. Default: 1.0
            max_combinations: Maximum number of column combinations to test.
                Limits execution time for tests on multiple column sets. Default: 100,000
            verbose: Verbosity level for progress output:
                -1: No output
                 0: Output only after completion
                 1: Display test names during execution
                 2: Display test descriptions and progress updates
        """

        set_warnings_levels()

        # Set class variables from the parameters
        # iqr_limit indicates how many multiples of the IQR below the 1st quartile or above the 3rd quartile are
        # considered outliers. This is used in numerous tests checking for small or large values. To reduce noise, a
        # stricter limit than the 1.5 or 2.2 normally used should be specified.
        self.iqr_limit = iqr_limit
        self.idr_limit = idr_limit
        self.max_combinations = max_combinations
        self.verbose = verbose

        # The tests to execute are specified in check_data_quality()
        self.execute_list = None
        self.exclude_list = None
        self.execution_test_list = []

        self.freq_contamination_level = -1
        self.rare_contamination_level = -1        

        # Synthetic data, if specified to create
        self.synth_df = None
        self.num_synth_rows = 1_000

        # The data to be examined and statistics related to it
        self.orig_df = None
        self.num_rows = -1
        self.column_medians = {}
        self.column_trimmed_means = {}
        self.column_unique_vals = {}  # Stored only for binary columns. Excludes None and NaN values.
        self.numeric_vals = {}  # An array of the truly numeric values in each numeric column
        self.numeric_vals_filled = {}
        self.binary_cols = []
        self.numeric_cols = []
        self.date_cols = []
        self.string_cols = []

        # Similar variables related to the sample df
        self.sample_numeric_vals_filled = {}

        # The number of tests executed. This will be the full set unless an list of tests to perform / excluded
        # is provided.
        self.n_tests_executed = 0
        self.execution_times = None

        # Information about the patterns found that have no exceptions
        self.patterns_arr = []
        self.patterns_df = None

        # A summary of the issues found, with a row for each test-feature describing the test and the exceptions.
        self.results_summary_arr = []
        self.exceptions_summary_df = None

        # A dataframe indicating which cells in the original data were flagged by which tests. The rows correspond
        # to the rows in the original data. There is a column for each test-feature set where the pattern is true, but
        # there are exceptions. For example a test on columns A, B, and C would have, if exceptions were found, a
        # single column in test_results_df for that test and set of columns.
        self.test_results_df = None

        self.results_dict = {}

        # HTML exports
        # For HTML exports, the images are saved as separate files in the same image and given sequential numbers
        self.image_output_num = 0
        self.output_folder = None

        # This maps the columns used in test_results_df and patterns_df to the original columns
        self.col_to_original_cols_dict = {}

        # Data that is collected once needed, and then saved for subsequent tests to use.
        self._init_variables()

        # This maps one-to-one with the original data, but in each cell contains a score for the corresponding cell in
        # the original data. It has the same rows and columns as the original data. If a test finds an issue with
        # columns A, B, and C in rows 10 and 13, then test_results_by_column_np will have, in rows 10 & 13, in columns
        # A, B, and C, 0.33 each (in addition to any other issues these cells have been flagged for).
        self.test_results_by_column_np = None

        # Safe versions of the exceptions found. These are stored to support restore_issues()
        self.safe_patterns_arr = []
        self.safe_patterns_df = None
        self.safe_results_summary_arr = []
        self.safe_exceptions_summary_df = None
        self.safe_test_results_df = None
        self.safe_results_dict = {}
        self.safe_test_results_by_column_np = None

        # Debugging information
        self.DEBUG_MSG = True
        self.num_exceptions = 0
        self.execution_failures: list[dict[str, Any]] = []

        # Variables used for calls to display_next()
        self.found_tests = []
        self.current_display_test = -1

        # A data structure to represent how each test (for the tests that run on a single feature) measures up for
        # each feature. This is saved as a dictionary during processing, then converted to a dataframe when requested.
        self.single_test_summary_dict = {}
        self.single_test_summary_df = None

        # Display options. There are relevant only when running this in a debugger or notebook.
        # Note: these can significantly slow down Jupyter in some environments, and so is set only for debugger
        # environments.
        if not is_notebook():
            pd.set_option('display.width', 32000)
            pd.set_option('display.max_columns', 3000)
            pd.set_option('display.max_colwidth', 3000)
            pd.set_option('display.max_rows', 5000)

        # A dictionary describing each test. For each, we have the ID, description, method to test for the pattern
        # and exceptions, a method to generate synthetic data to demonstrate the test, and in indicator if the
        # pattern is in the patterns short list (ie, the patterns listed by default in a call to get_patterns()),
        # and other properties of the tests. See the list of enums above.
        # Test definitions are now organized in the tests_definitions package for better maintainability.
        self.test_dict = get_all_test_definitions(self)

        # Remove any tests not yet implemented (legacy filter, kept for compatibility)
        self.test_dict: dict[str, TestDefinition] = {
            test_id: definition
            for test_id, definition in self.test_dict.items()
            if definition.implemented
        }


    def init_data(
        self,
        df: pd.DataFrame,
        known_date_cols: list[str] | None = None
    ) -> None:
        """
        Prepare data for quality checking. Must be called before check_data_quality().

        Args:
            df: DataFrame to be assessed for data quality
            known_date_cols: List of column names to treat as date columns.
                If None, date columns are auto-detected.

        Returns:
            None. Data is stored in instance variables.
        """

        self.orig_df = df.copy()

        # Check the dataframe does not contain duplicate column names. If it does, rename all columns with a _idx at
        # the end of each
        if len(df.columns) > len(set(df.columns)):
            df = df.copy()
            print(("Duplicate column names encountered. A suffix will be added to all column names to ensure they are "
                   f"unique. Duplicate column names: "
                   f"{set([x for x in df.columns if df.columns.tolist().count(x) > 1])}"))
            new_col_names = []
            for col_idx, col_name in enumerate(df.columns):
                new_col_names.append(f"{col_name}_{col_idx}")
            self.orig_df.columns = new_col_names

        # Ensure the dataframe has at least a very minimum number of rows.
        if len(df) < 10:
            print(f"The dataframe contains too few rows to be examined in a meaningful manner. Number of rows: "
                  f"{len(df)}")
            return None

        # Ensure the dataframe has a predictable index, which may be kept inline with parallel dataframes created
        # running the tests.
        self.orig_df = self.orig_df.reset_index(drop=True)
        self.num_rows = len(self.orig_df)
        self.num_valid_rows = None

        # Ensure the dataframe has column names in string format. Often dataframes contain numeric column names.
        self.orig_df.columns = [str(x) for x in self.orig_df.columns]

        # Remove any columns where there are not two or more unique values, or the values are all Null
        cols = [x for x in self.orig_df.columns if
                (self.orig_df[x].nunique(dropna=False) > 1) and (self.orig_df[x].isna().sum() < self.num_rows)]
        if self.verbose >= 2 and len(cols) < len(self.orig_df.columns):
            removed_cols = set(self.orig_df.columns) - set(cols)
            print()
            print(f"Removing columns with only one value: {removed_cols}")
        self.orig_df = self.orig_df[cols]

        # Determine which columns are binary, numeric, date, and none of these. Any that are not binary, numeric, or
        # date we consider as string.
        self.binary_cols = []
        self.numeric_cols = []
        self.date_cols = []
        self.string_cols = []

        # As this is called before check_data_quality, self.freq_contamination_level is not yet set. We use here the
        # default value in order to determine numeric columns with some non-numeric values.
        default_contamination_level = self.num_rows * 0.005

        for col_name in self.orig_df.columns:
            if self.orig_df[col_name].nunique() == 2:
                self.binary_cols.append(col_name)
            elif self.orig_df[col_name].dtype in [np.datetime64, 'datetime64[ns]']:
                self.date_cols.append(col_name)
            elif pandas_types.is_numeric_dtype(self.orig_df[col_name]) or \
                    self.orig_df[col_name].astype(str).str.replace('-', '', regex=False).str.\
                            replace('.', '', regex=False).str.isdigit().tolist().count(False) < default_contamination_level:
                self.numeric_cols.append(col_name)
            else:
                try:
                    _ = self.orig_df[col_name].astype(float)
                    self.numeric_cols.append(col_name)
                except Exception:
                    self.string_cols.append(col_name)

        # Try to convert any string columns we can to date format. The code below is likely sufficient, though may
        # erroneously convert some string or numeric columns to dates. If we find any legitimate date columns are
        # missed, we can use PyTime: https://github.com/shinux/PyTime. We are trying to minimize the pip installs
        # necessary for this tool, so will only add this if necessary.

        # As we cannot define the format for the date columns, attempts to cast values to datetime may present
        # warnings, which we ignore, but only during this process.
        warnings.filterwarnings(action='ignore', category=UserWarning)
        if known_date_cols is None:
            new_date_cols = []
            for col_name in self.string_cols + self.numeric_cols:
                avg_num_chars = statistics.median(self.orig_df[col_name].astype(str).str.len())
                num_rows_all_digits = self.orig_df[col_name].astype(str).str.isdigit().tolist().count(True)

                # Do not convert to date if the strings are too short. They must be at least yyyymm (6 characters)
                if avg_num_chars < 6:
                    continue

                # Do not convert to date if the strings are almost all digits and are too long
                if num_rows_all_digits > (self.num_rows / 2) and avg_num_chars > 8:
                    continue

                # Try some known formats before letting pandas attempt to determine the format
                is_date = False

                # Check if the column is of the form yyyymm or mmyyyy
                if avg_num_chars == 6:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name], format="%Y%M")
                        is_date = True
                    except:
                        pass
                if avg_num_chars == 6 and not is_date:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name], format="%M%Y")
                        is_date = True
                    except:
                        pass

                # Check if the column is of the form yyyymm or mmyyyy, but converted to float, so containing '.0'
                if avg_num_chars == 6:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name].astype(np.int64), format="%Y%M")
                        is_date = True
                    except:
                        pass
                if avg_num_chars == 6 and not is_date:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name].astype(np.int64), format="%M%Y")
                        is_date = True
                    except:
                        pass

                if col_name in self.string_cols and not is_date:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name])
                        is_date = True
                    except Exception:
                        pass

                if is_date:
                    new_date_cols.append(col_name)
                    self.date_cols.append(col_name)

            for datecol in new_date_cols:
                if datecol in self.string_cols:
                    self.string_cols.remove(datecol)
                if datecol in self.numeric_cols:
                    self.numeric_cols.remove(datecol)
        else:
            new_date_cols = []
            for col_name in known_date_cols:
                try:
                    self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name])
                    new_date_cols.append(col_name)
                    if col_name not in self.date_cols:
                        self.date_cols.append(col_name)
                except Exception:
                    pass

            for datecol in new_date_cols:
                if datecol in self.string_cols:
                    self.string_cols.remove(datecol)
                if datecol in self.numeric_cols:
                    self.numeric_cols.remove(datecol)
                if datecol in self.binary_cols:
                    self.binary_cols.remove(datecol)
        set_warnings_levels()

        # For any columns flagged as string columns, the dtype may be category.  Convert the columns to string to
        # ensure the code can compare values and perform other string operations
        for col_name in self.string_cols + self.binary_cols:
            if self.orig_df[col_name].dtype.name == 'category':
                self.orig_df[col_name] = self.orig_df[col_name].astype(str)

        for col_name in self.numeric_cols:
            if self.orig_df[col_name].dtype.name == 'category':
                try:
                    self.orig_df[col_name] = self.orig_df[col_name].astype(float)
                except:
                    self.orig_df[col_name] = self.orig_df[col_name].astype(str).astype(float)

        # For binary columns, find and cache the set of unique values per column
        for col_name in self.binary_cols:
            self.column_unique_vals[col_name] = sorted([x for x in self.orig_df[col_name].unique() if not is_missing(x)])

        # For all numeric columns, get the set of truly numeric values. This may have less than self.num_rows elements.
        for col_name in self.numeric_cols:
            self.numeric_vals[col_name] = pd.Series(
                [float(x) for x in self.orig_df[col_name]
                 if (isinstance(x, numbers.Number) or str(x).replace('-', '').replace('.', '').isdigit())])

        # Calculate and cache the median for each numeric column.
        self.column_medians = {}
        for col_name in self.numeric_cols:
            # It may be that there is the odd non-numeric value in the column. We take only valid numbers to
            # calculate the median.
            self.column_medians[col_name] = self.numeric_vals[col_name].median()

        # For all numeric columns, get the set of truly numeric values or the median. This will have the same rows
        # as orig_df.
        # Todo: remove calls to convert_to_numeric that use the median anyway
        for col_name in self.numeric_cols:
            self.numeric_vals_filled[col_name] = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])

        trimmed_orig_df = self.orig_df.copy()
        if len(self.numeric_cols) > 0:
            # Calculate and cache the pairwise correlations between each numeric column
            numeric_df = None
            for col_name in self.numeric_cols:
                if numeric_df is None:
                    numeric_df = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])
                else:
                    numeric_df = pd.concat([numeric_df, convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])], axis=1)
            numeric_df.columns = self.numeric_cols

            # Calculate the correlations between the numeric columns
            if len(self.numeric_cols) >= 2:
                self.pearson_corr = numeric_df.corr(method='pearson')
                self.spearman_corr = numeric_df.corr(method='spearman')

            # Calculate the trimmed mean for each numeric column
            for col_name in self.numeric_cols:
                lower_limit = self.numeric_vals[col_name].quantile(0.01)
                upper_limit = self.numeric_vals[col_name].quantile(0.99)
                reduced_numeric_vals_filled = self.numeric_vals_filled[col_name].loc[trimmed_orig_df.index]
                trimmed_orig_df = trimmed_orig_df[(reduced_numeric_vals_filled > lower_limit) &
                                                  (reduced_numeric_vals_filled < upper_limit)]
            self.column_trimmed_means = {}
            for col_name in self.numeric_cols:
                self.column_trimmed_means[col_name] = \
                    convert_to_numeric(trimmed_orig_df[col_name], self.column_medians[col_name]).mean()

        # Create a sample of the full data, which may be used for early stopping for expensive tests.
        # The sample_df will tend to not contain Null values.
        if len(trimmed_orig_df) > 50:
            self.sample_df = trimmed_orig_df.dropna().sample(n=min(len(trimmed_orig_df.dropna()), 50), random_state=0)
        elif len(self.orig_df.dropna()) > 50:
            self.sample_df = self.orig_df.dropna().sample(n=50, random_state=0)
        else:
            self.sample_df = self.orig_df.sample(n=min(len(self.orig_df), 50), random_state=0)

        if (len(self.sample_df) < 50) and (len(self.sample_df) < len(self.orig_df)):
            num_needed = 50 - len(self.sample_df)
            self.sample_df = pd.concat([self.sample_df,
                                        self.orig_df.sample(n=min(len(self.orig_df), 50), random_state=0)])

        # Similar to numeric_vals_filled, fill in this for the sample_df
        for col_name in self.numeric_cols:
            self.sample_numeric_vals_filled[col_name] = convert_to_numeric(self.sample_df[col_name], self.column_medians[col_name])

        # Calculate the number of valid (ie, not missing) values there are per row
        self.num_valid_rows = {}
        for col_name in self.orig_df.columns:
            self.num_valid_rows[col_name] = len([x for x in self.orig_df[col_name] if not is_missing(x)])

        # Replace any NA values that cannot be treated as None or NaN
        def test_NA(x):
            try:
                if x == x:
                    x = x
                return False
            except:
                return True

        for col_name in self.orig_df.columns:
            self.orig_df[col_name] = [x if not test_NA(x) else None for x in self.orig_df[col_name]]

        # patterns_df has a row for each test for each feature where there is a pattern with no exceptions.
        self.patterns_arr = []
        self.patterns_df = None

        # test_results_df has a column for each test for each original column where there is a pattern and also
        # exceptions.
        self.test_results_df = pd.DataFrame()
        self.n_tests_executed = 0

        # test_results_by_column_np is set initially to all zeros, as no
        self.test_results_by_column_np = np.zeros((self.num_rows, len(self.orig_df.columns)), dtype=float)

        # exceptions_summary_df has row for each test for each feature where there is a pattern and also exceptions.
        self.results_summary_arr = []
        self.exceptions_summary_df = None

        # Variables set as needed.
        self._init_variables()

        if self.verbose >= 2:
            print()
            print("Identified column types:")
            print(f"Number string: {len(self.string_cols)}")
            print(f"Number numeric: {len(self.numeric_cols)}")
            print(f"Number date/time: {len(self.date_cols)}")
            print(f"Number binary: {len(self.binary_cols)}")
            print()

    def _init_variables(self):
        self.lower_limits_dict = None
        self.upper_limits_dict = None
        self.larger_pairs_dict = None
        self.larger_or_equal_pairs_dict = None
        self.larger_pairs_with_bool_dict = None
        self.larger_or_equal_pairs_with_bool_dict = None
        self.is_missing_dict = None
        self.sample_is_missing_dict = None
        self.num_missing_dict = None
        self.percentiles_dict = None
        self.nunique_dict = None
        self.count_most_freq_value_dict = None
        self.words_list_dict = None
        self.word_counts_dict = None
        self.cols_same_bool_dict = None
        self.cols_same_count_dict = None
        self.cols_pairs_both_null_dict = None
        self.sample_cols_pairs_both_null_dict = None
        self.col_pairs_either_null_bool_dict = None
        self.col_triples_all_null_bool_dict = None
        self.common_vals_dict = None


    def _record_execution_failure(
        self,
        test_id: str,
        error: Exception,
        *,
        raise_on_error: bool,
    ) -> OutlierDetectionError:
        """Translate and retain one failed consistency-test execution.

        The original exception is preserved as ``__cause__`` by DataExcept.
        The serialized envelope is retained so callers can inspect failures
        without parsing console output.

        Args:
            test_id: Identifier of the consistency test that failed.
            error: Original exception raised by the test implementation.
            raise_on_error: If True, re-raise the structured failure.

        Returns:
            The structured DataExcept error representing the failed test.
        """
        structured = wrap_dataexcept(
            error,
            OutlierDetectionError,
            method=test_id,
            details=str(error),
        )
        self.execution_failures.append(
            {
                "test_id": test_id,
                "error": exception_to_dict(structured),
            }
        )
        self.num_exceptions += 1

        if raise_on_error:
            raise structured from error
        return structured

    def check_data_quality(
        self,
        append_results: bool = False,
        execute_list: list[str] | None = None,
        exclude_list: list[str] | None = None,
        test_start_id: int = 0,
        fast_only: bool = False,
        include_code_tests: bool = True,
        freq_contamination_level: int | float = 0.005,
        rare_contamination_level: int | float = 0.1,
        run_parallel: bool = False,
        raise_on_error: bool = False,
    ) -> None:
        """
        Execute data quality tests on the dataset specified in init_data().

        Identifies patterns and exceptions in the data. Use additional API calls
        to access and analyze the results.

        Args:
            append_results: If True, append to previous results; if False, clear previous results
            execute_list: Specific test IDs to execute. If None, runs all tests
            exclude_list: Test IDs to exclude. Cannot be used with execute_list
            test_start_id: Test number to start from (for resuming incomplete runs)
            fast_only: If True, run only fast single-column tests
            include_code_tests: If True, include tests for code/ID value columns
            freq_contamination_level: Max fraction (or count) of rows violating a pattern
                for tests that frequently find results. Lower values reduce false positives
            rare_contamination_level: Max fraction (or count) of rows violating a pattern
                for tests that rarely find results. Higher values reduce false negatives
            run_parallel: If True, run tests in parallel for faster execution
            raise_on_error: If True, stop on the first failed test and raise a
                structured OutlierDetectionError. If False, retain failures and
                continue running the remaining tests.

        Returns:
            None. Results stored in instance variables accessible via other methods.

        Raises:
            AssertionError: If both execute_list and exclude_list are specified
            OutlierDetectionError: If a test fails and raise_on_error is True
        """

        if self.orig_df is None or len(self.orig_df) == 0:
            print("Valid dataframe not specified, possibly due to not calling init_data(), or passing a null dataframe")
            return None

        # execute_list and exclude_list should not both be set.
        assert execute_list is None or exclude_list is None

        # The set of tests that are to be run / not run may optionally be specified. This may also effect any synthetic
        # data which is generated.
        self.execute_list = execute_list
        self.exclude_list = exclude_list
        self.execution_test_list = []  # The complete list for one call to check_data_quality()

        # Check if the specified tests are valid
        specified_test_list = []
        if exclude_list:
            specified_test_list.extend(exclude_list)
        if execute_list:
            specified_test_list.extend(execute_list)
        for t in specified_test_list:
            if t not in self.test_dict.keys():
                print(f"Error {t} is not a valid test")

        # Store the contamination_level in terms of number of rows. It may have been passed either in this form or as a
        # fraction.
        if freq_contamination_level > 1 and type(freq_contamination_level) == int:
            if freq_contamination_level > len(self.orig_df):
                print((f"Error. contamination rate set to {freq_contamination_level}, more than the number of rows in "
                       f"the dataframe passed. The contamination_level rate should be substantially smaller."))
                return None
            self.freq_contamination_level = freq_contamination_level
        elif freq_contamination_level < 1.0:
            self.freq_contamination_level = freq_contamination_level * len(self.orig_df)
            if self.freq_contamination_level < 1.0:
                if freq_contamination_level != 0.005:
                    print((f"Error. contamination rate set to {freq_contamination_level}, not allowing even 1 row to "
                           f"be in violation of the patterns. Must be set to a larger value given the number of rows "
                           "available."))
                    return None
                self.freq_contamination_level = 1

        # Adjust the test_start_id to 0 if necessary. 0 is the lowest valid value.
        if test_start_id < 0:
            test_start_id = 0

        # Determine the set of tests to execute
        self.execution_test_list = []
        for test_idx, test_id in enumerate(self.test_dict.keys()):
            if test_idx < test_start_id:
                continue
            if fast_only and not self.test_dict[test_id].fast:
                continue
            if (not include_code_tests) and self.test_dict[test_id].code:
                continue
            if (self.execute_list is None and self.exclude_list is None) or \
                    (self.execute_list and test_id in self.execute_list) or \
                    (self.exclude_list and test_id not in self.exclude_list):
                if self.test_dict[test_id].implemented:
                    self.execution_test_list.append(test_id)

        # Check at least one valid test was specified
        if len(self.execution_test_list) == 0:
            print("No valid tests specified.")
            return None

        # Initialize the variables related to the run
        self.n_tests_executed = 0
        self.execution_times = {}
        self.num_exceptions = 0
        self.execution_failures = []

        # Initialize the variables related to the results found, unless append_results is specified.
        if not append_results:
            self.patterns_arr = []
            self.patterns_df = None
            self.results_summary_arr = []
            self.exceptions_summary_df = None
            self.test_results_df = None
            self.results_dict = {}
            self.test_results_by_column_np = np.zeros((self.num_rows, len(self.orig_df.columns)), dtype=float)

        # Get the test index of each test
        test_idx_dict = {x: y for x, y in
                         zip(self.test_dict.keys(), range(len(self.test_dict.keys())))
                         if self.test_dict[x].implemented}

        # Initialize the results for tests on single columns
        self.single_test_summary_dict = {}
        self.single_test_summary_df = None

        if run_parallel:
            process_arr = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for test_id in self.execution_test_list:
                    self._output_current_test(test_idx_dict[test_id], test_id)
                    future = executor.submit(call_test, self, test_id)
                    process_arr.append((test_id, future))
                    self.n_tests_executed += 1
                for test_id, future in process_arr:
                    try:
                        future.result()
                    except Exception as error:
                        structured = self._record_execution_failure(
                            test_id,
                            error,
                            raise_on_error=raise_on_error,
                        )
                        if self.verbose >= 0:
                            print(f"Error executing {test_id}: {structured}")
        else:
            for test_id in self.execution_test_list:
                self._output_current_test(test_idx_dict[test_id], test_id)
                try:
                    t1 = time.time()
                    self.test_dict[test_id].test_func(test_id=test_id)
                    t2 = time.time()
                    self.execution_times[test_id] = t2 - t1
                except Exception as error:
                    structured = self._record_execution_failure(
                        test_id,
                        error,
                        raise_on_error=raise_on_error,
                    )
                    message = f"Error executing {test_id}: {structured}"
                    if colored:
                        print(colored(message, "red"))
                    else:
                        print(message)
                self.n_tests_executed += 1

        # Populate the test_results_df dataframe with all results found
        self.test_results_df = pd.DataFrame(self.results_dict)

        # Sum the the number of columns flagged per row
        self._calculate_final_scores()

        # Create the final dataframes of patterns and exceptions found
        self.patterns_df = pd.DataFrame(
            self.patterns_arr,
            columns=['Test ID', 'Column(s)', 'Description of Pattern', 'Display Information'])

        self.exceptions_summary_df = pd.DataFrame(
            self.results_summary_arr,
            columns=['Test ID', 'Column(s)', 'Description of Pattern', 'Number of Exceptions', 'Display Information'])

        # Add the Pattern Id column to patterns_df
        self.patterns_df['Pattern ID'] = list(range(len(self.patterns_df)))

        # Add the Issue Id column to exceptions_summary_df
        self.exceptions_summary_df['Issue ID'] = list(range(len(self.exceptions_summary_df)))

        # Display summary statistics
        self._output_stats()

        # Save safe versions of the patterns and exceptions found, to support restore_issues() if called
        self.safe_patterns_arr = self.patterns_arr.copy()
        self.safe_patterns_df = self.patterns_df.copy()
        self.safe_results_summary_arr = self.results_summary_arr.copy()
        self.safe_exceptions_summary_df = self.exceptions_summary_df.copy()
        self.safe_test_results_df = self.test_results_df.copy()
        self.safe_results_dict = self.results_dict.copy()
        self.safe_test_results_by_column_np = self.test_results_by_column_np.copy()

        # Initialize variables used by display_next()
        self.found_tests = self.get_test_ids_with_results()
        self.current_display_test = 0

    def check_data_quality_by_feature_pairs(self, max_features_shown=30):
        """
        An alternative to check_data_quality(). This runs similar (though fewer) tests on pairs of features and, for
        each test, presents a matrix indicating for what fraction of the rows a given relationship between the features
        holds true.

        max_features_shown: int
            Where there are many features, it can be infeasible to show a heatmap of all features. However, it may be
            useful to render a heatmap for the first features. max_features_shown specifies how many features, at most,
            will be included in the heatmaps rendered.
        """

        def handle_results(matrix):
            if matrix.sum().sum() == 0:
                print_text("No instances found")
                return
            d = pd.DataFrame(matrix, columns=self.numeric_cols[:max_features_shown],
                             index=self.numeric_cols[:max_features_shown])
            d = d.replace(0, np.nan)
            num_feats = min(len(self.numeric_cols), max_features_shown)
            fig, ax = plt.subplots(figsize=(num_feats, num_feats))
            s = sns.heatmap(d, annot=True, fmt='.4f', cmap="seismic", linewidths=1.1, linecolor='blue', cbar=False)
            for _, spine in s.spines.items():
                spine.set_visible(True)
            plt.show()

        def run_test(func):
            matrix = np.zeros((min(len(self.numeric_cols), max_features_shown),
                               min(len(self.numeric_cols), max_features_shown)))
            for col_name_1_idx, col_name_1 in enumerate(self.numeric_cols[:max_features_shown]):
                for col_name_2_idx, col_name_2 in enumerate(self.numeric_cols[:max_features_shown]):
                    df = func(col_name_1, col_name_2)
                    matrix[col_name_1_idx, col_name_2_idx] = len(df) / self.num_rows
            handle_results(matrix)

        print_text("Blank cells indicate counts of zero for the corresponding pair of features.")
        print_text(f"## Fraction of rows where both columns contain the same value")
        def test_same(col_name_1, col_name_2):
            return self.orig_df[self.orig_df[col_name_1] == self.orig_df[col_name_2]]
        run_test(test_same)

        print_text(f"## Fraction of rows where both columns contain null")
        def test_null(col_name_1, col_name_2):
            return self.orig_df[(self.orig_df[col_name_1].isna()) & (self.orig_df[col_name_2].isna())]
        run_test(test_null)

        print_text(f"## Fraction of rows where both columns contain zero")
        def test_zero(col_name_1, col_name_2):
            return self.orig_df[(self.orig_df[col_name_1] == 0) & (self.orig_df[col_name_2] == 0)]
        run_test(test_zero)

        print_text(f"## Fraction of rows where both columns contain positive values")
        def test_both_positive(col_name_1, col_name_2):
            return self.orig_df[(self.numeric_vals_filled[col_name_1] >= 0) & (self.numeric_vals_filled[col_name_2] >= 0)]
        run_test(test_both_positive)

        print_text(f"## Fraction of rows where both columns contain negative values")
        def test_both_negative(col_name_1, col_name_2):
            return self.orig_df[(self.numeric_vals_filled[col_name_1] < 0) & (self.numeric_vals_filled[col_name_2] < 0)]
        run_test(test_both_negative)

        print_text(f"## Fraction of rows where one columns is the negative of the other")
        def test_negative(col_name_1, col_name_2):
            return self.orig_df[self.numeric_vals_filled[col_name_1] == (-1)*self.numeric_vals_filled[col_name_2]]
        run_test(test_negative)

        print_text(f"## Fraction of rows where one columns is an even multiple of the other")
        def test_even_multiple(col_name_1, col_name_2):
            val_arr_1 = self.numeric_vals_filled[col_name_1]
            val_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series = np.where(val_arr_2 != 0, val_arr_1 / val_arr_2, val_arr_1).tolist()
            test_series = [float(x).is_integer() for x in test_series]
            return self.orig_df.loc[test_series]
        run_test(test_even_multiple)

        print_text(f"## Fraction of rows where one column is larger (using absolute values) than the other")
        def test_larger(col_name_1, col_name_2):
            return self.orig_df[abs(self.numeric_vals_filled[col_name_1]) > abs(self.numeric_vals_filled[col_name_2])]
        run_test(test_larger)

        print_text(f"## Fraction of rows where one columns is ten or more times (using absolute values) larger than the other")
        def test_much_larger(col_name_1, col_name_2):
            return self.orig_df[abs(self.numeric_vals_filled[col_name_1]) > abs((10.0)*self.numeric_vals_filled[col_name_2])]
        run_test(test_much_larger)

    # Public methods to output the results of the analysis in various ways
    ##################################################################################################################

    def plot_final_scores_distribution_by_row(self):
        """
        Display a probability plot and histogram representing the distribution of final scores by row.
        """

        if self.test_results_df is None or self.test_results_df.empty:
            return

        final_scores_df = self.test_results_df.copy()
        final_scores_df = final_scores_df.sort_values('FINAL SCORE', ascending=True)
        final_scores_df['Rank'] = range(self.num_rows)
        plt.subplots(figsize=(5, 3))
        s = sns.scatterplot(data=final_scores_df, x='Rank', y='FINAL SCORE')
        s.set_title("Distribution of Scores by Row, ordered lowest to highest scores")
        plt.show()

        plt.subplots(figsize=(5, 3))
        s = sns.histplot(data=final_scores_df, x='FINAL SCORE')
        s.set_title("Distribution of Scores per Row")
        plt.show()

        if final_scores_df['FINAL SCORE'].nunique() > 2:
            plt.subplots(figsize=(5, 3))
            s = sns.histplot(data=final_scores_df[final_scores_df['FINAL SCORE'] > 0], x='FINAL SCORE')
            s.set_title("Distribution of Scores per Row (Excluding Scores of 0)")
            plt.show()

    def plot_final_scores_distribution_by_feature(self):
        """
        Display a bar plot representing the distribution of final scores by feature.
        """

        final_scores_series = pd.Series(self.test_results_by_column_np.sum(axis=0)).sort_values(ascending=True).values
        final_scores_df = pd.DataFrame({'Feature': self.orig_df.columns,
                                        'Total Scores': final_scores_series})
        plt.subplots(figsize=(5, len(final_scores_df)*0.25))
        s = sns.barplot(data=final_scores_df, orient='h', y='Feature', x='Total Scores')
        s.set_title("Distribution of Total Scores per Column")
        plt.show()

    def plot_final_scores_distribution_by_test(self):
        """
        Display a bar plot representing the distribution of final scores by test.
        """

        if len(self.exceptions_summary_df) == 0:
            print("No exceptions found")
            return

        scores_by_test = pd.DataFrame(self.exceptions_summary_df.groupby('Test ID')['Number of Exceptions'].sum().sort_values())
        scores_by_test['Test ID'] = scores_by_test.index

        plt.subplots(figsize=(5, len(scores_by_test) * 0.25))
        s = sns.barplot(data=scores_by_test, orient='h', y='Test ID', x='Number of Exceptions')
        s.set_title("Distribution of Scores per Test")
        plt.show()

    def quick_report(self):
        """
        A convenience method, which calls several other APIs, to give an overview of the results in a single API.
        """

        def display_api_results(df, title):
            print("\n\n\n")
            if is_notebook():
                display(Markdown(f'# {title}'))
                display(df)
            else:
                print(title + ":")
                print(df)

        def display_plot(func, title):
            print("\n\n\n")
            if is_notebook():
                display(Markdown(f'# {title}'))
            else:
                print(title + ":")
            func()

        display_api_results(self.get_patterns_list(), 'Patterns List (short list only)')
        display_api_results(self.summarize_patterns_by_test_and_feature(), 'Patterns by Test and Feature')
        display_api_results(self.get_exceptions_list(), 'Exceptions List')
        display_api_results(self.summarize_exceptions_by_test_and_feature(), 'Exceptions Summary by Test and Feature')
        display_api_results(self.summarize_exceptions_by_test(), 'Exceptions Summary by Test')
        display_api_results(self.summarize_patterns_and_exceptions(), 'Summary of Patterns and Exceptions (all tests)')
        display_plot(self.plot_final_scores_distribution_by_row, "Final Scores by Row of the Data")
        display_plot(self.plot_final_scores_distribution_by_feature, "Final Scores by Feature")
        display_plot(self.plot_final_scores_distribution_by_test, "Final Scores by Test")

    def display_detailed_results(
            self,
            test_id_list=None,
            col_name_list=None,
            issue_id_list=None,
            pattern_id_list=None,
            row_id_list=None,
            show_patterns=True,
            show_exceptions=True,
            show_short_list_only=True,
            include_examples=True,
            plot_results=True,
            max_shown=-1,
            save_to_disk=False,
            output_folder=None,
            ):
        """
        Loops through each test specified, and each feature specified, and presents a detailed description of each. If
        filters are not specified, the set of identified patterns, with and without exceptions, can be very long in
        some cases, and in these cases, the method will not be able to display them. In this case, additional filters
        should be specified.

        test_id_list: Array of test IDs
            If specified, only these will be displayed. If None, all tests for which there is information to be display
            will be displayed.

        col_name_list: Array of column names, matching the column names in the passed dataframe.
            If specified, only these will be displayed, though the display will include any patterns or exceptions that
            include these columns, regardless of the other columns. If None, all columns for which there is information
            to display will be displayed.

        issue_id_list: Array of Issue IDs
            If specified, only these exceptions will be displayed. If set, patterns will not be shown.

        pattern_id_list: Array of Pattern IDs
            If specified, only these patterns will be displayed. If set, exceptions will not be shown.

        row_id_list: Array of ints, representing row numbers in the original dataset
            If specified, only patterns or exceptions found for these rows will be displayed.

        show_patterns: bool
            If set True, patterns without exceptions will be displayed. If set False, these will not be displayed.

        show_exceptions: bool
            If set True, patterns with exceptions will be displayed. If set False, these will not be displayed.

        show_short_list_only: bool.
            If False, all identified patterns matching the other parameters will be returned. If True, only the tests
            that are most relevant (least noisy) will be displayed as patterns. This does not affect the exceptions
            displayed.

        include_examples: bool
            If True, for any patterns found, examples of a random set of rows (other than cases where row order is
            relevant, in which case a consecutive set of rows will be used), with the relevant set of columns, will
            be display. As well, for any patterns found with exceptions, both a random set of rows that are not flagged
            and that are flagged will be displayed, also with only the relevant columns. May be set False to save
            time and space displaying the results.

        plot_results: bool
            If True, for any tests where plots are possible, one or more plots will be shown displaying the patterns.
            If exceptions are found, they will typically be shown in red. May be  set False to save
            time and space displaying the results.

        max_shown: int
            The maximum total number of patterns and exceptions shown. If no filters are set, the function will return
            if more patterns and/or exceptions are available. If filters are set, and the number is larger, the first
            max_shown will be set. If set to -1, a default will be used, which considers if plots and examples are
            to be displayed. The default is 200 without plots or examples, 100 with either, and 50 with both.

        save_to_disk: bool
            If set True, output will be written to a file on disk

        output_folder: str
            Used only if save_to_disk is True. Indicates the file path to use for the output. If not specified, the
            current folder will be used.
        """

        def print_test_header(test_id, f):
            nonlocal printed_test_header
            nonlocal test_id_list

            if printed_test_header:
                return
            if len(test_id_list) == 1 or (self.execute_list is not None and len(self.execute_list) == 1):
                return

            if f:
                f.write(''.join(['.'] * 100) + "<br>" + os.linesep)
                f.write("<H2>" + test_id + "</H2>" + os.linesep)
            elif is_notebook():
                print(''.join(['.'] * 100))
                display(Markdown(f"### {test_id}"))
            else:
                print("\n\n\n")
                print(stars)
                print(test_id)
                print(stars)

            printed_test_header = True

        def print_column_header(col_name, test_id, f):
            print_line(f)
            if not is_notebook():
                if f:
                    f.write(hyphens + "<br" + os.linesep)
                else:
                    print_line(f)
                    print(hyphens)

            s = f"Columns(s): {self._get_condensed_col_list(test_id, col_name)}"
            if f:
                f.write(s + "<br>" + os.linesep)
            elif is_notebook():
                display(Markdown(f"### {s}"))
            else:
                print(s)

        if (self.orig_df is None) or (len(self.orig_df) == 0):
            print("Empty dataset")
            return

        if ((self.patterns_df is None) or (len(self.patterns_df) == 0)) and \
                ((self.exceptions_summary_df is None) or (len(self.exceptions_summary_df) == 0)):
            print("No patterns or exceptions to display.")
            return

        # If issue_id_list or row_id_list are set, these apply only to exceptions, implying only exceptions should
        # be displayed.
        if issue_id_list or row_id_list:
            show_patterns = False

        if pattern_id_list:
            show_exceptions = False

        if max_shown == -1:
            if save_to_disk:
                max_shown = 50_000
            else:
                max_shown = 200  # This includes patterns & exceptions
            if include_examples:
                max_shown /= 2
            if plot_results:
                max_shown /= 2

        if (test_id_list is None) and (pattern_id_list is None) and (col_name_list is None) and \
                (issue_id_list is None) and (row_id_list is None):
            msg = ("This is beyond the limit to display all at once. Try specifying tests and/or columns to be "
                   "displayed here, specific issues, or row numbers, or setting include_examples and/or "
                   "plot_results to False.")

            if show_patterns and show_exceptions and \
                    ((len(self.exceptions_summary_df) + len(self.patterns_df)) > max_shown):
                print()
                print((f"{len(self.exceptions_summary_df) + len(self.patterns_df)} patterns and exceptions were "
                       f"identified. {msg}"))
                return
            if show_patterns and (len(self.patterns_df) > max_shown):
                print()
                print(f"{len(self.exceptions_summary_df) + len(self.patterns_df)} patterns were identified. {msg}")
                return
            if show_exceptions and (len(self.exceptions_summary_df) > max_shown):
                print()
                print(f"{len(self.exceptions_summary_df)} issues were identified. {msg}")
                return

        # Initialize the folder and file handle for HTML exports if specified
        f = None  # file handle used for HTML export
        if save_to_disk:
            if output_folder is None:
                self.output_folder = os.path.join(os.getcwd(), "Output")
            else:
                self.output_folder = output_folder
            os.makedirs(self.output_folder, exist_ok=True)

            f = open(os.path.join(self.output_folder, "Data_consistency.html"), 'w')
            f.write("<html>" + os.linesep)
            f.write("<head>" + os.linesep)
            f.write("</head>" + os.linesep)
            f.write("<body>" + os.linesep)
            f.write("<h1>Data Consistency Check Results</h1>" + os.linesep)
            f.write("<body>" + os.linesep)

        if test_id_list:
            if len(test_id_list) > 1:
                print_line(f)
                s = f"Displaying results for tests: {str(test_id_list).replace('[','').replace(']','')}"
                print_text(s, f)

            # Check for any tests that are specified that contradict the setting for show_short_list_only
            if show_short_list_only:
                for test_id in test_id_list:
                    if test_id not in self.get_patterns_shortlist():
                        sub_patterns_test = self.patterns_df[self.patterns_df['Test ID'] == test_id]
                        if len(sub_patterns_test):
                            print_text((f"Not displaying patterns without exceptions for {test_id}. This test is not in "
                                        f"the short list and the parameter show_short_list_only was set to True"), f)

            # Check for any invalid test IDs
            for test_id in test_id_list:
                if test_id not in self.get_test_list():
                    print_text(f"{test_id} is not a valid test ID", f)

        if col_name_list:
            print_line(f)
            print_text(f"Displaying results for columns: {col_name_list}", f)

            # Check for any invalid column names
            for col_name in col_name_list:
                if col_name not in self.orig_df.columns:
                    print_text(f"{col_name} is not a valid column name", f)

        if test_id_list is None:
            test_id_list = self.get_test_list()
        if col_name_list is None:
            col_name_list = self.orig_df.columns

        row_id_list_df = None
        if row_id_list:
            # Check the row_id_list is valid
            if (np.array(row_id_list) > self.num_rows).any():
                print_line(f)
                print_text("Row id specified was beyond the length of the dataframe. Use the 0-based row numbers.", f)
                return

            # Create a dataframe representing only the specified rows
            row_id_list_df = self.test_results_df.loc[row_id_list]

        stars = "******************************************************************************"
        hyphens = '----------------------------------------------------------------------------'
        count_shown = 0

        max_shown_msg = (f"**Showing the first {int(max_shown)} findings. To see additional patterns or exceptions, "
                         "specify more specific tests, columns, issue numbers, or row numbers, or increase max_shown**")

        for test_id in test_id_list:
            if self.execute_list and test_id not in self.execute_list:
                continue
            if self.exclude_list and test_id in self.exclude_list:
                continue

            printed_test_header = False

            sub_patterns_test = self.patterns_df[self.patterns_df['Test ID'] == test_id]
            sub_results_summary_test = self.exceptions_summary_df[self.exceptions_summary_df['Test ID'] == test_id]

            # Display patterns that have no exception
            if show_patterns and ((not show_short_list_only) or test_id in self.get_patterns_shortlist()):
                for columns_set in sub_patterns_test['Column(s)'].values:
                    if count_shown >= max_shown:
                        print_line(f)
                        print_text(max_shown_msg, f)
                        return

                    sub_patterns = self.patterns_df[(self.patterns_df['Test ID'] == test_id) &
                                                    (self.patterns_df['Column(s)'] == columns_set)]
                    pattern_columns_arr = self.col_to_original_cols_dict[columns_set]
                    if len(set(col_name_list).intersection(set(pattern_columns_arr))) == 0:
                        continue

                    # If pattern_id_list is specified, check the current pattern is in the list
                    pattern_id = sub_patterns['Pattern ID'].values[0]
                    if pattern_id_list and (pattern_id not in pattern_id_list):
                        continue

                    if len(sub_patterns) > 0:
                        count_shown += 1
                        print_test_header(test_id, f)
                        print_column_header(columns_set, test_id, f)
                        print_text("Pattern found (without exceptions)", f)
                        if test_id in ['PREV_VALUES_DT', 'DECISION_TREE_REGRESSOR', 'DECISION_TREE_CLASSIFIER',
                                       'PREDICT_NULL_DT']:
                            print_text("**Description**:", f)
                            print_text(sub_patterns.iloc[0]['Description of Pattern'], f)
                        else:
                            print_text(f"**Description**: {sub_patterns.iloc[0]['Description of Pattern']}", f)
                        cols = [x.lstrip('"').rstrip('"') for x in columns_set.split(" AND ")]
                        if include_examples:
                            self._display_examples_not_flagged(
                                test_id,
                                cols,
                                columns_set,
                                is_patterns=True,
                                display_info=sub_patterns.iloc[0]['Display Information'],
                                f=f)
                        if plot_results:
                            self._draw_results_plots(
                                test_id,
                                cols,
                                columns_set,
                                show_exceptions=False,
                                display_info=sub_patterns.iloc[0]['Display Information'],
                                f=f)

            # Display patterns with exceptions
            if show_exceptions:
                for columns_set in sub_results_summary_test['Column(s)'].values:
                    if count_shown >= max_shown:
                        print_line(f)
                        print_text(max_shown_msg, f)
                        return

                    if row_id_list:
                        if not row_id_list_df[self.get_results_col_name(test_id, columns_set)].any():
                            continue

                    # If columns_set_arr is specified, only report issues with some overlap of columns with
                    # columns_set_arr. The columns_set in the issues dataframe may be a single string. If so, convert
                    # to an array.
                    issue_columns_arr = self.col_to_original_cols_dict[self.get_results_col_name(test_id, columns_set)]
                    if len(set(col_name_list).intersection(set(issue_columns_arr))) == 0:
                        continue

                    # sub_summary should be one row, representing the current test ID and set of columns
                    sub_summary = self.exceptions_summary_df[
                        (self.exceptions_summary_df['Test ID'] == test_id) &
                        (self.exceptions_summary_df['Column(s)'] == columns_set)]
                    assert len(sub_summary) == 1
                    if len(sub_summary) == 0:
                        continue

                    # If issue_id_list is specified, check the current issue is in the list
                    issue_id = sub_summary['Issue ID'].values[0]
                    if issue_id_list and (issue_id not in issue_id_list):
                        continue

                    count_shown += 1
                    print_test_header(test_id, f)
                    print_column_header(columns_set, test_id, f)
                    print_text(f"**Issue ID**: {issue_id}", f)
                    if test_id in ['RARE_VALUES', 'VERY_SMALL', 'VERY_LARGE', 'VERY_SMALL_ABS', 'LARGE_GIVEN_DATE',
                                   'SMALL_GIVEN_DATE', 'LARGE_GIVEN_VALUE', 'SMALL_GIVEN_VALUE', 'LARGE_GIVEN_PREFIX',
                                   'SMALL_GIVEN_PREFIX', 'LARGE_GIVEN_PAIR', 'SMALL_GIVEN_PAIR']:
                        print_text("Unusual values were found.\n", f)
                    else:
                        print_text("A strong pattern, and exceptions to the pattern, were found.\n", f)
                    if test_id in ['GROUPED_STRINGS', 'GROUPED_STRINGS_BY_NUMERIC']:
                        # These display special output, so the formatting must be preserved.
                        print_text(f"**Description**:",f )
                        print_text(sub_summary.iloc[0]['Description of Pattern'])
                    elif test_id in ['PREV_VALUES_DT', 'DECISION_TREE_REGRESSOR', 'DECISION_TREE_CLASSIFIER',
                                     'PREDICT_NULL_DT']:
                        # These display a decision tree, so the formatting must be preserved.
                        print_text(f"**Description**:", f)
                        print_text(sub_summary.iloc[0]['Description of Pattern'], f)
                    else:
                        multiline_desc = wrap(sub_summary.iloc[0]['Description of Pattern'], 100)
                        print_text(f"**Description**: {'<br>'.join(multiline_desc)}", f)
                    num_exceptions = sub_summary.iloc[0]['Number of Exceptions']
                    print_text((f"**Number of exceptions**: {num_exceptions} "
                                f"({num_exceptions * 100.0 / self.num_rows:.4f}% of rows)"), f)

                    # Provide examples of the pattern and, for exceptions, of the exceptions
                    if include_examples:
                        # Display examples not flagged
                        result_col_name = self.get_results_col_name(test_id, columns_set)
                        cols = self.col_to_original_cols_dict[result_col_name]
                        self._display_examples_not_flagged(
                            test_id,
                            cols,
                            columns_set,
                            is_patterns=False,
                            display_info=sub_summary.iloc[0]['Display Information'],
                            f=f)

                        flagged_df = self._get_rows_flagged(test_id, columns_set)
                        if flagged_df is None:
                            continue
                        print_line(f)
                        if len(flagged_df) > 10:
                            print_text("**Examples of flagged values**:", f)
                        else:
                            print_text("**Flagged values**:", f)
                        display_cols = list(self.col_to_original_cols_dict[self.get_results_col_name(test_id, columns_set)])
                        flagged_df = flagged_df.head(10)
                        self._draw_sample_dataframe(
                            flagged_df[display_cols],
                            test_id,
                            cols,
                            display_info=sub_summary.iloc[0]['Display Information'],
                            is_patterns=False,
                            f=f)
                        print_line(f)

                        # For some tests, we display the rows before and after the flagged rows as well, to provide
                        # context
                        if test_id in ['PREV_VALUES_DT', 'COLUMN_ORDERED_ASC', 'COLUMN_ORDERED_DESC',
                                       'COLUMN_TENDS_ASC', 'COLUMN_TENDS_DESC', 'SIMILAR_PREVIOUS', 'RUNNING_SUM',
                                       'GROUPED_STRINGS', 'GROUPED_STRINGS_BY_NUMERIC']:
                            print_line(f)
                            print_text((f"Showing a flagged example (row {flagged_df.index[0]}) with 5 rows "
                                        f"before and 5 rows after (if available) the flagged row"), f)
                            self._draw_rows_around_flagged_row(
                                flagged_df[display_cols],
                                test_id,
                                cols,
                                display_info=sub_summary.iloc[0]['Display Information'],
                                f=f)

                    # For some tests, we display one or more plots to make the exceptions more clear
                    if plot_results:
                        result_col_name = self.get_results_col_name(test_id, columns_set)
                        cols = self.col_to_original_cols_dict[result_col_name]
                        self._draw_results_plots(
                            test_id,
                            cols,
                            columns_set,
                            show_exceptions=True,
                            display_info=sub_summary.iloc[0]['Display Information'],
                            f=f)

        if save_to_disk:
            f.write("</body>" + os.linesep)
            f.write("</html>" + os.linesep)
            f.close()

    def display_next(self):
        """
        This may be used where there are many results, and we wish to view detailed descriptions of all or most of
        these. This API calls display_detailed_results() for one test at a time, for each test that identified at least
        one pattern (with or without exceptions) during the last call to check_data_quality(). This allows, when working
        with notebooks, for output to be spread over multiple cells, which can make viewing it simpler. Note though,
        where many tests flag patterns, in most cases only a subset of these would be useful to examine in detail,
        though this varies for different projects.
        """

        if self.current_display_test >= len(self.found_tests):
            print("No further test results")
            return

        test_id = self.found_tests[self.current_display_test]

        if is_notebook():
            print(''.join(['.'] * 100))
            display(Markdown(f"### {test_id}"))
        else:
            print("\n\n\n")
            print(test_id)
        print_text(f"**Description**: {self.test_dict[test_id].description}")
        print()

        self.display_detailed_results(test_id_list=[test_id], max_shown=25)
        self.current_display_test += 1

    def display_least_flagged_rows(self, with_results=True, n_rows=10):
        """
        This displays the n_rows rows from the original data with the lowest scores. These are the rows with the least
        flagged issues. This may be called to provide context for the flagged rows. In rare cases, some returned rows
        may have non-zero scores, if all or most rows in the dataset are flagged at least once.

        with_results: bool
            If with_results is False, this displays a single dataframe showing the appropriate subset of the original
            data. If with_results is True, this displays a dataframe per original row, up to n_rows. For each, the
            original data is shown, along with all flagged issues, across all tests on all features.

        n_rows: int
            The maximum number of original rows to present.
        """

        sorted_df = self.test_results_df.sort_values('FINAL SCORE', ascending=True)
        if with_results:
            self._display_rows_with_tests(sorted_df, n_rows)
        else:
            df = self.orig_df.loc[sorted_df.index].head(n_rows).copy()
            df['FINAL SCORE'] = sorted_df['FINAL SCORE'][:len(df)]
            if is_notebook():
                display(df)
            else:
                print(df)

    def display_most_flagged_rows(self, with_results=True, n_rows=10):
        """
        This is similar to display_least_flagged_rows, but displays the rows with the most identified issues.

        with_results: bool
            If True, the flagged rows will be display in separate tables, with additional rows indicating which tests
            flagged which columns. If False, all displayed rows will be displayed in a single table; the flagged
            columns will be highlighted, but there will not be an indication of which tests flagged them.
        n_rows: int
            The maximum number of original rows to present.
        """

        if self.test_results_df is None or len(self.test_results_df) == 0:
            return None

        sorted_df = self.test_results_df.sort_values('FINAL SCORE', ascending=False)
        sorted_df.index = [x[0] if type(x) == tuple else x for x in sorted_df.index]
        if with_results:
            self._display_rows_with_tests(sorted_df, n_rows, check_score=True)
        else:
            row_idxs = np.array(sorted_df.index.to_list()).reshape(1, -1)[0]
            df = self.orig_df.loc[row_idxs].head(n_rows).copy()
            df['FINAL SCORE'] = sorted_df['FINAL SCORE'][:len(df)]
            df = df[df['FINAL SCORE'] > 0]
            if is_notebook():
                display(df.style.apply(styling_flagged_rows,
                                       flagged_cells=self.test_results_by_column_np,
                                       axis=None))
            else:
                print(df)

    ##################################################################################################################
    # Methods to find relationships between the data and the numbers of issues found.
    ##################################################################################################################

    def plot_columns_vs_final_scores(self):
        """
        Used to determine if there are any relationships between column values and the final scores of the rows. This
        displays tables and plots presenting any relationships found.
        """

        def clear_last_plots():
            if n_rows == 1:
                for i in range(num_feats, 4):
                    ax[i].set_visible(False)
            else:
                last_col_used = num_feats % 4
                for i in range(last_col_used, 4):
                    ax[n_rows-1][i].set_visible(False)

        if self.exceptions_summary_df is None or len(self.exceptions_summary_df) == 0:
            print("No exceptions found.")
            return

        df = self.orig_df.copy()
        df['FINAL SCORE'] = self.test_results_df['FINAL SCORE']

        num_feats = len(self.numeric_cols) + len(self.date_cols)
        feats = self.numeric_cols + self.date_cols
        if num_feats > 50:
            print(
                f"There are {num_feats} numeric and date features. Displaying only the 50 with the greatest correlation with the final score"
            )
            corr = {
                col: abs(df[col].astype(float).corr(df['FINAL SCORE'])) for col in feats
            }
            feats = [k for k, _ in sorted(corr.items(), key=lambda x: x[1], reverse=True)[:50]]
            num_feats = len(feats)

        if num_feats > 0:
            n_rows = math.ceil(num_feats / 4)
            fig, ax = plt.subplots(nrows=n_rows, ncols=4, figsize=(14, 4 * n_rows))
            for feat_idx, col_name in enumerate(feats):
                if n_rows == 1:
                    cur_ax = ax[feat_idx]
                else:
                    cur_ax = ax[feat_idx // 4][feat_idx % 4]
                s = sns.scatterplot(data=df, x=df[col_name], y=df['FINAL SCORE'], ax=cur_ax)
                s.set(xlabel=None)
                s.set_title(col_name)
            clear_last_plots()
            plt.suptitle("Relationship of features to Final Score (Numeric and Date features)")
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()

        num_feats = len(self.binary_cols) + len(self.string_cols)
        feats = self.binary_cols + self.string_cols
        if num_feats > 50:
            print(
                f"There are {num_feats} numeric and date features. Displaying only the 50 with the greatest correlation with the final score"
            )
            corr = {}
            for col in feats:
                codes = pd.Categorical(df[col]).codes
                corr[col] = abs(pd.Series(codes).corr(df['FINAL SCORE']))
            feats = [k for k, _ in sorted(corr.items(), key=lambda x: x[1], reverse=True)[:50]]
            num_feats = len(feats)

        if num_feats > 0:
            n_rows = math.ceil(num_feats / 4)
            fig, ax = plt.subplots(nrows=n_rows, ncols=4, figsize=(14, 4 * n_rows))
            for feat_idx, col_name in enumerate(feats):
                vc = self.orig_df[col_name].value_counts()
                if n_rows == 1:
                    cur_ax = ax[feat_idx]
                else:
                    cur_ax = ax[feat_idx // 4][feat_idx % 4]
                if len(vc) > 10:
                    common_vals = []
                    for v_idx in vc.index:
                        if vc[v_idx] > (self.num_rows / 10):
                            common_vals.append(v_idx)
                    if len(common_vals) == 0:
                        continue
                    map_dict = {x: x for x in common_vals}
                    sub_df = df.copy()
                    sub_df[col_name] = df[col_name].map(map_dict)
                    sub_df[col_name] = sub_df[col_name].fillna("Other")
                    s = sns.boxplot(data=sub_df,  x=col_name, y='FINAL SCORE', ax=cur_ax)
                else:
                    s = sns.boxplot(data=df,  x=col_name, y='FINAL SCORE', ax=cur_ax)
                s.set_title(col_name)

            clear_last_plots()
            plt.suptitle("Relationship of features to Final Score (String and Binary features)")
            plt.tight_layout()
            # See https://stackoverflow.com/questions/8248467/tight-layout-doesnt-take-into-account-figure-suptitle
            plt.subplots_adjust(top=0.90)
            plt.show()

    ##################################################################################################################
    ##################################################################################################################
    # Private methods to display tables of example rows from the original data
    ##################################################################################################################
    def _draw_sample_dataframe(self, df, test_id, cols, display_info, is_patterns, f):
        """
        Adds columns to the passed dataframe as is necessary to explain the pattern, then displays the dataframe.
        Many tests have additional columns which can make the pattern between the columns more clear.

        Also, for some test, sets the column order.

        df: dataframe
            the dataframe of the rows to display. Typically 10 rows.

        test_id: string
            the id of the test that flagged these rows

        cols: array of strings
            one set of columns flagged by this test

        display_info: dictionary
            dictionary of information specific to these columns for this test. Also used to display plots.

        is_patterns: bool
            Indicates if this is being called when displaying patterns or exceptions

        f: file handle
            Output file for HTML report
        """

        col_name, col_name_1, col_name_2, col_set = "", "", "", ""

        if len(cols) == 1:
            col_name = cols[0]
        if len(cols) == 2:
            col_name_1, col_name_2 = cols
        if len(cols) == 3:
            col_name_1, col_name_2, col_name_3 = cols
        if len(cols) >= 3:
            col_set = cols
            source_cols = col_set[:-1]

        # Add the additional columns to explain the relationship
        df = df.copy()
        if test_id in ['LARGER_THAN_SUM', 'CONSTANT_SUM', 'BINARY_MATCHES_SUM']:
            vals1 = convert_to_numeric(df[col_name_1], 0)
            vals2 = convert_to_numeric(df[col_name_2], 0)
            df['SUM'] = (vals1 + vals2).values
        elif test_id in ['RARE_VALUES']:
            df["Count of Value"] = [display_info['counts'][x] for x in df[col_name].astype(str)]
        elif test_id in ['NUMBER_DECIMALS']:
            df['Number decimals'] = [-1 if is_missing(x) else get_num_decimal_digits(x) for x in df[col_name]]
            df[col_name] = df[col_name].astype(str)
        elif test_id in ['ROUNDING']:
            vals = df[col_name].fillna(-9595959484)
            vals = convert_to_numeric(vals, 0)
            vals = vals.astype(int)
            vals = vals.astype(str)
            s = vals.str.replace('.0', '', regex=False).str.len() - \
                vals.str.replace('.0', '', regex=False).str.strip('0').str.len()
            vals = vals.replace('-9595959484', np.nan)
            df['Number Zeros'] = s.values
        elif test_id in ['SUM_OF_COLUMNS']:
            if display_info and display_info['operation'] == "plus":
                df[f"SUM PLUS {display_info['amount']}"] = df[source_cols].sum(axis=1) + display_info['amount']
            elif display_info and display_info['operation'] == "times":
                df[f"SUM TIMES {display_info['amount']}"] = df[source_cols].sum(axis=1) * display_info['amount']
            else:
                df['SUM'] = df[source_cols].sum(axis=1)
        elif test_id in ['SIMILAR_TO_DIFF', 'LARGER_THAN_ABS_DIFF', 'CONSTANT_DIFF']:
            df['ABSOLUTE DIFFERENCE'] = abs(df[col_name_1] - df[col_name_2])
        elif test_id in ['SIMILAR_TO_PRODUCT', 'CONSTANT_PRODUCT']:
            vals1 = convert_to_numeric(df[col_name_1], 0)
            vals2 = convert_to_numeric(df[col_name_2], 0)
            df['PRODUCT'] = (vals1 * vals2).values
        elif test_id in ['SIMILAR_TO_RATIO', 'CONSTANT_RATIO', 'EVEN_MULTIPLE']:
            vals1 = convert_to_numeric(df[col_name_1], 0)
            vals2 = convert_to_numeric(df[col_name_2], 0)
            df['DIVISION RESULTS'] = [safe_div(x, y) for x, y in zip(vals1, vals2)]
        elif test_id in ['MEAN_OF_COLUMNS']:
            df['MEAN'] = df[source_cols].mean(axis=1)
        elif test_id in ['MIN_OF_COLUMNS']:
            df['MIN'] = df[source_cols].min(axis=1)
        elif test_id in ['MAX_OF_COLUMNS']:
            df['MAX'] = df[source_cols].max(axis=1)
        elif test_id in ['RARE_PAIRS_FIRST_WORD_VAL', 'LARGE_GIVEN_PREFIX', 'SMALL_GIVEN_PREFIX']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            df[f'{col_name_1} FIRST WORD'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['LEADING_WHITESPACE']:
            df['NUM LEADING SPACES'] = df[col_name].astype(str).str.len() - df[col_name].astype(str).str.lstrip(' ').str.len()
        elif test_id in ['TRAILING_WHITESPACE']:
            df['NUM TRAILING SPACES'] = df[col_name].astype(str).str.len() - df[col_name].astype(str).str.rstrip(' ').str.len()
        elif test_id in ['MULTIPLE_OF_CONSTANT']:
            if is_patterns:
                df['NUM MULTIPLES'] = round(df[col_name] / display_info['value'])
            else:
                df['NUM MULTIPLES'] = df[col_name] / display_info['value']
        elif test_id in ['UNUSUAL_DAY_OF_WEEK']:
            df['Day of Week'] = pd.to_datetime(df[col_name]).dt.strftime('%A')
        elif test_id in ['UNUSUAL_DAY_OF_MONTH']:
            df['Day of Month'] = pd.to_datetime(df[col_name]).dt.day
        elif test_id in ['UNUSUAL_MONTH']:
            df['Month'] = pd.to_datetime(df[col_name]).dt.month
        elif test_id in ['UNUSUAL_HOUR']:
            df['Hour'] = pd.to_datetime(df[col_name]).dt.hour
        elif test_id in ['UNUSUAL_MINUTES']:
            df['Minutes'] = pd.to_datetime(df[col_name]).dt.minute
        elif test_id in ['CONSTANT_GAP', 'LARGE_GAP', 'SMALL_GAP', 'LATER']:
            df['Gap'] = pd.to_datetime(df[col_name_2]) - pd.to_datetime(df[col_name_1])
        elif test_id in ['NUMBER_ALPHA_CHARS']:
            df['Num Alpha Chars'] = df[col_name].astype(str).apply(lambda x: len([e for e in x if e.isalpha()]))
        elif test_id in ['NUMBER_NUMERIC_CHARS']:
            df['Num Numeric Chars'] = df[col_name].astype(str).apply(lambda x: len([e for e in x if e.isdigit()]))
        elif test_id in ['NUMBER_ALPHANUMERIC_CHARS']:
            df['Num Alpha-Numeric Chars'] = df[col_name].astype(str).apply(lambda x: len([e for e in x if e.isalnum()]))
        elif test_id in ['NUMBER_NON-ALPHANUMERIC_CHARS']:
            df['Num Non-Alpha-Numeric Chars'] = display_info['test_series'].loc[df.index]
        elif test_id in ['NUMBER_CHARS', 'MANY_CHARS', 'FEW_CHARS']:
            df['Num Chars'] = df[col_name].astype(str).str.len()
        elif test_id in ['FIRST_CHAR_ALPHA', 'FIRST_CHAR_NUMERIC', 'FIRST_CHAR_SMALL_SET', 'FIRST_CHAR_UPPERCASE',
                         'FIRST_CHAR_LOWERCASE']:
            df['First Char'] = df[col_name].astype(str).str.lstrip().str.slice(0,1)
        elif test_id in ['LAST_CHAR_SMALL_SET']:
            df['Last Char'] = df[col_name].astype(str).str.rstrip().str[-1:]
        elif test_id in ['FIRST_WORD_SMALL_SET']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            df['First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['LAST_WORD_SMALL_SET']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            df['Last Word'] = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['NUMBER_WORDS']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            df['Num Words'] = [len(x) for x in word_arr]
        elif test_id in ['LONGEST_WORDS']:
            col_vals = df[col_name].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            word_lens_arr = [[len(w) for w in x] for x in word_arr]
            df['Longest Word Len'] = [max(x) if len(x) > 0 else 0 for x in word_lens_arr]
        elif test_id in ['RARE_PAIRS_FIRST_CHAR', 'SAME_FIRST_CHARS']:
            df[f'{col_name_1} First Char'] = df[col_name_1].astype(str).str[:1]
            df[f'{col_name_2} First Char'] = df[col_name_2].astype(str).str[:1]
        elif test_id in ['RARE_PAIRS_FIRST_WORD', 'SAME_FIRST_WORD']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            df[f'{col_name_1} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            col_vals = df[col_name_2].astype(str).apply(replace_special_with_space)
            df[f'{col_name_2} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['SAME_LAST_WORD']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            df[f'{col_name_1} Last Word'] = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
            col_vals = df[col_name_2].astype(str).apply(replace_special_with_space)
            df[f'{col_name_2} Last Word'] = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
        elif test_id in ['SIMILAR_NUM_CHARS']:
            df[f'{col_name_1} Num Chars'] = df[col_name_1].astype(str).str.len()
            df[f'{col_name_2} Num Chars'] = df[col_name_2].astype(str).str.len()
        elif test_id in ['SIMILAR_NUM_WORDS']:
            col_vals = df[col_name_1].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            df[f'{col_name_1} Num Words'] = [len(x) for x in word_arr]
            col_vals = df[col_name_2].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            df[f'{col_name_2} Num Words'] = [len(x) for x in word_arr]
        elif test_id in ['SMALL_VS_CORR_COLS', 'LARGE_VS_CORR_COLS']:
            for c in display_info['cluster']:
                df[c] = self.orig_df[c].loc[df.index]
            df = df[list(df.columns[1:]) + [df.columns[0]]]  # Ensure the relevant column is the rightmost
        elif test_id in ['MISSING_VALUES_PER_ROW']:
            df['Number Missing Values'] = df.isna().sum(axis=1)
        elif test_id in ['ZERO_VALUES_PER_ROW']:
            df['Number Zero Values'] = df.applymap(lambda x: (x is None) or (x == 0)).sum(axis=1)
        elif test_id in ['UNIQUE_VALUES_PER_ROW']:
            df['Number Unique Values'] = df.apply(lambda x: len(set(x)), axis=1)
        elif test_id in ['NEGATIVE_VALUES_PER_ROW']:
            df['Number Negative Values'] = df.applymap(lambda x: isinstance(x, numbers.Number) and x < 0).sum(axis=1)
        elif test_id in ['DECISION_TREE_CLASSIFIER', 'DECISION_TREE_REGRESSOR', 'PREV_VALUES_DT', 'LINEAR_REGRESSION',
                         'PREDICT_NULL_DT']:
            df["PREDICTION"] = display_info['Pred'].loc[df.index]
        elif test_id in ['CORRELATED_NUMERIC', 'CORRELATED_DATES']:
            df['Column 1 Percentile'] = display_info['col_1_percentiles'].loc[df.index]
            df['Column 2 Percentile'] = display_info['col_2_percentiles'].loc[df.index]
        elif test_id in ['UNUSUAL_ORDER_MAGNITUDE']:
            df['ORDER OF MAGNITUDE'] = display_info['order of magnitude'][df.index]
        elif test_id in ['RUNNING_SUM']:
            df['RUNNING SUM'] = display_info['RUNNING SUM'][df.index]
        elif test_id in ['SMALL_AVG_RANK_PER_ROW', 'LARGE_AVG_RANK_PER_ROW']:
            df['AVG PERCENTILE  '] = display_info['percentiles'][df.index]
        elif test_id in ['C_IS_A_OR_B']:
            df['SAME AS'] = pd.Series(display_info['Same Column']).loc[df.index]
        elif test_id in ['SAME_SPECIAL_CHARS']:
            df['SPECIAL CHARS A'] = pd.Series(display_info['Special Chars A']).loc[df.index]
            df['SPECIAL CHARS B'] = pd.Series(display_info['Special Chars B']).loc[df.index]
        elif test_id in ['TWO_PAIRS']:
            df['A and B Match'] = pd.Series(display_info['match_1_2_arr']).loc[df.index]
            df['C and D Match'] = pd.Series(display_info['match_3_4_arr']).loc[df.index]
        elif test_id in ['BINARY_TWO_OTHERS_MATCH']:
            df['MATCH'] = pd.Series(display_info['Match']).loc[df.index]
        elif test_id in ['MUCH_LARGER']:
            df['RATIO'] = [safe_div(x, y) for x, y in zip(df[col_name_2].astype(float), df[col_name_1].astype(float))]
        elif test_id in ['LARGER_DIFF_RANGE', 'LARGER_SAME_RANGE']:
            if len(cols) == 2:
                df['DIFF'] = df[col_name_1].astype(float) - df[col_name_2].astype(float)
        elif test_id in ['SIMILAR_WRT_RATIO']:
            df['RATIO'] = pd.Series(display_info['Ratio']).loc[df.index]
        elif test_id in ['SIMILAR_WRT_DIFF']:
            df['DIFF'] = pd.Series(display_info['Diff']).loc[df.index]
        elif test_id in ['CORRELATED_ALPHA_ORDER']:
            df['Percentile Column 1'] = pd.Series(display_info['col_1_percentiles']).loc[df.index]
            df['Percentile Column 2'] = pd.Series(display_info['col_2_percentiles']).loc[df.index]
        elif test_id in ['LARGE_GIVEN_DATE', 'SMALL_GIVEN_DATE']:
            df[f'Bin Number {cols[0]}'] = pd.Series(display_info['bin_assignments']).loc[df.index]
        elif test_id in ['FEW_NEIGHBORS']:
            df['Closest smaller value'] = pd.Series(display_info['prev_val']).loc[df.index]
            df['Closest larger value'] = pd.Series(display_info['next_val']).loc[df.index]
        elif test_id in ['FEW_WITHIN_RANGE']:
            df['Number in Range'] = pd.Series(display_info['Number in Range']).loc[df.index]
        elif test_id in ['NONPRINTABLE_CHARS']:
            for i in df.index:
                v = str([x for x in df.loc[i, cols[0]] if not x.isprintable()])
                df.loc[i, 'Non-Printable Chars'] = v

        # Set the column order
        if test_id in ['LARGER_DIFF_RANGE', 'LARGER_SAME_RANGE', 'MUCH_LARGER']:
            col_medians = [self.column_medians[c] for c in cols]
            sorted_cols = np.array(cols)[np.argsort(col_medians)]
            df2 = df[sorted_cols]
            df = df2

        # Set the row order
        if test_id in ['SMALL_GIVEN_DATE', 'LARGE_GIVEN_DATE']:
            df = df.sort_values(df.columns[-1])
        else:
            df = df.sort_index()

        pd.options.display.float_format = '{:f}'.format
        if f:
            f.write(df.to_html())
            f.write("<br><br>")
        elif is_notebook():
            display(df)
        else:
            print(df)

    def _draw_rows_around_flagged_row(self, df, test_id, cols, display_info, f):
        """
        Called by display_detailed_results() to provide context for one flagged row, where the row order is relevant.
        This displays, where possible, 5 rows before and 5 rows after the flagged row.
        """

        row_id = df.index[0]
        row_start = max(0, row_id-5)
        row_end = min(self.num_rows, row_id+5)
        neighborhood_df = self.orig_df.iloc[row_start:row_end][df.columns]
        self._draw_sample_dataframe(
            neighborhood_df,
            test_id,
            cols,
            display_info=display_info,
            is_patterns=False,
            f=f)

    def _get_sample_not_flagged(self, test_id, col_name, n_examples=10, show_consecutive=False, sort_col=None,
                                 is_patterns=False, display_info=None, f=None):
        """
        Called by __display_examples_not_flagged()

        Return a random set of n_examples values from the specified columns that were not flagged by the
        specified test in the specified columns. This handles specific tests by balancing the the rows displayed
        in order to include examples of different types of rows.

        test_id: bool
            If test_id is None, this returns values not flagged by any test.

        col_name: string
            Single column or column set

        n_examples: int
            The maximum number of examples to return.

        show_consecutive: bool
            If True, this will return consecutive rows. This used for tests where row order is relevant.

        sort_col: string
            If specified, sort by this before returning a consecutive set of rows, if show_consecutive is True.

        is_patterns: bool
            Set True if this is being called to display examples of a pattern, in which case there are no exceptions.

        display_info: dict
            Set by individual test to describe each row with respect to the pattern

        f: file handle
        """

        if is_patterns:
            cols = [x.lstrip('"').rstrip('"') for x in col_name.split(" AND ")]
        else:
            results_col_name = self.get_results_col_name(test_id, col_name)
            cols = self.col_to_original_cols_dict[results_col_name]

        # If there are no values flagged for this test in this feature, there will not be a column in
        # test_results_df. In this case, return any values.
        if not is_patterns and results_col_name not in self.test_results_df.columns:
            assert False, "Should not happen"
            return self.orig_df[col_name].sample(n=n_examples, random_state=0)

        df = None
        if test_id in ['BINARY_SAME', 'BINARY_OPPOSITE', 'BINARY_IMPLIES', 'BINARY_AND', 'BINARY_OR',
                       'BINARY_XOR', 'BINARY_NUM_SAME', 'BINARY_TWO_OTHERS_MATCH', 'BINARY_MATCHES_SUM']:
            if len(cols) == 2 and cols[0] in self.binary_cols and cols[1] in self.binary_cols:
                v0_0, v0_1 = self.column_unique_vals[cols[0]]
                v1_0, v1_1 = self.column_unique_vals[cols[1]]
                df_v00 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_0) &
                    (self.orig_df[cols[1]] == v1_0)].head(n_examples // 4)
                df_v01 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_0) &
                    (self.orig_df[cols[1]] == v1_1)].head(n_examples // 4)
                df_v10 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_1) &
                    (self.orig_df[cols[1]] == v1_0)].head(n_examples // 4)
                df_v11 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_1) &
                    (self.orig_df[cols[1]] == v1_1)].head(n_examples // 4)
                df = pd.concat([df_v00, df_v01, df_v10, df_v11])[cols]
            elif len(cols) == 3 and cols[0] in self.binary_cols and \
                    cols[1] in self.binary_cols and cols[2] in self.binary_cols:
                v0_0, v0_1 = self.orig_df[cols[0]].dropna().unique()
                v1_0, v1_1 = self.orig_df[cols[1]].dropna().unique()
                v2_0, v2_1 = self.orig_df[cols[2]].dropna().unique()
                df_v000 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_0) &
                    (self.orig_df[cols[1]] == v1_0) &
                    (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                df_v001 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_0) &
                    (self.orig_df[cols[1]] == v1_0) &
                    (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                df_v010 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_0) &
                    (self.orig_df[cols[1]] == v1_1) &
                    (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                df_v011 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_0) &
                    (self.orig_df[cols[1]] == v1_1) &
                    (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                df_v100 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_1) &
                    (self.orig_df[cols[1]] == v1_0) &
                    (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                df_v101 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_1) &
                    (self.orig_df[cols[1]] == v1_0) &
                    (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                df_v110 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_1) &
                    (self.orig_df[cols[1]] == v1_1) &
                    (self.orig_df[cols[2]] == v2_0)].head(n_examples // 4)
                df_v111 = self.orig_df[
                    (self.orig_df[cols[0]] == v0_1) &
                    (self.orig_df[cols[1]] == v1_1) &
                    (self.orig_df[cols[2]] == v2_1)].head(n_examples // 4)
                df = pd.concat([df_v000, df_v001, df_v010, df_v011, df_v100, df_v101, df_v110, df_v111])[cols]
            elif len(cols) == 3 and cols[2] in self.binary_cols:
                v0, v1 = self.orig_df[cols[2]].dropna().unique()
                va = self.orig_df[cols[0]].dropna().value_counts().values[0]
                df_v0 = self.orig_df[self.orig_df[cols[2]] == v0] #.head(n_examples // 2)
                df_v1 = self.orig_df[self.orig_df[cols[2]] == v1] #.head(n_examples // 2)
                df_v0a = df_v0[df_v0[cols[0]] == va].head(n_examples // 4)
                df_v0b = df_v0[df_v0[cols[0]] != va].head((n_examples // 2) - len(df_v0a))
                df_v1a = df_v1[df_v1[cols[0]] == va].head(n_examples // 4)
                df_v1b = df_v1[df_v1[cols[0]] != va].head((n_examples // 2) - len(df_v1a))
                df = pd.concat([df_v0a, df_v0b, df_v1a, df_v1b])[cols]
        elif test_id in ['MATCHED_ZERO', 'MATCHED_ZERO_MISSING', 'MATCHED_SET_ZERO_NON_ZERO']:
            # Show where col1 is zero and non-zero
            col_name_1 = cols[0]
            col_name_2 = cols[1]
            df_v_zero = self.orig_df[cols][(self.orig_df[col_name_1] == 0) & (self.orig_df[col_name_1].notna())].head(n_examples // 2)
            if len(df_v_zero) < (n_examples // 2):
                df_v_zero = pd.concat([df_v_zero, self.orig_df[cols][(self.orig_df[col_name_1] == 0) & (self.orig_df[col_name_1].isna())].head((n_examples // 2) - len(df_v_zero))])
            df_v_non_zero = self.orig_df[cols][(self.orig_df[col_name_1] != 0) &
                                               (self.orig_df[col_name_1].notna())].head(n_examples - len(df_v_zero))
            df = pd.concat([df_v_zero, df_v_non_zero])
        elif test_id in ['SAME_OR_CONSTANT']:
            # Show where col1 == col2 and where doesn't (in both cases where col1 is not null)
            col_name_1, col_name_2 = cols
            df_same = self.orig_df[cols][
                (self.orig_df[col_name_1] == self.orig_df[col_name_2]) & self.orig_df[col_name_1].notna()
                ].head(n_examples // 2)
            df_not_same = self.orig_df[cols][
                (self.orig_df[col_name_1] != self.orig_df[col_name_2]) & self.orig_df[col_name_1].notna()
                ].head(n_examples // 2)
            df = pd.concat([df_same, df_not_same])
        elif test_id in ['POSITIVE']:
            # Show where col == 0 and where is > 0
            col_name_1 = cols[0]
            df_v_zero = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] == 0)].head(n_examples // 2)
            df_v_pos = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] > 0)].head(n_examples - len(df_v_zero))
            df = pd.concat([df_v_zero, df_v_pos])
        elif test_id in ['NEGATIVE']:
            # Show where col == 0 and where is < 0
            col_name_1 = cols[0]
            df_v_zero = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] == 0)].head(n_examples // 2)
            df_v_neg = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] < 0)].head(n_examples - len(df_v_zero))
            df = pd.concat([df_v_zero, df_v_neg])
        elif test_id in ['MATCHED_SET_POS_NEG']:
            # Show where col1 is positive and negative
            col_name_1 = cols[0]
            df_v_pos = self.orig_df[cols][(self.orig_df[col_name_1] > 0)].head(n_examples // 2)
            df_v_neg = self.orig_df[cols][(self.orig_df[col_name_1] < 0)].head(n_examples - len(df_v_pos))
            df = pd.concat([df_v_pos, df_v_neg])
        elif test_id in ['EVEN_MULTIPLE', 'MATCHED_MISSING', 'OPPOSITE_MISSING']:
            # Show where col 1 is null and non-null
            col_name_1, col_name_2 = cols[0], cols[1]
            df_v_null = self.orig_df[cols][(self.orig_df[col_name_1].apply(is_missing))].head(n_examples // 2)
            num_non_null = n_examples - len(df_v_null)
            df_v_non_null = self.orig_df[cols][(~self.orig_df[col_name_1].apply(is_missing))].head(num_non_null)
            df = pd.concat([df_v_null, df_v_non_null])
        elif test_id in ['PREDICT_NULL_DT']:
            # Show where target column is null and non-null
            target_col = cols[-1]
            df_v_null = self.orig_df[cols][(self.orig_df[target_col].apply(is_missing))].head(n_examples // 2)
            num_non_null = n_examples - len(df_v_null)
            df_v_non_null = self.orig_df[cols][(~self.orig_df[target_col].apply(is_missing))].head(num_non_null)
            df = pd.concat([df_v_null, df_v_non_null])
        elif test_id in ['C_IS_A_OR_B']:
            # Show examples where C matches both A and B, where the same column is col1 (where it is and is not null),
            # and where the same column is col2 (where it is and is not null).
            df_both = self.orig_df[np.array(display_info['Same Column']) == 'BOTH'].head(n_examples // 3)
            df_v1_null = self.orig_df[(np.array(display_info['Same Column']) == cols[0]) &
                                      (self.orig_df[cols[2]].isna())].head(n_examples // 4)
            df_v1_not_null = self.orig_df[(np.array(display_info['Same Column']) == cols[0]) &
                                          (self.orig_df[cols[2]].notna())].head((n_examples // 2) -
                                                                                len(df_v1_null) - (len(df_both) // 2))
            df_v2_null = self.orig_df[(np.array(display_info['Same Column']) == cols[1]) &
                                      (self.orig_df[cols[2]].isna())].head(n_examples // 4)
            df_v2_not_null = self.orig_df[(np.array(display_info['Same Column']) == cols[1]) &
                                          (self.orig_df[cols[2]].notna())].head((n_examples // 2) -
                                                                                len(df_v2_null) - (len(df_both) // 2))
            df = pd.concat([df_both, df_v1_null, df_v1_not_null, df_v2_null, df_v2_not_null])[cols]
        elif test_id in ['TWO_PAIRS']:
            # Show where the first pair match and where they do not
            df_match = self.orig_df[np.array(display_info['match_1_2_arr']) == True].head(n_examples // 2)
            df_not_match = self.orig_df[np.array(display_info['match_1_2_arr']) == False].head(n_examples - len(df_match))
            df = pd.concat([df_match, df_not_match])[cols]

        # If we do not yet have a df (the test was not specified above, or no rows matched the conditions), we
        # create a df simply trying to reduce the number of Null values and showing unique values in the last column.
        if not show_consecutive and ((df is None) or df.empty):
            # Test that test_results_df does not have duplicate values in the index
            assert (self.test_results_df is None) or \
                   (len(self.test_results_df.index) == len(set(self.test_results_df.index)))
            cols = list(cols)  # Ensure cols is not in tuple format
            df = self.orig_df[cols]

            # Collect rows with non-null values, other than for columns that are almost all null
            for col in cols:
                sub_df = self.orig_df.loc[df.index]
                mask = sub_df[col].notna()
                if mask.tolist().count(True) >= 5:
                    df = df[mask]

            # If there are too few rows, collect additional rows.
            if len(df) < n_examples:
                df = pd.concat([df, self.orig_df[cols].sample(n_examples - len(df))])
                df = df[~df.index.duplicated(keep='first')]

            # Try to get a good set of unique values
            if test_id in ['UNIQUE_VALUES']:
                df = df.sample(n=n_examples)
            else:
                last_col = cols[-1]
                vc = df[last_col].value_counts(dropna=False)
                vals = list(vc.index[:11])  # Try to cover the 3 to 10 most common values
                num_vals = len(vals)
                dfs_arr = []
                num_examples_found = 0
                num_per_value = 1
                if num_vals < n_examples:
                    num_per_value = n_examples // num_vals
                for v in vals:
                    if num_examples_found >= n_examples:
                        break
                    if v == v:
                        df_v = df[cols][(df[last_col] == v)].head(num_per_value)
                    else:
                        df_v = df[cols][(df[last_col].isna())].head(num_per_value)
                    num_examples_found += len(df_v)
                    dfs_arr.append(df_v)
                df = pd.concat(dfs_arr)

        if show_consecutive:
            assert df is None
            df = self.orig_df[cols]
            start_point = np.random.randint(0, len(df) - n_examples)
            if sort_col:
                df = df.sort_values(sort_col).iloc[start_point: start_point + n_examples]
            else:
                df = df.iloc[start_point: start_point + n_examples]
        else:
            # If we do not return a consecutive set of rows, df is likely of size n_examples. We ensure it is of size
            # n_examples, then sort it randomly.
            if len(df) < n_examples:
                df = pd.concat([df, self.orig_df[cols].sample(n_examples - len(df))])
                df = df[~df.index.duplicated(keep='first')]
            df = df.sample(n=min(len(df), n_examples), random_state=0)

        # Remove rows that were flagged. If is_patterns is True, no rows were flagged, and we skip this check.
        if not is_patterns:
            sub_df = self.test_results_df.loc[df.index]
            mask = sub_df[results_col_name] == 0
            df = df[mask]

        return df

    def _display_examples_not_flagged(self, test_id, cols, columns_set, is_patterns, display_info, f):
        """
        Called by display_detailed_results(). This prints a set of rows that were not flagged. May be called in cases
        where some rows were flagged, or where none were.
        """

        # Do not show examples for some tests
        if is_patterns and test_id in ['UNIQUE_VALUES']:
            print_text("Examples are not shown for this pattern.", f)
            return

        if test_id in ['MISSING_VALUES_PER_ROW', 'ZERO_VALUES_PER_ROW',
                       'NEGATIVE_VALUES_PER_ROW', 'GROUPED_STRINGS']:
            print_text("Examples are not shown for this pattern.", f)
            return

        show_consecutive = test_id in ['PREV_VALUES_DT', 'COLUMN_ORDERED_ASC', 'COLUMN_ORDERED_DESC',
                                       'COLUMN_TENDS_ASC', 'COLUMN_TENDS_DESC', 'SIMILAR_PREVIOUS', 'RUNNING_SUM',
                                       'GROUPED_STRINGS_BY_NUMERIC']

        sort_col = None
        if test_id in ['GROUPED_STRINGS_BY_NUMERIC']:
            sort_col = cols[0]

        consecutive_str = ""
        if show_consecutive:
            sort_msg = ""
            if sort_col:
                sort_msg = f' sorted by {sort_col}'
            consecutive_str = f" (showing a consecutive set of rows{sort_msg})"

        print_line(f)
        if is_patterns:
            print_text(f"**Examples{consecutive_str}**:", f)
        else:
            print_text(f"**Examples of values NOT flagged{consecutive_str}**:", f)

        vals = self._get_sample_not_flagged(
            test_id,
            columns_set,
            show_consecutive=show_consecutive,
            sort_col=sort_col,
            is_patterns=is_patterns,
            display_info=display_info,
            f=f)
        self._draw_sample_dataframe(vals, test_id, cols, display_info, is_patterns, f)

    ##################################################################################################################
    # Private helper methods to support outputting the results of the analysis in various ways
    ##################################################################################################################

    def _display_rows_with_tests(self, sorted_df, n_rows, check_score=False):
        """
        Called by display_most_flagged_rows() and display_least_flagged_rows()

        Display a set of dataframes, one per row in the original data, up to n_rows rows, each including the original
        row and the issues found in it, across all tests on all features.

        sorted_df: dataframe
            a sorted version of self.test_results_df
        n_rows: int
            The maximum number of rows from the original data to display
        check_score: bool
            if True, only rows with scores above zero will be displayed
        """

        flagged_idx_arr = sorted_df.index[:10]
        for row_idx in flagged_idx_arr[:n_rows]:
            if check_score and sorted_df.loc[row_idx]['FINAL SCORE'] == 0:
                print(f"The remaining rows have no flagged issues: cannot display {n_rows} flagged rows.")
                return

            # Get the row as it appears in the original data
            orig_row = self.orig_df.loc[row_idx:row_idx]

            # Insert a column to indicate the IDs of the tests that have flagged this row
            orig_row.insert(0, 'Test ID', '')

            colour_cells = [False] * len(self.orig_df.columns)

            # Loop through all tests, and add a row to the output for any that have flagged this row
            for test_id in self.get_test_list():
                test_row = [test_id] + [""] * len(self.orig_df.columns)

                # There may be multiple columns / column sets which have flagged this row.
                for column_set in self.exceptions_summary_df['Column(s)'].unique():
                    result_col_name = self.get_results_col_name(test_id, column_set)
                    if result_col_name not in self.test_results_df.columns:
                        continue
                    for column_name in self.col_to_original_cols_dict[result_col_name]:
                        if self.test_results_df[result_col_name][row_idx]:
                            column_idx = np.where(self.orig_df.columns == column_name)[0][0]
                            test_row[column_idx+1] = u'\u2714'  # Checkmark symbol
                            colour_cells[column_idx] = True
                if test_row.count(u'\u2714'):
                    orig_row = orig_row.append(pd.DataFrame(test_row, index=orig_row.columns).T)
            orig_row = orig_row.reset_index()
            orig_row = orig_row.drop(columns=['index'])

            # Display the dataframe representing this row from the original data
            print()
            if is_notebook():
                display(Markdown(f"**Row: {row_idx} " + u'\u2014' + f" Final Score: {sorted_df.loc[row_idx]['FINAL SCORE']}**"))
                display(orig_row.style.apply(styling_orig_row, row_idx=0, flagged_arr=colour_cells, axis=None))
            else:
                print(f"Row: {row_idx} Final Score: {sorted_df.loc[row_idx]['FINAL SCORE']}")
                print(orig_row.to_string(index=False))
            print()

    ##################################################################################################################
    # HTML Export
    ##################################################################################################################

    def gpt_export_html(self):

        def export_dataframes_to_html(dataframes, output_file):
            """
            Export multiple DataFrames to a single HTML file.

            Args:
                dataframes (list): List of DataFrames to export.
                output_file (str): Output file path for the HTML file.
            """

            # Create an HTML writer
            with open(output_file, 'w') as f:

                # Write the HTML header
                f.write('<html>\n<head>\n')
                f.write('<style>.hidden { display: none; }</style>\n')
                f.write('<script>\n')
                f.write('function toggleTable(tableId) {\n')
                f.write('\tvar table = document.getElementById(tableId);\n')
                f.write('\tvar button = document.getElementById("button-" + tableId);\n')
                f.write('\tif (table.classList.contains("hidden")) {\n')
                f.write('\t\ttable.classList.remove("hidden");\n')
                f.write('\t\tbutton.textContent = "Hide Table";\n')
                f.write('\t} else {\n')
                f.write('\t\ttable.classList.add("hidden");\n')
                f.write('\t\tbutton.textContent = "Show Table";\n')
                f.write('\t}\n')
                f.write('}\n')
                f.write('</script>\n')
                f.write('</head>\n<body>\n')

                # Write each DataFrame to the HTML file
                for i, df in enumerate(dataframes):
                    table_id = f'table-{i}'
                    button_id = f'button-{table_id}'
                    f.write(f'<h2>{df.name}</h2>\n')
                    f.write(f'<button id="{button_id}" onclick="toggleTable(\'{table_id}\')">Hide Table</button>\n')
                    f.write(f'<table id="{table_id}" class="hidden">\n')
                    f.write(df.to_html(index=False))
                    f.write('</table>\n')
                    f.write('<br>\n')

                # Write the HTML footer
                f.write('</body>\n</html>')

        print("get_export_html() not supported in this version")
        return

        df1 = self.summarize_patterns_by_test_and_feature(all_tests=True)
        df1.name = 'Summary 1'

        df2 = self.summarize_patterns_by_test_and_feature(all_tests=False)
        df2.name = 'Summary 2'

        output_file = "output.html"
        export_dataframes_to_html([df1, df2], output_file)

    def export_html(self, test_id_list=None, output_file: str = "Data_consistency.html"):
        """Export patterns and exceptions to an HTML report.

        Args:
            test_id_list: Optional list of test IDs to include. If ``None`` all
                available tests are exported.
            output_file: Output path for the generated HTML file.
        """

        def print_test_header(test_id, section_name, f):
            nonlocal test_id_list
            assert section_name in ['patterns', 'exceptions']

            f.write("<br>" + "\n")
            func_name = f"func_test_div_{section_name}_{test_id}"
            div_name = f"test_div_{section_name}_{test_id}"
            script_str = """
                <script>
                function func_name() {
                    var x = document.getElementById("div_name");
                    if (x.style.display === "none") {
                        x.style.display = "block";
                    } else {
                        x.style.display = "none";
                    }
                }
                </script>
                """
            script_str = script_str.replace("func_name", func_name)
            script_str = script_str.replace("div_name", div_name)
            f.write(script_str + "\n")

            f.write(f'<p onclick="{func_name}()">{test_id}</p>')
            return div_name

        def print_column_header(col_name, f):
            f.write(f"<br>Column(s): {col_name}" + "\n")

        if test_id_list is None:
            test_id_list = self.get_test_list()

        with open(output_file, 'w') as f:
            f.write("<html>" + os.linesep)
            f.write("<head>" + os.linesep)
            f.write("</head>" + os.linesep)
            f.write("<body>" + os.linesep)
            f.write("<h1>Data Consistency Check Results</h1>" + "\n")

            f.write("<a href='#Patterns'>Patterns</a><br/>" + "\n")
            f.write("<a href='#Exceptions'>Exceptions</a><br/>" + "\n")

            f.write("<h2 id='Patterns'>Patterns</h2>" + "\n")
            for test_id in test_id_list:
                sub_patterns_test = self.patterns_df[self.patterns_df['Test ID'] == test_id]

                for columns_set in sub_patterns_test['Column(s)'].values:
                    sub_patterns = self.patterns_df[(self.patterns_df['Test ID'] == test_id) &
                                                    (self.patterns_df['Column(s)'] == columns_set)]
                    if len(sub_patterns) > 0:
                        print_test_header(test_id, "patterns", f)
                        print_column_header(columns_set, f)
                        f.write("<p>Pattern found (without exceptions)</p>")
                        f.write(sub_patterns.iloc[0]['Description of Pattern'])

            f.write("<h2 id='Exceptions'>Exceptions</h2>")
            for test_id in test_id_list:
                sub_results_summary_test = self.exceptions_summary_df[self.exceptions_summary_df['Test ID'] == test_id]
                if len(sub_results_summary_test) == 0:
                    continue
                div_name = print_test_header(test_id, 'exceptions', f)
                f.write(f'<div id="{div_name}">')
                for columns_set in sub_results_summary_test['Column(s)'].values:
                    sub_summary = self.exceptions_summary_df[(self.exceptions_summary_df['Test ID'] == test_id) &
                                                          (self.exceptions_summary_df['Column(s)'] == columns_set)]
                    if len(sub_summary) == 0:
                        continue

                    print_column_header(columns_set, f)
                    f.write(f"<br>Issue index: {sub_summary.index[0]}")
                    f.write("<br>A strong pattern, and exceptions to the pattern, were found.<br>")
                    f.write(sub_summary.iloc[0]['Description of Pattern'])
                    num_exceptions = sub_summary.iloc[0]['Number of Exceptions']
                    f.write((f"<br>Number of exceptions: {num_exceptions} "
                            f"({num_exceptions * 100.0 / self.num_rows:.4f}% of rows)"))
                f.write('</div>')

            f.write("</body>")
            f.write("</html>")

    ##################################################################################################################
    # Internal methods to aid in analysing the data and executing tests
    ##################################################################################################################

    def get_similar_cols(self, full_cols_arr, include_self, lower_divisor, upper_multiplier,
                         check_larger_false=False, check_larger_true=False):
        """
        Used by __check_sum_of_columns, as well as min, max, and mean.
        This finds, for each numeric column, the set of other numeric columns that may reasonably be related to it.
        For min(), we look for columns larger, but not drastically larger. For max() and sum() columns that are
        smaller, but not drastically smaller. For mean(), columns of similar values, and for sum().

        lower_divisor and  upper_multiplier specify the range of median values we can accept.

        if check_larger_false is True (for min()), we check that the current column is not larger than any added cols.
        if check_larger_true is True (for max()), we check that the current column is not smaller than any added cols
        """

        # Get information about which columns have the same values
        cols_same_bool_dict = self.get_cols_same_bool_dict(force=True)

        # Each key in larger_dict is a pair of columns, (A, B), where A > B consistently, row by row.
        larger_dict = self.get_larger_or_equal_pairs_with_bool_dict()

        similar_cols_dict = {}
        similar_cols_idxs_dict = {}
        calc_size = 0
        for col_name in full_cols_arr:
            if include_self:
                similar_cols = [col_name]
            else:
                similar_cols = []
            similar_cols_idxs = []
            for c_idx, c in enumerate(full_cols_arr):
                if c == col_name:
                    continue
                # We provide for a large range, which can be found with, for example, tax values and so on, which may
                # be only 1 to 15% of the total column
                if (self.column_medians[col_name] / lower_divisor) < self.column_medians[c] < (self.column_medians[col_name] * upper_multiplier):
                    # Ensure the columns are correlated with col_name. Any columns summing to col_name should be
                    corr = self.pearson_corr.loc[col_name, c]
                    if corr > 0.2:
                        # Create a tuple representing the pair of features. If the tuple is in larger_dict and True,
                        # col_name is then larger, row by row, than c.
                        pair_tuple = tuple([col_name, c])
                        if not cols_same_bool_dict[tuple(sorted([col_name, c]))]: # cols_same_bool_dict uses sorted column names as the key
                            if (not check_larger_false and not check_larger_true) or \
                                    (check_larger_false and ((pair_tuple not in larger_dict) or (larger_dict[pair_tuple] == False))) or \
                                    (check_larger_true and (pair_tuple in larger_dict) and (larger_dict[pair_tuple] == True)):
                                similar_cols.append(c)
                                similar_cols_idxs.append(c_idx)
            similar_cols_dict[col_name] = similar_cols
            similar_cols_idxs_dict[col_name] = similar_cols_idxs
            calc_size += int(math.pow(2, len(similar_cols)))
        return similar_cols_dict, similar_cols_idxs_dict, calc_size

    def get_limit_subset_sizes(self, full_cols_arr, similar_cols_dict, calc_size):
        """
        Used by __check_sum_of_columns, as well as min, max, and mean. Used to determine how large of subsets we're
        able to support. If there are hundreds of numeric columns, we cannot support subsets of hundreds of columns
        but may be able to process subsets of size 2, 3, or more.

        Parameters:

        full_cols_arr: list
            The set of columns to use for the current test

        similar_cols_dict: dictionary
            Indicates which columns have similar values and so are comparable

        calc_size: int
            The initial calculation of the number of combinations, using all subset sizes. If this is within
            max_combinations, we can support subsets of any size.

        Returns:

            limit_subset_sizes: bool
                Set true if we can execute the test but only with a reduced subset size.

            max_subset_size: int
                Set to -1 if there is no need to limit the sizes of subsets.

            can_process: bool
                Set true if either using the full set of subsets is workable, or a smaller size was found that is valid
        """

        limit_subset_sizes = False
        max_subset_size = -1
        can_process = True
        if calc_size > self.max_combinations:

            # Try limiting the sizes of the subsets
            for max_subset_size in range(5, 1, -1):
                calc_size_limited = 0
                for col_name in full_cols_arr:
                    num_cols = len(similar_cols_dict[col_name])
                    for subset_size in range(2, max_subset_size + 1):
                        calc_size_limited += math.comb(num_cols, subset_size)
                if calc_size_limited < self.max_combinations:
                    limit_subset_sizes = True
                    break

            can_process = limit_subset_sizes

            if self.verbose >= 1:
                if limit_subset_sizes:
                    print((f"  Due to the potential number of combinations, limiting test to subsets of size "
                           f"{max_subset_size}."))
                else:
                    print((f"  Skipping test. Given the number of similar columns for each positive numeric "
                           f"column, there are {calc_size_limited:,} combinations, even limiting testing to subsets "
                           f"2 columns. max_combinations is currently set to {self.max_combinations:,}."))

        return limit_subset_sizes, max_subset_size, can_process

    def get_decision_tree_rules_as_categories(self, rules, categorical_features):
        rules_arr = rules.split('\n')
        for rule in rules_arr:
            for c_name in categorical_features:
                c_name_prefix = c_name + '_'
                if c_name_prefix in rule:
                    part_a, part_b = rule.split(c_name_prefix)
                    val_name = part_b.split()[0]
                    for v in self.orig_df[c_name].unique().astype(str):
                        if val_name.startswith(v):
                            val_name = v
                            break
                    replace_str = c_name_prefix + rule.split(c_name_prefix)[1]
                    if "<" in rule:
                        rules = rules.replace(replace_str, f'{c_name} is not {val_name}')
                    else:
                        rules = rules.replace(replace_str, f'{c_name} is {val_name}')
        return rules

    def _output_current_test(self, test_num, test_id):
        """
        Executed as tests run to allow monitoring progress.
        """
        if self.verbose <= 0:
            return
        if self.verbose == 1:
            print(f"Executing test {test_num:3}: {test_id:<30}")
            return

        # Many tests have sufficiently short descriptions, and so do not specify a separate short description.
        desc = self.test_dict[test_id].short_description
        if desc == "":
            desc = self.test_dict[test_id].description

        # Strip out any periods from short descriptions to be consistent.
        if desc[-1] == '.':
            desc = desc[:-1]

        if is_notebook():
            multiline_test_desc = wrap(desc, 70)
            print(f"Executing test {test_num:3}: {test_id}")
            print(f"  {multiline_test_desc[0]}")
            filler = ''.join([" "]*52)
            for line in multiline_test_desc[1:]:
               print(f'{filler} {line}')
        else:
            print(f"Executing test {test_num:3}: {test_id} \n  {desc}")

    def _add_synthetic_column(self, col_name, col_values):
        """
        Add a column with the specified name and values to self.synth_df.
        This uses concat(), instead of simply adding columns, to avoid inefficiency issues.
        """
        self.synth_df = pd.concat([
            self.synth_df,
            pd.DataFrame({col_name: col_values})],
            axis=1)

    ##################################################################################################################
    # Tune the contamination rate
    ##################################################################################################################
    def test_contamination_level(self, contamination_levels_arr=None):
        """
        This may be used to help determine an appropriate contamination rate to set for the process. Patterns in the
        date will only be recognized by the tool as patterns if there are less than the specified contamination rate
        of exceptions. Exceptions to patterns can only be found if the patterns are first recognized.
        Setting the contamination rate to a small value will identify only strong exceptions, but
        may miss some interesting patterns in the data. Setting a higher value will expose more patterns, but will
        also generate some noise.

        This presents a set of 3 plots: the number of issues found, the number of rows flagged at least once, and the
        number of columns flagged at least once, based on the contamination rate. It also returns these counts as
        three arrays.
        """
        if contamination_levels_arr is None:
            contamination_levels_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        pass

    ##################################################################################################################
    # Data consistency checks for single columns of any type
    ##################################################################################################################

    # Most __generate_xxx methods follow the pattern of generating three columns: one with no pattern, one with a
    # pattern with no exceptions, and one with a pattern with exception, typically in the final row. Many __generate_xxx
    # methods also create additional columns to further test the method.

    def _check_two_cols_larger(self, test_id, require_same_scale):
        """
        Used by __check_larger() and __check_larger_same_range()
        """

        num_pairs, col_pairs = self._get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1: 
                print((f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}."))
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()
        larger_dict = self.get_larger_pairs_dict(allow_equal=False, print_status=True)
        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict(force=True)
        larger_pairs_arr = []

        q1_dict = {}
        q3_dict = {}
        for col_name in self.numeric_cols:
            num_vals = self.numeric_vals[col_name]
            q1_dict[col_name] = num_vals.quantile(0.25)
            q3_dict[col_name] = num_vals.quantile(0.75)

        for cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            test_series = larger_dict[tuple([col_name_1, col_name_2])]
            if test_series is None:
                continue

            # Skip columns that do not have some correlation
            if abs(self.spearman_corr[col_name_1][col_name_2]) < 0.4:
                continue

            # Skip columns that are almost entirely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Skip columns that are almost entirely 0 or Null
            if (self.orig_df[col_name_1] == 0).tolist().count(False) < self.freq_contamination_level:
                continue
            if (self.orig_df[col_name_2] == 0).tolist().count(False) < self.freq_contamination_level:
                continue
            if self.orig_df[col_name_1].notna().sum() < self.freq_contamination_level:
                continue
            if self.orig_df[col_name_2].notna().sum() < self.freq_contamination_level:
                continue

            # Skip columns that are mostly a single value
            if self.orig_df[col_name_1].value_counts(normalize=True).values[0] > 0.8:
                continue
            if self.orig_df[col_name_2].value_counts(normalize=True).values[0] > 0.8:
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            if test_series.tolist().count(False) > self.freq_contamination_level:
                continue

            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            if test_series.tolist().count(False) > self.freq_contamination_level:
                continue

            col_1_q1 = q1_dict[col_name_1]
            col_1_q3 = q3_dict[col_name_1]
            col_2_q1 = q1_dict[col_name_2]
            col_2_q3 = q3_dict[col_name_2]
            if require_same_scale:
                if (self.column_medians[col_name_1] < col_2_q1) or (self.column_medians[col_name_1] > col_2_q3):
                    continue
                if (self.column_medians[col_name_2] < col_1_q1) or (self.column_medians[col_name_2] > col_1_q3):
                    continue
            else:
                if (self.column_medians[col_name_1] > col_2_q1) and (self.column_medians[col_name_1] < col_2_q3):
                    continue
                if (self.column_medians[col_name_2] > col_1_q1) and (self.column_medians[col_name_2] < col_1_q3):
                    continue

                # Test for an order of magnitude difference on the full column. If true, this is redundant with
                # MUCH_LARGER.
                vals_arr_1 = self.numeric_vals_filled[col_name_1]
                vals_arr_2 = self.numeric_vals_filled[col_name_2]
                order_mag_larger_series = np.where(
                    self.orig_df[col_name_2] != 0,
                    (vals_arr_1 / vals_arr_2) > 10.0,
                    False
                )
                order_mag_larger_series = order_mag_larger_series | \
                                          self.orig_df[col_name_1].isna() | \
                                          self.orig_df[col_name_2].isna()
                if order_mag_larger_series.tolist().count(False) < self.freq_contamination_level:
                    continue

            if test_series.tolist().count(False) > 0:
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_1}" is consistently larger than "{col_name_2}"'
                )
            else:
                larger_pairs_arr.append([col_name_1, col_name_2])

        patterns_arr = []  # The set of unique columns in each pattern
        patterns_pairs_arr = []  # The set of pair-wise relationships between columns in each pattern
        for col_name_1, col_name_2 in larger_pairs_arr:
            found_existing_pattern = False
            for p_idx, p in enumerate(patterns_arr):
                if (col_name_1 in p) or (col_name_2 in p):
                    patterns_arr[p_idx].append(col_name_1)
                    patterns_arr[p_idx].append(col_name_2)
                    patterns_arr[p_idx] = list(set(patterns_arr[p_idx]))
                    patterns_pairs_arr[p_idx].append([col_name_1, col_name_2])
                    found_existing_pattern = True
                    break
            if not found_existing_pattern:
                patterns_arr.append([col_name_1, col_name_2])
                patterns_pairs_arr.append([[col_name_1, col_name_2]])

        for pattern_idx, cols in enumerate(patterns_arr):

            # Order the columns in the pattern based on their median values
            col_medians = [self.column_medians[c] for c in cols]
            cols = np.array(cols)[np.argsort(col_medians)]

            if len(cols) == 2:
                desc = (f'"{patterns_pairs_arr[pattern_idx][0][0]}" is consistently larger than '
                        f'"{patterns_pairs_arr[pattern_idx][0][1]}"')
            else:
                desc = "There is a consistent relationship in size of values between the the columns."
                for p in patterns_pairs_arr[pattern_idx]:
                    desc += f'\n"{p[0]}" is consistently larger than "{p[1]}"'

            self._process_analysis_binary(
                test_id,
                sorted(cols),
                [True]*self.num_rows,
                desc,
                display_info={'patterns_pairs_arr': patterns_pairs_arr[pattern_idx]}
            )

    def _check_sum_exact(self, test_id):
        # todo: we can maybe combine this with checking for similar above, and adjust the threhold so there is a
        #   reasonable number of exceptions.
        pass

    def _check_grouped_strings_column(self, test_id, sort_col, col_name, col_values):
        """
        sort_col: The column used to sort the data, if any.
        col_name: The name of the column where we check if the values are grouped
        col_values: The values in col_name, either in the original order of the data, or sorted by sort_col if there
            is a sort_col

        Handling null values: this does not currently support many null values.
        """

        # Skip if there are any rare values
        min_count = col_values.value_counts().min()
        if min_count < self.freq_contamination_level:
            return

        if self.orig_df[col_name].isna().sum() > (self.num_rows * 0.9):
            return

        if self.orig_df[col_name].nunique() < 3:
            return

        # First test if a pattern holds when removing all null values
        col_df = pd.DataFrame({col_name: col_values})
        col_df.dropna()
        col_df['Next'] = col_df[col_name].shift(1)
        col_df['Same'] = col_df[col_name] == col_df['Next']
        num_same = col_df['Same'].tolist().count(True)
        # There will always be rows not like the next: where the list moves from one value to the next. So ideally,
        # the number of rows that are the same as the next is the total number of rows - (number values -1). As well,
        # the last row is always unlike the next, as the next is undefined.
        ideal_same = self.num_valid_rows[col_name] - self.orig_df[col_name].nunique()
        if (ideal_same - num_same) > self.freq_contamination_level:
            return

        # Test with the null values. This is necessary to maintain the actual row numbers
        col_df = pd.DataFrame({col_name: col_values})
        col_df['Next'] = col_df[col_name].shift(1)
        col_df['Same'] = col_df[col_name] == col_df['Next']
        num_same = col_df['Same'].tolist().count(True)
        ideal_same = self.num_valid_rows[col_name] - self.orig_df[col_name].nunique()
        if (ideal_same - num_same) > self.freq_contamination_level:
            return

        test_series = np.array([1]*self.num_rows)
        groups_str = ""
        group_lengths = []

        # Loop through each unique value and find the indexes where it occurs.
        # We then identify the runs of each unique value and flag any short runs.
        for v in col_df[col_name].dropna().unique():
            idxs = np.where(col_df[col_name] == v)[0]
            idx_diffs = pd.Series(idxs).shift(-1).values - idxs
            exceptions = list(np.where(idx_diffs != 1)[0])
            group_starts = [min(idxs)]
            group_ends = [max(idxs)]
            for e in exceptions:
                group_starts.append(idxs[e] + idx_diffs[e])
                group_ends.insert(0, idxs[e])

            group_starts = sorted([x for x in (set(group_starts)) if x == x])
            group_ends = sorted([x for x in (set(group_ends)) if x == x])
            groups_str += f'\nValue: "{v}": rows {group_starts[0]:.0f} to {group_ends[0]:.0f}'
            group_lengths.append(group_ends[0] - group_starts[0] + 1)
            for i in range(1, len(group_starts)):
                groups_str += f', {group_starts[i]:.0f} to {group_ends[i]:.0f}'
                group_lengths.append(group_ends[i] - group_starts[i] + 1)

            if num_same != ideal_same:
                for i in range(len(group_starts)):
                    group_len = group_ends[i] - group_starts[i] + 1
                    if group_len < (len(idxs) * 0.5):
                        for j in range(int(group_starts[i]), int(group_ends[i]+1)):
                            test_series[j] = False

        # All all runs are of length 1, skip this column
        if max(group_lengths) == 1:
            return

        pattern_cols = [col_name]
        if sort_col:
            pattern_cols = [sort_col, col_name]

            # Get the index in the original (unsorted) dataframe of the flagged rows.
            if test_series.tolist().count(False) < self.freq_contamination_level:
                idxs = list(np.where(test_series == False)[0])
                test_series = [True] * self.num_rows
                for idx in idxs:
                    test_series[self.orig_df.sort_values(sort_col).index[idx]] = False

        sort_msg = ""
        if sort_col:
            sort_msg = f" when sorted by {sort_col}"

        self._process_analysis_binary(
            test_id,
            pattern_cols,
            test_series,
            (f'The values in "{col_name}" are consistently grouped together{sort_msg}. The overall order is: '
             f'{groups_str}.')
        )

