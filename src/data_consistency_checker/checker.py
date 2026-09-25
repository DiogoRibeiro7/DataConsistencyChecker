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
from .data_init_mixin import DataInitMixin
from .execution_mixin import ExecutionMixin
from .export_mixin import ExportMixin

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


class DataConsistencyChecker(BaseTestsMixin, NumericTestsMixin, DateTestsMixin, StringTestsMixin, BinaryTestsMixin, MultiColumnTestsMixin, DisplayMixin, PlotsMixin, SynthDataMixin, ResultsMixin, AnalysisCacheMixin, DataInitMixin, ExecutionMixin, ExportMixin):
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

    ##################################################################################################################
    # Methods to find relationships between the data and the numbers of issues found.
    ##################################################################################################################

    ##################################################################################################################
    ##################################################################################################################
    # Private methods to display tables of example rows from the original data
    ##################################################################################################################
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

    ##################################################################################################################
    # Internal methods to aid in analysing the data and executing tests
    ##################################################################################################################

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
