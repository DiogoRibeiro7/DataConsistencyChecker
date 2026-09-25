from __future__ import annotations
from typing import Any, Callable

from dataexcept import OutlierDetectionError, exception_to_dict
from dataexcept import wrap as wrap_dataexcept

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
