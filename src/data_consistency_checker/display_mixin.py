"""Display mixin for DataConsistencyChecker.

This module contains helper methods related to displaying information
and statistics about the data. Splitting these out keeps the main
``check_data_consistency.py`` file more manageable.
"""

from __future__ import annotations

import numbers
import os
from collections.abc import Collection
from itertools import product
from textwrap import wrap

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from .checker_state import CheckerState
from .checker_utils import (
    convert_to_numeric,
    get_num_decimal_digits,
    is_missing,
    is_notebook,
    print_line,
    print_text,
    replace_special_with_space,
    safe_div,
    styling_flagged_rows,
    styling_orig_row,
)
from .test_metadata import TestMetadata, metadata_from_definition


class DisplayMixin(CheckerState):
    """Mixin providing display helper methods."""

    # ------------------------------------------------------------------
    # Public helper methods about the tool itself
    # ------------------------------------------------------------------
    def get_test_list(self) -> list[str]:
        """Return the list of implemented test IDs."""
        return [test_id for test_id, definition in self.test_dict.items() if definition.implemented]

    def get_test_descriptions(self) -> dict:
        """Return a dictionary mapping test IDs to short descriptions."""
        return {
            test_id: definition.description
            for test_id, definition in self.test_dict.items()
            if definition.implemented
        }

    def get_test_catalog(self) -> list[TestMetadata]:
        """Return typed metadata for all implemented consistency checks."""
        return [
            metadata_from_definition(test_id, definition)
            for test_id, definition in self.test_dict.items()
            if definition.implemented
        ]

    def print_test_descriptions(self, long_desc: bool = False, f: object | None = None) -> None:
        """Print test descriptions.

        Args:
            long_desc: If ``True``, include the longer description from the
                test function docstring.
            f: Optional file handle. If provided, output is written to this file
                as HTML.
        """
        for test_id in self.test_dict:
            text = self.test_dict[test_id].description
            if long_desc:
                doc_str = self.test_dict[test_id].test_func.__doc__
                if doc_str:
                    text += doc_str
                    text = " ".join(text.split())
            multiline_test_desc = wrap(text, 90)
            if f:
                f.write(f"{test_id+':':<30} {multiline_test_desc[0]}" + "<br>" + os.linesep)
            else:
                print(f"{test_id+':':<30} {multiline_test_desc[0]}")
            filler = "".join([" "] * 32)
            for line in multiline_test_desc[1:]:
                if f:
                    f.write(f"{filler} {line}" + "<br>" + os.linesep)
                else:
                    print(f"{filler} {line}")

    def get_patterns_shortlist(self) -> list[str]:
        """Return IDs of tests included in the short list."""
        return [test_id for test_id, definition in self.test_dict.items() if definition.shortlist]

    def get_tests_for_codes(self) -> list[str]:
        """Return IDs of tests related to code/ID style values."""
        return [test_id for test_id, definition in self.test_dict.items() if definition.code]

    def demo_test(self, test_id: str, include_nulls: bool = False) -> None:
        """Demonstrate a single test on synthetic data.

        Args:
            test_id: ID of the test to run.
            include_nulls: Whether to generate several versions of the demo data
                containing different amounts of ``None`` values.
        """
        if test_id not in self.test_dict:
            print(f"Error {test_id} is not a valid test")

        none_cases = ["none"]
        none_strs = [""]
        if include_nulls:
            none_cases = ["none", "one-row", "in-sync", "random", "80-percent"]
            none_strs = [
                "",
                "-- One row of Null values",
                "-- 50% Null values (same rows)",
                "-- 50% Null values (random locations)",
                "-- 80-percent Null Values",
            ]

        for none_idx, none_type in enumerate(none_cases):
            synth_df = self.generate_synth_data(all_cols=False, execute_list=[test_id], add_nones=none_type)
            print_text(f"## {test_id}")
            print_text(f"### Synthetic Data {none_strs[none_idx]}:")
            if is_notebook():
                display(synth_df)
            else:
                print(synth_df)

            self.verbose = 2
            self.init_data(synth_df)
            self.check_data_quality(execute_list=[test_id])
            self.display_detailed_results(show_short_list_only=False)

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
            return

        sorted_df = self.test_results_df.sort_values('FINAL SCORE', ascending=False)
        sorted_df.index = [x[0] if type(x) is tuple else x for x in sorted_df.index]
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
            max_shown = self._default_max_shown(save_to_disk, include_examples, plot_results)

        no_filters = (test_id_list is None) and (pattern_id_list is None) and (col_name_list is None) and \
            (issue_id_list is None) and (row_id_list is None)
        if no_filters and self._exceeds_display_limit(show_patterns, show_exceptions, max_shown):
            return

        # Initialize the folder and file handle for HTML exports if specified
        f = None  # file handle used for HTML export
        if save_to_disk:
            f = self._open_detailed_results_file(output_folder)

        try:
            self._describe_display_filters(test_id_list, col_name_list, show_short_list_only, f)

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

            # Each test's results are introduced by a header, unless results for only one test can be shown
            show_test_headers = not (len(test_id_list) == 1 or
                                     (self.execute_list is not None and len(self.execute_list) == 1))
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

                        count_shown += 1
                        self._print_finding_header(test_id, columns_set, show_test_headers and not printed_test_header, f)
                        printed_test_header = True
                        self._display_pattern_details(
                            test_id, columns_set, sub_patterns.iloc[0], include_examples, plot_results, f)

                # Display patterns with exceptions
                if show_exceptions:
                    for columns_set in sub_results_summary_test['Column(s)'].values:
                        if count_shown >= max_shown:
                            print_line(f)
                            print_text(max_shown_msg, f)
                            return

                        if row_id_list and not row_id_list_df[self.get_results_col_name(test_id, columns_set)].any():
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
                        self._print_finding_header(test_id, columns_set, show_test_headers and not printed_test_header, f)
                        printed_test_header = True
                        self._display_exception_details(
                            test_id, columns_set, issue_id, sub_summary.iloc[0], include_examples, plot_results, f)
        finally:
            # Complete and close the HTML report however the display ends, including early returns
            if f:
                self._close_detailed_results_file(f)

    @staticmethod
    def _default_max_shown(save_to_disk, include_examples, plot_results):
        """
        Called by display_detailed_results() to determine how many patterns and exceptions (in total) may be shown
        when max_shown is not specified. Fewer are shown when each includes examples or plots.
        """
        max_shown: float = 50_000 if save_to_disk else 200
        if include_examples:
            max_shown /= 2
        if plot_results:
            max_shown /= 2
        return max_shown

    def _exceeds_display_limit(self, show_patterns, show_exceptions, max_shown):
        """
        Called by display_detailed_results() where no filters are specified. If more patterns and/or exceptions were
        found than may be displayed at once, this explains how to narrow the results and returns True.
        """
        msg = ("This is beyond the limit to display all at once. Try specifying tests and/or columns to be "
               "displayed here, specific issues, or row numbers, or setting include_examples and/or "
               "plot_results to False.")

        if show_patterns and show_exceptions and \
                ((len(self.exceptions_summary_df) + len(self.patterns_df)) > max_shown):
            print()
            print(f"{len(self.exceptions_summary_df) + len(self.patterns_df)} patterns and exceptions were "
                   f"identified. {msg}")
            return True
        if show_patterns and (len(self.patterns_df) > max_shown):
            print()
            print(f"{len(self.exceptions_summary_df) + len(self.patterns_df)} patterns were identified. {msg}")
            return True
        if show_exceptions and (len(self.exceptions_summary_df) > max_shown):
            print()
            print(f"{len(self.exceptions_summary_df)} issues were identified. {msg}")
            return True
        return False

    def _open_detailed_results_file(self, output_folder):
        """
        Called by display_detailed_results() when saving to disk. Creates the output folder if necessary, and opens
        the HTML report, writing its header. Returns the open file handle.
        """
        if output_folder is None:
            self.output_folder = os.path.join(os.getcwd(), "Output")
        else:
            self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        f = open(os.path.join(self.output_folder, "Data_consistency.html"), 'w')  # noqa: SIM115 - closed by the caller
        f.write("<html>" + os.linesep)
        f.write("<head>" + os.linesep)
        f.write("</head>" + os.linesep)
        f.write("<body>" + os.linesep)
        f.write("<h1>Data Consistency Check Results</h1>" + os.linesep)
        f.write("<body>" + os.linesep)
        return f

    @staticmethod
    def _close_detailed_results_file(f):
        """Called by display_detailed_results() to complete and close the HTML report."""
        f.write("</body>" + os.linesep)
        f.write("</html>" + os.linesep)
        f.close()

    def _describe_display_filters(self, test_id_list, col_name_list, show_short_list_only, f):
        """
        Called by display_detailed_results() to describe the tests and columns requested, noting any that are
        invalid, or tests whose patterns are hidden by show_short_list_only.
        """
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

    def _print_finding_header(self, test_id, col_name, include_test_header, f):
        """
        Called by display_detailed_results() before each pattern or exception displayed. Prints a header for the test
        (if include_test_header is True, which is the case for the first finding shown for each test), then a header
        for the column(s) involved.
        """
        if include_test_header:
            if f:
                f.write(''.join(['.'] * 100) + "<br>" + os.linesep)
                f.write("<H2>" + test_id + "</H2>" + os.linesep)
            elif is_notebook():
                print(''.join(['.'] * 100))
                display(Markdown(f"### {test_id}"))
            else:
                stars = "******************************************************************************"
                print("\n\n\n")
                print(stars)
                print(test_id)
                print(stars)

        hyphens = '----------------------------------------------------------------------------'
        print_line(f)
        if not is_notebook():
            if f:
                f.write(hyphens + "<br>" + os.linesep)
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

    def _display_pattern_details(self, test_id, columns_set, pattern, include_examples, plot_results, f):
        """
        Called by display_detailed_results() to describe one pattern found without exceptions, optionally with
        example rows and plots.

        pattern: pd.Series
            The row of patterns_df describing the pattern.
        """
        print_text("Pattern found (without exceptions)", f)
        if test_id in ['PREV_VALUES_DT', 'DECISION_TREE_REGRESSOR', 'DECISION_TREE_CLASSIFIER',
                       'PREDICT_NULL_DT']:
            print_text("**Description**:", f)
            print_text(pattern['Description of Pattern'], f)
        else:
            print_text(f"**Description**: {pattern['Description of Pattern']}", f)
        cols = [x.lstrip('"').rstrip('"') for x in columns_set.split(" AND ")]
        if include_examples:
            self._display_examples_not_flagged(
                test_id,
                cols,
                columns_set,
                is_patterns=True,
                display_info=pattern['Display Information'],
                f=f)
        if plot_results:
            self._draw_results_plots(
                test_id,
                cols,
                columns_set,
                show_exceptions=False,
                display_info=pattern['Display Information'],
                f=f)

    def _display_exception_details(self, test_id, columns_set, issue_id, summary, include_examples, plot_results, f):
        """
        Called by display_detailed_results() to describe one pattern found with exceptions, optionally with example
        rows that were and were not flagged, and plots.

        summary: pd.Series
            The row of exceptions_summary_df describing the pattern and its exceptions.
        """
        print_text(f"**Issue ID**: {issue_id}", f)
        if test_id in ['RARE_VALUES', 'VERY_SMALL', 'VERY_LARGE', 'VERY_SMALL_ABS', 'LARGE_GIVEN_DATE',
                       'SMALL_GIVEN_DATE', 'LARGE_GIVEN_VALUE', 'SMALL_GIVEN_VALUE', 'LARGE_GIVEN_PREFIX',
                       'SMALL_GIVEN_PREFIX', 'LARGE_GIVEN_PAIR', 'SMALL_GIVEN_PAIR']:
            print_text("Unusual values were found.\n", f)
        else:
            print_text("A strong pattern, and exceptions to the pattern, were found.\n", f)
        if test_id in ['GROUPED_STRINGS', 'GROUPED_STRINGS_BY_NUMERIC']:
            # These display special output, so the formatting must be preserved.
            print_text("**Description**:", f)
            print_text(summary['Description of Pattern'], f)
        elif test_id in ['PREV_VALUES_DT', 'DECISION_TREE_REGRESSOR', 'DECISION_TREE_CLASSIFIER',
                         'PREDICT_NULL_DT']:
            # These display a decision tree, so the formatting must be preserved.
            print_text("**Description**:", f)
            print_text(summary['Description of Pattern'], f)
        else:
            multiline_desc = wrap(summary['Description of Pattern'], 100)
            print_text(f"**Description**: {'<br>'.join(multiline_desc)}", f)
        num_exceptions = summary['Number of Exceptions']
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
                display_info=summary['Display Information'],
                f=f)

            flagged_df = self._get_rows_flagged(test_id, columns_set)
            if flagged_df is None:
                return
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
                display_info=summary['Display Information'],
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
                    display_info=summary['Display Information'],
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
                display_info=summary['Display Information'],
                f=f)

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
        df = df.sort_values(df.columns[-1]) if test_id in ['SMALL_GIVEN_DATE', 'LARGE_GIVEN_DATE'] else df.sort_index()

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

    def _get_sample_not_flagged(self, test_id, col_name, n_examples=10, show_consecutive=False, sort_col=None,
                                 is_patterns=False, display_info=None, f=None):  # noqa: ARG002
        """
        Called by _display_examples_not_flagged()

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
            cols: Collection[str] = [x.lstrip('"').rstrip('"') for x in col_name.split(" AND ")]
        else:
            results_col_name = self.get_results_col_name(test_id, col_name)
            cols = self.col_to_original_cols_dict[results_col_name]

        # If there are no values flagged for this test in this feature, there will not be a column in
        # test_results_df. In this case, return any values.
        if not is_patterns and results_col_name not in self.test_results_df.columns:
            assert False, "Should not happen"  # noqa: B011
            return self.orig_df[col_name].sample(n=n_examples, random_state=0)

        df = self._get_balanced_sample(test_id, cols, n_examples, display_info)

        # If we do not yet have a df (the test was not specified above, or no rows matched the conditions), we
        # create a df simply trying to reduce the number of Null values and showing unique values in the last column.
        if not show_consecutive and ((df is None) or df.empty):
            cols = list(cols)  # Ensure cols is not in tuple format
            df = self._get_generic_sample(test_id, cols, n_examples)

        if show_consecutive:
            assert df is None
            df = self._get_consecutive_sample(cols, n_examples, sort_col)
        else:
            # If we do not return a consecutive set of rows, df is likely of size n_examples. We ensure it is of size
            # n_examples, then sort it randomly.
            df = self._pad_sample(df, cols, n_examples)
            df = df.sample(n=min(len(df), n_examples), random_state=0)

        # Remove rows that were flagged. If is_patterns is True, no rows were flagged, and we skip this check.
        if not is_patterns:
            sub_df = self.test_results_df.loc[df.index]
            mask = sub_df[results_col_name] == 0
            df = df[mask]

        return df

    def _get_balanced_sample(self, test_id, cols, n_examples, display_info):
        """
        Called by _get_sample_not_flagged(). For tests where it is informative, returns rows balanced between the
        different cases relevant to the test (for example, rows where a value is zero and where it is non-zero).
        Returns None for other tests.
        """
        if test_id in ['BINARY_SAME', 'BINARY_OPPOSITE', 'BINARY_IMPLIES', 'BINARY_AND', 'BINARY_OR',
                       'BINARY_XOR', 'BINARY_NUM_SAME', 'BINARY_TWO_OTHERS_MATCH', 'BINARY_MATCHES_SUM']:
            return self._get_binary_balanced_sample(cols, n_examples)

        if test_id in ['MATCHED_ZERO', 'MATCHED_ZERO_MISSING', 'MATCHED_SET_ZERO_NON_ZERO']:
            # Show where col1 is zero and non-zero
            col_name_1 = cols[0]
            col_name_2 = cols[1]
            df_v_zero = self.orig_df[cols][(self.orig_df[col_name_1] == 0) & (self.orig_df[col_name_1].notna())].head(n_examples // 2)
            if len(df_v_zero) < (n_examples // 2):
                df_v_zero = pd.concat([df_v_zero, self.orig_df[cols][(self.orig_df[col_name_1] == 0) & (self.orig_df[col_name_1].isna())].head((n_examples // 2) - len(df_v_zero))])
            df_v_non_zero = self.orig_df[cols][(self.orig_df[col_name_1] != 0) &
                                               (self.orig_df[col_name_1].notna())].head(n_examples - len(df_v_zero))
            return pd.concat([df_v_zero, df_v_non_zero])

        if test_id in ['SAME_OR_CONSTANT']:
            # Show where col1 == col2 and where doesn't (in both cases where col1 is not null)
            col_name_1, col_name_2 = cols
            df_same = self.orig_df[cols][
                (self.orig_df[col_name_1] == self.orig_df[col_name_2]) & self.orig_df[col_name_1].notna()
                ].head(n_examples // 2)
            df_not_same = self.orig_df[cols][
                (self.orig_df[col_name_1] != self.orig_df[col_name_2]) & self.orig_df[col_name_1].notna()
                ].head(n_examples // 2)
            return pd.concat([df_same, df_not_same])

        if test_id in ['POSITIVE']:
            # Show where col == 0 and where is > 0
            col_name_1 = cols[0]
            df_v_zero = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] == 0)].head(n_examples // 2)
            df_v_pos = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] > 0)].head(n_examples - len(df_v_zero))
            return pd.concat([df_v_zero, df_v_pos])

        if test_id in ['NEGATIVE']:
            # Show where col == 0 and where is < 0
            col_name_1 = cols[0]
            df_v_zero = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] == 0)].head(n_examples // 2)
            df_v_neg = self.orig_df[cols][(self.numeric_vals_filled[col_name_1] < 0)].head(n_examples - len(df_v_zero))
            return pd.concat([df_v_zero, df_v_neg])

        if test_id in ['MATCHED_SET_POS_NEG']:
            # Show where col1 is positive and negative
            col_name_1 = cols[0]
            df_v_pos = self.orig_df[cols][(self.orig_df[col_name_1] > 0)].head(n_examples // 2)
            df_v_neg = self.orig_df[cols][(self.orig_df[col_name_1] < 0)].head(n_examples - len(df_v_pos))
            return pd.concat([df_v_pos, df_v_neg])

        if test_id in ['EVEN_MULTIPLE', 'MATCHED_MISSING', 'OPPOSITE_MISSING']:
            # Show where col 1 is null and non-null
            col_name_1, col_name_2 = cols[0], cols[1]
            df_v_null = self.orig_df[cols][(self.orig_df[col_name_1].apply(is_missing))].head(n_examples // 2)
            num_non_null = n_examples - len(df_v_null)
            df_v_non_null = self.orig_df[cols][(~self.orig_df[col_name_1].apply(is_missing))].head(num_non_null)
            return pd.concat([df_v_null, df_v_non_null])

        if test_id in ['PREDICT_NULL_DT']:
            # Show where target column is null and non-null
            target_col = cols[-1]
            df_v_null = self.orig_df[cols][(self.orig_df[target_col].apply(is_missing))].head(n_examples // 2)
            num_non_null = n_examples - len(df_v_null)
            df_v_non_null = self.orig_df[cols][(~self.orig_df[target_col].apply(is_missing))].head(num_non_null)
            return pd.concat([df_v_null, df_v_non_null])

        if test_id in ['C_IS_A_OR_B']:
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
            return pd.concat([df_both, df_v1_null, df_v1_not_null, df_v2_null, df_v2_not_null])[cols]

        if test_id in ['TWO_PAIRS']:
            # Show where the first pair match and where they do not
            df_match = self.orig_df[np.array(display_info['match_1_2_arr']) == True].head(n_examples // 2)  # noqa: E712
            df_not_match = self.orig_df[np.array(display_info['match_1_2_arr']) == False].head(n_examples - len(df_match))  # noqa: E712
            return pd.concat([df_match, df_not_match])[cols]

        return None

    def _get_binary_balanced_sample(self, cols, n_examples):
        """
        Called by _get_balanced_sample() for tests on binary columns. Returns rows covering each combination of the
        binary values or, where only the last column is binary, rows balanced between its two values. Returns None
        if the columns are not of these forms.
        """
        if len(cols) == 2 and cols[0] in self.binary_cols and cols[1] in self.binary_cols:
            v0_0, v0_1 = self.column_unique_vals[cols[0]]
            v1_0, v1_1 = self.column_unique_vals[cols[1]]
            return self._rows_per_value_combination(cols, [(v0_0, v0_1), (v1_0, v1_1)], n_examples // 4)

        if len(cols) == 3 and cols[0] in self.binary_cols and \
                cols[1] in self.binary_cols and cols[2] in self.binary_cols:
            v0_0, v0_1 = self.orig_df[cols[0]].dropna().unique()
            v1_0, v1_1 = self.orig_df[cols[1]].dropna().unique()
            v2_0, v2_1 = self.orig_df[cols[2]].dropna().unique()
            return self._rows_per_value_combination(
                cols, [(v0_0, v0_1), (v1_0, v1_1), (v2_0, v2_1)], n_examples // 4)

        if len(cols) == 3 and cols[2] in self.binary_cols:
            v0, v1 = self.orig_df[cols[2]].dropna().unique()
            va = self.orig_df[cols[0]].dropna().value_counts().index[0]  # The most common value
            df_v0 = self.orig_df[self.orig_df[cols[2]] == v0]
            df_v1 = self.orig_df[self.orig_df[cols[2]] == v1]
            df_v0a = df_v0[df_v0[cols[0]] == va].head(n_examples // 4)
            df_v0b = df_v0[df_v0[cols[0]] != va].head((n_examples // 2) - len(df_v0a))
            df_v1a = df_v1[df_v1[cols[0]] == va].head(n_examples // 4)
            df_v1b = df_v1[df_v1[cols[0]] != va].head((n_examples // 2) - len(df_v1a))
            return pd.concat([df_v0a, df_v0b, df_v1a, df_v1b])[cols]

        return None

    def _rows_per_value_combination(self, cols, values_per_col, n_per_combination):
        """
        Return up to n_per_combination rows of the original data for each combination of the specified values of
        the specified columns, in the order of itertools.product(), restricted to the specified columns.
        """
        dfs_arr = []
        for combination in product(*values_per_col):
            mask = self.orig_df[cols[0]] == combination[0]
            for col, value in zip(cols[1:], combination[1:]):
                mask = mask & (self.orig_df[col] == value)
            dfs_arr.append(self.orig_df[mask].head(n_per_combination))
        return pd.concat(dfs_arr)[cols]

    def _get_generic_sample(self, test_id, cols, n_examples):
        """
        Called by _get_sample_not_flagged() where no balanced sample is available for the test. Prefers rows with
        non-null values, and covering the most common values of the last column.
        """
        # Test that test_results_df does not have duplicate values in the index
        assert (self.test_results_df is None) or \
               (len(self.test_results_df.index) == len(set(self.test_results_df.index)))
        df = self.orig_df[cols]

        # Collect rows with non-null values, other than for columns that are almost all null
        for col in cols:
            sub_df = self.orig_df.loc[df.index]
            mask = sub_df[col].notna()
            if mask.tolist().count(True) >= 5:
                df = df[mask]

        # If there are too few rows, collect additional rows.
        df = self._pad_sample(df, cols, n_examples)

        # Try to get a good set of unique values
        if test_id in ['UNIQUE_VALUES']:
            return df.sample(n=n_examples)

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
        return pd.concat(dfs_arr)

    def _get_consecutive_sample(self, cols, n_examples, sort_col):
        """
        Called by _get_sample_not_flagged() for tests where row order is relevant. Returns n_examples consecutive
        rows, starting at a random row, after sorting by sort_col if specified.
        """
        df = self.orig_df[cols]
        start_point = np.random.randint(0, len(df) - n_examples)
        if sort_col:
            return df.sort_values(sort_col).iloc[start_point: start_point + n_examples]
        return df.iloc[start_point: start_point + n_examples]

    def _pad_sample(self, df, cols, n_examples):
        """
        If df has fewer than n_examples rows, add randomly-selected rows of the original data, excluding any already
        present.
        """
        if len(df) < n_examples:
            df = pd.concat([df, self.orig_df[cols].sample(n_examples - len(df))])
            df = df[~df.index.duplicated(keep='first')]
        return df

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
                            test_row[column_idx+1] = '\u2714'  # Checkmark symbol
                            colour_cells[column_idx] = True
                if test_row.count('\u2714'):
                    orig_row = pd.concat([orig_row, pd.DataFrame([test_row], columns=orig_row.columns)])
            orig_row = orig_row.reset_index()
            orig_row = orig_row.drop(columns=['index'])

            # Display the dataframe representing this row from the original data
            print()
            if is_notebook():
                display(Markdown(f"**Row: {row_idx} " + '\u2014' + f" Final Score: {sorted_df.loc[row_idx]['FINAL SCORE']}**"))
                display(orig_row.style.apply(styling_orig_row, row_idx=0, flagged_arr=colour_cells, axis=None))
            else:
                print(f"Row: {row_idx} Final Score: {sorted_df.loc[row_idx]['FINAL SCORE']}")
                print(orig_row.to_string(index=False))
            print()


    # ------------------------------------------------------------------
    # Dataset statistics helpers
    # ------------------------------------------------------------------
    def display_columns_types_list(self) -> None:
        """Display lists of columns for each inferred type."""
        print()
        print_text("**String Columns**:")
        if len(self.string_cols) > 0:
            print_text(str(self.string_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

        print()
        print_text("**Numeric Columns**:")
        if len(self.numeric_cols) > 0:
            print_text(str(self.numeric_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

        print()
        print_text("**Binary Columns**:")
        if len(self.binary_cols) > 0:
            print_text(str(self.binary_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

        print()
        print_text("**Date/Time Columns**:")
        if len(self.date_cols) > 0:
            print_text(str(self.date_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

    def display_columns_types_table(self) -> None:
        """Display the inferred types along with sample data."""
        var_types = []
        for col_name in self.orig_df.columns:
            if col_name in self.string_cols:
                var_types.append("String")
            elif col_name in self.binary_cols:
                var_types.append("Binary")
            elif col_name in self.date_cols:
                var_types.append("Date")
            elif col_name in self.numeric_cols:
                var_types.append("Numeric")
            else:
                var_types.append("Unused")
        df1 = pd.DataFrame([var_types], columns=self.orig_df.columns)
        df2 = self.orig_df.head(5).copy()
        display_df = pd.concat([df1, df2])

        print()
        print_text("Assigned column types and example rows:")
        if is_notebook():
            display(display_df)
        else:
            print(display_df)
