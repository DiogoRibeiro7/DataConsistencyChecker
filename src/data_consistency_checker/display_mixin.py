"""Display mixin for DataConsistencyChecker.

This module contains helper methods related to displaying information
and statistics about the data. Splitting these out keeps the main
``check_data_consistency.py`` file more manageable.
"""

from __future__ import annotations

import numbers

import os
from textwrap import wrap
from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import display, Markdown

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


class DisplayMixin:
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

    def print_test_descriptions(self, long_desc: bool = False, f: Optional[object] = None) -> None:
        """Print test descriptions.

        Args:
            long_desc: If ``True``, include the longer description from the
                test function docstring.
            f: Optional file handle. If provided, output is written to this file
                as HTML.
        """
        for test_id in self.test_dict.keys():
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
        if test_id not in self.test_dict.keys():
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
                    orig_row = pd.concat([orig_row, pd.DataFrame([test_row], columns=orig_row.columns)])
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
