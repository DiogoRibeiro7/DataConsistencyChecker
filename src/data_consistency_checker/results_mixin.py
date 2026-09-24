"""Result access, summaries, and scoring for DataConsistencyChecker."""

from __future__ import annotations

import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .checker_utils import print_text, truncate_description
from .report import DataConsistencyReport


class ResultsMixin:
    """Mixin providing result queries, reports, summaries, state management, and row-level scoring."""

    def get_execution_failures(self) -> list[dict[str, Any]]:
        """Return a defensive copy of failures from the latest quality run."""
        return copy.deepcopy(self.execution_failures)

    def get_report(self) -> DataConsistencyReport:
        """Return a structured snapshot of the current analysis results.

        The report contains only JSON-safe primitives and does not expose live
        pandas objects or mutable checker state.
        """
        patterns_df = self.get_patterns_list(show_short_list_only=False)
        exceptions_df = self.get_exceptions_list()
        scores_df = self.get_outlier_score_summary()

        patterns = () if patterns_df is None else tuple(
            patterns_df.to_dict(orient="records")
        )
        exceptions = () if exceptions_df is None else tuple(
            exceptions_df.to_dict(orient="records")
        )

        score_records = scores_df.reset_index().rename(
            columns={"index": "row_id"}
        ).to_dict(orient="records")

        n_rows = 0 if self.orig_df is None else len(self.orig_df)
        n_columns = 0 if self.orig_df is None else len(self.orig_df.columns)

        return DataConsistencyReport.from_values(
            n_rows=n_rows,
            n_columns=n_columns,
            executed_tests=tuple(self.execution_test_list),
            patterns=patterns,
            exceptions=exceptions,
            row_scores=tuple(score_records),
            execution_failures=tuple(self.get_execution_failures()),
        )


    def get_test_ids_with_results(self, include_patterns=True, include_exceptions=True):
        """
        Gets a list of test ids, which may be used, for example, to loop through tests calling other APIs such as
        display_detailed_results(). These are the ids of tests that flagged at least one pattern and/or exeception.

        include_patterns: bool
            If True, the returned list will include all test ids for tests that flagged at least one pattern without
            exceptions

        include_exceptions: bool
            If True, the returned list will include all test ids for tests that flagged at least one pattern with
            exceptions

        Returns: array of test ids
            Returns an array of test ids, where each test found at least one pattern, with or without exceptions, as
            specified
        """

        ret_list = []
        if include_patterns:
            ret_list = self.patterns_df['Test ID'].unique().tolist()

        if include_exceptions:
            ret_list += self.exceptions_summary_df['Test ID'].unique().tolist()

        # Get the set of unique tests, and sort them based on their standard test order
        ret_list = list(set(ret_list))
        idx_full_list = [self.get_test_list().index(x) for x in ret_list]
        ret_list = np.array(ret_list)[np.argsort(idx_full_list)].tolist()
        return ret_list

    def get_single_feature_tests_matrix(self):
        """
        Returns a matrix with a column for every column in the original data and a row for each test executed. The cell
        values contain the percent of rows matching the pattern associated with the test. In many cases, this will be
        NaN (rendered as '-'), as not all tests execute on all columns. For example, tests for large numeric values
        execute only on numeric columns. As well, for efficiency, many tests skip examining columns that have many null
        values, many zero values, few or many unique values, etc. This in necessary to make the execution on many tests
        tractable where there are many columns and/or many rows.
        """

        if self.single_test_summary_df is None:
            self.single_test_summary_df = pd.DataFrame(self.single_test_summary_dict).T
            self.single_test_summary_df = self.single_test_summary_df * 100.0 / self.num_rows
            self.single_test_summary_df = self.single_test_summary_df.fillna("-")
        return self.single_test_summary_df

    def get_patterns_list(
        self,
        test_exclude_list: list[str] | None = None,
        column_exclude_list: list[str] | None = None,
        show_short_list_only: bool = True
    ) -> pd.DataFrame | None:
        """
        Get a DataFrame listing identified patterns without exceptions.

        Returns patterns discovered in the data, optionally filtered by test or column.
        Each row represents one pattern (one test on a set of columns).

        Args:
            test_exclude_list: Test IDs to exclude from results
            column_exclude_list: Column names to exclude from results
            show_short_list_only: If True, return only high-relevance (low-noise) patterns

        Returns:
            DataFrame with columns: Test ID, Column(s), Description of Pattern
            Returns None if no patterns have been identified yet.
        """

        if self.patterns_df is None:
            return None
        if test_exclude_list is None and column_exclude_list is None and not show_short_list_only:
            return self._clean_column_names(self.patterns_df.drop(columns=['Display Information']))
        df = self.patterns_df.copy()
        if test_exclude_list:
            df = df[~df['Test ID'].isin(test_exclude_list)].copy()
        if column_exclude_list:
            df = df[~df['Column(s)'].isin(column_exclude_list)].copy()
        if show_short_list_only:
            len_prev = len(df)
            df = df[df['Test ID'].isin(self.get_patterns_shortlist())].copy()
            if len(df) < len_prev:
                print_text("Some patterns not shown. Set show_short_list_only to False to see additional patterns.")

        # Ensure the descriptions are not too long
        df['Description of Pattern'] = df['Description of Pattern'].apply(truncate_description)

        return self._clean_column_names(df.drop(columns=['Display Information']))

    def get_exceptions_list(self) -> pd.DataFrame | None:
        """
        Get a DataFrame listing patterns with exceptions.

        Returns patterns that were discovered with violations. Similar to get_patterns_list()
        but includes an additional column for the number of exceptions found.

        Returns:
            DataFrame with columns: Test ID, Column(s), Description, Number of Exceptions
            Returns None if no exceptions have been identified yet.
        """

        def clean_col_names(x):
            col_name = x['Column(s)']
            test_id = x['Test ID']
            return self._get_condensed_col_list(test_id, col_name)

        if self.exceptions_summary_df is None or self.exceptions_summary_df.empty:
            return None
        df = self.exceptions_summary_df.drop(columns=['Display Information']).copy()
        df['Column(s)'] = df.apply(clean_col_names, axis=1)

        # Ensure the descriptions are not too long
        df['Description of Pattern'] = df['Description of Pattern'].apply(truncate_description)

        return df

    def get_exceptions(self):
        """
        Returns a dataframe with the same set of rows as the original dataframe, but a column for each pattern that was
        discovered that had exceptions, and a column indicating the final score for each row. This dataframe can
        be very large and is not generally useful to display, but may be collected for further analysis.

        Returns:
            pandas.DataFrame: DataFrame containing exceptions and final scores
        """

        return self.test_results_df

    def get_exceptions_by_column(self):
        """
        Returns a dataframe with the same shape as the original dataframe, but with each cell containing, instead
        of the original value for each feature for each row, a score allocated to that cell. Each pattern with
        exceptions has a score of 1.0, but patterns the cover multiple columns will give each cell a fraction of this.
        For example with a pattern covering 4 columns, any cells that are flagged will receive a score of 0.25 for
        this pattern. Each cell will have the sum of all patterns with exceptions where they are flagged.
        """

        return pd.DataFrame(self.test_results_by_column_np, columns=self.orig_df.columns)

    def summarize_patterns_by_test_and_feature(self, all_tests=False, heatmap=False):
        """
        Create and return a dataframe with a row for each test and a column for each feature in the original data. Each
        cell has a 0 or 1, indicating if the pattern was found without exceptions in that feature. Note, some tests to
        not identify patterns, such as VERY_LARGE. The dataframe is returned, and optionally displayed as a heatmap.

        all_tests: bool
            If all_tests is True, all tests are included in the output, even those that found no patterns. This may be
            used specifically to check which found no patterns.

        heatmap: bool
            If True, a heatmap will be displayed
        """

        if self.patterns_df is None:
            return None

        summary_arr = []
        cols = ['Test ID']
        cols.extend(self.orig_df.columns)
        for test_id in self.get_test_list():
            test_sub_df = self.patterns_df[self.patterns_df['Test ID'] == test_id]
            if (not all_tests) and test_sub_df.empty:
                continue
            test_arr = [test_id]
            flagged_cols = []
            for i in test_sub_df.index:
                sub_df_row_cols = test_sub_df.loc[i]['Column(s)']
                sub_df_row_cols = [x.lstrip('"').rstrip('"') for x in sub_df_row_cols.split(" AND ")]
                flagged_cols.extend(sub_df_row_cols)
            flagged_cols = list(set(flagged_cols))
            for col_name in self.orig_df.columns:
                if col_name in flagged_cols:
                    test_arr.append(1)
                else:
                    test_arr.append(0)
            summary_arr.append(test_arr)
        df = pd.DataFrame(summary_arr, columns=cols)
        df = df.set_index("Test ID", drop=True)

        if heatmap and len(df):
            plt.subplots(figsize=(len(df.columns) * 0.5, len(df) * 0.5))
            s = sns.heatmap(df, cmap='Blues', linewidths=1.1, linecolor='black', cbar=False)
            s.set(ylabel=None)
            plt.show()

        # Replace any '1' values with checkmarks to make the display more clear.
        df = df.replace(1, u'\u2714')
        df = df.replace(0, '')

        return df

    def summarize_exceptions_by_test_and_feature(self, all_tests=False, heatmap=False):
        """
        Create a dataframe with a row for each test and a column for each feature in the original data. Each cell has an
        integer, indicating, if the pattern was found in that feature, the number of rows that were flagged. Note,
        at most contamination_level of the rows (0.5% by default) may be flagged for any test in any feature, as this
        checks for exceptions to well-established patterns.

        all_tests: bool
            If all_tests is True, all tests are included in the output, even those that found no issues. This may be
            used specifically to check which found no issues.

        heatmap: bool:
            If set True, a heatmap of the dataframe will be displayed.
        """

        if self.exceptions_summary_df is None:
            return None

        summary_arr = []  # Has a row for each test that has has at least one pattern with exceptions.
        for test_id in self.get_test_list():
            test_sub_df = self.exceptions_summary_df[self.exceptions_summary_df['Test ID'] == test_id]
            if (not all_tests) and (len(test_sub_df) == 0):
                continue
            test_arr = [test_id]
            test_arr.extend([0]*len(self.orig_df.columns))

            for i in test_sub_df.index:
                num_exceptions = test_sub_df.loc[i, 'Number of Exceptions']
                sub_df_row_cols = test_sub_df.loc[i]['Column(s)']
                sub_df_row_cols = [x.lstrip('"').rstrip('"') for x in sub_df_row_cols.split(" AND ")]
                for col_name in sub_df_row_cols:
                    col_idx = self.orig_df.columns.tolist().index(col_name)
                    test_arr[col_idx + 1] += num_exceptions

            summary_arr.append(test_arr)

        cols = ['Test ID']
        cols.extend(self.orig_df.columns)
        df = pd.DataFrame(summary_arr, columns=cols)

        df = df.set_index("Test ID", drop=True)
        if heatmap:
            plt.subplots(figsize=(len(df.columns)*0.5, len(df) * 0.5))
            s = sns.heatmap(df, cmap='Blues', linewidths=1.1, linecolor='black', annot=True, fmt="d")
            s.set(ylabel=None)
            plt.show()

        # Replace any 0 values with a blank to make the display more clear.
        df = df.replace(0, '')

        return df

    def summarize_patterns_by_test(self, heatmap=False):
        """
        Create and return a dataframe with a row for each test, indicating the number of features where the pattern
        was found.

        heatmap: bool
            If True, a heatmap of the results are displayed
        """

        if self.patterns_df is None:
            return None

        g = self.patterns_df.groupby('Test ID')
        df = pd.DataFrame({
            'Test ID': list(g.groups),
            'Number of Columns Flagged': list(g['Column(s)'].nunique())
        })

        # Use the consistent ordering for the tests
        df['Test ID'] = df['Test ID'].astype("category")
        df['Test ID'] = df['Test ID'].cat.set_categories(self.get_test_list())
        df = df.sort_values(['Test ID'])

        df = df.set_index("Test ID", drop=True)
        if heatmap:
            fig, ax = plt.subplots(figsize=(len(df.columns) * 0.5, len(df) * 0.5))
            s = sns.heatmap(df, cmap='Blues', linewidths=0.75, linecolor='black', annot=True, fmt="d")
            s.set(ylabel=None)
            s.set(xticklabels=[])
            print("Counts: Number Columns Flagged at Least Once, and Total Number of Issues:")
            plt.show()
        return df

    def summarize_exceptions_by_test(self, heatmap=False):
        """
        Create and return a dataframe with a row for each test, indicating 1) the number of features where the pattern
        was found but also exceptions, 2) the number of issues total flagged (across all features and all rows).

        heatmap: bool
            If True, a heatmap of the results are displayed
        """

        if self.exceptions_summary_df is None:
            return None

        g = self.exceptions_summary_df.groupby('Test ID')
        row_counts = []
        for test_id in g.groups:
            result_cols = [c for c in self.test_results_df.columns if c.startswith(f"TEST {test_id} --")]
            if result_cols:
                num_rows = self.test_results_df[result_cols].any(axis=1).sum()
            else:
                num_rows = 0
            row_counts.append(num_rows)

        df = pd.DataFrame({
            'Test ID': list(g.groups),
            'Number of Columns Flagged At Least Once': list(g['Column(s)'].nunique()),
            'Number of Rows Flagged At Least Once': row_counts,
            'Number of Issues Total': list(g['Number of Exceptions'].sum())
        })

        # Use the consistent ordering for the tests
        df['Test ID'] = df['Test ID'].astype("category")
        df['Test ID'] = df['Test ID'].cat.set_categories(self.get_test_list())
        df = df.sort_values(['Test ID'])

        df = df.set_index("Test ID", drop=True)
        if heatmap:
            fig, ax = plt.subplots(figsize=(len(df.columns) * 0.5, len(df) * 0.5))
            s = sns.heatmap(df, cmap='Blues', linewidths=0.75, linecolor='black', annot=True, fmt="d")
            s.set(ylabel=None)
            s.set(xticklabels=[])
            print("Counts: Number Columns Flagged at Least Once, and Total Number of Issues:")
            plt.show()
        return df

    def summarize_patterns_and_exceptions(self, all_tests: bool = False, heatmap: bool = False) -> pd.DataFrame:
        """Summarize patterns and exceptions for each test.

        Args:
            all_tests: Include tests with no patterns when ``True``.
            heatmap: Display a heatmap when ``True``.

        Returns:
            ``pandas.DataFrame`` with counts of patterns with and without exceptions.
        """
        vals = []
        for test_id in self.get_test_list():
            if (not all_tests) and \
                    (test_id not in self.patterns_df['Test ID'].values) and \
                    (test_id not in self.exceptions_summary_df['Test ID'].values):
                continue
            sub_patterns_test = self.patterns_df[self.patterns_df['Test ID'] == test_id]
            sub_results_summary_test = self.exceptions_summary_df[self.exceptions_summary_df['Test ID'] == test_id]
            vals.append([test_id, len(sub_patterns_test), len(sub_results_summary_test)])
        df = pd.DataFrame(vals, columns=['Test ID',
                                         'Number Patterns without Exceptions',
                                         'Number Patterns with Exceptions'])
        if heatmap:
            plot_df = df.set_index('Test ID')
            fig, ax = plt.subplots(figsize=(len(plot_df.columns) * 0.6, len(plot_df) * 0.5))
            sns.heatmap(plot_df, cmap='YlGnBu', annot=True, fmt='d', linewidths=0.75, linecolor='black', ax=ax)
            ax.set_ylabel(None)
            plt.show()

        df = df.replace(0, '')
        return df

    def get_outlier_scores(self, normalized: bool = False) -> list[int] | list[float]:
        """Return row-level outlier scores.

        Args:
            normalized: If False, return the historical raw count of patterns
                flagging each row. If True, divide that count by the number of
                active exception-result columns, yielding values in [0, 1].

        Returns:
            One score per row in the original dataframe.
        """

        if self.test_results_df is not None and not self.test_results_df.empty:
            score_column = "NORMALIZED SCORE" if normalized else "FINAL SCORE"
            if score_column not in self.test_results_df.columns:
                self._calculate_final_scores()
            return self.test_results_df[score_column].tolist()

        if normalized:
            return [0.0] * len(self.orig_df)
        return [0] * len(self.orig_df)

    def get_outlier_score_summary(self) -> pd.DataFrame:
        """Return raw and normalized outlier scores in a stable dataframe.

        The returned index matches the checker row index. A copy is returned so
        callers cannot mutate checker state.
        """

        if self.test_results_df is None or self.test_results_df.empty:
            index = (
                self.orig_df.index.copy()
                if self.orig_df is not None
                else pd.RangeIndex(0)
            )
            return pd.DataFrame(
                {
                    "FINAL SCORE": [0] * len(index),
                    "NORMALIZED SCORE": [0.0] * len(index),
                },
                index=index,
            )

        required = {"FINAL SCORE", "NORMALIZED SCORE"}
        if not required.issubset(self.test_results_df.columns):
            self._calculate_final_scores()

        return self.test_results_df[
            ["FINAL SCORE", "NORMALIZED SCORE"]
        ].copy()

    def get_results_by_row_id(self, row_num):
        """
        Returns a list of tuples, with each tuple containing a test ID, and column name, for all issues flagged in the
        specified row.
        """

        if self.test_results_df is None:
            return []

        if row_num > len(self.test_results_df):
            print(f"Cannot display results for row {row_num}. {len(self.test_results_df)} rows available.")
            return []

        row = self.test_results_df.iloc[row_num]
        issues_list = []
        for c in row.index:
            if " -- " not in c:
                continue
            if row[c] == 0:
                continue
            test_id, col_name = c.split(" -- ")
            test_id = test_id.replace("TEST ", "")
            col_name = col_name.replace(" RESULT", "")
            issues_list.append((test_id, col_name))
        return issues_list

    def _get_condensed_col_list(self, test_id, col_name):
        if test_id in ['MISSING_VALUES_PER_ROW', 'UNIQUE_VALUES_PER_ROW']:
            s = "This test executes over all columns"
        elif test_id in ['ZERO_VALUES_PER_ROW', 'NEGATIVE_VALUES_PER_ROW', 'SMALL_AVG_RANK_PER_ROW',
                         'LARGE_AVG_RANK_PER_ROW']:
            s = "This test executes over all numeric columns"
        else:
            s = col_name
        return s

    def _clean_column_names(self, df):
        """
        Where tests operate over many features, the list of features can become difficult to read.
        """

        clean_df = df.copy()
        idxs = np.where(df['Test ID'].isin(['MISSING_VALUES_PER_ROW', 'UNIQUE_VALUES_PER_ROW']))[0].tolist()
        for idx in idxs:
            clean_df.loc[clean_df.index[idx], 'Column(s)'] = 'This test executes over all columns'

        idxs = np.where(df['Test ID'].isin(['ZERO_VALUES_PER_ROW', 'NEGATIVE_VALUES_PER_ROW', 'SMALL_AVG_RANK_PER_ROW',
                                            'LARGE_AVG_RANK_PER_ROW']))[0].tolist()
        for idx in idxs:
            clean_df.loc[clean_df.index[idx], 'Column(s)'] = 'This test executes over all numeric columns'
        return clean_df

    def _get_rows_flagged(self, test_id, col_name):
        """
        Return the subset of the original data where the specified test flagged an issue in the specified column
        """

        results_col_name = self.get_results_col_name(test_id, col_name)
        if results_col_name not in self.test_results_df.columns:
            return None
        df = self.test_results_df[self.test_results_df[results_col_name] == 1]
        if len(df) == 0:
            return None
        df.index = [x[0] if type(x) == tuple else x for x in df.index]
        row_idxs = df.index
        return self.orig_df.loc[row_idxs]

    def _add_result_column(self, col_name, col_values):
        """
        Add a column with the specified name and values to self.test_results_df.
        This uses concat(), instead of simply adding columns, to avoid inefficiency issues.
        """

        self.test_results_df = pd.concat([
            self.test_results_df,
            pd.DataFrame({col_name: col_values})],
            axis=1)

    def clear_results(
            self,
            test_id_list=None,
            col_name_list=None,
            pattern_id_list=None,
            issue_id_list=None,
            clear_code_tests=False,
            clear_all_patterns=False,
            clear_all_exceptions=False):
        """
        This may be used to iteratively clean the results until the DataConstencyChecker object has an appropriate
        set of patterns and exceptions. This may be done, for example, to pass the DataConstencyChecker on for
        further processing or to generate a report, or, for example, until all issues are acknowledged or understood,
        or until the set of results is zero.

        There are several parameters that may be used to specify which patterns or exceptions to remove. Only one
        may be specified at a time.

        test_id_list: array of test IDs
            If set, this will remove any patterns or exceptions based on any of these tests.

        col_name_list: array of column names
            If set, this will remove any patterns or exceptions based on any of these column names. This includes
            results that are based on other features as well.

        pattern_id_list: array of integers
            If set, this will remove the specified patterns. This will not affect the set of exceptions.

        issue_id_list: array of integers
            If set, this will remove the specified exceptions. This will not affect the set of patterns.

        clear_code_tests: bool
            If set, all patterns and exceptions related to all tests that are specifif to code and ID values will be
            removed.

        clear_all_patterns: bool
            If set, this will remove all patterns. This will not affect the set of exceptions.

        clear_all_exceptions: bool
            If set, this will remove all exceptions. This will not affect the set of patterns.
        """

        def check_col_includes_list(x):
            cols_arr = self.col_to_original_cols_dict[x]
            return len(set(cols_arr).intersection(col_name_list)) == 0

        def check_exception_col_includes_list(row):
            cols_arr = self.col_to_original_cols_dict[self.get_results_col_name(row[0], row[1])]
            return len(set(cols_arr).intersection(col_name_list)) == 0

        num_specfied = 0
        if test_id_list is not None:
            num_specfied += 1
        if col_name_list is not None:
            num_specfied += 1
        if issue_id_list is not None:
            num_specfied += 1
        if pattern_id_list is not None:
            num_specfied += 1
        if clear_code_tests:
            num_specfied += 1
        if clear_all_patterns:
            num_specfied += 1
        if clear_all_exceptions:
            num_specfied += 1

        if num_specfied == 0:
            print("No method of removing results specified. Cannot execute function.")
            return

        if num_specfied > 1:
            print(("Only one method or removing results may be specified in each call to clear_results(). The function "
                   "may be called any number of times."))
            return

        if (self.exceptions_summary_df is None) or (self.test_results_df is None) or (self.exceptions_summary_df is None):
            print("There are no results to clear. Cannot execute function.")
            return

        if clear_code_tests:
            test_id_list = self.get_tests_for_codes()

        if test_id_list is not None:
            # patterns_arr
            self.patterns_arr = [x for x in self.patterns_arr if x[0] not in test_id_list]

            # patterns_df
            self.patterns_df = self.patterns_df[~self.patterns_df['Test ID'].isin(test_id_list)]

            # results_summary_arr
            self.results_summary_arr = [x for x in self.results_summary_arr if x[0] not in test_id_list]

            # exceptions_summary_df
            self.exceptions_summary_df = self.exceptions_summary_df[~self.exceptions_summary_df['Test ID'].isin(test_id_list)]

            # test_results_df
            drop_cols = []
            for col_name in self.test_results_df:
                test_name = col_name.replace('TEST ', '').split(' -- ')[0]
                if test_name in test_id_list:
                    drop_cols.append(col_name)
            self.test_results_df = self.test_results_df.drop(columns=drop_cols)

            # results_dict
            for key in list(self.results_dict.keys()):
                test_name = key.replace('TEST ', '').split(' -- ')[0]
                if test_name in test_id_list:
                    self.results_dict.pop(key)

        if col_name_list is not None:
            # patterns_arr
            cols_list_arr = [self.col_to_original_cols_dict[x[1]] for x in self.patterns_arr]
            self.patterns_arr = [x for x, y in
                                 zip(self.patterns_arr, cols_list_arr) if not set(x).intersection(set(col_name_list))]

            # patterns_df
            self.patterns_df = self.patterns_df[self.patterns_df['Column(s)'].apply(check_col_includes_list)]

            # results_summary_arr
            cols_list_arr = [self.col_to_original_cols_dict[self.get_results_col_name(x[0], x[1])] for x in self.results_summary_arr]
            self.results_summary_arr = [x for x, y in
                                 zip(self.results_summary_arr, cols_list_arr) if not set(y).intersection(set(col_name_list))]

            # exceptions_summary_df
            self.exceptions_summary_df = self.exceptions_summary_df[
                self.exceptions_summary_df.apply(check_exception_col_includes_list, axis=1)]

            # test_results_df
            drop_cols = []
            for col_name in self.test_results_df:
                if col_name not in self.col_to_original_cols_dict:  # Skip "FINAL SCORE"
                    continue
                col_names = self.col_to_original_cols_dict[col_name]
                if set(col_names).intersection(col_name_list):
                    drop_cols.append(col_name)
            self.test_results_df = self.test_results_df.drop(columns=drop_cols)

            # results_dict
            for key in list(self.results_dict.keys()):
                col_names = self.col_to_original_cols_dict[key]
                if set(col_names).intersection(col_name_list):
                    self.results_dict.pop(key)

        if pattern_id_list is not None:
            # patterns_arr
            self.patterns_df.reset_index(drop=True)
            row_nums = self.patterns_df[self.patterns_df['Pattern ID'].isin(pattern_id_list)].index
            for row_num in sorted(row_nums, reverse=True):
                del self.patterns_arr[row_num]

            # patterns_df
            self.patterns_df = self.patterns_df[~self.patterns_df['Pattern ID'].isin(pattern_id_list)]

            # results_summary_arr, exceptions_summary_df, test_results_df, results_dict
            #   patterns_id_list applies only to patterns

        if issue_id_list is not None:
            # patterns_arr, patterns_df
            #   issue_id_list applies only to exceptions

            # results_summary_arr
            self.exceptions_summary_df.reset_index(drop=True)
            row_nums = self.exceptions_summary_df[self.exceptions_summary_df['Issue ID'].isin(issue_id_list)].index
            for row_num in sorted(row_nums, reverse=True):
                del self.results_summary_arr[row_num]

            # exceptions_summary_df
            # Before removing, get a list of the results column names for the issues
            results_col_names = []
            sub_df = self.exceptions_summary_df[self.exceptions_summary_df['Issue ID'].isin(issue_id_list)]
            for i in sub_df.index:
                row = sub_df.loc[i]
                results_col_names.append(self.get_results_col_name(row[0], row[1]))
            self.exceptions_summary_df = self.exceptions_summary_df[~self.exceptions_summary_df['Issue ID'].isin(issue_id_list)]

            # test_results_df
            self.test_results_df = self.test_results_df.drop(columns=results_col_names)

            # results_dict
            for key in list(self.results_dict.keys()):
                if key in results_col_names:
                    self.results_dict.pop(key)

        if clear_all_patterns:
            # patterns_arr
            self.patterns_arr = []

            # patterns_df
            self.patterns_df = self.patterns_df[0:0]

            # results_summary_arr, exceptions_summary_df, test_results_df, results_dict, test_results_by_column_np
            #   clear_all_patterns applies only to patterns

        if clear_all_exceptions:
            # patterns_arr, patterns_df
            #   clear_all_exceptions applies only to exceptions

            # results_summary_arr
            self.results_summary_arr = []

            # exceptions_summary_df
            self.exceptions_summary_df = self.exceptions_summary_df[0:0]

            # test_results_df
            self.test_results_df = self.test_results_df[0:0]

            # results_dict
            self.results_dict = {}

    def restore_results(self):
        """
        This resets all the discovered results back to the state when check_data_quality() was last called.
        """
        self.patterns_arr = self.safe_patterns_arr.copy()
        self.patterns_df = self.safe_patterns_df.copy()
        self.results_summary_arr = self.safe_results_summary_arr.copy()
        self.exceptions_summary_df = self.safe_exceptions_summary_df.copy()
        self.test_results_df = self.safe_test_results_df.copy()
        self.results_dict = self.safe_results_dict.copy()
        self.test_results_by_column_np = self.safe_test_results_by_column_np.copy()


    def _calculate_final_scores(self) -> None:
        """Calculate raw and normalized row-level outlier scores.

        Only per-pattern result columns contribute to the score. Derived score
        columns are excluded explicitly, so recalculating scores cannot inflate
        them during repeated or appended analyses.

        ``FINAL SCORE`` remains the historical raw count of exception-bearing
        patterns that flagged each row. ``NORMALIZED SCORE`` divides that count
        by the number of active result columns, producing a value in [0, 1].
        """

        if self.test_results_df is None:
            return

        result_columns = [
            column for column in self.test_results_df.columns if " -- " in column
        ]
        if not result_columns:
            self.test_results_df["FINAL SCORE"] = 0
            self.test_results_df["NORMALIZED SCORE"] = 0.0
            return

        raw_scores = self.test_results_df[result_columns].sum(axis=1)
        self.test_results_df["FINAL SCORE"] = raw_scores
        self.test_results_df["NORMALIZED SCORE"] = raw_scores / len(result_columns)
