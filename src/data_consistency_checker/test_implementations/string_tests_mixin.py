"""
StringTestsMixin: Test methods for string tests.

This mixin contains 118 test methods organized by category.
Extracted from check_data_consistency.py for better code organization.
"""

from __future__ import annotations

import datetime
import math
import random
import statistics
import string

import numpy as np
import pandas as pd
import pandas.api.types as pandas_types
from dateutil.relativedelta import relativedelta
from sklearn import metrics, tree
from sklearn.tree import DecisionTreeClassifier

try:
    from termcolor import colored
except ImportError:  # pragma: no cover - optional presentation dependency
    colored = None

from data_consistency_checker.checker_utils import (
    array_to_str,
    convert_to_numeric,
    get_non_alphanumeric,
    is_missing,
    is_uppercase,
    replace_special_with_space,
)

letters = string.ascii_letters
digits = string.digits
alphanumeric = letters + digits


def _shared_fraction(x, y) -> float:
    """Fraction of the distinct items in x or y that appear in both. Two empty collections are identical."""
    union = set(x).union(set(y))
    if not union:
        return 1.0
    return len(set(x).intersection(set(y))) / len(union)


class StringTestsMixin:
    """
    Mixin class containing string tests methods.

    This class contains 118 methods for testing data consistency
    related to string operations.

    This is a mixin class designed to be used with multiple inheritance.
    It does not have an __init__ method and relies on the parent class
    to provide necessary attributes and methods.
    """

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
                idxs = list(np.where(test_series == False)[0])  # noqa: E712
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



    def _generate_blank(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'blank_vals_most' has consistently non-blank values, with 2 exceptions
        """
        self._add_synthetic_column('blank_vals_rand',
                                    ["aaaa"] * (self.num_synth_rows//2) +
                                    [""] * (self.num_synth_rows//4) +
                                    ["   "] * (self.num_synth_rows//4))
        self._add_synthetic_column('blank_vals_most', ["aaaa"] * (self.num_synth_rows - 2) + [""] + ["   "])


    def _check_blank(self, test_id):
        for col_name in self.string_cols:
            test_series = (self.orig_df[col_name].astype(str) != "") & (~self.orig_df[col_name].astype(str).str.isspace())
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'Column "{col_name}" consistently contains non-blank values',
                allow_patterns=False
            )


    def _generate_leading_whitespace(self):
        """
        Patterns without exceptions: None. 'lead_white_all' has consistently zero leading white space, but this is
            not flagged as a pattern.
        Patterns with exception: 'lead_white_most_1' has consistently 0 leading whitespace characters, with 1 exception.
            'lead_white_most_2' has consistently 2 leading whitespace characters, with 2 exceptions
        """
        self._add_synthetic_column('lead_white_rand',
                                    ["aaaaaaa"] * (self.num_synth_rows // 4) +
                                    [" aaaaaa"] * (self.num_synth_rows // 4) +
                                    ["  aaaaa"] * (self.num_synth_rows // 4) +
                                    ["   aaaa"] * (self.num_synth_rows // 4))
        self._add_synthetic_column('lead_white_all', ["a"] * (self.num_synth_rows - 2) + ["aa"] + ["aaa"])
        self._add_synthetic_column('lead_white_most_1', ["aaaa"] * (self.num_synth_rows - 2) + ["      aaa"] + ["       aaa"])
        self._add_synthetic_column('lead_white_most_2', ["  aaaa"] * (self.num_synth_rows - 2) + ["aaa"] + ["      aaa"])


    def _check_leading_whitespace(self, test_id):
        """
        This does not report columns with consistently zero leading whitespace characters as a pattern, as this is
        normally the case and not considered interesting.
        """

        for col_name in self.string_cols:
            test_series_counts = self.orig_df[col_name].astype(str).str.len() - \
                                 self.orig_df[col_name].astype(str).str.lstrip().str.len()
            if test_series_counts.max() == 0:
                continue

            test_series_non_null_counts = pd.Series([
                (x-y)
                for w, x, y in zip(self.orig_df[col_name],
                                   self.orig_df[col_name].astype(str).str.len(),
                                   self.orig_df[col_name].astype(str).str.lstrip().str.len())
                if (not is_missing(w))])

            # Test if there are normally zero leading spaces
            test_series = self.orig_df[col_name].isna() | (test_series_counts == 0)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'Column "{col_name}" consistently contains values without leading spaces'
            )

            # Test if there are a normal number of trailing spaces
            median_num_spaces = test_series_non_null_counts.median()
            test_series = self.orig_df[col_name].apply(is_missing) | \
                          ((test_series_counts > (median_num_spaces / 2)) & \
                           (test_series_counts < (median_num_spaces * 2)))
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f'Column "{col_name}" consistently contains about {median_num_spaces} leading spaces (minimum: '
                 f'{test_series_counts.min()}, maximum: {test_series_counts.max()})'),
                ' with significantly more or less leading spaces'
            )


    def _generate_trailing_whitespace(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'trail_white_most' has consistently 2 trailing whitespace characters, with 2 exceptions
        """
        self._add_synthetic_column('trail_white_rand',
                                    ["aaaaaaa"] * (self.num_synth_rows // 4) + \
                                    ["aaaaaa "] * (self.num_synth_rows // 4) + \
                                    ["aaaaa  "] * (self.num_synth_rows // 4)+ \
                                    ["aaaa   "] * (self.num_synth_rows // 4))
        self._add_synthetic_column('trail_white_all', ["a"] * (self.num_synth_rows - 2) + ["aa"] + ["aaa"])
        self._add_synthetic_column('trail_white_most', ["aaaa  "] * (self.num_synth_rows - 2) + ["aaa"] + ["aaa      "])


    def _check_trailing_whitespace(self, test_id):
        for col_name in self.string_cols:
            test_series_counts = self.orig_df[col_name].astype(str).str.len() - self.orig_df[col_name].astype(str).str.rstrip().str.len()
            if test_series_counts.max() == 0:
                continue
            test_series_non_null_counts = pd.Series([
                (x-y)
                for w, x, y in zip(self.orig_df[col_name],
                                   self.orig_df[col_name].astype(str).str.len(),
                                   self.orig_df[col_name].astype(str).str.rstrip().str.len())
                if (not is_missing(w))])

            # Test if there are normally zero trailing spaces
            test_series = self.orig_df[col_name].isna() | (test_series_counts == 0)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'Column "{col_name}" consistently contains values without trailing spaces'
            )

            # Test if there are a normal number of trailing spaces
            median_num_spaces = test_series_non_null_counts.median()
            test_series = self.orig_df[col_name].apply(is_missing) | \
                          ((test_series_counts > (median_num_spaces / 2)) & \
                           (test_series_counts < (median_num_spaces * 2)))
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f'Column "{col_name}" consistently contains about {median_num_spaces} trailing spaces (minimum: '
                 f'{test_series_counts.min()}, maximum: {test_series_counts.max()})'),
                ' with significantly more or less trailing spaces'
            )


    def _generate_first_char_alpha(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('first_char_alpha_rand',
            [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_alpha_all',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_alpha_most',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['8abcd'])


    def _check_first_char_alpha(self, test_id):
        for col_name in self.string_cols:
            # Skip columns where there is only one character used for the first character in all values
            first_chars = self.orig_df[col_name].astype(str).str[:1]
            if first_chars.nunique() == 1:
                continue

            # Skip columns that have mostly a single character
            value_lens_arr = pd.Series(self.orig_df[col_name].astype(str).str.len())
            if value_lens_arr.quantile(0.9) <= 1:
                continue

            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.lstrip().str.slice(0, 1).str.isalpha()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.lstrip().str.slice(0, 1).str.isalpha()
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently starts with an alphabetic character")


    def _generate_first_char_numeric(self):
        """
        Patterns without exceptions: 'first_char_numeric_2' consistently begins with a digit
        Patterns with exception: 'first_char_numeric_3' consistently begins with a digit, with 1 exception
        """
        self._add_synthetic_column('first_char_numeric_1',
            [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_numeric_2',
            [['x'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_numeric_3',
            [['x'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])


    def _check_first_char_numeric(self, test_id):
        for col_name in self.string_cols:
            # Skip columns where there is only one character used for the first character in all values
            first_chars = self.orig_df[col_name].astype(str).str[:1]
            if first_chars.nunique() == 1:
                continue

            # Skip columns that have mostly a single character
            value_lens_arr = pd.Series(self.orig_df[col_name].astype(str).str.len())
            if value_lens_arr.quantile(0.9) <= 1:
                continue

            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.slice(0, 1).str.isdigit()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.slice(0, 1).str.isdigit()
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently starts with a numeric character")


    def _generate_first_char_small_set(self):
        """
        Patterns without exceptions: 'first_char_small_set_all' consistently start with a, b, or c
        Patterns with exception: 'first_char_small_set_most' consistently start with a, b, or c, with one exception
        """
        self._add_synthetic_column('first_char_small_set_rand',
            [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_small_set_all',
            [[''.join(random.choice(['A', 'B', 'C']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_small_set_most',
            [[''.join(random.choice(['A', 'B', 'C']) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['xabcde'])


    def _check_first_char_small_set(self, test_id):
        """
        This test applies to ID and code values, where the first characters may have a specific meaning.
        """

        for col_name in self.string_cols:
            # todo: Skip columns that have only one character for all values

            # Skip columns that have few distinct values
            if self.orig_df[col_name].nunique() < math.sqrt(self.num_rows):
                continue

            test_series = pd.Series([str(x)[:1] if not y else None for x, y in zip(self.orig_df[col_name], self.orig_df[col_name].isna())])
            num_non_null_vals = self.orig_df[col_name].notna().sum()

            counts_series = test_series.value_counts(normalize=False)  # Get the counts for each first letter

            # Check if there's a small set of characters that make up the bulk of the values
            count_most_common = np.where(
                counts_series.sort_values(ascending=False).cumsum() > (num_non_null_vals - self.freq_contamination_level))[0][0]
            if count_most_common <= 5:
                common_first_chars = list(counts_series.sort_values(ascending=False)[:count_most_common+1].index)
                rare_first_chars = []
                for c in counts_series.index:
                    if c not in common_first_chars:
                        rare_first_chars.append(c)
                results_col = test_series.isin(common_first_chars) | self.orig_df[col_name].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    results_col,
                    (f'The column "{col_name}" has {self.orig_df[col_name].nunique()} distinct values and contains '
                     f'values that consistently start with one of {common_first_chars}'),
                    f' with some values starting with one of {rare_first_chars}'
                )


    def _generate_first_char_uppercase(self):
        """
        Patterns without exceptions: 'first_char_upper all' consistently starts with an upper case character
        Patterns with exception: 'first_char_upper most' consistently starts with an upper case character, with one
            exception.
        """
        self._add_synthetic_column('first_char_upper rand',
            [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_upper all',
            [[''.join(random.choice(['A', 'B', 'C', 'Á']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_upper most',
            [[''.join(random.choice(['A', 'B', 'C', 'Á']) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])


    def _check_first_char_uppercase(self, test_id):
        for col_name in self.string_cols:
            # Skip columns where there is only one character used for the first character in all values
            first_chars = self.orig_df[col_name].astype(str).str[:1]
            if first_chars.nunique() == 1:
                continue

            # Skip columns that have mostly a single character
            value_lens_arr = pd.Series(self.orig_df[col_name].astype(str).str.len())
            if value_lens_arr.quantile(0.9) <= 1:
                continue

            test_series = self.orig_df[col_name].astype(str).str[:1].apply(is_uppercase)
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently starts with an uppercase character")


    def _generate_first_char_lowercase(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('first_char_lower rand',
                                    [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_lower all',
                                    [[''.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_char_lower most',
                                    [[''.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['Abcde'])


    def _check_first_char_lowercase(self, test_id):
        for col_name in self.string_cols:
            # Skip columns where there is only one character used for the first character in all values
            first_chars = self.orig_df[col_name].astype(str).str[:1]
            if first_chars.nunique() == 1:
                continue

            # Skip columns that have mostly a single character
            value_lens_arr = pd.Series(self.orig_df[col_name].astype(str).str.len())
            if value_lens_arr.quantile(0.9) <= 1:
                continue

            test_series = self.orig_df[col_name].astype(str).str[:1].isin(list(string.ascii_lowercase))
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently starts with an lowercase character")


    def _generate_last_char_small_set(self):
        """
        Patterns without exceptions: 'last_char_small_set_all' consistently ends with one of a, b, c
        Patterns with exception: 'last_char_small_set_most' consistently ends with one of a, b, c, with one exception
        """
        self._add_synthetic_column('last_char_small_set_rand',
            [[''.join(random.choice(alphanumeric) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('last_char_small_set_all',
            [[''.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('last_char_small_set_most',
            [[''.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])


    def _check_last_char_small_set(self, test_id):
        """
        This test applies to ID and code values, where the last characters may have a specific meaning.
        """

        for col_name in self.string_cols:

            # It may be trivially true that a column ends with a small set of characters if there are few unique
            # values in the column.
            if self.orig_df[col_name].nunique() < math.sqrt(self.num_rows):
                continue

            test_series = pd.Series(
                [str(x)[-1] if (not y) and (len(x) > 0) else None
                 for x, y in zip(self.orig_df[col_name].str.strip(), self.orig_df[col_name].isna())])
            num_non_null_vals = self.orig_df[col_name].notna().sum()
            counts_series = test_series.value_counts(normalize=False)

            # Check if there's a small set of characters that make up the bulk of the values
            count_most_common = np.where(
                counts_series.sort_values(ascending=False).cumsum() > (num_non_null_vals - self.freq_contamination_level))[0]
            if len(count_most_common) > 0:
                count_most_common = count_most_common[0]
            if count_most_common <= 5:
                common_last_chars = list(counts_series.sort_values(ascending=False)[:count_most_common+1].index)
                rare_last_chars = []
                for c in counts_series.index:
                    if c not in common_last_chars:
                        rare_last_chars.append(c)
                results_col = test_series.isin(common_last_chars) | self.orig_df[col_name].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    results_col,
                    (f'The column "{col_name}" has {self.orig_df[col_name].nunique()} distinct values and contains '
                     f'values that consistently end with one of {common_last_chars}'),
                    f' with some values ending with one of {rare_last_chars}'
                )


    def _generate_common_special_chars(self):
        """
        Patterns without exceptions: 'common_spec_chars all' consistently contains '*'
        Patterns with exception: 'common_spec_chars most' does as well, with 1 exception
        """
        self._add_synthetic_column('common_spec_chars rand',
            [[''.join(random.choice(list(alphanumeric) + ['*', '&', '#']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('common_spec_chars all',
            [['*'.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('common_spec_chars most',
            [['*'.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])


    def _check_common_special_chars(self, test_id):

        def get_non_alphanumeric(x):
            if x.isalnum():
                return []
            return [c for c in x if (not str(c).isalnum()) and (c != ' ')]

        for col_name in self.string_cols:
            # Skip columns which contain only alphanumeric characters
            if self.orig_df[col_name].astype(str).str.isalnum().tolist().count(False) == 0:
                continue

            # Get the set of special characters in each value
            special_chars_list = self.orig_df[col_name].astype(str).apply(get_non_alphanumeric)

            # Remove any empty lists
            special_chars_list = [x for x in special_chars_list if len(x)]

            # Skip columns where it is not the case that almost all values have special characters
            if (len(special_chars_list) + self.orig_df[col_name].isna().sum()) < (self.num_rows - self.freq_contamination_level):
                continue

            # Get the unique set of special characters
            special_chars_list = list({item for sublist in special_chars_list for item in sublist})

            # Examine each special character and determine if it is in most values
            common_special_chars_list = []
            for c in special_chars_list:
                if (self.orig_df[col_name].isna().sum() +
                    self.orig_df[col_name].astype(str).str.contains(c, regex=False).tolist().count(True)) > \
                        (self.num_rows - self.freq_contamination_level):
                    common_special_chars_list.append(c)

            if len(common_special_chars_list) == 0:
                continue

            # Having a space in each record is not interesting
            if common_special_chars_list == [' ']:
                continue

            # Identify the rows that do not contain all the common special characters
            test_series = [True] * self.num_rows
            for c in common_special_chars_list:
                test_series = test_series & self.orig_df[col_name].astype(str).str.contains(c, regex=False)
            test_series = test_series | self.orig_df[col_name].isna()

            chars_str = ""
            for c in common_special_chars_list:
                chars_str += c + ', '
            chars_str = chars_str[:-2]

            # In the string, remove the [] symbols from the string representation of an array.
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f"The column consistently contains the {chars_str} character(s)",
                allow_patterns=True)


    def _generate_common_chars(self):
        """
        Patterns without exceptions: None. 'repeated_chars all' contains an 'x' in all values, but this test does not
            generate patterns.
        Patterns with exception: 'repeated_chars most' contains an 'x' in most values, but not the last
        """
        self._add_synthetic_column('repeated_chars rand',
            [[''.join(random.choice(list(alphanumeric)) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('repeated_chars all',
            [['x'.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('repeated_chars most',
            [['x'.join(random.choice(['a', 'b', 'c']) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcde'])


    def _check_common_chars(self, test_id):
        def get_unique_chars(x):
            return list(set(x))

        for col_idx, col_name in enumerate(self.string_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 100 == 0:
                print(f"  Examining column {col_idx} of {len(self.string_cols)} string columns")

            # Skip columns that have mostly a single character
            value_lens_arr = pd.Series(self.orig_df[col_name].astype(str).str.len())
            if value_lens_arr.quantile(0.9) <= 1:
                continue

            # Skip columns that have over 10 characters. As this is for ID and code values, we do not check longer
            # strings
            if value_lens_arr.quantile(0.5) >= 10:
                continue

            # Skip columns that have few unique values
            if self.orig_df[col_name].nunique() < 10:
                continue

            # Test on a sample
            # Get the set of special characters in each value
            sample_unique_chars_list = self.sample_df[col_name].dropna().astype(str).apply(get_unique_chars)

            # Get the unique set of special characters
            sample_unique_chars_list = list({item for sublist in sample_unique_chars_list for item in sublist})

            # Examine each special character and determine if it is in most values
            common_chars_list = []
            for c in sample_unique_chars_list:
                if (self.sample_df[col_name].isna().sum() + \
                        self.sample_df[col_name].fillna("").astype(str).str.contains(c, regex=False).tolist().count(False)) <= 1:
                    common_chars_list.append(c)
            if len(common_chars_list) == 0:
                continue

            test_series = [True] * len(self.sample_df)
            for c in common_chars_list:
                test_series = test_series & self.sample_df[col_name].astype(str).str.contains(c, regex=False)
            test_series = test_series | (self.sample_df[col_name].isna())
            # Give some wiggle room, as the common_chars_list is based on a small sample
            if test_series.tolist().count(False) > 5:
                continue

            # Test on the full column
            # Get the set of special characters in each value
            unique_chars_list = self.orig_df[col_name].dropna().astype(str).apply(get_unique_chars)

            # Get the unique set of special characters
            unique_chars_list = list({item for sublist in unique_chars_list for item in sublist})

            # Examine each special character and determine if it is in most values
            common_chars_list = []
            for c in unique_chars_list:
                if (self.orig_df[col_name].isna().sum() + \
                        self.orig_df[col_name].fillna("").astype(str).str.contains(c, regex=False).tolist().count(True)) > \
                    (self.num_rows - self.freq_contamination_level):
                    common_chars_list.append(c)

            if len(common_chars_list) == 0:
                continue

            # Identify the rows that do not contain all the common special characters
            test_series = [True] * self.num_rows
            for c in common_chars_list:
                test_series = test_series & self.orig_df[col_name].astype(str).str.contains(c, regex=False)
            test_series = test_series | (self.orig_df[col_name].isnull())

            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f"The column consistently contains all of the following characters: {common_chars_list}",
                allow_patterns=False)


    def _generate_number_alpha_chars(self):
        """
        Patterns without exceptions: 'number_alpha_2' consistently has 5 alphabetic characters
        Patterns with exception: 'number_alpha_3 does as well, with 1 exception
        """
        self._add_synthetic_column('number_alpha_1',
            [[''.join(random.choice(letters)
                      for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('number_alpha_2',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('number_alpha_3',
            [[''.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['abcdef'])


    def _check_number_alpha_chars(self, test_id):
        """
        Handling null values: null values are considered zero-length strings, with no alphabetic, numeric, or
        special characters.
        """

        nunique_dict = self.get_nunique_dict()
        for col_name in self.string_cols:
            # Skip columns with very few unique values
            if nunique_dict[col_name] < 5:
                continue

            test_series = self.orig_df[col_name].fillna("").astype(str).apply(lambda x: len([e for e in x if e.isalpha()]))
            self._process_analysis_counts(
                test_id,
                [col_name],
                test_series,
                "The column contains values with consistently",
                "alphabetic characters")


    def _generate_number_numeric_chars(self):
        """
        Patterns without exceptions: 'number_numeric_2' consistently has 5 numeric characters.
        Patterns with exception: 'number_numeric_3' consistently has 5 numeric characters, with 1 exception.
        Not Flagged: 'number numeric 1' has random numbers of numeric characters.
        """
        self._add_synthetic_column('number_numeric_1',
            [[''.join(random.choice(letters + digits) for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('number_numeric_2',
            [['q'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('number_numeric_3',
            [['z'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['432'])


    def _check_number_numeric_chars(self, test_id):
        """
        Handling null values: null values are considered zero-length strings, with no alphabetic, numeric, or
        special characters.
        """

        n_unique_dict = self.get_nunique_dict()
        for col_name in self.string_cols:
            # Skip columns with very few unique values
            if n_unique_dict[col_name] < 5:
                continue

            test_series = self.orig_df[col_name].fillna("").astype(str).apply(lambda x: len([e for e in x if e.isdigit()]))
            self._process_analysis_counts(
                test_id,
                [col_name],
                test_series,
                "The column contains values with consistently",
                "numeric characters",
                allow_patterns=(test_series.max() != 0)
            )


    def _generate_number_alphanumeric_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('number_alphanumeric_1',
            [['$'.join(random.choice(digits) for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('number_alphanumeric_2',
            [['$'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('number_alphanumeric_3',
            [['$'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['$432'])


    def _check_number_alphanumeric_chars(self, test_id):
        """
        Handling null values: null values are considered zero-length strings, with no alphabetic, numeric, or
        special characters.
        """

        nunique_dict = self.get_nunique_dict()
        for col_name in self.string_cols:
            # Skip columns with very few unique values
            if nunique_dict[col_name] < 5:
                continue

            test_series = self.orig_df[col_name].fillna("").astype(str).apply(lambda x: len([e for e in x if e.isalnum()]))
            test_series = [test_series.median() if y else x for x, y in zip(test_series, self.orig_df[col_name].isnull())]
            self._process_analysis_counts(
                test_id,
                [col_name],
                test_series,
                "The column contains values with consistently",
                "alpha-numeric characters")


    def _generate_number_non_alphanumeric_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('non-alphanumeric rand',
            [['@'.join(random.choice(digits) for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('non-alphanumeric all',
            [['@'.join(random.choice(digits + ' ') for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('non-alphanumeric most',
            [['@'.join(random.choice(digits + ' ') for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['$432'])


    def _check_number_non_alphanumeric_chars(self, test_id):
        """
        Handling null values: null values are considered zero-length strings, with no alphabetic, numeric, or
        special characters.
        """

        n_unique_dict = self.get_nunique_dict()
        for col_name in self.string_cols:
            # Skip columns with very few unique values
            if n_unique_dict[col_name] < 5:
                continue

            test_series = (self.orig_df[col_name]
                           .fillna("")
                           .astype(str)
                           .apply(lambda x: len([e for e in x if (not e.isalnum()) and (e != ' ')])))
            self._process_analysis_counts(
                test_id,
                [col_name],
                test_series,
                "The column contains values with consistently",
                "non-alphanumeric characters",
                allow_patterns=(test_series.max() != 0),
                display_info={'test_series': test_series}
            )


    def _generate_number_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('num_chars rand',
            [['A'.join(random.choice(digits) for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('num_chars all',
            [['A'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('num_chars most',
            [['A'.join(random.choice(digits) for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['$432'])


    def _check_number_chars(self, test_id):
        """
        Handling null values: null values are considered zero-length strings, with no alphabetic, numeric, or
        special characters.
        """

        nunique_dict = self.get_nunique_dict()
        for col_name in self.string_cols:
            # Skip columns with very few unique values
            if nunique_dict[col_name] < 5:
                continue

            test_series = self.orig_df[col_name].fillna("").astype(str).str.len()
            self._process_analysis_counts(
                test_id,
                [col_name],
                test_series,
                "The column contains values with consistently",
                "characters")


    def _generate_nonprintable_chars(self):
        """
        Patterns without exceptions: This test does not generate patterns
        Patterns with exception: 'nonprintable one' has one value with a non-printable character
        """
        self._add_synthetic_column('nonprintable none',
            [['A'.join(random.choice(letters) for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('nonprintable one',
            [['A'.join(random.choice(letters) for _ in range(5))][0] for _ in range(self.num_synth_rows - 1)] + ['abcd\x10'])


    def _check_nonprintable_chars(self, test_id):
        """
        This test is different from most in that it simply considers any non-printable character an exception and does
        not look at the normal number or set of non-printable characters within a dataset.

        Handling null values: null values are considered zero-length strings, with no alphabetic, numeric, or
        special characters.
        """

        for col_name in self.string_cols:
            test_series = [len([y for y in x if not x.isprintable()]) == 0
                           for x in self.orig_df[col_name].fillna("").astype(str)]
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The strings contain one or more non-printable characters.",
                allow_patterns=False
            )


    def _generate_many_chars(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('many_chars all',
            [''.join(['a' for _ in range(np.random.randint(2, 20))]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('many_chars most',
            [''.join(['a' for _ in range(np.random.randint(2, 20))]) for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'many_chars most'] = ''.join(['a']*100)


    def _check_many_chars(self, test_id):
        for col_name in self.string_cols:
            val_lens = self.orig_df[col_name].fillna("").astype(str).str.len()
            q1 = val_lens.quantile(0.25)
            q3 = val_lens.quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))
            test_series = val_lens < upper_limit
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"Some values were unusually long. Any values with length greater than {upper_limit} characters "
                 f"were flagged, given the 25th percentile of string length is {q1} and the 75th is {q3}"),
                allow_patterns=False
            )


    def _generate_few_chars(self):
        """
        Patterns without exceptions: None: 'few_chars all' consistently has 100 to 200 characters, but this test is
            not flagged as a pattern.
        Patterns with exception: 'few_chars most' is similar, but has 1 exception with only 2 characters.
        """
        self._add_synthetic_column('few_chars all',
            [''.join(['a' for _ in range(np.random.randint(100, 200))]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('few_chars most',
            [''.join(['a' for _ in range(np.random.randint(100, 200))]) for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'few_chars most'] = 'aa'


    def _check_few_chars(self, test_id):
        """
        For this, we use the inter-decile range instead of inter-quartile, as IQR often sets the lower limit below zero
        which is impossible here, while small positive values may nevertheless be unusually small.
        """

        for col_name in self.string_cols:
            val_lens = self.orig_df[col_name].fillna("").astype(str).str.len()
            val_lens_no_null = self.orig_df[col_name].dropna().astype(str).str.len()
            d1 = val_lens_no_null.quantile(0.1)
            d9 = val_lens_no_null.quantile(0.9)
            lower_limit = d1 - (self.idr_limit * (d9 - d1))
            test_series = val_lens > lower_limit
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"Some values were unusually short. Any values with length less than {lower_limit} characters "
                 f"were flagged, given the 10th percentile of string length is {d1} and the 90th is {d9}. "
                 f"The coefficient is set at {self.idr_limit}"),
                allow_patterns=False
            )


    def _generate_position_non_alphanumeric(self):
        """
        Patterns without exceptions: 'posn-special all' consistently has a '-' in the 3rd position
        Patterns with exception: 'posn-special most' is similar, but with 1 exception
        """
        self._add_synthetic_column('posn-special rand',
            [[''.join(random.choice(list(letters) + ['-']) for _ in range(random.randint(1, 10)))][0]
             for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('posn-special all',
            [['ab-' + ''.join(random.choice(letters) for _ in range(5))][0]
             for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('posn-special most',
            [['ab-' + ''.join(random.choice(letters) for _ in range(5))][0]
             for _ in range(self.num_synth_rows-1)] + ['ab$43-2'])


    def _check_position_non_alphanumeric(self, test_id):
        """
        Check if the position of the non-alphanumeric columns is consistent. For example, a column may have values
        such as 'AB-123', 'WT-15445' etc. Here the hyphen in consistently in the 2nd (zero-based) position. We check,
        where the strings are not a consistent length, both the position from the front and from the back of the
        strings.

        The test COMMON_SPECIAL_CHARS checks if there are any special characters present in most values. This test
        checks, for any special characters that are in all values, if they are in the same position.
        """

        for col_name in self.string_cols:
            # todo: Skip columns that have only one character for all values

            # Skip columns which contain only alphanumeric characters
            if self.orig_df[col_name].astype(str).str.isalnum().tolist().count(False) == 0:
                continue

            # Get the set of special characters in each value. Do not include space characters.
            special_chars_list = self.orig_df[col_name].astype(str).apply(get_non_alphanumeric)

            # Remove any empty lists
            special_chars_list = [x for x in special_chars_list if len(x)]

            # Skip columns where it is not the case that all values have special characters
            if (self.orig_df[col_name].isna().sum() + len(special_chars_list)) < self.num_rows:
                continue

            # Get the unique set of special characters
            special_chars_list = list({item for sublist in special_chars_list for item in sublist})
            if ' ' in special_chars_list:
                special_chars_list.remove(' ')

            # Examine each special character and determine if it is in most values
            common_special_chars_list = []
            for c in special_chars_list:
                if (self.orig_df[col_name].isna().sum() + \
                        self.orig_df[col_name].fillna("").astype(str).str.contains(c, regex=False).tolist().count(True)) > \
                        (self.num_rows - self.freq_contamination_level):
                    common_special_chars_list.append(c)

            if len(common_special_chars_list) == 0:
                continue

            # Identify the common special characters which appear in a consistent location
            for c in common_special_chars_list:
                # Get the position of the character from the beginning of the strings
                list_1 = pd.Series(self.orig_df[col_name].astype(str).str.find(c)).replace(-1, np.nan)
                # Get the position of the character from the end of the strings
                list_2 = pd.Series([x - y if y >= 0 else -1 for x, y in
                                    zip(self.orig_df[col_name].fillna("").astype(str).str.len() ,
                                        self.orig_df[col_name].fillna("").astype(str).str.rfind(c))]).replace(-1, np.nan)
                vc1 = list_1.value_counts()
                vc2 = list_2.value_counts()
                if (self.orig_df[col_name].isna().sum() + vc1.iloc[0]) >= (self.num_rows - self.freq_contamination_level):
                    common_posn = int(vc1.index[0])
                    common_posn_str = str(common_posn + 1)
                    test_series = self.orig_df[col_name].astype(str).str.find(c) == common_posn
                elif (self.orig_df[col_name].isna().sum() + vc2.iloc[0]) >= (self.num_rows - self.freq_contamination_level):
                    common_posn = int(vc2.index[0])
                    common_posn_str = f"{common_posn} from the end"
                    test_series = (self.orig_df[col_name].astype(str).str.len() -
                                   self.orig_df[col_name].astype(str).str.rfind(c)) == common_posn
                else:
                    continue
                test_series = test_series | self.orig_df[col_name].isna()

                pattern_str = f"The column contains values consistently with '{c}' in position {common_posn_str}"
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    pattern_str)

                if test_series.tolist().count(False) < self.freq_contamination_level:
                    break


    def _generate_chars_pattern(self):
        """
        Patterns without exceptions: 'chars_pattern all' consistently contains values of the pattern: A@9.9-A
        Patterns with exception: 'chars_pattern most' consistently contains values of the pattern: A@9.9-A, with 1
            exception.
        Not flagged: 'chars_pattern rand' contains values with no pattern.
        """

        self._add_synthetic_column('chars_pattern rand',
            [[''.join(random.choice(list(letters) + ['-']) for _ in range(random.randint(1, 10)))][0]
             for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('chars_pattern all',
            [''.join([random.choice(list(letters)), '@', '5', '.', '44','-',random.choice(list(letters))])
             for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('chars_pattern most', self.synth_df['chars_pattern all'])
        self.synth_df.loc[999, 'chars_pattern most'] = 'g@-8-.42'


    def _check_chars_pattern(self, test_id):
        """
        This test checks for string values that follow a specific pattern with respect to the non-alphanumeric
        characters used, for example email addresses, telephone numbers, IDs, and other values with a specific,
        consistent pattern.
        """

        def get_str_format(x):
            if x is None:
                return ""
            x = x.strip()  # Remove leading and trailing spaces
            new_str = ' '  # Add a space to simplify tests for the previous character. Removed later.
            for c in x:
                if c.isdigit() or c.isalpha():
                    if new_str[-1] != 'X':
                        new_str += 'X'
                else:
                    new_str += c
            return new_str[1:]  # Strip the space added at the start

        for col_name in self.string_cols:
            # Skip columns that have only one character for all values
            avg_num_chars = statistics.median(self.orig_df[col_name].astype(str).str.len())
            if avg_num_chars <= 1:
                continue

            # Skip columns that are almost entirely Null. If there are even a small number of non-null values, though,
            # and they consistently follow a pattern, we flag that pattern.
            if self.orig_df[col_name].isna().sum() > (self.num_rows - self.freq_contamination_level):
                continue

            # Skip columns which contain only alphanumeric characters
            if self.orig_df[col_name].astype(str).str.isalnum().tolist().count(False) == 0:
                continue

            # First test on a sample
            patterns_arr = self.sample_df[col_name].astype(str).apply(get_str_format)
            if patterns_arr.nunique() > 1:
                continue

            # Skip where the main pattern is simply 'X', meaning all alpha-numeric characters
            vc = patterns_arr.value_counts()
            if vc.index[0] == 'X':
                continue

            patterns_arr = self.orig_df[col_name].astype(str).apply(get_str_format)
            vc = patterns_arr.value_counts()
            if len(vc) > self.freq_contamination_level:
                continue

            test_series = [y or (x == vc.index[0]) for x, y in zip(patterns_arr, self.orig_df[col_name].isna())]
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f'Column "{col_name}" consistently contains values with the pattern "{vc.index[0]}", where "X" '
                 f'represents alpha-numeric characters'),
            )


    def _generate_uppercase(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('uppercase rand',
            [[''.join(random.choice(string.ascii_letters) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('uppercase all',  [[''.join(random.choice(string.ascii_uppercase) for
                                        _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('uppercase most',
            [[''.join(random.choice(string.ascii_uppercase)
                      for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['YYa'])


    def _check_uppercase(self, test_id):
        for col_name in self.string_cols:
            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.isupper()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.isupper()
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column is consistently uppercase")


    def _generate_lowercase(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('lowercase rand',
            [[''.join(random.choice(string.ascii_letters)
            for _ in range(random.randint(1, 10)))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('lowercase all',
            [[''.join(random.choice(string.ascii_lowercase)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('lowercase most',
            [[''.join(random.choice(string.ascii_lowercase)
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['YYa'])


    def _check_lowercase(self, test_id):
        for col_name in self.string_cols:
            # Test on a sample of the full data
            sample_series = self.sample_df[col_name].astype(str).str.islower()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test of the full data
            test_series = self.orig_df[col_name].astype(str).str.islower()
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column is consistently lowercase")


    def _generate_characters_used(self):
        """
        Patterns without exceptions: 'char_counts_all' consistently has values using only a and/or b and/or c
        Patterns with exception: 'char_counts_most' is similar, but with 1 exception.
        Not Flagged: 'char_counts_rand' contains random values
        """
        self._add_synthetic_column('char_counts_rand',
            [[''.join(random.choice(string.ascii_letters)
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('char_counts_all',
            [[''.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('char_counts_most',
            [[''.join(random.choice(['a', 'b', 'c'])
            for _ in range(5))][0] for _ in range(self.num_synth_rows-1)] + ['Ybbca'])


    def _check_characters_used(self, test_id):
        """
        This test may be viewed as the opposite of COMMON_CHARS. Where COMMON_CHARS checks if there are any characters
        that appear in all rows, and flags values that do not contain any of these values, this test identifies a set of
        characters such that the values contain only those characters, and flags any values that contain other
        characters.

        This tests only string columns that have short strings, but not single characters, and many unique values.

        Handling null values: patterns are not considered violated if any cells contain null values.
        """

        for col_name in self.string_cols:

            # Only execute this test on columns that have consistently short strings, such as codes or IDs.
            if self.orig_df[col_name].astype(str).str.len().max() > 5:
                continue

            # Skip columns that consistently contain only 1 character
            if self.orig_df[col_name].astype(str).str.len().max() <= 1:
                continue

            # Skip columns that have few unique values
            if self.orig_df[col_name].nunique() < math.sqrt(self.num_rows):
                continue

            full_text = ''.join(list(self.orig_df[col_name].astype(str)))
            chars_used = list(set(full_text))
            rare_chars = {}
            common_chars = {}
            for c in chars_used:
                count_c = full_text.count(c)
                if count_c == 0:
                    continue
                if count_c < len(full_text) / 1000:
                    rare_chars[c] = count_c
                else:
                    common_chars[c] = count_c

            if (0 < len(common_chars) < 10) and (len(rare_chars) < 20):
                test_series = [True] * self.num_rows
                for rare_char in rare_chars:
                    test_series = test_series & ~self.orig_df[col_name].astype(str).str.contains(rare_char)
                test_series = test_series | self.orig_df[col_name].isna()

                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    (f'The column "{col_name}" consistently contains values containing only the characters: '
                     f'{list(common_chars.keys())}'),
                    f' Rare characters found: {list(rare_chars.keys())}'
                )


    def _generate_first_word(self):
        """
        Patterns without exceptions: 'first_word all' consistently begins with the word 'abc'
        Patterns with exception: 'first_word most' consistently begins with the word 'abc', with one exception.
        """
        self._add_synthetic_column('first_word rand',
            [''.join(np.random.choice(['a', 'b '], 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_word all',
            ["abc-" + np.random.choice(list(string.ascii_letters)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('first_word most',
            self.synth_df['first_word all'])
        self.synth_df.loc[999, 'first_word most'] = "wyxz"


    def _check_first_word(self, test_id):
        """
        Here we define words as strings separated by whitespace or common separator characters such as hyphens.
        This test skips columns that are primarily one word or have few unique values.
        """
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)

            # Skip columns that are primarily 1 word
            word_arr = col_vals.str.split()
            num_words_arr = [len(x) for x, y in zip(word_arr, self.orig_df[col_name].isna()) if not y]
            if pd.Series(num_words_arr).quantile(0.5) <= 1:
                continue

            # Skip columns that have few unique values
            if self.orig_df[col_name].nunique() < math.sqrt(self.num_rows):
                continue

            first_words = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            vc = pd.Series(first_words).value_counts()
            common_values = []
            for v in pd.Series(first_words).unique():
                if vc[v] > self.freq_contamination_level:
                    common_values.append(v)
            if len(common_values) > self.freq_contamination_level:
                continue
            test_series = [x in common_values for x in first_words] | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                np.array(test_series),
                f"The column consistently begins with one of {str(common_values)[1:-1]}",
                display_info={'counts': vc}
            )


    def _generate_last_word(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('last_word rand',
                                    [''.join(np.random.choice(['a', 'b '], 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('last_word all',
                                    [np.random.choice(list(string.ascii_letters)) + "-abc" for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('last_word most', self.synth_df['last_word all'])
        self.synth_df.loc[999, 'last_word most'] = "wyxz"


    def _check_last_word(self, test_id):
        """
        Here we define words as strings separated by whitespace or common separator characters such as hyphens.
        """
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)

            # Skip columns that are primarily 1 word
            word_arr = col_vals.str.split()
            num_words_arr = [len(x) for x, y in zip(word_arr, self.orig_df[col_name].isna()) if not y]
            if pd.Series(num_words_arr).quantile(0.5) <= 1:
                continue

            # Skip columns that have few unique values
            if self.orig_df[col_name].nunique() < math.sqrt(self.num_rows):
                continue

            last_words = [x[-1] if len(x) > 0 else "" for x in col_vals.str.split()]
            vc = pd.Series(last_words).value_counts()
            common_values = []
            for v in pd.Series(last_words).unique():
                if vc[v] > self.freq_contamination_level:
                    common_values.append(v)
            if len(common_values) > self.freq_contamination_level:
                continue
            test_series = [x in common_values for x in last_words] | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                np.array(test_series),
                f"The column consistently ends with one of {str(common_values)[1:-1]}")


    def _generate_num_words(self):
        """
        Patterns without exceptions: None. allow_patterns is set False.
        Patterns with exception: 'num_words most' consistently has 1 to 5 words, with one exception with many more
            words.
        """
        self._add_synthetic_column('num_words all',
            [[''.join(random.choice(string.ascii_letters[:10] + ' ') for _ in range(np.random.randint(1, 20)))][0]
                for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('num_words most', self.synth_df['num_words all'])
        self.synth_df.loc[999, 'num_words most'] = 'a b c a b c a b c a b c a b c a b c a b c a b c a b c a b c '


    def _check_num_words(self, test_id):
        """
        This flags values with unusually many words. Words are defined here as string separated by whitespace or any
        non-alphanumeric characters, so may count tokens within words.
        """

        word_counts_dict = self.get_word_counts_dict()

        for col_name in self.string_cols:
            word_counts_arr = pd.Series(word_counts_dict[col_name])
            q1 = word_counts_arr.quantile(0.25)
            q3 = word_counts_arr.quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))
            test_series = [x < upper_limit for x in word_counts_arr] | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f'The values in column "{col_name}" have word count with 25th percentile {q1} words and 75th '
                 f'percentile {q3} words'),
                f' Flagging any values with over {math.floor(upper_limit)} words.',
                allow_patterns=False
            )


    def _generate_longest_words(self):
        """
        Patterns without exceptions: None. This test does not generate patterns.
        Patterns with exception: 'long_word most' has one unusually long word.
        """
        self._add_synthetic_column('long_word all',
            [[''.join(random.choice(string.ascii_letters) for _ in range(np.random.randint(1, 12)))][0]
            for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('long_word most',
            [[''.join(random.choice(string.ascii_letters) for _ in range(np.random.randint(1, 12)))][0]
            for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'long_word most'] = 'ad dfddfdabdkdjsdlksjklsdjkldjklsdsjkldjklsdjklsd yy'


    def _check_longest_words(self, test_id):
        """
        This checks for invalid words within string, possibly due to missing white space between the words or otherwise
        invalid text content.
        """
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            word_arr = col_vals.str.split()
            if not any(len(x) for x in word_arr):
                continue
            word_lens_arr = [[len(w) for w in x] for x in word_arr]
            flat_word_lens_arr = pd.Series(np.concatenate(word_lens_arr).flat)
            q1 = flat_word_lens_arr.quantile(0.25)
            q3 = flat_word_lens_arr.quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * 1.5 * (q3 - q1))  # We use more than the normal limit
            max_word_lens = [max(x) if len(x) > 0 else 0 for x in word_lens_arr]
            test_series = [x < upper_limit for x in max_word_lens]
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f'The words in column "{col_name}" have lengths with 25th percentile {q1} characters and 75th '
                 f'percentile {q3} characters'),
                f' Flagging any values with words over {math.floor(upper_limit)} characters.',
                allow_patterns=False
            )


    def _generate_words_used(self):
            """
            Patterns without exceptions: 'words_used all' consistently has the word 'giblet'
            Patterns with exception: 'words_used most' consistently has the word 'giblet', with 1 exception
            """
            self._add_synthetic_column('words_used rand_a',
                np.random.choice(["aa", "bb", "cc", "dd"], self.num_synth_rows))
            self._add_synthetic_column('words_used rand_b',
                [[''.join(random.choice(string.ascii_letters) for _ in range(5))][0] for _ in range(self.num_synth_rows)])
            self._add_synthetic_column('words_used all',
                "giblet " + self.synth_df['words_used rand_a'] + "-" + self.synth_df['words_used rand_b'])
            self._add_synthetic_column('words_used most', self.synth_df['words_used all'])
            self.synth_df.loc[999, 'words_used most'] = "wyxz"


    def _check_words_used(self, test_id):
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            words_arr = col_vals.str.split()
            if pd.Series([len(x) for x in words_arr]).median() < 2:
                continue
            flat_words_arr = list(np.concatenate(words_arr).flat)
            vc = pd.Series(flat_words_arr).value_counts(ascending=False)
            common_words_arr = []
            for v in vc.index:
                if (self.orig_df[col_name].isna().sum() + vc[v]) >= (self.num_rows - self.freq_contamination_level):
                    common_words_arr.append(v)
                else:
                    break
            if len(common_words_arr) == 0:
                continue
            test_series = [True] * self.num_rows
            for c in common_words_arr:
                test_series = test_series & np.array([c in words_arr[i] for i in range(self.num_rows)])
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in column "{col_name}" consistently contain the word(s) {str(common_words_arr)[1:-1]}'
            )


    def _generate_rare_words(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'rare_words most' consitently contains non-rare words, with one exception
        """
        word_list = [''.join([random.choice(string.ascii_letters) for x in range(10)]) for _ in range(50)]
        self._add_synthetic_column('rare_words all',
            [' '.join([word_list[np.random.randint(0, len(word_list))]] * 4) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('rare_words most',
            [' '.join([word_list[np.random.randint(0, len(word_list))]] * 4) for _ in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'rare_words most'] = self.synth_df.loc[999, 'rare_words most'] + " xxxxxxx"


    def _check_rare_words(self, test_id):
        words_list_dict = self.get_words_list_dict()

        for col_name in self.string_cols:
            words_arr = words_list_dict[col_name]
            num_words_arr = [len(x) for x in words_arr]
            if pd.Series(num_words_arr).quantile(0.5) <= 1:
                continue

            flat_words_arr = list(np.concatenate(words_arr).flat)
            vc = pd.Series(flat_words_arr).value_counts(ascending=True)
            rare_words_arr = []
            for v in vc.index:
                if vc[v] < self.freq_contamination_level:
                    rare_words_arr.append(v)
                else:
                    break
            if len(rare_words_arr) == 0:
                continue
            if len(rare_words_arr) > self.freq_contamination_level:
                continue
            test_series = [len(set(x).intersection(set(rare_words_arr))) == 0 for x in words_arr]
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in column "{col_name}" consistently use only words common to this column',
                f' Flagging values containing the rare words: {array_to_str(rare_words_arr)}'
            )


    def _generate_grouped_strings(self):
        """
        Patterns without exceptions: 'grouped_str all' has the 3 values in this column in 3 consistent groups
        Patterns with exception: 'grouped_str most' is similar, with 1 exception
        """
        self._add_synthetic_column('grouped_str rand', [random.choice(['aaa', 'bbbb', 'cccc'])
                                                         for _ in range(self.num_synth_rows)])
        group_a_len = group_b_len = self.num_synth_rows // 3
        group_c_len = self.num_synth_rows - (group_a_len + group_b_len)
        self._add_synthetic_column('grouped_str all',  ['aaa'] * group_a_len +
                                                        ['bbb'] * group_b_len +
                                                        ['ccc'] * group_c_len)
        self._add_synthetic_column('grouped_str most', ['aaa'] * group_a_len +
                                                        ['bbb'] * group_b_len +
                                                        ['ccc'] * (group_c_len - 1) + ['aaa'])


    def _check_grouped_strings(self, test_id):
        """
        This is similar to GROUPED_STRINGS_BY_NUMERIC, but uses the row order the data is received in.
        """
        for col_name in self.string_cols + self.binary_cols:
            if self.orig_df[col_name].nunique() > math.sqrt(self.num_rows):
                continue
            df = self.orig_df.copy()
            col_series = df[col_name]
            self._check_grouped_strings_column(test_id, None, col_name, col_series)

    ##################################################################################################################
    # Data consistency checks for pairs of non-numeric columns
    ##################################################################################################################


    def _generate_a_implies_b(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """


    def _check_a_implies_b(self, test_id):
        pass


    def _generate_rare_pairs(self):
        """
        Patterns without exceptions: None: rand_c and rand_d columns appear with all combinations of values with the
            other columns, but this test does not flag patterns, only exceptions.
        Patterns with exception: the columns rand_a and rand_b have only 1 instance of 'b' and 'b'
        """
        vals = [
            ['a', 'a', 'x', 'x'],
            ['a', 'b', 'x', 'y'],
            ['b', 'a', 'y', 'x'],
            ['a', 'b', 'y', 'y'],
            ['a', 'b', 'x', 'y']]
        data = np.array([vals[np.random.choice(range(len(vals)))] for _ in range(self.num_synth_rows-1)])
        data = np.vstack([data, ['b', 'b', 'x', 'y']])
        self._add_synthetic_column('rare_pair rand_a', data[:, 0])
        self._add_synthetic_column('rare_pair rand_b', data[:, 1])
        self._add_synthetic_column('rare_pair rand_c', data[:, 2])
        self._add_synthetic_column('rare_pair rand_d', data[:, 3])


    def _check_rare_pairs(self, test_id):
        """
        This flags pairs of values, where neither value is by itself rare, but the combination is. This is performed
        on each pair of string columns. The test RARE_COMBINATION covers pairs of numeric columns. For pairs of
        columns with one string and one numeric, there are the VERY_LARGE_GIVEN_VALUE and VERY_SMALL_GIVEN_VALUE tests.
        """
        num_pairs, pairs = self._get_binary_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pair of binary columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for col_name_1, col_name_2 in pairs:
            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if self.orig_df[col_name_1].nunique() > math.sqrt(self.num_rows) or \
               self.orig_df[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue
            vc_1 = self.orig_df[col_name_1].value_counts()
            vc_2 = self.orig_df[col_name_2].value_counts()
            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < math.pow(self.freq_contamination_level, 2):
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < math.pow(self.freq_contamination_level, 2):
                        continue
                    sub_df = self.orig_df[(self.orig_df[col_name_1] == v1) &
                                          (self.orig_df[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue
                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    expected_count = ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows

                    # Check the count is both low and lower than expected.
                    if count_pair < self.freq_contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append((v1, v2))
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                "Rare combinations of values were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )


    def _generate_rare_pair_first_char(self):
        """
        Patterns without exceptions: None
        Patterns with exception: The first column starts with characters a, e, i, and the 2nd with b,d,f,h,j,l. a is
            frequently matched with b and d, but in only once with j
        """
        common_vals = [
            ["ax-xxx1", "bx-xxxxx1"],
            ["ax-xxx2", "fx-xxxxx2"],
            ["ex-xxx3", "fx-xxxxx3"],
            ["ex-xxx4", "fx-xxxxx4"],
            ["ix-xxx5", "jx-xxxxx5"],
            ["ix-xxx6", "bx-xxxxx6"]
        ]
        rare_vals = [["ax-xxxx", "jx-xxxxxx"]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self._add_synthetic_column('rare_pair_first_char all_a', data[:, 0])
        self._add_synthetic_column('rare_pair_first_char all_b', data[:, 1])


    def _check_rare_pair_first_char(self, test_id):
        """
        This skips columns where the strings tend to be longer, as this is concerned with values that resemble ids
        or codes. It also skips columns that tend to contain single characters. The test is not typically useful for
        columns which are not codes, and where the first character within the code does not have meaning. This does not
        flag pairs where either of the first characters is, by itself, rare, as the combination of it with any other
        value will necessarily also be rare.
        """

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        # The minimum number of times a first character must appear within it's column to check pairs including it
        freq_limit = math.pow(self.freq_contamination_level, 2)

        # The maximum number of unique values in a column to consider the column
        unique_vals_limit = math.sqrt(self.num_rows)

        first_chars_dict = {}   # The first character of each value with each column
        value_counts_dict = {}  # Counts of each first character within each column
        char_count_dict = {}    # The median string length of the values in each column
        for col_name in self.string_cols:
            first_chars_dict[col_name] = pd.Series(self.orig_df[col_name].astype(str).str[:1])
            value_counts_dict[col_name] = first_chars_dict[col_name].value_counts()
            val_lens = self.orig_df[col_name].astype(str).str.len()
            median_len = val_lens.quantile(0.5)
            char_count_dict[col_name] = median_len

        nunique_dict = self.get_nunique_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 1000 == 0:
                print(f"  Examining pair: {pair_idx:,} of {len(pairs):,} pairs of binary columns)")

            vc_1 = value_counts_dict[col_name_1]
            vc_2 = value_counts_dict[col_name_2]

            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if (len(vc_1) > unique_vals_limit) or (len(vc_2) > unique_vals_limit):
                continue

            # Skip columns that have as many unique values as unique first characters
            if len(value_counts_dict[col_name_1]) >= (nunique_dict[col_name_1] * 0.75):
                continue
            if len(value_counts_dict[col_name_2]) >= (nunique_dict[col_name_2] * 0.75):
                continue

            # Skip columns that have values that tend to be a single character
            if (char_count_dict[col_name_1] < 2) or (char_count_dict[col_name_2] < 2):
                continue

            # Skip columns that have values that tend to be long
            if (char_count_dict[col_name_1] > 10) or (char_count_dict[col_name_2] > 10):
                continue

            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < freq_limit:
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < freq_limit:
                        continue
                    sub_df = self.orig_df[(first_chars_dict[col_name_1] == v1) &
                                          (first_chars_dict[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue

                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    # This is ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows, but simplified
                    # here for efficiency.
                    expected_count = (vc_1[v1] / self.num_rows) * vc_2[v2]

                    # Check the count is both low and lower than expected.
                    if count_pair < self.freq_contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append((v1, v2))
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                "Rare combinations of first characters were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )


    def _generate_rare_pair_first_word(self):
        """
        Patterns without exceptions: None. This test does not flag patterns, only exceptions
        Patterns with exception: 'rare_pair_first_word all_a' and 'rare_pair_first_word all_b' have 1 rare combination.
        """
        common_vals = [
            ["ax-xxx1", "b1-xxxxx1"],
            ["ax-xxx2", "b1-xxxxx2"],
            ["ax-xxx3", "b2-xxxxx3"],
            ["ax-xxx4", "b2-xxxxx4"],
            ["ex-xxx5", "b3-xxxxx5"],
            ["ex-xxx6", "b3-xxxxx6"],
            ["ix-xxx7", "b4-xxxxx7"],
            ["ix-xxx8", "b4-xxxxx8"],
            ["ix-xxx9", "b4-xxxxx9"],
            ["ix-xxx10", "b6-xxxxx10"]
        ]
        rare_vals = [["ax-xxxx", "b4-xxxxxx"]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self._add_synthetic_column('rare_pair_first_word all_a', data[:, 0])
        self._add_synthetic_column('rare_pair_first_word all_b', data[:, 1])


    def _check_rare_pair_first_word(self, test_id):
        first_words_dict = {}
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            first_words_dict[col_name] = pd.Series([x[0] if len(x) >0 else "" for x in col_vals.str.split()])

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for _pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            # Skip columns if the first words are as unique as teh values in the column. In this case, the first words
            # have no real meaning on their own.
            if first_words_dict[col_name_1].nunique() >= (self.orig_df[col_name_1].nunique() / 2):
                continue
            if first_words_dict[col_name_2].nunique() >= (self.orig_df[col_name_2].nunique() / 2):
                continue

            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if first_words_dict[col_name_1].nunique() > math.sqrt(self.num_rows) or \
                    first_words_dict[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue
            vc_1 = first_words_dict[col_name_1].value_counts()
            vc_2 = first_words_dict[col_name_2].value_counts()
            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < math.pow(self.freq_contamination_level, 2):
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < math.pow(self.freq_contamination_level, 2):
                        continue
                    sub_df = self.orig_df[(first_words_dict[col_name_1] == v1) &
                                          (first_words_dict[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue
                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    expected_count = ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows

                    # Check the count is both low and lower than expected.
                    if count_pair < self.freq_contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append((v1, v2))
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                "Rare combinations of first words were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )


    def _generate_rare_pair_first_word_val(self):
        """
        Patterns without exceptions: None
        Patterns with exception: The combination of values in 'rare_pair_first_word_val all_a' and
        'rare_pair_first_word_val all_b' are consistently common, with 1 exception
        """
        common_vals = [
            ["ax-xxx1", "A"],
            ["ax-xxx2", "A"],
            ["ax-xxx3", "A"],
            ["ax-xxx4", "A"],
            ["ex-xxx5", "A"],
            ["ex-xxx6", "A"],
            ["ex-xxx7", "B"],
            ["ex-xxx8", "A"],
            ["ix-xxx9", "B"],
            ["ix-xxx10", "A"],
            ["ix-xxx11", "B"],
            ["ix-xxx12", "C"],
            ["ix-xxx13", "D"]
        ]
        rare_vals = [["ax-xxxx", "B"]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self._add_synthetic_column('rare_pair_first_word_val all_a', data[:, 0])
        self._add_synthetic_column('rare_pair_first_word_val all_b', data[:, 1])


    def _check_rare_pair_first_word_val(self, test_id):
        first_words_dict = {}
        word_count_dict = {}  # todo: call self.get_word_counts_dict()
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].astype(str).apply(replace_special_with_space)
            first_words_dict[col_name] = pd.Series([x[0] if len(x) > 0 else "" for x in col_vals.str.split()])
            word_count_dict[col_name] = pd.Series([len(x) for x in col_vals.str.split()])

        num_pairs, pairs = self._get_string_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 1000 == 0:
                print(f"  Examining pair number {pair_idx:,} of {len(pairs):,} pairs of string columns")

            # Skip col_name_1 if it's mostly single words
            non_null_vals = [x for x in word_count_dict[col_name_1] if x > 0]
            if len(non_null_vals) == 0:
                continue
            if statistics.median(non_null_vals) <= 1:
                continue

            # Skip col_name_1 if the first words are as unique as the values in the column. In this case, the first
            # words have no real meaning on their own.
            if first_words_dict[col_name_1].nunique() >= (self.orig_df[col_name_1].nunique() / 2):
                continue

            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if first_words_dict[col_name_1].nunique() > math.sqrt(self.num_rows) or \
                    self.orig_df[col_name_2].nunique() > math.sqrt(self.num_rows):
                continue
            vc_1 = first_words_dict[col_name_1].value_counts()
            vc_2 = self.orig_df[col_name_2].value_counts()
            rare_pairs_arr = []
            test_series = np.array([True] * self.num_rows)
            for v1 in vc_1.index:
                # Skip empty / Null values
                if v1 == "":
                    continue

                # Skip values that are themselves rare enough that rare combinations including it would be expected
                if vc_1[v1] < math.pow(self.freq_contamination_level, 2):
                    continue
                for v2 in vc_2.index:
                    if vc_2[v2] < math.pow(self.freq_contamination_level, 2):
                        continue
                    sub_df = self.orig_df[(first_words_dict[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]
                    count_pair = len(sub_df)
                    if count_pair == 0:
                        continue
                    # The expected_count is the number we would expect given the marginal frequencies of the two values
                    expected_count = ((vc_1[v1] / self.num_rows) * (vc_2[v2] / self.num_rows)) * self.num_rows

                    # Check the count is both low and lower than expected.
                    if count_pair < self.freq_contamination_level and count_pair < (expected_count * 0.5):
                        rare_pairs_arr.append([v1, v2])
                        test_series[sub_df.index] = False

            if len(rare_pairs_arr) == 0:
                continue
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f"Rare combinations of the first word in {col_name_1} and the value in {col_name_2} were found",
                f"{rare_pairs_arr}",
                allow_patterns=False
            )


    def _generate_similar_chars(self):
        """
        Patterns without exceptions: The columns "sim_chars rand" and "sim_chars all" consistently have a similar
            characters as each other
        Patterns with exception: "sim_chars rand" and "sim_chars most" consistently have a similar characters as each
            other, with 1 exception
        Not Flagged: "sim_chars all" and "sim_chars most" are almost identical, so not flagged by this test.
        """
        self._add_synthetic_column('sim_chars rand',
            [[''.join(random.choice(alphanumeric) for _ in range(10))][0] for i in range(self.num_synth_rows)])
        self._add_synthetic_column('sim_chars all', self.synth_df['sim_chars rand'] + "a")
        self._add_synthetic_column('sim_chars most', self.synth_df['sim_chars all'])
        self.synth_df.loc[999, 'sim_chars most'] = "abcd"


    def _check_similar_chars(self, test_id):
        """
        This is intended for stings that appear to be ids, codes or other such strings, so this test only covers
        columns that contain exactly one word in each value.
        """

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for col_idx_1 in range(len(self.string_cols)-1):
            col_name_1 = self.string_cols[col_idx_1]
            col_vals = self.orig_df[col_name_1].astype(str).apply(replace_special_with_space)
            # todo: call self.get_word_counts_dict()
            word_counts = [0 if x is None else len(x) for x in col_vals.str.split()]
            if max(word_counts) > 1:
                continue
            chars_list_1 = [[""] if x is None else list(x) for x in col_vals]
            for col_idx_2 in range(col_idx_1 + 1, len(self.string_cols)):
                col_name_2 = self.string_cols[col_idx_2]

                # Skip if the two columns are largely the same
                if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                    continue

                col_vals = self.orig_df[col_name_2].apply(replace_special_with_space)
                word_counts = [0 if x is None else len(x) for x in col_vals.str.split()]
                if max(word_counts) > 1:
                    continue
                chars_list_2 = [[""] if x is None else list(x) for x in col_vals]
                test_series = [(len(set(x).union(set(y))) > 0) and
                                   (len(set(x).intersection(set(y))) / len(set(x).union(set(y))) > 0.8)
                               for x, y in zip(chars_list_1, chars_list_2)]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    np.array(test_series),
                    (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar characters '
                     f'as each other')
                )


    def _generate_similar_num_chars(self):
        """
        Patterns without exceptions: "sim_num_chars rand" AND "sim_num_chars all" consistently have a similar number of
            characters
        Patterns with exception: "sim_num_chars rand" AND "sim_num_chars most" consistently have a similar number of
            characters, with 1 exception.
        Not Flagged: "sim_num_chars all" AND "sim_num_chars most" are almost identical, so not flagged by this test.
        """
        num_chars_arr = [np.random.randint(0, 20) for _ in range(self.num_synth_rows)]
        self._add_synthetic_column('sim_num_chars rand',
            [[''.join(random.choice(alphanumeric) for _ in range(num_chars_arr[i]))][0] for i in range(self.num_synth_rows)])
        self._add_synthetic_column('sim_num_chars all',
            [[''.join(random.choice(alphanumeric) for _ in range(num_chars_arr[i]))][0] for i in range(self.num_synth_rows)])
        self._add_synthetic_column('sim_num_chars most', self.synth_df['sim_num_chars all'])
        self.synth_df.loc[999, 'sim_num_chars most'] = "abcdefghijklmnopqrstuv"


    def _check_similar_num_chars(self, test_id):

        # Get the character length of each value in the string columns, and their variation in length
        char_len_dict = {}
        col_std_dev_len_dict = {}
        for col_name in self.string_cols:
            char_len_dict[col_name] = self.orig_df[col_name].astype(str).str.len().replace(0, 1)
            col_std_dev_len_dict[col_name] = char_len_dict[col_name].std()

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for _pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if (col_std_dev_len_dict[col_name_1] < 2.0) or (col_std_dev_len_dict[col_name_2] < 2.0):
                continue

            if self.orig_df[col_name_1].apply(is_missing).sum() > (self.num_rows * 0.75):
                continue
            if self.orig_df[col_name_2].apply(is_missing).sum() > (self.num_rows * 0.75):
                continue

            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            test_series = [0.9 < x/y < 1.1 for x, y in zip(char_len_dict[col_name_1], char_len_dict[col_name_2])]
            test_series = test_series | (self.orig_df[col_name_1].isna() & self.orig_df[col_name_2].isna())
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                np.array(test_series),
                (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar number of characters '
                 f'as each other')
            )


    def _generate_similar_words(self):
        """
        Patterns without exceptions: The columns "sim_words all" and "sim_words rand" consistently have a similar set of
            words as each other
        Patterns with exception: The columns "sim_words most" and "sim_words rand" consistently have a similar set of
            words as each other, with exceptions
        Not Flagged: "sim_words all" and "sim_words rand" consistently have a similar set of words, but the columns are
            almost identical, so not flagged by this test.
        """
        self._add_synthetic_column('sim_words rand',
            [' '.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(5, 8)))
            for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim_words all', [x + " xx" for x in self.synth_df['sim_words rand']])
        self._add_synthetic_column('sim_words most', [x + " xx" for x in self.synth_df['sim_words rand']])
        self.synth_df.loc[999, 'sim_words most'] = 'abcdef'


    def _check_similar_words(self, test_id):
        # todo: we can probably put this in process_data(). this is also used in __check_similar_num_words()
        word_counts_dict = {} # todo: call self.get_word_counts_dict()
        word_counts_medians = {}
        for col_name in self.string_cols:
            word_counts_dict[col_name] = pd.Series([len(x.split()) if (not is_missing(x)) else 0 for x in self.orig_df[col_name]])
            word_counts_not_null = pd.Series([len(x.split()) for x in self.orig_df[col_name] if (not is_missing(x))])
            word_counts_medians[col_name] = word_counts_not_null.median()

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for _pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            word_counts_medians_1 = word_counts_medians[col_name_1]
            word_counts_medians_2 = word_counts_medians[col_name_2]
            if word_counts_medians_1 < 3 or word_counts_medians_2 < 3:
                continue
            if word_counts_medians_2 < (word_counts_medians_1 / 2):
                continue
            if word_counts_medians_2 > (word_counts_medians_1 * 2):
                continue

            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            word_lists_1 = pd.Series([x.split() if (not is_missing(x)) else [] for x in self.orig_df[col_name_1]])
            word_lists_2 = pd.Series([x.split() if (not is_missing(x)) else [] for x in self.orig_df[col_name_2]])
            similarity_series = [0 if ((len(a) == 0) or (len(b) == 0)) else float(len(set(a).intersection(set(b)))) / len(set(a).union(set(b)))
                                 for a, b in zip(word_lists_1, word_lists_2)]
            for test_threshold in np.arange(1.0, 0.6, -0.1):
                test_series = [x >= test_threshold for x in similarity_series]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                if test_series.tolist().count(False) < self.freq_contamination_level:
                    self._process_analysis_binary(
                        test_id,
                        [col_name_1, col_name_2],
                        np.array(test_series),
                        (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar number of words '
                         f'as each other')
                    )
                    break


    def _generate_similar_num_words(self):
        """
        Patterns without exceptions: 'sim_num_words rand_b' and 'sim_num_words all' both have similar numbers of words
            row by row (though in this example the word count is also consistent within the columns).
        Patterns with exception: 'sim_num_words most' and 'sim_num_words rand_b' consistently have similar numbers of
            words. As do 'sim_num_words most' and 'sim_num_words all'
        """
        self._add_synthetic_column('sim_num_words rand_a',
            [' '.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(2, 8)))
             for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim_num_words rand_b',
            [' '.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(4, 6)))
             for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim_num_words all',
            self.synth_df['sim_num_words rand_b'].str.replace('a', 'A').str.replace('b', 'B'))
        self._add_synthetic_column('sim_num_words most',
            self.synth_df['sim_num_words rand_b'].str.replace('a', 'A').str.replace('b', 'B'))
        self.synth_df.loc[999, 'sim_num_words most'] = 'abcdef'


    def _check_similar_num_words(self, test_id):
        # todo: handle where one column or the other has None or Nan values but otherwise similar
        word_counts_dict = {}  # todo: call self.get_word_counts_dict()
        word_counts_medians = {}
        for col_name in self.string_cols:
            word_counts_dict[col_name] = pd.Series([len(x.split()) if (not is_missing(x)) else 0 for x in self.orig_df[col_name]])
            word_counts_not_null = pd.Series([len(x.split()) for x in self.orig_df[col_name] if (not is_missing(x))])
            word_counts_medians[col_name] = word_counts_not_null.median()

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for _pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            word_counts_medians_1 = word_counts_medians[col_name_1]
            word_counts_medians_2 = word_counts_medians[col_name_2]

            # Skip in the columns have very few words in most cells
            if word_counts_medians_1 < 3 or word_counts_medians_2 < 3:
                continue

            # Skip if the difference in the median words counts is too large
            if word_counts_medians_2 < (word_counts_medians_1 / 2):
                continue
            if word_counts_medians_2 > (word_counts_medians_1 * 2):
                continue

            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            word_counts_1 = word_counts_dict[col_name_1]
            word_counts_2 = word_counts_dict[col_name_2]

            # Skip if either column has all values with the same word count
            if len(word_counts_1.value_counts()) == 1 and len(word_counts_2.value_counts()) == 1:
                continue

            similarity_series = [abs(x - y) for x, y in zip(word_counts_1, word_counts_2)]
            compare_1_to_median_of_2_series = [abs(x - word_counts_medians_2) for x in word_counts_1]
            compare_2_to_median_of_1_series = [abs(x - word_counts_medians_1) for x in word_counts_2]
            for test_threshold in range(2, -1, -1):
                test_series = [(x <= test_threshold) and (x <= y) and (x <= z)
                               for x, y, z in zip(similarity_series,
                                                  compare_1_to_median_of_2_series,
                                                  compare_2_to_median_of_1_series)]
                test_series = (test_series | self.orig_df[col_name_1].isnull()) | self.orig_df[col_name_2].isnull()
                if test_series.tolist().count(False) < self.freq_contamination_level:
                    self._process_analysis_binary(
                        test_id,
                        [col_name_1, col_name_2],
                        np.array(test_series),
                        (f'The columns "{col_name_1}" and "{col_name_2}" consistently have a similar number of words '
                         f'as each other')
                    )
                    break


    def _generate_same_first_chars(self):
        """
        Patterns without exceptions: "same_start rand_a" and "same_start all" consistently share the same first 9
            characters
        Patterns with exception:
        """
        self._add_synthetic_column('same_start rand_a',
            [''.join(np.random.choice(list(string.ascii_lowercase), 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('same_start all', [x + "xxxx" for x in self.synth_df['same_start rand_a']])
        self._add_synthetic_column('same_start most', [x + "xxxx" for x in self.synth_df['same_start rand_a']])
        self.synth_df.loc[999, 'same_start most'] = 'ABCDE'


    def _check_same_first_chars(self, test_id):
        def check_column(col_name):
            first_chars_series = self.orig_df[col_name].astype(str).str.slice(0, 1)  # Get the first letter of each value
            counts_series = first_chars_series.value_counts(normalize=True)  # Get the fraction for each first letter
            if len(counts_series) < 10:
                return False
            # Skip columns where a single first letter covers more than half the values
            return not (counts_series.iloc[0] > 0.5)

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for _pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            # Check if either column always starts with the same characters anyway.
            if not check_column(col_name_1) or not check_column(col_name_2):
                continue

            # Determine how many characters we can potentially check for matches between the columns
            avg_len_1 = pd.Series([len(x) for x in self.orig_df[col_name_1].astype(str) if (not is_missing(x))]).mean()
            avg_len_2 = pd.Series([len(x) for x in self.orig_df[col_name_2].astype(str) if (not is_missing(x))]).mean()
            max_check = min(avg_len_1, avg_len_2)

            # Determine the length of longest substrings in the values in both columns that match, such that there
            # are sufficient matches to consider this a valid pattern (less than contamination_level exceptions)
            max_number_matching = -1
            # todo: handle where one or the other of the columns has Null values
            for num_chars_checking in range(1, int(max_check) + 1):
                first_chars_series_1 = self.orig_df[col_name_1].fillna("").astype(str).str.slice(0, num_chars_checking)
                first_chars_series_2 = self.orig_df[col_name_2].fillna("").astype(str).str.slice(0, num_chars_checking)
                num_matching = sum(x == y for x, y in zip(first_chars_series_1, first_chars_series_2))
                if num_matching < (self.num_rows - self.freq_contamination_level):
                    break
                max_number_matching = num_chars_checking

            if max_number_matching <= 0:
                continue

            first_chars_series_1 = self.orig_df[col_name_1].astype(str).str.slice(0, max_number_matching)
            first_chars_series_2 = self.orig_df[col_name_2].astype(str).str.slice(0, max_number_matching)
            test_series = np.array([x == y for x, y in zip(first_chars_series_1, first_chars_series_2)])
            test_series = (test_series | self.orig_df[col_name_1].isnull()) | self.orig_df[col_name_2].isnull()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                (f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same first {max_number_matching} '
                 f'characters'))


    def _generate_same_first_word(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        # Test case where words separated by white space
        self._add_synthetic_column('same_start_word rand_a',
            [''.join(np.random.choice(['a', 'b', ' '], 10)) for _ in range(self.num_synth_rows)])
        self.synth_df['same_start_word rand_a'] += " ab"  # Ensure no values are entirely blank
        self._add_synthetic_column('same_start_word all_a',
            [x[0] + " wxyz" for x in self.synth_df['same_start_word rand_a'].str.split()])
        self._add_synthetic_column('same_start_word most_a', self.synth_df['same_start_word all_a'])
        self.synth_df.loc[999, 'same_start_word most_a'] = 'ABCDE'

        # Test case where words separated by hyphens
        self._add_synthetic_column('same_start_word rand_b',
            [''.join(np.random.choice(['a', 'b', '-'], 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('same_start_word all_b',
            [x[0] + " wxyz" for x in  self.synth_df['same_start_word rand_b'].str.split()])
        self._add_synthetic_column('same_start_word most_b', self.synth_df['same_start_word all_b'])
        self.synth_df.loc[999, 'same_start_word most_b'] = 'ABCDE'


    def _check_same_first_word(self, test_id):

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        # Get the first word in each string
        nunique_dict = self.get_nunique_dict()
        words_list_dict = self.get_words_list_dict()
        first_words_dict = {}
        sample_first_words_dict = {}
        for col_name in self.string_cols:
            first_words_dict[col_name] = [x[0] if (x and (len(x) > 0)) else "" for x in words_list_dict[col_name]]
            sample_first_words_dict[col_name] = pd.Series([x[0] if (x and (len(x) > 0)) else "" for x in words_list_dict[col_name]]).loc[self.sample_df.index]

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for col_name_1, col_name_2 in pairs:
            # Skip columns that are primarily 1 word
            word_arr = words_list_dict[col_name_1]
            num_words_arr = [len(x) for x in word_arr]
            if pd.Series(num_words_arr).quantile(0.5) <= 1:
                continue
            word_arr = words_list_dict[col_name_2]
            num_words_arr = [len(x) for x in word_arr]
            if pd.Series(num_words_arr).quantile(0.5) <= 1:
                continue

            # Skip columns that have few unique values
            if nunique_dict[col_name_1] < 10:
                continue
            if nunique_dict[col_name_2] < 10:
                continue

            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample
            test_series = [x == y for x, y in zip(sample_first_words_dict[col_name_1], sample_first_words_dict[col_name_2])]
            if test_series.count(False) > 1:
                continue

            # Test on the full columns
            test_series = [x == y for x, y in zip(first_words_dict[col_name_1], first_words_dict[col_name_2])]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same first word'
            )


    def _generate_same_last_word(self):
        """
        Patterns without exceptions: 'same_last_word all_a' consistently has the same last word as
            'same_last_word rand_a'. As well, 'same_last_word all_b' consistently has the same last word as
            'same_last_word rand_b'
        Patterns with exception: 'same_last_word most_a' constistently has the same last word as 'same_last_word all_a'
            and 'same_last_word rand_a', with exceptions. Similar for the 'b' case using hyphens.
        """
        # Test case where words separated by white space
        self._add_synthetic_column('same_last_word rand_a',
            [''.join(np.random.choice(['a', 'b', ' '], 10)) for _ in range(self.num_synth_rows)])
        # Ensure no values are entirely blank
        self.synth_df['same_last_word rand_a'] = "ab " + self.synth_df['same_last_word rand_a']

        self._add_synthetic_column('same_last_word all_a',
            ["wxyz " + x[-1] for x in self.synth_df['same_last_word rand_a'].astype(str).str.split()])

        self._add_synthetic_column('same_last_word most_a', self.synth_df['same_last_word all_a'])
        self.synth_df.loc[999, 'same_last_word most_a'] = 'ABCDE'

        # Test case where words separated by hyphens
        self._add_synthetic_column('same_last_word rand_b',
            [''.join(np.random.choice(['a', 'b', '-'], 10)) for _ in range(self.num_synth_rows)])

        self._add_synthetic_column('same_last_word all_b',
            ["wxyz " + x[-1] for x in  self.synth_df['same_last_word rand_b'].astype(str).str.split()])

        self._add_synthetic_column('same_last_word most_b', self.synth_df['same_last_word all_b'])
        self.synth_df.loc[999, 'same_last_word most_b'] = 'ABCDE'


    def _check_same_last_word(self, test_id):
        """
        This skips columns that have few unique values.
        """

        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        nunique_dict = self.get_nunique_dict()
        words_list_dict = self.get_words_list_dict()
        last_words_dict = {}
        sample_last_words_dict = {}
        for col_name in self.string_cols:
            last_words_dict[col_name] = [x[-1] if (x and (len(x) > 0)) else "" for x in words_list_dict[col_name]]
            sample_last_words_dict[col_name] = pd.Series([x[-1] if (x and (len(x) > 0)) else ""
                                                          for x in words_list_dict[col_name]]).loc[self.sample_df.index]

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for col_name_1, col_name_2 in pairs:
            # Skip columns that are primarily 1 word
            word_arr = words_list_dict[col_name_1]
            num_words_arr = [len(x) for x in word_arr]
            if pd.Series(num_words_arr).quantile(0.5) <= 1:
                continue
            word_arr = words_list_dict[col_name_2]
            num_words_arr = [len(x) for x in word_arr]
            if pd.Series(num_words_arr).quantile(0.5) <= 1:
                continue

            # Skip columns that have few unique values
            if nunique_dict[col_name_1] < 10:
                continue
            if nunique_dict[col_name_2] < 10:
                continue

            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample
            test_series = [x == y for x, y in zip(sample_last_words_dict[col_name_1], sample_last_words_dict[col_name_2])]
            if test_series.count(False) > 1:
                continue

            # Test on the full columns
            test_series = [x == y for x, y in zip(last_words_dict[col_name_1], last_words_dict[col_name_2])]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same last word'
            )


    def _generate_same_alpha_chars(self):
        """
        Patterns without exceptions: "same_alpha rand" and "same_alpha all" consistently share the same alphabetic
            characters
        Patterns with exception: "same_alpha rand" and "same_alpha most" consistently share the same alphabetic
            characters, with 1 exception
        Not Flagged: "same_alpha all" and "same_alpha most" consistently share the same alphabetic characters, but are
            almost identical, so not flagged by this test.
        """
        self._add_synthetic_column('same_alpha rand',
            [''.join(np.random.choice(list(string.ascii_lowercase), 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('same_alpha all',
            self.synth_df['same_alpha rand'].str[1:] + [str(np.random.choice(list(string.digits)))])
        self._add_synthetic_column('same_alpha most', self.synth_df['same_alpha all'])
        self.synth_df.loc[999, 'same_alpha most'] = 'ABCDE'


    def _check_same_alpha_chars(self, test_id):
        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for _pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample.
            alpha_col_1 = self.sample_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_1 = [[c for c in x if c and c.isalpha()] for x in alpha_col_1]
            if [len(x) > 0 for x in alpha_col_1].count(False) > 1:
                continue
            alpha_col_2 = self.sample_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_2 = [[c for c in x if c and c.isalpha()] for x in alpha_col_2]
            if [len(x) > 0 for x in alpha_col_2].count(False) > 1:
                continue
            test_series = [_shared_fraction(x, y) > 0.80 if len(y) > 0 else False
                       for x, y in zip(alpha_col_1, alpha_col_2)]
            if test_series.count(False) > 1:
                continue

            # Test on the actual data
            alpha_col_1 = self.orig_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_1 = [[c for c in x if c and c.isalpha()] for x in alpha_col_1]
            alpha_col_2 = self.orig_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            alpha_col_2 = [[c for c in x if c and c.isalpha()] for x in alpha_col_2]
            test_series = [_shared_fraction(x, y) > 0.80
                           for x, y in zip(alpha_col_1, alpha_col_2)]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same alphabetic characters '
            )


    def _generate_same_numeric_chars(self):
        """
        Patterns without exceptions: 'same_num rand' and 'same_num all' consistently have the same numeric characters
        Patterns with exception: 'same_num rand' and 'same_num most' consistently have the same numeric characters, with
            one exception
        Not flagged: 'same_num most' and 'same_num all' are nearly identical, and not flagged by this test
        """
        self._add_synthetic_column('same_num rand',
            [''.join(np.random.choice(list(string.digits), 12)) + 'A' for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('same_num all',
            self.synth_df['same_num rand'] + ''.join(np.random.choice(list(string.ascii_lowercase))))
        self._add_synthetic_column('same_num most', self.synth_df['same_num all'])
        self.synth_df.loc[999, 'same_num most'] = '1234'


    def _check_same_numeric_chars(self, test_id):
        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        sample_digit_str_dict = {}
        digit_str_dict = {}
        for col_name in self.string_cols:
            digits_col = self.sample_df[col_name].astype(str).apply(lambda x: "" if is_missing(x) else x)
            digits_col = [[c for c in x if c and c.isdigit()] for x in digits_col]
            digits_col = [''.join(x) for x in digits_col]
            sample_digit_str_dict[col_name] = digits_col

            digits_col = self.orig_df[col_name].astype(str).apply(lambda x: "" if is_missing(x) else x)
            digits_col = [[c for c in x if c and c.isdigit()] for x in digits_col]
            digits_col = [''.join(x) for x in digits_col]
            digit_str_dict[col_name] = digits_col

        for _pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample
            digits_col_1 = sample_digit_str_dict[col_name_1]
            if [len(x) > 0 for x in digits_col_1].count(False) > 1:
                continue
            digits_col_2 = sample_digit_str_dict[col_name_2]
            if [len(x) > 0 for x in digits_col_2].count(False) > 1:
                continue
            test_series = [x == y for x, y in zip(digits_col_1, digits_col_2)]
            if test_series.count(False) > 1:
                continue

            # Test on the full columns
            digits_col_1 = digit_str_dict[col_name_1]
            digits_col_2 = digit_str_dict[col_name_2]
            test_series = [x == y for x, y in zip(digits_col_1, digits_col_2)]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same numeric characters '
            )


    def _generate_same_special_chars(self):
        """
        Patterns without exceptions: "same_special all" contains the same special characters as "same_special rand_b"
        Patterns with exception: "same_special most" contains the same special characters as "same_special rand_b"
            with one exception. Also, "same_special most" contains the same special characters as "same_special all"
            with one exception.
        """
        self._add_synthetic_column('same_special rand_a',
            [''.join(np.random.choice(['$', '%', '#', '@', '!'], 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('same_special rand_b',
            self.synth_df['same_special rand_a'].str[:1] + ''.join(np.random.choice(list(string.ascii_lowercase), 3)))
        # Test that Á is not considered a special character.
        self._add_synthetic_column('same_special all',
            self.synth_df['same_special rand_b'].str[:] + 'Á'.join(np.random.choice(list(string.ascii_lowercase), 10)))
        self._add_synthetic_column('same_special most', self.synth_df['same_special all'])
        self.synth_df.loc[999, 'same_special most'] = '&&^%'


    def _check_same_special_chars(self, test_id):
        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for col_name_1, col_name_2 in pairs:
            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample.
            special_col_1 = self.sample_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            special_col_1 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()) and (c != ' '))] for x in special_col_1]
            if [len(x) > 0 for x in special_col_1].count(False) > 1:
                continue
            special_col_2 = self.sample_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            special_col_2 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()) and (c != ' '))] for x in special_col_2]
            if [len(x) > 0 for x in special_col_2].count(False) > 1:
                continue
            test_series = [set(x) == set(y) for x, y in zip(special_col_1, special_col_2)]
            if test_series.count(False) > 1:
                continue

            # Test on the full columns
            special_col_1 = self.orig_df[col_name_1].apply(lambda x: "" if is_missing(x) else x)
            special_col_1 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()) and (c != ' '))] for x in special_col_1]
            if [len(x) > 0 for x in special_col_1].count(False) > self.freq_contamination_level:
                continue
            special_col_2 = self.orig_df[col_name_2].apply(lambda x: "" if is_missing(x) else x)
            special_col_2 = [[c for c in x if ((not c.isalpha()) and (not c.isdigit()) and (c != ' '))] for x in special_col_2]
            if [len(x) > 0 for x in special_col_2].count(False) > self.freq_contamination_level:
                continue
            test_series = [set(x) == set(y) for x, y in zip(special_col_1, special_col_2)]
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'Columns "{col_name_1}" and "{col_name_2}" consistently share the same special characters',
                display_info={
                    "Special Chars A": [str(list(set(x)))[1:-1] for x in special_col_1],
                    "Special Chars B": [str(list(set(x)))[1:-1] for x in special_col_2]
                }
            )


    def _generate_a_prefix_of_b(self):
        """
        Patterns without exceptions: "a_prefix_b rand_a" is consistently the same as the first characters of
            "a_prefix_b all"
        Patterns with exception: "a_prefix_b rand_a" is consistently the same as the first characters of
            "a_prefix_b most", with exceptions
        """
        self._add_synthetic_column('a_prefix_b rand_a',
            [''.join(np.random.choice(list(string.ascii_lowercase), 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('a_prefix_b all',
            self.synth_df['a_prefix_b rand_a'] + ''.join(np.random.choice(list(string.ascii_lowercase), 10)))
        self._add_synthetic_column('a_prefix_b most', self.synth_df['a_prefix_b all'])
        self.synth_df.loc[999, 'a_prefix_b most'] = 'abcdef'


    def _check_a_prefix_of_b(self, test_id):
        """
        Handling null values: This test skips columns that are primarily null. Any patterns are not considered violated
        in a given row if either cell is null.
        """

        is_missing_dict = self.get_is_missing_dict()
        sample_is_missing_dict = self.get_sample_is_missing_dict()

        num_pairs, pairs = self._get_string_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 500 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(pairs):,} pairs of string columns")

            # Skip columns that are primarily Null
            if is_missing_dict[col_name_1].tolist().count(True) > (self.num_rows / 2):
                continue
            if is_missing_dict[col_name_2].tolist().count(True) > (self.num_rows / 2):
                continue

            # Skip if the two columns are largely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test first on a sample
            test_series = [True if (w or x) else ((len(y) < len(z)) and (y == z[:len(y)]))
                           for w, x, y, z in zip(
                    sample_is_missing_dict[col_name_1],
                    sample_is_missing_dict[col_name_2],
                    self.sample_df[col_name_1].astype(str),
                    self.sample_df[col_name_2].astype(str)
                )]
            if test_series.count(False) > 1:
                continue

            test_series = [True if (w or x) else ((len(y) < len(z)) and (y == z[:len(y)]))
                           for w, x, y, z in zip(
                    is_missing_dict[col_name_1],
                    is_missing_dict[col_name_2],
                    self.orig_df[col_name_1].astype(str).str.strip(),
                    self.orig_df[col_name_2].astype(str).str.strip()
                )]
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'The column "{col_name_1}" is consistently the same as the first characters of "{col_name_2}"'
            )


    def _generate_a_suffix_of_b(self):
        """
        Patterns without exceptions: "a_suffix_b rand_a" is consistently the same as the last characters of
            "a_suffix_b all"
        Patterns with exception: "a_suffix_b rand_a" is consistently the same as the last characters of
            "a_suffix_b most", with exceptions
        """
        self._add_synthetic_column('a_suffix_b rand_a',
            [''.join(np.random.choice(list(string.ascii_lowercase), 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('a_suffix_b all',
            ''.join(np.random.choice(list(string.ascii_lowercase), 10)) + self.synth_df['a_suffix_b rand_a'])
        self._add_synthetic_column('a_suffix_b most', self.synth_df['a_suffix_b all'])
        self.synth_df.loc[999, 'a_suffix_b most'] = 'abcdef'


    def _check_a_suffix_of_b(self, test_id):
        is_missing_dict = self.get_is_missing_dict()
        sample_is_missing_dict = self.get_sample_is_missing_dict()

        num_pairs, pairs = self._get_string_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 500 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(pairs):,} pairs of string columns")

            # Skip columns that are primarily Null
            if is_missing_dict[col_name_1].tolist().count(True) > (self.num_rows / 2):
                continue
            if is_missing_dict[col_name_2].tolist().count(True) > (self.num_rows / 2):
                continue

            # Test first on a sample
            test_series = [True if (w or x) else ((len(y) < len(z)) and (y == z[-len(y):]))
                           for w, x, y, z in zip(
                    sample_is_missing_dict[col_name_1],
                    sample_is_missing_dict[col_name_2],
                    self.sample_df[col_name_1].astype(str),
                    self.sample_df[col_name_2].astype(str)
                )]
            if test_series.count(False) > 1:
                continue

            test_series = [True if (w or x) else ((len(y) < len(z)) and (y == z[-len(y):]))
                           for w, x, y, z in zip(
                    is_missing_dict[col_name_1],
                    is_missing_dict[col_name_2],
                    self.orig_df[col_name_1].astype(str),
                    self.orig_df[col_name_2].astype(str)
                )]
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'The column "{col_name_1}" is consistently the same as the last characters of "{col_name_2}"'
            )


    def _generate_b_contains_a(self):
        """
        Patterns without exceptions: "b_contains_a rand_a" is consistently contained in "b_contains_a all"
        Patterns with exception: "b_contains_a rand_a" is consistently contained in "b_contains_a rand_a", with
            exceptions
        """
        self._add_synthetic_column('b_contains_a rand_a',
            [''.join(np.random.choice(list(string.ascii_lowercase), 10)) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('b_contains_a all',
            ''.join(np.random.choice(list(string.ascii_lowercase), 10)) + self.synth_df['b_contains_a rand_a'] +
            ''.join(np.random.choice(list(string.ascii_lowercase), 10)))
        self._add_synthetic_column('b_contains_a most', self.synth_df['b_contains_a all'])
        self.synth_df.loc[999, 'b_contains_a most'] = 'abcdef'


    def _check_b_contains_a(self, test_id):
        num_pairs, pairs = self._get_string_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        is_missing_dict = self.get_is_missing_dict()
        sample_is_missing_dict = self.get_sample_is_missing_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 1000 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(pairs):,} pairs of string columns")

            if self.orig_df[col_name_1].apply(is_missing).sum() > (self.num_rows / 10.0):
                continue
            if self.orig_df[col_name_2].apply(is_missing).sum() > (self.num_rows / 10.0):
                continue

            # Test on a sample first
            test_series = [True if v and w else
                           ((not v) and (not w) and (len(x) < len(y)) and (x != y[:len(x)]) and (x != y[-len(x):]) and (x in y))
                           for v, w, x, y in zip(
                                sample_is_missing_dict[col_name_1],
                                sample_is_missing_dict[col_name_2],
                                self.sample_df[col_name_1],
                                self.sample_df[col_name_2])]
            if test_series.count(False) > 1:
                continue

            # Test on the full data
            test_series = [True if v and w else
                           ((not v) and (not w) and (len(x) < len(y)) and (x != y[:len(x)]) and (x != y[-len(x):]) and (x in y))
                           for v, w, x, y in zip(
                                is_missing_dict[col_name_1],
                                is_missing_dict[col_name_2],
                                self.orig_df[col_name_1],
                                self.orig_df[col_name_2])]
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                (f'The column "{col_name_1}" is consistently contained within "{col_name_2}", but not the first or last '
                 f'characters')
            )


    def _generate_correlated_alpha(self):
        """
        Patterns without exceptions: "correlated_alpha rand" is consistently similar, with regards to percentile, to
            "correlated_alpha all"
        Patterns with exception: "correlated_alpha rand" is consistently similar, with regards to percentile, to
            "correlated_alpha most", with exceptions,
        Not flagged "correlated_alpha rand_b" is consistently similar, with regards to percentile, to
            "correlated_alpha most", but is not flagged as the columns are almost identical
        """
        list_a = sorted([''.join(np.random.choice(list(string.ascii_lowercase), 10))
                         for _ in range(self.num_synth_rows)])
        list_b = sorted([''.join(np.random.choice(list(string.ascii_lowercase), 10))
                         for _ in range(self.num_synth_rows)])
        c = list(zip(list_a, list_b))
        random.shuffle(c)
        list_a, list_b = zip(*c)

        self._add_synthetic_column('correlated_alpha rand', list_a)
        self._add_synthetic_column('correlated_alpha all', list_b)
        list_b = list(list_b)
        list_b[-1] = 'aaaaaaaaaa'
        self._add_synthetic_column('correlated_alpha most', list_b)


    def _check_correlated_alpha(self, test_id):
        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        n_unique_dict = self.get_nunique_dict()
        is_missing_dict = self.get_is_missing_dict()

        percentiles_dict = {}
        for col_name in self.string_cols:
            percentiles_dict[col_name] = self.orig_df[col_name].rank(pct=True)

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 1000 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(pairs):,} pairs of string columns")

            # Check both columns have a significant number of unique values
            if n_unique_dict[col_name_1] < 3:
                continue
            if n_unique_dict[col_name_2] < 3:
                continue

            # Check neither column has too many missing values
            if is_missing_dict[col_name_1].sum() > (self.num_rows * 0.75):
                continue
            if is_missing_dict[col_name_2].sum() > (self.num_rows * 0.75):
                continue

            # Skip columns that are almost entirely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test first on a sample
            vals_arr_1 = self.sample_df[col_name_1].astype(str)
            vals_arr_2 = self.sample_df[col_name_2].astype(str)
            spearancorr = abs(vals_arr_1.rank().corr(vals_arr_2.rank(), method='spearman'))
            if spearancorr < 0.8:  # Require a less perfect correlation on samples, which may vary.
                continue

            # Test on the full columns
            vals_arr_1 = self.orig_df[col_name_1].astype(str)
            vals_arr_2 = self.orig_df[col_name_2].astype(str)

            # The class variable, self.spearman_corr, covers only numeric columns, so we calculate the correlation here.
            spearancorr = abs(vals_arr_1.rank().corr(vals_arr_2.rank(), method='spearman'))

            if spearancorr >= 0.95:
                col_1_percentiles = percentiles_dict[col_name_1]
                col_2_percentiles = percentiles_dict[col_name_2]

                # Test for positive correlation
                test_series = np.array([abs(x-y) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_1}" is consistently similar, with regards to percentile, to "{col_name_2}"',
                    display_info={"col_1_percentiles": col_1_percentiles, "col_2_percentiles": col_2_percentiles})

                # Test for negative correlation
                test_series = np.array([abs(x-(1.0 - y)) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    f'"{col_name_1}" is consistently inversely similar in percentile to "{col_name_2}"',
                    display_info={"col_1_percentiles": col_1_percentiles, "col_2_percentiles": col_2_percentiles})

    ##################################################################################################################
    # Data consistency checks for one non-numeric column and one numeric column
    ##################################################################################################################


    def _generate_large_given(self):
        """
        Patterns without exceptions: None. This test does not generate patterns.
        Patterns with exception: 'large_given most' contains one row with value 290, which is common for the column
            but not common when 'large_given rand' is 'C'. As well, 'large_given date_most' contains one row with
            value 01-8-2016, which is common for the column, but not common when 'large_given rand' is 'C'.
        """
        self._add_synthetic_column('large_given rand', ['A']*100 + ["B"]*100 + ['C']*(self.num_synth_rows - 200))
        self._add_synthetic_column('large_given all',
            np.concatenate([
                np.random.randint(200, 300, 100),
                np.random.randint(100, 200, 100),
                np.random.randint(0, 100, (self.num_synth_rows - 200))
            ]))
        self._add_synthetic_column('large_given most', self.synth_df['large_given all'])
        self.synth_df.loc[999, 'large_given most'] = 290

        test_date1 = datetime.datetime.strptime("01-7-2016", "%d-%m-%Y")
        test_date2 = datetime.datetime.strptime("01-7-2015", "%d-%m-%Y")
        test_date3 = datetime.datetime.strptime("01-7-2014", "%d-%m-%Y")
        self._add_synthetic_column('large_given date_most',
            np.concatenate([
                [test_date1 + relativedelta(days=np.random.randint(1, 300)) for _ in range(100)],
                [test_date2 + relativedelta(days=np.random.randint(1, 300)) for _ in range(100)],
                [test_date3 + relativedelta(days=np.random.randint(1, 300)) for _ in range(self.num_synth_rows - 200)]
            ]))
        self.synth_df.loc[999, 'large_given date_most'] = datetime.datetime.strptime("01-8-2016", "%d-%m-%Y")


    def _check_large_given(self, test_id):

        # Determine if there are too many combinations to execute
        total_combinations = (len(self.string_cols) + len(self.binary_cols)) * (len(self.numeric_cols) + len(self.date_cols))
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {(len(self.numeric_cols) + len(self.date_cols)):,} "
                       f"numeric and date columns, multiplied by {(len(self.string_cols) + len(self.binary_cols)):,} "
                       f"string and binary columns, results in {total_combinations:,} combinations. max_combinations "
                       f"is currently set to {self.max_combinations:,}.")
            return

        # Calculate and cache the upper limit based on q1 and q3 of each full numeric & date column
        upper_limits_dict = self.get_columns_iqr_upper_limit()

        for col_idx, col_name_1 in enumerate(self.string_cols + self.binary_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.string_cols) + len(self.binary_cols)} string and "
                       f"binary columns")

            # Get the common values in col_name_1. Below, we find large values in col_name_2 for each common value
            # in col_name_1
            if self.orig_df[col_name_1].nunique() > 10:
                continue
            vc = self.orig_df[col_name_1].value_counts()
            common_values = []
            sub_dfs_dict = {}
            for v in vc.index:
                if vc[v] > math.sqrt(self.num_rows):
                    common_values.append(v)
                    sub_dfs_dict[v] = self.orig_df[self.orig_df[col_name_1] == v]

            for col_name_2 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                for v in common_values:
                    sub_df = sub_dfs_dict[v]
                    if col_name_2 in self.numeric_cols:
                        # Get the upper limit (based on IQR), given the full column
                        upper_limit, col_q2, col_q3 = upper_limits_dict[col_name_2]

                        # Get the numeric values of the current subset
                        num_vals_no_null = pd.Series(
                            [float(x) for x in sub_df[col_name_2]
                             if str(x).replace('-', '').replace('.', '').isdigit()], dtype=float)
                        if len(num_vals_no_null) < 100:
                            continue
                        subset_med = num_vals_no_null.median()
                        num_vals = convert_to_numeric(sub_df[col_name_2], subset_med)
                        num_vals = num_vals.fillna(subset_med)

                        # We are only concerned in this test with subsets that tend to have smaller values in the
                        # numeric column, and flag large values in this case. We do not flag values in subsets that
                        # have any values that are large relative to the subset; these are flagged by other tests.
                        res_1 = num_vals > upper_limit
                        if res_1.tolist().count(True) > 0:
                            continue

                        # Get the upper limit given the current subset
                        q1 = num_vals.quantile(0.25)
                        q2 = num_vals.quantile(0.5)
                        q3 = num_vals.quantile(0.75)
                        upper_limit_subset = q3 + (self.iqr_limit * (q3 - q1))

                        # Check this subset is small compared to the full column
                        if (q2 >= col_q2) or (q3 >= col_q3):
                            continue

                        res = pd.Series([x <= upper_limit_subset for x in num_vals])
                    else:
                        # Get the iqr given the full column
                        upper_limit, col_q2, col_q3 = upper_limits_dict[col_name_2]
                        if upper_limit is None:
                            continue

                        res_1 = [x > upper_limit for x in pd.to_datetime(sub_df[col_name_2])]
                        if res_1.count(True) > 0:
                            continue

                        # Get the iqr given the current subset
                        q1 = pd.to_datetime(sub_df[col_name_2]).quantile(0.25, interpolation='midpoint')
                        q3 = pd.to_datetime(sub_df[col_name_2]).quantile(0.75, interpolation='midpoint')
                        try:
                            upper_limit_subset = q3 + (self.iqr_limit * (q3 - q1))
                        except Exception:
                            continue

                        res = pd.Series([x <= upper_limit_subset for x in pd.to_datetime(sub_df[col_name_2])])

                    if 0 < res.tolist().count(False) <= self.freq_contamination_level:
                        index_of_large = [x for x, y in zip(sub_df.index, res) if y == False]  # noqa: E712
                        for i in index_of_large:
                            test_series[i] = False

                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'"{col_name_2}" contains very large values given the specific value in "{col_name_1}" (Values '
                     f'that are large given any value of "{col_name_1}" are not flagged by this test.)'),
                    allow_patterns=False  # todo: ones that set allow_patterns=False may get ,. in string. check for that. maybe just do a replace()
                )


    def _generate_small_given(self):
        """
        Patterns without exceptions: None. This test does not generate patterns.
        Patterns with exception: 'small_given most' has one row with value 5, which is not small generally, but is
            small when 'small_given rand' has value 'C'. As well, 'small_given date_most' contains one row with
            value 01-8-2016, which is common for the column, but not common when 'small_given rand' is 'C'.
        """
        self._add_synthetic_column('small_given rand', ['A']*100 + ["B"]*100 + ['C']*(self.num_synth_rows - 200))
        self._add_synthetic_column('small_given all',
            np.concatenate([
                np.random.randint(0, 100, 100),
                np.random.randint(100, 200, 100),
                np.random.randint(200, 300, (self.num_synth_rows - 200))
            ]))
        self._add_synthetic_column('small_given most', self.synth_df['small_given all'])
        self.synth_df.loc[999, 'small_given most'] = 5

        test_date1 = datetime.datetime.strptime("01-7-2014", "%d-%m-%Y")
        test_date2 = datetime.datetime.strptime("01-7-2015", "%d-%m-%Y")
        test_date3 = datetime.datetime.strptime("01-7-2016", "%d-%m-%Y")
        self._add_synthetic_column('small_given date_most',
            np.concatenate([
                [test_date1 + relativedelta(days=np.random.randint(1, 400)) for _ in range(100)],
                [test_date2 + relativedelta(days=np.random.randint(1, 400)) for _ in range(100)],
                [test_date3 + relativedelta(days=np.random.randint(1, 400)) for _ in range(self.num_synth_rows - 200)]
            ]))
        self.synth_df.loc[999, 'small_given date_most'] = datetime.datetime.strptime("01-8-2014", "%d-%m-%Y")


    def _check_small_given(self, test_id):
        # Determine if there are too many combinations to execute
        total_combinations = (len(self.string_cols) + len(self.binary_cols)) * (len(self.numeric_cols) + len(self.date_cols))
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {(len(self.numeric_cols) + len(self.date_cols)):,} "
                       f"numeric and date columns, multiplied by {(len(self.string_cols) + len(self.binary_cols)):,} "
                       f"string and binary columns, results in {total_combinations:,} combinations. max_combinations "
                       f"is currently set to {self.max_combinations:,}.")
            return

        # Calculate and cache the lower limit based on q1 and q3 of each full numeric & date column
        lower_limits_dict = self.get_columns_iqr_lower_limit()

        for col_idx, col_name_1 in enumerate(self.string_cols + self.binary_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.string_cols) + len(self.binary_cols)} string and "
                       f"binary columns")

            # Get the common values in col_name_1. Below, we find small values in col_name_2 for each common value
            # in col_name_1
            if self.orig_df[col_name_1].nunique() > 10:
                continue
            vc = self.orig_df[col_name_1].value_counts()
            common_values = []
            sub_dfs_dict = {}
            for v in vc.index:
                if vc[v] > math.sqrt(self.num_rows):
                    common_values.append(v)
                    sub_dfs_dict[v] = self.orig_df[self.orig_df[col_name_1] == v]

            for col_name_2 in self.numeric_cols + self.date_cols:

                if self.orig_df[col_name_2].nunique() < math.sqrt(self.num_rows):
                    continue

                test_series = [True] * self.num_rows
                for v in common_values:
                    sub_df = sub_dfs_dict[v]
                    if col_name_2 in self.numeric_cols:
                        # Get the iqr given the full column
                        lower_limit, col_d1, col_q1 = lower_limits_dict[col_name_2]

                        # Get the iqr given the current subset
                        num_vals_no_null = pd.Series(
                            [float(x) for x in sub_df[col_name_2]
                             if str(x).replace('-', '').replace('.', '').isdigit()])
                        if len(num_vals_no_null) < 100:
                            continue
                        subset_med = num_vals_no_null.median()
                        num_vals = convert_to_numeric(sub_df[col_name_2], subset_med)
                        num_vals = num_vals.fillna(subset_med)

                        # We are only concerned in this test with subsets that tend to have larger values in the
                        # numeric column, and flag small values in this case. We do not flag values in subsets that
                        # have any values that are small relative to the subset; these are flagged by other tests.
                        res_1 = num_vals < lower_limit
                        if res_1.tolist().count(True) > 0:
                            continue

                        d1 = num_vals.quantile(0.1)
                        d9 = num_vals.quantile(0.9)
                        lower_limit_subset = d1 - (self.idr_limit * (d9 - d1))

                        # Ensure this subset is at least one decile shifted from the full column
                        if d1 < col_q1:
                            continue

                        res = pd.Series([x >= lower_limit_subset for x in num_vals])
                    else:
                        # Get the lower limit for the full column
                        lower_limit, col_d1, col_q1 = lower_limits_dict[col_name_2]
                        if lower_limit is None:
                            continue

                        res_1 = [x < lower_limit for x in pd.to_datetime(sub_df[col_name_2])]
                        if res_1.count(True) > 0:
                            continue

                        # Get the iqr given the current subset
                        q1 = pd.to_datetime(sub_df[col_name_2]).quantile(0.25, interpolation='midpoint')
                        q3 = pd.to_datetime(sub_df[col_name_2]).quantile(0.75, interpolation='midpoint')
                        try:
                            lower_limit_subset = q1 - (self.iqr_limit * (q3 - q1))
                        except Exception:
                            continue

                        res = pd.Series([x >= lower_limit_subset for x in pd.to_datetime(sub_df[col_name_2])])

                    if 0 < res.tolist().count(False) <= self.freq_contamination_level:
                        index_of_small = [x for x, y in zip(sub_df.index, res) if not y]
                        for i in index_of_small:
                            test_series[i] = False

                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'"{col_name_2}" contains very small values given the specific value in "{col_name_1}" (Values '
                     f'that are small given any value of "{col_name_1}" are not flagged by this test.)'),
                    allow_patterns=False
                )


    def _generate_large_given_prefix(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'large_given most' contains one row with value 290, which is common for the column
            but not common when 'large_given rand' is 'C'
        """
        self._add_synthetic_column('large_given_prefix rand',
                                    ['A-' + np.random.choice(list(string.ascii_letters)) for _ in range(100)] +
                                    ['B-' + np.random.choice(list(string.ascii_letters)) for _ in range(100)] +
                                    ['C-' + np.random.choice(list(string.ascii_letters)) for _ in range(self.num_synth_rows - 200)])
        self._add_synthetic_column('large_given_prefix all', np.concatenate([
            np.random.randint(200, 300, 100),
            np.random.randint(100, 200, 100),
            np.random.randint(0, 100, (self.num_synth_rows - 200))
        ]))
        self._add_synthetic_column('large_given_prefix most', self.synth_df['large_given_prefix all'])
        self.synth_df.loc[999, 'large_given_prefix most'] = 290

        # Date columns
        dates_arr = []
        for _i in range(100):
            y = np.random.randint(1990, 2025, 1)[0]
            dates_arr.append(datetime.datetime.strptime(f"01-7-{y}", "%d-%m-%Y"))
        for _i in range(100):
            y = np.random.randint(1980, 1989, 1)[0]
            dates_arr.append(datetime.datetime.strptime(f"01-7-{y}", "%d-%m-%Y"))
        for _i in range(self.num_synth_rows - 200):
            y = np.random.randint(1970, 1979, 1)[0]
            dates_arr.append(datetime.datetime.strptime(f"01-7-{y}", "%d-%m-%Y"))
        self._add_synthetic_column('large_given_prefix date_all', dates_arr)
        self._add_synthetic_column('large_given_prefix date_most', self.synth_df['large_given_prefix date_all'])
        self.synth_df.loc[999, 'large_given_prefix date_most'] = datetime.datetime.strptime("01-7-1986", "%d-%m-%Y")


    def _check_large_given_prefix(self, test_id):
        # Determine if there are too many combinations to execute
        total_combinations = len(self.string_cols) * (len(self.numeric_cols) + len(self.date_cols))
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {(len(self.numeric_cols) + len(self.date_cols)):,} "
                       f"numeric and date columns, multiplied by {len(self.string_cols):,} "
                       f"string columns, results in {total_combinations:,} combinations. max_combinations "
                       f"is currently set to {self.max_combinations:,}.")
            return

        # todo: when show non-flagged, show with the flagged prefixes
        # todo: dont flag Null values
        # Calculate and cache the upper limit based on q1 and q3 of each full numeric & date column
        upper_limits_dict = self.get_columns_iqr_upper_limit()
        nunique_dict = self.get_nunique_dict()

        for col_idx, col_name_1 in enumerate(self.string_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.string_cols) + len(self.binary_cols)} string and "
                      f"binary columns")

            col_vals = self.orig_df[col_name_1].astype(str).apply(replace_special_with_space)

            # Check the column's values are usually more than 1 word
            word_arr = col_vals.str.split()  # todo: call self.get_word_counts_dict()
            word_counts_arr = pd.Series([len(x) for x in word_arr])
            if word_counts_arr.quantile(0.75) <= 1:
                continue

            first_words = pd.Series([x[0] if len(x) > 0 else "" for x in col_vals.str.split()])
            if first_words.nunique() > 10:
                continue

            # Skip columns where the set of unique first words is almost as large as the set of unique strings. In
            # this case, the first word is not meaningful.
            if first_words.nunique() > (col_vals.nunique() / 2):
                continue
            vc = first_words.value_counts()
            common_values = []
            for v in vc.index:
                if (vc[v] > math.sqrt(self.num_rows)) and (vc[v] > 50):
                    common_values.append(v)

            for col_name_2 in self.numeric_cols + self.date_cols:
                if nunique_dict[col_name_2] < math.sqrt(self.num_rows):
                    continue
                test_series = [True] * self.num_rows
                flagged_prefixes = []
                for v in common_values:
                    sub_df = self.orig_df[[col_name_2]][first_words == v]

                    if col_name_2 in self.numeric_cols:
                        # Get the iqr given the full column
                        upper_limit, col_q2, col_q3 = upper_limits_dict[col_name_2]

                        # Get the iqr given the current subset
                        num_vals_no_null = pd.Series([float(x) for x in sub_df[col_name_2] if str(x).replace('-', '').replace('.', '').isdigit()])
                        if len(num_vals_no_null) < 100:
                            continue
                        subset_med = num_vals_no_null.median()
                        num_vals = convert_to_numeric(sub_df[col_name_2], subset_med)
                        num_vals = num_vals.fillna(subset_med)

                        # Check if this subset has values that would be flagged even in not separating by string values.
                        res_1 = num_vals > upper_limit
                        if res_1.tolist().count(True) > 0:
                            continue

                        q1 = num_vals.quantile(0.25)
                        q2 = num_vals.quantile(0.5)
                        q3 = num_vals.quantile(0.75)
                        upper_limit_subset = q3 + (self.iqr_limit * (q3 - q1))

                        # Check this subset is small compared to the full column
                        if (q2 >= col_q2) or (q3 >= col_q3):
                            continue

                        res = pd.Series([x <= upper_limit_subset for x in num_vals])
                    else:
                        # Get the iqr given the full column
                        upper_limit, col_q2, col_q3 = upper_limits_dict[col_name_2]
                        if upper_limit is None:
                            continue

                        res_1 = [x > upper_limit for x in pd.to_datetime(sub_df[col_name_2])]
                        if res_1.count(True) > 0:
                            continue

                        # Get the iqr given the current subset
                        q1 = pd.to_datetime(sub_df[col_name_2]).quantile(0.25, interpolation='midpoint')
                        q3 = pd.to_datetime(sub_df[col_name_2]).quantile(0.75, interpolation='midpoint')
                        try:
                            # Use a coeffiecient of 1.5 for dates, which tend to vary much less than numeric values.
                            upper_limit_subset = q3 + (1.5 * (q3 - q1))
                        except Exception:
                            continue

                        res_1 = [x > upper_limit for x in pd.to_datetime(sub_df[col_name_2])]
                        if res_1.count(True) > 0:
                            continue
                        res = pd.Series([x <= upper_limit_subset for x in pd.to_datetime(sub_df[col_name_2])])

                    if 0 < res.tolist().count(False) <= self.freq_contamination_level:
                        flagged_prefixes.append(v)
                        index_of_large = [x for x, y in zip(sub_df.index, res) if not y]
                        for i in index_of_large:
                            test_series[i] = False

                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'"{col_name_2}" contains very large values given the first word/prefix {flagged_prefixes} in '
                     f'"{col_name_1}" (Values that are large given any value of "{col_name_1}" are not flagged by '
                     f'this test.)'),
                    allow_patterns=False
                )


    def _generate_small_given_prefix(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('small_given_prefix rand',
                                    ['A-' + np.random.choice(list(string.ascii_letters))]*100 +
                                    ['B-' + np.random.choice(list(string.ascii_letters))]*100 +
                                    ['C-' + np.random.choice(list(string.ascii_letters))]*(self.num_synth_rows - 200))
        self._add_synthetic_column('small_given_prefix all', np.concatenate([
            np.random.randint(0, 100, 100),
            np.random.randint(100, 200, 100),
            np.random.randint(200, 300, (self.num_synth_rows - 200))
        ]))
        self._add_synthetic_column('small_given_prefix most', self.synth_df['small_given_prefix all'])
        self.synth_df.loc[999, 'small_given_prefix most'] = 8


    def _check_small_given_prefix(self, test_id):
        # todo: when show non-flagged, show with the flagged prefixes
        # todo: dont flag Null values

        # Determine if there are too many combinations to execute
        total_combinations = len(self.string_cols) * (len(self.numeric_cols) + len(self.date_cols))
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {(len(self.numeric_cols) + len(self.date_cols)):,} "
                       f"numeric and date columns, multiplied by {len(self.string_cols):,} "
                       f"string columns, results in {total_combinations:,} combinations. max_combinations "
                       f"is currently set to {self.max_combinations:,}.")
            return

        # Calculate and cache the lower limit based on q1 and q3 of each full numeric & date column
        lower_limits_dict = self.get_columns_iqr_lower_limit()
        nunique_dict = self.get_nunique_dict()

        for col_idx, col_name_1 in enumerate(self.string_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.string_cols) + len(self.binary_cols)} string and "
                       f"binary columns")

            col_vals = self.orig_df[col_name_1].astype(str).apply(replace_special_with_space)

            # Check the column's values are usually more than 1 word
            word_arr = col_vals.str.split() # todo: call self.get_word_counts_dict()
            word_counts_arr = pd.Series([len(x) for x in word_arr])
            if word_counts_arr.quantile(0.75) <= 1:
                continue

            first_words = pd.Series([x[0] if len(x) > 0 else "" for x in col_vals.str.split()])
            if first_words.nunique() > 10:
                continue
            vc = first_words.value_counts()
            common_values = []
            for v in vc.index:
                if (vc[v] > math.sqrt(self.num_rows)) and (vc[v] > 50):
                    common_values.append(v)

            for col_name_2 in self.numeric_cols + self.date_cols:
                if nunique_dict[col_name_2] < math.sqrt(self.num_rows):
                    continue
                test_series = [True] * self.num_rows
                flagged_prefixes = []
                for v in common_values:
                    sub_df = self.orig_df[[col_name_2]][first_words == v]
                    if col_name_2 in self.numeric_cols:
                        # Get the iqr given the full column
                        lower_limit, col_d1, col_q1 = lower_limits_dict[col_name_2]

                        # Get the iqr given the current subset
                        num_vals_no_null = pd.Series([float(x) for x in sub_df[col_name_2] if str(x).replace('-', '').replace('.', '').isdigit() ])
                        if len(num_vals_no_null) < 100:
                            continue
                        subset_med = num_vals_no_null.median()
                        num_vals = convert_to_numeric(sub_df[col_name_2], subset_med)
                        num_vals = num_vals.fillna(subset_med)

                        # We are only concerned in this test with subsets that tend to have larger values in the
                        # numeric column, and flag small values in this case. We do not flag values in subsets that
                        # have any values that are small relative to the subset; these are flagged by other tests.
                        res_1 = num_vals < lower_limit
                        if res_1.tolist().count(True) > 0:
                            continue

                        d1 = num_vals.quantile(0.1)
                        d9 = num_vals.quantile(0.9)
                        lower_limit_subset = d1 - (self.idr_limit * (d9 - d1))

                        # Ensure this subset is at least one decile shifted from the full column
                        if d1 < col_q1:
                            continue

                        res = pd.Series([x >= lower_limit_subset for x in num_vals])
                    else:
                        # Get the lower limit for the full column
                        lower_limit, col_d1, col_q1 = lower_limits_dict[col_name_2]
                        if lower_limit is None:
                            continue

                        res_1 = [x < lower_limit for x in pd.to_datetime(sub_df[col_name_2])]
                        if res_1.count(True) > 0:
                            continue

                        # Get the iqr given the current subset
                        q1 = pd.to_datetime(sub_df[col_name_2]).quantile(0.25, interpolation='midpoint')
                        q3 = pd.to_datetime(sub_df[col_name_2]).quantile(0.75, interpolation='midpoint')
                        try:
                            lower_limit_subset = q1 - (self.iqr_limit * (q3 - q1))
                        except Exception:
                            continue

                        res = pd.Series([x >= lower_limit_subset for x in pd.to_datetime(sub_df[col_name_2])])

                    if 0 < res.tolist().count(False) <= self.freq_contamination_level:
                        flagged_prefixes.append(v)
                        index_of_large = [x for x, y in zip(sub_df.index, res) if not y]
                        for i in index_of_large:
                            test_series[i] = False

                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    np.array(test_series),
                    (f'"{col_name_2}" contains very small values given the first word/prefix {flagged_prefixes} in '
                     f'"{col_name_1}" (Values that are small given any value of "{col_name_1}" are not flagged by this '
                     f'test.)'),
                    allow_patterns=False
                )


    def _generate_grouped_strings_by_numeric(self):
        """
        Patterns without exceptions: The values in 'num_grp_str all' are grouped (all the 'a' values, then all the 'b',
            and so on), when sorting by 'num_grp_str rand_i'
        Patterns with exception: Similar for columns  'num_grp_str most' and 'num_grp_str rand_i', with 1 exception.
        """
        group_a_len = group_b_len = self.num_synth_rows // 3
        group_c_len = self.num_synth_rows - (group_a_len + group_b_len)
        df = pd.DataFrame({
            'num_grp_str rand_i': sorted(np.random.random(self.num_synth_rows)),
            'num_grp_str rand_s': np.random.choice(['a', 'b', 'c'], self.num_synth_rows),
            'num_grp_str all': ['a'] * group_a_len + ['b'] * group_b_len + ['c'] * group_c_len,
            'num_grp_str most': ['a'] * group_a_len + ['b'] * group_b_len + ['c'] * (group_c_len - 1) + ['a']
        })
        df = df.sample(n=len(df), random_state=0)  # shuffle the dataframe
        df = df.reset_index(drop=True)
        for col_name in df.columns:
            self._add_synthetic_column(col_name, df[col_name])


    def _check_grouped_strings_by_numeric(self, test_id):
        """
        This is similar to GROUPED_STRINGS, but checks where the string/binary values are grouped when sorted by a
        numeric or date column, as opposed to the row order of the table.

        Handling null values: null values are treated as any other value and so if interspersed through the column
        will negate any grouping.
        """

        # Loop through the columns that we sort on.
        for col_idx, col_name_1 in enumerate(self.numeric_cols + self.date_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols) + len(self.date_cols)} numeric and "
                       f"date columns")

            if self.orig_df[col_name_1].nunique() < (self.num_valid_rows[col_name_1] - self.freq_contamination_level):
                continue

            # Loop through the columns that we check if they are grouped, when sorted by col_name_1
            df = None
            for col_name_2 in self.string_cols + self.binary_cols:
                if self.orig_df[col_name_2].nunique() > math.sqrt(self.num_valid_rows[col_name_2]):
                    continue
                if df is None:
                    df = self.orig_df.copy()
                    if col_name_1 in self.numeric_cols:
                        sort_order = self.numeric_vals_filled[col_name_1].sort_values().index
                    else:
                        sort_order = self.orig_df[col_name_1].sort_values().index
                    df = df.loc[sort_order]
                col_series = df[col_name_2]
                self._check_grouped_strings_column(test_id, col_name_1, col_name_2, col_series)

    ##################################################################################################################
    # Data consistency checks for two string and one numeric column
    ##################################################################################################################


    def _generate_large_given_pair(self):
        """
        Patterns without exceptions: None. This test does not generate patterns.
        Patterns with exception: Row 998 contains a value in 'large_given_pair most' which is common, but not for
            'd', 'b' in the two string columns. As well, Row 998 contains a a value for 'large_given_pair date_most'
            that is common for the column, but not with 'd' and 'b' in the two string columns.
        Values not flagged: Row 999 contains a rare combination of values, so it is not possible to check for
            rare numeric values given these.
        """
        common_vals = [
            ['a', 'b'],
            ['b', 'a'],
            ['a', 'c'],
            ['d', 'b']]
        rare_vals = [['b', 'c']]

        data = np.array([x*(self.num_synth_rows // len(common_vals)) for x in common_vals]).reshape(-1, 2)
        data = np.vstack([data[:self.num_synth_rows-1], rare_vals])
        self._add_synthetic_column('large_given_pair rand_a', data[:, 0])
        self._add_synthetic_column('large_given_pair rand_b', data[:, 1])
        self._add_synthetic_column('large_given_pair all',
            np.concatenate([
                np.random.randint(250, 300, 100),
                np.random.randint(100, 200, 100),
                np.random.randint(0, 40, (self.num_synth_rows - 200))
            ]))
        self._add_synthetic_column('large_given_pair most', self.synth_df['large_given_pair all'])
        self.synth_df.loc[998, 'large_given_pair most'] = 290

        test_date1 = datetime.datetime.strptime("01-7-2016", "%d-%m-%Y")
        test_date2 = datetime.datetime.strptime("01-7-2015", "%d-%m-%Y")
        test_date3 = datetime.datetime.strptime("01-7-2014", "%d-%m-%Y")
        self._add_synthetic_column('large_given_pair date_most',
            np.concatenate([
                [test_date1 + relativedelta(days=np.random.randint(1, 300)) for _ in range(200)],
                [test_date2 + relativedelta(days=np.random.randint(1, 300)) for _ in range(200)],
                [test_date3 + relativedelta(days=np.random.randint(1, 300)) for _ in range(self.num_synth_rows - 400)]
            ]))
        self.synth_df.loc[998, 'large_given_pair date_most'] = datetime.datetime.strptime("01-8-2018", "%d-%m-%Y")


    def _check_large_given_pair(self, test_id):
        """
        As this test examines many subsets, it sets the threshold for large values based on 2.0 * self.iqr_limit.

        This considers only subsets that have smaller values than normal for the column.

        Handling null values: with many Null values, the count of each pair of values in the 2 string columns may be
        too low to execute this test.
        """

        if len(self.string_cols) < 2:
            return
        if (len(self.numeric_cols) + len(self.date_cols)) < 1:
            return

        # Calculate and cache the upper limit based on q1 and q3 of each full numeric & date column
        upper_limits_dict = self.get_columns_iqr_upper_limit()

        # Get the set of common values in each string column
        common_vals_dict = self.get_common_values_dict()
        avg_num_common_vals = statistics.mean([len(x) for x in common_vals_dict.values()])

        # Determine if there are too many combinations to execute
        num_pairs, pairs = self._get_string_column_pairs_unique()  # todo: we should check the binary columns as well
        if pairs is None:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                      f"max_combinations is currently set to {self.max_combinations:,}.")
            return
        total_combinations = num_pairs * avg_num_common_vals * avg_num_common_vals * (len(self.numeric_cols) + len(self.date_cols))
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {int(total_combinations):,} combinations given the number"
                       f"of columns and unique values. max_combinations is currently set to {self.max_combinations:,}")
            return

        # Save the subset dataframe for each pair of values
        subsets_dict = {}
        store_index_only = len(self.date_cols) == 0
        for col_name_1, col_name_2 in pairs:
            for v1 in common_vals_dict[col_name_1]:
                for v2 in common_vals_dict[col_name_2]:
                    sub_df = self.orig_df[(self.orig_df[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]
                    if store_index_only:
                        subsets_dict[(col_name_1, col_name_2, v1, v2)] = sub_df.index
                    else:
                        subsets_dict[(col_name_1, col_name_2, v1, v2)] = sub_df

        # Loop through each pair of string columns, and for each pair: each pair of values, and each numeric/date column
        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 50 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(pairs):,} pairs of string columns")

            for col_name_3 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                col_upper_limit, col_q2, col_q3 = upper_limits_dict[col_name_3]
                if col_name_3 in self.numeric_cols:
                    col_q2_limit = col_q2 * 0.8
                    col_q3_limit = col_q3 * 0.8
                else:
                    col_q2_limit = col_q2
                    col_q3_limit = col_q3

                if col_q2_limit is None or col_q3_limit is None:
                    continue

                # found_many is set to True if we find any subset where many rows are flagged. In this case, we end
                # early, as exceptions are only flagged if they are few.
                found_many = False

                for v1 in common_vals_dict[col_name_1]:
                    if found_many:
                        break
                    for v2 in common_vals_dict[col_name_2]:
                        if found_many:
                            break

                        sub_df = subsets_dict[(col_name_1, col_name_2, v1, v2)]
                        if (len(sub_df) < 200) or (len(sub_df) < (self.num_rows * 0.05)):
                            continue

                        # Test the distribution relative to the full subset on a sample, if the subset is large
                        if (len(sub_df) > 2000) and (col_name_3 in self.numeric_cols):
                            if store_index_only:
                                sample_indexes = np.random.choice(sub_df, 50, replace=True)
                            else:
                                sample_indexes = sub_df.sample(n=50).index
                            num_vals = self.numeric_vals_filled[col_name_3].loc[sample_indexes]
                            q2, q3 = num_vals.quantile([0.5, 0.75])
                            if (q2 > col_q2_limit) or (q3 > col_q3_limit):
                                continue

                        if col_name_3 in self.numeric_cols:
                            if store_index_only:
                                num_vals = self.numeric_vals_filled[col_name_3].loc[sub_df]
                            else:
                                num_vals = self.numeric_vals_filled[col_name_3].loc[sub_df.index]
                            q1, q2, q3 = num_vals.quantile([0.25, 0.50, 0.75], interpolation='midpoint')
                            if q2 is None or q2 > col_q2_limit:
                                continue
                            if q3 is None or q3 > col_q3_limit:
                                continue
                            if q1 is None:
                                continue
                        else:
                            q2 = pd.to_datetime(sub_df[col_name_3]).quantile(0.50, interpolation='midpoint')
                            if q2 is None or q2 > col_q2_limit:
                                continue
                            q3 = pd.to_datetime(sub_df[col_name_3]).quantile(0.75, interpolation='midpoint')
                            if q3 is None or q3 > col_q3_limit:
                                continue
                            q1 = pd.to_datetime(sub_df[col_name_3]).quantile(0.25, interpolation='midpoint')
                            if q1 is None:
                                continue

                        try:
                            upper_limit = q3 + (self.iqr_limit * 2.0 * (q3 - q1))
                        except Exception:  # Date values can exceed limits
                            continue

                        if col_name_3 in self.numeric_cols:
                            res = num_vals <= upper_limit
                        else:
                            res = sub_df[col_name_3] <= upper_limit

                        if res.tolist().count(False) > self.freq_contamination_level:
                            found_many = True
                            break

                        if 0 < res.tolist().count(False) <= self.freq_contamination_level:
                            if store_index_only:
                                index_of_large = [x for x, y in zip(sub_df, res) if not y]
                            else:
                                index_of_large = [x for x, y in zip(sub_df.index, res) if not y]
                            for i in index_of_large:
                                test_series[i] = False

                test_series = test_series | \
                              self.orig_df[col_name_1].isna() | \
                              self.orig_df[col_name_2].isna() | \
                              self.orig_df[col_name_3].isna()

                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2, col_name_3],
                    test_series,
                    (f'"{col_name_3}" contains very large values given the values in "{col_name_1}" and '
                     f'"{col_name_2}"'),
                    allow_patterns=False
                )


    def _generate_small_given_pair(self):
        """
        Patterns without exceptions: None. This test does not generate patterns.
        Patterns with exception: Row 998 contains a value in 'small_given_pair most' which is common, but not for
            'd', 'b' in the two string columns. As well, Row 998 contains a a value for 'small_given_pair date_most'
                that is common for the column, but not with 'd' and 'b' in the two string columns.
        Values not flagged: Row 999 contains a rare combination of values, so it is not possible to check for
            rare numeric values given these.
        """
        common_vals = [
            ['a', 'a'],
            ['a', 'b'],
            ['b', 'a'],
            ['d', 'b']]
        rare_vals = [['b', 'c']]

        data = np.array([x*(self.num_synth_rows // len(common_vals)) for x in common_vals]).reshape(-1, 2)
        data = np.vstack([data[:self.num_synth_rows-1], rare_vals])
        self._add_synthetic_column('small_given_pair rand_a', data[:, 0])
        self._add_synthetic_column('small_given_pair rand_b', data[:, 1])
        self._add_synthetic_column('small_given_pair all', np.concatenate([
            np.random.randint(0, 50, 300),
            np.random.randint(100, 200, 300),
            np.random.randint(250, 300, (self.num_synth_rows - 600))
        ]))
        self._add_synthetic_column('small_given_pair most', self.synth_df['small_given_pair all'])
        self.synth_df.loc[998, 'small_given_pair most'] = 4

        test_date1 = datetime.datetime.strptime("01-7-2014", "%d-%m-%Y")
        test_date2 = datetime.datetime.strptime("01-7-2015", "%d-%m-%Y")
        test_date3 = datetime.datetime.strptime("01-7-2016", "%d-%m-%Y")
        self._add_synthetic_column('small_given_pair date_most',
            np.concatenate([
                [test_date1 + relativedelta(days=np.random.randint(1, 100)) for _ in range(300)],
                [test_date2 + relativedelta(days=np.random.randint(1, 100)) for _ in range(300)],
                [test_date3 + relativedelta(days=np.random.randint(1, 100)) for _ in range(self.num_synth_rows - 600)]
            ]))
        self.synth_df.loc[998, 'small_given_pair date_most'] = datetime.datetime.strptime("01-8-2014", "%d-%m-%Y")


    def _check_small_given_pair(self, test_id):
        """
        As this test examines many subsets, it sets the threshold for large values based on 2.0 * self.idr_limit.

        This considers only subsets that have larger values than normal for the column.

        With many Null values, the count of each pair of values in the 2 string columns may be too low to execute this
        test.
        """

        if len(self.string_cols) < 2:
            return
        if (len(self.numeric_cols) + len(self.date_cols)) < 1:
            return

        # Calculate and cache the lower limit based on q1 and q3 of each full numeric & date column
        lower_limits_dict = self.get_columns_iqr_lower_limit()

        # Get the set of common values in each string column
        common_vals_dict = self.get_common_values_dict()
        avg_num_common_vals = statistics.mean([len(x) for x in common_vals_dict.values()])

        # Determine if there are too many combinations to execute
        num_pairs, pairs = self._get_string_column_pairs_unique()  # todo: we should check the binary columns as well
        if pairs is None:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of string columns. "
                      f"max_combinations is currently set to {self.max_combinations:,}.")
            return
        total_combinations = num_pairs * avg_num_common_vals * avg_num_common_vals * (len(self.numeric_cols) + len(self.date_cols))
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {int(total_combinations):,} combinations given the number"
                       f"of columns and unique values. max_combinations is currently set to {self.max_combinations:,}")
            return

        # Save the subset for each pair of values
        subsets_dict = {}
        store_index_only = len(self.date_cols) == 0
        for col_name_1, col_name_2 in pairs:
            for v1 in common_vals_dict[col_name_1]:
                for v2 in common_vals_dict[col_name_2]:
                    sub_df = self.orig_df[(self.orig_df[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]
                    if store_index_only:
                        subsets_dict[(col_name_1, col_name_2, v1, v2)] = sub_df.index
                    else:
                        subsets_dict[(col_name_1, col_name_2, v1, v2)] = sub_df

        # Loop through each pair of string columns, and for each pair: each pair of values, and each numeric/date column
        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 50 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(pairs):,} pairs of string columns")

            for col_name_3 in self.numeric_cols + self.date_cols:
                test_series = [True] * self.num_rows
                if col_name_3 in self.numeric_cols:
                    _, col_d1, col_q1 = lower_limits_dict[col_name_3]
                    if col_d1 is None or col_q1 is None:
                        continue
                    col_d1_limit = col_d1 * 1.1
                    col_q1_limit = col_q1 * 1.1
                else:
                    _, col_q1, col_q3 = lower_limits_dict[col_name_3]
                    if col_q1 is None or col_q3 is None:
                        continue
                    col_q1_limit = col_q1
                    col_q3_limit = col_q3

                # found_many is set to True if we find any subset where many rows are flagged. In this case, we end
                # early, as exceptions are only flagged if they are few.
                found_many = False

                for v1 in common_vals_dict[col_name_1]:
                    if found_many:
                        break
                    for v2 in common_vals_dict[col_name_2]:
                        if found_many:
                            break

                        sub_df = subsets_dict[(col_name_1, col_name_2, v1, v2)]
                        if (len(sub_df) < 200) or (len(sub_df) < (self.num_rows * 0.05)):
                            continue

                        if col_name_3 in self.numeric_cols:
                            if store_index_only:
                                num_vals = self.numeric_vals_filled[col_name_3].loc[sub_df]
                            else:
                                num_vals = self.numeric_vals_filled[col_name_3].loc[sub_df.index]
                            d1, q1, q2, q3 = num_vals.quantile([0.1, 0.25, 0.50, 0.75], interpolation='midpoint')
                            if d1 is None or q1 is None or q2 is None or q3 is None:
                                continue
                            if d1 < col_d1_limit:
                                continue
                            if q1 < col_q1_limit:
                                continue
                        else:
                            q1 = pd.to_datetime(sub_df[col_name_3]).quantile(0.25, interpolation='midpoint')
                            if q1 is None or q1 < col_q1_limit:
                                continue
                            q3 = pd.to_datetime(sub_df[col_name_3]).quantile(0.75, interpolation='midpoint')
                            if q3 is None or q3 < col_q3_limit:
                                continue

                        try:
                            lower_limit = q1 - (self.iqr_limit * 2.0 * (q3 - q1))
                        except Exception:  # Date values can exceed limits
                            continue

                        if col_name_3 in self.numeric_cols:
                            res = num_vals >= lower_limit
                        else:
                            res = sub_df[col_name_3] >= lower_limit

                        if res.tolist().count(False) > self.freq_contamination_level:
                            found_many = True
                            break

                        if 0 < res.tolist().count(False) <= self.freq_contamination_level:
                            if store_index_only:
                                index_of_small = [x for x, y in zip(sub_df, res) if not y]
                            else:
                                index_of_small = [x for x, y in zip(sub_df.index, res) if not y]
                            for i in index_of_small:
                                test_series[i] = False

                test_series = test_series | \
                              self.orig_df[col_name_1].isna() | \
                              self.orig_df[col_name_2].isna() | \
                              self.orig_df[col_name_3].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2, col_name_3],
                    test_series,
                    (f'"{col_name_3}" contains very small values given the values in "{col_name_1}" and '
                     f'"{col_name_2}"'),
                    allow_patterns=False
                )

    ##################################################################################################################
    # Data consistency checks for one string/binary column and two numeric
    ##################################################################################################################


    def _generate_corr_given_val(self):
        """
        Patterns without exceptions: 'corr_given_val rand_a' and 'corr_given_val rand_b' are correlated when
            conditioning on 'corr_given_val rand_string'. When the string column as value 'A', then the 2 numeric
            columns are positively correlated, and with value 'B', they are negatively correlated. They are not
            correlated if not considering column 'corr_given_val rand_string'.
        Patterns with exception: 'corr_given_val rand_a' and 'corr_given_val most' have almost the same relationship
            with an exception in row 499 (the last row of list_b).
        """
        list_a = sorted([random.randint(1, 1_000) for _ in range(500)])
        list_b = sorted([random.randint(1, 2_000) for _ in range(500)])
        c = list(zip(list_a, list_b))
        random.shuffle(c)
        list_a, list_b = zip(*c)

        list_c = sorted([random.randint(1, 1_000) for _ in range(self.num_synth_rows - 500)])
        list_d = sorted([random.randint(1, 2_000) for _ in range(self.num_synth_rows - 500)], reverse=True)
        c = list(zip(list_c, list_d))
        random.shuffle(c)
        list_c, list_d = zip(*c)

        self._add_synthetic_column('corr_given_val rand_string', [None]*500 + ['B']*(self.num_synth_rows - 500))
        self._add_synthetic_column('corr_given_val rand_a', list_a + list_c)
        self._add_synthetic_column('corr_given_val rand_b', list_b + list_d)
        list_b = list(list_b)
        list_b[-1] = list_b[-1] / 10.0
        self._add_synthetic_column('corr_given_val most', list(list_b) + list(list_d))


    def _check_corr_given_val(self, test_id):

        # From: https://stackoverflow.com/questions/71844846/is-there-a-faster-way-to-get-correlation-coefficents
        # Modified to check for all-zero arrays
        def pairwise_correlation(a, b):
            am = a - np.mean(a, axis=0)
            bm = b - np.mean(b, axis=0)
            if all(am == 0.0) or all(bm == 0.0):
                return 0.0
            return am.T @ bm / (np.sqrt(
                np.sum(am**2, axis=0)).T * np.sqrt(
                np.sum(bm**2, axis=0)))

        if len(self.numeric_cols) < 2:
            return

        nunique_dict = self.get_nunique_dict()

        # Create a numpy array of just the numeric columns for efficiency
        numeric_df = None
        for col_name in self.numeric_cols:
            if numeric_df is None:
                numeric_df = self.numeric_vals_filled[col_name]
            else:
                numeric_df = pd.concat([numeric_df, self.numeric_vals_filled[col_name]], axis=1)
        numeric_np = numeric_df.values

        # Create a sample. We do not use self.sample_df, as it may have Nulls removed, and we do not wish to remove
        # rows where the conditioning column has Null values
        sample_df = self.orig_df.sample(n=50)

        # Create a sample array similarly
        numeric_sample_df = None
        for col_name in self.numeric_cols:
            if numeric_sample_df is None:
                numeric_sample_df = convert_to_numeric(sample_df[col_name], self.column_medians[col_name])
            else:
                numeric_sample_df = pd.concat([numeric_sample_df, convert_to_numeric(self.sample_df[col_name], self.column_medians[col_name])], axis=1)
        numeric_sample_df.columns = self.numeric_cols

        # Determine if there are too many combinations to execute
        num_pairs, numeric_pairs = self._get_numeric_column_pairs_unique()
        total_combinations = num_pairs * (len(self.string_cols) + len(self.binary_cols))
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {(len(self.string_cols) + len(self.binary_cols)):,} "
                       f"string and binary columns, multiplied by {num_pairs:,} pairs of numeric columns, results in "
                       f"{total_combinations:,} combinations. max_combinations is currently set to "
                       f"{self.max_combinations:,}")
            return

        # The string and binary columns are the columns we condition on to determine if the 2 numeric or date columns
        # are correlated when holding the values in the string/binary column constant.
        for col_idx, col_cond in enumerate(self.string_cols + self.binary_cols):
            if self.verbose >= 2:
                print(f"  Examining column {col_idx} of {len(self.string_cols + self.binary_cols)} string and binary "
                       f"columns")
            vals = self.orig_df[col_cond].unique()
            if len(vals) > 10:
                continue

            # Determine and cache subsets of numeric_np and numeric_sample_df for each value in the string/binary column
            conditioning_vals = []
            val_idxs_dict = {}
            for val in vals:
                if (val is None) or (val != val):
                    idxs = np.where(self.orig_df[col_cond].isna())[0]
                else:
                    idxs = np.where(self.orig_df[col_cond] == val)[0]
                if len(idxs) < 100:
                    continue
                conditioning_vals.append(val)
                val_idxs_dict[val] = idxs

            for num_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs):
                if self.verbose >= 2 and num_idx > 0 and num_idx % 10_000 == 0:
                    print(f"  Examining column set {num_idx:,} of {num_pairs:,} pairs of numeric columns.")

                if nunique_dict[col_name_1] < 10:
                    continue
                if nunique_dict[col_name_2] < 10:
                    continue

                # Get the column indexes of the 2 numeric columns
                col_idx_1 = self.numeric_cols.index(col_name_1)
                col_idx_2 = self.numeric_cols.index(col_name_2)

                # Skip any pairs of columns that are correlated even if not conditioning on another column
                try:
                    corr = pairwise_correlation(numeric_sample_df[col_name_1], numeric_sample_df[col_name_2])
                except Exception as e:
                    if self.DEBUG_MSG:
                        if colored:
                            print(colored(f"Error calculating correlation in {test_id}: {e}", 'red'))
                        else:
                            print(f"Error calculating correlation in {test_id}: {e}")
                    continue
                if abs(corr) >= 0.75:
                    continue

                # We ensure at least one subset was large enough to test and was correlated, and that there aren't
                # any subsets that are large enough, but uncorrelated.
                any_subsets_uncorrelated = False
                some_subset_correlated = False
                test_series = [True] * self.num_rows
                for val in conditioning_vals:
                    idxs = val_idxs_dict[val]
                    sub_np = numeric_np[idxs]

                    # First test on a sample of rows
                    sample_sub_np = sub_np[:50]
                    corr = pairwise_correlation(sample_sub_np[:, col_idx_1], sample_sub_np[:, col_idx_2])
                    if abs(corr) < 0.95:
                        any_subsets_uncorrelated = True
                        break

                    # Test with the full columns
                    corr = pairwise_correlation(sub_np[:, col_idx_1], sub_np[:, col_idx_2])
                    if abs(corr) < 0.95:
                        any_subsets_uncorrelated = True
                        break

                    # pairwise_correlation() is fast, but does not distinguish positive from negative correlation
                    if (val is None) or (val != val):
                        sub_df = self.orig_df[self.orig_df[col_cond].isna()]
                    else:
                        sub_df = self.orig_df[self.orig_df[col_cond] == val]
                    spearancorr = sub_df[col_name_1].corr(sub_df[col_name_2], method='spearman')
                    if abs(spearancorr) < 0.95:
                        any_subsets_uncorrelated = True
                        break
                    some_subset_correlated = True

                    col_1_percentiles = sub_df[col_name_1].rank(pct=True)
                    col_2_percentiles = sub_df[col_name_2].rank(pct=True)

                    if spearancorr > 0:
                        subset_series = np.array([abs(x-y) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                    else:
                        subset_series = np.array([abs(x-(1.0 - y)) < 0.1 for x, y in zip(col_1_percentiles, col_2_percentiles)])

                    for i_idx, i in enumerate(sub_df.index):
                        test_series[i] = subset_series[i_idx]

                if not any_subsets_uncorrelated and some_subset_correlated:
                    self._process_analysis_binary(
                        test_id,
                        [col_cond, col_name_1, col_name_2],
                        test_series,
                        (f'"{col_name_1}" is consistently correlated with "{col_name_2}" when conditioning on '
                         f'"{col_cond}"'))

    ##################################################################################################################
    # Data consistency checks for non-numeric columns in relation to all other columns
    ##################################################################################################################


    def _generate_dt_classifier(self):
        """
        Patterns without exceptions: 'dt cls. 2' may be predicted from 'dt cls. 1a' and 'dt cls. 1b', and optionally
            'dt cls. 3'
        Patterns with exception: 'dt cls. 3' may be predicted from 'dt cls. 1a' and 'dt cls. 1b', and optionally
            'dt cls. 2', with exceptions.
        """
        self._add_synthetic_column('dt cls. 1a', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('dt cls. 1b', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        arr = []
        for i in range(self.num_synth_rows):
            if self.synth_df['dt cls. 1a'][i] > 50:
                if self.synth_df['dt cls. 1b'][i] > 50:
                    arr.append('A')
                else:
                    arr.append('B')
            else:
                if self.synth_df['dt cls. 1b'][i] > 50:
                    arr.append('C')
                else:
                    arr.append('D')
        self._add_synthetic_column('dt cls. 2', arr)
        self._add_synthetic_column('dt cls. 3', self.synth_df['dt cls. 2'])
        if self.synth_df.loc[999, 'dt cls. 3'] == 'A':
            self.synth_df.at[999, 'dt cls. 3'] = 'B'
        else:
            self.synth_df.at[999, 'dt cls. 3'] = 'A'


    def _check_dt_classifier(self, test_id):
        """"
        Very similar to the test using a regression decision tree, but used where the target column is string or binary.
        The accuracy is measured in terms f1_score.
        """
        # todo: if there are many values, group the rare ones into "other"
        cols_same_bool_dict = self.get_cols_same_bool_dict()

        # Set the seed to ensure the DT behaves the same each execution of this test
        random.seed(0)
        np.random.seed(0)

        drop_features = []
        categorical_features = []
        for col_name in self.orig_df.columns:
            if not pandas_types.is_numeric_dtype(self.orig_df[col_name]):
                if self.orig_df[col_name].nunique() > 5:
                    drop_features.append(col_name)
                else:
                    categorical_features.append(col_name)

        for col_idx, col_name in enumerate(self.string_cols + self.binary_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 100 == 0:
                print(f"  Examining column {col_idx:,} of {(len(self.string_cols) + len(self.binary_cols)):,} "
                      f"string and binary columns.")

            # Skip columns where one value dominates and a trivial decision tree could predict well
            vc = self.orig_df[col_name].value_counts(normalize=True)
            if len(vc) == 0:
                continue
            if vc.iloc[0] > 0.95:
                continue

            if col_name in drop_features:
                continue
            clf = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0)
            x_data = self.orig_df.drop(columns=set(drop_features + [col_name]))

            # Remove any columns that are almost the same as the target column
            same_cols = []
            for c in x_data.columns:
                pair_tuple = tuple(sorted([col_name, c]))
                # The pair of columns may not be in cols_same_bool_dict if they are of different types, in which case
                # we know they are not the same values.
                if (pair_tuple in cols_same_bool_dict) and cols_same_bool_dict[pair_tuple]:
                    same_cols.append(c)
            x_data = x_data.drop(columns=same_cols)

            if len(x_data.columns) == 0:
                continue

            # One-hot encoded any categorical features
            use_categorical_features = categorical_features.copy()
            if col_name in use_categorical_features:
                use_categorical_features.remove(col_name)
            use_categorical_features = [x for x in use_categorical_features if x in x_data.columns]
            x_data = pd.get_dummies(x_data, columns=use_categorical_features)
            for c in self.orig_df.columns:
                if c in x_data.columns:
                    x_data[c] = x_data[c].replace([np.inf, -np.inf, np.nan], self.orig_df[c].median())

            # Ensure the target column is clean
            y = self.orig_df[col_name]
            y = y.replace([np.inf, -np.inf, np.nan], statistics.mode(y))
            y = y.astype(str)

            # Fit a model, get the predictions, and the accuracy
            clf.fit(x_data, y)
            y_pred = clf.predict(x_data)
            if pd.Series(y_pred).isna().sum() > (len(y_pred) * 0.75):
                continue
            f1_dt = metrics.f1_score(y, y_pred, average='macro')
            ft_naive = metrics.f1_score(y, [statistics.mode(y)] * len(y), average='macro')

            # If the model is accurate, format the tree description and save the pattern
            if (f1_dt > 0.8) and (f1_dt > (ft_naive * 1.5)):
                rules = tree.export_text(clf)

                # Clean the rules to use the original feature names. We go through in reverse order so we don't
                # have, for example, feature_1 matching feature_11
                cols = []
                for c_idx, c_name in reversed(list(enumerate(x_data.columns))):
                    rule_col_name = f'feature_{c_idx}'
                    if rule_col_name in rules:
                        orig_col = c_name
                        for cat_col in categorical_features:
                            if c_name.startswith(cat_col):
                                orig_col = cat_col
                        cols.append(orig_col)
                        rules = rules.replace(rule_col_name, c_name)

                # Some columns may be included multiple times. Put the columns used into a consistent, list without
                # duplicates
                cols = sorted(set(cols))

                # Clean the split points for categorical features to use the values, not 0.5
                rules = self.get_decision_tree_rules_as_categories(rules, categorical_features)

                test_series = (y == y_pred)
                self._process_analysis_binary(
                    test_id,
                    cols + [col_name],
                    test_series,
                    f'The values in column "{col_name}" are consistently predictable from {cols} based using a decision '
                    f'tree with the following rules: \n{rules}',
                    display_info={'Pred': pd.Series(y_pred)}
                )

    ##################################################################################################################
    # Data consistency checks for sets of three columns of any type
    ##################################################################################################################


