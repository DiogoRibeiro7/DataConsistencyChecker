"""
NumericTestsMixin: Test methods for numeric tests.

This mixin contains 118 test methods organized by category.
Extracted from check_data_consistency.py for better code organization.
"""

from __future__ import annotations

import copy
import datetime
import math
import random
import statistics
import string
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
from dateutil.relativedelta import relativedelta
from sklearn import metrics, tree
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from termcolor import colored
except ImportError:  # pragma: no cover - optional presentation dependency
    colored = None

from data_consistency_checker.checker_state import CheckerState
from data_consistency_checker.checker_utils import (
    array_to_str,
    as_str,
    convert_to_numeric,
    get_num_decimal_digits,
    is_missing,
    safe_div,
)

digits = string.digits


class NumericTestsMixin(CheckerState):
    """
    Mixin class containing numeric tests methods.

    This class contains 118 methods for testing data consistency
    related to numeric operations.

    This is a mixin class designed to be used with multiple inheritance.
    It does not have an __init__ method and relies on the parent class
    to provide necessary attributes and methods.
    """

    def _check_two_cols_larger(self, test_id, require_same_scale):
        """
        Used by __check_larger() and __check_larger_same_range()
        """

        num_pairs, col_pairs = self._get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
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

        for _cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            test_series = larger_dict[(col_name_1, col_name_2)]
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


    def _generate_positive_values(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('positive rand', [random.random() - 0.5 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('positive all',  [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('positive most', [random.random() for _ in range(self.num_synth_rows-1)] + [-0.5])


    def _check_positive_values(self, test_id):
        for col_name in self.numeric_cols:
            if self.orig_df[col_name].notna().sum() < self.freq_contamination_level:
                continue
            vals_arr = convert_to_numeric(self.orig_df[col_name], 1)
            test_series = (self.orig_df[col_name].isna()) | (vals_arr >= 0)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column contains consistently positive values")


    def _generate_negative_values(self):
        """
        Patterns without exceptions: 'negative all' consistently contains negative values
        Patterns with exception: 'negative most' consistently contains negative values with one exception.
        """
        self._add_synthetic_column('negative rand', [-random.random() + 0.5 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('negative all', [-random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('negative most', [-random.random() for _ in range(self.num_synth_rows-1)] + [0.5])


    def _check_negative_values(self, test_id):
        for col_name in self.numeric_cols:
            if self.orig_df[col_name].notna().sum() < self.freq_contamination_level:
                continue
            vals_arr = convert_to_numeric(self.orig_df[col_name], -1)
            test_series_neg = (vals_arr < 0)
            if test_series_neg.tolist().count(False) > (self.num_rows * 0.75):
                continue
            test_series = (self.orig_df[col_name].isna()) | (vals_arr <= 0)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"The column contains consistently zero or negative values (with "
                 f"{(vals_arr == 0).tolist().count(True)} zero values and {self.orig_df[col_name].isna().sum()} null "
                 f"values)")
            )


    def _generate_number_decimals(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('num_decimals rand', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('num_decimals all',  [round(random.random(), 2) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('num_decimals most',
                                    [round(random.random(), 2) for _ in range(self.num_synth_rows-2)] + [0.5665] + [0.00083838383334445])


    def _check_number_decimals(self, test_id):
        for col_name in self.numeric_cols:

            # Test first on a sample, unless the column is entirely null within the sample
            if self.sample_df[col_name].isna().sum() < len(self.sample_df):
                vals_arr = convert_to_numeric(self.sample_df[col_name], 1)
                num_digits_series = vals_arr.apply(get_num_decimal_digits)
                # convert_to_numeric() fills the missing values, so these are removed using the original values
                num_digits_non_null_series = num_digits_series[self.sample_df[col_name].notna().to_numpy()]

                counts_series = num_digits_non_null_series.value_counts(normalize=False)
                most_common_num_digits = counts_series.sort_values().index[-1]
                rare_num_digits = [x for x in counts_series.index if (x > (most_common_num_digits * 1.2)) and (x >= (most_common_num_digits + 2))]

                test_series = [x not in rare_num_digits for x in num_digits_series]
                if test_series.count(False) > 1:
                    continue
            else:
                pass  # todo: test with columns 90% null. In this case, get another sample, of just this column

            # Test on the full column
            vals_arr = convert_to_numeric(self.orig_df[col_name], 1)
            num_digits_series = vals_arr.apply(get_num_decimal_digits)
            num_digits_non_null_series = num_digits_series[self.orig_df[col_name].notna().to_numpy()]

            counts_series = num_digits_non_null_series.value_counts(normalize=False)
            if len(counts_series) == 0:
                continue
            most_common_num_digits = counts_series.sort_values().index[-1]
            rare_num_digits = [x for x in counts_series.index if (x > (most_common_num_digits * 1.2)) and (x >= (most_common_num_digits + 2))]

            # An alternative formulation for rare numbers of digits
            # [x for x, y in zip(counts_series.sort_values(ascending=True).index,
            #   counts_series.sort_values(ascending=True).values) if y <= self.freq_contamination_level]

            common_num_digits = [x for x in counts_series.index if x <= most_common_num_digits]
            test_series = [x not in rare_num_digits for x in num_digits_series]

            # We just flag values that have significantly more decimals than normal. Where values have less
            # than normal, this may simply be that there are zeros in the least significant positions.
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The column contains values consistently with {array_to_str(common_num_digits)} decimals',
                allow_patterns=((len(common_num_digits) > 1) or (common_num_digits[0] != 0))
            )
            # todo: we need to display the decimals better, likely converting to str; otherwise only a few decimals are shown.


    def _generate_rare_decimals(self):
        """
        Patterns without exceptions: 'rare_decimals_all' consistently has values ending in a small set of values after
            the decimal point.
        Patterns with exception: 'most' consistently has values ending in a small set of values after
            the decimal point, with 1 exception.
        """
        self._add_synthetic_column('rare_decimals rand',
                                    [np.random.random() * 1000.0 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('rare_decimals all',
                                    [np.random.randint(1, 100) + np.random.choice([0.49, 0.98, 0.99])
                                     for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('rare_decimals most',
                                    [np.random.randint(1, 100) + np.random.choice([0.49, 0.98, 0.99])
                                     for _ in range(self.num_synth_rows - 1)] + [5.74])


    def _check_rare_decimals(self, test_id):
        for col_name in self.numeric_cols:
            vals_arr = self.numeric_vals[col_name]  # This has the non-numeric values removed, but may contain NaNs.
            vals_arr = vals_arr.dropna()
            if len(vals_arr) == 0:
                continue

            # The format_float_positional function ensures the strings are not in scientific notation
            vals_arr = pd.Series([x[1] for x in as_str(vals_arr.apply(np.format_float_positional)).str.split('.')])
            vc = vals_arr.value_counts()
            common_values = [x for x, y in zip(vc.index, vc.values) if y > (self.num_rows * 0.01)]
            if len(common_values) > 10:
                continue
            if len(common_values) == 0:
                continue
            if set(common_values) == {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                continue
            if set(common_values) == {'', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                continue
            test_series = [
                True if y else x[1] in common_values
                for x, y in zip(as_str(self.numeric_vals_filled[col_name].apply(np.format_float_positional)).str.split('.'),
                self.orig_df[col_name].isna())]
            if len(common_values) == 1:
                common_values_str = str(common_values)[1:-1].replace("\'\'", '0')
            else:
                common_values_str = "one of " + str(common_values)[1:-1].replace("\'\'", '0')
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f"The column consistently contains values with {common_values_str} after the decimal point",
                allow_patterns=(len(common_values) > 1) or (common_values[0] != "")
            )


    def _generate_column_increasing(self):
        """
        Patterns without exceptions: 'col_asc all' is consistently increasing
        Patterns with exception: 'col_asc most' is consistently increasing, with one exception. As well,
            'col_asc date_most' is consistently increasing, with one exception.
        """
        self._add_synthetic_column('col_asc rand', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('col_asc all', sorted([round(random.random(), 2) for _ in range(self.num_synth_rows)]))
        self._add_synthetic_column('col_asc most', sorted([round(random.random(), 2) for _ in range(self.num_synth_rows-1)]) + [0.0001])

        test_date = datetime.datetime.strptime("1-1-2000", "%d-%m-%Y")
        self._add_synthetic_column('col_asc date_most', pd.date_range(test_date, periods=self.num_synth_rows, freq='D'))
        self.synth_df.loc[999, 'col_asc date_most'] = datetime.datetime.strptime("1-1-2000", "%d-%m-%Y")


    def _check_column_increasing(self, test_id):
        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique() < 3:
                continue
            if col_name in self.numeric_cols:
                # Check if there are any invalid numeric values
                if len(self.numeric_vals[col_name]) < self.num_rows:
                    continue

                # Compare each value to the previous non-missing value. Rows with missing values have no difference.
                diff_series = self.orig_df[col_name].astype(float).dropna().diff().reindex(self.orig_df.index)

                # Check there are few decreasing values
                decr_series = diff_series < 0
                num_decr = decr_series.tolist().count(True)
                if num_decr > self.freq_contamination_level:
                    continue

                # Check the number of increases is significantly more than the number of decreases
                incr_series = diff_series > 0
                num_incr = incr_series.tolist().count(True)
                if num_decr > (num_incr / 10.0):
                    continue

                # Check there are a decent number increasing
                if num_incr < (self.num_valid_rows[col_name] / 20):
                    continue

                test_series = (diff_series >= 0) | \
                              np.array([is_missing(x) for x in diff_series])
            else:
                # Compare each value to the previous non-missing value. Rows with missing values have no difference.
                gap_series = pd.to_datetime(self.orig_df[col_name]).dropna().diff().reindex(self.orig_df.index)

                # Check there are few decreasing values
                decr_series = gap_series.dt.total_seconds() < 0
                num_decr = decr_series.tolist().count(True)
                if num_decr > self.freq_contamination_level:
                    continue

                # Check the number of increases is significantly more than the number of decreases
                incr_series = gap_series.dt.total_seconds() > 0
                num_incr = incr_series.tolist().count(True)
                if num_decr > (num_incr / 10.0):
                    continue

                # Check there are a decent number increasing
                if num_incr < (self.num_valid_rows[col_name] / 20):
                    continue

                test_series = (gap_series.dt.total_seconds() >= 0) | gap_series.isna()
            # The shift operation is undefined for the first row, which results in a NaN that we fill here.
            test_series[0] = True
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column contains consistently ascending values")


    def _generate_column_decreasing(self):
        """
        Patterns without exceptions: 'col desc all' is consistently decreasing.
        Patterns with exception: 'col desc most' is consistently decreasing, with one exception.
        """
        self._add_synthetic_column('col_desc rand',
                                    [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('col_desc all',
                                    sorted([round(random.random(), 2) for _ in range(self.num_synth_rows)], reverse=True))
        self._add_synthetic_column('col_desc most',
                                    sorted([round(random.random(), 2) for _ in range(self.num_synth_rows-1)], reverse=True) + [5.1])

        test_date = datetime.datetime.strptime("1-1-2000", "%d-%m-%Y")
        self._add_synthetic_column('col_desc date_most',
                                    sorted(pd.date_range(test_date, periods=self.num_synth_rows, freq='D'), reverse=True))
        self.synth_df.loc[999, 'col_desc date_most'] = datetime.datetime.strptime("1-1-2001", "%d-%m-%Y")


    def _check_column_decreasing(self, test_id):
        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique() < 3:
                continue
            if col_name in self.numeric_cols:
                # Check if there are any invalid numeric values
                if len(self.numeric_vals[col_name]) < self.num_rows:
                    continue

                # Compare each value to the previous non-missing value. Rows with missing values have no difference.
                diff_series = self.orig_df[col_name].astype(float).dropna().diff().reindex(self.orig_df.index)

                # Check there are few increasing values
                incr_series = diff_series > 0
                num_incr = incr_series.tolist().count(True)
                if num_incr > self.freq_contamination_level:
                    continue

                # Check the number of decreases is significantly more than the number of increases
                decr_series = diff_series < 0
                num_decr = decr_series.tolist().count(True)
                if num_incr > (num_decr / 10.0):
                    continue

                # Check there are a decent number decreasing
                if num_decr < (self.num_valid_rows[col_name] / 20):
                    continue

                test_series = (diff_series <= 0) | \
                              np.array([is_missing(x) for x in diff_series])
            else:
                # Compare each value to the previous non-missing value. Rows with missing values have no difference.
                gap_series = pd.to_datetime(self.orig_df[col_name]).dropna().diff().reindex(self.orig_df.index)

                # Check there are few increasing values
                incr_series = gap_series.dt.total_seconds() > 0
                num_incr = incr_series.tolist().count(True)
                if num_incr > self.freq_contamination_level:
                    continue

                # Check the number of decreases is significantly more than the number of increases
                decr_series = gap_series.dt.total_seconds() < 0
                num_decr = decr_series.tolist().count(True)
                if num_incr > (num_decr / 10.0):
                    continue

                # Check there are a decent number decreasing
                if num_decr < (self.num_valid_rows[col_name] / 20):
                    continue

                test_series = (gap_series.dt.total_seconds() <= 0) | gap_series.isna()
            # The shift operation is undefined for the first row, which results in a NaN we fill here.
            test_series[0] = True
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column contains consistently descending values")


    def _generate_column_tends_asc(self):
        """
        Patterns without exceptions: 'tends_asc all' tends to increase
        Patterns with exception: 'tends_asc most' tends to increase, with one exception. As well, 'tends_asc date_most'
            tends to increase, with one exception.
        """
        self._add_synthetic_column('tends_asc rand', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('tends_asc all',  [x + (100 * random.random()) for x in range(self.num_synth_rows)])
        self._add_synthetic_column('tends_asc most', [x + (100 * random.random()) for x in range(self.num_synth_rows)])
        self.synth_df.loc[999, 'tends_asc most'] = 400

        dates_arr = []
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        for i in range(self.num_synth_rows):
            dates_arr.append(test_date + relativedelta(days=(i + np.random.randint(-50, 50))))
        self._add_synthetic_column('tends_asc date_most', dates_arr)
        self.synth_df.loc[999, 'tends_asc date_most'] = datetime.datetime.strptime("15-7-2022", "%d-%m-%Y")


    def _check_column_tends_asc(self, test_id):
        """
        Check if the values in the column are correlated with the row numbers. Identify any outliers.
        """

        row_numbers = list(range(self.num_rows))
        row_numbers_percentiles = pd.Series(row_numbers).rank(pct=True)
        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique(dropna=True) < 3:
                continue
            if col_name in self.numeric_cols:
                spearman_corr = abs(self.numeric_vals[col_name].corr(pd.Series(row_numbers), method='spearman'))
            else:
                col_vals = [x.timestamp() for x, y in
                            zip(pd.to_datetime(self.orig_df[col_name]), self.orig_df[col_name].isna()) if not y]
                spearman_corr = abs(pd.Series(col_vals).corr(pd.Series(list(range(len(col_vals)))), method='spearman'))
            if spearman_corr >= 0.95:
                col_percentiles = self.orig_df[col_name].rank(pct=True)

                # Test for positive correlation
                test_series = np.array([(abs(x-y) < 0.1) or is_missing(x)
                                        for x, y in zip(col_percentiles, row_numbers_percentiles)])
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    f'"{col_name}" is consistently similar, with regards to percentile, to the row number')


    def _generate_column_tends_desc(self):
        """
        Patterns without exceptions: 'tends_desc all' tends to decrease
        Patterns with exception: 'tends_desc most' tends to decrease, with one exception. As well, 'tends_desc date_most'
            tends to decrease, with one exception.
        """
        self._add_synthetic_column('tends_desc rand', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('tends_desc all',  [x + (100 * random.random()) for x in range(self.num_synth_rows-1, -1, -1)])
        self._add_synthetic_column('tends_desc most', [x + (100 * random.random()) for x in range(self.num_synth_rows-1, -1, -1)])
        self.synth_df.loc[999, 'tends_desc most'] = 1000

        dates_arr = []
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        for i in range(self.num_synth_rows):
            dates_arr.append(test_date + relativedelta(days=(i + np.random.randint(-50, 50))))
        self._add_synthetic_column('tends_desc date_most', sorted(dates_arr, reverse=True))
        self.synth_df.loc[999, 'tends_desc date_most'] = datetime.datetime.strptime("1-7-2024", "%d-%m-%Y")


    def _check_column_tends_desc(self, test_id):
        """
        Check if the values in the column are inversely correlated with the row numbers. Identify any outliers.
        """

        row_numbers = list(range(self.num_rows))
        row_numbers_percentiles = pd.Series(row_numbers).rank(pct=True)
        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique(dropna=True) < 3:
                continue
            if col_name in self.numeric_cols:
                spearman_corr = abs(self.numeric_vals[col_name].corr(pd.Series(row_numbers), method='spearman'))
            else:
                col_vals = [x.timestamp() for x, y in zip(pd.to_datetime(self.orig_df[col_name]), self.orig_df[col_name].isna()) if not y]
                spearman_corr = abs(pd.Series(col_vals).corr(pd.Series(list(range(len(col_vals)))), method='spearman'))
            if spearman_corr >= 0.95:
                col_percentiles = self.orig_df[col_name].rank(pct=True)

                # Test for negative correlation
                test_series = np.array([(abs(x-(1.0 - y)) < 0.1) or is_missing(x) for x, y in
                                        zip(col_percentiles, row_numbers_percentiles)])
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    f'"{col_name}" is consistently inversely similar, with regards to percentile, to the row number')


    def _generate_similar_previous(self):
        """
        Patterns without exceptions: 'sim_prev all' has values consistently similar to the previous values
        Patterns with exception: 'sim_prev most' has values consistently similar to the previous values, with 1
            exception. As well, sim_prev date_most' has values consistently similar to the previous values, with 1
            exception.
        Not Flagged: 'sim_prev_rand' has no relationship to the previous values
        """
        random_walk = [10.0]
        prev_val = random_walk[0]
        for _i in range(self.num_synth_rows-1):
            new_val = prev_val + ((random.random() - 0.5) * 2.0)
            random_walk.append(new_val)
            prev_val = new_val

        self._add_synthetic_column('sim_prev rand', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim_prev all', random_walk)
        self._add_synthetic_column('sim_prev most', random_walk)
        self.synth_df.loc[999, 'sim_prev most'] += 15

        random_walk = [datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")]
        prev_val = random_walk[0]
        for _i in range(self.num_synth_rows-1):
            new_val = prev_val + relativedelta(days=np.random.randint(-10, 10))
            random_walk.append(new_val)
            prev_val = new_val
        self._add_synthetic_column('sim_prev date_most', random_walk)
        self.synth_df.loc[999, 'sim_prev date_most'] = datetime.datetime.strptime("01-7-2025", "%d-%m-%Y")


    def _check_similar_previous(self, test_id):
        """
        This test checks if each value within a numeric or date column is similar to the previous value in the column.
        The test cannot be performed on the first row of each column. It often identifies where values are steadily
        increasing, and then set back to a lower value, and restart increasing from there, a common pattern in data.
        It can also detect where values follow a random walk, where each value is similar to the previous value
        with some random movement up or down.
        """

        for col_name in self.numeric_cols + self.date_cols:
            if self.orig_df[col_name].nunique(dropna=True) <= 2:
                continue
            # Missing values are skipped: each value is compared to the previous non-missing value
            non_null_arr = self.orig_df[col_name].notna()
            if col_name in self.numeric_cols:
                num_vals = self.numeric_vals_filled[col_name][non_null_arr]
                diff_to_prev_arr = abs(num_vals.diff())
                col_med = self.column_medians[col_name]
                diff_to_median_arr = abs(num_vals - self.column_medians[col_name])
            else:
                date_vals = pd.to_datetime(self.orig_df[col_name])[non_null_arr]
                diff_to_prev_arr = abs(date_vals.diff())
                col_med = pd.to_datetime(self.orig_df[col_name]).quantile(0.5, interpolation='midpoint')
                diff_to_median_arr = abs(date_vals - col_med)
            test_series = diff_to_prev_arr < diff_to_median_arr
            test_series.iloc[0] = True  # The first value has no difference from the previous, so can not be tested

            # Test a reasonable number of values are closer to the previous value than to the median. It does not
            # have to be almost all, as many values may be close to the median as well.
            if test_series.tolist().count(True) < (len(test_series) * 0.75):
                continue

            # Test that those values not closer to the previous value, are still close
            if col_name in self.numeric_cols:
                q1 = self.numeric_vals[col_name].quantile(0.25)
                q3 = self.numeric_vals[col_name].quantile(0.75)
            else:
                q1 = pd.to_datetime(self.orig_df[col_name]).quantile(0.25, interpolation='midpoint')
                q3 = pd.to_datetime(self.orig_df[col_name]).quantile(0.75, interpolation='midpoint')

            iqr = abs(q3 - q1)
            test_series = diff_to_prev_arr < iqr
            test_series.iloc[0] = True
            test_series = test_series.reindex(self.orig_df.index, fill_value=True)  # Missing values are not flagged

            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f'The values in "{col_name}" are consistently similar to the previous value, more so than they are '
                 f'similar to the median value of the column ({col_med})')
            )


    def _generate_unusual_order_magnitude(self):
        """
        Patterns without exceptions: 'unusual_number all' has a strong pattern, as all values are within 1 order of
            magnitude.
        Patterns with exception: 'unusual_number most_1' has a strong pattern, with most values having a value within an
            order of magnitude of each other, with one value many orders of magnitude larger. Similar for
            'unusual_number most_2'
        """
        self._add_synthetic_column('unusual_number all',
                                    [random.randint(4, 8) * 100 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('unusual_number most_1',
                                    [random.randint(1, 5) * 100 for _ in range(self.num_synth_rows-1)] + [999_999_999])
        self._add_synthetic_column('unusual_number most_2',
                                    sorted([random.randint(1000, 1_000_000) for _ in range(self.num_synth_rows)], reverse=True))


    def _check_unusual_order_magnitude(self, test_id):
        for col_name in self.numeric_cols:
            # We consider only columns that do not contain fractions
            if abs(self.numeric_vals[col_name]).min() < 1:
                continue
            vals_arr = self.numeric_vals_filled[col_name]
            test_series = round(np.log10(abs(vals_arr.replace(0, 1)))).values
            self._process_analysis_counts(
                test_id,
                [col_name],
                test_series,
                (f"This test checks for values of an unusual order of magnitude. Each value is described in terms "
                 f"of its order, or power of 10. For example, 10 is of order 1, 100 is of order 2, 1000 is of order 3. "
                 f"All numbers are rounded to the nearest integer order of magnitude. For example, 3.0 will be "
                 f"considered of order 0, and 7.0 of order 1. "
                 f"The column contains values in the range {self.numeric_vals[col_name].min()} to "
                 f"{self.numeric_vals[col_name].max()}, and consistently in the order of"),
                "",
                display_info={"order of magnitude": test_series}
            )


    def _generate_few_neighbors(self):
        """
        Patterns without exceptions: None. This test does not support patterns.
        Patterns with exception: values in 'few neighbors most' are consistently close to their neighbors, with one
            exception. Similar for 'few neighbors date_most'.
        """
        self._add_synthetic_column('few neighbors rand',
                                    [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('few neighbors all',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows-100)] +
                                    [random.randint(10, 20) for _ in range(100)])
        self._add_synthetic_column('few neighbors most',
                                    [random.randint(10_000, 20_000) for _ in range(self.num_synth_rows-100)] +
                                    [5000] +
                                    [random.randint(1000, 2000) for _ in range(99)])

        date_1 = datetime.datetime.strptime("01-7-2018", "%d-%m-%Y")
        date_2 = datetime.datetime.strptime("01-7-2025", "%d-%m-%Y")
        self._add_synthetic_column('few neighbors date_most',
            [date_1 + relativedelta(days=(i + np.random.randint(-50, 50))) for i in range(self.num_synth_rows-100)] +
            [date_2 + relativedelta(days=(i + np.random.randint(-50, 50))) for i in range(99)] +
            [datetime.datetime.strptime("01-7-2023", "%d-%m-%Y")])


    def _check_few_neighbors(self, test_id):
        """
        Handling Null values: Null values do not establish or violate a pattern.

        This operates on the sorted values, not considering the original row orders.

        The test examines all numeric and date columns.
        """

        # This test does not identify patterns, but does identify rows that are distant from both the nearest point
        # before and point afterwards. This does not flag extreme values, only internal outliers.

        for col_name in self.numeric_cols:
            # Skip columns with any non-numeric values
            if len(self.numeric_vals[col_name]) < self.num_rows:
                continue

            sorted_vals = copy.copy(self.orig_df[col_name].astype(float).values)
            sorted_vals.sort()
            sorted_vals = pd.Series(sorted_vals)
            diff_from_prev = sorted_vals.diff(1)
            diff_from_next = sorted_vals.diff(-1)
            diff_threshold = (self.numeric_vals[col_name].max() - self.numeric_vals[col_name].min()) / 10.0

            test_arr = [bool(not math.isnan(x) and not math.isnan(y) and abs(x) > diff_threshold and abs(y) > diff_threshold) for x, y in zip(diff_from_prev, diff_from_next)]
            num_isolated_points = test_arr.count(True)

            if num_isolated_points > 0:
                # Flag the correct rows. We currently have their indexes based on a sorted array.
                vals_arr = [x for x, y in zip(sorted_vals, test_arr) if y]
                test_series = [x not in vals_arr for x in self.orig_df[col_name].values]
                # Some versions of pandas cannot work with NaN values here, so use the min & max
                prev_arr = np.array(
                    sorted([self.orig_df[col_name].min()] +
                    self.numeric_vals_filled[col_name].values.tolist())
                )[list(self.numeric_vals_filled[col_name].rank().astype(int)-1)].tolist()
                next_arr = np.array(
                    sorted(self.numeric_vals_filled[col_name].values.tolist() +
                           [self.orig_df[col_name].max(), self.orig_df[col_name].max()])
                )[list(self.numeric_vals_filled[col_name].rank().astype(int)+0)].tolist()

                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    (f"The test marked any values more than {diff_threshold:.3f} away from both the next smallest and "
                     f"next largest values in the column. The minimum value for the column is "
                     f"{self.numeric_vals[col_name].min()} and the maximum is {self.numeric_vals[col_name].max()}. "),
                    allow_patterns=False,
                    display_info={"prev_val": prev_arr, "next_val": next_arr}
                )

        for col_name in self.date_cols:
            sorted_vals = copy.copy(pd.to_datetime(self.orig_df[col_name]).values)
            sorted_vals.sort()
            sorted_vals = pd.Series(sorted_vals)
            diff_from_prev = sorted_vals.diff(1)
            diff_from_next = sorted_vals.diff(-1)
            diff_threshold = (pd.to_datetime(self.orig_df[col_name]).max() -
                              pd.to_datetime(self.orig_df[col_name]).min()) / 10.0

            test_arr = [bool(x == x and y == y and abs(x) > diff_threshold and abs(y) > diff_threshold) for x, y in zip(diff_from_prev, diff_from_next)]
            num_isolated_points = test_arr.count(True)

            if num_isolated_points > 0:
                # Flag the correct rows. We currently have their indexes based on a sorted array.
                vals_arr = [x for x, y in zip(sorted_vals, test_arr) if y]
                test_series = [is_missing(x) or x not in vals_arr for x in self.orig_df[col_name].values]
                # Find the closest values among the non-missing values. Rows with missing values have none.
                non_null_vals = self.orig_df[col_name].dropna()
                prev_arr = np.array(sorted(non_null_vals.values))[list(non_null_vals.rank().astype(int)-2)]
                next_arr = np.array([pd.Timestamp(x) for x in np.concatenate(  # It converts to integer otherwise
                        [np.array(sorted(non_null_vals.values)), np.array([non_null_vals.max()])]
                    ).astype(pd.Timestamp)])[list(non_null_vals.rank().astype(int))]
                prev_arr = pd.Series(prev_arr, index=non_null_vals.index).reindex(self.orig_df.index).values
                next_arr = pd.Series(next_arr, index=non_null_vals.index, dtype=object).reindex(
                    self.orig_df.index, fill_value=pd.NaT).values  # type: ignore[arg-type]  # NaT is not in the stubs

                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    (f"The test marked any values more than {diff_threshold.days} days away from both the next "
                     "smallest and next largest values in the column"),
                    allow_patterns=False,
                    display_info={"prev_val": prev_arr, "next_val": next_arr}
                )


    def _generate_few_within_range(self):
        """
        Patterns without exceptions: 'few in range all' consistently has values with other similar values.
        Patterns with exception: 'few in range most' consistently has values with other similar values, with three
            exceptions -- a set of three numbers similar to each other, but few other values. Similar for
            'few in range date_most'.
        """
        self._add_synthetic_column('few in range rand',
                                    [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('few in range all',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows - 200)] +
                                    [random.randint(10_000, 20_000) for _ in range(200)])
        self._add_synthetic_column('few in range most',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows - 200)] +
                                    [random.randint(10_000, 20_000) for _ in range(197)] + [5000, 5001, 5002])

        date_1 = datetime.datetime.strptime("01-7-2018", "%d-%m-%Y")
        date_2 = datetime.datetime.strptime("01-7-2025", "%d-%m-%Y")
        self._add_synthetic_column('few in range date_most',
            [date_1 + relativedelta(days=(np.random.randint(-50, 50))) for _ in range(self.num_synth_rows - 200)] +
            [date_2 + relativedelta(days=(np.random.randint(-50, 50))) for _ in range(197)] +
            [datetime.datetime.strptime("01-7-2023", "%d-%m-%Y"),
             datetime.datetime.strptime("02-7-2023", "%d-%m-%Y"),
             datetime.datetime.strptime("03-7-2023", "%d-%m-%Y")])


    def _check_few_within_range(self, test_id):
        """
        This identifies only internal outliers and not extreme values. It flags values where there are few other values
        within a range on either side. To do this efficiently, it divides the column into 50 equal-width bins.
        It is concerned with bins of width 1/10 the full data range, but to do this in a more robust manner, it
        uses 50 bins, and considers sets of 7 bins.

        Handling null values: Null values do not add to the count within any bin. Null values will not be flagged as
        having few within a range.
        """

        num_bins = 50

        for col_name in self.numeric_cols:
            # Skip columns with any non-numeric values
            if len(self.numeric_vals[col_name]) < self.num_rows:
                continue

            # Skip columns with few unique values
            if self.orig_df[col_name].nunique() < num_bins:
                continue

            bins = [-np.inf]
            min_val = self.numeric_vals[col_name].min()
            max_val = self.numeric_vals[col_name].max()
            col_range = max_val - min_val
            bin_width = col_range / num_bins
            for i in range(num_bins):
                bins.append(min_val + (i * bin_width))
            bins.append(np.inf)
            bin_labels = [int(x) for x in range(len(bins)-1)]
            # Keep one bin per row (NaN for missing values) so the results stay aligned with the rows
            binned_values = pd.cut(self.numeric_vals[col_name], bins, labels=bin_labels)
            bin_counts = binned_values.dropna().value_counts()

            rare_bins = []

            # Set combined bins counts for elements outside the normal range
            combined_bins_counts = [0]*len(bin_counts)
            list_counts = bin_counts.sort_index().tolist()
            combined_bins_counts[0] = list_counts[0] + list_counts[1] + list_counts[2]
            combined_bins_counts[1] = list_counts[0] + list_counts[1] + list_counts[2] + list_counts[3]
            combined_bins_counts[2] = list_counts[0] + list_counts[1] + list_counts[2] + list_counts[3] + list_counts[4]
            combined_bins_counts[-1] = list_counts[-1] + list_counts[-2] + list_counts[-3]
            combined_bins_counts[-2] = list_counts[-1] + list_counts[-2] + list_counts[-3] + list_counts[-4]
            combined_bins_counts[-3] = list_counts[-1] + list_counts[-2] + list_counts[-3] + list_counts[-4] + list_counts[-5]

            for bin_id in bin_labels[3:-3]:
                rows_count = bin_counts.loc[bin_id-3] + \
                             bin_counts.loc[bin_id-2] + \
                             bin_counts.loc[bin_id-1] + \
                             bin_counts.loc[bin_id]   + \
                             bin_counts.loc[bin_id+1] + \
                             bin_counts.loc[bin_id+2] + \
                             bin_counts.loc[bin_id+3]
                combined_bins_counts[bin_id] = rows_count
                if (bin_counts.sort_index().loc[bin_id+1:].sum() > (self.num_valid_rows[col_name] / 10.0)) and \
                    (bin_counts.sort_index().loc[:bin_id].sum() > (self.num_valid_rows[col_name] / 10.0)) and \
                    (rows_count < self.freq_contamination_level):
                    rare_bins.append(bin_id)
            if len(rare_bins) == 0:
                continue
            test_series = np.array([x not in rare_bins for x in binned_values])

            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently contains values that have several similar values in the column",
                (f" -- any values with fewer than an average of {math.floor(self.freq_contamination_level)} neighbors "
                 f"within their and the neighboring bins (width {bin_width:.4f})"),
                display_info={'Number in Range': [np.nan if pd.isna(x) else combined_bins_counts[x] for x in binned_values]}
            )

        for col_name in self.date_cols:
            # Skip columns with few unique values
            if self.orig_df[col_name].nunique() < num_bins:
                continue

            bins = []
            min_val = self.orig_df[col_name].min()
            max_val = self.orig_df[col_name].max()
            col_range = max_val - min_val
            bin_width = col_range / num_bins
            for i in range(num_bins + 2):
                bins.append(min_val + (i * bin_width))
            bin_labels = [int(x) for x in range(len(bins)-1)]
            binned_values = pd.cut(pd.to_datetime(self.orig_df[col_name]), bins=bins, labels=bin_labels, include_lowest=True)
            bin_counts = binned_values.value_counts()

            rare_bins = []

            # Set combined bins counts for elements outside the normal range
            combined_bins_counts = [0]*len(bin_counts)
            list_counts = bin_counts.sort_index().tolist()
            combined_bins_counts[0] = list_counts[0] + list_counts[1] + list_counts[2]
            combined_bins_counts[1] = list_counts[0] + list_counts[1] + list_counts[2] + list_counts[3]
            combined_bins_counts[2] = list_counts[0] + list_counts[1] + list_counts[2] + list_counts[3] + list_counts[4]
            combined_bins_counts[-1] = list_counts[-1] + list_counts[-2] + list_counts[-3]
            combined_bins_counts[-2] = list_counts[-1] + list_counts[-2] + list_counts[-3] + list_counts[-4]
            combined_bins_counts[-3] = list_counts[-1] + list_counts[-2] + list_counts[-3] + list_counts[-4] + list_counts[-5]

            for bin_id in bin_labels[3:-3]:
                rows_count = bin_counts.loc[bin_id-3] + \
                             bin_counts.loc[bin_id-2] + \
                             bin_counts.loc[bin_id-1] + \
                             bin_counts.loc[bin_id]   + \
                             bin_counts.loc[bin_id+1] + \
                             bin_counts.loc[bin_id+2] + \
                             bin_counts.loc[bin_id+3]
                combined_bins_counts[bin_id] = rows_count
                if (bin_counts.sort_index().loc[bin_id+1:].sum() > (self.num_valid_rows[col_name] / 10.0)) and \
                        (bin_counts.sort_index().loc[:bin_id].sum() > (self.num_valid_rows[col_name] / 10.0)) and \
                        (rows_count < self.freq_contamination_level) and (rows_count > 0):
                    rare_bins.append(bin_id)
            if len(rare_bins) == 0:
                continue
            test_series = np.array([x not in rare_bins for x in binned_values])

            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently contains values that have several similar values in the column",
                (f" -- any values with fewer than an average of {math.floor(self.freq_contamination_level)} neighbors "
                 f"within their and the neighboring bins (width {bin_width.days} days)"),
                display_info={'Number in Range': [np.nan if pd.isna(x) else combined_bins_counts[x] for x in binned_values]})


    def _generate_very_small(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('very small rand',
                                    [random.randint(1, 100_000_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('very small all',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('very small most',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows-1)] + [3])


    def _check_very_small(self, test_id):
        """
        This uses inter-decile range.
        """

        for col_name in self.numeric_cols:
            d1 = self.numeric_vals[col_name].quantile(0.1)
            d9 = self.numeric_vals[col_name].quantile(0.9)
            lower_limit = d1 - (self.idr_limit * (d9 - d1))
            col_vals = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])
            test_series = self.orig_df[col_name].isna() | (col_vals > lower_limit)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"The test marked any values less than {lower_limit:,.2f} as very small given the 10th decile is "
                 f"{d1:,.2f} and the 90th is {d9:,.2f}. The coefficient is set at {self.idr_limit}"),
                allow_patterns=False,
                display_info={"lower_limit": lower_limit}
            )


    def _generate_very_large(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'very large most' contains one value much larger than the others in this column.
        """
        self._add_synthetic_column('very large rand1',
                                    [random.randint(1, 100_000_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('very large rand2',
                                    [random.randint(100_000_000, sys.maxsize) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('very large all',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('very large most',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows-1)] + [30_000])


    def _check_very_large(self, test_id):
        for col_name in self.numeric_cols:
            q1 = self.numeric_vals[col_name].quantile(0.25)
            q3 = self.numeric_vals[col_name].quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))  # Using a stricter threshold than the 2.2 normally used
            vals_arr = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])
            test_series = self.orig_df[col_name].isna() | (vals_arr < upper_limit)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"The test marked any values larger than {upper_limit:,.2f} as very large given the 25th quartile "
                 f"is {q1:,.2f} and 75th is {q3:,.2f}"),
                allow_patterns=False,
                display_info={"upper_limit": upper_limit}
            )


    def _generate_very_small_abs(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'very_small_abs most' has one very small value.
        """
        self._add_synthetic_column('very_small_abs rand',
                                    [random.randint(1, 100_000_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('very_small_abs all',
                                    [random.randint(5000, 10000) for _ in range(self.num_synth_rows // 2)]
                                    + [random.randint(-10000, -5000)
                                       for _ in range(self.num_synth_rows - (self.num_synth_rows // 2))])
        self._add_synthetic_column('very_small_abs most', self.synth_df['very_small_abs all'])
        self.synth_df.loc[999, 'very_small_abs most'] = 3


    def _check_very_small_abs(self, test_id):
        """
        Detect where values are unusually close to zero, where a feature has a significant number of both positive
        and negative values. Note, it is not necessary to check for very large absolute values, as these would be caught
        by the tests for very small and very large values. This test catches internal outliers, which are closer than
        normal to zero.
        """
        for col_name in self.numeric_cols:
            num_pos = (self.numeric_vals[col_name] > 0).tolist().count(True)
            num_neg = (self.numeric_vals[col_name] < 0).tolist().count(True)
            if num_pos < (self.num_valid_rows[col_name] * 0.10) or num_neg < (self.num_valid_rows[col_name] * 0.10):
                continue
            abs_vals = self.numeric_vals[col_name].abs()
            d1 = abs_vals.quantile(0.1)
            d9 = abs_vals.quantile(0.9)
            lower_limit = d1 - (self.idr_limit * (d9 - d1))
            vals_arr = abs(convert_to_numeric(self.orig_df[col_name], self.numeric_vals[col_name].max()))
            test_series = self.orig_df[col_name].isna() | (vals_arr > lower_limit)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"Some values were unusually close to zero. Any values with absolute value less than {lower_limit:,.2f} "
                 f"were flagged, given the 10th percentile of absolute values is {d1:,.2f} and the 90th is {d9:,.2f}. "
                 f" The coefficient is set at {self.idr_limit}"),
                "",
                allow_patterns=False,
                display_info={'lower_limit': lower_limit}
            )


    def _generate_multiple_constant(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('constant multiple rand',
                                    [random.randint(1, 100_000_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('constant multiple all',
                                    [random.randint(1, 2000) * 13.3 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('constant multiple most',
                                    [random.randint(1, 2000) * 13.3 for _ in range(self.num_synth_rows-1)] + [30_000])


    def _check_multiple_constant(self, test_id):
        # todo: it is also possible to look at the gaps between values to find the common denominator
        # todo: this considers only constants greater than 1.0, but in some cases, constants less than 1.0 may be
        # meaningful. Eg a constant of .25 or .5 is meaningless if the values are all integers, but is meaningful
        # if most values have decimals.

        def test_divisor(col_name, v):
            vals_arr = convert_to_numeric(self.orig_df[col_name], v)
            test_series = [math.isclose(x, y) or is_missing(x) or is_missing(y)
                           for x, y in zip(round(vals_arr / v), vals_arr / v)]
            n_multiples = test_series.count(True)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f"The column contains values that are consistently multiples of {v}",
                display_info={"value": v}
            )
            return bool(n_multiples > (self.num_rows - self.freq_contamination_level))

        exclude_list = [0, 1, -1]
        for col_name in self.numeric_cols:

            # Ensure there are many unique values in the column. Otherwise they may be multiples of a common
            # value in a trivial way.
            if self.orig_df[col_name].nunique() < (self.num_rows / 10):
                continue

            # Test if the rows tend to be multiples of any of the 5 smallest values. We test more than 1, as the
            # small values may themselves be the outliers.
            min_vals = [x for x in self.numeric_vals[col_name].unique() if x > 0.5 and x not in exclude_list]
            if len(min_vals) == 0:
                continue
            min_vals = sorted(set(min_vals))[:5]
            found = False
            for v in min_vals:
                if test_divisor(col_name, v):
                    found = True
                    break

            # It may be that all records are several times the common multiple and the common multiple itself is not
            # in the data. Check if the smallest value available is any multiple up to 10 of the common multiple.
            if found:
                continue
            for v_div in range(2, 11):
                v = min_vals[0] / v_div
                if v > 0.5 and v not in exclude_list and test_divisor(col_name, v):
                    break


    def _generate_rounding(self):
        """
        Patterns without exceptions: 'rounding 1' and 'rounding 2' both have sets of values with similar numbers of
            rounded values.
        Patterns with exception: 'rounding 3' has one value with significantly more zeros. 'rounding 4' does as well,
            and is an example with a float column.
        """
        self._add_synthetic_column('rounding 1', [random.randint(1, 1_000_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('rounding 2', [int(random.randint(1, 100) * (math.pow(10, random.randint(1, 4))))
                                                   for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('rounding 3', [int(random.randint(1, 100) * (math.pow(10, random.randint(1, 4))))
                                                   for _ in range(self.num_synth_rows-1)] + [100_000_000])
        self._add_synthetic_column('rounding 4', [(random.randint(1, 100) * 0.25 * (math.pow(10, random.randint(1, 4))))
                                                   for _ in range(self.num_synth_rows-1)] + [100_000_000])


    def _check_rounding(self, test_id):
        """
        For each numeric column, we get the number of trailing zeros in each value. This test applies only to integer
        columns. We take the normal range of number of zeros: for example, 1 to 4, meaning most values have 1, 2, 3,
        or 4 trailing zeros. We flag any values with less, or more than 2 more zeros. In this example, anything with
        0 or 7 or more zeros. Patterns without exceptions will be reported only if the normal numbers of zeros is
        at least 1 (not zero).
        """

        is_missing_dict = self.get_is_missing_dict()

        for col_name in self.numeric_cols:
            if self.orig_df[col_name].notna().sum() < self.freq_contamination_level:
                continue

            # Skip any columns than have more than a few non-integer values. We do not use self.numeric_vals_filled
            # as the median values may have decimals or not.
            numeric_vals = convert_to_numeric(self.orig_df[col_name], 0)
            numeric_vals = numeric_vals.fillna(0)
            if (numeric_vals.astype(int) == numeric_vals).tolist().count(False) > \
                    self.freq_contamination_level:
                continue

            # Create an array called vals, with the string representation of the values in col_name, where we
            # convert any non-numeric values to 0. Leave NaN values as NaN.
            vals = self.orig_df[col_name].fillna(-9595959484)
            vals = convert_to_numeric(vals, 0)
            vals = as_str(vals.astype(int)).replace('-9595959484', np.nan)
            vals = vals.replace("0", "1")  # We count 0 as having no trailing zeros; it is essentially a 1-digit number.

            num_zeros_arr = vals.str.replace('.0', '', regex=False).str.len() - \
                            vals.str.replace('.0', '', regex=False).str.strip('0').str.len()
            counts_series = num_zeros_arr.value_counts()  # This counts only the non-missing values
            cum_sum_series = np.where(counts_series.sort_values(ascending=False).cumsum() >
                                      (counts_series.sum() - self.freq_contamination_level))
            if (len(cum_sum_series) == 0) or (len(cum_sum_series[0]) == 0):
                continue
            last_normal_index = cum_sum_series[0][0]
            normal_vals = counts_series.index[:last_normal_index + 1]
            # The numbers of zeros are floats where there are missing values
            min_normal = int(min(normal_vals))
            max_normal = int(max(normal_vals))
            test_series = ((num_zeros_arr >= min_normal) & (num_zeros_arr <= max_normal + 2)) | is_missing_dict[col_name]

            desc_str = f'The column has values with consistently {min_normal} to {max_normal} trailing zeros.'
            if min_normal == max_normal:
                desc_str = f'The column has values with consistently {min_normal} trailing zeros.'

            if min_normal > 0:
                exception_str = f" -- flagging values with with {max_normal + 3} or more trailing zeros"
            else:
                exception_str = (f" -- flagging values with less than {min_normal} or with more than {max_normal + 2} "
                                 "trailing zeros")

            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                desc_str,
                exception_str,
                allow_patterns=(min_normal > 0)
            )


    def _generate_non_zero(self):
        """
        Patterns without exceptions: 'non-zero all'  consistently contains non-zero values
        Patterns with exception: 'non-zero most'  consistently contains non-zero values, with 1 exception
        """
        self._add_synthetic_column('non-zero rand', [random.randint(0, 50) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('non-zero all',  [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('non-zero most', [random.randint(1, 100) for _ in range(self.num_synth_rows-1)] + [0])


    def _check_non_zero(self, test_id):
        for col_name in self.numeric_cols:

            # Check where are many values on either side of zero. Zero may be rare simply because there are many
            # unique values.
            num_below = (self.numeric_vals_filled[col_name] < 0).tolist().count(True)
            num_above = (self.numeric_vals_filled[col_name] > 0).tolist().count(True)
            if (num_below > self.freq_contamination_level) and (num_above > self.freq_contamination_level):
                continue

            test_series = np.array([x != 0 for x in self.orig_df[col_name]])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently contains non-zero values")


    def _generate_less_than_one(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('less_than_one rand', [random.random() * 2.0 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('less_than_one all',  [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('less_than_one most', [random.random() for _ in range(self.num_synth_rows-1)] + [1.2])


    def _check_less_than_one(self, test_id):
        for col_name in self.numeric_cols:

            # Skip columns that are almost entirely zero. In this case, the pattern is really that the column is
            # mostly zero, not that it is less than 1.0.
            if (self.orig_df[col_name] == 0).tolist().count(False) <= (self.num_rows / 10.0):
                continue

            if self.orig_df[col_name].nunique() < 3:
                continue

            if self.orig_df[col_name].notna().sum() < self.freq_contamination_level:
                continue

            # We do not use self.numeric_vals_filled, as we wish to fill with 0.5 and not the median.
            vals_arr = convert_to_numeric(self.orig_df[col_name], 0.5)
            test_series = np.array([(abs(x) <= 1.0) or is_missing(x) for x in vals_arr])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently contains values between -1.0, and 1.0 (inclusive)")


    def _generate_greater_than_one(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('greater_than_one rand', [random.random() * 2.0 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('greater_than_one all',  [random.random() + 2.0 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('greater_than_one most', [random.random() + 2.0 for _ in range(self.num_synth_rows-1)] + [0.6])


    def _check_greater_than_one(self, test_id):
        for col_name in self.numeric_cols:
            vals_arr = convert_to_numeric(self.orig_df[col_name], 10.0)
            test_series = np.array([(x == 0) or (abs(x) >= 1.0) for x in vals_arr.fillna(1)])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently contains absolute values greater than or equal to 1.0")


    def _generate_invalid_numbers(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'invalid_number most' contains consistently valid numbers, with one exception.
        Not Flagged: 'invalid_number rand' has many non-valid numeric value, and would not be considered a numeric
            column
        """
        self._add_synthetic_column('invalid_number rand',
                                    [random.choice(list(digits) + ['a']) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('invalid_number most',
                                    [random.random() + 2.0 for _ in range(self.num_synth_rows-1)] + ['0.6%'])


    def _check_invalid_numbers(self, test_id):
        for col_name in self.numeric_cols:
            test_series = [True if z else (x == y) for x, y, z in zip(self.orig_df[col_name],
                                                                      self.numeric_vals_filled[col_name],
                                                                      self.orig_df[col_name].isna())]
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column consistently contains values that are valid numbers",
                allow_patterns=False
            )

    ##################################################################################################################
    # Data consistency checks for pairs of numeric columns
    ##################################################################################################################


    def _generate_larger_diff_range(self):
        """
        Patterns without exceptions: 'larger all_2' is consistently larger than 'larger all', which is consistently
            larger than 'larger rand'
        Patterns with exception: 'larger most' is consistently larger than 'larger rand' with 1 exception
        """
        self._add_synthetic_column('larger rand', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('larger all',   self.synth_df['larger rand'] + 50)
        self._add_synthetic_column('larger all_2', self.synth_df['larger rand'] + 200)
        self._add_synthetic_column('larger most',  self.synth_df['larger rand'] + 50)
        self.synth_df.loc[999, 'larger most'] = -5


    def _check_larger_diff_range(self, test_id):
        """
        Where a pair of numeric columns is flagged, this may indicate a anomaly with either column, or with the
        relationship between the two columns.

        This test checks pairs of columns that are on different scales, while LARGER_SAME_RANGE tests pairs of
        columns that are on the same scale. Typically, exceptions are more interesting in this test than in
        LARGER_SAME_SCALE, while patterns without exceptions are more interesting in LARGER_SAME_SCALE than in this
        test. This requires the featuers have some correlation.
        """
        self._check_two_cols_larger(test_id, require_same_scale=False)


    def _generate_larger_same_range(self):
        """
        Patterns without exceptions: 'larger_same_rng all_2' is consistently larger than 'larger_same_rng all', which is
            consistently larger than 'larger_same_rng rand'
        Patterns with exception: 'larger_same_rng most' is consistently larger than 'larger_same_rng rand' with one
            exception
        """
        self._add_synthetic_column('larger_same_rng rand',
            [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('larger_same_rng all',
            self.synth_df['larger_same_rng rand'] + np.random.randint(1, 10, self.num_synth_rows))
        self._add_synthetic_column('larger_same_rng all_2',
            self.synth_df['larger_same_rng all'] + np.random.randint(1, 10, self.num_synth_rows))
        self._add_synthetic_column('larger_same_rng most',
            self.synth_df['larger_same_rng all'] + np.random.randint(1, 10, self.num_synth_rows))
        self.synth_df.loc[999, 'larger_same_rng most'] = 0


    def _check_larger_same_range(self, test_id):
        """
        This is a similar test as LARGER, but this requires the two columns be on similar scales, where LARGER will
        consider where there are two columns, with one strictly larger than the other, to be a pattern. For example,
        if column A has a range 10 to 100 and column B has a range 110 to 150, this would be a pattern with respect
        to LARGER, but not LARGER_SAME_RANGE. With LARGER_SAME_RANGE, the two columns need to be in the same range,
        with each row having larger values in one of the two columns than the other.
        """
        self._check_two_cols_larger(test_id, require_same_scale=True)


    def _generate_much_larger(self):
        """
        Patterns without exceptions: 'much larger all' is consistently much larger than 'much larger rand', but this
            test is not in the shortlist of patterns to list.
        Patterns with exception: 'much larger most' is consistently much larger than 'much larger rand', with one
            exception: row 99 has a similar value.
        """
        self._add_synthetic_column('much larger rand',
                                    [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('much larger all',
                                    [random.randint(2000, 2100) for _ in range(self.num_synth_rows)])
        self.synth_df['much larger all'] = self.synth_df['much larger all'] + (3.0 * self.synth_df['much larger rand'])
        self.synth_df['much larger most'] = self.synth_df['much larger all']
        self.synth_df.loc[999, 'much larger most'] = 50


    def _check_much_larger(self, test_id):

        # Test if col_name_1 is consistently 10x larger than col_name_2
        num_pairs, col_pairs = self._get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        larger_pairs_with_bool_dict = self.get_larger_pairs_with_bool_dict()
        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict(force=True)
        much_larger_pairs_arr = []

        for cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining pair {cols_idx:,} of {len(col_pairs):,} pairs of numeric columns")

            if self.column_medians[col_name_1] < ((self.column_medians[col_name_2]) * 5):
                continue

            # Check the column is at least larger if not an order of magnitude larger
            if not larger_pairs_with_bool_dict[(col_name_1, col_name_2)]:
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample that the columns are correlated
            corr = scipy.stats.spearmanr(self.sample_df[col_name_1].fillna(self.sample_df[col_name_1].median()),
                                         self.sample_df[col_name_2].fillna(self.sample_df[col_name_2].median()))
            if corr.correlation < 0.9:
                continue

            # Test on a sample
            vals_arr_1 = self.sample_numeric_vals_filled[col_name_1]
            vals_arr_2 = self.sample_numeric_vals_filled[col_name_2]
            sample_series = np.where(
                vals_arr_2 != 0,
                (vals_arr_1 / vals_arr_2) > 10.0,
                False
            )
            sample_series = sample_series | self.sample_df[col_name_1].isna() | self.sample_df[col_name_2].isna()
            if sample_series.tolist().count(False) > 1:
                continue

            # Test on the full column
            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series = np.where(
                self.orig_df[col_name_2] != 0,
                (vals_arr_1 / vals_arr_2) > 10.0,
                False
            )
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | \
                          (self.orig_df[col_name_2] == 0)

            if test_series.tolist().count(False) > 0:
                self._process_analysis_binary(
                    test_id,
                    [col_name_2, col_name_1],
                    test_series,
                    f'"{col_name_1}" is consistently an order of magnitude or more larger than "{col_name_2}"',
                    ' (where values may still be larger, but not by the normal extent)')
            else:
                much_larger_pairs_arr.append([col_name_1, col_name_2])

        # Consolidate patterns where possible, in order to generate fewer, and more useful, patterns.
        patterns_arr = []  # The set of unique columns in each pattern
        patterns_pairs_arr = []  # The set of pair-wise relationships between columns in each pattern
        for col_name_1, col_name_2 in much_larger_pairs_arr:
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
                    desc = (f'"{patterns_pairs_arr[pattern_idx][0][0]}" is larger than '
                            f'"{patterns_pairs_arr[pattern_idx][0][1]}" consistently by an order of magnitude or more')
            else:
                desc = "There is a consistent relationship in size of values between the the columns."
                for p in patterns_pairs_arr[pattern_idx]:
                    desc += f'\n"{p[0]}" is consistently an order of magnitude or more larger than "{p[1]}"'

            self._process_analysis_binary(
                test_id,
                cols,
                [True]*self.num_rows,
                desc,
                display_info={'patterns_pairs_arr': patterns_pairs_arr[pattern_idx]}
            )


    def _generate_similar_wrt_ratio(self):
        """
        Patterns without exceptions: 'sim wrt ratio rand_a' and 'sim wrt ratio all' are consistently similar
        Patterns with exception: 'sim wrt ratio rand_a' and 'sim wrt ratio most' are consistently similar, with 1
            exception. As well, 'sim wrt ratio all' and 'sim wrt ratio most' are consistently similar, with 1
            exception, but this is skipped as they have almost the same values.
        """
        self._add_synthetic_column('sim wrt ratio rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim wrt ratio rand_b', [random.choice([0, random.randint(1, 1000)]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim wrt ratio all', self.synth_df['sim wrt ratio rand_a'] * 1.01)
        self._add_synthetic_column('sim wrt ratio most', self.synth_df['sim wrt ratio rand_a'] * 1.01)
        self.synth_df.loc[999, 'sim wrt ratio most'] = 100_000


    def _check_similar_wrt_ratio(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        nunique_dict = self.get_nunique_dict()
        cols_same_bool_dict = self.get_cols_same_bool_dict()
        cols_same_count_dict = self.get_cols_same_count_dict()
        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and len(numeric_pairs_list) > 5000 and pair_idx > 0 and pair_idx % 5000 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(numeric_pairs_list):,} pairs of numeric columns")

            # Skip where the columns have few unique values
            if (nunique_dict[col_name_1] < 10) or (nunique_dict[col_name_2] < 10):
                continue

            # Skip where the columns are almost the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue
            num_same = cols_same_count_dict[tuple(sorted([col_name_1, col_name_2]))]

            # Skip where there are many zero values. Rows with null values are not tested.
            if self.orig_df[col_name_1].tolist().count(0) > (self.num_valid_rows[col_name_1] * 0.75):
                continue
            if self.orig_df[col_name_2].tolist().count(0) > (self.num_valid_rows[col_name_2] * 0.75):
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series_a = np.where(
                self.orig_df[col_name_2] != 0,
                abs(vals_arr_1 / vals_arr_2),
                1.0
            )
            test_series = np.where((test_series_a > 0.5) & (test_series_a < 2.0), True, False)
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                (f'"{col_name_1}" and "{col_name_2}" have consistently similar values in terms of their ratio (with '
                 f'{int(num_same)} rows having identical values). Rows with values of zero are not tested.'),
                display_info={'Ratio': test_series_a}
            )


    def _generate_similar_wrt_difference(self):
        """
        Patterns without exceptions: 'sim wrt diff all' is consistently similar to 'sim wrt diff rand'
        Patterns with exception: 'sim wrt diff most' is consistently similar to 'sim wrt diff rand' with one exception
        """
        self._add_synthetic_column('sim wrt diff rand', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim wrt diff all', self.synth_df['sim wrt diff rand'] - 2.5)
        self._add_synthetic_column('sim wrt diff most', self.synth_df['sim wrt diff rand'] - 2.5)
        self.synth_df.loc[999, 'sim wrt diff most'] = 100_000


    def _check_similar_wrt_difference(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        nunique_dict = self.get_nunique_dict()
        cols_same_bool_dict = self.get_cols_same_bool_dict()
        cols_same_count_dict = self.get_cols_same_count_dict()
        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and (pair_idx > 0) and (pair_idx % 5000) == 0:
                    print(f"  Examining column set {pair_idx:,} of {len(numeric_pairs_list):,} pairs of numeric columns")

            # Skip columns that have few unique values.
            # todo: ensure this is a consistent mimimum for all tests, make it a class variable
            if (nunique_dict[col_name_1] < 10) or (nunique_dict[col_name_2] < 10):
                continue

            # Skip where the columns are almost the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Skip pairs where only rare rows have no nulls. Rows with null values are not tested.
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            diff_medians = abs(self.column_medians[col_name_1] - self.column_medians[col_name_2])
            if diff_medians > min(self.column_medians[col_name_1], self.column_medians[col_name_2]):
                continue

            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series_a = (vals_arr_1 - vals_arr_2)
            test_series = abs(test_series_a) < (min(self.column_medians[col_name_1], self.column_medians[col_name_2]) / 10.0)
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                (f'"{col_name_1}" and "{col_name_2}" have consistently similar values in terms of their absolute '
                 f'difference (with {int(cols_same_count_dict[tuple(sorted([col_name_1, col_name_2]))])} rows having '
                 f'identical values). Rows with values of zero are not tested.'),
                display_info={"Diff": test_series_a}
            )


    def _generate_similar_to_inverse(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('sim to inv rand', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim to inv all', 1 / self.synth_df['sim to inv rand'])
        self._add_synthetic_column('sim to inv most', 1 / self.synth_df['sim to inv rand'])
        self.synth_df.loc[999, 'sim to inv most'] = 500


    def _check_similar_to_inverse(self, test_id):
        """
        This skips columns that contain many 0 values.
        """

        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        zeros_limit = self.num_rows // 10
        count_zeros_dict = {}
        for col_name in self.numeric_cols:
            count_zeros_dict[col_name] = (self.orig_df[col_name] == 0).tolist().count(True)

        for _pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if (count_zeros_dict[col_name_1] > zeros_limit) or (count_zeros_dict[col_name_2] > zeros_limit):
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample
            vals_arr_1 = self.sample_numeric_vals_filled[col_name_1]
            vals_arr_2 = self.sample_numeric_vals_filled[col_name_2]
            sample_series = [math.isclose(x, 1/y) if y != 0 else True
                             for x, y in zip(vals_arr_1, vals_arr_2)]
            if sample_series.count(False) > 1:
                continue

            # Test on the full columns
            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series = np.array([math.isclose(x, 1/y) if y != 0 else True
                                    for x, y in zip(vals_arr_1, vals_arr_2)])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently the inverse of "{col_name_2}"')


    def _generate_similar_to_negative(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('sim to neg rand', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sim to neg all', -1 * self.synth_df['sim to neg rand'])
        self._add_synthetic_column('sim to neg most', -1 * self.synth_df['sim to neg rand'])
        self.synth_df.loc[999, 'sim to neg most'] = 500


    def _check_similar_to_negative(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        for _pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            vals_arr_1 = convert_to_numeric(self.sample_df[col_name_1], self.column_medians[col_name_1])
            vals_arr_2 = convert_to_numeric(self.sample_df[col_name_2], self.column_medians[col_name_2])
            sample_series = [math.isclose(x, -y) if y != 0 else True
                             for x, y in zip(vals_arr_1, vals_arr_2)]
            if sample_series.count(False) > 1:
                continue

            vals_arr_1 = convert_to_numeric(self.orig_df[col_name_1], self.column_medians[col_name_1])
            vals_arr_2 = convert_to_numeric(self.orig_df[col_name_2], self.column_medians[col_name_2])
            test_series = np.array([math.isclose(x, -y) and x != 0
                                    for x, y in zip(vals_arr_1, vals_arr_2)])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently the negative of "{col_name_2}"')


    def _generate_constant_sum(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('constant sum 1', [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('constant sum 2', 5000 - self.synth_df['constant sum 1'])
        self._add_synthetic_column('constant sum 3', 5000 - self.synth_df['constant sum 1'])
        self.synth_df.at[999, 'constant sum 3'] = self.synth_df.at[999, 'constant sum 3'] * 2.0


    def _check_constant_sum(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        min_unique_vals = math.sqrt(self.num_rows)
        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and (pair_idx > 0) and (pair_idx % 5000) == 0:
                print(f"  Examining column set {pair_idx:,} of {num_pairs:,} pairs of numeric columns")

            # Check there are a sufficient number of unique values in both columns
            if self.orig_df[col_name_1].nunique() < min_unique_vals:
                continue
            if self.orig_df[col_name_2].nunique() < min_unique_vals:
                continue

            if not self.check_columns_same_scale_2(col_name_1, col_name_2, order=10):
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample
            vals_arr_1 = self.sample_numeric_vals_filled[col_name_1]
            vals_arr_2 = self.sample_numeric_vals_filled[col_name_2]
            sample_sums = vals_arr_1 + vals_arr_2
            nmad = np.nanmedian(np.absolute(sample_sums - np.nanmedian(sample_sums))) / np.nanmedian(sample_sums) \
                if np.nanmedian(sample_sums) != 0 \
                else 0.0
            if nmad > 0.01:
                continue

            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            sums_series = vals_arr_1 + vals_arr_2
            # Get the median absolute deviation of the differences, normalized by the median. Note, the scipy
            # implementation does not handle Null values.
            nmad = np.nanmedian(np.absolute(sums_series - np.nanmedian(sums_series))) / np.nanmedian(sums_series) \
                if np.nanmedian(sums_series) != 0 \
                else 0.0
            if nmad < 0.01:
                test_series = abs(sums_series - np.nanmedian(sums_series)) < \
                              abs(0.01 * np.nanmedian(sums_series))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The sum of "{col_name_1}" and "{col_name_2}" is consistently close to '
                     f'{np.nanmedian(sums_series)}'),
                    display_info={'Sum': sums_series}
                )


    def _generate_constant_diff(self):
        """
        Patterns without exceptions: the difference in 'constant diff 1' and 'constant diff 2' is consistently 1000
        Patterns with exception: the difference in 'constant diff 1' and 'constant diff 3' is consistently 1000 with
            the exception of row 999
        """
        self._add_synthetic_column('constant diff 1', [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('constant diff 2', 1000 + self.synth_df['constant diff 1'])
        self._add_synthetic_column('constant diff 3', 1000 + self.synth_df['constant diff 1'])
        self.synth_df.at[999, 'constant diff 3'] = self.synth_df.at[999, 'constant diff 3'] * 2.0


    def _check_constant_diff(self, test_id):
        # todo: in examples not flagged, don't include None/NaN -- do that always with exceptions for some tests i think
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        min_unique_vals = math.sqrt(self.num_rows)
        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and (pair_idx > 0) and (pair_idx % 5000) == 0:
                print(f"  Examining column set {pair_idx:,} of {num_pairs:,} pairs of numeric columns")

            # Check there are a sufficient number of unique values in both columns
            if self.orig_df[col_name_1].nunique() < min_unique_vals:
                continue
            if self.orig_df[col_name_2].nunique() < min_unique_vals:
                continue

            if not self.check_columns_same_scale_2(col_name_1, col_name_2, order=100):
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Test on a sample, using the rows with values in both columns
            sample_non_null = (self.sample_df[col_name_1].notna() & self.sample_df[col_name_2].notna()).values
            if not sample_non_null.any():
                continue
            vals_arr_1 = self.sample_numeric_vals_filled[col_name_1][sample_non_null]
            vals_arr_2 = self.sample_numeric_vals_filled[col_name_2][sample_non_null]
            diffs_series = vals_arr_1 - vals_arr_2
            if diffs_series.median() == 0:
                continue  # If the two columns are the same, there is a separate test for that.
            nmad = scipy.stats.median_abs_deviation(diffs_series) / statistics.median(diffs_series) \
                if statistics.median(diffs_series) != 0 \
                else 0.0
            if nmad > 0.01:
                continue

            # Get the median absolute deviation of the differences, normalized by the median. Rows with a missing value
            # in either column neither support nor violate the pattern, so these use only the other rows.
            non_null = self.orig_df[col_name_1].notna() & self.orig_df[col_name_2].notna()
            vals_arr_1 = self.numeric_vals_filled[col_name_1][non_null]
            vals_arr_2 = self.numeric_vals_filled[col_name_2][non_null]
            diffs_series = abs(vals_arr_1 - vals_arr_2)
            diffs_series = diffs_series.replace([np.inf, -np.inf, np.nan], diffs_series.median())
            nmad = scipy.stats.median_abs_deviation(diffs_series) / statistics.median(diffs_series)\
                if statistics.median(diffs_series) != 0 \
                else 0.0
            if nmad < 0.01:
                test_series = abs(diffs_series - statistics.median(diffs_series)) <= \
                              abs(0.01 * statistics.median(diffs_series))
                test_series = test_series.reindex(self.orig_df.index, fill_value=True)
                # todo: for all check_constant_* tests, check it wouldn't work as well to just use one column. Is the other just zero?
                # here, checking the same scale should work but doesn't seem to.

                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The difference of "{col_name_1}" and "{col_name_2}" is consistently close to '
                     f'{statistics.median(diffs_series)}'),
                    display_info={'Diff': diffs_series.reindex(self.orig_df.index)}
                )


    def _generate_constant_product(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('constant product 1',
                                    [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('constant product 2', 5000 / self.synth_df['constant product 1'])
        self._add_synthetic_column('constant product 3', 5000 / self.synth_df['constant product 1'])
        self.synth_df.at[999, 'constant product 3'] = self.synth_df.at[999, 'constant product 3'] * 2.0


    def _check_constant_product(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        min_unique_vals = math.sqrt(self.num_rows)
        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and (pair_idx > 0) and (pair_idx % 5000) == 0:
                print(f"  Examining column set {pair_idx:,} of {num_pairs:,} pairs of numeric columns")

            # Check there are a sufficient number of unique values in both columns
            if self.orig_df[col_name_1].nunique() < min_unique_vals:
                continue
            if self.orig_df[col_name_2].nunique() < min_unique_vals:
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # For this test, we do not check the 2 columns are on the same scale, as they can be on quite different
            # scales and still produce a constant product in a meaningful way.

            # Test on a sample, using the rows with values in both columns
            sample_non_null = (self.sample_df[col_name_1].notna() & self.sample_df[col_name_2].notna()).values
            if not sample_non_null.any():
                continue
            vals_arr_1 = self.sample_numeric_vals_filled[col_name_1][sample_non_null]
            vals_arr_2 = self.sample_numeric_vals_filled[col_name_2][sample_non_null]
            sample_series = vals_arr_1 * vals_arr_2
            nmad = scipy.stats.median_abs_deviation(sample_series) / statistics.median(sample_series) \
                if statistics.median(sample_series) != 0 \
                else 0.0
            if nmad > 0.01:
                continue

            # Rows with a missing value in either column neither support nor violate the pattern: their product is NaN
            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series_a = abs(vals_arr_1 * vals_arr_2).where(
                self.orig_df[col_name_1].notna() & self.orig_df[col_name_2].notna())

            # Get the median absolute deviation of the products, normalized by the median
            nmad = np.nanmedian(np.absolute(test_series_a - np.nanmedian(test_series_a))) / np.nanmedian(test_series_a) \
                if np.nanmedian(test_series_a) != 0 \
                else 0.0
            if nmad < 0.01:
                test_series = abs(test_series_a - np.nanmedian(test_series_a)) < abs(0.01 * np.nanmedian(test_series_a))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The product of "{col_name_1}" and "{col_name_2}" is consistently close to '
                     f'{np.nanmedian(test_series_a)}'),
                    display_info={'Product': test_series_a}
                )


    def _generate_constant_ratio(self):
        """
        Patterns without exceptions: 'constant ratio 2' is consistently 5000 * 'constant ratio 1'
        Patterns with exception: 'constant ratio 3' is consistently 5000 * 'constant ratio 1', with 1 exception
        """
        self._add_synthetic_column('constant ratio 1', [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('constant ratio 2', 5000 * self.synth_df['constant ratio 1'])
        self._add_synthetic_column('constant ratio 3', 5000 * self.synth_df['constant ratio 1'])
        self.synth_df.at[999, 'constant ratio 3'] = self.synth_df.at[999, 'constant ratio 3'] * 2.0


    def _check_constant_ratio(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        # Determine the number of zeros in each numeric column
        count_zeros_dict = {}
        for col_name in self.numeric_cols:
            count_zeros_dict[col_name] = self.num_rows - np.count_nonzero(self.orig_df[col_name])

        min_unique_vals = math.sqrt(self.num_rows)
        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                print(f"  Examining pair number {pair_idx:,} of {len(numeric_pairs_list):,} pairs of numeric columns")

            # Check there are a sufficient number of unique values in both columns
            if self.orig_df[col_name_1].nunique() < min_unique_vals:
                continue
            if self.orig_df[col_name_2].nunique() < min_unique_vals:
                continue

            # Check if both columns have too many zeros.
            if (count_zeros_dict[col_name_1] > self.freq_contamination_level) and \
                    (count_zeros_dict[col_name_2] > self.freq_contamination_level):
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # If just col_name_2 has many zeros, swap the columns
            if count_zeros_dict[col_name_2] > self.freq_contamination_level:
                temp = col_name_1
                col_name_1 = col_name_2
                col_name_2 = temp

            # Test first on a sample, using the rows with values in both columns
            sample_non_null = (self.sample_df[col_name_1].notna() & self.sample_df[col_name_2].notna()).values
            if not sample_non_null.any():
                continue
            vals_arr_1 = self.sample_numeric_vals_filled[col_name_1][sample_non_null]
            vals_arr_2 = self.sample_numeric_vals_filled[col_name_2][sample_non_null]
            sample_series = list(map(safe_div, vals_arr_1, vals_arr_2))
            nmad = scipy.stats.median_abs_deviation(sample_series) / np.nanmedian(sample_series) \
                if np.nanmedian(sample_series) != 0 \
                else 0.0

            # Skip if the variance in the ratios (measured as median absolute deviation) is too large relative to
            # the median for the column.
            if abs(nmad) > 0.01:
                continue

            # Skip if there is a trivial ratio, if the columns are the same, or the negative of each other, for which
            # there are specific tests.
            if np.nanmedian(sample_series) in [1.0, -1.0, 0.0]:
                continue

            # Rows with a missing value in either column neither support nor violate the pattern: their ratio is NaN
            non_null = self.orig_df[col_name_1].notna() & self.orig_df[col_name_2].notna()
            vals_arr_1 = self.numeric_vals_filled[col_name_1].where(non_null)
            vals_arr_2 = self.numeric_vals_filled[col_name_2].where(non_null)
            test_series_a = list(map(safe_div, vals_arr_1, vals_arr_2))

            # Get the median absolute deviation of the ratios, normalized by the median
            nmad = np.nanmedian(np.absolute(test_series_a - np.nanmedian(test_series_a))) / np.nanmedian(test_series_a) \
                if np.nanmedian(test_series_a) != 0 \
                else 0.0
            if abs(nmad) < 0.01:
                test_series = abs(test_series_a - np.nanmedian(test_series_a)) < abs(0.01 * np.nanmedian(test_series_a))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The ratio of "{col_name_1}" and "{col_name_2}" is consistently close to '
                     f'{np.nanmedian(test_series_a)}'),
                    display_info={'Ratio': test_series_a}
                )


    def _generate_even_multiple(self):
        """
        Patterns without exceptions: 'even_multiple all' is consistently an even integer multiple of
            'even_multiple rand'
        Patterns with exception: 'even_multiple most' is consistently an even integer multiple of
            'even_multiple rand' with one exception
        """
        self._add_synthetic_column('even_multiple rand',
                                    [random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('even_multiple all',
                                    np.random.randint(1, 1_000, 1000) * self.synth_df['even_multiple rand'])
        self._add_synthetic_column('even_multiple most', self.synth_df['even_multiple all'])
        self.synth_df.at[999, 'even_multiple most'] = self.synth_df.at[999, 'even_multiple most'] * 1.25 + 1


    def _check_even_multiple(self, test_id):
        """
        For each pair of numeric columns, A and B, this checks if A is consistently an even integer multiple of B.
        This skips rows where B is 0 or missing.
        """

        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                print(f"  Examining pair number {pair_idx:,} of {num_pairs:,} pairs of numeric columns")

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Rows with a missing value in either column neither support nor violate the pattern, so are not counted
            non_null = self.orig_df[col_name_1].notna() & self.orig_df[col_name_2].notna()
            num_valid = len(np.where(
                non_null &
                (self.orig_df[col_name_1] != 0) &
                (self.orig_df[col_name_1] != 1) &
                (self.orig_df[col_name_1] != -1) &
                (self.orig_df[col_name_2] != 0) &
                (self.orig_df[col_name_2] != 1) &
                (self.orig_df[col_name_2] != -1))[0])
            if num_valid < (non_null.sum() / 2):
                continue

            if self.orig_df[col_name_1].isna().sum() > (self.num_rows * 0.75):
                continue
            if self.orig_df[col_name_2].isna().sum() > (self.num_rows * 0.75):
                continue
            if self.orig_df[col_name_2].tolist().count(0) > (self.num_rows * 0.75):
                continue

            val_arr_1 = self.numeric_vals_filled[col_name_1].where(non_null)
            val_arr_2 = self.numeric_vals_filled[col_name_2].where(non_null)
            test_series = np.where(val_arr_2 != 0, val_arr_1 / val_arr_2, val_arr_1).tolist()

            # Remove cases where there is trivially an even multiple
            if (test_series.count(1) + test_series.count(0) + test_series.count(-1) + \
                test_series.count(np.inf)) > self.freq_contamination_level:
                continue

            test_series = [float(x).is_integer() for x in test_series]
            test_series = np.array(test_series) | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'"{col_name_1}" is consistently an even integer multiple of "{col_name_2}"')


    def _generate_rare_combination(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        common_vals = []
        for i in range(8):
            for j in range(80):
                if i == 0 or i == 7 or j == 0 or j == 70:
                    common_vals.append([i, j])
        rare_vals = [[3, 30]]
        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self._add_synthetic_column('rare_combo all_a', data[:, 0])
        self._add_synthetic_column('rare_combo all_b', data[:, 1])


    def _check_rare_combination(self, test_id):
        """
        This flags only pairs of values where the combination is unusual, but neither value by itself would be flagged.
        No rows are removed from the data during evaluation, only from the output once generated.
        If the data contains clusters, this will flag points outside of the clusters or on the fringes. If there is
        a single cluster, it will flag points on the fringes of that cluster. Data is often concentrated in the
        lower-left of the 2d space; in this case the test will flag any points far from the origin, which will typically
        be points with both values not quite extreme enough to be flagged in either single dimension.
        """
        def get_bins(col_name, num_bins):
            if self.orig_df[col_name].dropna().nunique() < num_bins:
                return None, []
            bins = []
            min_val = self.numeric_vals[col_name].min()
            max_val = self.numeric_vals[col_name].max()
            col_range = max_val - min_val
            bin_width = col_range / num_bins
            for i in range(num_bins+1):
                bins.append(min_val + (i * bin_width))
            bins[0] = bins[0] - (bin_width / 10.0)
            bins[-1] = bins[-1] + (bin_width / 10.0)
            bin_labels = [int(x) for x in range(len(bins)-1)]
            # Missing values are left out of the bins, so are not in any cell
            vals_arr = self.numeric_vals_filled[col_name].where(self.orig_df[col_name].notna())
            binned_values = pd.cut(vals_arr, bins, labels=bin_labels)
            return binned_values, bins

        column_bins = {}
        column_bin_boundaries = {}
        num_bins = 6
        for col_name in self.numeric_cols:
            bins, bin_boundaries = get_bins(col_name, num_bins)
            column_bins[col_name] = bins
            column_bin_boundaries[col_name] = bin_boundaries

        cell_limit = math.ceil(self.freq_contamination_level / 9)

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 5_000 == 0:
                print(f"  Examining pair {pair_idx:,} of {num_pairs:,} pairs of numeric columns.")

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            bins_1 = column_bins[col_name_1]
            bins_2 = column_bins[col_name_2]
            if (bins_1 is None) or (bins_2 is None):
                continue

            vc1 = bins_1.value_counts()
            vc2 = bins_2.value_counts()
            counts_col_1 = vc1.sort_index().values
            counts_col_2 = vc2.sort_index().values

            # There are 6x6=36 cells in the 2d space. However, this test does not flag values that are extreme in
            # either dimension, so does not flag the outer ring of cells, checking only the inner 4x4=16 cells.
            # Loop through the 16 internal cells and identify cells that both have low count and their 8 neighboring
            # cells have low count. We first get the counts of all 36 cells.
            cell_counts = np.zeros((num_bins, num_bins))
            test_series = [1] * self.num_rows
            for i in range(num_bins):
                subset_i = [(x, y) for x, y in zip(bins_1, bins_2) if x == i]
                for j in range(num_bins):
                    cell_counts[i][j] = len([1 for x, y in subset_i if y == j])
            flat_list = [item for sublist in cell_counts for item in sublist]

            # Check there are at least 9 cells with counts less than the contamination rate
            num_less_contamination = len([1 for x in flat_list if x < self.freq_contamination_level])
            if num_less_contamination < 9:
                continue

            # Identify the internal cells with low counts.
            low_count_cells = []
            for i in range(1, num_bins-1):
                for j in range(1, num_bins-1):
                    if cell_counts[i][j] <= cell_limit:
                        if counts_col_1[i] > self.freq_contamination_level and counts_col_2[j] > self.freq_contamination_level:
                            low_count_cells.append((i, j))

            flagged_cells = []
            for i, j in low_count_cells:
                # These checks are ordered most to least likely to fail
                if cell_counts[i][j] == 0 or \
                   cell_counts[i][j-1] > cell_limit or \
                   cell_counts[i][j+1] > cell_limit or \
                   cell_counts[i-1][j] > cell_limit or \
                   cell_counts[i+1][j] > cell_limit or \
                   cell_counts[i-1][j-1] > cell_limit or \
                   cell_counts[i-1][j+1] > cell_limit or \
                   cell_counts[i+1][j-1] > cell_limit or \
                   cell_counts[i+1][j+1] > cell_limit:
                    continue
                flagged_cells.append((i, j))

            if len(flagged_cells) == 0:
                continue

            # Check if enough rows have been flagged that we do not consider this a pattern
            total_flagged = 0
            for i, j in flagged_cells:
                total_flagged += cell_counts[i][j]
            if total_flagged > self.freq_contamination_level:
                continue

            # Loop through each flagged cell and flag all rows in those cells
            for i, j in flagged_cells:
                flagged_idxs = [x for x in bins_1.index if bins_1[x] == i and bins_2[x] == j]
                for f in flagged_idxs:
                    test_series[f] = False

            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                "One or more rare combinations of values were found",
                allow_patterns=False,
                display_info={'bins_1': column_bin_boundaries[col_name_1], 'bins_2': column_bin_boundaries[col_name_2]}
            )


    def _generate_correlated(self):
        """
        Patterns without exceptions: "correlated rand_a" is consistently correlated with "correlated rand_b"
        Patterns with exception: 'correlated most' is consistently correlated with "correlated rand_a" and
            "correlated rand_b" with exceptions. We do not flag the correlation between 'correlated most' and
            'correlated rand_b' as they are almost the same.
        """
        list_a = sorted([random.randint(1, 1_000) for _ in range(self.num_synth_rows)])
        list_b = sorted([random.randint(1, 2_000) for _ in range(self.num_synth_rows)])
        c = list(zip(list_a, list_b))
        random.shuffle(c)
        list_a, list_b = zip(*c)

        self._add_synthetic_column('correlated rand_a', list_a)
        self._add_synthetic_column('correlated rand_b', list_b)
        list_b = list(list_b)
        list_b[-1] = list_b[-1] * 10.0
        self._add_synthetic_column('correlated most', list_b)


    def _check_correlated(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()
        cols_same_count_dict = self.get_cols_same_count_dict()
        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(numeric_pairs_list):,} pairs of numeric columns")

            if self.orig_df[col_name_1].nunique(dropna=True) < self.freq_contamination_level:
                continue
            if self.orig_df[col_name_2].nunique(dropna=True) < self.freq_contamination_level:
                continue

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Rows with a missing value in either column neither support nor violate the pattern, so the correlation
            # and the percentiles are calculated on the other rows
            non_null = self.orig_df[col_name_1].notna() & self.orig_df[col_name_2].notna()
            val_arr_1 = self.numeric_vals_filled[col_name_1].where(non_null)
            val_arr_2 = self.numeric_vals_filled[col_name_2].where(non_null)
            if val_arr_1.nunique() < 3:
                continue
            if val_arr_2.nunique() < 3:
                continue

            # Skip columns that are almost entirely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue
            num_same = cols_same_count_dict[tuple(sorted([col_name_1, col_name_2]))]

            spearman_corr = abs(val_arr_1.corr(val_arr_2, method='spearman'))
            if spearman_corr >= 0.995:
                col_1_percentiles = self.orig_df[col_name_1].where(non_null).rank(pct=True)
                col_2_percentiles = self.orig_df[col_name_2].where(non_null).rank(pct=True)

                # Test for positive correlation
                test_series = np.array([abs(x-y) < 0.2 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'"{col_name_1}" is consistently similar (with {num_same * 100.0 / self.num_rows}% identical '
                     f'values), with respect to rank, to "{col_name_2}" with a Spearman correlation of {spearman_corr}'),
                    display_info={'col_1_percentiles': col_1_percentiles, 'col_2_percentiles': col_2_percentiles}
                )

                # Test for negative correlation
                test_series = np.array([abs(x-(1.0 - y)) < 0.2 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'"{col_name_1}" is consistently inversely similar in rank (with '
                     f'{num_same * 100.0 / self.num_rows}% identical values),to "{col_name_2}" with an absolute '
                     f'Spearman correlation of {spearman_corr}'),
                    display_info={'col_1_percentiles': col_1_percentiles, 'col_2_percentiles': col_2_percentiles}
                )


    def _generate_matched_zero(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('matched zero rand_a', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched zero rand_b', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched zero all', self.synth_df['matched zero rand_a'])
        self._add_synthetic_column('matched zero most', self.synth_df['matched zero rand_a'])
        if self.synth_df.loc[999, 'matched zero most'] == 0:
            self.synth_df.loc[999, 'matched zero most'] = 1
        else:
            self.synth_df.loc[999, 'matched zero most'] = 0


    def _check_matched_zero(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()
        is_missing_dict = self.get_is_missing_dict()

        for _pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            num_not_missing_1 = is_missing_dict[col_name_1].tolist().count(False)
            num_not_missing_2 = is_missing_dict[col_name_2].tolist().count(False)
            zero_indicator_1 = self.orig_df[col_name_1] == 0
            num_zeros_1 = zero_indicator_1.tolist().count(True)
            zero_indicator_2 = self.orig_df[col_name_2] == 0
            num_zeros_2 = zero_indicator_2.tolist().count(True)
            if num_zeros_1 < (num_not_missing_1 * 0.01) or \
                    num_zeros_1 > (num_not_missing_1 * 0.99) or \
                    num_zeros_2 < (num_not_missing_2 * 0.01) or \
                    num_zeros_2 > (num_not_missing_2 * 0.99):
                continue
            test_series = np.array([x == y for x, y in zip(zero_indicator_1, zero_indicator_2)])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()

            # Determine if there is already a pattern found, which these columns are part of. If so, simply add
            # the new column to the pattern.
            found_existing_pattern = False
            if test_series.tolist().count(False) == 0:
                for pattern_idx, existing_pattern in enumerate(self.patterns_arr):
                    if existing_pattern[0] != test_id:
                        continue
                    pattern_cols = [x.lstrip('"').rstrip('"') for x in existing_pattern[1].split(" AND ")]
                    if col_name_1 in pattern_cols or col_name_2 in pattern_cols:
                        if f'"{col_name_1}"' not in existing_pattern[1]:
                            existing_pattern[1] += f' AND "{col_name_1}"'
                        if f'"{col_name_2}"' not in existing_pattern[1]:
                            existing_pattern[1] += f' AND "{col_name_2}"'
                        existing_pattern[2] = (f'The columns consistently have zero values in the same rows, '
                                               f'with {num_zeros_1} zero values.')
                        self.patterns_arr[pattern_idx] = existing_pattern

                        # Update self.col_to_original_cols_dict
                        self.col_to_original_cols_dict[existing_pattern[1]] = pattern_cols

                        found_existing_pattern = True
                        break
            if found_existing_pattern:
                continue

            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                (f'The columns "{col_name_1}" (with {num_zeros_1} zero values) and "{col_name_2}" (with '
                 f'{num_zeros_2} zero values) consistently have zero values in the same rows')
            )


    def _generate_opposite_zero(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('opp_zero all_a',
                                    [random.randint(1, 500) if x % 2 == 0 else 0 for x in range(self.num_synth_rows)])
        self._add_synthetic_column('opp_zero all_b',
                                    [random.randint(1000, 2000) if x % 2 == 1 else 0 for x in range(self.num_synth_rows)])
        self._add_synthetic_column('opp_zero most', self.synth_df['opp_zero all_b'])
        self.synth_df.loc[998, 'opp_zero most'] = 600


    def _check_opposite_zero(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        for _pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            zero_indicator_1 = self.orig_df[col_name_1] == 0
            num_zeros_1 = zero_indicator_1.tolist().count(True)
            zero_indicator_2 = self.orig_df[col_name_2] == 0
            num_zeros_2 = zero_indicator_2.tolist().count(True)
            if num_zeros_1 < (self.num_rows * 0.01) or \
                num_zeros_1 > (self.num_rows * 0.99) or \
                num_zeros_2 < (self.num_rows * 0.01) or \
                num_zeros_2 > (self.num_rows * 0.99):
                continue
            test_series = np.array([x != y for x, y in zip(zero_indicator_1, zero_indicator_2)])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                (f'The columns "{col_name_1}" (with {num_zeros_1} zero values) and "{col_name_2}" (with '
                 f'{num_zeros_2} zero values) consistently have zero values in the opposite rows')
            )


    def _generate_running_sum(self):
        """
        Patterns without exceptions: 'run_sum all' is a running total of 'run sum rand'
        Patterns with exception: 'run_sum most' is a running total of 'run sum rand' with the exception of row 999.
            Note: this may be missed where Null values are added.
        """
        self._add_synthetic_column('run_sum rand', [random.randint(1, 500) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('run_sum all', self.synth_df['run_sum rand'].cumsum())
        self._add_synthetic_column('run_sum most', self.synth_df['run_sum all'])
        self.synth_df.loc[999, 'run_sum most'] = 100


    def _check_running_sum(self, test_id):
        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()
        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(numeric_pairs_list):,} pairs of numeric columns")

            # Skip pairs where only rare rows have no nulls. Rows with null values, or following a null value, are not
            # tested.
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # Skip where the columns have few unique values
            if self.orig_df[col_name_1].nunique() < 10:
                continue
            if self.orig_df[col_name_2].nunique() < 10:
                continue

            # Skip where the two columns are very similar (suggesting a different relationship than a running sum)
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # This is more robust than checking the cumulative sum, which can be thrown off by missing or
            # inaccurate values.
            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series_a = vals_arr_1.shift(1) + vals_arr_2
            test_series = self.orig_df[col_name_1] == test_series_a
            test_series.loc[0] = True  # The first row can not be tested for running totals.
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_1].shift(1).isna()
            self._process_analysis_binary(
                test_id,
                [col_name_2, col_name_1],
                test_series,
                f'Column "{col_name_1}" consistently contains a running sum of "{col_name_2}"',
                '',
                display_info={'RUNNING SUM': test_series_a}
            )


    def _generate_a_rounded_b(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        # todo: create more rand columns, so there's less overlap and smaller results

        self._add_synthetic_column('a_rounded_b rand', [random.random() * 10_000 for _ in range(self.num_synth_rows)])

        # Test floor function
        self._add_synthetic_column('a_rounded_b all_a', self.synth_df['a_rounded_b rand'].apply(np.floor))
        self._add_synthetic_column('a_rounded_b most_a', self.synth_df['a_rounded_b all_a'])
        self.synth_df.loc[999, 'a_rounded_b most_a'] = 100.8

        # Test ceil function
        self._add_synthetic_column('a_rounded_b all_b', self.synth_df['a_rounded_b rand'].apply(np.ceil))
        self._add_synthetic_column('a_rounded_b most_b', self.synth_df['a_rounded_b all_b'])
        self.synth_df.loc[999, 'a_rounded_b most_b'] = 100.8

        # Test rounding to 1's
        self._add_synthetic_column('a_rounded_b all_c', self.synth_df['a_rounded_b rand'].apply(np.round))
        self._add_synthetic_column('a_rounded_b most_c', self.synth_df['a_rounded_b all_c'])
        self.synth_df.loc[999, 'a_rounded_b most_c'] = 100.8

        # Test rounding to 10's
        self._add_synthetic_column('a_rounded_b all_d', (self.synth_df['a_rounded_b rand'] / 10).apply(round) * 10)
        self._add_synthetic_column('a_rounded_b most_d', self.synth_df['a_rounded_b all_d'].astype(float))
        self.synth_df.loc[999, 'a_rounded_b most_d'] = 100.8

        # Test rounding to 100's
        self._add_synthetic_column('a_rounded_b all_e',(self.synth_df['a_rounded_b rand'] / 100).apply(round) * 100)
        self._add_synthetic_column('a_rounded_b most_e', self.synth_df['a_rounded_b all_e'].astype(float))
        self.synth_df.loc[999, 'a_rounded_b most_e'] = 100.8

        # Test rounding to 1000's
        self._add_synthetic_column('a_rounded_b all_f', (self.synth_df['a_rounded_b rand'] / 1000).apply(round) * 1000)
        self._add_synthetic_column('a_rounded_b most_f', self.synth_df['a_rounded_b all_f'].astype(float))
        self.synth_df.loc[999, 'a_rounded_b most_f'] = 100.8


    def _check_a_rounded_b(self, test_id):
        """
        This tests checks where one numeric column is the result of rounding, taking the ceiling, or taking the floor
        function onf another numeric column. This checks rounding to the even integer, 10's, 100's, and 1000's
        """

        # Cache the rounded, floor and ceiling values of each numeric column, as well as the number of digits after
        # the decimal point
        number_decimals_dict = {}

        sample_floor_dict = {}
        sample_ceil_dict = {}
        sample_round_dict = {}
        sample_round_10_dict = {}
        sample_round_100_dict = {}
        sample_round_1000_dict = {}

        floor_dict = {}
        ceil_dict = {}
        round_dict = {}
        round_10_dict = {}
        round_100_dict = {}
        round_1000_dict = {}

        for col_name in self.numeric_cols:
            # Missing values are kept as NaN, so they neither match nor violate any of the relationships
            vals_arr = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name]).where(
                self.orig_df[col_name].notna().to_numpy())
            vals_arr_sample = convert_to_numeric(self.sample_df[col_name], self.column_medians[col_name]).where(
                self.sample_df[col_name].notna().to_numpy())
            number_decimals_dict[col_name] = vals_arr.dropna().apply(get_num_decimal_digits)

            sample_floor_dict[col_name] = vals_arr_sample.apply(np.floor)
            sample_ceil_dict[col_name] = vals_arr_sample.apply(np.ceil)
            sample_round_dict[col_name] = vals_arr_sample.apply(np.round)
            sample_round_10_dict[col_name] = (vals_arr_sample / 10).apply(np.round) * 10
            sample_round_100_dict[col_name] = (vals_arr_sample / 100).apply(np.round) * 100
            sample_round_1000_dict[col_name] = (vals_arr_sample / 1000).apply(np.round) * 1000

            floor_dict[col_name] = vals_arr.apply(np.floor)
            ceil_dict[col_name] = vals_arr.apply(np.ceil)
            round_dict[col_name] = vals_arr.apply(np.round)
            round_10_dict[col_name] = (vals_arr / 10).apply(np.round) * 10
            round_100_dict[col_name] = (vals_arr / 100).apply(np.round) * 100
            round_1000_dict[col_name] = (vals_arr / 1000).apply(np.round) * 1000

        num_pairs, numeric_pairs_list = self._get_numeric_column_pairs()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        cols_same_count_dict = self.get_cols_same_count_dict()
        cols_same_bool_dict = self.get_cols_same_bool_dict()
        get_col_pairs_either_null_bool_dict = self.get_col_pairs_either_null_bool_dict(force=True)

        for pair_idx, (col_name_1, col_name_2) in enumerate(numeric_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(numeric_pairs_list):,} pairs of numeric columns")

            # Skip pairs of columns that are mostly identical
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue
            num_same = cols_same_count_dict[tuple(sorted([col_name_1, col_name_2]))]

            # Skip pairs where only rare rows have no nulls
            if get_col_pairs_either_null_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            # todo: exclude pairs where the medians are different. We should maybe get the number of digits in each
            #   column above, create a feature at the class level to store this, then in this loop skip any columns
            #   where more than contamination_level values have diff # digits.

            # Test floor function
            if number_decimals_dict[col_name_2].median() > 0:
                test_series = [is_missing(x) or is_missing(y) or x == math.floor(y)
                               for x, y in zip(self.sample_df[col_name_1], sample_floor_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [is_missing(x) or is_missing(y) or x == round(y)
                                   for x, y in zip(self.orig_df[col_name_1], floor_dict[col_name_2])]
                    if test_series.count(False) < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            (f'Column "{col_name_1}" is consistently the same as the floor of "{col_name_2}" (with '
                             f'{num_same} identical values)')
                        )
                        continue

            # Test ceil function
            if number_decimals_dict[col_name_2].median() > 0:
                test_series = [is_missing(x) or is_missing(y) or x == math.ceil(y)
                               for x, y in zip(self.sample_df[col_name_1], sample_ceil_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [is_missing(x) or is_missing(y) or x == round(y)
                                   for x, y in zip(self.orig_df[col_name_1], ceil_dict[col_name_2])]
                    if test_series.count(False) < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            (f'Column "{col_name_1}" is consistently the same as the ceiling of "{col_name_2}" (with '
                             f'{num_same} identical values')
                        )
                        continue

            # Test rounding to a single digit
            if (abs(self.column_medians[col_name_1]) >= 1.0) and (abs(self.column_medians[col_name_2]) >= 1.0):
                test_series = [is_missing(x) or is_missing(y) or x == round(y)
                               for x, y in zip(self.sample_df[col_name_1], sample_round_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [is_missing(x) or is_missing(y) or x == round(y)
                                   for x, y in zip(self.orig_df[col_name_1], round_dict[col_name_2])]
                    if test_series.count(False) < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            (f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2}" (with '
                             f'{num_same} identical values)')
                        )
                        continue

            # Test rounding to 10's
            if (abs(self.column_medians[col_name_1]) >= 10.0) and (abs(self.column_medians[col_name_2]) >= 10.0):
                test_series = [is_missing(x) or is_missing(y) or x == round(y)
                               for x, y in zip(self.sample_df[col_name_1], sample_round_10_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [is_missing(x) or is_missing(y) or x == round(y)
                                   for x, y in zip(self.orig_df[col_name_1], round_10_dict[col_name_2])]
                    if test_series.count(False) < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            (f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2}" to the 10s '
                             f'(with {num_same} identical values)')
                        )
                        continue

            # Test rounding to 100's
            if (abs(self.column_medians[col_name_1]) >= 100.0) and (abs(self.column_medians[col_name_2]) >= 100.0):
                test_series = [is_missing(x) or is_missing(y) or x == round(y)
                               for x, y in zip(self.sample_df[col_name_1], sample_round_100_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [is_missing(x) or is_missing(y) or x == round(y)
                                   for x, y in zip(self.orig_df[col_name_1], round_100_dict[col_name_2])]
                    if test_series.count(False) < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            (f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2}" to the 100s '
                             f'(with {num_same} identical values)')
                        )
                        continue

            # Test rounding to 1000's
            if (abs(self.column_medians[col_name_1]) >= 1000.0) and (abs(self.column_medians[col_name_2]) >= 1000.0):
                test_series = [is_missing(x) or is_missing(y) or x == round(y)
                               for x, y in zip(self.sample_df[col_name_1], sample_round_1000_dict[col_name_2])]
                if test_series.count(False) <= 1:
                    test_series = [is_missing(x) or is_missing(y) or x == round(y)
                                   for x, y in zip(self.orig_df[col_name_1], round_1000_dict[col_name_2])]
                    if test_series.count(False) < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            [col_name_2, col_name_1],
                            np.array(test_series),
                            (f'Column "{col_name_1}" is consistently the same as rounding "{col_name_2}" to the 1000s'
                             f'(with {num_same} identical values)')
                        )
                        continue

    ##################################################################################################################
    # Data consistency checks for pairs of columns, where one must be numeric
    ##################################################################################################################


    def _generate_matched_zero_missing(self):
        """
        Patterns without exceptions: 'matched zero miss rand_a' and 'matched zero miss all' are matched such that when
             'matched zero miss rand_a' has 0, 'matched zero miss all' has NaN.
        Patterns with exception: 'matched zero miss rand_a' and 'matched zero miss most' are matched such that when
             'matched zero miss rand_a' has 0, 'matched zero miss most' has NaN, with 1 exception.
        """
        self._add_synthetic_column('matched zero miss rand_a', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched zero miss rand_b', [random.randint(0, 10) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched zero miss all', self.synth_df['matched zero miss rand_a'])
        self.synth_df['matched zero miss all'] = self.synth_df['matched zero miss all'].replace(0, np.nan)
        self._add_synthetic_column('matched zero miss most', self.synth_df['matched zero miss rand_a'])
        self.synth_df['matched zero miss most'] = self.synth_df['matched zero miss most'].replace(0, np.nan)
        if pd.isna(self.synth_df.loc[999, 'matched zero miss most']):
            self.synth_df.loc[999, 'matched zero miss most'] = 1
        else:
            self.synth_df.loc[999, 'matched zero miss most'] = np.nan


    def _check_matched_zero_missing(self, test_id):
        """
        This may occur, for example if the numeric column is "Number of Children" and the other column is
        "Age First Child". If there are zero children, we expect Null in the other column.
        """

        # Calculate and cache the col_missing_arr for every column
        sample_col_missing_dict = {}
        col_missing_dict = {}
        count_missing_dict = {}
        for col_name in self.orig_df.columns:
            sample_col_missing_dict[col_name] = [is_missing(x) for x in self.orig_df.head(50)[col_name]]
            col_missing_dict[col_name] = [is_missing(x) for x in self.orig_df[col_name]]
            count_missing_dict[col_name] = col_missing_dict[col_name].count(True)

        for col_idx, col_name_1 in enumerate(self.numeric_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 50 == 0:
                    print(f"  Examining column: {col_idx} of {len(self.numeric_cols)} numeric columns)")
            # We use the first 50 rows of orig_df instead of sample_df, as it has few Null values.
            sample_col_1_zero_arr = [x == 0 for x in self.orig_df.head(50)[col_name_1]]
            col_1_zero_arr = [x == 0 for x in self.orig_df[col_name_1]]
            num_zero_1 = col_1_zero_arr.count(True)
            if num_zero_1 < (self.num_rows / 100.0):
                continue
            for col_name_2 in self.orig_df.columns:
                if col_name_1 == col_name_2:
                    continue
                sample_col_2_missing_arr = sample_col_missing_dict[col_name_2]
                col_2_missing_arr = col_missing_dict[col_name_2]
                num_missing_2 = count_missing_dict[col_name_2]
                if num_missing_2 < (self.num_rows / 100.0):
                    continue

                # If the difference between the number of missing values is too large, there is not a pattern
                if abs(num_zero_1 - num_missing_2) > (self.freq_contamination_level * 2.0):
                    continue

                # Test first on a sample
                test_series = np.array([x == y for x, y in zip(sample_col_1_zero_arr, sample_col_2_missing_arr)])
                if test_series.tolist().count(False) > 1:
                    continue

                # Test of the full columns
                test_series = np.array([x == y for x, y in zip(col_1_zero_arr, col_2_missing_arr)])
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'The column "{col_name_1}" is 0 in {num_zero_1} rows and not in {self.num_rows - num_zero_1} '
                     f'rows. The column "{col_name_2}" is Null in {num_missing_2} rows and not in '
                     f'{self.num_rows - num_missing_2} rows. '
                     f'Where "{col_name_1}" is 0, "{col_name_2}" is consistently Null and not Null otherwise')
                )

    ##################################################################################################################
    # Data consistency checks for sets of 3 numeric columns
    ##################################################################################################################


    def _generate_similar_to_diff(self):
        """
        Patterns without exceptions: 'similar_to_diff all' is the same as the difference in 'similar_to_diff rand_a' and
            'similar_to_diff rand_b'
        Patterns with exception: 'similar_to_diff most' is the same as the difference in 'similar_to_diff rand_a' and
            'similar_to_diff rand_b', with 1 exception
        """
        self._add_synthetic_column('similar_to_diff rand_a',
                                    [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('similar_to_diff rand_b',
                                    [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('similar_to_diff all',
                                    self.synth_df['similar_to_diff rand_a'] - self.synth_df['similar_to_diff rand_b'])
        self._add_synthetic_column('similar_to_diff most', self.synth_df['similar_to_diff all'])
        self.synth_df.at[999, 'similar_to_diff most'] = 12.0


    def _check_similar_to_diff(self, test_id):
        num_combos = len(self.numeric_cols) * (len(self.numeric_cols) * (len(self.numeric_cols)-1)/2)
        if num_combos > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {int(num_combos):,}  triples of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        col_triples_any_null_bool_dict = self.get_col_triples_any_null_bool_dict()
        nunique_dict = self.get_nunique_dict()

        flagged_tuples = {}
        for col_idx, col_name_3 in enumerate(self.numeric_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns.")
            _, column_pairs = self._get_numeric_column_pairs_unique()
            for _cols_idx, (col_name_1, col_name_2) in enumerate(column_pairs):
                if col_name_1 == col_name_3 or col_name_2 == col_name_3:
                    continue

                if (nunique_dict[col_name_1] == 1) or (nunique_dict[col_name_2] == 1) or (nunique_dict[col_name_3] == 1):
                    continue

                # Check the two columns that have the values that are checked have a reasonable number of at least
                # two unique values, relative to the number of non-null values
                if self.orig_df[col_name_1].value_counts().values[1] < \
                        self.freq_contamination_level * (self.num_valid_rows[col_name_1] / self.num_rows):
                    continue
                if self.orig_df[col_name_2].value_counts().values[1] < \
                        self.freq_contamination_level * (self.num_valid_rows[col_name_2] / self.num_rows):
                    continue

                if col_triples_any_null_bool_dict[tuple(sorted([col_name_1, col_name_2, col_name_3]))]:
                    continue

                med_1 = abs(self.column_medians[col_name_1])
                med_2 = abs(self.column_medians[col_name_2])
                med_3 = abs(self.column_medians[col_name_3])

                # Test that subtracting the 1st and 2nd columns makes sense (they are on the same scale)
                if ((med_2 != 0) and ((med_1 / med_2) < 0.1)) or ((med_2 != 0) and ((med_1 / med_2) > 10.0)):
                    continue

                # Test that the columns may be related checking the medians of the 3 columns
                if med_3 > max(med_1, med_2):
                    continue
                if (med_3 < (abs(med_2 - med_1) * 0.5)) or (med_3 > (abs(med_2 - med_1) * 2.0)):
                    continue

                # Test the relationship on a small sample of the full data
                test_series_a = (self.sample_numeric_vals_filled[col_name_3] / \
                    (self.sample_numeric_vals_filled[col_name_1] - self.sample_numeric_vals_filled[col_name_2]))
                if test_series_a.isna().sum() > (self.num_rows * 0.75):
                    continue
                test_series_a = test_series_a.replace(np.nan, 1.0)
                test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
                # Rows with missing values neither support nor violate the pattern
                test_series = test_series | \
                    self.sample_df[[col_name_1, col_name_2, col_name_3]].isna().any(axis=1).values
                num_not_matching = test_series.tolist().count(False)
                if num_not_matching > 1:
                    continue

                # Check if this set of columns has already been flagged
                current_tuple = tuple(sorted([col_name_1, col_name_2, col_name_3]))
                if current_tuple in flagged_tuples:
                    continue

                # Test on the full data. Rows with missing values are not tested, so skip where few rows have none.
                is_missing_arr = self.orig_df[[col_name_1, col_name_2, col_name_3]].isna().any(axis=1).values
                if (~is_missing_arr).sum() < self.freq_contamination_level:
                    continue
                test_series_a = (self.numeric_vals_filled[col_name_3] / \
                                    (self.numeric_vals_filled[col_name_1] - self.numeric_vals_filled[col_name_2]))
                test_series_a = test_series_a.replace(np.nan, 1.0)
                test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
                test_series = test_series | is_missing_arr
                num_matching = test_series.tolist().count(True)
                if num_matching < (self.num_rows - self.freq_contamination_level):
                    continue

                # Test the match wouldn't be as close simply using col_name_1, using the rows without missing values
                test_series_a = abs(1.0 - (self.numeric_vals_filled[col_name_3] / \
                                           abs(self.numeric_vals_filled[col_name_1] - self.numeric_vals_filled[col_name_2])))
                test_series_b = abs(1.0 - (self.numeric_vals_filled[col_name_3] / abs(self.numeric_vals_filled[col_name_1])))
                if test_series_a[~is_missing_arr].median() < test_series_b[~is_missing_arr].median():
                    self._process_analysis_binary(
                        test_id,
                        [col_name_1, col_name_2, col_name_3],
                        test_series,
                        (f'"{col_name_3}" is consistently similar (within 10%) to the difference of "{col_name_1}" and '
                         f'"{col_name_2}". '))
                    flagged_tuples[current_tuple] = True


    def _generate_diff_exact(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """


    def _check_diff_exact(self, test_id):
        pass


    def _generate_similar_to_product(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('similar to prod 1a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('similar to prod 1b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('similar to prod 2', self.synth_df['similar to prod 1a'] * self.synth_df['similar to prod 1b'])
        self._add_synthetic_column('similar to prod 3', self.synth_df['similar to prod 2'])
        self.synth_df.at[999, 'similar to prod 3'] = 12.0


    def _check_similar_to_product(self, test_id):

        # Track the triples of columns already reported. If we know A = B * C, we do not need to check if B = C* A etc.
        # This saves some execution time, but primarily reduces double reporting.
        reported_dict = {}

        col_triples_any_null_bool_dict = self.get_col_triples_any_null_bool_dict()

        num_triples, column_triples = self._get_numeric_column_triples()
        if num_triples > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {int(num_triples):,} triples of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        # Test if col_name_3 is approximately the product of col_name_1 and col_name_2
        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 100_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(column_triples):,} pairs.")
            columns_tuple = tuple(sorted([col_name_1, col_name_2, col_name_3]))
            if columns_tuple in reported_dict:
                continue

            if col_triples_any_null_bool_dict[tuple(sorted([col_name_1, col_name_2, col_name_3]))]:
                continue

            med_1 = abs(self.column_medians[col_name_1])
            med_2 = abs(self.column_medians[col_name_2])
            med_3 = abs(self.column_medians[col_name_3])

            # Test that the columns may be related checking the medians of the 3 columns
            if (med_3 < med_1) or (med_3 < med_2) or (med_3 < (med_1 * med_2 * 0.5)) or (med_3 > (med_1 * med_2 * 2.0)):
                continue

            # Skip cases where the product is trivially true because some columns are largely zeros. Rows with null
            # values are not tested, and triples with few rows without nulls are skipped above.
            if self.orig_df[col_name_3].tolist().count(0) > (self.num_valid_rows[col_name_3] * 0.75):
                continue

            # Test the relationship on a small sample of the full data
            test_series_a = self.sample_numeric_vals_filled[col_name_3] / \
                            (self.sample_numeric_vals_filled[col_name_1] * self.sample_numeric_vals_filled[col_name_2])
            if test_series_a.isna().sum() > (self.num_rows * 0.75):
                continue
            test_series_a = test_series_a.replace(np.nan, 1.0)
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            test_series = test_series | self.sample_df[[col_name_1, col_name_2, col_name_3]].isna().any(axis=1).values
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching > 1:
                continue

            # Test on the full data
            is_missing_arr = self.orig_df[[col_name_1, col_name_2, col_name_3]].isna().any(axis=1).values
            test_series_a = self.numeric_vals_filled[col_name_3] / \
                            (self.numeric_vals_filled[col_name_1] * self.numeric_vals_filled[col_name_2])

            # First check there is a reasonable number of matches before filling the null values, in the rows without
            # null values.
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False) & ~is_missing_arr
            if test_series.tolist().count(True) < self.freq_contamination_level:
                continue

            # We fill the null values to allow null values to not violate the pattern.
            test_series_a = test_series_a.replace(np.nan, 1.0)
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            test_series = test_series | is_missing_arr
            num_matching = test_series.tolist().count(True)
            if num_matching >= (self.num_rows - self.freq_contamination_level):
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2, col_name_3],
                    test_series,
                    (f'"{col_name_3}" is consistently similar (within 10%) to the product of "{col_name_1}" and '
                     f'"{col_name_2}"'))
                reported_dict[columns_tuple] = True


    def _generate_product_exact(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """


    def _check_product_exact(self, test_id):
        pass


    def _generate_similar_to_ratio(self):
        """
        Patterns without exceptions: 'similar_to_ratio all' is consistently the ratio of rand_a / rand_b
        Patterns with exception: 'similar_to_ratio most' is consistently as well, but with one exception.
        """
        self._add_synthetic_column('similar_to_ratio rand_a',
                                    [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('similar_to_ratio rand_b',
                                    [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('similar_to_ratio all',
                                    self.synth_df['similar_to_ratio rand_a'] / self.synth_df['similar_to_ratio rand_b'])
        self._add_synthetic_column('similar_to_ratio most', self.synth_df['similar_to_ratio all'])
        self.synth_df.at[999, 'similar_to_ratio most'] = 12.0


    def _check_similar_to_ratio(self, test_id):
        # todo: take the abs value of each numeric column here, create a new df, and just use that below.

        # Test if col_name_3 is approximately the ratio of col_name_1 and col_name_2
        # There are 6 combinations. __get_numeric_column_triples() will return each. Each execution of the loop
        # we check if col_name_3 is similar to (col_name_1 / col_name_2)
        num_triples, column_triples = self._get_numeric_column_triples()
        if num_triples > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {int(num_triples):,} triples of numeric columns."
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        col_triples_any_null_bool_dict = self.get_col_triples_any_null_bool_dict()

        # If A == B * C, then C will be the ratio A / B and B will be the ratio A / C. To avoid flagging both, we
        # keep track of the triples flagged.
        flagged_sets = []

        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 100_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(column_triples):,} combinations of columns.")

            if {col_name_1, col_name_2, col_name_3} in flagged_sets:
                continue

            if col_triples_any_null_bool_dict[tuple(sorted([col_name_1, col_name_2, col_name_3]))]:
                continue

            # Test that the columns may be related just checking the medians of the 3 columns
            med_1 = self.column_medians[col_name_1]
            med_2 = self.column_medians[col_name_2]
            med_3 = self.column_medians[col_name_3]
            if med_2 != 0:
                ratio_meds_1_2 = abs(med_1 / med_2)
                if ratio_meds_1_2 != 0:
                    if ((med_3 / ratio_meds_1_2) < 0.5) or ((med_3 / ratio_meds_1_2) > 2.0):
                        continue

            # Test the relationship on a sample of the full data
            test_series_a = self.sample_numeric_vals_filled[col_name_3] / \
                            abs(self.sample_numeric_vals_filled[col_name_1] / self.sample_numeric_vals_filled[col_name_2])
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching > 1:
                continue

            # Test col1 / col2 on the full data
            test_series_a = self.numeric_vals_filled[col_name_3] / \
                            abs(self.numeric_vals_filled[col_name_1] / self.numeric_vals_filled[col_name_2])
            if test_series_a.isna().sum() > (self.num_rows * 0.75):
                continue
            test_series = np.where((test_series_a > 0.9) & (test_series_a < 1.1), True, False)
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_3].isna()
            num_matching = test_series.tolist().count(True)
            if num_matching >= (self.num_rows - self.freq_contamination_level):
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2, col_name_3],  # Put in order such that col_3 == col_1 / col_2
                    test_series,
                    (f'"{col_name_3}" is consistently similar (within 10%) to the ratio of "{col_name_1}" and '
                     f'"{col_name_2}"'))
                flagged_sets.append({col_name_1, col_name_2, col_name_3})
                continue


    def _generate_ratio_exact(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """


    def _check_ratio_exact(self, test_id):
        pass


    def _generate_larger_than_sum(self):
        """
        Patterns without exceptions: 'larger_sum all' is consistently larger than the sum of 'larger_sum rand_a' and
            'larger_sum rand_b'
        Patterns with exception: 'larger_sum most' is consistently larger than the sum of 'larger_sum rand_a' and
            'larger_sum rand_b', with exceptions.
        """
        self._add_synthetic_column('larger_sum rand_a',
                                    [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('larger_sum rand_b',
                                    [random.randint(1000, 2000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('larger_sum all',
                                    self.synth_df['larger_sum rand_b'] + self.synth_df['larger_sum rand_a'] + random.randint(1, 10))
        self._add_synthetic_column('larger_sum most',
                                    self.synth_df['larger_sum all'].astype(float))
        self.synth_df.at[999, 'larger_sum most'] = self.synth_df.at[999, 'larger_sum most'] * 0.2


    def _check_larger_than_sum(self, test_id):
        """
        Where the 3 columns are A, B, and C, we try: A > (B+C)
        This is done for all 3 columns in the place of A, so there are 3 tests performed.
        """

        def test_larger(col_a, col_b, col_c):
            # Check col_a and col_b are on the same scale and may reasonably be added.
            if not self.check_columns_same_scale_2(col_a, col_b, order=5):
                return False

            # Check both features have some correlation with the summed feature
            if self.spearman_corr[col_a][col_c] < 0.40:
                return False
            if self.spearman_corr[col_b][col_c] < 0.40:
                return False

            # Test the relationship on a small sample of the full data. Rows with missing values do not violate it.
            test_series = self.sample_df[col_c].astype(float) > (self.sample_df[col_a].astype(float) +
                                                                 self.sample_df[col_b].astype(float))
            test_series = test_series | self.sample_df[[col_a, col_b, col_c]].isna().any(axis=1)
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching > 1:
                return False

            # Test there is a correlation on a sample, using the rows with values in all three columns
            corr = scipy.stats.spearmanr(self.sample_df[col_a].astype(float) + self.sample_df[col_b].astype(float),
                                            self.sample_df[col_c].astype(float), nan_policy='omit')
            if corr.correlation < 0.9:
                return False

            test_series = self.numeric_vals_filled[col_c].astype(float) > (self.numeric_vals_filled[col_a] +
                                                                           self.numeric_vals_filled[col_b])
            test_series = test_series | self.orig_df[col_a].isna() | self.orig_df[col_b].isna() | self.orig_df[col_c].isna()
            if test_series.tolist().count(False) < self.freq_contamination_level:
                self._process_analysis_binary(
                    test_id,
                    [col_a, col_b, col_c],
                    test_series,
                    (f'The values in "{col_c}" are consistently larger than the sum of "{col_a}", '
                     f' and "{col_b}"'))
                return True
            return False

        # Track which columns are mostly positive. We execute this test only on those columns.
        column_pos_dict = {}
        for col_name in self.numeric_cols:
            vals_arr = convert_to_numeric(self.orig_df[col_name], 1)
            column_pos_dict[col_name] = \
                (((vals_arr >= 0) | self.orig_df[col_name].isna()).tolist().count(False) < self.freq_contamination_level)

        num_triples, column_triples = self._get_numeric_column_triples_unique()
        if num_triples > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {int(num_triples):,} triples of numeric columns."
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        col_triples_any_null_bool_dict = self.get_col_triples_any_null_bool_dict()

        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {num_triples:,} combinations of columns.")

            if not self.check_columns_same_scale_3(col_name_1, col_name_2, col_name_3, order=5):
                continue

            # This test applies only to columns which are largely positive.
            if not column_pos_dict[col_name_1]:
                continue
            if not column_pos_dict[col_name_2]:
                continue
            if not column_pos_dict[col_name_3]:
                continue

            if col_triples_any_null_bool_dict[tuple(sorted([col_name_1, col_name_2, col_name_3]))]:
                continue

            med_1 = self.column_medians[col_name_1]
            med_2 = self.column_medians[col_name_2]
            med_3 = self.column_medians[col_name_3]

            # Test if col_name_1 is larger than col_name_2 + col_name_3
            if (med_1 > med_2) and (med_1 > med_3) and (med_1 > (med_2 + med_3 * 0.5)):
                if test_larger(col_name_2, col_name_3, col_name_1):
                    continue

            # Test if col_name_2 is larger than col_name_1 + col_name_3
            if (med_2 > med_1) and (med_2 > med_3) and (med_2 > (med_1 + med_3 * 0.5)):
                if test_larger(col_name_1, col_name_3, col_name_2):
                    continue

            # Test if col_name_3 is larger than col_name_1 + col_name_2
            if (med_3 > med_1) and (med_3 > med_2) and (med_3 > (med_1 + med_2 * 0.5)):
                if test_larger(col_name_1, col_name_2, col_name_3):
                    continue


    def _generate_larger_than_abs_diff(self):
        """
        Patterns without exceptions: 'larger_diff all' is consistently larger than the abs differnce between
            'larger_diff rand_a' and 'larger_diff rand_b'
        Patterns with exception: 'larger_diff most' is consistently larger than the abs differnce between
            'larger_diff rand_a' and 'larger_diff rand_b'
        """
        self._add_synthetic_column('larger_diff rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('larger_diff rand_b', [random.randint(1000, 2000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('larger_diff all',
                                    self.synth_df['larger_diff rand_b'] - self.synth_df['larger_diff rand_a'] + random.randint(1, 10))
        self._add_synthetic_column('larger_diff most', self.synth_df['larger_diff all'])
        self.synth_df.at[999, 'larger_diff most'] = self.synth_df.at[999, 'larger_diff most'] * 0.2


    def _check_larger_than_abs_diff(self, test_id):
        """
        For example, with the blood-transfusion dataset on OpenML, there are columns for time since first donation and
        time since the last donation. The difference in these is the time between their first & last donations. There
        is also a column for number of donations. This cannot be larger than the difference between their first and
        last donation times. This test is only done where all 3 columns are on the same scale (none is more than
        double the other with respect to their medians).

        Where the 3 columns are A, B, and C, we try: C > abs(A-B)
        This is done for all 3 columns in the place of A, so there are 3 tests performed.
        """

        def test_larger(col_a, col_b, col_c):
            # Skip where any 2 of the columns are usually the same
            # return True to skip other permutations of these 3 columns.
            num_same = (self.orig_df[col_a] == self.orig_df[col_b]).tolist().count(True)
            if num_same > (self.num_rows * 0.9):
                return True
            num_same = (self.orig_df[col_a] == self.orig_df[col_c]).tolist().count(True)
            if num_same > (self.num_rows * 0.9):
                return True
            num_same = (self.orig_df[col_a] == self.orig_df[col_c]).tolist().count(True)
            if num_same > (self.num_rows * 0.9):
                return True

            # Test the relationship col_c > abs(col_a - col_b) on a small sample of the full data
            test_series = self.sample_df[col_c] > abs(self.sample_df[col_a] - self.sample_df[col_b])
            num_not_matching = test_series.tolist().count(False)
            if num_not_matching >= 1:
                return False

            # Test of the full dataset
            test_series = self.orig_df[col_c] > abs(self.orig_df[col_a] - self.orig_df[col_b])
            test_series = test_series | self.orig_df[col_a].isna() | self.orig_df[col_b].isna() | self.orig_df[col_c].isna()

            # Check the values are not strange in themselves, just the relationship between them.
            # And, only flag values significantly greater than the abs diff
            if test_series.tolist().count(False) < self.freq_contamination_level:
                flagged_rows = np.where(~test_series)[0]
                for row_num in flagged_rows:
                    # This may help, but leads to plots with the orange values mixed with the blue
                    # if (percentiles_dict[col_a][row_num] > 0.98) | (percentiles_dict[col_a][row_num] < 0.02) | \
                    #     (percentiles_dict[col_b][row_num] > 0.98) | (percentiles_dict[col_b][row_num] < 0.02) | \
                    #     (percentiles_dict[col_c][row_num] > 0.98) | (percentiles_dict[col_c][row_num] < 0.02):
                    #     test_series[row_num] = True
                    if (abs(self.orig_df.iloc[row_num][col_a ] - self.orig_df.iloc[row_num][col_b]) - self.orig_df.iloc[row_num][col_c]) < (self.column_medians[col_c] / 5.0):
                        test_series[row_num] = True
            else:
                return False

            if test_series.tolist().count(False) < self.freq_contamination_level:
                self._process_analysis_binary(
                    test_id,
                    [col_a, col_b, col_c],  # Order such that the last depends on the others
                    test_series,
                    (f'The values in "{col_c}" are consistently larger than the difference between "{col_a}", '
                     f' and "{col_b}"'))
                return True
            return False

        # todo: we should also check the spearman correlations match. maybe. this catches something different.
        #   maybe 2 different tests.

        # Test if col_name_3 is consistently larger than the absolute difference in col_name_1 and col_name_2
        num_triples, column_triples = self._get_numeric_column_triples_unique()
        if num_triples > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. \nThere are {int(num_triples):,} triples of numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        col_triples_any_null_bool_dict = self.get_col_triples_any_null_bool_dict()
        self.get_percentiles_dict()

        for cols_idx, (col_name_1, col_name_2, col_name_3) in enumerate(column_triples):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(column_triples):,} combinations of columns.")

            if not self.check_columns_same_scale_3(col_name_1, col_name_2, col_name_3, order=5):
                continue

            if col_triples_any_null_bool_dict[tuple(sorted([col_name_1, col_name_2, col_name_3]))]:
                continue

            med_1 = abs(self.column_medians[col_name_1])
            med_2 = abs(self.column_medians[col_name_2])
            med_3 = abs(self.column_medians[col_name_3])

            # Test if col_name_1 is larger than abs(col_name_2 - col_name_3)
            if med_1 > (abs(med_2 - med_3) * 0.5) and test_larger(col_name_2, col_name_3, col_name_1):
                continue

            # Test if col_name_2 is larger than abs(col_name_1 - col_name_3)
            if med_2 > (abs(med_1 - med_3) * 0.5) and test_larger(col_name_1, col_name_3, col_name_2):
                continue

            # Test if col_name_3 is larger than abs(col_name_2 - col_name_3)
            if med_3 > (abs(med_1 - med_2) * 0.5) and test_larger(col_name_1, col_name_2, col_name_3):
                continue

    ##################################################################################################################
    # Data consistency checks for numeric column in relation to all other numeric columns.
    ##################################################################################################################


    def _generate_sum_of_columns(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        self._add_synthetic_column('sum of cols rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sum of cols rand_b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sum of cols rand_c', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('sum of cols rand_d', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self._add_synthetic_column('sum of cols all', self.synth_df[[
            'sum of cols rand_a',
            'sum of cols rand_b',
            'sum of cols rand_c']].sum(axis=1))
        self._add_synthetic_column('sum of cols most', self.synth_df['sum of cols all'].copy())
        self.synth_df.at[999, 'sum of cols most'] = self.synth_df.at[999, 'sum of cols most'] * 5.0

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c plus a constant are always, and mostly
        # true
        self._add_synthetic_column('sum of cols plus all', self.synth_df[[
            'sum of cols rand_a',
            'sum of cols rand_b',
            'sum of cols rand_c']].sum(axis=1))
        self.synth_df['sum of cols plus all'] += 67.3
        self._add_synthetic_column('sum of cols plus most', self.synth_df['sum of cols plus all'].copy())
        self.synth_df.at[999, 'sum of cols plus most'] = self.synth_df.at[999, 'sum of cols plus most'] * 5.0

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c times a constant are always, and mostly
        # true
        self._add_synthetic_column('sum of cols times all', self.synth_df[[
            'sum of cols rand_a',
            'sum of cols rand_b',
            'sum of cols rand_c']].sum(axis=1))
        self.synth_df['sum of cols times all'] *= 1.6
        self._add_synthetic_column('sum of cols times most', self.synth_df['sum of cols times all'].copy())
        self.synth_df.at[999, 'sum of cols times most'] = self.synth_df.at[999, 'sum of cols times most'] * 5.0


    def _check_sum_of_columns(self, test_id):
        """
        Loop through each numeric column. For each, find a set of plausible other columns, and test each subset of
        these. Plausible columns have smaller, but not drastically smaller, medians to the current column.
        This test also checks if there is a constant difference or constant ratio between the column sums and the
        values in the column.
        """

        # Track which columns are mostly positive. We execute this test only on those columns.
        column_pos_arr = []
        for col_name in self.numeric_cols:
            if (self.numeric_vals_filled[col_name] >= 0).tolist().count(False) < self.freq_contamination_level:
                column_pos_arr.append(col_name)

        # Identify the set of similar columns for each positive numeric column
        similar_cols_dict, _, calc_size = self.get_similar_cols(
            column_pos_arr, include_self=False, lower_divisor=20.0, upper_multiplier=1.0)

        # Check if there are too many combinations to execute this test
        limit_subset_sizes, max_subset_size, can_process = \
            self.get_limit_subset_sizes(column_pos_arr, similar_cols_dict, calc_size)
        if not can_process:
            return

        for col_idx, col_name in enumerate(column_pos_arr):
            if (self.verbose == 2 and col_idx > 0 and col_idx % 10 == 0) or (self.verbose >= 3):
                print(f"  Examining column: {col_idx} of {len(column_pos_arr)} positive numeric columns)")

            similar_cols = similar_cols_dict[col_name]
            if len(similar_cols) == 0:
                continue

            found_any = False

            # For any subsets whose sum is too small to match col_name, there is no use trying any smaller subsets.
            know_failed_subsets = {}

            starting_size = len(similar_cols)
            if limit_subset_sizes:
                starting_size = min(starting_size, max_subset_size)
            for subset_size in range(starting_size, 1, -1):
                if found_any:
                    break

                subsets = list(combinations(similar_cols, subset_size))
                if self.verbose >= 3 and len(similar_cols) > 15:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
                for subset_idx, subset in enumerate(subsets):
                    if self.verbose >= 3 and len(similar_cols) > 15 and subset_idx > 0 and subset_idx % 10_000 == 0:
                        print(f"    Examining subset {subset_idx:,}")

                    # Check if this subset is a subset of any subsets previously tried which were too small.
                    subset_matches = True
                    for kfs in know_failed_subsets:
                        if set(kfs).issuperset(set(subset)):
                            subset_matches = False
                            break
                    if not subset_matches:
                        continue

                    # Check if this set of columns summing to col_name is plausible, checking if the sum of medians is
                    # significantly smaller or larger. actually, no -- we check ading/ multiply by constant below
                    sum_of_medians = 0
                    for c in subset:
                        sum_of_medians += self.column_medians[c]
                    if sum_of_medians > (self.column_medians[col_name] * 1.1):
                        continue

                    subset = list(subset)

                    # Check all 3 sub-tests on a sample first, skipping rows with missing values
                    sample_df = self.sample_df[[*subset, col_name]].dropna().astype(float)
                    col_sums = sample_df[subset].sum(axis=1)
                    sample_diffs_series = sample_df[col_name] - col_sums
                    sample_col_values = sample_diffs_series == 0
                    subtest_1_okay = sample_col_values.tolist().count(False) <= 1

                    # Check on a sample for 2nd sub-test
                    median_diff = sample_diffs_series.median()
                    sample_col_values = [math.isclose(x, median_diff) for x in sample_diffs_series]
                    subtest_2_okay = sample_col_values.count(False) <= 1

                    # Check on a sample for 3rd sub-test
                    ratios_series = sample_df[col_name] / col_sums
                    median_ratio = ratios_series.median()
                    sample_col_values = [math.isclose(x, median_ratio) for x in ratios_series]
                    subtest_3_okay = sample_col_values.count(False) <= 1

                    # Check if col_name is the sum of subset
                    if subtest_1_okay or subtest_2_okay or subtest_3_okay:
                        # Rows with missing values are not tested, so skip where few rows have none
                        if self.orig_df[[*subset, col_name]].notna().all(axis=1).sum() < self.freq_contamination_level:
                            continue
                        df = self.orig_df[subset].astype(float)
                        col_sums = df.sum(axis=1, skipna=False)  # The sums are NaN where any value is missing
                        diffs_series = self.orig_df[col_name].astype(float) - col_sums
                        col_values = diffs_series == 0
                        col_values = self.check_results_for_null(col_values, col_name, subset)
                        if col_values.tolist().count(False) < self.freq_contamination_level:
                            self._process_analysis_binary(
                                test_id,
                                subset + [col_name],
                                col_values,
                                f'The column "{col_name}" is consistently equal to the sum of the values in the columns {subset}',
                                "",
                                display_info={"Sum": col_sums, "operation": ''}
                            )
                            found_any = True
                            break
                        too_small_arr = self.orig_df[col_name].astype(float) > col_sums
                        if too_small_arr.tolist().count(True) > self.freq_contamination_level:
                            know_failed_subsets[tuple(subset)] = True
                    else:
                        too_small_arr = sample_df[col_name] > col_sums
                        if too_small_arr.tolist().count(True) > 1:
                            know_failed_subsets[tuple(subset)] = True

                    # Check if there is a constant difference between the column sums and the values in col_name. If so,
                    # column col_name is the sum of the columns plus a constant. We determine if the median value in
                    # the differences is this constant.
                    if subtest_2_okay:
                        median_diff = diffs_series.median()
                        col_values = np.array([math.isclose(x, median_diff) for x in diffs_series])
                        col_values = self.check_results_for_null(col_values, col_name, subset)
                        if col_values.tolist().count(False) < self.freq_contamination_level:
                            self._process_analysis_binary(
                                test_id,
                                subset + [col_name],
                                np.array(col_values),
                                (f'The column "{col_name}" is consistently equal to the sum of the values in the '
                                 f"columns {subset} plus {median_diff}"),
                                "",
                                display_info={"Sum": col_sums, "operation": "plus", "amount": median_diff}
                            )
                            found_any = True
                            break

                    # Check if there is a constant ratio between the column sums and the values in col_name. If so,
                    # column col_name is the sum of the columns times a constant. We determine if the median value in
                    # the differences is this constant.
                    if subtest_3_okay:
                        ratios_series = self.orig_df[col_name].astype(float) / col_sums
                        median_ratio = ratios_series.median()
                        col_values = np.array([math.isclose(x, median_ratio) for x in ratios_series])
                        col_values = self.check_results_for_null(col_values, col_name, subset)
                        if col_values.tolist().count(False) < self.freq_contamination_level:
                            self._process_analysis_binary(
                                test_id,
                                subset + [col_name],
                                np.array(col_values),
                                (f"The column {col_name} is consistently equal to the sum of the values in the columns "
                                 f" {subset} times {median_ratio}"),
                                "",
                                display_info={"Sum": col_sums, "operation": "times", "amount": median_ratio}
                            )
                            found_any = True
                            break


    def _generate_min_of_columns(self):
        """
        Unlike sum_of_columns, there is not the concept of adding or multiplying a constant to the minimum of another
        set of columns.

        Patterns without exceptions: 'min_of_cols all' is consistently the min of "min_of_cols rand_a",
            "min_of_cols rand_b" AND "min_of_cols rand_c"
        Patterns with exception: 'min_of_cols most' is consistently the min of "min_of_cols rand_a",
            "min_of_cols rand_b" AND "min_of_cols rand_c", with 1 exception
        """
        self._add_synthetic_column('min_of_cols rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('min_of_cols rand_b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('min_of_cols rand_c', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('min_of_cols rand_d', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self._add_synthetic_column('min_of_cols all', self.synth_df[[
            'min_of_cols rand_a',
            'min_of_cols rand_b',
            'min_of_cols rand_c']].min(axis=1))
        self._add_synthetic_column('min_of_cols most', self.synth_df['min_of_cols all'].copy())
        self.synth_df.at[999, 'min_of_cols most'] = self.synth_df.at[999, 'min_of_cols most'] * 5.0


    def _check_min_of_columns(self, test_id):
        """
        This checks for subsets of maximum 10 columns.
        """
        # Try all subsets where it's median is less than this column's, but more than 1/20 of it.
        # For each target column, try all subsets whose minimum values match this column.

        two_rows_np = self.sample_df[self.numeric_cols].sample(n=2, random_state=0).values

        # Identify the set of similar columns for each positive numeric column
        similar_cols_dict, similar_cols_idxs_dict, calc_size = self.get_similar_cols(
            self.numeric_cols, include_self=False, lower_divisor=1.0, upper_multiplier=10.0, check_larger_false=True)

        # Check if there are too many combinations to execute this test
        limit_subset_sizes, max_subset_size, can_process = \
            self.get_limit_subset_sizes(self.numeric_cols, similar_cols_dict, calc_size)
        if not can_process:
            return

        for col_idx, col_name in enumerate(self.numeric_cols):
            printed_column_status = False
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")
                printed_column_status = True

            similar_cols = similar_cols_dict[col_name]
            similar_cols_idxs = similar_cols_idxs_dict[col_name]

            found_any = False
            starting_size = len(similar_cols)
            if limit_subset_sizes:
                starting_size = min(starting_size, max_subset_size)
            for subset_size in range(starting_size, 2, -1):
                if found_any:
                    break

                subsets = list(combinations(similar_cols_idxs, subset_size))
                if self.verbose >= 2 and len(similar_cols) > 15:
                    if not printed_column_status:
                        print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")
                        printed_column_status = True
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset in subsets:
                    subset = list(subset)

                    # Test on just 2 rows
                    test_np = two_rows_np[:, subset]
                    col_mins = test_np.min(axis=1)
                    test_series = np.where(two_rows_np[:, col_idx] == col_mins, True, False)
                    if test_series.tolist().count(False) > 0:
                        continue

                    # Test on a subset of the rows
                    subset_names = np.array(self.numeric_cols)[subset].tolist()
                    df = self.sample_df[subset_names]
                    col_mins = df.min(axis=1)
                    test_series = np.where(self.sample_df[col_name].values == col_mins, True, False)
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on the full columns
                    df = self.orig_df[subset_names]
                    col_mins = df.min(axis=1)
                    diffs_series = self.orig_df[col_name] - col_mins
                    test_series = diffs_series == 0
                    test_series = self.check_results_for_null(test_series, col_name, subset_names)
                    if test_series.tolist().count(False) < self.freq_contamination_level:
                        # Check the target column is not identical to any of the source columns
                        subset_okay = True
                        for col in subset_names:
                            equal_series = [x == y or is_missing(x) or is_missing(y)
                                            for x, y in zip(self.orig_df[col_name], self.orig_df[col])]
                            if equal_series.count(False) < self.freq_contamination_level:
                                subset_okay = False
                                break
                        if not subset_okay:
                            continue

                        # Check if this is a pattern in a trivial way. Check each column in the subset is the min at
                        # least once.
                        subset_okay = True
                        for col in subset_names:
                            if [(x == y) for x, y, z in zip(self.orig_df[col], col_mins, test_series) if z].count(True) == 0:
                                subset_okay = False
                                break
                        if not subset_okay:
                            continue

                        self._process_analysis_binary(
                            test_id,
                            subset_names + [col_name],
                            test_series,
                            (f'The column "{col_name}" is consistently equal to the minimum of the values in the '
                             f'columns {subset_names}'),
                            "",
                            display_info={"Min": col_mins, "operation": ''}
                        )
                        found_any = True
                        break


    def _generate_max_of_columns(self):
        self._add_synthetic_column('max_of_cols rand_a', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('max_of_cols rand_b', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('max_of_cols rand_c', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('max_of_cols rand_d', [random.randint(1, 1000) for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self._add_synthetic_column('max_of_cols all', self.synth_df[[
            'max_of_cols rand_a',
            'max_of_cols rand_b',
            'max_of_cols rand_c']].max(axis=1))
        self._add_synthetic_column('max_of_cols most', self.synth_df['max_of_cols all'].copy())
        self.synth_df.at[999, 'max_of_cols most'] = self.synth_df.at[999, 'max_of_cols most'] * 5.0


    def _check_max_of_columns(self, test_id):
        """
        This checks for subsets of maximum 10 columns.
        """

        # Try all subsets where it's median is less than this column's, but more than 1/10 of it.
        # For each target column, try all subsets whose minimum values match this column.

        two_rows_np = self.sample_df[self.numeric_cols].sample(n=2, random_state=0).values

        # Identify the set of similar columns for each positive numeric column
        similar_cols_dict, similar_cols_idxs_dict, calc_size = self.get_similar_cols(
            self.numeric_cols, include_self=False, lower_divisor=10.0, upper_multiplier=1.0, check_larger_true=True)

        # Check if there are too many combinations to execute this test
        limit_subset_sizes, max_subset_size, can_process = \
            self.get_limit_subset_sizes(self.numeric_cols, similar_cols_dict, calc_size)
        if not can_process:
            return

        for col_idx, col_name in enumerate(self.numeric_cols):
            printed_column_status = False
            if self.verbose >= 2 and col_idx > 0 and col_idx % 25 == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")
                printed_column_status = True

            similar_cols = similar_cols_dict[col_name]
            similar_cols_idxs = similar_cols_idxs_dict[col_name]

            found_any = False
            starting_size = len(similar_cols)
            if limit_subset_sizes:
                starting_size = min(starting_size, max_subset_size)
            for subset_size in range(starting_size, 2, -1):
                if found_any:
                    break

                subsets = list(combinations(similar_cols_idxs, subset_size))
                if self.verbose >= 2 and len(similar_cols) > 15:
                    if not printed_column_status:
                        print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")
                        printed_column_status = True
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset in subsets:
                    subset = list(subset)

                    # Test on just 2 rows
                    test_np = two_rows_np[:, subset]
                    col_maxs = test_np.max(axis=1)
                    test_series = np.where(two_rows_np[:, col_idx] == col_maxs, True, False)
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on a subset of the rows
                    subset_names = np.array(self.numeric_cols)[subset].tolist()
                    df = self.sample_df[subset_names]
                    col_maxs = df.max(axis=1)
                    test_series = np.where(self.sample_df[col_name].values == col_maxs, True, False)
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on the full columns
                    df = self.orig_df[subset_names]
                    col_maxs = df.max(axis=1)
                    diffs_series = self.orig_df[col_name] - col_maxs
                    test_series = diffs_series == 0
                    test_series = self.check_results_for_null(test_series, col_name, subset_names)
                    if test_series.tolist().count(False) < self.freq_contamination_level:
                        # Check the target column is not identical to any of the source columns
                        subset_okay = True
                        for col in subset_names:
                            equal_series = [x == y or is_missing(x) or is_missing(y) for x, y in zip(self.orig_df[col_name], self.orig_df[col])]
                            if equal_series.count(False) < self.freq_contamination_level:
                                subset_okay = False
                                break
                        if not subset_okay:
                            continue

                        # Check if this is a pattern in a trivial way. Check each column in the subset is the min at
                        # least once.
                        subset_okay = True
                        for col in subset_names:
                            if [(x == y) for x, y, z in zip(self.orig_df[col], col_maxs, test_series) if z].count(True) == 0:
                                subset_okay = False
                                break
                        if not subset_okay:
                            continue

                        self._process_analysis_binary(
                            test_id,
                            subset_names + [col_name],
                            test_series,
                            (f'The column "{col_name}" is consistently equal to the maximum of the values in the '
                             f'columns {subset_names}'),
                            "",
                            display_info={"Max": col_maxs, "operation": ''}
                        )
                        found_any = True
                        break


    def _generate_mean_of_columns(self):
        """
        Unlike sum_of_columns, there is not the concept of adding or multiplying a constant to the minimum of another
        set of columns.

        Patterns without exceptions: ''mean_of_cols all' is consistently the mean of rand_a, rand_b, and rand_c
        Patterns with exception: 'mean_of_cols most' is consistently the mean of rand_a, rand_b, and rand_c, with
            one exception.
        """
        self._add_synthetic_column('mean_of_cols rand_a', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('mean_of_cols rand_b', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('mean_of_cols rand_c', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('mean_of_cols rand_d', [random.random() for _ in range(self.num_synth_rows)])

        # Add columns where the pattern based on cols rand_a, rand_b, and rand_c are always and mostly true
        self._add_synthetic_column('mean_of_cols all', self.synth_df[[
            'mean_of_cols rand_a',
            'mean_of_cols rand_b',
            'mean_of_cols rand_c']].mean(axis=1))
        self._add_synthetic_column('mean_of_cols most', self.synth_df['mean_of_cols all'].copy())
        self.synth_df.at[999, 'mean_of_cols most'] = self.synth_df.at[999, 'mean_of_cols most'] * 5.0


    def _check_mean_of_columns(self, test_id):
        """
        This test loops through each numeric column and determines if that column is the mean of some subset of the
        other numeric columns. Doing this, it is able to cache information from the first  columns checked, and
        consequently, subsequent columns tend to execute much faster.
        """

        # Track which columns are mostly positive. We execute this test only on those columns.
        column_pos_arr = []
        for col_name in self.numeric_cols:
            if (self.numeric_vals_filled[col_name] >= 0).tolist().count(False) < self.freq_contamination_level:
                column_pos_arr.append(col_name)

        # We loop through the positive numeric columns and for each find the set of columns with similar values.
        # For each subset of these, we check if any is the mean of the others. We then do not need to check any of these
        # columns again, but continue through the positive numeric columns for numeric columns in other ranges.
        skip_col_sets = []

        # Identify the set of similar columns for each positive numeric column
        similar_cols_dict, _, calc_size = self.get_similar_cols(
            column_pos_arr, include_self=True, lower_divisor=5.0, upper_multiplier=5.0)

        # Check if there are too many combinations to execute this test
        limit_subset_sizes, max_subset_size, can_process = \
            self.get_limit_subset_sizes(column_pos_arr, similar_cols_dict, calc_size)
        if not can_process:
            return

        for col_idx, col_name in enumerate(column_pos_arr):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f'  Examining column: {col_idx} of {len(column_pos_arr)} positive numeric columns (and all '
                       f'numeric columns of similar ranges)')

            similar_cols = similar_cols_dict[col_name]

            # Check if we've evaluated the same set, or a superset of this
            similar_cols_set = set(similar_cols)
            subset_matches = False
            for sct in skip_col_sets:
                if sct.issuperset(similar_cols_set):
                    subset_matches = True
                    break
            if subset_matches:
                continue
            skip_col_sets.append(similar_cols_set)

            found_any = False
            starting_size = len(similar_cols)
            if limit_subset_sizes:
                starting_size = min(starting_size, max_subset_size)
            for subset_size in range(starting_size, 2, -1):
                if found_any:
                    break

                subsets = list(combinations(similar_cols, subset_size))
                if self.verbose >= 3 and len(similar_cols) > 15:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset_idx, subset in enumerate(subsets):
                    if self.verbose >= 3 and len(similar_cols) > 15 and subset_idx > 0 and subset_idx % 10_000 == 0:
                        print(f"    Examining subset {subset_idx:,}")

                    # Check if this set of columns summing to col_name is plausible, checking if the mean of medians is
                    # significantly smaller or larger.
                    medians_arr = []
                    for c in subset:
                        medians_arr.append(self.column_medians[c])
                    mean_of_medians = statistics.mean(medians_arr)
                    if mean_of_medians < (self.column_medians[col_name] * 0.9):
                        continue
                    if mean_of_medians > (self.column_medians[col_name] * 1.1):
                        continue

                    subset = list(subset)

                    # Test on a sample of the rows in the columns, skipping rows with missing values. Using numpy works
                    # faster in this case.
                    cols_idxs = [self.orig_df.columns.tolist().index(x) for x in subset]
                    sample_non_null_arr = self.sample_df[subset].notna().all(axis=1).values
                    sample_np = self.sample_df.values[sample_non_null_arr][:, cols_idxs]
                    col_mean = sample_np.mean(axis=1)

                    # We loop through all the columns in the subset, to avoid duplicate work later
                    matching_column = []
                    for c in subset:
                        if np.allclose(self.sample_df[c][sample_non_null_arr], col_mean.astype(float)):
                            matching_column = c
                            break
                    if not matching_column:
                        continue

                    # Rows with missing values are not tested, so skip where few rows have none
                    if self.orig_df[[*subset, col_name]].notna().all(axis=1).sum() < self.freq_contamination_level:
                        continue

                    # Test on the full columns
                    df = self.orig_df[subset].copy()
                    col_mean = df.mean(axis=1)
                    diffs_series = self.orig_df[matching_column] - col_mean
                    test_series = np.isclose(diffs_series, 0)
                    test_series = self.check_results_for_null(test_series, col_name, subset)

                    # Get the set of columns to report, keeping the matching column the right-most
                    all_cols = list(set(subset + [col_name] + [matching_column]))
                    rest_of_cols = all_cols.copy()
                    rest_of_cols.remove(matching_column)
                    if test_series.tolist().count(False) < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            sorted(rest_of_cols) + [matching_column],
                            test_series,
                            (f'The column "{matching_column}" is consistently equal to the mean of the values in the '
                             f'columns {rest_of_cols}'),
                            "",
                            display_info={"Mean": col_mean}
                        )
                        found_any = True
                        break


    def generate_mathed_set_pos_neg(self):
        """
        Patterns without exceptions: None: 'matched_pos_neg all' consistently has the same sign as 'matched_pos_neg rand_a'.
            However, it is not reported as a pair of columns as it is part of a set of 3 features with near-perfect
            matching.
        Patterns with exception: 'matched_pos_neg most' consistently has the same sign as 'matched_pos_neg rand_a',
            and 'matched_pos_neg all', with the exception of row 999.
        """
        self._add_synthetic_column('matched_pos_neg rand_a', [random.randint(-1000, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched_pos_neg rand_b', [random.randint(-1000, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched_pos_neg all', self.synth_df['matched_pos_neg rand_a'] * random.randint(1, 100))
        self._add_synthetic_column('matched_pos_neg most', self.synth_df['matched_pos_neg rand_a'] * random.randint(1, 100))
        self.synth_df.at[999, 'matched_pos_neg most'] = self.synth_df.at[999, 'matched_pos_neg most'] * -1.0


    def check_mathed_set_pos_neg(self, test_id):
        """
        We identify sets of numeric columns, not necessarily on the same scale, that are all both frequently
        positive and frequently negative. We then find the subsets of these columns that are consistently positive and
        negative together.
        """

        # Check all columns examined have at least 10% positive and 10% negative values.
        cols = []
        sample_pos_dict = {}
        sample_neg_dict = {}
        pos_dict = {}
        neg_dict = {}
        for _col_idx, col_name in enumerate(self.numeric_cols):
            vals_arr = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])
            num_pos = len([x for x in vals_arr if x > 0])
            num_neg = len([x for x in vals_arr if x < 0])
            if num_pos > (self.num_rows * 0.1) and num_neg > (self.num_rows * 0.1):
                cols.append(col_name)

                # The sample data tends not to include Null values
                pos_arr = self.sample_df[col_name] > 0
                neg_arr = self.sample_df[col_name] < 0
                sample_pos_dict[col_name] = pos_arr
                sample_neg_dict[col_name] = neg_arr

                pos_arr = ((self.orig_df[col_name] > 0) | self.orig_df[col_name].isna())
                neg_arr = ((self.orig_df[col_name] < 0) | self.orig_df[col_name].isna())
                pos_dict[col_name] = pos_arr
                neg_dict[col_name] = neg_arr

        # Loop through the subsets, from biggest to smallest. Consider only subsets of at least 2 columns.
        # We may flag multiple subsets of the same size, but none that are smaller. This will allow some subsets to
        # be skipped, but is a useful heuristic to reduce execution time.
        num_cols = len(cols)
        found_any = False
        know_failed_subsets = {}  # dictionary of dictionaries, with and element for each subset size.
        printed_subset_size_msg = False
        for subset_size in range(num_cols, 1, -1):
            know_failed_subsets[subset_size] = {}
            if found_any:
                break

            calc_size = math.comb(len(cols), subset_size)
            if calc_size > self.max_combinations:
                if self.verbose >= 2 and not printed_subset_size_msg :
                    print(f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. "
                           f"max_combinations is currently set to {self.max_combinations:,}.")
                    printed_subset_size_msg = True
                continue

            subsets = list(combinations(cols, subset_size))
            if self.verbose >= 3:
                print(f"  Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset_idx, subset in enumerate(subsets):
                if self.verbose >= 3 and subset_idx > 0 and subset_idx % 10_000 == 0:
                    print(f"    Examining subset {subset_idx:,} of {len(subsets):,} subsets.")

                # Check if this subset has already been determined to be impossible from a portion of a previously
                # checked subset. That is, if we've tried a set of columns that is a subset of the current subset, and
                # the pattern did not hold there, it could not hold with a larger set. We catch these smaller subsets
                # below. Though the size of the subsets decreases in the main loop, when checking sets of columns,
                # we stop early with any mismatch, and save this.
                subset_matches = True
                for prev_subset_size in range(num_cols, subset_size, -1):
                    if prev_subset_size not in know_failed_subsets:
                        continue
                    for kfs in know_failed_subsets[prev_subset_size]:
                        if set(subset).issuperset(set(kfs)):  # True if subset is a superset of kfs
                            subset_matches = False
                            break
                if not subset_matches:
                    continue

                # Test on a sample of rows
                pos_matching_arr = np.full(len(self.sample_df), True)
                neg_matching_arr = np.full(len(self.sample_df), True)
                for c_idx, c in enumerate(subset[1:]):  # noqa: B007 - used after the loop
                    pos_matching_arr = pos_matching_arr & (sample_pos_dict[subset[0]] == sample_pos_dict[c])
                    if pos_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                    neg_matching_arr = neg_matching_arr & (sample_neg_dict[subset[0]] == sample_neg_dict[c])
                    if neg_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                if not subset_matches:
                    know_failed_subsets[subset_size][tuple(subset[:c_idx+2])] = True
                    continue

                # Test on the full columns
                pos_matching_arr = np.full(self.num_rows, True)
                neg_matching_arr = np.full(self.num_rows, True)
                subset_matches = True
                for c in subset[1:]:
                    pos_matching_arr = pos_matching_arr & (pos_dict[subset[0]] == pos_dict[c])
                    pos_matching_arr = self.check_results_for_null(pos_matching_arr, None, subset)
                    if pos_matching_arr.tolist().count(False) > self.freq_contamination_level:
                        subset_matches = False
                        break
                    neg_matching_arr = neg_matching_arr & (neg_dict[subset[0]] == neg_dict[c])
                    neg_matching_arr = self.check_results_for_null(neg_matching_arr, None, subset)
                    if neg_matching_arr.tolist().count(False) > self.freq_contamination_level:
                        subset_matches = False
                        break
                if not subset_matches:
                    continue
                found_any = True
                self._process_analysis_binary(
                    test_id,
                    list(subset),
                    pos_matching_arr & neg_matching_arr,
                    f"The columns in {str(subset)[1:-1]} are consistently positive and negative together",
                    ""
                )


    def generate_matched_set_zero_non_zero(self):
        """
        Patterns without exceptions: None: 'all_zero_or_not all' is consistently zero or non-zero with
            'all_zero_or_not rand_a'. However, it is not reported as it is part of a set of 3 feature with near-perfect
            matching.
        Patterns with exception: 'all_zero_or_not most' consistently has the same sign as 'all_zero_or_not rand_a',
            and 'all_zero_or_not all', with the exception of row 999.
        """
        self._add_synthetic_column('all_zero_or_not rand_a',
                                    [random.randint(-2, 2) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('all_zero_or_not rand_b',
                                    [random.randint(-1000, 1000) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('all_zero_or_not all',
                                    self.synth_df['all_zero_or_not rand_a'] * random.randint(1, 100))
        self._add_synthetic_column('all_zero_or_not most',
                                    self.synth_df['all_zero_or_not rand_a'] * random.randint(1, 100))
        if self.synth_df.at[999, 'all_zero_or_not most'] == 0:
            self.synth_df.at[999, 'all_zero_or_not most'] = 1
        else:
            self.synth_df.at[999, 'all_zero_or_not most'] = 0


    def check_matched_set_zero_non_zero(self, test_id):
        """
        We identify sets of numeric columns, not necessarily on the same scale, that are all both frequently
        zero and frequently non-zero. We then find the subsets of these columns that are consistently zero and
        non-zero together.

        This is partially redundant with MATCHED_ZERO, though identify slightly different things. That test will
        check pairs of features, and will combine patterns without exceptions into larger patterns, but does not look
        at larger groups of features with exceptions, which this will do.

        Handling null values: Null is treated as a non-zero value.
        """

        # Check all columns examined have at least 10% zero and 10% non-zero values.
        cols = []
        sample_zero_dict = {}
        sample_non_zero_dict = {}
        zero_dict = {}
        non_zero_dict = {}
        for _col_idx, col_name in enumerate(self.numeric_cols):
            num_zero = len([x for x in self.orig_df[col_name] if (x == 0)])
            num_non_zero = len([x for x in self.orig_df[col_name] if (x != 0)])
            if num_zero > (self.num_rows * 0.1) and num_non_zero > (self.num_rows * 0.1):
                cols.append(col_name)

                sample_zero_dict[col_name] = self.sample_df[col_name] == 0
                sample_non_zero_dict[col_name] = self.sample_df[col_name] != 0

                zero_dict[col_name] = self.orig_df[col_name] == 0
                non_zero_dict[col_name] = self.orig_df[col_name] != 0

        know_failed_subsets = {}
        printed_subset_size_msg = False

        # Loop through the subsets, from biggest to smallest. Consider only subsets of at least 2 columns.
        # We may flag multiple subsets of the same size, but none that are smaller. This will allow some subsets to
        # be skipped, but is a useful heuristic to reduce execution time.
        num_cols = len(cols)
        found_any = False
        for subset_size in range(num_cols, 1, -1):
            if found_any:
                break

            calc_size = math.comb(len(cols), subset_size)
            skip_subsets = calc_size > self.max_combinations
            if skip_subsets:
                if self.verbose >= 2 and not printed_subset_size_msg:
                    print(f"    Skipping subsets of size {subset_size}. There are {calc_size} subsets. max_combinations"
                           f"is currently set to {self.max_combinations:,}.")
                    printed_subset_size_msg = True
                continue

            subsets = list(combinations(cols, subset_size))
            if self.verbose >= 3:
                print(f"  Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset_idx, subset in enumerate(subsets):
                if self.verbose >= 3 and subset_idx > 0 and subset_idx % 10_000 == 0:
                    print(f"    Examining subset {subset_idx}")
                subset_matches = True

                # Check if this subset has already been determined to be impossible from a portion of a previously
                # checked subset
                for kfs in know_failed_subsets:
                    if set(subset).issuperset(set(kfs)):
                        subset_matches = False
                        break
                if not subset_matches:
                    continue

                sample_zero_matching_arr = np.full(len(self.sample_df), True)
                sample_non_zero_matching_arr = np.full(len(self.sample_df), True)
                zero_matching_arr = np.full(self.num_rows, True)
                non_zero_matching_arr = np.full(self.num_rows, True)

                # Test on a sample of rows
                for c_idx, c in enumerate(subset):  # noqa: B007 - used after the loop
                    # We compare all columns to the first column in the set, so this must be identical
                    if c == subset[0]:
                        continue
                    sample_zero_matching_arr = sample_zero_matching_arr & (sample_zero_dict[subset[0]] == sample_zero_dict[c])
                    if sample_zero_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                    sample_non_zero_matching_arr = sample_non_zero_matching_arr & (sample_non_zero_dict[subset[0]] == sample_non_zero_dict[c])
                    if sample_non_zero_matching_arr.tolist().count(False) > 1:
                        subset_matches = False
                        break
                if not subset_matches:
                    know_failed_subsets[tuple(subset[:c_idx+1])] = True
                    continue

                # Test on the full set of rows
                for c in subset:
                    # We compare all columns to the first column in the set, so this must be identical
                    if c == subset[0]:
                        continue
                    zero_matching_arr = zero_matching_arr & (zero_dict[subset[0]] == zero_dict[c])
                    zero_matching_arr = self.check_results_for_null(zero_matching_arr, None, subset)
                    if zero_matching_arr.tolist().count(False) > self.freq_contamination_level:
                        subset_matches = False
                        break
                    non_zero_matching_arr = non_zero_matching_arr & (non_zero_dict[subset[0]] == non_zero_dict[c])
                    non_zero_matching_arr = self.check_results_for_null(non_zero_matching_arr, None, subset)
                    if non_zero_matching_arr.tolist().count(False) > self.freq_contamination_level:
                        subset_matches = False
                        break
                if not subset_matches:
                    continue
                found_any = True

                num_zeros_str = ""
                if len(subset) < 5:
                    for col_name in subset:
                        num_zeros = self.orig_df[col_name].tolist().count(0)
                        num_zeros_str += (f"Column {col_name} contains {num_zeros} zero values and "
                                          f"{self.num_rows - num_zeros} non-zero values. ")
                self._process_analysis_binary(
                    test_id,
                    list(subset),
                    zero_matching_arr & non_zero_matching_arr,
                    f"{num_zeros_str}The columns are consistently zero or non-zero together.",
                    ""
                )


    def _generate_dt_regressor(self):
        """
        Patterns without exceptions: 'dt regr. 2' may be derived from 'dt regr. 1a' and  'dt regr. 1b'
        Patterns with exception: 'dt regr. 3' may be derived from 'dt regr. 1c' and 'dt regr. 1d' with 1 exception
        """
        def set_y(col1, col2):
            arr = []
            for i in range(self.num_synth_rows):
                if self.synth_df[col1][i] > 50:
                    if self.synth_df[col2][i] > 50:
                        arr.append(0.0 + random.randint(0, 5))
                    else:
                        arr.append(20.0 + random.randint(0, 5))
                else:
                    if self.synth_df[col2][i] > 50:
                        arr.append(40.0 + random.randint(0, 5))
                    else:
                        arr.append(60.0 + random.randint(0, 5))
            return arr

        self.synth_df['dt regr. 1a'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 1b'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 1c'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 1d'] = [random.randint(1, 100) for _ in range(self.num_synth_rows)]
        self.synth_df['dt regr. 2'] = set_y('dt regr. 1a', 'dt regr. 1b')
        self.synth_df['dt regr. 3'] = set_y('dt regr. 1c', 'dt regr. 1d')
        self.synth_df.at[999, 'dt regr. 3'] = self.synth_df.at[999, 'dt regr. 3'] * 10.0


    def _check_dt_regressor(self, test_id):
        # We determine if it's possible to create a small, interpretable decision tree which is accurate and
        # substantially more accurate than a naive model. We check the train score on the decision tree, as
        # it is a constricted model so unlikely to overfit, and does not need to predict new instances, just
        # summarize well the existing data. We measure accuracy in terms of normalized root mean squared error.

        # todo:  this does not work with all_cols=True -- it seems to get confused with many columns and not enough
        #   rows. There may be some way to improve that. There are about 530 cols vs 1000 rows, so maybe just comment.

        # Set the seed to ensure the DT behaves the same each execution of this test
        random.seed(0)
        np.random.seed(0)

        # Determine which columns to drop, and which are categorical and should be one-hot encoded.
        drop_features = self.date_cols.copy()
        categorical_features = self.binary_cols.copy()
        for col_name in self.string_cols:
            if self.orig_df[col_name].nunique() > 5:
                drop_features.append(col_name)
            else:
                categorical_features.append(col_name)

        cols_same_bool_dict = self.get_cols_same_bool_dict(force=True)

        for col_idx, col_name in enumerate(self.numeric_cols):
            if (self.verbose >= 2) and (col_idx > 0) and (col_idx % 10) == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")

            # Skip columns that are largely zero or largely Null
            if (self.num_rows - np.count_nonzero(self.orig_df[col_name])) > (self.num_rows / 2):
                continue
            if self.orig_df[col_name].isna().sum() > (self.num_rows / 2):
                continue
            # Skip columns where both the median & mean are zero, as there is no way to evaluate the accuracy of the
            # model currently.
            if (self.column_medians[col_name] == 0) and (self.orig_df[col_name].mean() == 0):
                continue

            regr = DecisionTreeRegressor(max_leaf_nodes=4, random_state=0)

            x_data = self.orig_df.drop(columns=[col_name])
            x_data = x_data.drop(columns=drop_features)
            uncorrelated_cols = []
            for c in self.numeric_cols:
                if c in x_data.columns:
                    if abs(self.spearman_corr.loc[col_name, c]) > 0.2:
                        x_data[c] = self.numeric_vals_filled[c]
                    elif cols_same_bool_dict[tuple(sorted([col_name, c]))]:
                        uncorrelated_cols.append(c)  # Actually over-correlated, but will remove as well.
                    else:
                        uncorrelated_cols.append(c)
            x_data = x_data.drop(columns=uncorrelated_cols)
            if len(x_data.columns) == 0:
                continue
            if len(categorical_features) > 0:
                x_data = pd.get_dummies(x_data, columns=categorical_features)
            for c in x_data.columns:
                x_data[c] = x_data[c].replace([np.inf, -np.inf, np.nan], x_data[c].median())

            y = self.numeric_vals_filled[col_name]
            y = y.replace([np.inf, -np.inf, np.nan], y.median())

            # Remove the extreme values from y to make the predictor less fit to outliers
            upper_y = y.quantile(0.99)
            lower_y = y.quantile(0.01)
            train_y = y[(y > lower_y) & (y < upper_y)]
            train_x_data = x_data.loc[train_y.index]
            if len(train_x_data) < 50:
                continue

            # A simple model should be derivable from a small number of records. We train on 900 rows for robustness
            # and speed
            train_y = train_y.sample(min(900, len(train_y)), random_state=0)
            train_x_data = x_data.loc[train_y.index]

            regr.fit(train_x_data, train_y)
            y_pred = regr.predict(x_data)

            # Evaluate on a sample
            y_sample = y[:50]
            y_pred_sample = y_pred[:50]
            mae_dt = metrics.median_absolute_error(y_sample, y_pred_sample)
            mae_naive = metrics.median_absolute_error(y_sample, [statistics.mean(y)] * len(y_sample))
            # Normalize the MAE by dividing by the median (or mean if median is zero)
            if self.column_medians[col_name] != 0:
                norm_mae_dt = abs(mae_dt / self.column_medians[col_name])
                norm_mae_naive = abs(mae_naive / self.column_medians[col_name])
            else:
                norm_mae_dt = abs(mae_dt / self.orig_df[col_name].mean())
                norm_mae_naive = abs(mae_naive / self.orig_df[col_name].mean())
            if norm_mae_dt > 0.3:
                continue
            if norm_mae_naive < norm_mae_dt:
                continue

            # Evaluate on the full column
            mae_dt = metrics.median_absolute_error(y, y_pred)
            mae_naive = metrics.median_absolute_error(y, [statistics.mean(y)] * len(y))
            # Normalize the MAE by dividing by the median (or mean if median is zero)
            if self.column_medians[col_name] != 0:
                norm_mae_dt = abs(mae_dt / self.column_medians[col_name])
                norm_mae_naive = abs(mae_naive / self.column_medians[col_name])
            else:
                norm_mae_dt = abs(mae_dt / self.orig_df[col_name].mean())
                norm_mae_naive = abs(mae_naive / self.orig_df[col_name].mean())

            if (norm_mae_dt < 0.1) and (norm_mae_dt < (norm_mae_naive / 8.0)):
                rules = tree.export_text(regr)
                # The export has the column names in the format 'feature_1' and so on. We replace these with the
                # actual column names
                cols = []
                for c_idx, c_name in reversed(list(enumerate(x_data.columns))):
                    rule_col_name = f'feature_{c_idx}'
                    if rule_col_name in rules:
                        #cols.append(c_name)
                        orig_col = c_name
                        for cat_col in categorical_features:
                            if c_name.startswith(cat_col):
                                orig_col = cat_col
                        cols.append(orig_col)
                        rules = rules.replace(rule_col_name, c_name)

                # Clean the split points for categorical features to use the values, not 0.5
                rules = self.get_decision_tree_rules_as_categories(rules, categorical_features)

                errors_arr = (y_pred - y)
                normalized_errors_arr = abs(errors_arr / self.column_medians[col_name])
                test_series = normalized_errors_arr < 0.1
                # Rows with missing values in the target column or the columns used by the DT are not tested
                test_series = test_series | self.orig_df[cols + [col_name]].isna().any(axis=1)
                self._process_analysis_binary(
                    test_id,
                    cols + [col_name],
                    test_series,
                    f'The values in column "{col_name}" are consistently predictable from {cols} based using a decision '
                    f'tree with the following rules: \n{rules}',
                    display_info={'Pred': pd.Series(y_pred)}
                )


    def _generate_lin_regressor(self):
        """
        Patterns without exceptions: 'lin regr 2' is is based on '1a', '1b', and '1c'
        Patterns with exception: 'lin regr 3' is based on '1d', '1e', '1f', with one exception.
        """
        self._add_synthetic_column('lin regr 1a', [random.randint(1, 10)  for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('lin regr 1b', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('lin regr 1c', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('lin regr 1d', [random.randint(1, 10)  for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('lin regr 1e', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('lin regr 1f', [random.randint(1, 100) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column(
            'lin regr 2',
            (40.1 * self.synth_df['lin regr 1a']) + (2.1 * self.synth_df['lin regr 1b']) + (5.1 * self.synth_df['lin regr 1c']))
        self._add_synthetic_column(
            'lin regr 3',
            (40.1 * self.synth_df['lin regr 1d']) + (3.1 * self.synth_df['lin regr 1e']) + (5.1 * self.synth_df['lin regr 1f']))
        self.synth_df.at[999, 'lin regr 3'] = self.synth_df.at[999, 'lin regr 3'] * 10.0


    def _check_lin_regressor(self, test_id):
        """
        We determine if it's possible to create a small, interpretable linear regression which is accurate and
        substantially more accurate than a naive model, which simply predicts the median. We check the in-sample score
        on the linear regression, as it does not need to predict new instances, just summarize well the existing data.
        We measure accuracy in terms of normalized root mean squared error. The linear regression uses only the numeric
        features in the data.
        """

        cols_same_bool_dict = self.get_cols_same_bool_dict(force=True)

        for col_idx, col_name in enumerate(self.numeric_cols):

            if (self.verbose >= 2) and (col_idx > 0) and (col_idx % 10) == 0:
                print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")

            # Skip columns that are largely zero
            if (self.num_rows - np.count_nonzero(self.orig_df[col_name])) > (self.num_rows / 10.0):
                continue

            # Skip columns that are largely Null
            if self.orig_df[col_name].isna().sum() > (self.num_rows / 10.0):
                continue

            regr = Lasso(alpha=0.0)

            # Collect the relevant features and remove any null or inf values from these
            x_data = self.orig_df.drop(columns=[col_name])
            x_data = x_data.drop(columns=self.binary_cols + self.date_cols + self.string_cols)
            uncorrelated_cols = []
            for c in x_data.columns:
                if cols_same_bool_dict[tuple(sorted([col_name, c]))]:
                    uncorrelated_cols.append(c)  # Actually over-correlated, but removed as well.
                elif abs(self.pearson_corr.loc[col_name, c]) > 0.2:
                    x_data[c] = self.numeric_vals_filled[c]
                    x_data[c] = x_data[c].fillna(self.column_medians[c])
                    x_data[c] = x_data[c].replace([np.inf, -np.inf], self.column_medians[c])
                else:
                    uncorrelated_cols.append(c)
            x_data = x_data.drop(columns=uncorrelated_cols)
            if len(x_data.columns) == 0:
                continue

            # Clean the target column
            y = self.numeric_vals_filled[col_name]
            y = y.fillna(self.column_medians[col_name])
            y = y.replace([np.inf, -np.inf], self.column_medians[col_name])

            # Remove the extreme values from y to make the predictor less fit to outliers
            upper_y = y.quantile(0.99)
            lower_y = y.quantile(0.01)
            train_y = y[(y > lower_y) & (y < upper_y)]
            train_x_data = x_data.loc[train_y.index]
            if len(train_x_data) < 50:
                continue

            # A simple model should be derivable from a small number of records. We train on 900 rows for robustness
            # and speed
            train_y = train_y.sample(min(900, len(train_y)), random_state=0)
            train_x_data = x_data.loc[train_y.index]

            # Scale the data
            scaler = RobustScaler()
            cols = train_x_data.columns
            train_x_data = scaler.fit_transform(train_x_data)
            train_x_data = pd.DataFrame(train_x_data, columns=cols)
            x_data_scaled = scaler.transform(x_data)
            x_data_scaled = pd.DataFrame(x_data_scaled, columns=cols)

            # Scale the target
            orig_y = y
            scaler = RobustScaler()
            train_y = scaler.fit_transform(train_y.values.reshape(-1, 1)).reshape(1, -1)[0]
            y = scaler.fit_transform(y.values.reshape(-1, 1)).reshape(1, -1)[0]

            try:
                regr.fit(train_x_data, train_y)
            except Exception as e:
                if self.DEBUG_MSG:
                    if colored:
                        print(colored(f"Error fitting Linear Regression in {test_id}: {e}", 'red'))
                    else:
                        print(f"Error fitting Linear Regression in {test_id}: {e}")
                continue

            y_pred = regr.predict(x_data_scaled)
            mae_lr = metrics.median_absolute_error(y, y_pred)
            mae_naive = metrics.median_absolute_error(y, [statistics.median(y)] * len(y))

            # Normalize the MAE by dividing by the median (or mean if median is zero)
            if self.column_medians[col_name] != 0:
                norm_mae_lr = abs(mae_lr / self.column_medians[col_name])
                norm_mae_naive = abs(mae_naive / self.column_medians[col_name])
            else:
                norm_mae_lr = abs(mae_lr / self.orig_df[col_name].mean())
                norm_mae_naive = abs(mae_naive / self.orig_df[col_name].mean())

            if (norm_mae_lr < 0.1) and (norm_mae_lr < (norm_mae_naive / 10.0)):
                # Get a string representation of the linear regression formula
                regression_formula = f"{regr.intercept_:.3f} + "
                num_coefs = 0
                features_used = []
                for x_col_idx, x_col in enumerate(x_data.columns):
                    # todo: this is too stringent if the columns are on different scales, but this does keep the
                    #  formulas interpretable
                    if regr.coef_[x_col_idx] >= 0.01:
                        regression_formula += f' {regr.coef_[x_col_idx]:.2f} * "{x_col}" + '
                        num_coefs += 1
                        features_used.append(x_col)
                regression_formula = regression_formula[:-2]  # remove the trailing + sign

                # Ignore cases where the column is a constant that can be predicted from a single intercept. There are
                # simpler tests for this case.
                if num_coefs == 0:
                    continue

                # Ignore cases where the column is a constant plus another column.
                if num_coefs == 1 and round(max(regr.coef_)) == 1.0:
                    continue

                # Determine if there are too many coefficients given the number of rows. If so, remove the least
                # predictive (not sure how to figure that out, maybe with 1d linear regressions -- can't use the
                # coefficients since not scaled, and when scale, the intercepts are weird) and see if we still have
                # a good model. todo: code this.

                # Determine if the formula is still accurate when using just the selected columns
                # todo: fill in

                # Switch back to the original scale to calculate the errors and to display the predictions
                y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(1, -1)[0]

                errors_arr = (y_pred - orig_y)
                normalized_errors_arr = abs(errors_arr / self.column_medians[col_name])
                test_series = normalized_errors_arr < 0.5
                self._process_analysis_binary(
                    test_id,
                    features_used + [col_name],
                    test_series,
                    (f'The column "{col_name}" contains values that are consistently predictable based on a linear '
                     f'regression formula: \n{regression_formula}. (The co-efficients are based on scaled values '
                     f'and do not apply the the original values. This is to represent the relative importances of the '
                     f'features)'),
                    display_info={'Pred': pd.Series(y_pred)}
                )


    def _generate_small_vs_corr_cols(self):
        """
        Patterns without exceptions: None. This test does not flag patterns. There are 2 clusters of columns, but this
            is not flagged as a pattern.
        Patterns with exception: 'small_vs_corr_cols_2'
        """
        for i in range(5):
            self._add_synthetic_column(f'small_vs_corr_cols_{i}',
                                        sorted([random.random() for _ in range(self.num_synth_rows)], reverse=False))
        self.synth_df.loc[999, 'small_vs_corr_cols_2'] = 0.0000001
        for i in range(3):
            self._add_synthetic_column(f'small_vs_corr_cols_{i+5}',
                                        sorted([random.random() for _ in range(self.num_synth_rows)], reverse=True))
        for i in range(3):
            self._add_synthetic_column(f'small_vs_corr_cols_{i+8}',
                                        [random.random() for _ in range(self.num_synth_rows)])

    def _get_column_clusters(self):
        def get_next_corr_pair(corr_matrix, correlated_sets_arr):
            for i in corr_matrix.index:
                for j in corr_matrix.index:
                    if i == j:
                        continue
                    if corr_matrix[i][j] > 0.9:
                        already_in_set = False
                        for csa in correlated_sets_arr:
                            if (i in csa) or (j in csa):
                                already_in_set = True
                        if already_in_set:
                            continue
                        return {i, j}
            return None

        # todo: tighten up so they are all reasonably correlated with each other -- at least 0.9 with at least 1/2 of the other columns in the set
        def get_rest_set(corr_matrix, curr_set, correlated_sets_arr):
            for i in corr_matrix.index:
                for j in corr_matrix.index:
                    if i == j:
                        continue
                    if (corr_matrix[i][j] > 0.9) and ((i in curr_set) or (j in curr_set)):
                        for csa in correlated_sets_arr:
                            if (i in csa) or (j in csa):
                                continue
                        curr_set.add(i)
                        curr_set.add(j)
            return curr_set

        sub_df = self.orig_df[self.numeric_cols].sample(n=min(self.num_rows, 2000), random_state=0)
        corr_matrix = sub_df.corr(method='spearman')
        correlated_sets_arr=[]
        pair = get_next_corr_pair(corr_matrix, correlated_sets_arr)
        while pair:
            full_set = get_rest_set(corr_matrix, pair, correlated_sets_arr)
            correlated_sets_arr.append(full_set)
            pair = get_next_corr_pair(corr_matrix, correlated_sets_arr)
        return correlated_sets_arr


    def _check_small_vs_corr_cols(self, test_id):
        """
        This test is currently disabled. The idea is: for each column, we compare each value to the values in the
        same row for all correlated columns. This first finds the clusters of correlated columns. Probably instead
        it should just find, for each column, the set of columns that are reasonably correlated. Then it converts
        all values to the percentile relative to the column. It compares each percentile to the average percentile
        for that row. This works okay, but it may not give much more than comparing correlated columns pairwise, and
        is more difficult to explain and present.
        """
        correlated_sets_arr = self._get_column_clusters()
        if len(correlated_sets_arr) == 0:
            return
        if self.verbose >= 2:
            print("  Identified the clusters of correlated columns: ", correlated_sets_arr)

        for cluster_idx, cluster in enumerate(correlated_sets_arr):
            if self.verbose >= 2:
                print(f"  Examining cluster {cluster_idx} of {len(correlated_sets_arr)} clusters of columns")
            df_rank = pd.DataFrame()
            for col_name in cluster:
                df_rank[col_name] = self.orig_df[col_name].rank(pct=True)
            df_rank['Avg Percentile'] = df_rank.mean(axis=1)

            for col_name in cluster:
                test_series = (df_rank['Avg Percentile'] - df_rank[col_name]) < 0.5
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    (f'Column "{col_name}" contains values that are small compared to the values in similar columns: '
                     f'{cluster}'),
                    display_info={"cluster": cluster}
                )


    def _generate_large_vs_corr_cols(self):
        """
        Patterns without exceptions: None. This test does not flag patterns. There are 2 clusters of columns, but this
            is not flagged as a pattern.
        Patterns with exception: 'large_vs_corr_cols_2'
        """
        for i in range(3):
            self._add_synthetic_column(f'large_vs_corr_cols_{i}',
                                        sorted([random.random() for _ in range(self.num_synth_rows)], reverse=False))
        for i in range(5):
            self._add_synthetic_column(f'large_vs_corr_cols_{i+3}',
                                        sorted([random.random() for _ in range(self.num_synth_rows)], reverse=True))
        self.synth_df.loc[999, 'large_vs_corr_cols_5'] = 2.0
        for i in range(3):
            self._add_synthetic_column(f'large_vs_corr_cols_{i+8}',
                                        [random.random() for _ in range(self.num_synth_rows)])


    def _check_large_vs_corr_cols(self, test_id):
        correlated_sets_arr = self._get_column_clusters()
        if len(correlated_sets_arr) == 0:
            return

        if self.verbose >= 2:
            print("  Identified the clusters of correlated columns: ", correlated_sets_arr)

        for cluster_idx, cluster in enumerate(correlated_sets_arr):
            if self.verbose >= 2:
                print(f"  Examining cluster {cluster_idx} of {len(correlated_sets_arr)} clusters of columns")
            df_rank = pd.DataFrame()
            for col_name in cluster:
                df_rank[col_name] = self.orig_df[col_name].rank(pct=True)
            df_rank['Avg Percentile'] = df_rank.mean(axis=1)

            for col_name in cluster:
                test_series = ( df_rank[col_name] - df_rank['Avg Percentile']) < 0.5
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    test_series,
                    (f'Column "{col_name}" contains values that are large compared to the values in similar columns: '
                     f'{cluster}'),
                    display_info={"cluster": cluster}
                )


    def _generate_predict_null(self):
        """
        Patterns without exceptions: the Null's in 'predict_null all' are consistently predictable from
            'predict_null rand_b'
        Patterns with exception: the Null's in 'predict_null most' are consistently predictable from
            'predict_null rand_b'
        """
        self._add_synthetic_column('predict_null rand_a', np.random.choice(['a', 'b', 'c', 'd'], self.num_synth_rows))
        self._add_synthetic_column('predict_null rand_b',
                                    [random.randint(1, 1000) for _ in range(self.num_synth_rows)])
        self.synth_df.at[999, 'predict_null rand_b'] = 985  # Set to be known to be over 800,
        vals2 = np.random.randint(1, 1000, size=self.num_synth_rows)
        vals2 = np.where(self.synth_df['predict_null rand_b'] > 800, None, vals2)
        self._add_synthetic_column('predict_null all', vals2)
        self._add_synthetic_column('predict_null most', vals2)
        self.synth_df.at[999, 'predict_null most'] = 833


    def _check_predict_null(self, test_id):
        """
        This creates a set of binary classifiers, in the form of small, interpretable decision trees, to predict the
        null values in each column. Each model uses all other columns, other than those that have Null values in
        approximately the same rows. The MATCHED_MISSING test is intended to detect where two or more columns have
        missing values in the same rows. Often datasets have multiple columns with in-sync missing values. This test,
        however, attempts to predict those missing values from the other columns that are not in sync.
        """

        # Set the seed to ensure the DT behaves the same each execution of this test
        random.seed(0)
        np.random.seed(0)
        is_missing_dict = self.get_is_missing_dict()

        # drop_features are the columns we do not use as feature columns. We may predict the nulls in these columns.
        drop_features = []
        categorical_features = []
        for col_name in self.orig_df.columns:
            if col_name in self.string_cols:
                if self.orig_df[col_name].nunique() > 5:
                    drop_features.append(col_name)
                else:
                    categorical_features.append(col_name)
            if col_name in self.binary_cols:
                categorical_features.append(col_name)
            if col_name in self.date_cols:
                drop_features.append(col_name)

        # Pre-calculate the number of missing values in each column
        num_missing_dict = {}
        for col_name in self.orig_df.columns:
            num_missing_dict[col_name] = is_missing_dict[col_name].tolist().count(True)

        for col_name in self.orig_df.columns:
            target_col = is_missing_dict[col_name]

            # Check if almost all Null or non-Null
            if target_col.tolist().count(True) < math.sqrt(self.num_rows):
                continue
            if target_col.tolist().count(False) < math.sqrt(self.num_rows):
                continue

            x_df = self.orig_df.copy()

            # Drop categorical columns with too many unique values
            x_df = x_df.drop(columns=drop_features)

            # Drop the target column
            if col_name not in drop_features:
                x_df = x_df.drop(columns=col_name)

            # Drop any columns that have almost the same set of Null values as the target column
            matching_cols = []
            for col_name_2 in x_df.columns:
                if abs(num_missing_dict[col_name] - num_missing_dict[col_name_2]) > self.freq_contamination_level:
                    continue
                num_matching = len([1 for x, y in zip(is_missing_dict[col_name], is_missing_dict[col_name_2]) if x == y])
                if num_matching > (self.num_rows - self.freq_contamination_level):
                    matching_cols.append(col_name_2)
            x_df = x_df.drop(columns=matching_cols)

            if len(x_df.columns) == 0:
                continue

            # Rows with a missing value in any of the X columns neither support nor violate the pattern, so the DT is
            # fit and evaluated only on the other rows. Check again these are not almost all Null or non-Null.
            rows_used = ~pd.concat([is_missing_dict[c] for c in x_df.columns], axis=1).any(axis=1)
            if (target_col[rows_used].tolist().count(True) < math.sqrt(self.num_rows)) or \
                    (target_col[rows_used].tolist().count(False) < math.sqrt(self.num_rows)):
                continue
            x_df = x_df[rows_used]

            # One-hot encode any categorical X columns
            use_categorical_features = categorical_features.copy()
            if col_name in use_categorical_features:
                use_categorical_features.remove(col_name)
            use_categorical_features = [x for x in use_categorical_features if x in x_df.columns]
            x_df = pd.get_dummies(x_df, columns=use_categorical_features)
            x_df = x_df.fillna(sys.maxsize)

            # Ensure any numeric columns are treated as numeric and have NaN values filled
            for num_col in self.numeric_cols:
                if num_col in x_df.columns:
                    x_df[num_col] = self.numeric_vals_filled[num_col]
                    x_df[num_col] = x_df[num_col].fillna(self.column_medians[num_col])

            # Create, fit, and test the DT. We create as simple of a tree as possible to get a good level of accuracy.
            for max_leaf_nodes in range(2, 9):
                clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
                clf.fit(x_df, target_col[rows_used])
                y_pred = clf.predict(x_df)
                f1_dt = metrics.f1_score(target_col[rows_used], y_pred, average='macro')
                if f1_dt > 0.9:
                    break

            if f1_dt > 0.9:
                rules = tree.export_text(clf)

                # Clean the rules to use the original feature names. We go through in reverse order so we don't
                # have, for example, feature_1 matching feature_11
                cols = []
                for c_idx, c_name in reversed(list(enumerate(x_df.columns))):
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

                # There is no prediction for the rows not used
                y_pred = pd.Series(y_pred, index=x_df.index).reindex(self.orig_df.index)
                test_series = (target_col == y_pred) | ~rows_used
                self._process_analysis_binary(
                    test_id,
                    cols + [col_name],
                    test_series,
                    f'The Null values in column "{col_name}" (with {num_missing_dict[col_name]} null and '
                    f'{self.num_rows - num_missing_dict[col_name]} non-null values) are consistently '
                    f'predictable from {cols} based using a decision tree with the following rules: \n{rules}',
                    display_info={'Pred': y_pred.replace({True: 'Null', False: 'Non-Null'})}
                )

    ##################################################################################################################
    # Data consistency checks for single date columns
    ##################################################################################################################


