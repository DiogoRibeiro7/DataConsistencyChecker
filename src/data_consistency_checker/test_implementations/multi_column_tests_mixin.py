"""
MultiColumnTestsMixin: Test methods for multi column tests.

This mixin contains 18 test methods organized by category.
Extracted from check_data_consistency.py for better code organization.
"""

from __future__ import annotations

import math
import numbers
import random
import statistics
from itertools import combinations

import numpy as np
import pandas as pd

from data_consistency_checker.checker_state import CheckerState
from data_consistency_checker.checker_utils import map_elements


class MultiColumnTestsMixin(CheckerState):
    """
    Mixin class containing multi column tests methods.

    This class contains 18 methods for testing data consistency
    related to multi column operations.

    This is a mixin class designed to be used with multiple inheritance.
    It does not have an __init__ method and relies on the parent class
    to provide necessary attributes and methods.
    """

    def _generate_c_is_a_or_b(self):
        """
        Patterns without exceptions: 'c_is_a_or_b_all' is consistently the same as either 'c_is_a_or_b_a' or
            'c_is_a_or_b_b'
        Patterns with exception: 'c_is_a_or_b_most' is consistently the same as either 'c_is_a_or_b_a' or
            'c_is_a_or_b_b', with 1 exception.
        """
        self._add_synthetic_column('c_is_a_or_b_a', np.random.randint(0, 50, self.num_synth_rows))
        self._add_synthetic_column('c_is_a_or_b_b', np.random.randint(0, 50, self.num_synth_rows))
        self._add_synthetic_column('c_is_a_or_b_all',
                                    self.synth_df['c_is_a_or_b_a'][:500].tolist() +
                                    self.synth_df['c_is_a_or_b_b'][500:].tolist())
        self._add_synthetic_column('c_is_a_or_b_most',
                                    self.synth_df['c_is_a_or_b_a'][:500].tolist() +
                                    self.synth_df['c_is_a_or_b_b'][500:].tolist())
        self.synth_df.loc[999, 'c_is_a_or_b_most'] = -87

    def _check_c_is_a_or_b(self, test_id):
        """
        Given a set of columns, referred to as Column A, Column B, and Column C, check if Column C is consistently
        the same value as in either Column A or Column B, but not consistently the same as Column A or as Column B.
        This is done for string, numeric, and date columns, but not binary.

        Handling null values: A and C are considered a match if both are Null, and not a match if one, but not both,
        is Null. Similar for B and C.
        """
        def check_triple():
            if col_name_c in (col_name_a, col_name_b):
                return

            # if C is A or B, we don't check as well if A is C or B, or if B is A or C. This is not to save time, but
            # to remove over-reporting.
            columns_tuple = tuple(sorted([col_name_a, col_name_b, col_name_c]))
            if columns_tuple in reported_dict:
                return

            # Test if any of the 3 columns are mostly a single value.
            if most_freq_value_dict[col_name_a] > most_freq_value_limit:
                return
            if most_freq_value_dict[col_name_b] > most_freq_value_limit:
                return
            if most_freq_value_dict[col_name_c] > most_freq_value_limit:
                return

            # Test if any 2 columns are mostly the same
            if cols_same_bool_dict[tuple(sorted([col_name_a, col_name_b]))]:
                return
            if cols_same_bool_dict[tuple(sorted([col_name_a, col_name_c]))]:
                return
            if cols_same_bool_dict[tuple(sorted([col_name_b, col_name_c]))]:
                return

            # Test if either A or B is almost never equal to C
            if cols_same_count_dict[tuple(sorted([col_name_a, col_name_c]))] < math.sqrt(self.num_rows):
                return
            if cols_same_count_dict[tuple(sorted([col_name_b, col_name_c]))] < math.sqrt(self.num_rows):
                return

            # Test on a sample
            test_series = [z in (x, y) for x, y, z in
                           zip(self.sample_df[col_name_a], self.sample_df[col_name_b], self.sample_df[col_name_c])]
            test_series = np.array(test_series) | \
                          sample_col_pair_both_null_dict[tuple(sorted([col_name_a, col_name_c]))] | \
                          sample_col_pair_both_null_dict[tuple(sorted([col_name_b, col_name_c]))]

            if test_series.tolist().count(False) > 1:
                return

            # Test of the full data
            test_series = [z in (x, y) for x, y, z in
                           zip(self.orig_df[col_name_a], self.orig_df[col_name_b], self.orig_df[col_name_c])]
            test_series = np.array(test_series) | \
                          col_pair_both_null_dict[tuple(sorted([col_name_a, col_name_c]))] | \
                          col_pair_both_null_dict[tuple(sorted([col_name_b, col_name_c]))]
            num_matching = test_series.tolist().count(True)
            if num_matching >= (self.num_rows - self.freq_contamination_level):
                matching_arr = [
                    "BOTH" if ((a == b == c) or (a_na and b_na and c_na))
                    else col_name_a if ((c == a) or (c_na and a_na))
                    else col_name_b if ((c == b) or (c_na and b_na))
                    else "NONE"
                    for a, b, c, a_na, b_na, c_na in
                    zip(self.orig_df[col_name_a], self.orig_df[col_name_b], self.orig_df[col_name_c],
                        self.orig_df[col_name_a].isna(), self.orig_df[col_name_b].isna(),
                        self.orig_df[col_name_c].isna())]

                # Determine again how often C matched A and B. The test above allowed cases where it equalled both
                if matching_arr.count(col_name_a) < self.freq_contamination_level:
                    return
                if matching_arr.count(col_name_b) < self.freq_contamination_level:
                    return

                # Calculating where they match on Null values is simply to include this information in the pattern
                # description
                matching_on_none_arr = [
                    "BOTH" if (a_na and b_na and c_na)
                    else col_name_a if (c_na and a_na)
                    else col_name_b if (c_na and b_na)
                    else "NONE"
                    for a_na, b_na, c_na in
                    zip(self.orig_df[col_name_a].isna(), self.orig_df[col_name_b].isna(),
                        self.orig_df[col_name_c].isna())]

                null_frac_both_str = ""
                if matching_arr.count("BOTH") > 0:
                    null_frac_both_str = (f'({matching_on_none_arr.count("BOTH") * 100.0 / matching_arr.count("BOTH"):.3f}'
                                          f'% of these on Null values)')

                self._process_analysis_binary(
                    test_id,
                    [col_name_a, col_name_b, col_name_c],
                    test_series,
                    (f'Column "{col_name_c}" matches "{col_name_a}" {matching_arr.count(col_name_a)} times '
                     f'({matching_on_none_arr.count(col_name_a) * 100.0 / matching_arr.count(col_name_a):.3f}% of '
                     f'these on Null values); '
                     f'matches "{col_name_b}" {matching_arr.count(col_name_b)} times '
                     f'({matching_on_none_arr.count(col_name_b) * 100.0 / matching_arr.count(col_name_b):.3f}% of '
                     f'these on Null values); '
                     f'matches both {matching_arr.count("BOTH")} times {null_frac_both_str}; '
                     f'and matches neither {matching_arr.count("NONE")} times. '
                     f'The values in "{col_name_c}" are consistently the same as those in either "{col_name_a}" '
                     f'or "{col_name_b}"'),
                    display_info={'Same Column': matching_arr}
                )
                reported_dict[columns_tuple] = True

        def calc_num_combos(num_cols):
            return num_cols * (num_cols * (num_cols-1) / 2)

        reported_dict = {}
        cols_same_bool_dict = self.get_cols_same_bool_dict()
        cols_same_count_dict = self.get_cols_same_count_dict()
        most_freq_value_dict = self.get_count_most_freq_value_dict()
        most_freq_value_limit = self.num_rows * 0.9

        # Check triples of numeric columns
        num_combos = calc_num_combos(len(self.numeric_cols))
        if num_combos > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping numeric columns. There are {len(self.numeric_cols)} numeric columns, which leads to "
                       f"{int(num_combos):,} combinations. max_combinations is currently set to {self.max_combinations:,}.")
        else:
            sample_col_pair_both_null_dict = self.get_sample_col_pair_both_null_dict(force=True)
            col_pair_both_null_dict = self.get_col_pair_both_null_dict(force=True)
            for col_idx, col_name_c in enumerate(self.numeric_cols):  # noqa: B007 - read by the nested check function
                if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                    print(f"  Examining column {col_idx} of {len(self.numeric_cols)} numeric columns")
                num_pairs, pairs_arr = self._get_numeric_column_pairs_unique()
                for _pair_idx, (col_name_a, col_name_b) in enumerate(pairs_arr):  # noqa: B007 - read by the nested check function
                    check_triple()

        # Check triples of string columns
        num_combos = calc_num_combos(len(self.string_cols))
        if num_combos > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping string columns. There are {len(self.string_cols)} string columns, which leads to "
                       f"{int(num_combos):,} combinations. max_combinations is currently set to {self.max_combinations:,}.")
        else:
            sample_col_pair_both_null_dict = self.get_sample_col_pair_both_null_dict(force=True)
            col_pair_both_null_dict = self.get_col_pair_both_null_dict(force=True)
            for col_idx, col_name_c in enumerate(self.string_cols):  # noqa: B007 - read by the nested check function
                if self.verbose >= 2 and col_idx > -1 and col_idx % 1 == 0:
                    print(f"  Examining column {col_idx} of {len(self.string_cols)} string columns")
                num_pairs, pairs_arr = self._get_string_column_pairs_unique()
                for _pair_idx, (col_name_a, col_name_b) in enumerate(pairs_arr):  # noqa: B007 - read by the nested check function
                    check_triple()

        # Check triples of date columns
        num_combos = calc_num_combos(len(self.date_cols))
        if num_combos > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping date columns. There are {len(self.date_cols)} numeric columns, which leads to "
                       f"{int(num_combos):,} combinations. max_combinations is currently set to {self.max_combinations:,}.")
        else:
            sample_col_pair_both_null_dict = self.get_sample_col_pair_both_null_dict(force=True)
            col_pair_both_null_dict = self.get_col_pair_both_null_dict(force=True)
            for col_idx, col_name_c in enumerate(self.date_cols):  # noqa: B007 - read by the nested check function
                if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                    print(f"  Examining column {col_idx} of {len(self.date_cols)} date columns")
                num_pairs, pairs_arr = self._get_date_column_pairs_unique()
                for _pair_idx, (col_name_a, col_name_b) in enumerate(pairs_arr):  # noqa: B007 - read by the nested check function
                    check_triple()

    '''
    # ChatGPT below. Looks better, but needs testing

    def _check_c_is_a_or_b(self, test_id):
        def check_triple(col_name_a, col_name_b, col_name_c):
            if col_name_c in (col_name_a, col_name_b):
                return

            columns_tuple = tuple(sorted([col_name_a, col_name_b, col_name_c]))
            if columns_tuple in reported_dict:
                return

            if most_freq_value_dict[col_name_a] > most_freq_value_limit:
                return
            if most_freq_value_dict[col_name_b] > most_freq_value_limit:
                return
            if most_freq_value_dict[col_name_c] > most_freq_value_limit:
                return

            if cols_same_bool_dict[tuple(sorted([col_name_a, col_name_b]))]:
                return
            if cols_same_bool_dict[tuple(sorted([col_name_a, col_name_c]))]:
                return
            if cols_same_bool_dict[tuple(sorted([col_name_b, col_name_c]))]:
                return

            if cols_same_count_dict[tuple(sorted([col_name_a, col_name_c]))] < self.freq_contamination_level:
                return
            if cols_same_count_dict[tuple(sorted([col_name_b, col_name_c]))] < self.freq_contamination_level:
                return

            test_series = [
                (z == x) or (z == y)
                for x, y, z in zip(self.sample_df[col_name_a], self.sample_df[col_name_b], self.sample_df[col_name_c])
            ]
            test_series = (
                    test_series
                    | sample_col_pair_both_null_dict[tuple(sorted([col_name_a, col_name_c]))]
                    | sample_col_pair_both_null_dict[tuple(sorted([col_name_b, col_name_c]))]
            )

            if test_series.tolist().count(False) > 1:
                return

            test_series = [
                (z == x) or (z == y)
                for x, y, z in zip(self.orig_df[col_name_a], self.orig_df[col_name_b], self.orig_df[col_name_c])
            ]
            test_series = (
                    test_series
                    | col_pair_both_null_dict[tuple(sorted([col_name_a, col_name_c]))]
                    | col_pair_both_null_dict[tuple(sorted([col_name_b, col_name_c]))]
            )
            num_matching = test_series.tolist().count(True)
            if num_matching >= (self.num_rows - self.freq_contamination_level):
                self._process_analysis_binary(
                    test_id,
                    self.get_col_set_name([col_name_a, col_name_b, col_name_c]),
                    [col_name_a, col_name_b, col_name_c],
                    test_series,
                    (f'The values in "{col_name_c}" are consistently the same as those in either "{col_name_a}" '
                     f'or "{col_name_b}".')
                )
                reported_dict[columns_tuple] = True

        def process_column_pairs(column_type, column_list):
            def calc_num_combos(num_cols):
                return num_cols * (num_cols * (num_cols-1) / 2)

            num_combos = calc_num_combos(len(column_list))
            if num_combos > self.max_combinations:
                if self.verbose >= 1:
                    print(f"  Skipping {column_type} columns. There are {len(column_list)} {column_type} columns, "
                          f"which leads to {int(num_combos):,} combinations. "
                          f"max_combinations is currently set to {self.max_combinations:,}.")
            else:
                sample_col_pair_both_null_dict = self.get_sample_col_pair_both_null_dict(force=True)
                col_pair_both_null_dict = self.get_col_pair_both_null_dict(force=True)
                for col_idx, col_name_c in enumerate(column_list):
                    if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                        print(f"  Examining column {col_idx} of {len(column_list)} {column_type} columns")
                    pairs_arr = get_column_pairs_unique(column_type)
                    for col_name_a, col_name_b in pairs_arr:
                        check_triple(col_name_a, col_name_b, col_name_c)

        reported_dict = {}
        cols_same_bool_dict = self.get_cols_same_bool_dict()
        cols_same_count_dict = self.get_cols_same_count_dict()
        most_freq_value_dict = self.get_count_most_freq_value_dict()
        most_freq_value_limit = self.num_rows * 0.9

        process_column_pairs("numeric", self.numeric_cols)
        process_column_pairs("string", self.string_cols)
        process_column_pairs("date", self.date_cols)
    '''

    ##################################################################################################################
    # Data consistency checks for sets of four columns of any type
    ##################################################################################################################


    def _generate_two_pairs(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'two_pairs rand_a' and 'two_pairs rand_b' have equal values in the same rows as
            'two_pairs rand_c' and 'two_pairs rand_d', with 1 exception
        """
        self._add_synthetic_column('two_pairs_rand_a', np.random.randint(0, 5, self.num_synth_rows))
        self._add_synthetic_column('two_pairs_rand_b', np.random.randint(0, 5, self.num_synth_rows))
        self._add_synthetic_column('two_pairs_rand_c', np.random.randint(0, 5, self.num_synth_rows))
        a_b_match_arr = [x == y for x, y, in zip(self.synth_df['two_pairs_rand_a'], self.synth_df['two_pairs_rand_b'])]
        self._add_synthetic_column('two_pairs_rand_d',
                                    [x if y else x + 1 for x, y in zip(self.synth_df['two_pairs_rand_c'], a_b_match_arr)])
        self.synth_df.loc[999, 'two_pairs_rand_d'] = \
            self.synth_df.loc[999, 'two_pairs_rand_c'] + 1 if a_b_match_arr[999] \
                else self.synth_df.loc[999, 'two_pairs_rand_c']


    def _check_two_pairs(self, test_id):

        # Get the column type of each column
        column_types_dict = {}
        for col_name in self.binary_cols:
            column_types_dict[col_name] = 0  # Using integer codes for fast comparison
        for col_name in self.numeric_cols:
            column_types_dict[col_name] = 1
        for col_name in self.string_cols:
            column_types_dict[col_name] = 2
        for col_name in self.date_cols:
            column_types_dict[col_name] = 3

        sample_is_na_dict = {}
        for col_name in self.orig_df.columns:
            sample_is_na_dict[col_name] = self.sample_df[col_name].isna()

        is_na_dict = {}
        for col_name in self.orig_df.columns:
            is_na_dict[col_name] = self.orig_df[col_name].isna()

        # Set the minimum number of rows where 2 columns must match and, also, must not match. We require a minimum
        # amount of both.
        match_sample_okay_limit = len(self.sample_df) / 10.0
        match_okay_limit = self.num_rows / 10.0

        # To avoid calculating the matching for pairs of columns multiple times, we cache their matching.
        sample_pairs_match_bool_dict = {}
        pairs_match_arr_dict = {}

        num_combinations_tested = 0

        for col_idx_1 in range(len(self.orig_df.columns)-3):
            col_name_1 = self.orig_df.columns[col_idx_1]
            if self.verbose >= 2 and col_idx_1 > 0 and col_idx_1 % 25 == 0:
                print(f"  Examining column {col_idx_1} of {len(self.orig_df.columns)} columns")
            for col_idx_2 in range(col_idx_1+1, len(self.orig_df.columns)):
                col_name_2 = self.orig_df.columns[col_idx_2]
                if column_types_dict[col_name_1] != column_types_dict[col_name_2]:
                    continue

                if (col_name_1, col_name_2) in sample_pairs_match_bool_dict:
                    if not sample_pairs_match_bool_dict[(col_name_1, col_name_2)]:
                        continue

                # Test on the sample data if col_name_1 and col_name_2 are often, but not too often, the same value
                match_1_2_sample_arr = [(x == y) or (w and z) for x, y, w, z in
                                        zip(self.sample_df[col_name_1],
                                            self.sample_df[col_name_2],
                                            sample_is_na_dict[col_name_1],
                                            sample_is_na_dict[col_name_2])]
                if (match_1_2_sample_arr.count(True) < match_sample_okay_limit) or \
                        (match_1_2_sample_arr.count(False) < match_sample_okay_limit):
                    sample_pairs_match_bool_dict[(col_name_1, col_name_2)] = False
                    continue
                sample_pairs_match_bool_dict[(col_name_1, col_name_2)] = False

                if (col_name_1, col_name_2) in pairs_match_arr_dict:
                    if not pairs_match_arr_dict[(col_name_1, col_name_2)]:
                        continue

                match_1_2_arr = [(x == y) or (w and z) for x, y, w, z in
                                        zip(self.orig_df[col_name_1],
                                            self.orig_df[col_name_2],
                                            is_na_dict[col_name_1],
                                            is_na_dict[col_name_2])]
                pairs_match_arr_dict[(col_name_1, col_name_2)] = match_1_2_arr
                if (match_1_2_arr.count(True) < match_okay_limit) or (match_1_2_arr.count(False) < match_okay_limit):
                    continue

                for col_idx_3 in range(col_idx_1+1, len(self.orig_df.columns)-1):
                    if col_idx_3 == col_idx_2:
                        continue
                    col_name_3 = self.orig_df.columns[col_idx_3]
                    for col_idx_4 in range(col_idx_3+1, len(self.orig_df.columns)):
                        if col_idx_4 == col_idx_2:
                            continue
                        col_name_4 = self.orig_df.columns[col_idx_4]
                        if column_types_dict[col_name_4] != column_types_dict[col_name_3]:
                            continue

                        # Test on the sample data if col_name_3 and col_name_3 are often, but not too often, the same
                        # value
                        if (col_name_3, col_name_4) in sample_pairs_match_bool_dict:
                            if not sample_pairs_match_bool_dict[(col_name_3, col_name_4)]:
                                continue
                        else:
                            match_3_4_sample_arr = [(x == y) or (w and z) for x, y, w, z in
                                                    zip(self.sample_df[col_name_3],
                                                        self.sample_df[col_name_4],
                                                        sample_is_na_dict[col_name_3],
                                                        sample_is_na_dict[col_name_4])]
                            if (match_3_4_sample_arr.count(True) < match_sample_okay_limit) or \
                                    (match_3_4_sample_arr.count(False) < match_sample_okay_limit):
                                sample_pairs_match_bool_dict[(col_name_3, col_name_4)] = False
                                continue
                            sample_pairs_match_bool_dict[(col_name_3, col_name_4)] = True

                        # With this test, it is not possible to determine a priori how many combinations it will
                        # check. The theoretical limit may be calculated, where all columns of the same type pair
                        # up well, but in practice, this vastly over-estimates the workload.
                        num_combinations_tested += 1
                        if num_combinations_tested > self.max_combinations:
                            if self.verbose >= 1:
                                print(f"  Skipping further testing pairs of pairs of columns. This test checked "
                                       f"{num_combinations_tested:,} combinations. "
                                       f"max_combinations is currently set to {self.max_combinations:,}.")
                            return

                        # Rows with a missing value in any of the four columns neither support nor violate the pattern
                        sample_is_na_arr = sample_is_na_dict[col_name_1].values | sample_is_na_dict[col_name_2].values | \
                            sample_is_na_dict[col_name_3].values | sample_is_na_dict[col_name_4].values
                        sample_series = [x == y or z for x, y, z in
                                         zip(match_1_2_sample_arr, match_3_4_sample_arr, sample_is_na_arr)]
                        if sample_series.count(False) > 1:
                            continue

                        if (col_name_3, col_name_4) in pairs_match_arr_dict:
                            match_3_4_arr = pairs_match_arr_dict[(col_name_3, col_name_4)]
                        else:
                            match_3_4_arr = [(x == y) or (w and z) for x, y, w, z in
                                             zip(self.orig_df[col_name_3],
                                                 self.orig_df[col_name_4],
                                                 is_na_dict[col_name_3],
                                                 is_na_dict[col_name_4])]
                            pairs_match_arr_dict[(col_name_3, col_name_4)] = match_3_4_arr
                        if (match_3_4_arr.count(True) < match_okay_limit) or \
                                (match_3_4_arr.count(False) < match_okay_limit):
                            continue

                        is_na_arr = is_na_dict[col_name_1].values | is_na_dict[col_name_2].values | \
                            is_na_dict[col_name_3].values | is_na_dict[col_name_4].values
                        test_series = [x == y or z for x, y, z in zip(match_1_2_arr, match_3_4_arr, is_na_arr)]
                        self._process_analysis_binary(
                            test_id,
                            [col_name_1, col_name_2, col_name_3, col_name_4],
                            test_series,
                            (f'Columns "{col_name_1}" and "{col_name_2}" match {match_1_2_arr.count(True)} times and '
                             f'mismatch {match_1_2_arr.count(False)} times. '
                             f'Columns "{col_name_3}" and "{col_name_4}" match {match_3_4_arr.count(True)} times and '
                             f'mismatch {match_3_4_arr.count(False)} times. '
                             f'Columns "{col_name_1}" and "{col_name_2}" have equal values in the same rows as '
                             f'"{col_name_3}" and "{col_name_4}"'
                             ),
                            display_info={'match_1_2_arr': match_1_2_arr, 'match_3_4_arr': match_3_4_arr}
                        )

    ##################################################################################################################
    # Data consistency checks for sets of columns of any type
    ##################################################################################################################

    def _generate_unique_sets_values(self):
        """
        Patterns without exceptions:
        Patterns with exception: 'unique_sets most_a' and 'unique_sets most_b' have 10 and 100 unique values
            respectively.
        """
        data_a = []
        for i in range(10):
            data_a.append([i]*100)
        data_a = np.array(data_a).reshape(1, -1)[0]

        data_b = []
        for _ in range(10):
            data_b.append(list(range(100)))
        data_b = np.array(data_b).reshape(1, -1)[0]

        self._add_synthetic_column('unique_sets rand', data_a)
        self._add_synthetic_column('unique_sets all', data_b)
        self._add_synthetic_column('unique_sets most', data_b)
        self.synth_df.loc[999, 'unique_sets most'] = 98  # repeating a combination with 9 in column 'all_a'


    def _check_unique_sets_values(self, test_id):
        """
        """

        # Get the set of columns that have not too many unique values. This is not strictly necessary, but a set of
        # columns with a unique combination of values is less meaningful if one or more of the columns have many
        # unique values in themselves. As well, get and cache the number of unique values per column.
        cols = []
        num_unique_vals_dict = {}
        for col_name in self.orig_df.columns:
            if self.orig_df[col_name].nunique() <= (self.num_rows / 2):  # todo: makes sense???
                cols.append(col_name)
                num_unique_vals_dict[col_name] = self.orig_df[col_name].nunique()

        found_any = False
        printed_subset_size_msg = False
        for subset_size in range(len(cols), 1, -1):
            if found_any:
                break

            calc_size = math.comb(len(cols), subset_size)
            skip_subsets = calc_size > self.max_combinations
            if skip_subsets:
                if self.verbose >= 2 and not printed_subset_size_msg:
                    print(f"    Skipping subsets of size {subset_size} and smaller. There are {calc_size:,} subsets. "
                           f"max_combinations is currently set to {self.max_combinations:,}.")
                    printed_subset_size_msg = True
                continue

            subsets = list(combinations(cols, subset_size))
            if self.verbose >= 2 and len(cols) > 15:
                print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset in subsets:
                max_combinations = 1
                for c in subset:
                    max_combinations *= num_unique_vals_dict[c]

                # If there are too few combinations to make unique combinations impossible (the number of combinations
                # is less than the number of rows), do not test. Also do not test if it will be unremarkable if there
                # are all unique combinations. The threshold for this is arbitrary, but set to a small multiple of the
                # number of rows.
                if self.num_rows <= max_combinations <= (self.num_rows * 2):
                    # Rows with a missing value in any of the columns neither support nor violate the pattern
                    test_series = self.orig_df.duplicated(subset=subset) & self.orig_df[list(subset)].notna().all(axis=1)
                    num_dup = test_series.tolist().count(True)
                    if 0 < num_dup < self.freq_contamination_level:
                        self._process_analysis_binary(
                            test_id,
                            subset,
                            ~test_series,
                            f'The set {subset} consistently contain a unique combination of values',
                            ""
                        )
                        found_any = True
                        break

    ##################################################################################################################
    # Data consistency checks for complete rows of values
    ##################################################################################################################


    def _generate_missing_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 2 Null values (if only this test is tested),
            with the exception of 1 row
        """
        data_vals = [['a', 'b', None, None], ['a', None, None, 'd'], [None, 'b', None, 'd'], [None, None, 'c', 'd']]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('miss_per_row_0', data[0].values)
        self._add_synthetic_column('miss_per_row_1', data[1].values)
        self._add_synthetic_column('miss_per_row_2', data[2].values)
        self._add_synthetic_column('miss_per_row_3', data[3].values)
        self.synth_df.loc[999, 'miss_per_row_0'] = None
        self.synth_df.loc[999, 'miss_per_row_1'] = None
        self.synth_df.loc[999, 'miss_per_row_2'] = None
        self.synth_df.loc[999, 'miss_per_row_3'] = None


    def _check_missing_values_per_row(self, test_id):
        test_series = self.orig_df.isna().sum(axis=1)
        self._process_analysis_counts(
            test_id,
            list(self.orig_df.columns),
            test_series,
            "The dataset consistently has",
            "null values per row"
        )


    def _generate_zero_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 2 zero values (if only this test is tested),
            with the exception of 1 row
        """
        data_vals = [[1, 2, 0, 0],
                     [2, 0, 0, 1],
                     [0, 0, 2, 2],
                     [0, 0, 3, 3]]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('zero_per_row_0', data[0].values)
        self._add_synthetic_column('zero_per_row_1', data[1].values)
        self._add_synthetic_column('zero_per_row_2', data[2].values)
        self._add_synthetic_column('zero_per_row_3', data[3].values)
        self.synth_df.loc[999, 'zero_per_row_0'] = 0
        self.synth_df.loc[999, 'zero_per_row_1'] = 0
        self.synth_df.loc[999, 'zero_per_row_2'] = 0
        self.synth_df.loc[999, 'zero_per_row_3'] = 0


    def _check_zero_values_per_row(self, test_id):
        # Skip if the dataset does not have at least two numeric columns
        if len(self.numeric_cols) < 2:
            return

        num_zeros_arr = map_elements(self.orig_df, lambda x: x == 0).sum(axis=1)

        # Missing values may or may not be 0, so the most common count is taken from the rows without missing values,
        # and a row with missing values is an exception only if no values in their place could give it that count.
        num_missing_arr = self.orig_df.isna().sum(axis=1)
        if (num_missing_arr > 0).all():
            return

        # If there are consistently no 0 values per row, this is not an interesting pattern
        most_common_count = statistics.mode(num_zeros_arr[num_missing_arr == 0])
        if most_common_count == 0:
            return

        test_series = (num_zeros_arr <= most_common_count) & (num_zeros_arr + num_missing_arr >= most_common_count)
        self._process_analysis_binary(
            test_id,
            list(self.orig_df.columns),
            test_series,
            f"The dataset consistently has {most_common_count} elements with value 0 per row"
        )


    def _generate_unique_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 3 unique values (if only this test is
            tested), with the exception of 1 row
        """
        data_vals = [[1, 1, 3, 4],
                     [1, 6, 6, 4],
                     [0, 1, 0, 5],
                     [4, 0, 3, 4]]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('unique_per_row_0', data[0].values)
        self._add_synthetic_column('unique_per_row_1', data[1].values)
        self._add_synthetic_column('unique_per_row_2', data[2].values)
        self._add_synthetic_column('unique_per_row_3', data[3].values)
        self.synth_df.loc[999, 'unique_per_row_0'] = 1
        self.synth_df.loc[999, 'unique_per_row_1'] = 2
        self.synth_df.loc[999, 'unique_per_row_2'] = 3
        self.synth_df.loc[999, 'unique_per_row_3'] = 4


    def _check_unique_values_per_row(self, test_id):
        # Missing values are not counted. They may or may not repeat other values in their row, so the common counts
        # are taken from the rows without missing values, and a row with missing values is an exception only if no
        # values in their place could give it a common count.
        counts_per_row = self.orig_df.apply(lambda x: len({v for v in x if not pd.isna(v)}), axis=1)
        num_missing_arr = self.orig_df.isna().sum(axis=1)
        counts_series = counts_per_row[num_missing_arr == 0].value_counts(normalize=False)
        uncommon_counts = [x for x, y in zip(counts_series.index, counts_series.values) if y < self.freq_contamination_level]
        common_counts = sorted([x for x in counts_series.index if x not in uncommon_counts])
        # There may be too few rows without missing values for any count to be common
        if len(common_counts) == 0:
            return
        min_common_counts = min(common_counts)
        max_common_counts = max(common_counts)

        # It is not interesting if the rows always have all unique values
        if min_common_counts == len(self.orig_df.columns):
            return

        # It is not interesting if the number of unique values varies greatly from row to row
        if (max_common_counts / min_common_counts) > 1.1:
            return

        if min_common_counts == max_common_counts:
            test_series = (counts_per_row <= max_common_counts) & \
                          (counts_per_row + num_missing_arr >= max_common_counts)
            common_str = str(min_common_counts)
            exceptions_str = "Flagging rows with other counts of unique values"
        else:
            lower_limit = min_common_counts / 2
            upper_limit = max_common_counts * 2
            test_series = (counts_per_row + num_missing_arr >= lower_limit) & (counts_per_row <= upper_limit)
            if lower_limit > 0:
                exceptions_str = (f"Flagging rows with other counts less than {lower_limit} or greater than "
                                   f"{upper_limit} unique values")
            else:
                exceptions_str = f"Flagging rows with other counts greater than {upper_limit} unique values"
            common_str = str(min_common_counts) + " to " + str(max_common_counts)

        self._process_analysis_binary(
            test_id,
            list(self.orig_df.columns),
            test_series,
            f"The dataset consistently has {common_str} unique values per row",
            exceptions_str,
            display_info={'min_common_counts':min_common_counts, 'max_common_counts': max_common_counts}
        )


    def _generate_negative_values_per_row(self):
        """
        Patterns without exceptions: None. As this operates on all columns, it is not possible to have examples of
            both patterns and exceptions
        Patterns with exception:  The full set of columns consistently have 2 negative values (if only this test is
            tested), with the exception of 1 row
        """
        data_vals = [[-1, -2, 0, 0], [-1, 0, 0, -4], [1, -2, 0, -4], [0, 2, -3, -4]]
        data = pd.DataFrame([random.choice(data_vals) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('neg_per_row_0', data[0].values)
        self._add_synthetic_column('neg_per_row_1', data[1].values)
        self._add_synthetic_column('neg_per_row_2', data[2].values)
        self._add_synthetic_column('neg_per_row_3', data[3].values)
        self.synth_df.loc[999, 'neg_per_row_0'] = 0
        self.synth_df.loc[999, 'neg_per_row_1'] = 0
        self.synth_df.loc[999, 'neg_per_row_2'] = 0
        self.synth_df.loc[999, 'neg_per_row_3'] = 0


    def _check_negative_values_per_row(self, test_id):

        # Skip if the dataset does not have at least two numeric columns
        if len(self.numeric_cols) < 2:
            return

        test_series = map_elements(self.orig_df, lambda x: isinstance(x, numbers.Number) and x < 0).sum(axis=1)
        # Missing values may or may not be negative, so rows with missing values do not have a count
        test_series = test_series.where(self.orig_df.notna().all(axis=1))
        self._process_analysis_counts(
            test_id,
            list(self.orig_df.columns),
            test_series,
            "The dataset consistently has",
            "negative values per row"
        )


    def _generate_small_avg_rank_per_row(self):
        """
        Patterns without exceptions: None. This test does not flag patterns.
        Patterns with exception: Row 999, over all numeric columns, has values with, on average, low percentiles.
        """
        for i in range(10):
            self._add_synthetic_column(f'small_avg_rank_rand_{i}',
                                        [random.random() for _ in range(self.num_synth_rows - 1)] + [0.000001])


    def _check_small_avg_rank_per_row(self, test_id):
        # Skip if the dataset does not have at least two numeric columns
        if len(self.numeric_cols) < 2:
            return

        rand_df = pd.DataFrame()
        for col_name in self.numeric_cols:
            rand_df = pd.concat([
                rand_df,
                pd.DataFrame({col_name: self.orig_df[col_name].rank(pct=True)})
            ], axis=1)
        rand_df['Avg Percentile'] = rand_df.mean(axis=1)

        d1 = rand_df['Avg Percentile'].quantile(0.1)
        d9 = rand_df['Avg Percentile'].quantile(0.9)
        idr = abs(d9 - d1)
        lower_limit = d1 - (self.idr_limit * idr)
        # Rows with all numeric values missing have no average percentile, so neither support nor violate the pattern
        test_series = (rand_df['Avg Percentile'] >= lower_limit) | rand_df['Avg Percentile'].isna()
        flagged_vals = [x for x in rand_df['Avg Percentile'] if x < lower_limit]
        self._process_analysis_binary(
            test_id,
            self.numeric_cols,
            test_series,
            ("This test considered all numeric columns, and calculated the percentile of each value, relative to its "
             'column. The average percentile was then calculated per row. '
             "The flagged rows contain average percentiles that are unusually small, suggesting consistently low "
             "values across all or most numeric columns. This flags any rows with an average percentile of "
             f"{lower_limit:.3f} or lower"),
            allow_patterns=False,
            display_info={"percentiles": rand_df['Avg Percentile'],
                          "flagged_vals": flagged_vals}
        )


    def _generate_large_avg_rank_per_row(self):
        """
        Patterns without exceptions: None. This test does not flag patterns.
        Patterns with exception: Row 999 over all numeric columns has values with, on average, low percentiles.
        """
        for i in range(10):
            self._add_synthetic_column(
                f'large_avg_rank_rand_{i}', [random.random() for _ in range(self.num_synth_rows -1)] + [2.0])


    def _check_large_avg_rank_per_row(self, test_id):
        # Skip if the dataset does not have at least two numeric columns
        if len(self.numeric_cols) < 2:
            return

        rank_df = pd.DataFrame()
        for col_name in self.numeric_cols:
            rank_df = pd.concat([
                rank_df,
                pd.DataFrame({col_name: self.orig_df[col_name].rank(pct=True)})
            ], axis=1)
        rank_df['Avg Percentile'] = rank_df.mean(axis=1)

        q1 = rank_df['Avg Percentile'].quantile(0.25)
        q3 = rank_df['Avg Percentile'].quantile(0.75)
        iqr = abs(q3 - q1)
        # As the value is limited to 1.0, we do not use self.iqr_limit here, but instead the standard coefficient of
        # 2.2 for testing for outliers.
        upper_limit = q3 + (2.2 * iqr)
        # Rows with all numeric values missing have no average percentile, so neither support nor violate the pattern
        test_series = (rank_df['Avg Percentile'] <= upper_limit) | rank_df['Avg Percentile'].isna()
        flagged_vals = [x for x in rank_df['Avg Percentile'] if x > upper_limit]
        self._process_analysis_binary(
            test_id,
            self.numeric_cols,
            test_series,
            ("This test considered all numeric columns, and calculated the percentile of each value, relative to its "
             'column. The average percentile was then calculated per row. '
             "The flagged rows contain average percentiles that are unusually high, suggesting consistently high "
             "values across all or most numeric columns. This flags any rows with an average percentile of "
             f"{upper_limit:.3f} or higher"),
            allow_patterns=False,
            display_info={"percentiles": rank_df['Avg Percentile'],
                          "flagged_vals": flagged_vals}
        )


