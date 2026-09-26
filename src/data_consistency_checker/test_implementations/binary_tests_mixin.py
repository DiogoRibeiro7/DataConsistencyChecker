"""
BinaryTestsMixin: Test methods for binary tests.

This mixin contains 24 test methods organized by category.
Extracted from check_data_consistency.py for better code organization.
"""

from __future__ import annotations

import datetime
import math
import random
import statistics
from itertools import combinations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score

from data_consistency_checker.checker_state import CheckerState
from data_consistency_checker.checker_utils import as_str, is_missing


class BinaryTestsMixin(CheckerState):
    """
    Mixin class containing binary tests methods.

    This class contains 24 methods for testing data consistency
    related to binary operations.

    This is a mixin class designed to be used with multiple inheritance.
    It does not have an __init__ method and relies on the parent class
    to provide necessary attributes and methods.
    """

    def _generate_binary_same(self):
        """
        Patterns without exceptions: 'bin sim all_1' and 'bin sim all_2' consistently have the same binary values
        Patterns with exception: 'bin sim most' compared to both 'bin sim all_1' and 'bin sim all_2', has the same
            values, but with 1 exception.

        We use values 7 & 9 to test where the binary values are not 0 and 1.
        """
        self._add_synthetic_column('bin sim all_1', [7]*(self.num_synth_rows-500) + [9]*500)
        self._add_synthetic_column('bin sim all_2', [7]*(self.num_synth_rows-500) + [9]*500)
        self._add_synthetic_column('bin sim most', [7]*(self.num_synth_rows-501) + [9]*501)


    def _check_binary_same(self, test_id):
        # todo: all binary tests: do on all binary & numeric columns, but first convert the numeric to binary as
        #   being 0 or non-0.

        # If the dataset is large, test on a 2nd sample
        sample_1000_df = None
        if self.num_rows > 10_000:
            sample_1000_df = self.orig_df[self.binary_cols].sample(n=1000, random_state=0)

        num_pairs, pairs = self._get_binary_column_pairs_unique(same_vocabulary=True)
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of binary columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 500 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(pairs):,} pairs of binary columns")

            # Test on a sample first
            test_series = np.array([x == y for x, y in zip(self.sample_df[col_name_1], self.sample_df[col_name_2])])
            test_series = test_series | self.sample_df[col_name_1].isna() | self.sample_df[col_name_2].isna()
            if test_series.tolist().count(False) > 1:
                continue

            # Test on a larger sample
            if sample_1000_df is not None:
                test_series = np.array([x == y for x, y in zip(sample_1000_df[col_name_1], sample_1000_df[col_name_2])])
                test_series = test_series | sample_1000_df[col_name_1].isna() | sample_1000_df[col_name_2].isna()
                if test_series.tolist().count(False) > 10:
                    continue

                # Consider imbalanced arrays. We also check the macro f1 score
                sub_df = sample_1000_df[[col_name_1, col_name_2]].dropna()
                f1score = f1_score(as_str(sub_df[col_name_1]), as_str(sub_df[col_name_2]), average='macro')
                if f1score < 0.75:
                    continue

            # Test on the full columns
            test_series = np.array([x == y for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])])
            test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
            if test_series.tolist().count(False) > self.freq_contamination_level:
                continue

            # Consider imbalanced arrays. We also check the macro f1 score
            sub_df = self.orig_df[[col_name_1, col_name_2]].dropna()
            f1score = f1_score(as_str(sub_df[col_name_1]), as_str(sub_df[col_name_2]), average='macro')
            if f1score < 0.9:
                continue

            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                "The columns consistently have the same value",
                "")


    def _generate_binary_opposite(self):
        """
        Patterns without exceptions: 'bin opp all_1' and 'bin opp all_2' consistently contain the opposite values.
        Patterns with exception: 'bin opp most' consistently has the opposite values as 'bin opp all_2', with one
            exception.
        """
        self._add_synthetic_column('bin opp all_1', [0]*(self.num_synth_rows-500) + [1]*500)
        self._add_synthetic_column('bin opp all_2', [1]*(self.num_synth_rows-500) + [0]*500)
        self._add_synthetic_column('bin opp most',  [0]*(self.num_synth_rows-501) + [1]*501)


    def _check_binary_opposite(self, test_id):
        num_pairs, column_pairs = self._get_binary_column_pairs_unique(same_vocabulary=True)
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of binary columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(column_pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 100 == 0:
                print(f"  Examining pair of binary columns: {pair_idx} of {len(column_pairs)}")

            if self.orig_df[col_name_1].isna().sum() > (self.num_rows / 2):
                continue
            if self.orig_df[col_name_2].isna().sum() > (self.num_rows / 2):
                continue

            test_series = np.array([x != y for x, y in zip(self.orig_df[col_name_1], self.orig_df[col_name_2])])
            if test_series.tolist().count(False) > self.freq_contamination_level:
                continue

            # Consider imbalanced arrays. We also check the macro f1 score
            v1, v2 = self.orig_df[col_name_2].dropna().unique()
            opp_arr = self.orig_df[col_name_2].map({v1: v2, v2: v1})
            opp_arr = opp_arr.fillna("NONE")
            # todo: flip col2 properly, using map, not ~ -- it may not be 0 & 1
            f1score = f1_score(as_str(self.orig_df[col_name_1].fillna("NONE")), as_str(opp_arr), average='macro')
            if f1score < 0.9:
                continue

            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                "The columns consistently have the opposite value", "")


    def _generate_binary_implies(self):
        """
        Patterns without exceptions: 'bin implies all_1' consistently implies 'bin implies all_2'. A value of 0 in
            'bin implies all_1' consistently implies a 1 in 'bin implies all_2'.
        Patterns with exception: 'bin implies all_1' consistently implies 'bin implies most', with 1 exception. A
            value of 1 in 'bin implies all_1' implies 'bin implies most' will have a value of 0, but does not in
            the last row.
        """
        self._add_synthetic_column('bin implies all_1', [0]*(self.num_synth_rows-500) + [1]*500)
        self._add_synthetic_column('bin implies all_2', [1]*(self.num_synth_rows-500) + [0]*500)
        self._add_synthetic_column('bin implies most', [0]*(self.num_synth_rows-800) + [1]*799 + [0])


    def _check_binary_implies(self, test_id):
        """
        This essentially checks for rare combinations, and does not work well with columns with many Null values.
        """

        num_pairs, column_pairs = self._get_binary_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of binary columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for pair_idx, (col_name_1, col_name_2) in enumerate(column_pairs):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 500 == 0:
                print(f"  Examining pair: {pair_idx:,} of {len(column_pairs):,}  pairs of binary columns")

            if self.orig_df[col_name_1].apply(is_missing).sum() > self.freq_contamination_level:
                continue
            if self.orig_df[col_name_2].apply(is_missing).sum() > self.freq_contamination_level:
                continue

            vals_1 = self.column_unique_vals[col_name_1]
            val_1a = vals_1[0]
            val_1b = vals_1[1]
            vals_2 = self.column_unique_vals[col_name_2]
            val_2a = vals_2[0]
            val_2b = vals_2[1]

            # Ensure there are at least 10% of the rows with both values in both columns.
            count_min = self.num_rows * 0.1

            mask_1a = (self.orig_df[col_name_1] == val_1a) | self.orig_df[col_name_1].isna()
            mask_1b = (self.orig_df[col_name_1] == val_1b) | self.orig_df[col_name_1].isna()
            mask_2a = (self.orig_df[col_name_2] == val_2a) | self.orig_df[col_name_2].isna()
            mask_2b = (self.orig_df[col_name_2] == val_2b) | self.orig_df[col_name_2].isna()

            count_1a = mask_1a.tolist().count(True)
            if count_1a < count_min:
                continue
            count_1b = mask_1b.tolist().count(True)
            if count_1b < count_min:
                continue
            count_2a = mask_2a.tolist().count(True)
            if count_2a < count_min:
                continue
            count_2b = mask_2b.tolist().count(True)
            if count_2b < count_min:
                continue

            mask_aa = mask_1a & mask_2a
            mask_ab = mask_1a & mask_2b
            mask_ba = mask_1b & mask_2a
            mask_bb = mask_1b & mask_2b

            count_aa = mask_aa.tolist().count(True)
            count_ab = mask_ab.tolist().count(True)
            count_ba = mask_ba.tolist().count(True)
            count_bb = mask_bb.tolist().count(True)

            test_series = None
            pattern_str = ""
            if count_aa < self.freq_contamination_level:
                # The expected count is based on the fraction of value a in column 1 and value b in column two,
                # multiplied to the give the expected fraction of both, times the number of rows to give the expected
                # number of rows. This is then: (count_1a / self_num_rows) * (count_1a / self_num_rows) * self.num_rows
                # which is simplified below for efficiency
                expected_count = (count_1a / self.num_rows) * (count_2a)
                if count_aa < expected_count:
                    test_series = np.array(~mask_aa)
                    pattern_str = f'"{col_name_1}" value: {val_1a} consistently implies "{col_name_2}" value: {val_2b}'
            elif count_ab < self.freq_contamination_level:
                expected_count = (count_1a / self.num_rows) * (count_2b)
                if count_ab < expected_count:
                    test_series = np.array(~mask_ab)
                    pattern_str = f'"{col_name_1}" value: {val_1a} consistently implies "{col_name_2}" value: {val_2a}'
            elif count_ba < self.freq_contamination_level:
                expected_count = (count_1a / self.num_rows) * (count_2b)
                if count_ba < expected_count:
                    test_series = np.array(~mask_ba)
                    pattern_str = f'"{col_name_1}" value: {val_1b} consistently implies "{col_name_2}" value: {val_2b}'
            elif count_bb < self.freq_contamination_level:
                expected_count = (count_1a / self.num_rows) * (count_2b)
                if count_bb < expected_count:
                    test_series = np.array(~mask_bb)
                    pattern_str = f'"{col_name_1}" value: "{val_1b}" consistently implies "{col_name_2}" value: {val_2a}'

            if test_series is not None:
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    pattern_str,
                    "")

    ##################################################################################################################
    # Data consistency checks for sets of binary columns
    ##################################################################################################################

    def _generate_binary_and(self):
        """
        Patterns without exceptions: 'bin_and all' is consistently the AND of 'bin_and rand_1' and 'bin_and rand_2'
        Patterns with exception: 'bin_and most' is consistently the AND of 'bin_and rand_1' and 'bin_and rand_2' with
            1 exception.
        """
        self._add_synthetic_column('bin_and rand_1', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_and rand_2', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_and all', self.synth_df['bin_and rand_1'] & self.synth_df['bin_and rand_2'])
        self._add_synthetic_column('bin_and most', self.synth_df['bin_and rand_1'] & self.synth_df['bin_and rand_2'])
        self.synth_df.loc[999, 'bin_and most'] = (self.synth_df.loc[999, 'bin_and most'] + 1) % 2


    def _check_binary_and(self, test_id):
        """
        Handling null values: patterns are not considered violated if any cells contain null values.
        """

        def test_val(col_name, col_vals):
            def get_and(x):
                and_v = 1
                for v in x:
                    and_v = and_v & (is_missing(v) | v)
                return and_v

            # Determine which other columns may potentially be AND'd to match col_name.
            # Ensure all columns considered have the same two values and at least 10% of both values.
            # Also check they each have enough of the 2nd value (treated as equivalent to 1 in the binary sense)
            # to form an AND relationship with col_name.
            set_other_cols = []
            count_min = self.num_rows * 0.1
            val0, val1 = self.column_unique_vals[col_name]
            target_count_1 = self.orig_df[col_name].tolist().count(val1)

            for col_name_other in self.binary_cols:
                if col_name_other == col_name:
                    continue
                if val0 == val0:
                    num_val0 = self.orig_df[col_name_other].dropna().tolist().count(val0)
                else:
                    num_val0 = self.orig_df[col_name_other].isna().sum()
                if val1 == val1:
                    num_val1 = self.orig_df[col_name_other].dropna().tolist().count(val1)
                else:
                    num_val1 = self.orig_df[col_name_other].isna().sum()
                if num_val0 < count_min:
                    continue
                if num_val1 < count_min:
                    continue
                # If this column has less 1's then the target column, then no amount of ANDing with other columns
                # will allow it to match the target column
                if num_val1 < target_count_1:
                    continue
                num_ones_not_matching = len([1 for x, y in zip(self.orig_df[col_name], self.orig_df[col_name_other])
                                             if (((x == val1) and (y != val1)) and (not is_missing(x)) and (not is_missing(y)))])
                if num_ones_not_matching > self.freq_contamination_level:
                    continue
                set_other_cols.append(col_name_other)
            if len(set_other_cols) < 2:
                return

            # todo: (for binary_or as well) skip subset sizes that would produce too many combinations. We can just calculate (n chose m)
            # todo: we can determine for each subset_size n, if there are no viable subsets, then there is no
            # use checking any smaller subsets. This occurs if, for each subset, the result of the AND operation
            # is such that smaller subsets could never do better. Need to code this to be actually faster than just
            # looping through.
            for subset_size in range(min(5, len(set_other_cols)), 1, -1):
                subset_list = list(combinations(set_other_cols, subset_size))
                if len(subset_list) > 5000:
                    continue
                for subset_idx, subset in enumerate(subset_list):
                    if self.verbose >= 2 and subset_idx > 0 and subset_idx % 1000 == 0:
                        print(f"    Examining subset: {subset_idx:,} of {len(subset_list):,} other binary columns")

                    # The AND can work based on either value if the column does not contain 0 and 1.
                    subset_df = self.orig_df[list(subset)].copy()
                    for c in subset_df.columns:
                        subset_df[c] = subset_df[c].fillna(-1).map({col_vals[0]: 0, col_vals[1]: 1}).fillna(-1).astype(int)

                    # Test first on a sample
                    sample_df = subset_df.head(10)
                    and_of_cols = sample_df[list(subset)].apply(get_and, axis=1)
                    test_series = np.array(self.orig_df.head(10)[col_name] == and_of_cols)
                    num_matching = test_series.tolist().count(True)
                    if num_matching < 9:
                        continue

                    # Test on the full columns
                    and_of_cols = subset_df.apply(get_and, axis=1)
                    test_series = np.array(self.orig_df[col_name] == and_of_cols)
                    test_series = self.check_results_for_null(test_series, col_name, subset)
                    num_matching = test_series.tolist().count(True)
                    if num_matching > (self.num_rows - self.freq_contamination_level):
                        self._process_analysis_binary(
                            test_id,
                            list(subset) + [col_name],
                            test_series,
                            f'"{col_name}" is consistently the result of an AND operation over the columns {subset_list}',
                            ""
                        )
                        return

        for col_idx, col_name in enumerate(self.binary_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column: {col_idx:,} of {len(self.binary_cols):,} binary columns")
            col_vals = self.column_unique_vals[col_name]
            if self.orig_df[col_name].tolist().count(col_vals[0]) < (self.num_rows * 0.1) or \
                    self.orig_df[col_name].tolist().count(col_vals[1]) < (self.num_rows * 0.1):
                continue
            test_val(col_name, col_vals)


    def _generate_binary_or(self):
        """
        Patterns without exceptions: 'bin_or all' is consistently the OR of "bin_or rand_1", "bin_or rand_2" AND
            "bin_or most"
        Patterns with exception: "bin_or most" is consistently the OR of "bin_or rand_1" and "bin_or rand_2" with
            1 exception.
        """
        self._add_synthetic_column('bin_or rand_1', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_or rand_2', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_or all', self.synth_df['bin_or rand_1'] | self.synth_df['bin_or rand_2'])
        self._add_synthetic_column('bin_or most', self.synth_df['bin_or rand_1'] | self.synth_df['bin_or rand_2'])
        self.synth_df.loc[999, 'bin_or most'] = (self.synth_df.loc[999, 'bin_or most'] + 1) % 2


    def _check_binary_or(self, test_id):
        def test_val(col_name, col_vals):
            def get_or(x):
                or_v = 0
                for v in x:
                    or_v = or_v | (is_missing(v) | v)
                return or_v

            # Determine which other columns may potentially be OR'd to match col_name.
            # Ensure all columns considered have at least 10% of both values. This also checks the 2 columns have the
            # same vocabulary.
            # Also check they each have few enough of the 2nd value (treated as equivalent to 1 in the binary sense)
            # to form an OR relationship with col_name.
            set_other_cols = []
            count_min = self.num_rows * 0.1
            val0, val1 = self.column_unique_vals[col_name]
            target_count_1 = self.orig_df[col_name].tolist().count(val1)
            for col_name_other in self.binary_cols:
                if col_name_other == col_name:
                    continue
                if val0 == val0:
                    num_val0 = self.orig_df[col_name_other].dropna().tolist().count(val0)
                else:
                    num_val0 = self.orig_df[col_name_other].isna().sum()
                if val1 == val1:
                    num_val1 = self.orig_df[col_name_other].dropna().tolist().count(val1)
                else:
                    num_val1 = self.orig_df[col_name_other].isna().sum()
                if num_val0 < count_min:
                    continue
                if num_val1 < count_min:
                    continue
                # If this column has more 1's then the target column, then no amount of ORing with other columns
                # will allow it to match the target column
                if num_val1 > target_count_1:
                    continue
                num_ones_not_matching = len([1 for x, y in zip(self.orig_df[col_name], self.orig_df[col_name_other])
                                             if ((x == val0) and (y == val1) and (not is_missing(x)) and (not is_missing(y)))])
                if num_ones_not_matching > self.freq_contamination_level:
                    continue
                set_other_cols.append(col_name_other)
            if len(set_other_cols) < 2:
                return None

            # Consider subsets of size 5 at maximum.
            for subset_size in range(min(5, len(set_other_cols)), 1, -1):
                subset_list = list(combinations(set_other_cols, subset_size))
                if len(subset_list) > 5000: # As this test can be slow, we do not extend it to the full max_combinations
                    continue
                for subset_idx, subset in enumerate(subset_list):
                    if self.verbose >= 2 and subset_idx > 0 and subset_idx % 1000 == 0:
                        print(f"    Examining subset: {subset_idx:,} of {len(subset_list):,} other binary columns")

                    # The OR can work based on either value if the column does not contain 0 and 1.
                    subset_df = self.orig_df[list(subset)].copy()
                    for c in list(subset_df.columns):
                        subset_df[c] = subset_df[c].fillna(-1).map({col_vals[0]: 0, col_vals[1]: 1}).fillna(-1).astype(int)

                    # Test first on a sample
                    sample_df = subset_df.head(10)
                    or_of_cols = sample_df[list(subset)].apply(get_or, axis=1)
                    test_series = np.array(self.orig_df.head(10)[col_name] == or_of_cols)
                    num_matching = test_series.tolist().count(True)
                    if num_matching < 9:
                        continue

                    # Test on the full columns
                    or_of_cols = subset_df[list(subset)].apply(get_or, axis=1)
                    test_series = np.array(self.orig_df[col_name] == or_of_cols)
                    test_series = self.check_results_for_null(test_series, col_name, subset)
                    num_matching = test_series.tolist().count(True)
                    if num_matching > (self.num_rows - self.freq_contamination_level):
                        self._process_analysis_binary(
                            test_id,
                            list(subset) + [col_name],
                            test_series,
                            f'"{col_name}" is consistently the result of an OR operation over the columns {subset_list}',
                            ""
                        )
                        return True
            return None

        for col_idx, col_name in enumerate(self.binary_cols):
            if self.verbose >= 2 and col_idx > 0 and col_idx % 10 == 0:
                print(f"  Examining column: {col_idx} of {len(self.binary_cols)} binary columns")
            col_vals = self.column_unique_vals[col_name]
            if self.orig_df[col_name].tolist().count(col_vals[0]) < (self.num_rows * 0.1) or \
                    self.orig_df[col_name].tolist().count(col_vals[1]) < (self.num_rows * 0.1):
                continue
            test_val(col_name, col_vals)


    def _generate_binary_xor(self):
        """
        Patterns without exceptions: 'bin_xor all' is consistently the XOR of 'bin_xor rand_1 and 'bin_xor rand_2'
        Patterns with exception: 'bin_xor most' is consistently the XOR of 'bin_xor rand_1 and 'bin_xor rand_2', with
            1 exception.
        """
        self._add_synthetic_column('bin_xor rand_1', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_xor rand_2', [random.choice([0, 1]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_xor all', self.synth_df['bin_xor rand_1'] ^ self.synth_df['bin_xor rand_2'])
        self._add_synthetic_column('bin_xor most', self.synth_df['bin_xor rand_1'] ^ self.synth_df['bin_xor rand_2'])
        self.synth_df.loc[999, 'bin_xor most'] = (self.synth_df.loc[999, 'bin_xor most'] + 1) % 2


    def _check_binary_xor(self, test_id):
        """
        This defines XOR for multiple inputs as "1" if exactly 1 input has value 1. However, as the binary columns
        may contain any pair of values, not necessarily "0" and "1", this test is repeated for both values.
        """
        # Determine if there are too many combinations to execute
        num_bin_cols = len(self.binary_cols)
        total_combinations = num_bin_cols * (num_bin_cols * (num_bin_cols - 1)) / 2
        if total_combinations > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test {test_id}. There are {num_bin_cols:,} binary columns, resulting in "
                       f"{int(total_combinations):,} combinations. max_combinations is currently set to "
                       f"{self.max_combinations:,}")
            return

        reported_dict = {}
        for col_idx_1, col_name_1 in enumerate(self.binary_cols):
            if self.verbose >= 2 and col_idx_1 > 0 and col_idx_1 % 10 == 0:
                print(f"  Examining column {col_idx_1} of {len(self.binary_cols)} binary columns.")

            min_rows_per_value = self.num_rows * 0.1

            col_vals_1 = self.column_unique_vals[col_name_1]
            if self.orig_df[col_name_1].tolist().count(col_vals_1[0]) < min_rows_per_value or \
                    self.orig_df[col_name_1].tolist().count(col_vals_1[1]) < min_rows_per_value:
                continue

            num_pairs, pairs = self._get_binary_column_pairs_unique()
            for (col_name_2, col_name_3) in pairs:
                columns_tuple = tuple(sorted([col_name_1, col_name_2, col_name_3]))
                if columns_tuple in reported_dict:  # todo: maybe faster to remove
                    continue

                if col_name_1 in (col_name_2, col_name_3):
                    continue

                col_vals_2 = self.column_unique_vals[col_name_2]
                if self.orig_df[col_name_2].tolist().count(col_vals_2[0]) < (self.num_rows * 0.1) or \
                        self.orig_df[col_name_2].tolist().count(col_vals_2[1]) < (self.num_rows * 0.1):
                    continue

                col_vals_3 = self.column_unique_vals[col_name_3]
                if self.orig_df[col_name_3].tolist().count(col_vals_3[0]) < (self.num_rows * 0.1) or \
                        self.orig_df[col_name_3].tolist().count(col_vals_1[1]) < (self.num_rows * 0.1):
                    continue

                # Test on sample first
                # todo: fill in

                # Test on the full columns
                vals_col_2 = self.orig_df[col_name_2].fillna(-1).map({col_vals_2[0]: 0, col_vals_2[1]: 1}).fillna(-1).astype(int)
                vals_col_3 = self.orig_df[col_name_3].fillna(-1).map({col_vals_3[0]: 0, col_vals_3[1]: 1}).fillna(-1).astype(int)
                test_series = np.array(self.orig_df[col_name_1] == (vals_col_2 ^ vals_col_3))
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna() | self.orig_df[col_name_3].isna()
                num_matching = test_series.tolist().count(True)
                if num_matching >= (self.num_rows - self.freq_contamination_level):
                    self._process_analysis_binary(
                        test_id,
                        [col_name_2, col_name_3, col_name_1],
                        test_series,
                        (f'"{col_name_1}" is consistently the result of an XOR operation with columns "{col_name_2}" '
                         f'and "{col_name_3}"'),
                        "")
                    reported_dict[columns_tuple] = True


    def _generate_binary_num_same(self):
        """
        Patterns without exceptions: None: the first 4 columns have consistently 2 "2" values and 2 "1" values. However,
            these will not be flagged, as they are part of a larger pattern with all 5 columns
        Patterns with exception: the full set of 5 columns have consistently 3 "1" values, other than the last row
        """
        # We have a set of 4 binary columns where there are always 2 set to 1
        vals = [[2, 2, 1, 1], [2, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1, 2], [1, 2, 2, 1]]
        data = np.array([vals[np.random.choice(range(len(vals)))] for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_num_same rand_a', data[:, 0])
        self._add_synthetic_column('bin_num_same rand_b', data[:, 1])
        self._add_synthetic_column('bin_num_same rand_c', data[:, 2])
        self._add_synthetic_column('bin_num_same rand_d', data[:, 3])
        self._add_synthetic_column('bin_num_same most', [1] * (self.num_synth_rows - 1) + [2])


    def _check_binary_num_same(self, test_id):
        """
        Identify sets of binary columns that have the same two values. For each such set, determine if there is a
        consistent number of these columns that has the same value. For example, if a set of 10 columns contains
        values 'y' and 'n', a pattern would exist if there are consistently, say, 7 columns with 'y' and 3 with 'n',
        though the specific columns with each value may vary from row to row.
        """

        # Each execution of the loop processes the sets of binary columns for a given pair of values (eg 0 and 1,
        # Y and N, etc). We first identify one binary column that has not been examined, and then find the other
        # binary columns with the same vocabulary.

        vals_processed = {}
        for col_idx, col_name in enumerate(self.binary_cols):
            col_vals_tuple = tuple(self.column_unique_vals[col_name])
            col_vals_set = set(col_vals_tuple)
            if col_vals_tuple in vals_processed:
                continue
            vals_processed[col_vals_tuple] = True
            if self.verbose >= 2:
                print(f"  Examining column: {col_idx} of {len(self.binary_cols)} binary columns) "
                      f"(and all other columns with values {col_vals_tuple}).")
            similar_cols = [col_name]
            for c_idx in range(col_idx + 1, len(self.binary_cols)):
                c_name = self.binary_cols[c_idx]
                c_vals = set(self.column_unique_vals[c_name])
                if len(col_vals_set.intersection(c_vals)) != 2:
                    continue
                similar_cols.append(c_name)
            if self.verbose >= 2 and len(similar_cols) > 1:
                print(f"    There are {len(similar_cols)} binary columns with this pair of values")

            # Create a dataframe where all the values are mapped to 0 and 1. This allows us to perform sum() operations
            # by row on subsets of this (taking subsets of columns below).
            one_zero_df = self.orig_df[list(similar_cols)].copy()
            for c in one_zero_df.columns:
                one_zero_df[c] = one_zero_df[c].map({col_vals_tuple[0]: 0, col_vals_tuple[1]: 1}).fillna(0).astype(int)
            sample_one_zero_df = one_zero_df.sample(n=min(self.num_rows, 50), random_state=0)
            sample_one_zero_np = sample_one_zero_df.values

            found_any = False
            # Check subsets of at least size 3. With 2 columns, this is equivalent to the BINARY_OPPOSITE test
            printed_subset_size_msg = False
            for subset_size in range(len(similar_cols), 2, -1):
                if found_any:
                    break
                calc_size = math.comb(len(similar_cols), subset_size)
                skip_subsets = calc_size > self.max_combinations
                if skip_subsets:
                    if self.verbose >= 2 and not printed_subset_size_msg:
                        print(f"    Skipping subsets of size {subset_size}. There are {calc_size:,} subsets. "
                               f"max_combinations is currently set to {self.max_combinations:,}.")
                        printed_subset_size_msg = True
                    continue

                subsets = list(combinations(list(range(len(similar_cols))), subset_size))
                if self.verbose >= 3:
                    print(f"    Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")

                for subset in subsets:
                    # Test first on a sample of rows for the subset of columns
                    sample_subset_np = sample_one_zero_np[:, list(subset)]
                    sums = sample_subset_np.sum(axis=1)
                    count_most_common_sum = sums.tolist().count(statistics.mode(sums))
                    # Allow only 1 deviation for a small sample
                    if count_most_common_sum < (len(sample_one_zero_df) - 1):
                        continue

                    # If count_most_common_sum is 0, this is really a case of identifying rare values, which is another
                    # test
                    if statistics.mode(sums) == 0:
                        continue

                    # If the sample seems to match, test on the full subset of columns
                    col_names = np.array(similar_cols)[list(subset)]
                    subset_df = one_zero_df[col_names].copy()
                    sums = subset_df.sum(axis=1)
                    vc = sums.value_counts(normalize=False).sort_values(ascending=False)
                    if vc.iloc[0] > (self.num_rows - self.freq_contamination_level):
                        test_series = np.array([sums == vc.index[0]][0])
                        self._process_analysis_binary(
                                test_id,
                                list(col_names),
                                test_series,
                                (f"The set of {len(col_names)} columns: {col_names} consistently have exactly "
                                 f"{vc.index[0]} columns with value {col_vals_tuple[1]}."),
                                ""
                            )
                        found_any = True


    def _generate_binary_rare_combo(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        common_vals = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1],
        ]
        rare_vals = [1, 1, 1, 1, 1]

        data = np.array([common_vals[np.random.choice(len(common_vals))] for _ in range(self.num_synth_rows - 1)])
        data = np.vstack([data, rare_vals])
        self._add_synthetic_column('bin_rare_combo all_a', data[:, 0])
        self._add_synthetic_column('bin_rare_combo all_b', data[:, 1])
        self._add_synthetic_column('bin_rare_combo all_c', data[:, 2])
        self._add_synthetic_column('bin_rare_combo all_d', data[:, 3])
        self._add_synthetic_column('bin_rare_combo all_e', data[:, 4])


    def _check_binary_rare_combo(self, test_id):
        """
        This is similar to RARE_PAIRS, which can check columns with any number of values, but is limited to pairs of
        columns. This test can check sets of any number of binary columns, three or more.

        This test considers all sets of binary columns of size 3 or more columns, where the columns have at least 10% of
        their values as both values in that column. The columns do not need to have the same vocabulary.
        It identifies rare combinations of values. Rare is defined as occurring both 1) less than the contamination rate
        times; and 2) less than 5% as would occur given the number of columns in the set if the values were equally
        distributed.
        """

        # Find the set of columns used by this test.
        cols = []
        min_count = self.num_rows / 10.0
        for col_name in self.binary_cols:
            val0, val1 = self.column_unique_vals[col_name]
            if (self.orig_df[col_name].tolist().count(val0) > min_count) and \
                    (self.orig_df[col_name].tolist().count(val1) > min_count) and \
                    (self.orig_df[col_name].isna().sum() < min_count):
                cols.append(col_name)

        max_subset_size = int(np.floor(np.log2(self.num_rows / 20)))

        # Convert all values to 0 and 1, which can be handled more efficiently by numpy than any string or float
        # values which may be in the binary columns.
        cols_df = self.orig_df[cols].copy()
        for col_name in cols:
            cols_df[col_name] = cols_df[col_name].map({self.column_unique_vals[col_name][0]: 0,
                                                       self.column_unique_vals[col_name][1]: 1})
            cols_df[col_name] = cols_df[col_name].fillna(cols_df[col_name].mode()[0])

        cols_np = cols_df.values
        skipped_subsets = {}

        # Determine the upper limit on the number of distinct combinations that would allow any rows to be flagged.
        # We only flag a combination if its count is less than 1/50 what we would expect if all unique values
        # were equal count, given the number of unique values.
        max_combination_count = self.num_rows / 50

        printed_subset_size_msg = False
        for subset_size in range(3, min(len(cols), max_subset_size) + 1):

            # Determine if the current subset_size would generate too many subsets.
            calc_size = math.comb(len(cols), subset_size)
            skip_subsets = calc_size > self.max_combinations
            if skip_subsets:
                if self.verbose >= 2 and not printed_subset_size_msg:
                    print(f"  Skipping subsets of size {subset_size} and larger. There are {calc_size:,} subsets. "
                           f"max_combinations is currently set to {self.max_combinations:,}.")
                    printed_subset_size_msg = True
                continue

            subsets = list(combinations(range(len(cols)), subset_size))

            # Determine the upper limit for the count of a combination to be flagged. The count must be both below
            # the contamination rate and 1/10 of the expected count given a uniform joint distribution.
            flagged_limit = int(min(self.freq_contamination_level, (self.num_rows / math.pow(2, subset_size)) / 10.0))

            if self.verbose >= 2:
                print(f"  Examining subsets of size {subset_size}. There are {len(subsets):,} subsets.")
            for subset in subsets:

                # Do not check subsets of columns if we have already flagged a subset of this subset. Note, this
                # may skip some rare combinations, but is an efficient method to reduce over-reporting. Also
                # do not check subsets of columns if a subset of this subset already has too many distinct combinations
                # to support flagging any combination.
                subset_matches = False
                for fs in skipped_subsets:
                    if set(subset).issuperset(set(fs)):
                        subset_matches = True
                        break
                if subset_matches:
                    continue

                subset_np = cols_np[:, list(subset)].astype(int)
                counts_info = np.unique(subset_np, axis=0, return_counts=True)

                if len(counts_info[1]) > max_combination_count:
                    skipped_subsets[tuple(subset)] = True

                sum_rare = sum([x for x in counts_info[1] if x < self.freq_contamination_level])
                if 0 < sum_rare < self.freq_contamination_level:
                    skipped_subsets[tuple(subset)] = True
                    test_series = [True] * self.num_rows

                    # Loop through each rare combination for this subset of columns
                    for count_idx, count_val in enumerate(counts_info[1]):
                        if count_val < flagged_limit and count_val < ((self.num_rows / len(counts_info[1])) / 50):
                            rare_vals = counts_info[0][count_idx]
                            sub_df = cols_df
                            for col_idx, col_name in enumerate(subset):
                                sub_df = sub_df[sub_df[cols[col_name]] == rare_vals[col_idx]]
                            for i in sub_df.index:
                                test_series[i] = False

                    subset_names = [cols[x] for x in list(subset)]
                    self._process_analysis_binary(
                        test_id,
                        subset_names,
                        test_series,
                        f'For columns {subset_names}, the combinations of values flagged are rare',
                        allow_patterns=False
                    )

    ##################################################################################################################
    # Data consistency checks for pairs of columns where one is binary and one is numeric
    ##################################################################################################################


    def _generate_binary_matches_values(self):
        """
        Patterns without exceptions: 'bin_match_val all' is consistently 0 when 'bin_match_val rand_a' is under 600 and
            consistently 1 when it is over 600
        Patterns with exception: 'bin_match_val most' is the same, with 1 exception
        """
        self._add_synthetic_column('bin_match_val rand_a', np.random.randint(1, 1000, self.num_synth_rows))
        self._add_synthetic_column('bin_match_val all', self.synth_df['bin_match_val rand_a'] > 600)
        self._add_synthetic_column('bin_match_val most', self.synth_df['bin_match_val all'])
        self.synth_df.loc[999, 'bin_match_val most'] = not self.synth_df.loc[999, 'bin_match_val most']


    def _check_binary_matches_values(self, test_id):
        """
        For each pair of columns, where one is binary and the other is numeric, determine if it's consistently the case
        that low values in the numeric column are associated with one value in the binary column, and high values with
        the other value in the binary column.

        This skips binary columns that are almost entirely a single value.
        """

        num_pairs = len(self.binary_cols) * len(self.numeric_cols)
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of binary and numeric columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        for bin_idx, bin_col in enumerate(self.binary_cols):
            if self.verbose >= 2 and bin_idx > 0 and bin_idx % 10 == 0:
                print(f"  Examining column {bin_idx} of {len(self.binary_cols)} binary columns")

            val0, val1 = self.column_unique_vals[bin_col]
            count_val_0 = self.orig_df[bin_col].tolist().count(val0)
            count_val_1 = self.orig_df[bin_col].tolist().count(val1)
            if count_val_0 < (self.num_rows * 0.1):
                continue
            if count_val_1 < (self.num_rows * 0.1):
                continue
            counts_str = (f'Column "{bin_col}" has {count_val_0:,} rows with value "{val0}" and {count_val_1:,} '
                          f'rows with value "{val1}"')
            sub_df_0 = self.orig_df[self.orig_df[bin_col] == val0]
            sub_df_1 = self.orig_df[self.orig_df[bin_col] == val1]

            for num_col in self.numeric_cols:
                # Check the numeric column that has the values has a reasonable number of values besides the most frequent
                if self.orig_df[num_col].value_counts().values[0] > (self.num_rows - self.freq_contamination_level):
                    return

                set_0_numeric_vals = pd.Series([float(x) for x in sub_df_0[num_col] if str(x).replace('-', '').replace('.', '').isdigit()])
                set_1_numeric_vals = pd.Series([float(x) for x in sub_df_1[num_col] if str(x).replace('-', '').replace('.', '').isdigit()])

                set_0_min = set_0_numeric_vals.min()
                set_0_max = set_0_numeric_vals.max()
                set_1_min = set_1_numeric_vals.min()
                set_1_max = set_1_numeric_vals.max()
                set_0_01_percentile = set_0_numeric_vals.quantile(0.01)
                set_0_99_percentile = set_0_numeric_vals.quantile(0.99)
                set_1_01_percentile = set_1_numeric_vals.quantile(0.01)
                set_1_99_percentile = set_1_numeric_vals.quantile(0.99)

                # Test if the numeric values are strictly larger for value 0
                if set_0_min > set_1_max:
                    # Test if the binary column is consistently vol_0 for the larger values in the numeric column
                    threshold = statistics.mean([set_0_min, set_1_max])
                    test_series = [bool(x == val0 and y > threshold or x == val1 and y <= threshold)
                                   for x, y in zip(self.orig_df[bin_col], self.orig_df[num_col])]
                    test_series = np.array(test_series) | self.orig_df[bin_col].isna() | self.orig_df[num_col].isna()
                    self._process_analysis_binary(
                        test_id,
                        [num_col, bin_col],
                        test_series,
                        (f'{counts_str} and is consistently "{val1}" when Column "{num_col}" contains '
                         f'values under {threshold} and "{val0}" when values are over {threshold}')
                    )

                # Test if the numeric values are strictly larger for value 1
                elif set_1_min > set_0_max:
                    # Test if the binary column is consistently vol_0 for the larger values in the numeric column
                    threshold = statistics.mean([set_1_min, set_0_max])
                    test_series = [bool(x == val1 and y > threshold or x == val0 and y <= threshold)
                                   for x, y in zip(self.orig_df[bin_col], self.orig_df[num_col])]
                    test_series = np.array(test_series) | self.orig_df[bin_col].isna() | self.orig_df[num_col].isna()
                    self._process_analysis_binary(
                        test_id,
                        [num_col, bin_col],
                        test_series,
                        (f'{counts_str} and is consistently "{val0}" when Column "{num_col}" contains '
                         f'values under {threshold} and "{val1}" when values are over {threshold}')
                    )

                # Test if the numeric values tend to be larger for value 0
                elif set_0_01_percentile > set_1_99_percentile:
                    # Test if the binary column is consistently vol_0 for the larger values in the numeric column
                    threshold = statistics.mean([set_0_01_percentile, set_1_99_percentile])
                    test_series = [bool(x == val0 and y > threshold or x == val1 and y <= threshold)
                                   for x, y in zip(self.orig_df[bin_col], self.orig_df[num_col])]
                    test_series = np.array(test_series) | self.orig_df[bin_col].isna() | self.orig_df[num_col].isna()
                    self._process_analysis_binary(
                        test_id,
                        [num_col, bin_col],
                        test_series,
                        (f'{counts_str} and is consistently "{val1}" when Column "{num_col}" contains '
                         f'values under {threshold} and "{val0}" when values are over {threshold}')
                    )

                # Test if the numeric values tend to be larger for value 0
                elif set_1_01_percentile > set_0_99_percentile:
                    # Test if the binary column is consistently vol_1 for the larger values in the numeric column
                    threshold = statistics.mean([set_1_01_percentile, set_0_99_percentile])
                    test_series = [bool(x == val1 and y > threshold or x == val0 and y <= threshold)
                                   for x, y in zip(self.orig_df[bin_col], self.orig_df[num_col])]
                    test_series = np.array(test_series) | self.orig_df[bin_col].isna() | self.orig_df[num_col].isna()
                    self._process_analysis_binary(
                        test_id,
                        [num_col, bin_col],
                        test_series,
                        (f'{counts_str} and is consistently "{val0}" when Column "{num_col}" contains '
                         f'values under {threshold} and "{val1}" when values are over {threshold}')
                    )

    ##################################################################################################################
    # Data consistency checks for sets of three columns, where one must be binary
    ##################################################################################################################


    def _generate_binary_two_others_match(self):
        """
        Patterns without exceptions: 'bin_match_others all' is consistently true iff 'bin_match_others rand_a' ==
            'bin_match_others rand_b'
        Patterns with exception: similar for 'bin_match_others most', with one exception. As well,
            'bin_match_others date_most' is consitently true iff 'bin_match_others date_rand_a' matches
            'bin_match_others date_rand_b', with one exception.
        """
        self._add_synthetic_column('bin_match_others rand_a', np.random.randint(1, 10, self.num_synth_rows))
        self._add_synthetic_column('bin_match_others rand_b', np.random.randint(1, 5, self.num_synth_rows))
        self._add_synthetic_column('bin_match_others all',
            self.synth_df['bin_match_others rand_a'] == self.synth_df['bin_match_others rand_b'])
        self._add_synthetic_column('bin_match_others most', self.synth_df['bin_match_others all'])
        self.synth_df.loc[999, 'bin_match_others most'] = not self.synth_df.loc[999, 'bin_match_others most']

        test_date1 = datetime.datetime.strptime("5-8-2020", "%d-%m-%Y")
        self._add_synthetic_column('bin_match_others date_rand_a',
            [test_date1 + relativedelta(days=(np.random.randint(1, 5))) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_match_others date_rand_b',
            [test_date1 + relativedelta(days=(np.random.randint(1, 5))) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('bin_match_others date_most',
            self.synth_df['bin_match_others date_rand_a'] == self.synth_df['bin_match_others date_rand_b'])
        self.synth_df.loc[999, 'bin_match_others date_most'] = not self.synth_df.loc[999, 'bin_match_others date_most']


    def _check_binary_two_others_match(self, test_id):
        """
        Handling Null values: The two columns are considered a match if both are Null, and not o match if one, but not
        both, is Null. Rows where the binary column is Null do not count for or against the pattern.
        """

        def test_set(bin_col, col_name_2, col_name_3):
            nonlocal sub_df_0, sub_df_1

            # When checking pairs of binary columns, one may be the same as the bin_col
            if bin_col in (col_name_2, col_name_3):
                return

            # Skip where the columns are almost the same
            if cols_same_bool_dict[tuple(sorted([col_name_2, col_name_3]))]:
                return

            # Check the two columns that have the values that are checked have a reasonable number of values besides
            # the most frequent
            if self.orig_df[col_name_2].value_counts().values[0] > (self.num_rows - self.freq_contamination_level):
                return
            if self.orig_df[col_name_3].value_counts().values[0] > (self.num_rows - self.freq_contamination_level):
                return

            # Test first on a sample of the rows where the bin_col has value 0
            if sample_sub_df_0[col_name_2].dtype.name == 'category' or \
                    sample_sub_df_0[col_name_3].dtype.name == 'category':
                test_series_0 = as_str(sample_sub_df_0[col_name_2]) == as_str(sample_sub_df_0[col_name_3])
            else:
                test_series_0 = sample_sub_df_0[col_name_2] == sample_sub_df_0[col_name_3]
            test_series_0 = test_series_0 | (sample_sub_df_0[col_name_2].isna() & sample_sub_df_0[col_name_3].isna())

            # Before calculating test_series_1, determine if test_series_0 is mostly True or mostly False. If so,
            # we can return.
            if (test_series_0.tolist().count(True) > 2) and (test_series_0.tolist().count(False) > 2):
                return

            # Test on a sample of the rows where bin_col has value 1
            if sample_sub_df_0[col_name_2].dtype.name == 'category' or \
                    sample_sub_df_0[col_name_3].dtype.name == 'category':
                test_series_1 = as_str(sample_sub_df_1[col_name_2]) == as_str(sample_sub_df_1[col_name_3])
            else:
                test_series_1 = sample_sub_df_1[col_name_2] == sample_sub_df_1[col_name_3]
            test_series_1 = test_series_1 | (sample_sub_df_1[col_name_2].isna() & sample_sub_df_1[col_name_3].isna())

            # Test if the binary column has val_0 iff the other 2 columns are equal
            sample_okay_val_0 = False
            if (test_series_0.tolist().count(False) < 1) and (test_series_1.tolist().count(True) < 1):
                sample_okay_val_0 = True

            # Test if the binary column has val_1 iff the other 2 columns are equal
            sample_okay_val_1 = False
            if (test_series_0.tolist().count(True) < 1) and (test_series_1.tolist().count(False) < 1):
                sample_okay_val_1 = True

            if not sample_okay_val_0 and not sample_okay_val_1:
                return

            # Test on the full columns
            if sub_df_0 is None:
                sub_df_0 = self.orig_df[self.orig_df[bin_col] == val0]
                sub_df_1 = self.orig_df[self.orig_df[bin_col] == val1]

            # Todo: this handles when the dtype is 'category', but it may be faster to treat as category
            if sub_df_0[col_name_2].dtype.name == 'category' or sub_df_0[col_name_3].dtype.name == 'category':
                test_series_0 = as_str(sub_df_0[col_name_2]) == as_str(sub_df_0[col_name_3])
                test_series_1 = as_str(sub_df_1[col_name_2]) == as_str(sub_df_1[col_name_3])
            else:
                test_series_0 = sub_df_0[col_name_2] == sub_df_0[col_name_3]
                test_series_1 = sub_df_1[col_name_2] == sub_df_1[col_name_3]
            test_series_0 = test_series_0 | (sub_df_0[col_name_2].isna() & sub_df_0[col_name_3].isna())
            test_series_1 = test_series_1 | (sub_df_1[col_name_2].isna() & sub_df_1[col_name_3].isna())
            value_when_match = -1

            test_series = None

            # Test if the binary column has val_0 iff the other 2 columns are equal
            if (test_series_0.tolist().count(False) < self.freq_contamination_level) and \
                    (test_series_1.tolist().count(True) < self.freq_contamination_level):
                value_when_match = val0
                value_when_dont = val1
                test_series = [True] * self.num_rows
                for i in sub_df_0.index:
                    if not test_series_0[i]:
                        test_series[i] = False
                for i in sub_df_1.index:
                    if test_series_1[i]:
                        test_series[i] = False

            # Test if the binary column has val_1 iff the other 2 columns are equal
            if (test_series_0.tolist().count(True) < self.freq_contamination_level) and \
                    (test_series_1.tolist().count(False) < self.freq_contamination_level):
                value_when_match = val1
                value_when_dont = val0
                test_series = [True] * self.num_rows
                for i in sub_df_0.index:
                    if test_series_0[i]:
                        test_series[i] = False
                for i in sub_df_1.index:
                    if not test_series_1[i]:
                        test_series[i] = False

            if not test_series:
                return

            test_series = np.array(test_series) | \
                          self.orig_df[bin_col].isna() | \
                          (self.orig_df[col_name_2].isna() & self.orig_df[col_name_3].isna())

            self._process_analysis_binary(
                test_id,
                [col_name_2, col_name_3, bin_col],
                test_series,
                (f"Column {bin_col} (with value {val0} {self.orig_df[bin_col].tolist().count(val0)} times and value "
                 f"{val1} {self.orig_df[bin_col].tolist().count(val1)} times) consistently has value "
                 f"{value_when_match} when columns {col_name_2} and {col_name_3} match values, and value "
                 f"{value_when_dont} when they do not match"),
                display_info={"Match": (self.orig_df[col_name_2] == self.orig_df[col_name_3]) |
                                       (self.orig_df[col_name_2].isna() & self.orig_df[col_name_3].isna())}
            )

        if len(self.binary_cols) == 0:
            return

        # Check if there are too many combinations of any type
        skip_numeric = False
        num_pairs, numeric_pairs = self._get_numeric_column_pairs_unique()
        if numeric_pairs is None:
            skip_numeric = True
        else:
            if (num_pairs * len(self.binary_cols)) > self.max_combinations:
                if self.verbose >= 1:
                    print(f"  Skipping testing numeric columns. There are {num_pairs * len(self.binary_cols):,} combinations of binary and numeric columns. "
                           f"max_combinations is currently set to {self.max_combinations:,}.")
                skip_numeric = True

        skip_string = False
        num_pairs, string_pairs = self._get_string_column_pairs_unique()
        if string_pairs is None:
            skip_string = True
        else:
            if (num_pairs * len(self.binary_cols)) > self.max_combinations:
                if self.verbose >= 1:
                    print(f"  Skipping testing string columns. There are {num_pairs * len(self.binary_cols):,} combinations of binary and string columns. "
                           f"max_combinations is currently set to {self.max_combinations:,}.")
                skip_string = True

        skip_date = False
        num_pairs, date_pairs = self._get_date_column_pairs_unique()
        if date_pairs is None:
            skip_date = True
        else:
            if (num_pairs * len(self.binary_cols)) > self.max_combinations:
                if self.verbose >= 1:
                    print(f"  Skipping testing date columns. There are {num_pairs * len(self.binary_cols):,} combinations of binary and date columns. "
                           f"max_combinations is currently set to {self.max_combinations:,}.")
                skip_date = True

        skip_binary = False
        num_pairs, binary_pairs = self._get_binary_column_pairs_unique()
        if binary_pairs is None:
            skip_binary = True
        else:
            if (num_pairs * len(self.binary_cols)) > self.max_combinations:
                if self.verbose >= 1:
                    print(f"  Skipping testing binary columns. There are {num_pairs * len(self.binary_cols):,} combinations of binary columns. "
                           f"max_combinations is currently set to {self.max_combinations:,}.")
                skip_binary = True

        cols_same_bool_dict = self.get_cols_same_bool_dict()
        for bin_idx, bin_col in enumerate(self.binary_cols):
            if self.verbose >= 2 and bin_idx > 0 and bin_idx % 10 == 0:
                print(f"  Examining column {bin_idx} of {len(self.binary_cols)} binary columns")

            # Test the binary column contains both values a reasonable amount.
            val0, val1 = self.column_unique_vals[bin_col]
            if self.orig_df[bin_col].tolist().count(val0) < (self.num_rows * 0.1):
                continue
            if self.orig_df[bin_col].tolist().count(val1) < (self.num_rows * 0.1):
                continue

            # Create 2 dataframes that each cover one of the two values in the binary column. Do for the sample
            # dataframe as well.
            sample_sub_df_0 = self.sample_df[self.sample_df[bin_col] == val0]
            sample_sub_df_1 = self.sample_df[self.sample_df[bin_col] == val1]
            sub_df_0 = None
            sub_df_1 = None

            # Test for each pair columns, where the pair are of the same type.
            if not skip_numeric:
                for col_name_2, col_name_3 in numeric_pairs:
                    test_set(bin_col, col_name_2, col_name_3)

            if not skip_string:
                for col_name_2, col_name_3 in string_pairs:
                    test_set(bin_col, col_name_2, col_name_3)

            if not skip_date:
                for col_name_2, col_name_3 in date_pairs:
                    test_set(bin_col, col_name_2, col_name_3)

            if not skip_binary:
                for col_name_2, col_name_3 in binary_pairs:
                    test_set(bin_col, col_name_2, col_name_3)

    ##################################################################################################################
    # Data consistency checks for sets of three columns, where one is binary and the other two string
    ##################################################################################################################

    def _generate_binary_two_str_match(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        # todo: think through how to code this. I think break into smaller, simple tests:
        # binary col true if: 1) the 2 string cols have the same length, but are many lengths in the columns;
        #   2) the 2 string cols have almost the same set of characters
        # self._add_synthetic_column('bin_str_match rand_a', np.random.choice(['a', 'aa', 'aaa', 'aaaa'], self.num_synth_rows))
        # self._add_synthetic_column('bin_str_match rand_b', np.random.choice(['a', 'aa', 'b', 'bb'], self.num_synth_rows))
        # self._add_synthetic_column('bin_str_match all', self.synth_df['bin_str_match rand_a'] == self.synth_df['bin_str_match rand_b'])
        # self._add_synthetic_column('bin_str_match most', self.synth_df['bin_str_match all'])


    def _check_binary_two_str_match(self, test_id):
        """
        """

    ##################################################################################################################
    # Data consistency sets of multiple columns, where one is binary and the others are numeric
    ##################################################################################################################

    def _generate_binary_matches_sum(self):
        """
        Patterns without exceptions: 'bin_match_sum all' is consistently False when the sum of 'bin_match_sum rand_a'
            and 'bin_match_sum rand_b' is below 1000 and True when the sum is above
        Patterns with exception: 'bin_match_sum most' is consistently False when the sum of 'bin_match_sum rand_a'
            and 'bin_match_sum rand_b' is below 1000 and True when the sum is above, with 1 exception in row 999.
        """
        self._add_synthetic_column('bin_match_sum rand_a', np.random.randint(1, 1000, self.num_synth_rows))
        self._add_synthetic_column('bin_match_sum rand_b', np.random.randint(1, 1000, self.num_synth_rows))
        self._add_synthetic_column('bin_match_sum all',
            self.synth_df['bin_match_sum rand_a'] + self.synth_df['bin_match_sum rand_b'] > 1000)
        self._add_synthetic_column('bin_match_sum most', self.synth_df['bin_match_sum all'])
        self.synth_df.loc[999, 'bin_match_sum most'] = not self.synth_df.loc[999, 'bin_match_sum most']


    def _check_binary_matches_sum(self, test_id):

        def check_nulls_matching(bin_col, num_col_1, num_col_2):
            """
            Check if either numeric value is mostly Null for either of the binary values.
            """
            sub_df_v0 = self.orig_df[[bin_col, num_col_1, num_col_2]][self.orig_df[bin_col] == val0]
            if sub_df_v0[num_col_1].isna().sum() > (len(sub_df_v0) * 0.75):
                return False
            if sub_df_v0[num_col_2].isna().sum() > (len(sub_df_v0) * 0.75):
                return False
            sub_df_v1 = self.orig_df[[bin_col, num_col_1, num_col_2]][self.orig_df[bin_col] == val0]
            if sub_df_v1[num_col_1].isna().sum() > (len(sub_df_v1) * 0.75):
                return False
            return not (sub_df_v1[num_col_2].isna().sum() > (len(sub_df_v1) * 0.75))

        if len(self.binary_cols) == 0:
            return

        _, pairs = self._get_numeric_column_pairs_unique()
        if pairs is None:
            return

        nunique_dict = self.get_nunique_dict()

        # Calculate and cache the sums of each pair of numeric columns
        sums_arr_dict = {}
        sorted_sums_arr_dict = {}
        for num_col_1, num_col_2 in pairs:
            # Check the two columns that have the values that are checked have a reasonable number of values besides
            # the most frequent
            if self.orig_df[num_col_1].value_counts().values[0] > (self.num_rows - self.freq_contamination_level):
                return
            if self.orig_df[num_col_2].value_counts().values[0] > (self.num_rows - self.freq_contamination_level):
                return

            if not self.check_columns_same_scale_2(num_col_1, num_col_2):
                continue
            sum_arr = pd.Series(self.numeric_vals_filled[num_col_1] + self.numeric_vals_filled[num_col_2])
            sum_arr = sum_arr.fillna(self.column_medians[num_col_1] + self.column_medians[num_col_2])
            sums_arr_dict[(num_col_1, num_col_2)] = sum_arr
            sorted_sums_arr_dict[(num_col_1, num_col_2)] = pd.Series(sorted(sum_arr))

        for bin_idx, bin_col in enumerate(self.binary_cols):
            if self.verbose >= 2 and bin_idx > 0 and bin_idx % 1 == 0:
                print(f"  Examining column {bin_idx} of {len(self.binary_cols)} binary columns")

            val0, val1 = self.column_unique_vals[bin_col]
            num_val_0 = self.orig_df[bin_col].tolist().count(val0)
            num_val_1 = self.orig_df[bin_col].tolist().count(val1)
            if num_val_0 < (self.num_rows * 0.1):
                continue
            if num_val_1 < (self.num_rows * 0.1):
                continue

            for num_col_1, num_col_2 in pairs:
                if not self.check_columns_same_scale_2(num_col_1, num_col_2):
                    continue

                if (nunique_dict[num_col_1] < 10) or (nunique_dict[num_col_2] < 10):
                    continue

                sum_arr = sums_arr_dict[(num_col_1, num_col_2)]
                sorted_sum_arr = sorted_sums_arr_dict[(num_col_1, num_col_2)]

                # We check if it appears the binary column has val_0 for smaller sums, or val_1 for smaller sums.
                # If either is True, we set a flag and do not check the other.
                val_0_for_smaller = False
                val_1_for_smaller = False

                # Test if the binary column is consistently val_0 for the larger values in the numeric column.
                # If bin_col is val_0 for smaller values, then the threshold will be at the point in the sorted
                # array corresponding to the number of instances of val_0
                val_at_frac_0 = sorted_sum_arr[num_val_0]
                idxs_below_threshold_0 = np.where(sum_arr[:50] < val_at_frac_0)
                sub_df_0 = self.orig_df[bin_col].loc[idxs_below_threshold_0]
                if sub_df_0.tolist().count(val1) < self.freq_contamination_level:
                    val_0_for_smaller = True

                if not val_0_for_smaller:
                    val_at_frac_1 = sorted_sum_arr[num_val_1]
                    idxs_below_threshold_1 = np.where(sum_arr[:50] < val_at_frac_1)
                    sub_df_1 = self.orig_df[bin_col].loc[idxs_below_threshold_1]
                    if sub_df_1.tolist().count(val0) < self.freq_contamination_level:
                        val_1_for_smaller = True

                if not val_0_for_smaller and not val_1_for_smaller:
                    continue

                sample_size = 25

                # Test if the binary column is consistently val_0 for the smaller values in the numeric column
                if val_0_for_smaller:
                    threshold = val_at_frac_0

                    # Test on a sample of rows
                    test_series = [bool(x == val0 and y <= threshold or x == val1 and y >= threshold)
                                   for x, y in zip(self.orig_df[bin_col].head(sample_size), sum_arr.head(sample_size))]
                    test_series = np.array(test_series) | \
                                  self.orig_df[bin_col].head(sample_size).isna() | \
                                  self.orig_df[num_col_1].head(sample_size).isna() | \
                                  self.orig_df[num_col_2].head(sample_size).isna()
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on the full columns
                    test_series = [bool(x == val0 and y <= threshold or x == val1 and y >= threshold)
                                   for x, y in zip(self.orig_df[bin_col], sum_arr)]
                    if not check_nulls_matching(bin_col, num_col_1, num_col_2):
                        continue
                    test_series = np.array(test_series) | \
                                  self.orig_df[bin_col].isna() | \
                                  self.orig_df[num_col_1].isna() | \
                                  self.orig_df[num_col_2].isna()

                    self._process_analysis_binary(
                        test_id,
                        [num_col_1, num_col_2, bin_col],
                        test_series,
                        (f'Column "{bin_col}" is consistently {val0} when the sum of columns "{num_col_1}" and '
                         f'"{num_col_2}" is under {threshold} and {val1} when the sum is over.'),
                    )

                # Test if the binary column is consistently val_0 for the larger values in the numeric column
                else:
                    threshold = val_at_frac_1

                    # Test on a sample of rows
                    test_series = [bool(x == val1 and y > threshold or x == val0 and y <= threshold)
                                   for x, y in zip(self.orig_df[bin_col].head(sample_size), sum_arr.head(sample_size))]
                    test_series = np.array(test_series) | \
                                  self.orig_df[bin_col].head(sample_size).isna() | \
                                  self.orig_df[num_col_1].head(sample_size).isna() | \
                                  self.orig_df[num_col_2].head(sample_size).isna()
                    if test_series.tolist().count(False) > 1:
                        continue

                    # Test on the full columns
                    test_series = [bool(x == val1 and y > threshold or x == val0 and y <= threshold)
                                   for x, y in zip(self.orig_df[bin_col], sum_arr)]
                    if not check_nulls_matching(bin_col, num_col_1, num_col_2):
                        continue
                    test_series = np.array(test_series) | \
                                  self.orig_df[bin_col].isna() | \
                                  self.orig_df[num_col_1].isna() | \
                                  self.orig_df[num_col_2].isna()

                    self._process_analysis_binary(
                        test_id,
                        [num_col_1, num_col_2, bin_col],
                        test_series,
                        (f'Column "{bin_col}" is consistently {val1} when the sum of columns "{num_col_1}" and '
                         f'"{num_col_2}" is over {threshold} and {val0} when the sum is under'),
                    )

    ##################################################################################################################
    # Data consistency checks for single non-numeric columns
    ##################################################################################################################

