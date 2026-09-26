"""Lazy analysis caches and reusable column-combination helpers."""

from __future__ import annotations

import math

import pandas as pd

from .checker_utils import is_missing, replace_special_with_space


class AnalysisCacheMixin:
    """Mixin providing cached statistics and reusable column-set helpers."""

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
            similar_cols = [col_name] if include_self else []
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
                        pair_tuple = (col_name, c)
                        # cols_same_bool_dict uses sorted column names as the key. Values in larger_dict may be
                        # None, arrays or booleans, so they are compared to False / True explicitly.
                        if not cols_same_bool_dict[tuple(sorted([col_name, c]))] and (
                                (not check_larger_false and not check_larger_true) or
                                (check_larger_false and ((pair_tuple not in larger_dict) or
                                                         (larger_dict[pair_tuple] == False))) or  # noqa: E712
                                (check_larger_true and (pair_tuple in larger_dict) and
                                 (larger_dict[pair_tuple] == True))):  # noqa: E712
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
                    print(f"  Due to the potential number of combinations, limiting test to subsets of size "
                           f"{max_subset_size}.")
                else:
                    print(f"  Skipping test. Given the number of similar columns for each positive numeric "
                           f"column, there are {calc_size_limited:,} combinations, even limiting testing to subsets "
                           f"2 columns. max_combinations is currently set to {self.max_combinations:,}.")

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


    def check_columns_same_scale_2(self, col_name_1, col_name_2, order=2):
        med_1 = self.column_medians[col_name_1]
        med_2 = self.column_medians[col_name_2]
        if med_1 != 0 and med_2 != 0:
            ratio_1_2 = abs(med_1 / med_2)
            if ratio_1_2 < (1.0/order) or ratio_1_2 > order:
                return False
        return True

    def check_columns_same_scale_3(self, col_name_1, col_name_2, col_name_3, order=2):
        med_1 = self.column_medians[col_name_1]
        med_2 = self.column_medians[col_name_2]
        med_3 = self.column_medians[col_name_3]
        if med_1 != 0 and med_2 != 0:
            ratio_1_2 = abs(med_1 / med_2)
            if ratio_1_2 < (1/order) or ratio_1_2 > order:
                return False
        if med_1 != 0 and med_3 != 0:
            ratio_1_3 = abs(med_1 / med_3)
            if ratio_1_3 < (1/order) or ratio_1_3 > order:
                return False
        if med_2 != 0 and med_3 != 0:
            ratio_2_3 = abs(med_2 / med_3)
            if ratio_2_3 < (1/order) or ratio_2_3 > order:
                return False
        return True

    def check_results_for_null(self, test_series, col_name, subset):
        """
        Used by tests that work with any number of columns, and where Null values do not violate the general pattern.
        """
        if col_name:
            test_series = test_series | self.orig_df[col_name].isna()
        for col in subset:
            test_series = test_series | self.orig_df[col].isna()
        return test_series

    ##################################################################################################################
    # Methods to populate the caches used by some of the tests. These are not set in init(), as these may not be used
    # in the tests actually executed.
    ##################################################################################################################

    def get_columns_iqr_upper_limit(self):
        """
        For each numeric & date column, calculate q2, q3, and the IQR upper limit, which is based on q3 and the IQR.
        """

        if self.upper_limits_dict:
            return self.upper_limits_dict
        self.upper_limits_dict = {}
        for col_name in self.numeric_cols:
            num_vals = self.numeric_vals[col_name]
            q1 = num_vals.quantile(0.25)
            q2 = num_vals.quantile(0.5)
            q3 = num_vals.quantile(0.75)
            upper_limit = q3 + (self.iqr_limit * (q3 - q1))
            self.upper_limits_dict[col_name] = (upper_limit, q2, q3)
        for col_name in self.date_cols:
            q1 = pd.to_datetime(self.orig_df[col_name]).quantile(0.25, interpolation='midpoint')
            q2 = pd.to_datetime(self.orig_df[col_name]).quantile(0.50, interpolation='midpoint')
            q3 = pd.to_datetime(self.orig_df[col_name]).quantile(0.75, interpolation='midpoint')
            try:
                upper_limit = q3 + (self.iqr_limit * (q3 - q1))
            except Exception:
                self.upper_limits_dict[col_name] = (None, None, None)
                continue
            self. upper_limits_dict[col_name] = (upper_limit, q2, q3)
        return self.upper_limits_dict

    def get_columns_iqr_lower_limit(self):
        """
        For each numeric column, calculate d1, d9, and the IDR lower limit, which is based on d1 and the IDR.
        For each date column, calculate q1, q3, and the IQR lower limit, which is based on q1 and the IQR.
        """

        if self.lower_limits_dict:
            return self.lower_limits_dict
        self.lower_limits_dict= {}
        for col_name in self.numeric_cols:
            num_vals = self.numeric_vals[col_name]
            d1 = num_vals.quantile(0.1)
            q1 = num_vals.quantile(0.25)
            d9 = num_vals.quantile(0.9)
            lower_limit = d1 - (self.idr_limit * (d9 - d1))
            self.lower_limits_dict[col_name] = (lower_limit, d1, q1)
        for col_name in self.date_cols:
            # todo: possibly use deciles, the same as the numeric case, though dates are distributed differently.
            q1 = pd.to_datetime(self.orig_df[col_name]).quantile(0.25, interpolation='midpoint')
            q3 = pd.to_datetime(self.orig_df[col_name]).quantile(0.75, interpolation='midpoint')
            try:
                lower_limit = q1 - (self.iqr_limit * (q3 - q1))
            except Exception:
                self.lower_limits_dict[col_name] = (None, None, None)
                continue
            self.lower_limits_dict[col_name] = (lower_limit, q1, q3)
        return self.lower_limits_dict

    def get_larger_pairs_dict(self, allow_equal, print_status=False):
        """
        Populates self.larger_pairs_dict and self.larger_or_equal_pairs_dict. There is an element for each
        pair of numeric columns where one column is potentially larger than the other (or larger than or equal to),
        row by row.
        For any given pair of columns, A and B, the dictionary may have a key of (A, B) or (B, A), or may have neither.
        Where the dictionary contains (A, B), this indicates A is potentially larger than B, given their medians.
        The value for the tuple will indicate, row by row, where A is, in fact, larger than B.
        """

        if self.larger_pairs_dict:
            if allow_equal:
                return self.larger_or_equal_pairs_dict
            return self.larger_pairs_dict

        self.larger_pairs_dict = {}
        self.larger_or_equal_pairs_dict = {}
        num_pairs, col_pairs = self._get_numeric_column_pairs()
        if num_pairs == 0 or col_pairs is None:
            return self.larger_pairs_dict
        for cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            key = (col_name_1, col_name_2)

            if print_status and self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining pair {cols_idx:,} of {len(col_pairs):,} pairs of numeric columns.")

            if self.column_medians[col_name_1] < (self.column_medians[col_name_2] * 0.95):
                self.larger_pairs_dict[key] = None
                continue

            vals_arr_1 = self.sample_numeric_vals_filled[col_name_1]
            vals_arr_2 = self.sample_numeric_vals_filled[col_name_2]
            sample_series = ((vals_arr_1 - vals_arr_2) >= 0) | \
                            self.sample_df[col_name_1].isna().values | \
                            self.sample_df[col_name_2].isna().values
            if sample_series.tolist().count(False) > 1:
                self.larger_pairs_dict[key] = None
                continue

            vals_arr_1 = self.numeric_vals_filled[col_name_1]
            vals_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series = ((vals_arr_1 - vals_arr_2) > 0) | \
                          self.orig_df[col_name_1].isna() | \
                          self.orig_df[col_name_2].isna()
            self.larger_pairs_dict[key] = test_series
            test_series = ((vals_arr_1 - vals_arr_2) >= 0) | \
                          self.orig_df[col_name_1].isna() | \
                          self.orig_df[col_name_2].isna()
            self.larger_or_equal_pairs_dict[key] = test_series
        if allow_equal:
            return self.larger_or_equal_pairs_dict
        return self.larger_pairs_dict

    def get_larger_pairs_with_bool_dict(self):
        """
        Populates self.larger_pairs_with_bool_dict, which has a key for each pair of columns, (A, B) where A is
        potentially larger than B, based on their medians. The value for each key is a boolean value indicating
        if A is actually, row by row, larger than B consistently, with no more than self.freq_contamination_level exceptions.
        """
        if self.larger_pairs_with_bool_dict:
            return self.larger_pairs_with_bool_dict
        self.larger_pairs_with_bool_dict = {}
        larger_dict = self.get_larger_pairs_dict(allow_equal=False)
        for key in larger_dict:
            if larger_dict[key] is not None:
                self.larger_pairs_with_bool_dict[key] = \
                    larger_dict[key].tolist().count(True) > self.freq_contamination_level
            else:
                self.larger_pairs_with_bool_dict[key] = False
        return self.larger_pairs_with_bool_dict

    def get_larger_or_equal_pairs_with_bool_dict(self):
        if self.larger_or_equal_pairs_with_bool_dict:
            return self.larger_or_equal_pairs_with_bool_dict
        self.larger_or_equal_pairs_with_bool_dict = {}
        larger_or_equal_dict = self.get_larger_pairs_dict(allow_equal=True)
        for key in larger_or_equal_dict:
            if larger_or_equal_dict[key] is not None:
                self.larger_or_equal_pairs_with_bool_dict[key] = \
                    larger_or_equal_dict[key].tolist().count(True) > self.freq_contamination_level
            else:
                self.larger_or_equal_pairs_with_bool_dict[key] = False
        return self.larger_or_equal_pairs_with_bool_dict

    # todo: call this everywhere to reduce work

    def get_is_missing_dict(self):
        """
        For all columns, creates an array the length of the original data, indicating which cells are flagged by
        is_missing(), which is more general that isna().
        """

        if self.is_missing_dict:
            return self.is_missing_dict
        self.is_missing_dict = {}
        for col_name in self.orig_df.columns:
            self.is_missing_dict[col_name] = self.orig_df[col_name].apply(is_missing)
        return self.is_missing_dict

    def get_num_missing_dict(self):
        """
        For all columns, creates an integer indicated the number of missing values.
        """
        if self.num_missing_dict:
            return self.num_missing_dict
        self.num_missing_dict = {}
        is_missing_dict = self.get_is_missing_dict()
        for col_name in self.orig_df.columns:
            self.num_missing_dict[col_name] = is_missing_dict[col_name].tolist().count(True)
        return self.num_missing_dict

    def get_sample_is_missing_dict(self):
        """
        Equivalent to get_is_missing_dict(), but done on the sample dataframe.
        """

        if self.sample_is_missing_dict:
            return self.sample_is_missing_dict
        self.sample_is_missing_dict = {}
        for col_name in self.orig_df.columns:
            self.sample_is_missing_dict[col_name] = self.sample_df[col_name].apply(is_missing)
        return self.sample_is_missing_dict

    def get_percentiles_dict(self):
        """
        For each numeric column, calculate an array of values, the same length as the original data, representing
        the percentile value of the original numeric value.
        """

        if self.percentiles_dict:
            return self.percentiles_dict
        self.percentiles_dict = {}
        for col_name in self.numeric_cols:
            self.percentiles_dict[col_name] = self.orig_df[col_name].rank(pct=True)
        return self.percentiles_dict

    # todo: call this were we are now calling nunique() on the full columns

    def get_nunique_dict(self):
        """
        For each column, calculate an integer value representing the number of unique values in the column.
        """

        if self.nunique_dict:
            return self.nunique_dict
        self.nunique_dict = {}
        for col_name in self.orig_df.columns:
            self.nunique_dict[col_name] = self.orig_df[col_name].nunique()
        return self.nunique_dict

    # todo: call this were we are now calling value_counts on the full columns

    def get_count_most_freq_value_dict(self):
        """
        For each column, calculate a single value: the most frequent unique value in that column.
        """

        if self.count_most_freq_value_dict:
            return self.count_most_freq_value_dict
        self.count_most_freq_value_dict = {}
        for col_name in self.orig_df.columns:
            self.count_most_freq_value_dict[col_name] = self.orig_df[col_name].value_counts().values[0]
        return self.count_most_freq_value_dict

    # todo: call this were get word lists

    def get_words_list_dict(self):
        """
        For each string column, calculate an array of values of the same length as the original data. Each value is an
        array containing the words within the original value, split by whitespace or non-alphanumeric characters.
        """

        if self.words_list_dict:
            return self.words_list_dict
        self.words_list_dict = {}
        for col_name in self.string_cols:
            self.words_list_dict[col_name] = \
                self.orig_df[col_name].astype(str).apply(replace_special_with_space).str.split().values
        return self.words_list_dict

    def get_word_counts_dict(self):
        """
        For each string column, calculate an array of values of the same length as the original data. Each value is an
        integer count representing the number of words in the cell in the original data, split by whitespace or
        non-alphanumeric characters. This counts null values as having 0 words.
        """

        if self.word_counts_dict:
            return self.word_counts_dict
        self.word_counts_dict = {}
        for col_name in self.string_cols:
            col_vals = self.orig_df[col_name].fillna("").astype(str).apply(replace_special_with_space)
            word_counts_arr = [0 if x is None else len(x) for x in col_vals.str.split()]
            self.word_counts_dict[col_name] = word_counts_arr
        return self.word_counts_dict

    def get_cols_same_bool_dict(self, force=False):
        """
        For each pair of columns, store a binary value indicating if the two columns are the same (other than up to
        contamination_level rows). This also calculates cols_same_count_dict, though this is estimated for efficiency.
        The pair of columns are keyed by a sorted tuple.
        This adds an element for every pair of features of the same type.
        """

        def check_match(col_name_a, col_name_b):
            pairs_tuple = tuple(sorted([col_name_a, col_name_b]))

            # Test first on a sample
            are_same_arr = [(x == y) or (n1 and n2)
                            for x, y, n1, n2 in zip(
                                    self.sample_df[col_name_a],
                                    self.sample_df[col_name_b],
                                    self.sample_df[col_name_a].isna(),
                                    self.sample_df[col_name_b].isna())]
            if are_same_arr.count(False) > 1:
                self.cols_same_bool_dict[pairs_tuple] = False
                self.cols_same_count_dict[pairs_tuple] = (are_same_arr.count(True) / len(self.sample_df)) * self.num_rows
                return

            # Test on the full columns
            are_same_arr = [(x == y) or (n1 and n2)
                            for x, y, n1, n2 in zip(self.orig_df[col_name_a],
                                                    self.orig_df[col_name_b],
                                                    self.orig_df[col_name_a].isna(),
                                                    self.orig_df[col_name_b].isna())]
            if are_same_arr.count(True) > (self.num_rows - self.freq_contamination_level):
                self.cols_same_bool_dict[pairs_tuple] = True
            else:
                self.cols_same_bool_dict[pairs_tuple] = False
            self.cols_same_count_dict[pairs_tuple] = are_same_arr.count(True)

        if self.cols_same_bool_dict:
            return self.cols_same_bool_dict
        self.cols_same_bool_dict = {}
        self.cols_same_count_dict = {}

        # Check pairs of numeric columns
        num_pairs, pairs_arr = self._get_numeric_column_pairs_unique(force=force)
        if pairs_arr is not None:
            for _pair_idx, (col_name_a, col_name_b) in enumerate(pairs_arr):
                check_match(col_name_a, col_name_b)

        # Check pairs of string columns
        num_pairs, pairs_arr = self._get_string_column_pairs_unique(force=force)
        for _pair_idx, (col_name_a, col_name_b) in enumerate(pairs_arr):
            check_match(col_name_a, col_name_b)

        # Check pairs of binary columns
        num_pairs, pairs_arr = self._get_binary_column_pairs_unique(force=force)
        for _pair_idx, (col_name_a, col_name_b) in enumerate(pairs_arr):
            check_match(col_name_a, col_name_b)

        # Check pairs of date columns
        num_pairs, pairs_arr = self._get_date_column_pairs_unique(force=force)
        for _pair_idx, (col_name_a, col_name_b) in enumerate(pairs_arr):
            check_match(col_name_a, col_name_b)

        return self.cols_same_bool_dict

    def get_cols_same_count_dict(self):
        """
        Similar to get_cols_same_bool_dict(), but returns a count of how many values are the same, as opposed to a
        boolean flag. This count is approximate, extrapolated from a sample in cases where they do not appear to be
        similar.
        """

        if self.cols_same_count_dict:
            return self.cols_same_count_dict
        self.get_cols_same_bool_dict()
        return self.cols_same_count_dict

    def get_col_pair_both_null_dict(self, force=False):
        """
        Creates an element for each pair of columns. Each element is an array with the same length as the original data
        containing a bool value indicating if both cell values are Null.
        """

        if self.cols_pairs_both_null_dict:
            return self.cols_pairs_both_null_dict
        self.cols_pairs_both_null_dict = {}
        _, pairs = self._get_column_pairs_unique(force=force)
        if pairs is None:
            return None
        for col_name_a, col_name_b in pairs:
            pairs_tuple = tuple(sorted([col_name_a, col_name_b]))
            match_arr = (self.orig_df[col_name_a].isna() & self.orig_df[col_name_b].isna())
            self.cols_pairs_both_null_dict[pairs_tuple] = match_arr
        return self.cols_pairs_both_null_dict

    def get_sample_col_pair_both_null_dict(self, force=False):
        if self.sample_cols_pairs_both_null_dict:
            return self.sample_cols_pairs_both_null_dict
        self.sample_cols_pairs_both_null_dict = {}
        _, pairs = self._get_column_pairs_unique(force=force)
        if pairs is None:
            return None
        for col_name_a, col_name_b in pairs:
            pairs_tuple = tuple(sorted([col_name_a, col_name_b]))
            match_arr = (self.sample_df[col_name_a].isna() & self.sample_df[col_name_b].isna())
            self.sample_cols_pairs_both_null_dict[pairs_tuple] = match_arr
        return self.sample_cols_pairs_both_null_dict

    def get_col_pairs_either_null_bool_dict(self, force=False):
        """
        Similar to get_col_pair_both_null_dict(), but checks if either are null, not if both are, and contains a single
        boolean value for each pair of columns indicating True if there are at least 90% of the rows having either null.

        Set force=True if the results will not be used to loop through tests, only to create a dictionary for reference.
        """
        if self.col_pairs_either_null_bool_dict:
            return self.col_pairs_either_null_bool_dict
        self.col_pairs_either_null_bool_dict = {}
        threshold = self.num_rows * 0.9
        _, pairs = self._get_column_pairs_unique(force=force)
        if pairs is None:
            return None
        for col_name_a, col_name_b in pairs:
            pairs_tuple = tuple(sorted([col_name_a, col_name_b]))
            match_arr = (self.orig_df[col_name_a].isna() | self.orig_df[col_name_b].isna())
            self.col_pairs_either_null_bool_dict[pairs_tuple] = match_arr.tolist().count(True) > threshold
        return self.col_pairs_either_null_bool_dict

    def get_col_triples_any_null_bool_dict(self):
        """
        Similar to get_col_pairs_either_null_bool_dict(), but checks triples of numeric columns and triples where one
        is binary and two are numeric. Each element contains a single boolean value for each pair of columns indicating
        True if there are at least 10% of the rows having no nulls.
        """
        if self.col_triples_all_null_bool_dict:
            return self.col_triples_all_null_bool_dict
        self.col_triples_all_null_bool_dict = {}

        # Get triples of numeric columns
        num_triples, triples_arr = self._get_numeric_column_triples_unique()
        if triples_arr is None:
            return None

        # Get triples where one is binary and two are numeric
        for bin_col in self.binary_cols:
            num_pairs, pairs = self._get_numeric_column_pairs_unique()
            for col_name_1, col_name_2 in pairs:
                triples_arr.append(tuple(sorted([bin_col, col_name_1, col_name_2])))

        # Examine each triple
        threshold = self.num_rows * 0.9
        for triple in triples_arr:
            triple = list(triple)
            self.col_triples_all_null_bool_dict[tuple(sorted(triple))] = \
                self.orig_df[triple].isna().sum(axis=1).tolist().count(True) > threshold

        return self.col_triples_all_null_bool_dict

    def get_common_values_dict(self):
        """
        For each string column, find the set of common values.
        """

        if self.common_vals_dict:
            return self.common_vals_dict
        self.common_vals_dict = {}
        for col_name in self.string_cols:
            common_values = []
            vc = self.orig_df[col_name].value_counts()
            # Skip columns that have many unique values, as all combinations will be somewhat rare.
            if len(vc) < math.sqrt(self.num_rows):
                for v in vc.index:
                    # Ensure the value occurs frequently, but not the majority of the column
                    if (vc[v] > math.sqrt(self.num_rows)) and (vc[v] > 100) and (vc[v] < self.num_rows * 0.75):
                        common_values.append(v)
            self.common_vals_dict[col_name] = common_values
        return self.common_vals_dict

    ##################################################################################################################
    # Internal methods to get sets of columns
    ##################################################################################################################

    def _get_column_pairs_unique(self, force=False):
        """
        Returns a set of all pairs of columns A & B, other than where A == B. This will return both
        (A,B) and (B,A). This may be used to test, for example, where A >> B and B >> A.
        """
        # Check if there would be too many pairs.
        num_pairs = math.comb(len(self.orig_df.columns), 2)
        if (not force) and (num_pairs > self.max_combinations):
                return num_pairs, None

        pairs_arr = []
        for col_idx_1 in range(len(self.orig_df.columns)-1):
            col_name_1 = self.orig_df.columns[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.orig_df.columns)):
                col_name_2 = self.orig_df.columns[col_idx_2]
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def _get_binary_column_pairs(self, same_vocabulary=True):
        # Check if there would be too many pairs.
        num_pairs = len(self.binary_cols) * (len(self.binary_cols) - 1)
        if num_pairs > self.max_combinations:
            return num_pairs, None

        pairs_arr = []
        for col_name_1 in self.binary_cols:
            col_1_vals = set(self.orig_df[col_name_1].unique())
            for col_name_2 in self.binary_cols:
                col_2_vals = set(self.orig_df[col_name_2].unique())
                if col_name_1 == col_name_2:
                    continue
                if same_vocabulary and len(col_1_vals.intersection(col_2_vals)) != 2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def _get_binary_column_pairs_unique(self, same_vocabulary=True, force=False):
        """
        if same_vocabulary is True, this returns only pairs of binary columns that contain the same 2 values.
        """

        # Check if there would be too many pairs. We can not say, though, if same_vocabulary is set True
        num_pairs = math.comb(len(self.binary_cols), 2)
        if (not same_vocabulary) and (not force) and (num_pairs > self.max_combinations):
            return num_pairs, None

        pairs_arr = []
        for col_idx_1 in range(len(self.binary_cols)-1):
            col_name_1 = self.binary_cols[col_idx_1]
            col_1_vals = set(self.column_unique_vals[col_name_1])
            for col_idx_2 in range(col_idx_1+1, len(self.binary_cols)):
                col_name_2 = self.binary_cols[col_idx_2]
                col_2_vals = set(self.column_unique_vals[col_name_2])
                if same_vocabulary and len(col_1_vals.intersection(col_2_vals)) != 2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return len(pairs_arr), pairs_arr

    def _get_numeric_column_pairs(self):
        """
        This behaves the same as __get_column_pairs(), but returns only pairs where both columns are numeric.
        """
        num_pairs = len(self.numeric_cols) * (len(self.numeric_cols) - 1)
        if num_pairs > self.max_combinations:
            return num_pairs, None

        pairs_arr = []
        for col_name_1 in self.numeric_cols:
            for col_name_2 in self.numeric_cols:
                if col_name_1 == col_name_2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def _get_numeric_column_pairs_unique(self, force=False):
        """
        Similar to __get_numeric_column_pairs(), but returns each unique pair; this will return (A,B), but not (B,A)
        """
        num_pairs = math.comb(len(self.numeric_cols), 2)
        if (not force) and (num_pairs > self.max_combinations):
            return num_pairs, None

        pairs_arr = []
        for col_idx_1 in range(len(self.numeric_cols)-1):
            col_name_1 = self.numeric_cols[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.numeric_cols)):
                col_name_2 = self.numeric_cols[col_idx_2]
                if col_name_1 == col_name_2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def _get_numeric_column_triples(self):
        num_triples = len(self.numeric_cols) * (len(self.numeric_cols) - 1) * (len(self.numeric_cols) - 2)
        if num_triples > self.max_combinations:
            return num_triples, None

        triples_arr = []
        for col_name_1 in self.numeric_cols:
            for col_name_2 in self.numeric_cols:
                if col_name_1 == col_name_2:
                    continue
                for col_name_3 in self.numeric_cols:
                    if col_name_3 in (col_name_1, col_name_2):
                        continue
                    triples_arr.append((col_name_1, col_name_2, col_name_3))
        return num_triples, triples_arr

    def _get_numeric_column_triples_unique(self):
        num_triples = math.comb(len(self.numeric_cols), 3)
        if num_triples > self.max_combinations:
            return num_triples, None

        triples_arr = []
        for col_ix_1 in range(len(self.numeric_cols)-2):
            col_name_1 = self.numeric_cols[col_ix_1]
            for col_ix_2 in range(col_ix_1 + 1, len(self.numeric_cols)-1):
                col_name_2 = self.numeric_cols[col_ix_2]
                for col_ix_3 in range(col_ix_2+1, len(self.numeric_cols)):
                    col_name_3 = self.numeric_cols[col_ix_3]
                    triples_arr.append((col_name_1, col_name_2, col_name_3))
        return num_triples, triples_arr

    def _get_string_column_pairs_unique(self, force=False):
        num_pairs = math.comb(len(self.string_cols), 2)
        if (not force) and (num_pairs > self.max_combinations):
            return num_pairs, None

        pairs_arr = []
        for col_idx_1 in range(len(self.string_cols)-1):
            col_name_1 = self.string_cols[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.string_cols)):
                col_name_2 = self.string_cols[col_idx_2]
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def _get_string_column_pairs(self):
        """
        This behaves the same as __get_column_pairs(), but returns only pairs where both columns are string.
        """
        num_pairs = len(self.string_cols) * (len(self.string_cols) - 1)
        if num_pairs > self.max_combinations:
            return num_pairs, None

        pairs_arr = []
        for col_name_1 in self.string_cols:
            for col_name_2 in self.string_cols:
                if col_name_1 == col_name_2:
                    continue
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    def _get_date_column_pairs_unique(self, force=False):
        num_pairs = math.comb(len(self.date_cols), 2)
        if (not force) and (num_pairs > self.max_combinations):
            return num_pairs, None

        pairs_arr = []
        for col_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_idx_1]
            for col_idx_2 in range(col_idx_1+1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_idx_2]
                pairs_arr.append((col_name_1, col_name_2))
        return num_pairs, pairs_arr

    ##################################################################################################################
    # Clear issues
    ##################################################################################################################
