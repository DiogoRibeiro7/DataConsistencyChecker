"""
BaseTestsMixin: Test methods for base tests.

This mixin contains 18 test methods organized by category.
Extracted from check_data_consistency.py for better code organization.
"""

from __future__ import annotations

import datetime
import math
import random
import string
import sys

import numpy as np
import pandas as pd
from sklearn import metrics, tree
from sklearn.metrics import f1_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_consistency_checker.checker_state import CheckerState
from data_consistency_checker.checker_utils import is_missing


class BaseTestsMixin(CheckerState):
    """
    Mixin class containing base tests methods.

    This class contains 18 methods for testing data consistency
    related to base operations.

    This is a mixin class designed to be used with multiple inheritance.
    It does not have an __init__ method and relies on the parent class
    to provide necessary attributes and methods.
    """

    def _generate_missing(self):
        """
        Patterns without exceptions: 'missing vals all' has consistently non-missing values.
        Patterns with exception: 'missing vals most' has consistently non-missing values, with the exception of a None.
        """
        self._add_synthetic_column('missing vals rand',
            [random.choice(['a', 'b', 'c', None]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('missing vals all',
            [random.choice(['a', 'b', 'c']) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('missing vals most',
            [random.choice(['a', 'b', 'c']) for _ in range(self.num_synth_rows-2)] + [None] + [np.nan])
        self._add_synthetic_column('missing vals most null',
            [None] * (self.num_synth_rows-1) + ['a'])


    def _check_missing(self, test_id):
        """
        Handling null values: This test specifically checks for null values.
        """

        single_test_results = {}
        for col_name in self.orig_df.columns:
            single_test_results[col_name] = self.orig_df[col_name].isna().sum()
        self.single_test_summary_dict[test_id] = single_test_results

        for col_name in self.orig_df.columns:
            null_arr = self.orig_df[col_name].apply(is_missing)
            non_null_arr = ~null_arr
            num_null = null_arr.sum()
            if (self.num_rows - self.freq_contamination_level) < num_null < self.num_rows:
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    null_arr,
                    "The column contains values that are consistently NULL"
                )
            if 0 <= num_null < self.freq_contamination_level:
                self._process_analysis_binary(
                    test_id,
                    [col_name],
                    non_null_arr,
                    "The column contains values that are consistently non-NULL"
                )


    def _generate_rare_values(self):
        """
        Patterns without exceptions: 'rare vals all' has consistently frequent values.
        Patterns with exception: 'rare vals most' has consistently frequent values, with the exception of 'z', a rare
            value.
        """
        self._add_synthetic_column('rare_vals rand',
            [random.choice(string.ascii_letters) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('rare_vals all',
            [random.choice(['a', 'b', 'c']) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('rare_vals most',
            [random.choice(['a', 'b', 'c']) for _ in range(self.num_synth_rows - 1)] + ['z'])


    def _check_rare_values(self, test_id):
        """
        Handling null values: This test does not flag null values. The rareness of values is measured based on their
        frequency, independent if the other values are null or non-null.

        This skips binary columns and columns with few unique values.
        """
        # todo: allow users to specify a set of common values, and just flag anything else. Or in the clear issues,
        #   allow to clear there.

        single_test_results = {}

        # Check all column types except binary, where there are often rare values.
        for col_name in self.numeric_cols + self.date_cols + self.string_cols:
            if self.orig_df[col_name].nunique() > math.log2(self.num_rows):
                continue

            counts_series = self.orig_df[col_name].astype(str).value_counts(normalize=False, dropna=False)
            all_rare_vals = [str(x) for x, y in zip(counts_series.index, counts_series.values)
                             if y < self.freq_contamination_level]
            non_null_rare_vals = [str(x) for x in all_rare_vals if not is_missing(x)]
            # It is not possible to sort None or NaN values
            rare_vals = sorted(non_null_rare_vals)
            for v in all_rare_vals:
                if v not in rare_vals:
                    rare_vals.append(v)

            all_common_vals = [str(x).strip() for x in counts_series.index if x not in rare_vals]
            non_null_common_vals = [str(x).strip() for x in all_common_vals if not is_missing(x)]
            common_vals = sorted(non_null_common_vals)
            for v in all_common_vals:
                if v not in common_vals:
                    common_vals.append(v)

            test_series = ~self.orig_df[col_name].isin(rare_vals)
            single_test_results[col_name] = test_series.tolist().count(False)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f"The column consistently contains a small set of common values: {common_vals}",
                f"The rare values: {rare_vals}",
                allow_patterns=False,
                display_info={"counts": counts_series}
            )

        self.single_test_summary_dict[test_id] = single_test_results


    def _generate_unique_values(self):
        """
        Patterns without exceptions: 'unique_vals all' has consistently unique values.
        Patterns with exception: 'unique_vals most' has consistently unique values, with the exception of 500, which
            appears twice.
        """
        self._add_synthetic_column('unique_vals rand', [random.randint(0, 250) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('unique_vals all', np.arange(0, self.num_synth_rows))
        self._add_synthetic_column('unique_vals most', np.arange(0, self.num_synth_rows-1).tolist() + [500])


    def _check_unique_values(self, test_id):
        """
        Handling null values: Null is considered a single, unique value, similar to any other value. Columns with up to
        one null value may be considered as having all unique values if the other values are also unique.

        This test does not check floating point columns, where small numbers of duplicate values are common. This
        test is useful for ID columns which should be unique, which are typically string or integer values. It will
        also check where datetime values should be unique.
        """

        single_test_results = {}

        for col_name in self.orig_df.columns:
            # The values are not considered to be all unique if more than 1 Null value is present
            if self.orig_df[col_name].isna().sum() > 1:
                continue

            # Test for floating point columns
            if col_name in self.numeric_cols:
                numeric_vals = self.numeric_vals_filled[col_name]
                numeric_vals = numeric_vals.fillna(0)
                if (numeric_vals.astype(int) == numeric_vals).tolist().count(False) > \
                        self.freq_contamination_level:
                    continue

            # Test on a sample first
            counts_arr = self.sample_df[col_name].value_counts()
            test_series = [counts_arr[x] == 1 for x in self.sample_df[col_name]]
            if test_series.count(False) > 1:
                continue

            counts_arr = self.orig_df[col_name].value_counts(dropna=False)
            repeated_vals = [x for x, y in zip(counts_arr.index, counts_arr.values) if y > 1]
            test_series = [counts_arr[x] == 1 for x in self.orig_df[col_name]]
            single_test_results[col_name] = test_series.count(False)
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                "The column contains values that are consistently unique",
                f" including {repeated_vals[:5]} ({len(repeated_vals)} identified repeated value(s))"
            )

        self.single_test_summary_dict[test_id] = single_test_results


    def _generate_prev_values_dt(self):
        """
        Patterns without exceptions:
            'pattern_history_df_str all_a' has a repeating pattern.
            'pattern_history_df_num all_a' has a repeating pattern.
        Patterns with exception:
            'pattern_history_df_str most_a' has a repeating pattern, with exceptions (a new value).
            'pattern_history_df_str most_b' has a repeating pattern, with exceptions (an existing value in the wrong
                location).
            'pattern_history_df_num most_a' has a repeating pattern, with one exception
        Columns not flagged:
            'pattern_history_df_num not_1' has a trend, but it is no stronger than predicting the previous value
            'pattern_history_df_num not_2' has a trend, but it is no stronger than predicting the previous value
        """
        # Categorical test cases - repeating values
        self._add_synthetic_column('pattern_history_df_str all_a',
                                    ['a', 'a', 'b', 'c', 'd'] * (self.num_synth_rows // 5))
        self._add_synthetic_column('pattern_history_df_str most_a',
                                    ['a', 'a', 'b', 'c', 'd'] * (self.num_synth_rows // 5))
        self.synth_df.loc[999, 'pattern_history_df_str most_a'] = 'x'
        self._add_synthetic_column('pattern_history_df_str most_b',
                                    ['a', 'b', 'c', 'd'] * (self.num_synth_rows // 4))
        self.synth_df.loc[999, 'pattern_history_df_str most_b'] = 'a'

        # Numeric test cases - repeating values
        self._add_synthetic_column('pattern_history_df_num all_a', [10, 20, 30, 40] * (self.num_synth_rows // 4))
        self._add_synthetic_column('pattern_history_df_num most_a', [10, 20, 30, 40] * (self.num_synth_rows // 4))
        self.synth_df.loc[999, 'pattern_history_df_num most_a'] = 10

        # Numeric test cases - trend
        # The DT can't beat just predicting the previous value
        self._add_synthetic_column('pattern_history_df_num not_1', np.arange(self.num_synth_rows))
        self._add_synthetic_column('pattern_history_df_num not_2', np.arange(self.num_synth_rows))


    def _check_prev_values_dt(self, test_id):
        """
        This checks binary, string, and numeric columns. It currently does test date columns.

        Handling null values: null values are treated as any other value, both in the lag values and the predicted
        values.

        This test cannot test the first 10 rows, as the test requires 10 prior rows of history.

        The test ensures that any patterns found by decision trees are more predictive than predicting the previous
        value or predicting the mode/median for the column.
        """

        look_back_range = 10  # The number of previous values examined to predict the current value

        # Set the seed to ensure the DT behaves the same each execution of this test
        random.seed(0)
        np.random.seed(0)

        for col_name in self.orig_df.columns:
            df = pd.DataFrame()
            df[col_name] = self.orig_df[col_name]
            if col_name in self.numeric_cols:
                df[col_name] = self.numeric_vals_filled[col_name].astype(float).fillna(sys.maxsize)

            # Create the lag features
            for i in range(1, look_back_range + 1):
                df[f"Lag_{i}"] = df[col_name].shift(i)

            # For any simple DTs, we should be able to train on a fairly small set of rows. This is done to ensure
            # the pattern is simple and for speed. We skip the first set of rows, as they do not have the lag values.
            # The sample is set smaller than the synthetic data size to provide increased testing.
            df_sample = df.iloc[look_back_range:].copy()
            df_sample = df_sample.sample(n=min(len(df_sample), 900), random_state=0)

            n_unique_vals = self.orig_df[col_name].nunique()

            x_train = df_sample.drop(columns=[col_name])
            y_lag_1 = df_sample['Lag_1']
            y_train = df_sample[col_name]

            x_train = x_train.replace([np.inf, -np.inf], sys.maxsize)
            y_lag_1 = y_lag_1.replace([np.inf, -np.inf], sys.maxsize)
            y_train = y_train.replace([np.inf, -np.inf], sys.maxsize)
            x_train = x_train.fillna(sys.maxsize)
            y_lag_1 = y_lag_1.fillna(sys.maxsize)
            y_train = y_train.fillna(sys.maxsize)

            if col_name in self.string_cols or col_name in self.binary_cols or n_unique_vals <= look_back_range:
                if self.orig_df[col_name].nunique() > look_back_range:
                    continue

                # Do not flag rare values, as there is another test for that.
                rare_values = []
                for v in self.orig_df[col_name].unique():
                    if (v != v) or (self.orig_df[col_name].tolist().count(v) < self.freq_contamination_level):
                        rare_values.append(v)

                # Create a binary (one-hot) column for each value for each lag. Sklearn decision trees can not work
                # with non-numeric values.
                x_train = pd.get_dummies(x_train, columns=x_train.columns)

                # Ensure the Y column is in a consistent format, using ordinal values
                encode_dict = dict(zip(y_train.unique(), range(y_train.nunique())))
                decode_dict = {y: x for x, y in zip(y_train.unique(), range(y_train.nunique()))}
                y_train_numeric = y_train.map(encode_dict)
                y_lag_1_numeric = y_lag_1.map(encode_dict)

                # Allow the DT to create a rule for each value. Each rule may include multiple features in the
                # decision path.
                clf = DecisionTreeClassifier(max_leaf_nodes=max(2, self.orig_df[col_name].nunique()), random_state=0)
                clf.fit(x_train, y_train_numeric)

                # Predict on a smaller sample first
                y_pred = clf.predict(x_train)
                # We do not use macro, as some rare classes may have poor f1 scores even for good trees.
                f1 = f1_score(y_train_numeric, y_pred, average='micro')
                if f1 < 0.9:
                    continue

                # Test simply predicting the most common value
                f1_naive = f1_score(y_train_numeric, [y_train_numeric.mode()[0]] * len(y_pred), average='micro')
                if f1_naive >= (f1 - 0.05):
                    continue

                # Test simply predicting the previous value
                y_train_numeric = y_train_numeric.fillna(y_train_numeric.mode())
                y_lag_1_numeric = y_lag_1_numeric.fillna(y_lag_1_numeric.mode())
                y_train_numeric = y_train_numeric.replace([np.inf, -np.inf, None, np.nan, pd.NaT], y_train_numeric.mode()[0])
                y_lag_1_numeric = y_lag_1_numeric.replace([np.inf, -np.inf, None, np.nan, pd.NaT], y_lag_1_numeric.mode()[0])
                f1_naive = f1_score(y_train_numeric, y_lag_1_numeric, average='micro')
                if f1_naive >= (f1 - 0.05):
                    continue

                # Once we establish the accuracy on roughly 1000 rows, predict for the full column. Use as few lags
                # as possible to get a good score.
                for num_lags in range(1, 11):
                    cols_used = [col_name]
                    for i in range(num_lags):
                        cols_used.append(f'Lag_{i+1}')

                    test_df = df.iloc[look_back_range:][cols_used]
                    x_test = test_df.drop(columns=[col_name])
                    y_test = test_df[col_name]

                    x_test = x_test.replace([np.inf, -np.inf], sys.maxsize)
                    y_test = y_test.replace([np.inf, -np.inf], sys.maxsize)
                    x_test = x_test.fillna(sys.maxsize)
                    y_test = y_test.fillna(sys.maxsize)

                    x_test = pd.get_dummies(x_test, columns=x_test.columns)
                    y_test_numeric = y_test.map(encode_dict)

                    clf = DecisionTreeClassifier(max_leaf_nodes=max(2, self.orig_df[col_name].nunique()), random_state=0)
                    clf.fit(x_test, y_test_numeric)
                    y_pred = clf.predict(x_test)

                    f1 = f1_score(y_test_numeric, y_pred, average='micro')
                    if f1 >= 0.9:
                        break

                test_series = [x == y or x in rare_values for x, y in zip(y_test_numeric, y_pred)]
                test_series = [True]*look_back_range + test_series
                rules = tree.export_text(clf)
            elif col_name in self.numeric_cols:  # todo: try to cover date columns here too
                # Skip columns that have non-numeric values. In the future, we may support these as well.
                if len(self.numeric_vals[col_name]) < self.num_rows:
                    continue

                # Skip columns that have very little variance
                q1 = self.numeric_vals[col_name].quantile(0.25)
                med = self.numeric_vals[col_name].quantile(0.50)
                q3 = self.numeric_vals[col_name].quantile(0.75)
                if med != 0:
                    if ((q3 - q1) / self.numeric_vals[col_name].quantile(0.5)) < 0.5:
                        continue
                else:
                    if (q1 == med) or (q3 == med):
                        continue

                # We do not encode/decode values with a numeric column.
                decode_dict = None

                # Get the normalized MAE & R2 when using a decision tree
                regr = DecisionTreeRegressor(max_leaf_nodes=look_back_range, random_state=0)
                regr.fit(x_train, y_train)

                # Predict on a smaller sample first
                y_pred = regr.predict(x_train)
                mae = metrics.median_absolute_error(y_train, y_pred)
                norm_mae = abs(mae / self.column_medians[col_name]) if self.column_medians[col_name] != 0 else np.inf
                if norm_mae > 0.1:
                    continue

                # Get the normalized MAE when simply predicting the median
                naive_mae = metrics.median_absolute_error(y_train.astype(float),
                                                          [y_train.astype(float).quantile(0.5)] * len(y_train))
                if mae > (naive_mae * 0.5):
                    continue

                # Test simply predicting the previous value
                y_prev = y_train.shift(1)
                # Element 0 will have NaN, so cannot be evaluated.
                naive_mae = metrics.median_absolute_error(y_train[1:], y_prev[1:])
                if mae > (naive_mae * 0.5):
                    continue

                # Once we establish the accuracy on roughly 1000 rows, predict for the full column. Use as few lags
                # as possible.
                found = False
                for num_lags in range(1, 11):
                    cols_used = [col_name]
                    for i in range(num_lags):
                        cols_used.append(f'Lag_{i+1}')

                    test_df = df.iloc[look_back_range:][cols_used]
                    x_test = test_df.drop(columns=[col_name])
                    y_test = test_df[col_name]

                    regr = DecisionTreeRegressor(max_leaf_nodes=look_back_range, random_state=0)
                    regr.fit(x_test, y_test)
                    y_pred = regr.predict(x_test)

                    # Ensure both the NRMSE and the R2 score are strong
                    mae = metrics.median_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    if self.column_medians[col_name] != 0:
                        norm_mae = abs(mae / self.column_medians[col_name])
                    else:
                        norm_mae = np.inf
                    if (r2 > 0.9) and (norm_mae <= 0.1):
                        found = True
                        break

                if not found:
                    continue

                test_series = [True if y == 0 else (x/y) < (y / 10.0) for x, y in zip(y_test, y_pred)]
                test_series = [True]*look_back_range + test_series

                # test_series = [True if y == 0 else (x/y) < (y / 10.0) for x, y in zip(y_train, y_pred)]
                test_series = [True]*look_back_range + test_series
                rules = tree.export_text(regr)
            else:
                continue  # todo: can remove once support date columns

            # The export of the rules has the column names in the format 'feature_1' and so on. We replace these with
            # the actual column names. These are cleaned further below.
            cols = []
            for c_idx, c_name in enumerate(x_train.columns):
                rule_col_name = f'feature_{c_idx} '
                if rule_col_name in rules:
                    cols.append(c_name)
                    rules = rules.replace(rule_col_name, c_name + ' ')

            # We map the numeric values in the target column back to their original values
            if decode_dict:
                for v in decode_dict:
                    rules = rules.replace(f"class: {v}", f"value: {decode_dict[v]}")

            # todo: this is copied to clean_dt_splitpoints(). Call that instead.
            # We map the split points (in the form of Lag_1_[value] <= 0.50) to a more readable format
            if decode_dict is None:  # Regression
                for i in range(look_back_range):
                    rules = rules.replace(f"Lag_{i}", f"The value {i} rows previously")
            else:  # Classification
                for i in range(look_back_range):
                    for v in decode_dict:
                        val = decode_dict[v]
                        rules = rules.replace(f"Lag_{i}_{val} <= 0.50", f"The value {i} rows previously was not '{val}'")
                        rules = rules.replace(f"Lag_{i}_{val} >  0.50", f"The value {i} rows previously was '{val}'")
                        # In some cases, '.0' is added to the end of the vals
                        rules = rules.replace(f"Lag_{i}_{val}.0 <= 0.50", f"The value {i} rows previously was not '{val}'")
                        rules = rules.replace(f"Lag_{i}_{val}.0 >  0.50", f"The value {i} rows previously was '{val}'")

            # Correct grammar for single row case
            rules = rules.replace('value 1 rows previously', 'value 1 row previously')

            # Check the tree is not trivial in that it has the same prediction in each leaf (the most common value)
            # todo: add this check for regression as well
            rules_lines = rules.split('\n')
            tree_preds = set()
            for line in rules_lines:
                if "class: " in line:
                    tree_preds.add(line.split('class: ')[1])
            if len(tree_preds) == 1:
                continue

            pred_series = pd.Series(self.orig_df[col_name].iloc[:10].tolist() + y_pred.tolist())
            if decode_dict:
                pred_series = pred_series.map(decode_dict)
            self._process_analysis_binary(
                test_id,
                [col_name],
                np.array(test_series),
                (f"The values in {col_name} can consistently be predicted from the previous values in the column "
                 f"with a decision tree using the following rules: \n\n{rules}"),
                display_info={'Pred': pred_series}
                )

    ##################################################################################################################
    # Data consistency checks for pairs of columns of any type
    ##################################################################################################################


    def _generate_matched_missing(self):
        """
        Patterns without exceptions: 'matched_missing_vals rand_a' and 'matched_missing_vals all' have Null values in
            the same rows.
        Patterns with exception: 'matched_missing_vals rand_a' and 'matched_missing_vals most' have Null values in the
            same rows, with 1 exception.
        """
        self._add_synthetic_column('matched_missing_vals rand_a',
                                    [random.choice(['a', 'b', 'c', None]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched_missing_vals rand_b',
                                    [random.choice(['a', 'b', 'c', None]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('matched_missing_vals all',
                                    self.synth_df['matched_missing_vals rand_a'])
        self._add_synthetic_column('matched_missing_vals most', self.synth_df['matched_missing_vals rand_a'])
        if self.synth_df.loc[999, 'matched_missing_vals most'] is None:
            self.synth_df.loc[999, 'matched_missing_vals most'] = 'a'
        else:
            self.synth_df.loc[999, 'matched_missing_vals most'] = None


    def _check_matched_missing(self, test_id):
        """
        Handling null values: This test specifically checks for null and non-null values. It requires a minimal number
        of both null and non-null values in each pair of columns examined.
        """

        is_missing_dict = self.get_is_missing_dict()

        for col_idx_1 in range(len(self.orig_df.columns)-1):
            col_name_1 = self.orig_df.columns[col_idx_1]
            col_1_missing_arr = is_missing_dict[col_name_1].tolist()
            num_missing_1 = col_1_missing_arr.count(True)
            if self.verbose >= 2 and col_idx_1 > 0 and col_idx_1 % 100 == 0:
                print(f"  Examining column {col_idx_1} of {len(self.orig_df.columns)} columns")
            if num_missing_1 < self.freq_contamination_level:
                continue
            if num_missing_1 > (self.num_rows - self.freq_contamination_level):
                continue

            for col_idx_2 in range(col_idx_1 + 1, len(self.orig_df.columns)):
                col_name_2 = self.orig_df.columns[col_idx_2]
                col_2_missing_arr = is_missing_dict[col_name_2].tolist()
                num_missing_2 = col_2_missing_arr.count(True)
                if num_missing_2 < self.freq_contamination_level:
                    continue
                if num_missing_2 > (self.num_rows - self.freq_contamination_level):
                    continue

                # If the difference between the number of missing values is too large, there is not a pattern.
                # For example, if col 1 has 400 missing and col 2 has 300, the difference is 100. So, there are at
                # least 100 rows where col 1 has Null and col 2 does not. Assuming this is greater than the
                # contamination level (probably set to between 1 and 50), there can not be a match.
                if abs(num_missing_1 - num_missing_2) > self.freq_contamination_level:
                    continue

                test_series = [x == y for x, y in zip(col_1_missing_arr, col_2_missing_arr)]

                # Determine if there is already a pattern found, which these columns are part of. If so, simply add
                # the new column to the pattern.
                found_existing_pattern = False
                if test_series.count(False) == 0:
                    for pattern_idx, existing_pattern in enumerate(self.patterns_arr):
                        if existing_pattern[0] != test_id:
                            continue
                        pattern_cols = [x.lstrip('"').rstrip('"') for x in existing_pattern[1].split(" AND ")]
                        if col_name_1 in pattern_cols or col_name_2 in pattern_cols:
                            if f'"{col_name_1}"' not in existing_pattern[1]:
                                existing_pattern[1] += f' AND "{col_name_1}"'
                            if f'"{col_name_2}"' not in existing_pattern[1]:
                                existing_pattern[1] += f' AND "{col_name_2}"'
                            existing_pattern[2] = (f'The columns consistently have missing values in the same rows, '
                                                   f'with {num_missing_1} missing values.')
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
                    (f'The columns "{col_name_1}" (with {num_missing_1} Null values) and "{col_name_2}" (with '
                     f'{num_missing_2} Null values) consistently have missing values in the same rows.')
                )


    def _generate_opposite_missing(self):
        """
        Patterns without exceptions: 'opposite_missing_vals rand_a', 'opposite_missing_vals all_1' have Null values
            strictly different rows. 'opposite_missing_vals all_2' matches as well, creating a pattern with 3
            columns.
        Patterns with exception: 'opposite_missing_vals rand_a' and 'opposite_missing_vals most' have Null values in
            consistently different rows, with 1 exception.
        """
        self._add_synthetic_column('opposite_missing_vals rand_a', ['a'] * 500 + [None] * (self.num_synth_rows - 500))
        self._add_synthetic_column('opposite_missing_vals rand_b',
                                    [random.choice(['a', 'b', 'c', None]) for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('opposite_missing_vals all_1', [None] * (self.num_synth_rows - 500) + ['a'] * 500)
        self._add_synthetic_column('opposite_missing_vals all_2', [None] * (self.num_synth_rows - 500) + ['b'] * 500)
        self._add_synthetic_column('opposite_missing_vals most', self.synth_df['opposite_missing_vals all_1'])
        if self.synth_df.loc[999, 'opposite_missing_vals most'] is None:
            self.synth_df.loc[999, 'opposite_missing_vals most'] = 'a'
        else:
            self.synth_df.loc[999, 'opposite_missing_vals most'] = None


    def _check_opposite_missing(self, test_id):
        """
        Handling null values: This test specifically checks for null and non-null values. It requires a minimal number
        of both null and non-null values in each pair of columns examined.
        """

        is_missing_dict = self.get_is_missing_dict()

        for col_idx_1 in range(len(self.orig_df.columns)-1):
            col_name_1 = self.orig_df.columns[col_idx_1]
            col_1_missing_arr = is_missing_dict[col_name_1].tolist()
            num_missing_1 = col_1_missing_arr.count(True)
            if self.verbose >= 2 and col_idx_1 > 0 and col_idx_1 % 100 == 0:
                print(f"  Examining column {col_idx_1} of {len(self.orig_df.columns)} columns")
            if num_missing_1 < (self.num_rows * 0.1):
                continue
            if num_missing_1 > (self.num_rows * 0.9):
                continue
            for col_idx_2 in range(col_idx_1 + 1, len(self.orig_df.columns)):
                col_name_2 = self.orig_df.columns[col_idx_2]
                col_2_missing_arr = is_missing_dict[col_name_2].tolist()
                num_missing_2 = col_2_missing_arr.count(True)
                if num_missing_2 < (self.num_rows * 0.1):
                    continue
                if num_missing_2 > (self.num_rows * 0.9):
                    continue

                # If the sum of the number of missing values is too large, there is not a pattern. For example, if
                # the dataset has 1000 rows and column 1 has 600 missing and column 2 has 700 missing, then there must
                # be many rows where they both have Null values.
                if abs(num_missing_1 + num_missing_2) > (self.num_rows + self.freq_contamination_level):
                    continue

                test_series = [x != y for x, y in zip(col_1_missing_arr, col_2_missing_arr)]

                # Determine if there is already a pattern found, which these columns are part of. If so, simply add
                # the new column to the pattern.
                found_existing_pattern = False
                if test_series.count(False) == 0:
                    for pattern_idx, existing_pattern in enumerate(self.patterns_arr):
                        if existing_pattern[0] != test_id:
                            continue
                        pattern_cols = [x.lstrip('"').rstrip('"') for x in existing_pattern[1].split(" AND ")]
                        if col_name_1 in pattern_cols or col_name_2 in pattern_cols:
                            if f'"{col_name_1}"' not in existing_pattern[1]:
                                existing_pattern[1] += f' AND "{col_name_1}"'
                            if f'"{col_name_2}"' not in existing_pattern[1]:
                                existing_pattern[1] += f' AND "{col_name_2}"'
                            existing_pattern[2] = (f'The columns consistently have missing values in the different rows, '
                                                   f'with {num_missing_1} or {num_missing_2} missing values.')
                            self.patterns_arr[pattern_idx] = existing_pattern #  ff

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
                    (f'The columns "{col_name_1}" (with {num_missing_1} Null values) and "{col_name_2}" (with '
                     f'{num_missing_2} Null values) consistently have missing values in different rows.')
                )


    def _generate_same(self):
        """
        Patterns without exceptions: 'same rand' and 'same all' are consistently the same
        Patterns with exception: 'same rand' and 'same most' are consistently the same with 1 exception. Similarly for
            "same all" AND "same most".
            'same rand_date' and 'same most_date' are consistently the same with 1 exception.
        Not matching: 'same null_a' and 'same null_b' are mostly the same, but only because they are mostly Null.
            These should not be flagged.
        """
        self._add_synthetic_column('same rand', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('same all',  self.synth_df['same rand'])
        self._add_synthetic_column('same most', self.synth_df['same rand'])
        self.synth_df.loc[999, 'same most'] = 2.1

        dates_arr = []
        for _ in range(self.num_synth_rows):
            d = np.random.randint(1, 25)
            dates_arr.append(datetime.datetime.strptime(f"{d}-01-2022", "%d-%m-%Y"))
        self._add_synthetic_column('same rand_date', dates_arr)
        self._add_synthetic_column('same most_date', dates_arr)
        self.synth_df.loc[999, 'same most_date'] = datetime.datetime.strptime("26-01-2022", "%d-%m-%Y")

        self._add_synthetic_column('same null_a', [None] * self.num_synth_rows)
        self.synth_df.loc[0, 'same null_a'] = 1.0
        self.synth_df.loc[1, 'same null_a'] = 1.0
        self._add_synthetic_column('same null_b', [None] * self.num_synth_rows)
        self.synth_df.loc[1, 'same null_b'] = 2.0


    def _check_same(self, test_id):
        """
        Handling null values: Matches if both values are Null. If one value is Null and the other is not, does not
        count as a match.
        """

        def test_arrs(arr1, arr2, is_sample):
            if is_missing_dict[col_name_1].sum() > missing_limit:
                return False
            if is_missing_dict[col_name_2].sum() > missing_limit:
                return False
            if count_most_freq_value_dict[col_name_1] > most_freq_limit:
                return False
            if count_most_freq_value_dict[col_name_2] > most_freq_limit:
                return False

            test_series = [x == y or (is_missing(x) and is_missing(y)) for x, y in zip(arr1, arr2)]
            if is_sample:
                return test_series.count(False) < 1
            self._process_analysis_binary(
                test_id,
                [col_name_1, col_name_2],
                test_series,
                f'The values in "{col_name_2}" are consistently the same as those in "{col_name_1}"')
            return None

        def test_pair():
            if not test_arrs(self.sample_df[col_name_1], self.sample_df[col_name_2], is_sample=True):
                return
            test_arrs(self.orig_df[col_name_1], self.orig_df[col_name_2], is_sample=False)

        count_most_freq_value_dict = self.get_count_most_freq_value_dict()
        is_missing_dict = self.get_is_missing_dict()
        missing_limit = self.num_rows * 0.75
        most_freq_limit = self.num_rows * 0.99

        # Numeric columns
        num_pairs, pairs = self._get_numeric_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping testing pairs of numeric columns. There are {num_pairs:,} pairs. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
        else:
            for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):  # noqa: B007 - read by the nested check function
                if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                    print(f"  Examining pair number {pair_idx:,} of {num_pairs:,} pairs of numeric columns")
                test_pair()

        # Binary columns
        num_pairs, pairs = self._get_binary_column_pairs_unique(same_vocabulary=True)
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping testing pairs of binary columns. There are {num_pairs:,} pairs. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
        else:
            for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):  # noqa: B007 - read by the nested check function
                if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                    print(f"  Examining pair number {pair_idx:,} of {num_pairs:,} pairs of binary columns")
                test_pair()

        # String columns
        num_pairs, pairs = self._get_string_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping testing pairs of string columns. There are {num_pairs:,} pairs. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
        else:
            for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):  # noqa: B007 - read by the nested check function
                if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                    print(f"  Examining pair number {pair_idx:,} of {num_pairs:,} pairs of string columns")
                test_pair()

        # Date columns
        num_pairs, pairs = self._get_date_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping testing pairs of date columns. There are {num_pairs:,} pairs. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
        else:
            for pair_idx, (col_name_1, col_name_2) in enumerate(pairs):  # noqa: B007 - read by the nested check function
                if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                    print(f"  Examining pair number {pair_idx:,} of {num_pairs:,} pairs of date columns")
                test_pair()


    def _generate_same_or_constant(self):
        """
        Patterns without exceptions: 'same_or_const all' consistently has either the same value as 'same_or_const rand'
            or a constant value, 34.5
        Patterns with exception: 'same_or_const most' usually has either the same value as 'same_or_const rand'
            or a constant value, 34.5, with the exception of row 999
        """
        self._add_synthetic_column('same_or_const rand', [random.random() - 0.5 for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('same_or_const all',
                                    [x if y % 2 == 0 else 34.5  for x, y in
                                     zip(self.synth_df['same_or_const rand'], range(self.num_synth_rows))])
        self._add_synthetic_column('same_or_const most', self.synth_df['same_or_const all'].copy())
        self.synth_df.loc[999, 'same_or_const most'] = 2.1


    def _check_same_or_constant(self, test_id):
        """
        For each pair of columns we check both directions, if A is either the same as B or a small set of other values,
        or if B is either the same as A or a small set of other values.
        """

        def test_arrs(arr1, arr2, is_sample):
            """
            This may be run on either a sample of the full data or the full data. If run on a sample, we check if the
            two arrays appear to exhibit the pattern and return True if so and False otherwise. If True, this will
            be called again on the full dataset, which may save a pattern or exception if the pattern is true for
            the majority of rows.
            """
            # Skip columns that are largely Null
            if arr1.isna().sum() > (len(arr1) * 0.75):
                return False
            if arr2.isna().sum() > (len(arr2) * 0.75):
                return False

            # Skip columns that have few unique values
            if is_sample:
                if arr1.nunique() < math.sqrt(len(arr1)):
                    return False
                if arr2.nunique() < math.sqrt(len(arr2)):
                    return False
            else:
                if arr1.nunique() < math.sqrt(self.num_rows):
                    return False
                if arr2.nunique() < math.sqrt(self.num_rows):
                    return False

            same_indicator = [x == y or (is_missing(x) and is_missing(y)) for x, y in zip(arr1, arr2)]

            # Exclude column pairs which are not the same value in at least 10% of rows
            if same_indicator.count(True) < (len(arr1) * 0.1):
                return False

            # Exclude column pairs which are the same value in over 95% of rows
            if same_indicator.count(True) > (len(arr1) * 0.95):
                return False

            other_values_1 = pd.Series([np.nan if y == 1 else x for x, y in zip(arr1, same_indicator)])
            other_values_2 = pd.Series([np.nan if y == 1 else x for x, y in zip(arr2, same_indicator)])
            vc_1 = other_values_1.value_counts()
            vc_2 = other_values_2.value_counts()
            if (len(vc_1) <= 5) and (arr1.nunique() >= 10):
                common_alternatives = [x for x, y in zip(vc_1.index, vc_1.values) if y > self.freq_contamination_level]
                col_values = np.array([bool(x == 1 or y in common_alternatives)
                                       for x, y in zip(same_indicator, arr1)])
                if is_sample:
                    return col_values.tolist().count(False) <= 1
                self._process_analysis_binary(
                    test_id,
                    [col_name_2, col_name_1],
                    col_values,
                    (f'The values in "{col_name_1}" are consistently either the same as those in "{col_name_2}", '
                     f'or one of {common_alternatives}'))
            elif (len(vc_2) <= 5) and (arr2.nunique() >= 10):
                common_alternatives = [x for x, y in zip(vc_2.index, vc_2.values) if y > self.freq_contamination_level]
                col_values = np.array([bool(x == 1 or y in common_alternatives)
                                       for x, y in zip(same_indicator, arr2)])
                if is_sample:
                    return col_values.tolist().count(False) <= 1
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    col_values,
                    (f'The values in "{col_name_2}" are consistently either the same as those in "{col_name_1}", '
                     f'or one of {common_alternatives}'))
            else:
                return False
            return None

        num_pairs, col_pairs = self._get_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print(f"  Skipping test. There are {num_pairs:,} pairs of columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}.")
            return

        # For this test, we do not use self.sample_df, as it has the extreme values removed, which may be preferable
        # to keep for this test
        sample_df = self.orig_df.sample(n=min(len(self.orig_df), 50), random_state=0)

        # Determine and cache the fraction of the column of the most frequent value per column
        most_freq_per_col = {}
        for col_name in self.orig_df.columns:
            vc = self.orig_df[col_name].value_counts(normalize=True)
            most_freq_per_col[col_name] = vc.sort_values(ascending=False).values[0]

        for cols_idx, (col_name_1, col_name_2) in enumerate(col_pairs):
            if self.verbose >= 2 and cols_idx > 0 and cols_idx % 10_000 == 0:
                print(f"  Examining column set {cols_idx:,} of {len(col_pairs):,} combinations of columns.")

            # Skip binary columns
            if col_name_1 in self.binary_cols or col_name_2 in self.binary_cols:
                continue

            # Skip cases where one column is almost entirely a single value.
            if most_freq_per_col[col_name_1] > 0.99:
                continue
            if most_freq_per_col[col_name_2] > 0.99:
                continue

            if not test_arrs(sample_df[col_name_1], sample_df[col_name_2], is_sample=True):
                continue
            test_arrs(self.orig_df[col_name_1], self.orig_df[col_name_2], is_sample=False)


    def _generate_unique_pair(self):
        """
        Patterns without exceptions: all_1 and all_2 have unique pairs of values
        Patterns with exception: all_1 and most have unique pairs of values other than the last 2 rows
        """
        self._add_synthetic_column('unique_pair rand', [random.random() for _ in range(self.num_synth_rows)])
        self._add_synthetic_column('unique_pair all_1',  [1]*500 + [2]*(self.num_synth_rows - 500))
        self._add_synthetic_column('unique_pair all_2', list(range(500)) + list(range(self.num_synth_rows - 500)))
        self._add_synthetic_column('unique_pair most', self.synth_df['unique_pair all_2'])
        self.synth_df.loc[999, 'unique_pair most'] = self.synth_df.loc[998, 'unique_pair most']


    def _check_unique_pair(self, test_id):
        nunique_dict = self.get_nunique_dict()
        num_missing_dict = self.get_num_missing_dict()

        nunique_threshold = (self.num_rows / 2)
        nunique_sample_pairs_threshold = len(self.sample_df) * 0.9
        missing_threshold = self.num_rows * 0.1
        min_number_combinations = self.num_rows - self.freq_contamination_level

        sample_df = self.sample_df.copy().fillna('NONE')
        for col_name_1_idx, col_name_1 in enumerate(self.orig_df.columns):
            if num_missing_dict[col_name_1] > missing_threshold:
                continue
            if nunique_dict[col_name_1] < 2:
                continue
            if nunique_dict[col_name_1] > nunique_threshold:
                continue
            for col_name_2 in self.orig_df.columns[col_name_1_idx + 1:]:
                if num_missing_dict[col_name_2] > missing_threshold:
                    continue
                if nunique_dict[col_name_2] < 2:
                    continue
                if nunique_dict[col_name_2] > nunique_threshold:
                    continue
                if nunique_dict[col_name_1] * nunique_dict[col_name_2] < min_number_combinations:
                    continue

                counts_arr = sample_df[[col_name_1, col_name_2]].value_counts(dropna=False)
                if len(counts_arr) < nunique_sample_pairs_threshold:
                    continue
                df = self.orig_df[[col_name_1, col_name_2]].copy().fillna('NONE')
                counts_arr = df.fillna('NONE').value_counts(dropna=False)
                repeated_vals = [x for x, y in zip(counts_arr.index, counts_arr.values) if y > 1]
                test_series = [counts_arr[x, y] == 1
                               for x, y in zip(df[col_name_1], df[col_name_2])]
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f"{col_name_1} contains {self.orig_df[col_name_1].nunique()} unique values and "
                     f"{col_name_2} contains {self.orig_df[col_name_2].nunique()} unique values. "
                     "The columns contain pairs of values that are consistently unique"),
                    f" including {repeated_vals[:5]} ({len(repeated_vals)} identified repeated value(s))"
                )

    ##################################################################################################################
    # Data consistency checks for single numeric columns
    ##################################################################################################################


