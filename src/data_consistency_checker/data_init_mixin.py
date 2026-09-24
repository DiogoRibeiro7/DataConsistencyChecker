"""Dataset initialization helpers for DataConsistencyChecker."""

from __future__ import annotations

import numbers
import statistics
import warnings

import numpy as np
import pandas as pd
import pandas.api.types as pandas_types

from .checker_utils import convert_to_numeric, is_missing, set_warnings_levels


class DataInitMixin:
    """Mixin providing dataset initialization and cache reset helpers."""

    def init_data(
        self,
        df: pd.DataFrame,
        known_date_cols: list[str] | None = None
    ) -> None:
        """
        Prepare data for quality checking. Must be called before check_data_quality().

        Args:
            df: DataFrame to be assessed for data quality
            known_date_cols: List of column names to treat as date columns.
                If None, date columns are auto-detected.

        Returns:
            None. Data is stored in instance variables.
        """

        self.orig_df = df.copy()

        # Check the dataframe does not contain duplicate column names. If it does, rename all columns with a _idx at
        # the end of each
        if len(df.columns) > len(set(df.columns)):
            df = df.copy()
            print(("Duplicate column names encountered. A suffix will be added to all column names to ensure they are "
                   f"unique. Duplicate column names: "
                   f"{set([x for x in df.columns if df.columns.tolist().count(x) > 1])}"))
            new_col_names = []
            for col_idx, col_name in enumerate(df.columns):
                new_col_names.append(f"{col_name}_{col_idx}")
            self.orig_df.columns = new_col_names

        # Ensure the dataframe has at least a very minimum number of rows.
        if len(df) < 10:
            print(f"The dataframe contains too few rows to be examined in a meaningful manner. Number of rows: "
                  f"{len(df)}")
            return None

        # Ensure the dataframe has a predictable index, which may be kept inline with parallel dataframes created
        # running the tests.
        self.orig_df = self.orig_df.reset_index(drop=True)
        self.num_rows = len(self.orig_df)
        self.num_valid_rows = None

        # Ensure the dataframe has column names in string format. Often dataframes contain numeric column names.
        self.orig_df.columns = [str(x) for x in self.orig_df.columns]

        # Remove any columns where there are not two or more unique values, or the values are all Null
        cols = [x for x in self.orig_df.columns if
                (self.orig_df[x].nunique(dropna=False) > 1) and (self.orig_df[x].isna().sum() < self.num_rows)]
        if self.verbose >= 2 and len(cols) < len(self.orig_df.columns):
            removed_cols = set(self.orig_df.columns) - set(cols)
            print()
            print(f"Removing columns with only one value: {removed_cols}")
        self.orig_df = self.orig_df[cols]

        # Determine which columns are binary, numeric, date, and none of these. Any that are not binary, numeric, or
        # date we consider as string.
        self.binary_cols = []
        self.numeric_cols = []
        self.date_cols = []
        self.string_cols = []

        # As this is called before check_data_quality, self.freq_contamination_level is not yet set. We use here the
        # default value in order to determine numeric columns with some non-numeric values.
        default_contamination_level = self.num_rows * 0.005

        for col_name in self.orig_df.columns:
            if self.orig_df[col_name].nunique() == 2:
                self.binary_cols.append(col_name)
            elif self.orig_df[col_name].dtype in [np.datetime64, 'datetime64[ns]']:
                self.date_cols.append(col_name)
            elif pandas_types.is_numeric_dtype(self.orig_df[col_name]) or \
                    self.orig_df[col_name].astype(str).str.replace('-', '', regex=False).str.\
                            replace('.', '', regex=False).str.isdigit().tolist().count(False) < default_contamination_level:
                self.numeric_cols.append(col_name)
            else:
                try:
                    _ = self.orig_df[col_name].astype(float)
                    self.numeric_cols.append(col_name)
                except Exception:
                    self.string_cols.append(col_name)

        # Try to convert any string columns we can to date format. The code below is likely sufficient, though may
        # erroneously convert some string or numeric columns to dates. If we find any legitimate date columns are
        # missed, we can use PyTime: https://github.com/shinux/PyTime. We are trying to minimize the pip installs
        # necessary for this tool, so will only add this if necessary.

        # As we cannot define the format for the date columns, attempts to cast values to datetime may present
        # warnings, which we ignore, but only during this process.
        warnings.filterwarnings(action='ignore', category=UserWarning)
        if known_date_cols is None:
            new_date_cols = []
            for col_name in self.string_cols + self.numeric_cols:
                avg_num_chars = statistics.median(self.orig_df[col_name].astype(str).str.len())
                num_rows_all_digits = self.orig_df[col_name].astype(str).str.isdigit().tolist().count(True)

                # Do not convert to date if the strings are too short. They must be at least yyyymm (6 characters)
                if avg_num_chars < 6:
                    continue

                # Do not convert to date if the strings are almost all digits and are too long
                if num_rows_all_digits > (self.num_rows / 2) and avg_num_chars > 8:
                    continue

                # Try some known formats before letting pandas attempt to determine the format
                is_date = False

                # Check if the column is of the form yyyymm or mmyyyy
                if avg_num_chars == 6:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name], format="%Y%M")
                        is_date = True
                    except:
                        pass
                if avg_num_chars == 6 and not is_date:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name], format="%M%Y")
                        is_date = True
                    except:
                        pass

                # Check if the column is of the form yyyymm or mmyyyy, but converted to float, so containing '.0'
                if avg_num_chars == 6:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name].astype(np.int64), format="%Y%M")
                        is_date = True
                    except:
                        pass
                if avg_num_chars == 6 and not is_date:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name].astype(np.int64), format="%M%Y")
                        is_date = True
                    except:
                        pass

                if col_name in self.string_cols and not is_date:
                    try:
                        self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name])
                        is_date = True
                    except Exception:
                        pass

                if is_date:
                    new_date_cols.append(col_name)
                    self.date_cols.append(col_name)

            for datecol in new_date_cols:
                if datecol in self.string_cols:
                    self.string_cols.remove(datecol)
                if datecol in self.numeric_cols:
                    self.numeric_cols.remove(datecol)
        else:
            new_date_cols = []
            for col_name in known_date_cols:
                try:
                    self.orig_df[col_name] = pd.to_datetime(self.orig_df[col_name])
                    new_date_cols.append(col_name)
                    if col_name not in self.date_cols:
                        self.date_cols.append(col_name)
                except Exception:
                    pass

            for datecol in new_date_cols:
                if datecol in self.string_cols:
                    self.string_cols.remove(datecol)
                if datecol in self.numeric_cols:
                    self.numeric_cols.remove(datecol)
                if datecol in self.binary_cols:
                    self.binary_cols.remove(datecol)
        set_warnings_levels()

        # For any columns flagged as string columns, the dtype may be category.  Convert the columns to string to
        # ensure the code can compare values and perform other string operations
        for col_name in self.string_cols + self.binary_cols:
            if self.orig_df[col_name].dtype.name == 'category':
                self.orig_df[col_name] = self.orig_df[col_name].astype(str)

        for col_name in self.numeric_cols:
            if self.orig_df[col_name].dtype.name == 'category':
                try:
                    self.orig_df[col_name] = self.orig_df[col_name].astype(float)
                except:
                    self.orig_df[col_name] = self.orig_df[col_name].astype(str).astype(float)

        # For binary columns, find and cache the set of unique values per column
        for col_name in self.binary_cols:
            self.column_unique_vals[col_name] = sorted([x for x in self.orig_df[col_name].unique() if not is_missing(x)])

        # For all numeric columns, get the set of truly numeric values. This may have less than self.num_rows elements.
        for col_name in self.numeric_cols:
            self.numeric_vals[col_name] = pd.Series(
                [float(x) for x in self.orig_df[col_name]
                 if (isinstance(x, numbers.Number) or str(x).replace('-', '').replace('.', '').isdigit())])

        # Calculate and cache the median for each numeric column.
        self.column_medians = {}
        for col_name in self.numeric_cols:
            # It may be that there is the odd non-numeric value in the column. We take only valid numbers to
            # calculate the median.
            self.column_medians[col_name] = self.numeric_vals[col_name].median()

        # For all numeric columns, get the set of truly numeric values or the median. This will have the same rows
        # as orig_df.
        # Todo: remove calls to convert_to_numeric that use the median anyway
        for col_name in self.numeric_cols:
            self.numeric_vals_filled[col_name] = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])

        trimmed_orig_df = self.orig_df.copy()
        if len(self.numeric_cols) > 0:
            # Calculate and cache the pairwise correlations between each numeric column
            numeric_df = None
            for col_name in self.numeric_cols:
                if numeric_df is None:
                    numeric_df = convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])
                else:
                    numeric_df = pd.concat([numeric_df, convert_to_numeric(self.orig_df[col_name], self.column_medians[col_name])], axis=1)
            numeric_df.columns = self.numeric_cols

            # Calculate the correlations between the numeric columns
            if len(self.numeric_cols) >= 2:
                self.pearson_corr = numeric_df.corr(method='pearson')
                self.spearman_corr = numeric_df.corr(method='spearman')

            # Calculate the trimmed mean for each numeric column
            for col_name in self.numeric_cols:
                lower_limit = self.numeric_vals[col_name].quantile(0.01)
                upper_limit = self.numeric_vals[col_name].quantile(0.99)
                reduced_numeric_vals_filled = self.numeric_vals_filled[col_name].loc[trimmed_orig_df.index]
                trimmed_orig_df = trimmed_orig_df[(reduced_numeric_vals_filled > lower_limit) &
                                                  (reduced_numeric_vals_filled < upper_limit)]
            self.column_trimmed_means = {}
            for col_name in self.numeric_cols:
                self.column_trimmed_means[col_name] = \
                    convert_to_numeric(trimmed_orig_df[col_name], self.column_medians[col_name]).mean()

        # Create a sample of the full data, which may be used for early stopping for expensive tests.
        # The sample_df will tend to not contain Null values.
        if len(trimmed_orig_df) > 50:
            self.sample_df = trimmed_orig_df.dropna().sample(n=min(len(trimmed_orig_df.dropna()), 50), random_state=0)
        elif len(self.orig_df.dropna()) > 50:
            self.sample_df = self.orig_df.dropna().sample(n=50, random_state=0)
        else:
            self.sample_df = self.orig_df.sample(n=min(len(self.orig_df), 50), random_state=0)

        if (len(self.sample_df) < 50) and (len(self.sample_df) < len(self.orig_df)):
            num_needed = 50 - len(self.sample_df)
            self.sample_df = pd.concat([self.sample_df,
                                        self.orig_df.sample(n=min(len(self.orig_df), 50), random_state=0)])

        # Similar to numeric_vals_filled, fill in this for the sample_df
        for col_name in self.numeric_cols:
            self.sample_numeric_vals_filled[col_name] = convert_to_numeric(self.sample_df[col_name], self.column_medians[col_name])

        # Calculate the number of valid (ie, not missing) values there are per row
        self.num_valid_rows = {}
        for col_name in self.orig_df.columns:
            self.num_valid_rows[col_name] = len([x for x in self.orig_df[col_name] if not is_missing(x)])

        # Replace any NA values that cannot be treated as None or NaN
        def test_NA(x):
            try:
                if x == x:
                    x = x
                return False
            except:
                return True

        for col_name in self.orig_df.columns:
            self.orig_df[col_name] = [x if not test_NA(x) else None for x in self.orig_df[col_name]]

        # patterns_df has a row for each test for each feature where there is a pattern with no exceptions.
        self.patterns_arr = []
        self.patterns_df = None

        # test_results_df has a column for each test for each original column where there is a pattern and also
        # exceptions.
        self.test_results_df = pd.DataFrame()
        self.n_tests_executed = 0

        # test_results_by_column_np is set initially to all zeros, as no
        self.test_results_by_column_np = np.zeros((self.num_rows, len(self.orig_df.columns)), dtype=float)

        # exceptions_summary_df has row for each test for each feature where there is a pattern and also exceptions.
        self.results_summary_arr = []
        self.exceptions_summary_df = None

        # Variables set as needed.
        self._init_variables()

        if self.verbose >= 2:
            print()
            print("Identified column types:")
            print(f"Number string: {len(self.string_cols)}")
            print(f"Number numeric: {len(self.numeric_cols)}")
            print(f"Number date/time: {len(self.date_cols)}")
            print(f"Number binary: {len(self.binary_cols)}")
            print()

    def _init_variables(self):
        self.lower_limits_dict = None
        self.upper_limits_dict = None
        self.larger_pairs_dict = None
        self.larger_or_equal_pairs_dict = None
        self.larger_pairs_with_bool_dict = None
        self.larger_or_equal_pairs_with_bool_dict = None
        self.is_missing_dict = None
        self.sample_is_missing_dict = None
        self.num_missing_dict = None
        self.percentiles_dict = None
        self.nunique_dict = None
        self.count_most_freq_value_dict = None
        self.words_list_dict = None
        self.word_counts_dict = None
        self.cols_same_bool_dict = None
        self.cols_same_count_dict = None
        self.cols_pairs_both_null_dict = None
        self.sample_cols_pairs_both_null_dict = None
        self.col_pairs_either_null_bool_dict = None
        self.col_triples_all_null_bool_dict = None
        self.common_vals_dict = None


