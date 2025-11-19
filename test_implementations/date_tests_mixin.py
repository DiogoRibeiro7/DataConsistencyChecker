"""
DateTestsMixin: Test methods for date tests.

This mixin contains 36 test methods organized by category.
Extracted from check_data_consistency.py for better code organization.
"""

from __future__ import annotations
from typing import Any

import pandas as pd
import numpy as np
import numbers
import sys
import math
import statistics
import datetime
import calendar
import random
import string
import copy
import scipy
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import tree, metrics
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from itertools import combinations
from decimal import Decimal, ROUND_HALF_UP

from checker_utils import (
    safe_div,
    is_number,
    convert_to_numeric,
    get_num_decimal_digits,
    get_non_alphanumeric,
    is_missing,
    array_to_str,
    replace_special_with_space,
)


class DateTestsMixin:
    """
    Mixin class containing date tests methods.

    This class contains 36 methods for testing data consistency
    related to date operations.

    This is a mixin class designed to be used with multiple inheritance.
    It does not have an __init__ method and relies on the parent class
    to provide necessary attributes and methods.
    """

    def _generate_early_dates(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('early date all', pd.date_range(test_date, periods=self.num_synth_rows))
        self._add_synthetic_column('early date most', pd.date_range(test_date, periods=self.num_synth_rows-1))
        self.synth_df.loc[999, 'early date most'] = datetime.datetime.strptime("01-7-2005", "%d-%m-%Y")

        # This function also tests that string columns that should be recognized as dates are properly converted to
        # dates. Test with the month in string format
        self._add_synthetic_column('date_str_1',
                                    ['Jan 12 2021',
                                     '2019-02-20 09:31:13.450',
                                     'Jan. 13 2022 ',
                                     '01/01/2003'] +
                                    ['April 15, 2005'] * (self.num_synth_rows-4))
        # Test in YYYYMM format
        self._add_synthetic_column('date_str_2',
                                    [202101,
                                     '2021/02',
                                     '2021-03'] +
                                    ['April 15, 2005'] * (self.num_synth_rows-3))


    def _check_early_dates(self, test_id):
        for col_name in self.date_cols:
            # todo: may need to cast back to datetime: pd.to_datetime(self.orig_df[col_name]) -- do all these methods
            #   also add the interpolation all these methods
            q1 = pd.to_datetime(self.orig_df[col_name]).quantile(0.25, interpolation='midpoint')
            q3 = pd.to_datetime(self.orig_df[col_name]).quantile(0.75, interpolation='midpoint')
            try:
                lower_limit = q1 - (self.iqr_limit * (q3 - q1))
            except: # There is a limit for pd.Timestamp objects. They can not go beyond Timestamp.min or .max
                continue
            test_series = pd.to_datetime(self.orig_df[col_name]) > lower_limit
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"The test flagged any values earlier than {lower_limit} as very early given the 25th quartile is "
                 f"{q1} and 75th is {q3}"),
                allow_patterns=False,
                display_info={'lower_limit': lower_limit}
            )


    def _generate_late_dates(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('late date all', pd.date_range(test_date, periods=self.num_synth_rows))
        self._add_synthetic_column('late date most', pd.date_range(test_date, periods=self.num_synth_rows-1))
        self.synth_df.loc[999, 'late date most'] = datetime.datetime.strptime("01-7-2045", "%d-%m-%Y")


    def _check_late_dates(self, test_id):
        for col_name in self.date_cols:
            q1 = pd.to_datetime(self.orig_df[col_name]).quantile(0.25)
            q3 = pd.to_datetime(self.orig_df[col_name]).quantile(0.75)
            try:
                upper_limit = q3 + (self.iqr_limit * (q3 - q1))  # Using a stricter threshold than the 2.2 normally used
            except:
                continue
            test_series = pd.to_datetime(self.orig_df[col_name]) < upper_limit
            test_series = test_series | self.orig_df[col_name].isna()
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                (f"The test flagged any values later than {upper_limit} as very late given the 25th quartile is "
                 f"{q1} and 75th is {q3}"),
                allow_patterns=False,
                display_info={'upper_limit': upper_limit}
            )

    # todo: also check for unusual dates (with distant neighbors)
    # difference in days between each entry:
    # pd.Series(sorted_test_series).diff().dt.days


    def _generate_unusual_dow(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('dow rand', pd.date_range(test_date, periods=self.num_synth_rows))
        self._add_synthetic_column('dow all',  pd.date_range(test_date, periods=self.num_synth_rows, freq='W'))
        self._add_synthetic_column('dow most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='W'))
        self.synth_df.loc[999, 'dow most'] = datetime.datetime.strptime("02-7-2022", "%d-%m-%Y")


    def _check_unusual_dow(self, test_id):
        for col_name in self.date_cols:
            # Check the dates span enough weeks to check the dow in a meaningful way
            min_date = pd.to_datetime(self.orig_df[col_name]).min()
            max_date = pd.to_datetime(self.orig_df[col_name]).max()
            if (max_date - min_date).days < 100:
                continue

            # datetime objects have day_of_week(); timestamps have dayofweek()
            # todo: test all methods with timestamps
            dow_list = pd.Series([x.day_of_week if hasattr(x, 'day_of_week') else x.dayofweek for x in pd.to_datetime(self.orig_df[col_name])])
            counts_list = dow_list.value_counts()
            rare_dow = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.freq_contamination_level]
            if len(rare_dow) == 0:
                continue
            test_series = np.array([x not in rare_dow for x in dow_list])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific days of the week',
                f" on the {rare_dow}th days of the week"
            )


    def _generate_unusual_dom(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-1922", "%d-%m-%Y")
        self._add_synthetic_column('dom rand', pd.date_range(test_date, periods=self.num_synth_rows))
        self._add_synthetic_column('dom all',  pd.date_range(test_date, periods=self.num_synth_rows, freq='M'))
        self._add_synthetic_column('dom most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='M'))
        self.synth_df.loc[999, 'dom most'] = datetime.datetime.strptime("02-7-2022", "%d-%m-%Y")


    def _check_unusual_dom(self, test_id):
        for col_name in self.date_cols:
            # Check the dates span enough months to check the dom in a meaningful way
            min_date = pd.to_datetime(self.orig_df[col_name]).min()
            max_date = pd.to_datetime(self.orig_df[col_name]).max()
            if (max_date - min_date).days < 300:
                continue

            dom_list = pd.Series([x.day for x in pd.to_datetime(self.orig_df[col_name])])
            counts_list = dom_list.value_counts()
            rare_dom = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.freq_contamination_level]
            if len(rare_dom) == 0:
                continue
            test_series = np.array([x not in rare_dom for x in dom_list])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific days of the month',
                f" on the {rare_dom}th days of the month"
            )


    def _generate_unusual_month(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('month rand', pd.date_range(test_date, periods=self.num_synth_rows, freq='3D'))
        dates = pd.date_range(test_date, periods=self.num_synth_rows, freq='3D')
        dates = [x if x.month != 12 else x + relativedelta(months=1) for x in dates]
        self._add_synthetic_column('month all', dates)
        self._add_synthetic_column('month most', dates)
        self.synth_df.loc[999, 'month most'] = datetime.datetime.strptime("02-12-2022", "%d-%m-%Y")


    def _check_unusual_month(self, test_id):
        for col_name in self.date_cols:
            # Check the dates span enough years to check the dom in a meaningful way
            min_date = pd.to_datetime(self.orig_df[col_name]).min()
            max_date = pd.to_datetime(self.orig_df[col_name]).max()
            if (max_date - min_date).days < (365 * 3):
                continue

            month_list = pd.Series([x.month for x in pd.to_datetime(self.orig_df[col_name])])
            counts_list = month_list.value_counts()
            rare_months = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.freq_contamination_level]
            if len(rare_months) == 0:
                continue
            test_series = np.array([x not in rare_months for x in month_list])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific months of the year',
                f" on the {rare_months}th months of the year"
            )


    def _generate_unusual_hour(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022 02:35:5", "%d-%m-%Y %H:%M:%S")
        self._add_synthetic_column('hour rand', pd.date_range(test_date, periods=self.num_synth_rows, freq='s'))
        self._add_synthetic_column('hour all', pd.date_range(test_date, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('hour most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='D'))
        self.synth_df.loc[999, 'hour most'] = datetime.datetime.strptime("01-7-2022 04:35:5", "%d-%m-%Y %H:%M:%S")


    def _check_unusual_hour(self, test_id):
        for col_name in self.date_cols:
            # Check the times span enough days to check the dom in a meaningful way
            min_date = pd.to_datetime(self.orig_df[col_name]).min()
            max_date = pd.to_datetime(self.orig_df[col_name]).max()
            if (max_date - min_date).days < 5:
                continue

            # todo: don't check the hour is consistent if it's just consitently missing, which is common. check the
            #   minutes vary, or something.

            hour_list = pd.Series([x.hour for x in pd.to_datetime(self.orig_df[col_name])])
            counts_list = hour_list.value_counts()
            rare_hours = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.freq_contamination_level]
            if len(rare_hours) == 0:
                continue
            test_series = np.array([x not in rare_hours for x in hour_list])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific hours of the day',
                f" on the {rare_hours}th hours of the day",
                display_info={"hour_list": hour_list}
            )


    def _generate_unusual_minutes(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date = datetime.datetime.strptime("01-7-2022 02:35:5", "%d-%m-%Y %H:%M:%S")
        self._add_synthetic_column('minutes rand', pd.date_range(test_date, periods=self.num_synth_rows, freq='s'))
        self._add_synthetic_column('minutes all',  pd.date_range(test_date, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('minutes most', pd.date_range(test_date, periods=self.num_synth_rows-1, freq='D'))
        self.synth_df.loc[999, 'minutes most'] = datetime.datetime.strptime("01-7-2022 04:59:5", "%d-%m-%Y %H:%M:%S")


    def _check_unusual_minutes(self, test_id):
        for col_name in self.date_cols:
            # Check the times span enough days to check the dom in a meaningful way
            min_date = pd.to_datetime(self.orig_df[col_name]).min()
            max_date = pd.to_datetime(self.orig_df[col_name]).max()
            # Ensure there is at least 100 hours of range
            if ((max_date - min_date) / pd.Timedelta(hours=1)) < 100:
                continue

            # todo: don't check the hour is consistent if it's just consitently missing, which is common. check the
            #   minutes vary, or something.

            minutes_list = pd.Series([x.minute for x in pd.to_datetime(self.orig_df[col_name])])
            counts_list = minutes_list.value_counts()
            rare_minutes = [x for x, y in zip(counts_list.index, counts_list.values) if y < self.freq_contamination_level]
            if len(rare_minutes) == 0:
                continue
            test_series = np.array([x not in rare_minutes for x in minutes_list])
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on specific minutes of the hour',
                f" on the {rare_minutes}th minutes of the hour"
            )


    def _generate_constant_dom(self):
        """
        Patterns without exceptions: The column 'constant_dom all' consistently has dates on the 1st of the month
        Patterns with exception: The column 'constant_dom most' consistently has dates on the 1st of the month, with
            one exception.
        """
        rand_dates = []
        for i in range(self.num_synth_rows):
            d = np.random.randint(1, 28, 1)[0]
            m = np.random.randint(1, 12, 1)[0]
            y = np.random.randint(2010, 2023, 1)[0]
            rand_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} 02:35:5", "%d-%m-%Y %H:%M:%S"))

        all_constant_dates = []
        most_constant_dates = []
        for i in range(self.num_synth_rows):
            m = np.random.randint(1, 12, 1)[0]
            y = np.random.randint(2010, 2023, 1)[0]
            all_constant_dates.append(datetime.datetime.strptime(f"01-{m}-{y} 02:35:5", "%d-%m-%Y %H:%M:%S"))
            most_constant_dates.append(datetime.datetime.strptime(f"01-{m}-{y} 02:35:5", "%d-%m-%Y %H:%M:%S"))
        most_constant_dates[-1] = datetime.datetime.strptime(f"13-7-2021 02:35:5", "%d-%m-%Y %H:%M:%S")

        self._add_synthetic_column('constant_dom rand', rand_dates)
        self._add_synthetic_column('constant_dom all', all_constant_dates)
        self._add_synthetic_column('constant_dom most', most_constant_dates)


    def _check_constant_dom(self, test_id):
        for col_name in self.date_cols:
            min_date = pd.to_datetime(self.orig_df[col_name]).min()
            max_date = pd.to_datetime(self.orig_df[col_name]).max()

            # Ensure there is at least 3 months of range
            if ((max_date - min_date) / np.timedelta64(1, 'D')) < 91:
                continue

            dom_arr = pd.to_datetime(self.orig_df[col_name]).dt.day
            most_common = statistics.mode(dom_arr)
            test_series = (dom_arr == most_common) | (self.orig_df[col_name].isna())
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on day {most_common} of the month',
                ""
            )


    def _generate_last_dom(self):
        """
        Patterns without exceptions: The column 'constant_last_dom all' consistently has dates on the last of the month
        Patterns with exception: The column 'constant_last_dom most' consistently has dates on the last of the month,
            with one exception.
        """
        rand_dates = []
        for i in range(self.num_synth_rows):
            d = np.random.randint(1, 28, 1)[0]
            m = np.random.randint(1, 12, 1)[0]
            y = np.random.randint(2010, 2023, 1)[0]
            rand_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} 02:35:5", "%d-%m-%Y %H:%M:%S"))

        all_constant_dates = []
        most_constant_dates = []
        dom_arr = [(1, 31), (3, 31), (4, 30), (5, 31), (6, 30)]
        for i in range(self.num_synth_rows):
            m, d = dom_arr[np.random.randint(0, len(dom_arr), 1)[0]]
            y = np.random.randint(2010, 2023, 1)[0]
            all_constant_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} 02:35:5", "%d-%m-%Y %H:%M:%S"))
            most_constant_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} 02:35:5", "%d-%m-%Y %H:%M:%S"))
        most_constant_dates[-1] = datetime.datetime.strptime(f"13-7-2021 02:35:5", "%d-%m-%Y %H:%M:%S")

        self._add_synthetic_column('constant_last_dom rand', rand_dates)
        self._add_synthetic_column('constant_last_dom all', all_constant_dates)
        self._add_synthetic_column('constant_last_dom most', most_constant_dates)


    def _check_last_dom(self, test_id):

        def check_last_dom(x):
            if x is None or x != x:
                return False
            y = x.year
            m = x.month
            d = x.day
            last_dom = calendar.monthrange(y, m)[1]
            return d == last_dom

        for col_name in self.date_cols:
            min_date = pd.to_datetime(self.orig_df[col_name]).min()
            max_date = pd.to_datetime(self.orig_df[col_name]).max()

            # Ensure there is at least 3 months of range
            if ((max_date - min_date) / np.timedelta64(1, 'D')) < 91:
                continue

            test_series = pd.to_datetime(self.orig_df[col_name]).apply(check_last_dom) | (self.orig_df[col_name].isna())
            self._process_analysis_binary(
                test_id,
                [col_name],
                test_series,
                f'The values in Column "{col_name}" were consistently on last day of the month',
                ""
            )

    ##################################################################################################################
    # Data consistency checks pairs of date columns
    ##################################################################################################################


    def _generate_constant_date_gap(self):
        """
        Patterns without exceptions: 'const_gap all_1', 'const_gap all_2', and 'const_gap all_3' are all a constant
            gap from each other. 'const_gap all_1' contains daily values starting July 1. ''const_gap all_2'
            contains daily values starting 1 week later, so is consistently 7 days after 'const_gap all_1'.
        Patterns with exception: 'const_gap most' is the same as 'const_gap all_2', so also has a consistent 7 day
            gap from 'const_gap all_1', 'const_gap all_2' and 'const_gap all_3', but with one exception.
        """

        # 'const_gap rand' simply tests this is identified as a date column
        rand_dates = []
        for _ in range(self.num_synth_rows):
            y = np.random.randint(1990, 2010)
            m = np.random.randint(1, 13)
            if m >= 10:
                s = str(y) + str(m)
            else:
                s = str(y) + "0" + str(m)
            rand_dates.append(s)
        self._add_synthetic_column('const_gap rand', rand_dates)

        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        test_date2 = datetime.datetime.strptime("07-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('const_gap all_1', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('const_gap all_2', pd.date_range(test_date2, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('const_gap all_3', pd.date_range(test_date2, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('const_gap most', pd.date_range(test_date2, periods=self.num_synth_rows-1, freq='D'))
        self.synth_df.loc[999, 'const_gap most'] = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")


    def _check_constant_date_gap(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                gap_array = pd.Series([
                    0 if (is_missing(x) or is_missing(y)) else (x-y).days
                    for x, y in zip(pd.to_datetime(self.orig_df[col_name_1]), pd.to_datetime(self.orig_df[col_name_2]))
                ])
                self._process_analysis_counts(
                    test_id,
                    [col_name_1, col_name_2],
                    gap_array,
                    f"Columns {col_name_1} and {col_name_2} are consistently",
                    "days apart"
                )


    def _generate_large_date_gap(self):
        """
        Patterns without exceptions: 'large_gap all_2' consistently has a small gap (1 to 10 days) after
            'large_gap all_1'
        Patterns with exception: 'large_gap most' consistently has a small gap (1 to 10 days) after
            'large_gap all_1, with 1 exception'
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('large_gap all_1', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('large_gap all_2', [self.synth_df.loc[x]['large_gap all_1'] + datetime.timedelta(days=random.randint(1, 10)) for x in self.synth_df.index] )
        self._add_synthetic_column('large_gap most', [self.synth_df.loc[x]['large_gap all_1'] + datetime.timedelta(days=random.randint(1, 10)) for x in self.synth_df.index] )
        self.synth_df.loc[999, 'large_gap most'] = datetime.datetime.strptime("01-7-2028", "%d-%m-%Y")


    def _check_large_date_gap(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            med_1 = pd.to_datetime(self.orig_df[col_name_1]).quantile(0.5, interpolation='midpoint')
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                med_2 = pd.to_datetime(self.orig_df[col_name_2]).quantile(0.5, interpolation='midpoint')
                if med_1 > med_2:
                    col_big = col_name_1
                    col_small = col_name_2
                else:
                    col_big = col_name_2
                    col_small = col_name_1
                gap_array = pd.Series([0 if (is_missing(x) or is_missing(y))
                                       else (x-y).days
                                       for x, y in zip(pd.to_datetime(self.orig_df[col_big]), pd.to_datetime(self.orig_df[col_small]))])
                gap_array_no_null = pd.Series([(x-y).days
                                               for x, y in zip(pd.to_datetime(self.orig_df[col_big]), pd.to_datetime(self.orig_df[col_small]))
                                               if not is_missing(x) and not is_missing(y)])
                q1 = gap_array_no_null.quantile(0.25)
                q3 = gap_array_no_null.quantile(0.75)
                iqr = q3 - q1
                try:
                    threshold = q3 + (iqr * self.iqr_limit)
                except:
                    continue
                test_series = gap_array < threshold
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_small, col_big],
                    test_series,
                    (f'The gap between "{col_small}" and "{col_big}" is larger than normal. We flag any gaps '
                     f'larger than {threshold} days, as the 25th percentile is {q1} days and the 75th {q3} days.'),
                    f"",
                    allow_patterns=False
                )


    def _generate_small_date_gap(self):
        """
        Patterns without exceptions: None
        Patterns with exception: 'small_gap most' has a consistently small gap from both 'small_gap all_1' and
            'small_gap all_2', with exceptions.
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('small_gap all_1', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('small_gap all_2',
            [self.synth_df.loc[x]['small_gap all_1'] + datetime.timedelta(days=random.randint(15, 20)) for x in self.synth_df.index])
        self._add_synthetic_column('small_gap most',
            [self.synth_df.loc[x]['small_gap all_1'] + datetime.timedelta(days=random.randint(15, 20)) for x in self.synth_df.index] )
        self.synth_df.loc[999, 'small_gap most'] = self.synth_df.loc[999, 'small_gap all_1']


    def _check_small_date_gap(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            med_1 = pd.to_datetime(self.orig_df[col_name_1]).quantile(0.5, interpolation='midpoint')
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                med_2 = pd.to_datetime(self.orig_df[col_name_2]).quantile(0.5, interpolation='midpoint')
                if med_1 > med_2:
                    col_big = col_name_1
                    col_small = col_name_2
                else:
                    col_big = col_name_2
                    col_small = col_name_1
                gap_array = pd.Series([0 if (is_missing(x) or is_missing(y))
                                       else (x-y).days
                                       for x, y in zip(pd.to_datetime(self.orig_df[col_big]),
                                                       pd.to_datetime(self.orig_df[col_small]))])
                gap_array_no_null = pd.Series([(x-y).days
                                               for x, y in zip(pd.to_datetime(self.orig_df[col_big]),
                                                               pd.to_datetime(self.orig_df[col_small]))
                                               if not is_missing(x) and not is_missing(y)])
                q1 = gap_array_no_null.quantile(0.25)
                q3 = gap_array_no_null.quantile(0.75)
                iqr = q3 - q1
                try:
                    threshold = q1 - (iqr * self.iqr_limit)
                except:
                    continue
                test_series = gap_array > threshold
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_small, col_big],
                    test_series,
                    (f'The gap between "{col_small}" and "{col_big}" is smaller than normal. We flag any gaps '
                     f'smaller than {threshold} days, as the 25th percentile is {q1} days and the 75th {q3} days.'),
                    f"",
                    allow_patterns=False
                )


    def _generate_date_later(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('date_later all_1',
                                    pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('date_later all_2',
                                    [self.synth_df.loc[x]['date_later all_1'] + datetime.timedelta(days=random.randint(15, 20))
             for x in self.synth_df.index])
        self._add_synthetic_column('date_later most',
                                    [self.synth_df.loc[x]['date_later all_1'] + datetime.timedelta(days=random.randint(15, 20))
                                    for x in self.synth_df.index] )
        self.synth_df.loc[999, 'date_later most'] = test_date1


    def _check_date_later(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            med_1 = pd.to_datetime(self.orig_df[col_name_1]).quantile(0.5, interpolation='midpoint')
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                med_2 = pd.to_datetime(self.orig_df[col_name_2]).quantile(0.5, interpolation='midpoint')
                if med_1 > med_2:
                    col_big = col_name_1
                    col_small = col_name_2
                else:
                    col_big = col_name_2
                    col_small = col_name_1
                gap_array = pd.Series([0 if (is_missing(x) or is_missing(y))
                                       else (x-y).days
                                       for x, y in zip(pd.to_datetime(self.orig_df[col_big]), pd.to_datetime(self.orig_df[col_small]))])
                test_series = gap_array > 0
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_small, col_big],
                    test_series,
                    f'"{col_big}" is consistently later than "{col_small}"',
                    f"",
                    allow_patterns=False
                )


    def _generate_same_date(self):
        """
        Patterns without exceptions: "same_date rand" AND "same_date all" consistently have the same date
        Patterns with exception: "same_date rand" AND "same_date most", as well as "same_date all" AND "same_date most"
            consistently have the same date, with one exception.
        """
        rand_dates = []
        all_same_dates = []
        most_same_dates = []
        for i in range(self.num_synth_rows):
            d = np.random.randint(1, 28, 1)[0]
            m = np.random.randint(1, 12, 1)[0]
            y = np.random.randint(2010, 2023, 1)[0]
            h = np.random.randint(1, 20, 1)[0]
            min = np.random.randint(1, 50, 1)[0]
            rand_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} {h}:{min}:5", "%d-%m-%Y %H:%M:%S"))

            h = np.random.randint(1, 20, 1)[0]
            min = np.random.randint(1, 50, 1)[0]
            all_same_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} {h}:{min}:5", "%d-%m-%Y %H:%M:%S"))

            h = np.random.randint(1, 20, 1)[0]
            min = np.random.randint(1, 50, 1)[0]
            most_same_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} {h}:{min}:5", "%d-%m-%Y %H:%M:%S"))
        most_same_dates[-1] = datetime.datetime.strptime(f"{23}-{2}-{2024} {8}:{43}:5", "%d-%m-%Y %H:%M:%S")

        self._add_synthetic_column('same_date rand', rand_dates)
        self._add_synthetic_column('same_date all', all_same_dates)
        self._add_synthetic_column('same_date most', most_same_dates)


    def _check_same_date(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                test_series = [(y1 == y2) and (m1 == m2) and (d1 == d1) for y1, y2, m1, m2, d1, d2 in
                               zip(pd.to_datetime(self.orig_df[col_name_1]).dt.year,
                                   pd.to_datetime(self.orig_df[col_name_2]).dt.year,
                                   pd.to_datetime(self.orig_df[col_name_1]).dt.month,
                                   pd.to_datetime(self.orig_df[col_name_2]).dt.month,
                                   pd.to_datetime(self.orig_df[col_name_1]).dt.day,
                                   pd.to_datetime(self.orig_df[col_name_2]).dt.day,
                                )]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    f'The dates in "{col_name_1}" and "{col_name_2}" are consistently on the same date.'
                )


    def _generate_same_month(self):
        """
        Patterns without exceptions: "same_month rand" AND "same_month all" consistently have the same month.
        Patterns with exception: "same_month rand" AND "same_month most", as well as "same_month all" AND
            same_month most" consistently have the same month, with one exception.
        """
        rand_dates = []
        all_same_months = []
        most_same_months = []
        for i in range(self.num_synth_rows):
            y = np.random.randint(2010, 2023, 1)[0]
            m = np.random.randint(1, 12, 1)[0]
            d = np.random.randint(1, 28, 1)[0]
            h = np.random.randint(1, 20, 1)[0]
            min = np.random.randint(1, 50, 1)[0]
            rand_dates.append(datetime.datetime.strptime(f"{d}-{m}-{y} {h}:{min}:5", "%d-%m-%Y %H:%M:%S"))

            d = np.random.randint(1, 28, 1)[0]
            h = np.random.randint(1, 20, 1)[0]
            min = np.random.randint(1, 50, 1)[0]
            all_same_months.append(datetime.datetime.strptime(f"{d}-{m}-{y} {h}:{min}:5", "%d-%m-%Y %H:%M:%S"))

            d = np.random.randint(1, 28, 1)[0]
            h = np.random.randint(1, 20, 1)[0]
            min = np.random.randint(1, 50, 1)[0]
            most_same_months.append(datetime.datetime.strptime(f"{d}-{m}-{y} {h}:{min}:5", "%d-%m-%Y %H:%M:%S"))
        most_same_months[-1] = datetime.datetime.strptime(f"{23}-{2}-{2024} {8}:{43}:5", "%d-%m-%Y %H:%M:%S")

        self._add_synthetic_column('same_month rand', rand_dates)
        self._add_synthetic_column('same_month all', all_same_months)
        self._add_synthetic_column('same_month most', most_same_months)


    def _check_same_month(self, test_id):
        for col_name_idx_1 in range(len(self.date_cols)-1):
            col_name_1 = self.date_cols[col_name_idx_1]
            for col_name_idx_2 in range(col_name_idx_1 + 1, len(self.date_cols)):
                col_name_2 = self.date_cols[col_name_idx_2]
                test_series = [(y1 == y2) and (m1 == m2) for y1, y2, m1, m2 in
                               zip(pd.to_datetime(self.orig_df[col_name_1]).dt.year,
                                   pd.to_datetime(self.orig_df[col_name_2]).dt.year,
                                   pd.to_datetime(self.orig_df[col_name_1]).dt.month,
                                   pd.to_datetime(self.orig_df[col_name_2]).dt.month,
                                )]
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    f'The dates in "{col_name_1}" and "{col_name_2}" are consistently on the same month.'
                )


    def _generate_correlated_dates(self):
        rand_dates = []
        correlated_dates = []
        for i in range(self.num_synth_rows):
            y = np.random.randint(2010, 2023, 1)[0]
            m = np.random.randint(1, 12, 1)[0]
            d = np.random.randint(1, 28, 1)[0]
            date_val = datetime.datetime.strptime(f"{d}-{m}-{y}", "%d-%m-%Y")
            rand_dates.append(date_val)
            correlated_dates.append(date_val + relativedelta(days=(np.random.randint(-50, 50))))

        self._add_synthetic_column('corr_dates rand', rand_dates)
        self._add_synthetic_column('corr_dates all', correlated_dates)
        self._add_synthetic_column('corr_dates most', correlated_dates)
        self.synth_df.loc[999, 'corr_dates most'] = datetime.datetime.strptime(f"1-1-2009", "%d-%m-%Y")


    def _check_correlated_dates(self, test_id):
        num_pairs, date_pairs_list = self._get_date_column_pairs_unique()
        if num_pairs > self.max_combinations:
            if self.verbose >= 1:
                print((f"  Skipping test. There are {num_pairs:,} pairs of date columns. "
                       f"max_combinations is currently set to {self.max_combinations:,}."))
            return

        cols_same_bool_dict = self.get_cols_same_bool_dict()

        for pair_idx, (col_name_1, col_name_2) in enumerate(date_pairs_list):
            if self.verbose >= 2 and pair_idx > 0 and pair_idx % 10_000 == 0:
                print(f"  Examining pair {pair_idx:,} of {len(date_pairs_list):,} pairs of numeric columns")

            if self.orig_df[col_name_1].nunique(dropna=True) < self.freq_contamination_level:
                continue
            if self.orig_df[col_name_2].nunique(dropna=True) < self.freq_contamination_level:
                continue

            val_arr_1 = pd.to_numeric(pd.to_datetime(self.orig_df[col_name_1]))
            val_arr_2 = pd.to_numeric(pd.to_datetime(self.orig_df[col_name_2]))
            if val_arr_1.nunique() < 3:
                continue
            if val_arr_2.nunique() < 3:
                continue
            if self.orig_df[col_name_1].isna().sum() > (self.num_rows * 0.75):
                continue
            if self.orig_df[col_name_2].isna().sum() > (self.num_rows * 0.75):
                continue

            # Skip columns that are almost entirely the same
            if cols_same_bool_dict[tuple(sorted([col_name_1, col_name_2]))]:
                continue

            spearman_corr = abs(val_arr_1.corr(val_arr_2, method='spearman'))
            if spearman_corr >= 0.995:
                col_1_percentiles = self.orig_df[col_name_1].rank(pct=True)
                col_2_percentiles = self.orig_df[col_name_2].rank(pct=True)

                # Test for positive correlation
                test_series = np.array([abs(x-y) < 0.2 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'"{col_name_1}" is consistently similar, with respect to rank, to "{col_name_2}" with a Spearman '
                     f'correlation of {spearman_corr}'),
                    display_info={'col_1_percentiles': col_1_percentiles, 'col_2_percentiles': col_2_percentiles}
                )

                # Test for negative correlation
                test_series = np.array([abs(x-(1.0 - y)) < 0.2 for x, y in zip(col_1_percentiles, col_2_percentiles)])
                test_series = test_series | self.orig_df[col_name_1].isna() | self.orig_df[col_name_2].isna()
                self._process_analysis_binary(
                    test_id,
                    [col_name_1, col_name_2],
                    test_series,
                    (f'"{col_name_1}" is consistently inversely similar in rank to "{col_name_2}" with an absolute '
                     f'Spearman correlation of {spearman_corr}'),
                    display_info={'col_1_percentiles': col_1_percentiles, 'col_2_percentiles': col_2_percentiles}
                )

    ##################################################################################################################
    # Data consistency checks for two columns, where one is date and the other is numeric
    ##################################################################################################################


    def _generate_large_given_date(self):
        """
        Patterns without exceptions: This method identifies exceptions, but not patterns.
        Patterns with exception: 'large_given_date all' is well correlated with the date column and has no exceptions.
            'large_given_date most' has one value in the last date bin which is very for that bin, though is normal
            relative to the full column.
        # todo: should test with a largish value in one of the early rows.
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column(
            'large_given_date rand', pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column(
            'large_given_date all',  sorted([x for x in range(self.num_synth_rows)], reverse=True))
        self._add_synthetic_column(
            'large_given_date most', sorted([x for x in range(self.num_synth_rows - 1)], reverse=True) + [990])


    def _check_large_given_date(self, test_id):

        # Ensure all 10 bins have at least 50 rows
        if self.num_rows < 500:
            return

        # Calculate and cache the upper limit based on q1 and q3 of each full numeric & date column
        upper_limits_dict = self.get_columns_iqr_upper_limit()
        nunique_dict = self.get_nunique_dict()

        for date_idx, date_col in enumerate(self.date_cols):
            if self.verbose >= 2 and date_idx > 0 and date_idx % 10 == 0:
                print(f"  Examining column {date_idx} of {len(self.date_cols)} date columns")

            if self.orig_df[date_col].nunique() < 10:
                continue

            # Create 10 equal-width bins for the dates in the current date column
            bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            bin_assignments, bin_edges = pd.cut(pd.to_datetime(self.orig_df[date_col]), bins=10, labels=bin_labels,
                                                retbins=True)
            if bin_assignments.isna().sum() > 0:
                bin_labels.append(-1)
                bin_assignments = pd.Series([-1 if is_missing(x) else x for x in bin_assignments])
            bin_row_idxs = {}
            sub_dfs_arr_dict = {}

            # Create a dictionary, keyed by the bin ids, containing a dataframe covering the rows represented by each
            # bin.
            for bin_id in bin_labels:
                bin_row_idxs[bin_id] = np.where(bin_assignments == bin_id)
                sub_dfs_arr_dict[bin_id] = self.orig_df.loc[bin_row_idxs[bin_id]]

            for num_col in self.numeric_cols:

                if nunique_dict[num_col] < math.sqrt(self.num_rows):
                    continue

                test_series = [True] * self.num_rows
                for bin_id in bin_labels:
                    # Get the upper limit (based on IQR), given the full column
                    upper_limit, col_q2, col_q3 = upper_limits_dict[num_col]

                    # Get the stats for the numeric column for the subset
                    sub_df = sub_dfs_arr_dict[bin_id]
                    num_vals = pd.Series(
                        [float(x) for x in sub_df[num_col] if str(x).replace('-', '').replace('.', '').isdigit()])
                    q1 = num_vals.quantile(0.25)
                    med = num_vals.quantile(0.50)
                    q3 = num_vals.quantile(0.75)
                    iqr = q3 - q1
                    # As we look at multiple subsets, we use a higher coefficient than normally when looking for large
                    # values, though smaller than in LARGE_GIVEN_PAIRS, which looks at still more subsets.
                    threshold = q3 + (iqr * 1.5 * self.iqr_limit)

                    if q1 is None or med is None or q3 is None or upper_limit is None or col_q2 is None or col_q3 is None:
                        continue

                    # We are only concerned in this test with subsets that tend to have smaller values in the
                    # numeric column, and flag large values in this case. We do not flag values in subsets that
                    # have any values that are large relative to the subset; these are flagged by other tests.
                    res_1 = num_vals > upper_limit
                    if res_1.tolist().count(True) > 0:
                        continue

                    # Check this subset is small compared to the full column
                    if (med >= col_q2) or (q3 >= col_q3):
                        continue

                    # We can not use self.numeric_value_filled, as that was filled with the median for the full column,
                    # not the median for this subset.
                    num_vals_all = convert_to_numeric(sub_df[num_col], med)
                    sub_test_series = pd.Series([(x < threshold) or (x!=x) or (x==None) for x in num_vals_all])

                    if 0 < sub_test_series.tolist().count(False) <= self.freq_contamination_level:
                        index_of_large = \
                            [x for x, y in zip(list(bin_row_idxs[bin_id][0]), list(sub_test_series.values)) if not y]
                        for i in index_of_large:
                            test_series[i] = False

                self._process_analysis_binary(
                    test_id,
                    [date_col, num_col],
                    test_series,
                    f'"{num_col}" is unusually large given the bin of the date column: "{date_col}"',
                    f"",
                    allow_patterns=False,
                    display_info={"bin_assignments": bin_assignments, "bin_edges": bin_edges}
                )


    def _generate_small_given_date(self):
        """
        Patterns without exceptions:
        Patterns with exception:
        """
        test_date1 = datetime.datetime.strptime("01-7-2022", "%d-%m-%Y")
        self._add_synthetic_column('small_given_date rand',
                                    pd.date_range(test_date1, periods=self.num_synth_rows, freq='D'))
        self._add_synthetic_column('small_given_date all', [x for x in range(self.num_synth_rows)])
        self._add_synthetic_column('small_given_date most', [x for x in range(self.num_synth_rows - 1)] + [2])


    def _check_small_given_date(self, test_id):

        # Ensure all 10 bins have at least 50 rows
        if self.num_rows < 500:
            return

        lower_limits_dict = self.get_columns_iqr_lower_limit()
        nunique_dict = self.get_nunique_dict()

        for date_idx, date_col in enumerate(self.date_cols):
            if self.verbose >= 2 and date_idx > 0 and date_idx % 10 == 0:
                print(f"  Examining column {date_idx} of {len(self.date_cols)} date columns")

            if self.orig_df[date_col].nunique() < 10:
                continue

            # Create 10 equal-width bins for the dates
            bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            bin_assignments, bin_edges = pd.cut(pd.to_datetime(self.orig_df[date_col]), bins=10, labels=bin_labels,
                                                retbins=True)
            if bin_assignments.isna().sum() > 0:
                bin_labels.append(-1)
                bin_assignments = pd.Series([-1 if is_missing(x) else x for x in bin_assignments])
            bin_row_idxs = {}
            sub_dfs_arr_dict = {}

            # Create a dictionary, keyed by the bin ids, containing a dataframe covering the rows represented by each
            # bin.
            for bin_id in bin_labels:
                bin_row_idxs[bin_id] = np.where(bin_assignments == bin_id)
                sub_dfs_arr_dict[bin_id] = self.orig_df.loc[bin_row_idxs[bin_id]]

            for num_col in self.numeric_cols:

                if nunique_dict[num_col] < math.sqrt(self.num_rows):
                    continue

                test_series = [True] * self.num_rows
                for bin_id in bin_labels:
                    # Get the lower limit (based on IQR), given the full column
                    lower_limit, col_q1, col_q3 = lower_limits_dict[num_col]

                    # Get the stats for the numeric column for the subset
                    sub_df = sub_dfs_arr_dict[bin_id]
                    num_vals = pd.Series(
                        [float(x) for x in sub_df[num_col] if str(x).replace('-', '').replace('.', '').isdigit()])
                    q1 = num_vals.quantile(0.25)
                    med = num_vals.quantile(0.50)
                    q3 = num_vals.quantile(0.75)
                    iqr = q3 - q1
                    # As we look at multiple subsets, we use a higher coefficient than normally when looking for large
                    # values, though smaller than in LARGE_GIVEN_PAIRS, which looks at still more subsets.
                    threshold = q1 - (iqr * 1.5 * self.iqr_limit)

                    if q1 is None or med is None or q3 is None or lower_limit is None or col_q1 is None or col_q3 is None:
                        continue

                    # We are only concerned in this test with subsets that tend to have larger values in the
                    # numeric column, and flag small values in this case. We do not flag values in subsets that
                    # have any values that are small relative to the subset; these are flagged by other tests.
                    res_1 = num_vals < lower_limit
                    if res_1.tolist().count(True) > 0:
                        continue

                    # Check this subset is small compared to the full column
                    if q1 <= col_q1:
                        continue

                    # We can not use self.numeric_value_filled, as that was filled with the median for the full column,
                    # not the median for this subset.
                    num_vals_all = convert_to_numeric(sub_df[num_col], med)
                    sub_test_series = pd.Series([(x >= threshold) or (x!=x) or (x==None) for x in num_vals_all])

                    if 0 < sub_test_series.tolist().count(False) <= self.freq_contamination_level:
                        index_of_small = \
                            [x for x, y in zip(list(bin_row_idxs[bin_id][0]), list(sub_test_series.values)) if not y]
                        for i in index_of_small:
                            test_series[i] = False

                self._process_analysis_binary(
                    test_id,
                    [date_col, num_col],
                    test_series,
                    f'"{num_col}" is unusually small given the bin of the date column: "{date_col}"',
                    f"",
                    allow_patterns=False,
                    display_info={"bin_assignments": bin_assignments, "bin_edges": bin_edges}
                )

    ##################################################################################################################
    # Data consistency checks for pairs of binary columns
    ##################################################################################################################


