"""
Date/Time Test Definitions for DataConsistencyChecker

This module contains test definitions for date and datetime columns including:
- Single date column tests (unusual dates, consistent patterns)
- Pair date column tests (gaps, relationships)
- Date-numeric combination tests
"""

from ..test_registry import TestDefinition


def _test(short_description, description, test_func, gen_func, shortlist, implemented, fast, code):
    """Build one typed test definition."""
    return TestDefinition(
        short_description=short_description,
        description=description,
        test_func=test_func,
        gen_func=gen_func,
        shortlist=shortlist,
        implemented=implemented,
        fast=fast,
        code=code,
    )




def get_date_tests(checker):
    """
    Get test definitions for date/datetime column tests.

    Args:
        checker: DataConsistencyChecker instance to bind test methods

    Returns:
        dict: Test definitions for date tests
    """
    return {
        # Tests on single Date columns
        'EARLY_DATES': _test(
            'Check for unusually early dates.',
            'Check for dates significantly earlier than the other dates in the column.',
            checker._check_early_dates,
            checker._generate_early_dates,
            True, True, True, False
        ),
        'LATE_DATES': _test(
            '',
            'Check for dates significantly later than the other dates in the column.',
            checker._check_late_dates,
            checker._generate_late_dates,
            True, True, True, False
        ),
        'UNUSUAL_DAY_OF_WEEK': _test(
            '',
            'Check if a date column contains any unusual days of the week.',
            checker._check_unusual_dow,
            checker._generate_unusual_dow,
            True, True, True, False
        ),
        'UNUSUAL_DAY_OF_MONTH': _test(
            '',
            'Check if a date column contains any unusual days of the month.',
            checker._check_unusual_dom,
            checker._generate_unusual_dom,
            True, True, True, False
        ),
        'UNUSUAL_MONTH': _test(
            '',
            'Check if a date column contains any unusual months of the year.',
            checker._check_unusual_month,
            checker._generate_unusual_month,
            True, True, True, False
        ),
        'UNUSUAL_HOUR': _test(
            'Check if a datetime / time column contains any unusual hours.',
            ('Check if a datetime / time column contains any unusual hours of the day. '
             'This and UNUSUAL_MINUTES also identify where it is inconsistent if the time '
             'is included in the column.'),
            checker._check_unusual_hour,
            checker._generate_unusual_hour,
            True, True, True, False
        ),
        'UNUSUAL_MINUTES': _test(
            'Check if a datetime / time column contains any unusual minutes.',
            ('Check if a datetime / time column contains any unusual minutes of the hour. '
             'This and UNUSUAL_MINUTES also identify where it is inconsistent if the time '
             'is included in the column.'),
            checker._check_unusual_minutes,
            checker._generate_unusual_minutes,
            True, True, True, False
        ),
        'CONSTANT_DOM': _test(
            'Check if dates are the same day of the month.',
            ('Check if a date column spans multiple months and the values are consistently '
             'the same day of the month'),
            checker._check_constant_dom,
            checker._generate_constant_dom,
            True, True, True, False
        ),
        'CONSTANT_LAST_DOM': _test(
            'Check if dates are the last day of the month',
            ('Check if a date column spans multiple months and the values are consistently '
             'the last day of the month'),
            checker._check_last_dom,
            checker._generate_last_dom,
            True, True, True, False
        ),

        # Tests on pairs of date columns
        'CONSTANT_GAP': _test(
            'Check for consistent gaps between dates.',
            ('Check if there is consistently a specific gap in time between two date columns.'),
            checker._check_constant_date_gap,
            checker._generate_constant_date_gap,
            True, True, False, False
        ),
        'LARGE_GAP': _test(
            'Check for large gaps between dates.',
            ('Check if there is an unusually large gap in time between dates in two date columns.'),
            checker._check_large_date_gap,
            checker._generate_large_date_gap,
            True, True, False, False
        ),
        'SMALL_GAP': _test(
            'Check for small gaps between dates.',
            ('Check if there is an unusually small gap in time between dates in two date columns.'),
            checker._check_small_date_gap,
            checker._generate_small_date_gap,
            True, True, False, False
        ),
        'LATER': _test(
            '',
            'Check if one date column is consistently later than another date column.',
            checker._check_date_later,
            checker._generate_date_later,
            True, True, False, False
        ),
        'SAME_DATE': _test(
            'Check if two date columns consistently contain the same date',
            ('Check if two date columns consistently contain the same date, but may have '
             'different times.'),
            checker._check_same_date,
            checker._generate_same_date,
            True, True, False, False
        ),
        'SAME_MONTH': _test(
            'Check if two date columns consistently contain the same month',
            ('Check if two date columns consistently contain the same month, but may have '
             'different days or times.'),
            checker._check_same_month,
            checker._generate_same_month,
            True, True, False, False
        ),
        'CORRELATED_DATES': _test(
            '',
            'Check if two date columns are correlated',
            checker._check_correlated_dates,
            checker._generate_correlated_dates,
            True, True, False, False
        ),

        # Tests on two columns, where one is date and the other is numeric
        'LARGE_GIVEN_DATE': _test(
            'Check if a numeric value is large given the value in a date column',
            ('Check if a numeric value is very large given the value in a given date column.'),
            checker._check_large_given_date,
            checker._generate_large_given_date,
            True, True, False, False
        ),
        'SMALL_GIVEN_DATE': _test(
            'Check if a numeric value is small given the value in a date column',
            ('Check if a numeric value is very small given the value in a given date column.'),
            checker._check_small_given_date,
            checker._generate_small_given_date,
            True, True, False, False
        ),
    }
