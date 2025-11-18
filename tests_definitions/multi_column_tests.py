"""
Multi-Column Test Definitions for DataConsistencyChecker

This module contains test definitions for tests operating on:
- Sets of three or more columns (any type)
- Row-level aggregation tests
- Complex multi-column relationships
"""


def get_multi_column_tests(checker):
    """
    Get test definitions for multi-column and row-level tests.

    Args:
        checker: DataConsistencyChecker instance to bind test methods

    Returns:
        dict: Test definitions for multi-column tests
    """
    return {
        # Tests on sets of three columns of any type
        'C_IS_A_OR_B': (
            '',
            ('Check if one column is consistently equal to the value in one of two other columns, '
             'though not consistently either one of the two columns.'),
            checker._DataConsistencyChecker__check_c_is_a_or_b,
            checker._DataConsistencyChecker__generate_c_is_a_or_b,
            True, True, False, False
        ),

        # Tests on sets of four columns of any type
        'TWO_PAIRS': (
            '',
            ('Check that, given two pairs of columns, the first pair of columns have matching '
             'values in, and only in, the same rows as the other pair of columns.'),
            checker._DataConsistencyChecker__check_two_pairs,
            checker._DataConsistencyChecker__generate_two_pairs,
            True, True, False, False
        ),

        # Tests on sets of columns of any type
        'UNIQUE_SETS_VALUES': (
            '',
            'Check if a set of columns has consistently unique combinations of values.',
            checker._DataConsistencyChecker__check_unique_sets_values,
            checker._DataConsistencyChecker__generate_unique_sets_values,
            True, True, False, False
        ),

        # Tests on rows of values
        'MISSING_VALUES_PER_ROW': (
            '',
            'Check if there is a consistent number of missing values per row.',
            checker._DataConsistencyChecker__check_missing_values_per_row,
            checker._DataConsistencyChecker__generate_missing_values_per_row,
            True, True, True, False
        ),
        'ZERO_VALUES_PER_ROW': (
            '',
            'Check if there is a consistent number of zero values per row.',
            checker._DataConsistencyChecker__check_zero_values_per_row,
            checker._DataConsistencyChecker__generate_zero_values_per_row,
            True, True, True, False
        ),
        'UNIQUE_VALUES_PER_ROW': (
            '',
            'Check if there is a consistent number of unique values per row.',
            checker._DataConsistencyChecker__check_unique_values_per_row,
            checker._DataConsistencyChecker__generate_unique_values_per_row,
            True, True, True, False
        ),
        'NEGATIVE_VALUES_PER_ROW': (
            '',
            'Check if there is a consistent number of negative values per row.',
            checker._DataConsistencyChecker__check_negative_values_per_row,
            checker._DataConsistencyChecker__generate_negative_values_per_row,
            True, True, True, False
        ),
        'SMALL_AVG_RANK_PER_ROW': (
            'Check for rows with many small values',
            ('Check if the numeric values in a row have a small average percentile value relative '
             'to their columns. This indicates the numeric values in a row are typically unusually '
             'small for their columns.'),
            checker._DataConsistencyChecker__check_small_avg_rank_per_row,
            checker._DataConsistencyChecker__generate_small_avg_rank_per_row,
            False, True, True, False
        ),
        'LARGE_AVG_RANK_PER_ROW': (
            'Check for rows with many large values',
            ('Check if the numeric values in a row have a large average percentile value relative '
             'to their columns. This indicates the numeric values in a row are typically unusually '
             'large for their columns.'),
            checker._DataConsistencyChecker__check_large_avg_rank_per_row,
            checker._DataConsistencyChecker__generate_large_avg_rank_per_row,
            False, True, True, False
        ),
    }
