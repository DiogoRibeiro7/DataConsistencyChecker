"""
Base Test Definitions for DataConsistencyChecker

This module contains test definitions for single columns and pairs of columns
of any type (numeric, categorical, string, date, etc.).

Test Categories:
- Single column tests (any type): MISSING_VALUES, RARE_VALUES, UNIQUE_VALUES, PREV_VALUES_DT
- Pair column tests (any type): MATCHED_MISSING, OPPOSITE_MISSING, SAME_VALUES,
                                 SAME_OR_CONSTANT, UNIQUE_PAIR
"""


def get_base_tests(checker):
    """
    Get test definitions for base tests on single and pair columns of any type.

    Args:
        checker: DataConsistencyChecker instance to bind test methods

    Returns:
        dict: Test definitions for base tests
    """
    return {
        # Tests on single columns of any type
        'MISSING_VALUES': (
            'Check if all values in a column are consistently present / missing',
            ('Check if all values in a column are consistently present / consistently missing.'),
            checker._DataConsistencyChecker__check_missing,
            checker._DataConsistencyChecker__generate_missing,
            False, True, True, False
        ),
        'RARE_VALUES': (
            '',
            'Check if there are any rare values in a column.',
            checker._DataConsistencyChecker__check_rare_values,
            checker._DataConsistencyChecker__generate_rare_values,
            False, True, True, False
        ),
        'UNIQUE_VALUES': (
            '',
            'Check if there are consistently unique values with a column.',
            checker._DataConsistencyChecker__check_unique_values,
            checker._DataConsistencyChecker__generate_unique_values,
            True, True, True, False
        ),
        'PREV_VALUES_DT': (
            'Check if the values in a column can be predicted from previous values',
            ('Check if the values in a column can be predicted from previous values in '
             'that column using a simple decision tree.'),
            checker._DataConsistencyChecker__check_prev_values_dt,
            checker._DataConsistencyChecker__generate_prev_values_dt,
            True, True, True, False
        ),

        # Tests on pairs of columns of any type
        'MATCHED_MISSING': (
            '',
            'Check if two columns have missing values consistently in the same rows.',
            checker._DataConsistencyChecker__check_matched_missing,
            checker._DataConsistencyChecker__generate_matched_missing,
            True, True, False, False
        ),
        'OPPOSITE_MISSING': (
            'Check if two columns have null values consistently in different rows',
            ('Check if two columns both frequently have null values, but consistently '
             'not in the same rows.'),
            checker._DataConsistencyChecker__check_opposite_missing,
            checker._DataConsistencyChecker__generate_opposite_missing,
            True, True, False, False
        ),
        'SAME_VALUES': (
            '',
            'Check if two columns consistently have the same values.',
            checker._DataConsistencyChecker__check_same,
            checker._DataConsistencyChecker__generate_same,
            True, True, False, False
        ),
        'SAME_OR_CONSTANT': (
            'Check for values matching another column, or small set of other values',
            ('Check one column consistently has either the same value as another column, '
             'or a small number of other values.'),
            checker._DataConsistencyChecker__check_same_or_constant,
            checker._DataConsistencyChecker__generate_same_or_constant,
            True, True, False, False
        ),
        'UNIQUE_PAIR': (
            '',
            'Check if two columns consistently have a unique pair of values.',
            checker._DataConsistencyChecker__check_unique_pair,
            checker._DataConsistencyChecker__generate_unique_pair,
            True, True, False, False
        ),
    }
