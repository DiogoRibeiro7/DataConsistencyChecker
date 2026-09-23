"""
Base Test Definitions for DataConsistencyChecker

This module contains test definitions for single columns and pairs of columns
of any type (numeric, categorical, string, date, etc.).

Test Categories:
- Single column tests (any type): MISSING_VALUES, RARE_VALUES, UNIQUE_VALUES, PREV_VALUES_DT
- Pair column tests (any type): MATCHED_MISSING, OPPOSITE_MISSING, SAME_VALUES,
                                 SAME_OR_CONSTANT, UNIQUE_PAIR
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
        'MISSING_VALUES': _test(
            'Check if all values in a column are consistently present / missing',
            ('Check if all values in a column are consistently present / consistently missing.'),
            checker._check_missing,
            checker._generate_missing,
            False, True, True, False
        ),
        'RARE_VALUES': _test(
            '',
            'Check if there are any rare values in a column.',
            checker._check_rare_values,
            checker._generate_rare_values,
            False, True, True, False
        ),
        'UNIQUE_VALUES': _test(
            '',
            'Check if there are consistently unique values with a column.',
            checker._check_unique_values,
            checker._generate_unique_values,
            True, True, True, False
        ),
        'PREV_VALUES_DT': _test(
            'Check if the values in a column can be predicted from previous values',
            ('Check if the values in a column can be predicted from previous values in '
             'that column using a simple decision tree.'),
            checker._check_prev_values_dt,
            checker._generate_prev_values_dt,
            True, True, True, False
        ),

        # Tests on pairs of columns of any type
        'MATCHED_MISSING': _test(
            '',
            'Check if two columns have missing values consistently in the same rows.',
            checker._check_matched_missing,
            checker._generate_matched_missing,
            True, True, False, False
        ),
        'OPPOSITE_MISSING': _test(
            'Check if two columns have null values consistently in different rows',
            ('Check if two columns both frequently have null values, but consistently '
             'not in the same rows.'),
            checker._check_opposite_missing,
            checker._generate_opposite_missing,
            True, True, False, False
        ),
        'SAME_VALUES': _test(
            '',
            'Check if two columns consistently have the same values.',
            checker._check_same,
            checker._generate_same,
            True, True, False, False
        ),
        'SAME_OR_CONSTANT': _test(
            'Check for values matching another column, or small set of other values',
            ('Check one column consistently has either the same value as another column, '
             'or a small number of other values.'),
            checker._check_same_or_constant,
            checker._generate_same_or_constant,
            True, True, False, False
        ),
        'UNIQUE_PAIR': _test(
            '',
            'Check if two columns consistently have a unique pair of values.',
            checker._check_unique_pair,
            checker._generate_unique_pair,
            True, True, False, False
        ),
    }
