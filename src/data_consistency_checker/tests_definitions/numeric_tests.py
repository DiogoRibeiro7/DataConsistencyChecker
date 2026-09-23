"""
Numeric Test Definitions for DataConsistencyChecker

This module contains test definitions for numeric columns including:
- Single numeric column tests
- Pair numeric column tests
- Triple numeric column tests (3-column relationships)
- Multi-column numeric tests and aggregations
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




def get_numeric_tests(checker):
    """
    Get test definitions for numeric column tests.

    Args:
        checker: DataConsistencyChecker instance to bind test methods

    Returns:
        dict: Test definitions for numeric tests
    """
    return {
        # Tests on single numeric columns
        'POSITIVE': _test(
            '',
            'Check if all numbers in a column are positive.',
            checker._check_positive_values,
            checker._generate_positive_values,
            False, True, True, False
        ),
        'NEGATIVE': _test(
            '',
            'Check if all numbers in a column are negative.',
            checker._check_negative_values,
            checker._generate_negative_values,
            True, True, True, False
        ),
        'NUMBER_DECIMALS': _test(
            'Check if there is a consistent number of decimal digits',
            ('Check if there is a consistent number of decimal digits in each value in a column.'),
            checker._check_number_decimals,
            checker._generate_number_decimals,
            False, True, True, False
        ),
        'RARE_DECIMALS': _test(
            '',
            'Check if there are any uncommon sets of digits after the decimal point.',
            checker._check_rare_decimals,
            checker._generate_rare_decimals,
            True, True, True, False
        ),
        'COLUMN_ORDERED_ASC': _test(
            '',
            'Check if a column is monotonically increasing.',
            checker._check_column_increasing,
            checker._generate_column_increasing,
            True, True, True, False
        ),
        'COLUMN_ORDERED_DESC': _test(
            '',
            'Check if a column is monotonically decreasing.',
            checker._check_column_decreasing,
            checker._generate_column_decreasing,
            True, True, True, False
        ),
        'COLUMN_TENDS_ASC': _test(
            '',
            'Check if a column is generally increasing.',
            checker._check_column_tends_asc,
            checker._generate_column_tends_asc,
            True, True, True, False
        ),
        'COLUMN_TENDS_DESC': _test(
            '',
            'Check if a column is generally decreasing.',
            checker._check_column_tends_desc,
            checker._generate_column_tends_desc,
            True, True, True, False
        ),
        'SIMILAR_PREVIOUS': _test(
            'Check if all values are similar to the previous value in the column',
            ('Check if all values are similar to the previous value in the column, '
             'relative to the range of values in the column.'),
            checker._check_similar_previous,
            checker._generate_similar_previous,
            True, True, True, False
        ),
        'UNUSUAL_ORDER_MAGNITUDE': _test(
            'Check if any values have an unusual order of magnitude',
            ('Check if there are any unusual numeric values, in the sense of having an '
             'unusual order of magnitude for the column.'),
            checker._check_unusual_order_magnitude,
            checker._generate_unusual_order_magnitude,
            False, True, True, False
        ),
        'FEW_NEIGHBORS': _test(
            'Check if any values have no similar values',
            ('Check if there are any unusual numeric values, in the sense of being '
             'distant from both the next smallest and next largest values within the column.'),
            checker._check_few_neighbors,
            checker._generate_few_neighbors,
            False, True, True, False
        ),
        'FEW_WITHIN_RANGE': _test(
            'Check if any values have few similar values',
            ('Check if there are any unusual numeric values, in the sense of having few '
             'other values in the column within a small range.'),
            checker._check_few_within_range,
            checker._generate_few_within_range,
            False, True, True, False
        ),
        'VERY_SMALL': _test(
            '',
            'Check if there are any very small values relative to their column.',
            checker._check_very_small,
            checker._generate_very_small,
            True, True, True, False
        ),
        'VERY_LARGE': _test(
            '',
            'Check if there are any very large values relative to their column.',
            checker._check_very_large,
            checker._generate_very_large,
            True, True, True, False
        ),
        'VERY_SMALL_ABS': _test(
            'Check for very small absolute values',
            'Check if there are any very small absolute values relative to its column.',
            checker._check_very_small_abs,
            checker._generate_very_small_abs,
            True, True, True, False
        ),
        'MULTIPLE_OF_CONSTANT': _test(
            '',
            'Check if all values in a column are multiples of some constant.',
            checker._check_multiple_constant,
            checker._generate_multiple_constant,
            True, True, True, False
        ),
        'ROUNDING': _test(
            '',
            'Check if all values in a column are rounded to the same degree.',
            checker._check_rounding,
            checker._generate_rounding,
            True, True, True, False
        ),
        'NON_ZERO': _test(
            '',
            'Check if all values in a column are non-zero.',
            checker._check_non_zero,
            checker._generate_non_zero,
            False, True, True, False
        ),
        'LESS_THAN_ONE': _test(
            '',
            'Check if all values in a column are between -1.0 and 1.0, inclusive.',
            checker._check_less_than_one,
            checker._generate_less_than_one,
            True, True, True, False
        ),
        'GREATER_THAN_ONE': _test(
            'Check if all values are between -1.0 and 1.0',
            ('Check if all values in a column are less than -1.0 or greater than 1.0, inclusive.'),
            checker._check_greater_than_one,
            checker._generate_greater_than_one,
            False, True, True, False
        ),
        'INVALID_NUMBERS': _test(
            'Check for invalid characters in numeric columns',
            ('Check for values in numeric columns that are not valid numbers, including '
             'values that include parenthesis, brackets, percent signs and other values.'),
            checker._check_invalid_numbers,
            checker._generate_invalid_numbers,
            False, True, True, False
        ),

        # Tests on pairs of numeric columns
        'LARGER_DIFF_RANGE': _test(
            'Check if one column is consistently larger than another',
            ('Check if one column is consistently larger than another, but within one '
             'order of magnitude, where the two columns have different ranges of values.'),
            checker._check_larger_diff_range,
            checker._generate_larger_diff_range,
            False, True, False, False
        ),
        'LARGER_SAME_RANGE': _test(
            'Check if one column is consistently larger than another row by row',
            ('Check if one column is consistently larger than another column, where the '
             'two columns have the same range of values'),
            checker._check_larger_same_range,
            checker._generate_larger_same_range,
            True, True, False, False
        ),
        'MUCH_LARGER': _test(
            'Check if one column is consistently significantly larger than another',
            ('Check if one column is consistently at least one order of magnitude larger than another.'),
            checker._check_much_larger,
            checker._generate_much_larger,
            False, True, False, False
        ),
        'SIMILAR_WRT_RATIO': _test(
            'Check if two columns are consistently similar (first test)',
            ('Check if two columns are consistently similar, with respect to their ratio, to each other.'),
            checker._check_similar_wrt_ratio,
            checker._generate_similar_wrt_ratio,
            True, True, False, False
        ),
        'SIMILAR_WRT_DIFF': _test(
            'Check if two columns are consistently similar (second test)',
            ('Check if two columns are consistently similar, with respect to absolute difference, to each other.'),
            checker._check_similar_wrt_difference,
            checker._generate_similar_wrt_difference,
            True, True, False, False
        ),
        'SIMILAR_TO_INVERSE': _test(
            'Check if one column is the inverse of another. ',
            ('Check if one column is consistently similar to the inverse of another column.'),
            checker._check_similar_to_inverse,
            checker._generate_similar_to_inverse,
            True, True, False, False
        ),
        'SIMILAR_TO_NEGATIVE': _test(
            'Check if one column is the negative of another.',
            ('Check if one column is consistently similar to the negative of another column.'),
            checker._check_similar_to_negative,
            checker._generate_similar_to_negative,
            True, True, False, False
        ),
        'CONSTANT_SUM': _test(
            'Check if two columns have a consistent sum.',
            ('Check if the sum of two columns is consistently similar to a constant value.'),
            checker._check_constant_sum,
            checker._generate_constant_sum,
            True, True, False, False
        ),
        'CONSTANT_DIFF': _test(
            'Check if two columns have a consistent difference.',
            ('Check if the difference between two columns is consistently similar to a constant value.'),
            checker._check_constant_diff,
            checker._generate_constant_diff,
            True, True, False, False
        ),
        'CONSTANT_PRODUCT': _test(
            'Check if two columns have a consistent product.',
            ('Check if the product of two columns is consistently similar to a constant value.'),
            checker._check_constant_product,
            checker._generate_constant_product,
            True, True, False, False
        ),
        'CONSTANT_RATIO': _test(
            'Check if two columns have a consistent ratio.',
            ('Check if the ratio of two columns is consistently similar to a constant value.'),
            checker._check_constant_ratio,
            checker._generate_constant_ratio,
            True, True, False, False
        ),
        'EVEN_MULTIPLE': _test(
            '',
            'Check if one column is consistently an even integer multiple of another.',
            checker._check_even_multiple,
            checker._generate_even_multiple,
            True, True, False, False
        ),
        'RARE_COMBINATION': _test(
            '',
            'Check if two columns have any unusual pairs of values.',
            checker._check_rare_combination,
            checker._generate_rare_combination,
            True, True, False, False
        ),
        'CORRELATED_NUMERIC': _test(
            '',
            'Check if two numeric columns are consistently correlated.',
            checker._check_correlated,
            checker._generate_correlated,
            True, True, False, False
        ),
        'MATCHED_ZERO': _test(
            '',
            'Check if two columns have a value of zero consistently in the same rows.',
            checker._check_matched_zero,
            checker._generate_matched_zero,
            True, True, False, False
        ),
        'OPPOSITE_ZERO': _test(
            '',
            ('Check if two columns are consistently such that one column contains a '
             'zero and the other contains a non-zero value.'),
            checker._check_opposite_zero,
            checker._generate_opposite_zero,
            True, True, False, False
        ),
        'RUNNING_SUM': _test(
            '',
            ('Check if one column is consistently the sum of its own value from the '
             'previous row and another column in the current row.'),
            checker._check_running_sum,
            checker._generate_running_sum,
            True, True, False, False
        ),
        'A_ROUNDED_B': _test(
            '',
            ('Check if one column is consistently the result of rounding another column.'),
            checker._check_a_rounded_b,
            checker._generate_a_rounded_b,
            True, True, False, False
        ),

        # Tests on pairs of columns where one must be numeric
        'MATCHED_ZERO_MISSING': _test(
            'Check for pairs of columns with zero in one and null in the other',
            ('Check if two columns consistently have a zero in one column and a '
             'missing value in the other.'),
            checker._check_matched_zero_missing,
            checker._generate_matched_zero_missing,
            True, True, False, False
        ),

        # Tests on sets of 3 numeric columns
        'SIMILAR_TO_DIFF': _test(
            'Check for columns similar to the difference in two others',
            ('Check if one column is consistently similar to the difference of two other columns.'),
            checker._check_similar_to_diff,
            checker._generate_similar_to_diff,
            True, True, False, False
        ),
        'DIFF_EXACT': _test(
            '',
            ('Check if one column is consistently exactly the difference of two other columns.'),
            checker._check_diff_exact,
            checker._generate_diff_exact,
            True, False, False, False
        ),
        'SIMILAR_TO_PRODUCT': _test(
            'Check for columns similar to the product of two other',
            ('Check if one column is consistently similar to the product of two other columns.'),
            checker._check_similar_to_product,
            checker._generate_similar_to_product,
            True, True, False, False
        ),
        'PRODUCT_EXACT': _test(
            '',
            ('Check if one column is consistently exactly the product of two other columns.'),
            checker._check_product_exact,
            checker._generate_product_exact,
            True, False, False, False
        ),
        'SIMILAR_TO_RATIO': _test(
            'Check for columns similar to the ratio of two other',
            ('Check if one column is consistently similar to the ratio of two other columns.'),
            checker._check_similar_to_ratio,
            checker._generate_similar_to_ratio,
            True, True, False, False
        ),
        'RATIO_EXACT': _test(
            '',
            ('Check if one column is consistently exactly the ratio of two other columns.'),
            checker._check_ratio_exact,
            checker._generate_ratio_exact,
            True, False, False, False
        ),
        'LARGER_THAN_SUM': _test(
            'Check for columns larger than to sum of two others',
            ('Check if one column is consistently larger than the sum of two other columns.'),
            checker._check_larger_than_sum,
            checker._generate_larger_than_sum,
            False, True, False, False
        ),
        'LARGER_THAN_ABS_DIFF': _test(
            'Check for columns larger than the absolute difference of two others',
            ('Check if one column is consistently larger than the difference between two other columns.'),
            checker._check_larger_than_abs_diff,
            checker._generate_larger_than_abs_diff,
            False, False, False, False
        ),

        # Tests on single numeric columns in relation to all other numeric columns
        'SUM_OF_COLUMNS': _test(
            'Check if one column is the sum of two others.',
            ('Check if one column is consistently similar to the sum of two or more other columns.'),
            checker._check_sum_of_columns,
            checker._generate_sum_of_columns,
            True, True, False, False
        ),
        'MEAN_OF_COLUMNS': _test(
            'Check if one column is the mean of a set of other columns.',
            ('Check if one column is consistently similar to the mean of two or more other columns.'),
            checker._check_mean_of_columns,
            checker._generate_mean_of_columns,
            True, True, False, False
        ),
        'MIN_OF_COLUMNS': _test(
            'Check if one column is the minimum of a set of other columns.',
            ('Check if one column is consistently similar to the minimum of two or more other columns.'),
            checker._check_min_of_columns,
            checker._generate_min_of_columns,
            True, True, False, False
        ),
        'MAX_OF_COLUMNS': _test(
            'Check if one column is the maximum of a set of other columns.',
            ('Check if one column is consistently similar to the maximum of two or more other columns.'),
            checker._check_max_of_columns,
            checker._generate_max_of_columns,
            True, True, False, False
        ),
        'MATCHED_SET_POS_NEG': _test(
            'Check for sets of columns that are positive and negative together.',
            ('Identify sets of columns where the values are consistently either all positive, or all negative.'),
            checker.check_mathed_set_pos_neg,
            checker.generate_mathed_set_pos_neg,
            True, True, False, False
        ),
        'MATCHED_SET_ZERO_NON_ZERO': _test(
            'Check for sets of columns that are zero and non-zero together.',
            ('Identify sets of columns where the values are consistently either all zero or non-zero.'),
            checker.check_matched_set_zero_non_zero,
            checker.generate_matched_set_zero_non_zero,
            True, True, False, False
        ),
        'DECISION_TREE_REGRESSOR': _test(
            'Check for columns that can be predicted from the other columns from a tree',
            ('Check if a numeric column can be derived from the other columns using a small decision tree.'),
            checker._check_dt_regressor,
            checker._generate_dt_regressor,
            True, True, False, False
        ),
        'LINEAR_REGRESSION': _test(
            'Check for columns that can be predicted from the others with a regression',
            ('Check if a numeric column can be derived from the other numeric columns using linear regression.'),
            checker._check_lin_regressor,
            checker._generate_lin_regressor,
            True, True, False, False
        ),
        'SMALL_VS_CORR_COLS': _test(
            '',
            ('Check if a value has an unusually small rank within its column compared '
             'to other ranks within that row for correlated columns.'),
            checker._check_small_vs_corr_cols,
            checker._generate_small_vs_corr_cols,
            False, False, False, False
        ),
        'LARGE_VS_CORR_COLS': _test(
            '',
            ('Check if a value has an unusually large rank within its column compared '
             'to other ranks within that row for correlated columns.'),
            checker._check_large_vs_corr_cols,
            checker._generate_large_vs_corr_cols,
            False, False, False, False
        ),
        'PREDICT_NULL_DT': _test(
            'Check for columns where can predict the null values with a decision tree',
            ('Check if the Null values in one column can be predicted from the values in the other columns.'),
            checker._check_predict_null,
            checker._generate_predict_null,
            True, True, False, False
        ),
    }
