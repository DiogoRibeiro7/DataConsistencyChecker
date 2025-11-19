"""
Binary Column Test Definitions for DataConsistencyChecker

This module contains test definitions for binary columns including:
- Pair binary column tests (same, opposite, implies)
- Sets of binary columns (AND, OR, XOR operations)
- Binary-numeric combinations
- Binary-string combinations
"""


def get_binary_tests(checker):
    """
    Get test definitions for binary column tests.

    Args:
        checker: DataConsistencyChecker instance to bind test methods

    Returns:
        dict: Test definitions for binary tests
    """
    return {
        # Tests on pairs of binary columns
        'BINARY_SAME': (
            'Check if two binary columns are the same.',
            ('For each pair of binary columns with the same set of two values, check if they '
             'consistently have the same value.'),
            checker._check_binary_same,
            checker._generate_binary_same,
            True, True, False, False
        ),
        'BINARY_OPPOSITE': (
            'Check if two binary columns have opposite values.',
            ('For each pair of binary columns with the same set of two values, check if they '
             'consistently have the opposite value.'),
            checker._check_binary_opposite,
            checker._generate_binary_opposite,
            True, True, False, False
        ),
        'BINARY_IMPLIES': (
            'Check if one value in a binary column implies a value in another column.',
            ('For each pair of binary columns with the same set of two values, check if when '
             'one has a given value, the other consistently does as well, though the other '
             'direction may not be true.'),
            checker._check_binary_implies,
            checker._generate_binary_implies,
            True, True, False, False
        ),

        # Tests on sets of binary columns
        'BINARY_AND': (
            'Check if one column is the AND of other binary columns.',
            ('For sets of binary columns with the same set of two values, check if one column '
             'is consistently the result of ANDing the other columns.'),
            checker._check_binary_and,
            checker._generate_binary_and,
            True, True, False, False
        ),
        'BINARY_OR': (
            'Check if one column is the OR of other binary columns.',
            ('For sets of binary columns with the same set of two values, check if one column '
             'is consistently the result of ORing the other columns.'),
            checker._check_binary_or,
            checker._generate_binary_or,
            True, True, False, False
        ),
        'BINARY_XOR': (
            'Check if one column is the XOR of other binary columns.',
            ('For sets of binary columns with the same set of two values, check if one column '
             'is consistently the result of XORing the other columns.'),
            checker._check_binary_xor,
            checker._generate_binary_xor,
            True, True, False, False
        ),
        'BINARY_NUM_SAME': (
            'Check for sets of columns with a constant number of matching values',
            ('For sets of binary columns with the same set of two values, check if there is a '
             'consistent number of these columns with the same value.'),
            checker._check_binary_num_same,
            checker._generate_binary_num_same,
            True, True, False, False
        ),
        'BINARY_RARE_COMBINATION': (
            '',
            'Check for rare sets of values in sets of three or more binary columns.',
            checker._check_binary_rare_combo,
            checker._generate_binary_rare_combo,
            True, True, False, False
        ),

        # Tests on pairs of columns where one is binary and one is numeric
        'BINARY_MATCHES_VALUES': (
            'Check for binary columns that match the values in a numeric column',
            ('Check if the binary column is consistently one value when the values in a numeric '
             'column have low values, or when they have high values.'),
            checker._check_binary_matches_values,
            checker._generate_binary_matches_values,
            True, True, False, False
        ),

        # Tests on sets of three columns, where one must be binary
        'BINARY_TWO_OTHERS_MATCH': (
            'Check for binary columns that indicate if two other columns match',
            ('Check if a binary column is consistently one value when two other columns have '
             'the same value as each other.'),
            checker._check_binary_two_others_match,
            checker._generate_binary_two_others_match,
            True, True, False, False
        ),

        # Tests on sets of three columns, where one is binary and the other two string
        'BINARY_TWO_STR_SIMILAR': (
            '',
            ('Check if a binary column is consistently one value when two other string have '
             'similar values as each other, with respect to string length and the characters used.'),
            checker._check_binary_two_str_match,
            checker._generate_binary_two_str_match,
            True, False, False, False
        ),

        # Tests on sets of multiple columns, where one is binary and the others are numeric
        'BINARY_MATCHES_SUM': (
            'Check for binary columns that match the sum of two numeric columns',
            ('Check if the binary column is consistently true when the sum of a set of numeric '
             'columns is over some threshold.'),
            checker._check_binary_matches_sum,
            checker._generate_binary_matches_sum,
            True, True, False, False
        ),
    }
