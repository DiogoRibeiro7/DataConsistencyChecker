"""
String Test Definitions for DataConsistencyChecker

This module contains test definitions for string/text columns including:
- Single string column tests (character patterns, word patterns)
- Pair string column tests (similarity, relationships)
- String-numeric combination tests
"""


def get_string_tests(checker):
    """
    Get test definitions for string/text column tests.

    Args:
        checker: DataConsistencyChecker instance to bind test methods

    Returns:
        dict: Test definitions for string tests
    """
    return {
        # Tests on single string columns
        'BLANK_VALUES': (
            '',
            'Check for blank strings and values that are entirely whitespace.',
            checker._DataConsistencyChecker__check_blank,
            checker._DataConsistencyChecker__generate_blank,
            False, True, True, False
        ),
        'LEADING_WHITESPACE': (
            '',
            'Check for strings with unusual leading whitespace for the column.',
            checker._DataConsistencyChecker__check_leading_whitespace,
            checker._DataConsistencyChecker__generate_leading_whitespace,
            True, True, True, False
        ),
        'TRAILING_WHITESPACE': (
            '',
            'Check for blank strings with unusual trailing whitespace for the column.',
            checker._DataConsistencyChecker__check_trailing_whitespace,
            checker._DataConsistencyChecker__generate_trailing_whitespace,
            True, True, True, False
        ),
        'FIRST_CHAR_ALPHA': (
            '',
            ('Check if the first characters are consistently alphabetic within a column. '
             'Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_first_char_alpha,
            checker._DataConsistencyChecker__generate_first_char_alpha,
            False, True, True, True
        ),
        'FIRST_CHAR_NUMERIC': (
            '',
            ('Check if the first characters are consistently numeric within a column. '
             'Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_first_char_numeric,
            checker._DataConsistencyChecker__generate_first_char_numeric,
            True, True, True, True
        ),
        'FIRST_CHAR_SMALL_SET': (
            'Check for columns with a small number of first characters',
            ('Check if there are a small number of distinct characters used for the first '
             'character within a column. Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_first_char_small_set,
            checker._DataConsistencyChecker__generate_first_char_small_set,
            True, True, True, True
        ),
        'FIRST_CHAR_UPPERCASE': (
            '',
            'Check if the first character is consistently uppercase within a column.',
            checker._DataConsistencyChecker__check_first_char_uppercase,
            checker._DataConsistencyChecker__generate_first_char_uppercase,
            True, True, True, False
        ),
        'FIRST_CHAR_LOWERCASE': (
            '',
            'Check if the first character is consistently lowercase within a column.',
            checker._DataConsistencyChecker__check_first_char_lowercase,
            checker._DataConsistencyChecker__generate_first_char_lowercase,
            False, True, True, False
        ),
        'LAST_CHAR_SMALL_SET': (
            'Check for columns with a small number of last characters',
            ('Check if there are a small number of distinct characters used for the last '
             'character within a column. Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_last_char_small_set,
            checker._DataConsistencyChecker__generate_last_char_small_set,
            True, True, True, True
        ),
        'COMMON_SPECIAL_CHARS': (
            'Check for special characters that are in most values.',
            ('Check if there are one or more non-alphanumeric characters that consistently '
             'appear in the values within a column.'),
            checker._DataConsistencyChecker__check_common_special_chars,
            checker._DataConsistencyChecker__generate_common_special_chars,
            True, True, True, False
        ),
        'COMMON_CHARS': (
            'Check for characters that are in most values',
            ('Check if there is consistently a small number of characters repeated in each '
             'value in a column. Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_common_chars,
            checker._DataConsistencyChecker__generate_common_chars,
            True, True, True, True
        ),
        'NUMBER_ALPHA_CHARS': (
            'Check for features with a consistent number of alphabetic characters.',
            ('Check if there is a consistent number of alphabetic characters in each value in '
             'a column. Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_number_alpha_chars,
            checker._DataConsistencyChecker__generate_number_alpha_chars,
            True, True, True, True
        ),
        'NUMBER_NUMERIC_CHARS': (
            'Check for features with a consistent number of numeric characters.',
            ('Check if there is a consistent number of numeric characters in each value in '
             'a column. Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_number_numeric_chars,
            checker._DataConsistencyChecker__generate_number_numeric_chars,
            True, True, True, True
        ),
        'NUMBER_ALPHANUMERIC_CHARS': (
            'Check for features with a consistent number of alpha-numeric characters.',
            ('Check if there is a consistent number of alphanumeric characters in each value in '
             'a column. Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_number_alphanumeric_chars,
            checker._DataConsistencyChecker__generate_number_alphanumeric_chars,
            True, True, True, True
        ),
        'NUMBER_NON-ALPHANUMERIC_CHARS': (
            'Check for features with a consistent number of non-alpha-numeric characters.',
            ('Check if there is a consistent number of non-alphanumeric characters in each '
             'value in a column. Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_number_non_alphanumeric_chars,
            checker._DataConsistencyChecker__generate_number_non_alphanumeric_chars,
            True, True, True, True
        ),
        'NUMBER_CHARS': (
            'Check for features with a consistent number of characters.',
            ('Check if there is a consistent number of characters in each value in a column.'),
            checker._DataConsistencyChecker__check_number_chars,
            checker._DataConsistencyChecker__generate_number_chars,
            True, True, True, False
        ),
        'NONPRINTABLE_CHARS': (
            '',
            'Check for features with non-printable characters.',
            checker._DataConsistencyChecker__check_nonprintable_chars,
            checker._DataConsistencyChecker__generate_nonprintable_chars,
            True, True, True, False
        ),
        'MANY_CHARS': (
            '',
            ('Check if any values have an unusually large number of characters for the column.'),
            checker._DataConsistencyChecker__check_many_chars,
            checker._DataConsistencyChecker__generate_many_chars,
            False, True, True, False
        ),
        'FEW_CHARS': (
            '',
            ('Check if any values have an unusually small number of characters for the column.'),
            checker._DataConsistencyChecker__check_few_chars,
            checker._DataConsistencyChecker__generate_few_chars,
            False, True, True, False
        ),
        'POSITION_NON-ALPHANUMERIC': (
            '',
            ('Check if the positions of the non-alphanumeric characters is consistent within a column.'),
            checker._DataConsistencyChecker__check_position_non_alphanumeric,
            checker._DataConsistencyChecker__generate_position_non_alphanumeric,
            True, True, True, False
        ),
        'CHARS_PATTERN': (
            'Check for features with a consistent pattern of characters',
            ('Check if there is a consistent pattern of alphabetic, numeric and special '
             'characters in each value in a column.'),
            checker._DataConsistencyChecker__check_chars_pattern,
            checker._DataConsistencyChecker__generate_chars_pattern,
            True, True, True, False
        ),
        'UPPERCASE': (
            '',
            ('Check if all alphabetic characters in a column are consistently uppercase.'),
            checker._DataConsistencyChecker__check_uppercase,
            checker._DataConsistencyChecker__generate_uppercase,
            True, True, True, False
        ),
        'LOWERCASE': (
            '',
            ('Check if all alphabetic characters in a column are consistently lowercase.'),
            checker._DataConsistencyChecker__check_lowercase,
            checker._DataConsistencyChecker__generate_lowercase,
            False, True, True, False
        ),
        'CHARACTERS_USED': (
            '',
            ('Check if there is a consistent set of characters used in each value in a column. '
             'Intended primarily for ID/code columns.'),
            checker._DataConsistencyChecker__check_characters_used,
            checker._DataConsistencyChecker__generate_characters_used,
            True, True, True, True
        ),
        'FIRST_WORD_SMALL_SET': (
            'Check for words that are in most values',
            ('Check if there is a small set of words consistently used for the first word of '
             'each value in a column.'),
            checker._DataConsistencyChecker__check_first_word,
            checker._DataConsistencyChecker__generate_first_word,
            True, True, True, False
        ),
        'LAST_WORD_SMALL_SET': (
            '',
            ('Check if there is a small set of words consistently used for the last word of '
             'each value in a column.'),
            checker._DataConsistencyChecker__check_last_word,
            checker._DataConsistencyChecker__generate_last_word,
            True, True, True, False
        ),
        'NUMBER_WORDS': (
            '',
            ('Check if there is a consistent number of words used in each value in a column.'),
            checker._DataConsistencyChecker__check_num_words,
            checker._DataConsistencyChecker__generate_num_words,
            True, True, True, False
        ),
        'LONGEST_WORDS': (
            '',
            'Check if a column contains any unusually long words.',
            checker._DataConsistencyChecker__check_longest_words,
            checker._DataConsistencyChecker__generate_longest_words,
            True, True, True, False
        ),
        'COMMON_WORDS': (
            '',
            ('Check if there is a consistent set of words used in each value in a column.'),
            checker._DataConsistencyChecker__check_words_used,
            checker._DataConsistencyChecker__generate_words_used,
            True, True, True, False
        ),
        'RARE_WORDS': (
            '',
            'Check if there are words which occur rarely in a given column.',
            checker._DataConsistencyChecker__check_rare_words,
            checker._DataConsistencyChecker__generate_rare_words,
            True, True, True, False
        ),
        'GROUPED_STRINGS': (
            '',
            'Check if a string or binary column is sorted into groups.',
            checker._DataConsistencyChecker__check_grouped_strings,
            checker._DataConsistencyChecker__generate_grouped_strings,
            True, True, True, False
        ),

        # Tests on pairs string columns
        'A_IMPLIES_B': (
            '',
            ('Check if specific values in one categorical column imply specific values in '
             'another categorical column.'),
            checker._DataConsistencyChecker__check_a_implies_b,
            checker._DataConsistencyChecker__generate_a_implies_b,
            True, False, False, False
        ),
        'RARE_PAIRS': (
            '',
            ('Check for pairs of values in two columns, where neither is rare, but the '
             'combination is rare.'),
            checker._DataConsistencyChecker__check_rare_pairs,
            checker._DataConsistencyChecker__generate_rare_pairs,
            True, True, False, False
        ),
        'RARE_PAIRS_FIRST_CHAR': (
            'Check in pairs of columns for rare pairs of first character',
            ('Check for pairs of values in two columns, where neither begins with a rare '
             'character, but the combination or first characters is rare. Intended primarily '
             'for pairs of ID/code columns.'),
            checker._DataConsistencyChecker__check_rare_pair_first_char,
            checker._DataConsistencyChecker__generate_rare_pair_first_char,
            True, True, False, True
        ),
        'RARE_PAIRS_FIRST_WORD': (
            '',
            ('Check for pairs of values in two columns, where neither begins with a rare word, '
             'but the combination of words is rare.'),
            checker._DataConsistencyChecker__check_rare_pair_first_word,
            checker._DataConsistencyChecker__generate_rare_pair_first_word,
            True, True, False, False
        ),
        'RARE_PAIRS_FIRST_WORD_VAL': (
            '',
            ('Check for pairs of values in two columns, where the combination of the first word '
             'in one and the value in the other is rare.'),
            checker._DataConsistencyChecker__check_rare_pair_first_word_val,
            checker._DataConsistencyChecker__generate_rare_pair_first_word_val,
            True, True, False, False
        ),
        'SIMILAR_CHARACTERS': (
            '',
            ('Check if two string columns, with one word each, consistently have a significant '
             'overlap in the characters used.'),
            checker._DataConsistencyChecker__check_similar_chars,
            checker._DataConsistencyChecker__generate_similar_chars,
            True, True, False, False
        ),
        'SIMILAR_NUM_CHARS': (
            '',
            ('Check if two string columns consistently have similar numbers of characters while '
             'the range of string lengths varies within both columns.'),
            checker._DataConsistencyChecker__check_similar_num_chars,
            checker._DataConsistencyChecker__generate_similar_num_chars,
            True, True, False, False
        ),
        'SIMILAR_WORDS': (
            '',
            ('Check if two string columns consistently have a significant overlap in the words used.'),
            checker._DataConsistencyChecker__check_similar_words,
            checker._DataConsistencyChecker__generate_similar_words,
            True, True, False, False
        ),
        'SIMILAR_NUM_WORDS': (
            '',
            'Check if two string columns consistently have similar numbers of words.',
            checker._DataConsistencyChecker__check_similar_num_words,
            checker._DataConsistencyChecker__generate_similar_num_words,
            True, True, False, False
        ),
        'SAME_FIRST_CHARS': (
            '',
            ('Check if two string columns consistently start with the same set of characters. '
             'Intended primarily for pairs of ID/code columns.'),
            checker._DataConsistencyChecker__check_same_first_chars,
            checker._DataConsistencyChecker__generate_same_first_chars,
            True, True, False, True
        ),
        'SAME_FIRST_WORD': (
            '',
            'Check if two string columns consistently start with the same word.',
            checker._DataConsistencyChecker__check_same_first_word,
            checker._DataConsistencyChecker__generate_same_first_word,
            True, True, False, False
        ),
        'SAME_LAST_WORD': (
            '',
            'Check if two string columns consistently end with the same word.',
            checker._DataConsistencyChecker__check_same_last_word,
            checker._DataConsistencyChecker__generate_same_last_word,
            True, True, False, False
        ),
        'SAME_ALPHA_CHARS': (
            '',
            ('Check if two string columns consistently contain the same set of alphabetic '
             'characters. Intended primarily for pairs of ID/code columns.'),
            checker._DataConsistencyChecker__check_same_alpha_chars,
            checker._DataConsistencyChecker__generate_same_alpha_chars,
            True, True, False, True
        ),
        'SAME_NUMERIC_CHARS': (
            '',
            ('Check if two string columns consistently contain the same set of numeric characters. '
             'Intended primarily for pairs of ID/code columns.'),
            checker._DataConsistencyChecker__check_same_numeric_chars,
            checker._DataConsistencyChecker__generate_same_numeric_chars,
            True, True, False, True
        ),
        'SAME_SPECIAL_CHARS': (
            '',
            ('Check if two string columns consistently contain the same set of special characters.'),
            checker._DataConsistencyChecker__check_same_special_chars,
            checker._DataConsistencyChecker__generate_same_special_chars,
            True, True, False, False
        ),
        'A_PREFIX_OF_B': (
            '',
            'Check if one column is consistently the prefix of another column.',
            checker._DataConsistencyChecker__check_a_prefix_of_b,
            checker._DataConsistencyChecker__generate_a_prefix_of_b,
            True, True, False, False
        ),
        'A_SUFFIX_OF_B': (
            '',
            'Check if one column is consistently the suffix of another column.',
            checker._DataConsistencyChecker__check_a_suffix_of_b,
            checker._DataConsistencyChecker__generate_a_suffix_of_b,
            True, True, False, False
        ),
        'B_CONTAINS_A': (
            '',
            ('Check if one column is consistently contained in another columns, but is neither '
             'the prefix, nor suffix of the second column.'),
            checker._DataConsistencyChecker__check_b_contains_a,
            checker._DataConsistencyChecker__generate_b_contains_a,
            True, True, False, False
        ),
        'CORRELATED_ALPHA_ORDER': (
            '',
            ('Check if the alphabetic orderings of two columns are consistently correlated.'),
            checker._DataConsistencyChecker__check_correlated_alpha,
            checker._DataConsistencyChecker__generate_correlated_alpha,
            True, True, False, False
        ),

        # Tests with one string and one numeric column
        'LARGE_GIVEN_VALUE': (
            '',
            ('Check if a value in a numeric column is very large given the value in a '
             'categorical column.'),
            checker._DataConsistencyChecker__check_large_given,
            checker._DataConsistencyChecker__generate_large_given,
            True, True, False, False
        ),
        'SMALL_GIVEN_VALUE': (
            '',
            ('Check if a value in a numeric column is very small given the value in a '
             'categorical column.'),
            checker._DataConsistencyChecker__check_small_given,
            checker._DataConsistencyChecker__generate_small_given,
            True, True, False, False
        ),
        'LARGE_GIVEN_PREFIX': (
            '',
            ('Check if a value in a numeric column is very large given the first word in a '
             'categorical column.'),
            checker._DataConsistencyChecker__check_large_given_prefix,
            checker._DataConsistencyChecker__generate_large_given_prefix,
            True, True, False, False
        ),
        'SMALL_GIVEN_PREFIX': (
            '',
            ('Check if a value in a numeric column is very small given the first word in a '
             'categorical column.'),
            checker._DataConsistencyChecker__check_small_given_prefix,
            checker._DataConsistencyChecker__generate_small_given_prefix,
            True, True, False, False
        ),
        'GROUPED_STRINGS_BY_NUMERIC': (
            '',
            ('Check if a string or binary column is sorted into groups when the table is ordered '
             'by a numeric or date column.'),
            checker._DataConsistencyChecker__check_grouped_strings_by_numeric,
            checker._DataConsistencyChecker__generate_grouped_strings_by_numeric,
            True, True, False, False
        ),

        # Tests on two string and one numeric column
        'LARGE_GIVEN_PAIR': (
            '',
            ('Check if a value in a numeric or date column is large given a pair of values in '
             'two string or binary columns.'),
            checker._DataConsistencyChecker__check_large_given_pair,
            checker._DataConsistencyChecker__generate_large_given_pair,
            True, True, False, False
        ),
        'SMALL_GIVEN_PAIR': (
            '',
            ('Check if a value in a numeric or date column is small given the a of values in two '
             'string or binary columns.'),
            checker._DataConsistencyChecker__check_small_given_pair,
            checker._DataConsistencyChecker__generate_small_given_pair,
            True, True, False, False
        ),

        # Tests on one string/binary column and two numeric
        'CORRELATED_GIVEN_VALUE': (
            '',
            ('Check if two numeric columns are correlated if conditioning on a string or binary column.'),
            checker._DataConsistencyChecker__check_corr_given_val,
            checker._DataConsistencyChecker__generate_corr_given_val,
            True, True, False, False
        ),

        # Tests on one string column related to the rest of the columns
        'DECISION_TREE_CLASSIFIER': (
            '',
            ('Check if a categorical column can be derived from the other columns using a decision tree.'),
            checker._DataConsistencyChecker__check_dt_classifier,
            checker._DataConsistencyChecker__generate_dt_classifier,
            True, True, False, False
        ),
    }
