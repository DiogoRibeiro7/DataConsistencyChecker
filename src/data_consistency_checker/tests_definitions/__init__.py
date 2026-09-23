"""
Tests Definitions Package for DataConsistencyChecker

This package organizes the 164 tests into logical modules by category:
- base_tests: Single and pair column tests (any type)
- numeric_tests: Numeric column tests
- date_tests: Date/time column tests
- string_tests: String column tests
- binary_tests: Binary column tests
- multi_column_tests: Tests on 3+ columns

Each module defines test metadata using the structure:
    test_id: (short_desc, description, test_func, gen_func, shortlist, implemented, fast, code)
"""

from .base_tests import get_base_tests
from .numeric_tests import get_numeric_tests
from .date_tests import get_date_tests
from .string_tests import get_string_tests
from .binary_tests import get_binary_tests
from .multi_column_tests import get_multi_column_tests

__all__ = [
    'get_base_tests',
    'get_numeric_tests',
    'get_date_tests',
    'get_string_tests',
    'get_binary_tests',
    'get_multi_column_tests',
    'get_all_test_definitions',
]


def get_all_test_definitions(checker_instance):
    """
    Aggregate all test definitions from all modules.

    Args:
        checker_instance: Instance of DataConsistencyChecker to bind test methods

    Returns:
        dict[str, TestDefinition]: Complete typed test dictionary
    """
    test_dict = {}

    # Collect tests from all modules
    test_dict.update(get_base_tests(checker_instance))
    test_dict.update(get_numeric_tests(checker_instance))
    test_dict.update(get_date_tests(checker_instance))
    test_dict.update(get_string_tests(checker_instance))
    test_dict.update(get_binary_tests(checker_instance))
    test_dict.update(get_multi_column_tests(checker_instance))

    return test_dict
