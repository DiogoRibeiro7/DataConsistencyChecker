"""Typed test-definition aggregation for DataConsistencyChecker."""

from __future__ import annotations

from data_consistency_checker.test_registry import TestDefinition

from .base_tests import get_base_tests
from .binary_tests import get_binary_tests
from .date_tests import get_date_tests
from .multi_column_tests import get_multi_column_tests
from .numeric_tests import get_numeric_tests
from .string_tests import get_string_tests

__all__ = [
    "get_base_tests",
    "get_numeric_tests",
    "get_date_tests",
    "get_string_tests",
    "get_binary_tests",
    "get_multi_column_tests",
    "get_all_test_definitions",
]


def get_all_test_definitions(checker_instance) -> dict[str, TestDefinition]:
    """Aggregate all typed test definitions."""
    test_dict: dict[str, TestDefinition] = {}
    test_dict.update(get_base_tests(checker_instance))
    test_dict.update(get_numeric_tests(checker_instance))
    test_dict.update(get_date_tests(checker_instance))
    test_dict.update(get_string_tests(checker_instance))
    test_dict.update(get_binary_tests(checker_instance))
    test_dict.update(get_multi_column_tests(checker_instance))
    return test_dict
