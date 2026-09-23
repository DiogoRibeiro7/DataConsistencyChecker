"""Tests for directly typed test-definition modules."""

from __future__ import annotations

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker.test_registry import TestDefinition
from data_consistency_checker.tests_definitions.base_tests import get_base_tests
from data_consistency_checker.tests_definitions.binary_tests import get_binary_tests
from data_consistency_checker.tests_definitions.date_tests import get_date_tests
from data_consistency_checker.tests_definitions.multi_column_tests import (
    get_multi_column_tests,
)
from data_consistency_checker.tests_definitions.numeric_tests import get_numeric_tests
from data_consistency_checker.tests_definitions.string_tests import get_string_tests


def test_core_definition_modules_return_typed_definitions() -> None:
    """Every definition module should return typed definitions directly."""
    checker = DataConsistencyChecker(verbose=-1)
    factories = (
        get_base_tests,
        get_binary_tests,
        get_date_tests,
        get_multi_column_tests,
        get_numeric_tests,
        get_string_tests,
    )

    for factory in factories:
        definitions = factory(checker)
        assert definitions
        assert all(
            isinstance(definition, TestDefinition)
            for definition in definitions.values()
        )
