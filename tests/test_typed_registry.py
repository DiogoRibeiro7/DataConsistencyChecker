"""Tests for the typed internal test registry."""

from __future__ import annotations

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker.test_registry import TestDefinition


def test_live_registry_contains_only_typed_definitions() -> None:
    """All aggregated test definitions are normalized before execution."""
    checker = DataConsistencyChecker(verbose=-1)

    assert checker.test_dict
    assert all(
        isinstance(definition, TestDefinition)
        for definition in checker.test_dict.values()
    )


def test_test_definition_named_fields_are_executable() -> None:
    """Representative registry entries expose named executable fields."""
    checker = DataConsistencyChecker(verbose=-1)
    definition = checker.test_dict["MISSING_VALUES"]

    assert definition.implemented is True
    assert callable(definition.test_func)
    assert definition.gen_func is not None
    assert isinstance(definition.description, str)
    assert isinstance(definition.fast, bool)
