"""Tests for the typed consistency-check catalog."""

from __future__ import annotations

from data_consistency_checker import DataConsistencyChecker, TestMetadata


def test_catalog_returns_typed_metadata() -> None:
    checker = DataConsistencyChecker(verbose=-1)

    catalog = checker.get_test_catalog()

    assert catalog
    assert all(isinstance(item, TestMetadata) for item in catalog)
    assert all(item.implemented for item in catalog)
    assert {item.test_id for item in catalog} == set(checker.get_test_list())


def test_catalog_metadata_is_json_safe() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    item = next(
        entry for entry in checker.get_test_catalog()
        if entry.test_id == "MISSING_VALUES"
    )

    payload = item.to_dict()

    assert payload["test_id"] == "MISSING_VALUES"
    assert isinstance(payload["description"], str)
    assert isinstance(payload["fast"], bool)
    assert "test_func" not in payload
    assert "gen_func" not in payload
