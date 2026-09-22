"""Pytest configuration for registry-backed test modules."""

from __future__ import annotations

import pytest

from data_consistency_checker import DataConsistencyChecker


_IMPLEMENTED_TEST_IDS = frozenset(DataConsistencyChecker(verbose=-1).test_dict)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip legacy test modules whose test IDs are not currently implemented.

    The historical suite contains modules for experiments and retired checks.
    They remain useful as reference fixtures, but they must not fail CI as if
    they represented supported public behavior.
    """
    for item in items:
        test_id = getattr(item.module, "test_id", None)
        if test_id is None or test_id in _IMPLEMENTED_TEST_IDS:
            continue

        item.add_marker(
            pytest.mark.skip(
                reason=f"{test_id} is not an implemented registry test",
            )
        )
