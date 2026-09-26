"""Integration tests for structured consistency-test execution failures."""

from __future__ import annotations

import pandas as pd
import pytest
from dataexcept import OutlierDetectionError

from check_data_consistency import DataConsistencyChecker
from data_consistency_checker.test_registry import TestDefinition

TEST_ID = "DATAEXCEPT_FAILURE_TEST"


def _failing_test(*, test_id: str) -> None:
    """Deterministic failing test used to exercise the real execution loop."""
    raise ValueError(f"{test_id} exploded")


def _checker_with_failing_test() -> DataConsistencyChecker:
    """Return an initialized checker with one injected failing test."""
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(
        pd.DataFrame(
            {
                "value": list(range(12)),
                "group": ["a", "b"] * 6,
            }
        )
    )
    checker.test_dict[TEST_ID] = TestDefinition(
        short_description="Intentional execution failure",
        description="Intentional execution failure for DataExcept integration testing.",
        test_func=_failing_test,
        gen_func=None,
        shortlist=False,
        implemented=True,
        fast=True,
        code=False,
    )
    return checker


def test_failed_test_is_retained_as_dataexcept_envelope() -> None:
    """A failed consistency test remains inspectable without aborting the run."""
    checker = _checker_with_failing_test()

    checker.check_data_quality(execute_list=[TEST_ID])

    failures = checker.get_execution_failures()
    assert checker.num_exceptions == 1
    assert len(failures) == 1
    assert failures[0]["test_id"] == TEST_ID

    envelope = failures[0]["error"]
    assert envelope["type"] == "OutlierDetectionError"
    assert envelope["attributes"]["method"] == TEST_ID
    assert "exploded" in envelope["attributes"]["details"]
    assert envelope["cause"]["type"] == "ValueError"
    assert TEST_ID in envelope["cause"]["message"]


def test_execution_failure_copy_is_defensive() -> None:
    """Callers cannot mutate the checker by editing the returned diagnostics."""
    checker = _checker_with_failing_test()
    checker.check_data_quality(execute_list=[TEST_ID])

    failures = checker.get_execution_failures()
    failures[0]["test_id"] = "changed"

    assert checker.get_execution_failures()[0]["test_id"] == TEST_ID


def test_execution_failures_reset_for_each_quality_run() -> None:
    """Execution diagnostics describe the latest call, not all historical calls."""
    checker = _checker_with_failing_test()

    checker.check_data_quality(execute_list=[TEST_ID])
    checker.check_data_quality(execute_list=[TEST_ID])

    assert checker.num_exceptions == 1
    assert len(checker.get_execution_failures()) == 1


def test_raise_on_error_raises_structured_failure_with_cause() -> None:
    """Fail-fast mode raises DataExcept while retaining the original exception."""
    checker = _checker_with_failing_test()

    with pytest.raises(OutlierDetectionError) as captured:
        checker.check_data_quality(
            execute_list=[TEST_ID],
            raise_on_error=True,
        )

    error = captured.value
    assert error.method == TEST_ID
    assert isinstance(error.__cause__, ValueError)
    assert TEST_ID in str(error.__cause__)
    assert len(checker.get_execution_failures()) == 1
