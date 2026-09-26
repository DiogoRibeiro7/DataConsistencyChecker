"""Execution orchestration for DataConsistencyChecker."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from dataexcept import OutlierDetectionError, exception_to_dict
from dataexcept import wrap as wrap_dataexcept

try:
    from termcolor import colored
except ImportError:  # pragma: no cover - optional presentation dependency
    colored = None


class ExecutionMixin:
    """Mixin providing consistency-test execution orchestration."""

    def _record_execution_failure(
        self,
        test_id: str,
        error: Exception,
        *,
        raise_on_error: bool,
    ) -> OutlierDetectionError:
        """Translate and retain one failed consistency-test execution.

        The original exception is preserved as ``__cause__`` by DataExcept.
        The serialized envelope is retained so callers can inspect failures
        without parsing console output.

        Args:
            test_id: Identifier of the consistency test that failed.
            error: Original exception raised by the test implementation.
            raise_on_error: If True, re-raise the structured failure.

        Returns:
            The structured DataExcept error representing the failed test.
        """
        structured = wrap_dataexcept(
            error,
            OutlierDetectionError,
            method=test_id,
            details=str(error),
        )
        self.execution_failures.append(
            {
                "test_id": test_id,
                "error": exception_to_dict(structured),
            }
        )
        self.num_exceptions += 1

        if raise_on_error:
            raise structured from error
        return structured

    def check_data_quality(
        self,
        append_results: bool = False,
        execute_list: list[str] | None = None,
        exclude_list: list[str] | None = None,
        test_start_id: int = 0,
        fast_only: bool = False,
        include_code_tests: bool = True,
        freq_contamination_level: float = 0.005,
        rare_contamination_level: float = 0.1,  # noqa: ARG002
        run_parallel: bool = False,
        raise_on_error: bool = False,
    ) -> None:
        """
        Execute data quality tests on the dataset specified in init_data().

        Identifies patterns and exceptions in the data. Use additional API calls
        to access and analyze the results.

        Args:
            append_results: If True, append to previous results; if False, clear previous results
            execute_list: Specific test IDs to execute. If None, runs all tests
            exclude_list: Test IDs to exclude. Cannot be used with execute_list
            test_start_id: Test number to start from (for resuming incomplete runs)
            fast_only: If True, run only fast single-column tests
            include_code_tests: If True, include tests for code/ID value columns
            freq_contamination_level: Max fraction (or count) of rows violating a pattern
                for tests that frequently find results. Lower values reduce false positives
            rare_contamination_level: Max fraction (or count) of rows violating a pattern
                for tests that rarely find results. Higher values reduce false negatives
            run_parallel: Not supported; if True, a RuntimeWarning is issued and the tests run sequentially
            raise_on_error: If True, stop on the first failed test and raise a
                structured OutlierDetectionError. If False, retain failures and
                continue running the remaining tests.

        Returns:
            None. Results stored in instance variables accessible via other methods.

        Raises:
            AssertionError: If both execute_list and exclude_list are specified
            OutlierDetectionError: If a test fails and raise_on_error is True
        """

        if self.orig_df is None or len(self.orig_df) == 0:
            print("Valid dataframe not specified, possibly due to not calling init_data(), or passing a null dataframe")
            return

        # execute_list and exclude_list should not both be set.
        assert execute_list is None or exclude_list is None

        # The set of tests that are to be run / not run may optionally be specified. This may also effect any synthetic
        # data which is generated.
        self.execute_list = execute_list
        self.exclude_list = exclude_list
        self.execution_test_list = []  # The complete list for one call to check_data_quality()

        # Check if the specified tests are valid
        specified_test_list = []
        if exclude_list:
            specified_test_list.extend(exclude_list)
        if execute_list:
            specified_test_list.extend(execute_list)
        for t in specified_test_list:
            if t not in self.test_dict:
                print(f"Error {t} is not a valid test")

        # Store the contamination_level in terms of number of rows. It may have been passed either in this form or as a
        # fraction.
        if freq_contamination_level >= 1 and type(freq_contamination_level) is int:
            if freq_contamination_level > len(self.orig_df):
                print(f"Error. contamination rate set to {freq_contamination_level}, more than the number of rows in "
                       f"the dataframe passed. The contamination_level rate should be substantially smaller.")
                return
            self.freq_contamination_level = freq_contamination_level
        elif freq_contamination_level < 1.0:
            self.freq_contamination_level = freq_contamination_level * len(self.orig_df)
            if self.freq_contamination_level < 1.0:
                if freq_contamination_level != 0.005:
                    print(f"Error. contamination rate set to {freq_contamination_level}, not allowing even 1 row to "
                           f"be in violation of the patterns. Must be set to a larger value given the number of rows "
                           "available.")
                    return
                self.freq_contamination_level = 1

        # Adjust the test_start_id to 0 if necessary. 0 is the lowest valid value.
        if test_start_id < 0:
            test_start_id = 0

        # Determine the set of tests to execute
        self.execution_test_list = []
        for test_idx, test_id in enumerate(self.test_dict.keys()):
            if test_idx < test_start_id:
                continue
            if fast_only and not self.test_dict[test_id].fast:
                continue
            if (not include_code_tests) and self.test_dict[test_id].code:
                continue
            if ((self.execute_list is None and self.exclude_list is None) or \
                    (self.execute_list and test_id in self.execute_list) or \
                    (self.exclude_list and test_id not in self.exclude_list)) and self.test_dict[test_id].implemented:
                self.execution_test_list.append(test_id)

        # Check at least one valid test was specified
        if len(self.execution_test_list) == 0:
            print("No valid tests specified.")
            return

        # Initialize the variables related to the run
        self.n_tests_executed = 0
        self.execution_times = {}
        self.num_exceptions = 0
        self.execution_failures = []

        # Initialize the variables related to the results found, unless append_results is specified.
        if not append_results:
            self.patterns_arr = []
            self.patterns_df = None
            self.results_summary_arr = []
            self.exceptions_summary_df = None
            self.test_results_df = None
            self.results_dict = {}
            self.test_results_by_column_np = np.zeros((self.num_rows, len(self.orig_df.columns)), dtype=float)

        # Get the test index of each test
        test_idx_dict = {x: y for x, y in
                         zip(self.test_dict.keys(), range(len(self.test_dict.keys())))
                         if self.test_dict[x].implemented}

        # Initialize the results for tests on single columns
        self.single_test_summary_dict = {}
        self.single_test_summary_df = None

        if run_parallel:
            # Tests record their results on the checker itself, so tests run in worker processes could not report
            # results back. Previously this silently produced no results.
            warnings.warn(
                "run_parallel is not supported: running the tests sequentially instead.",
                RuntimeWarning,
                stacklevel=2,
            )

        for test_id in self.execution_test_list:
            self._output_current_test(test_idx_dict[test_id], test_id)
            try:
                t1 = time.time()
                self.test_dict[test_id].test_func(test_id=test_id)
                t2 = time.time()
                self.execution_times[test_id] = t2 - t1
            except Exception as error:
                structured = self._record_execution_failure(
                    test_id,
                    error,
                    raise_on_error=raise_on_error,
                )
                message = f"Error executing {test_id}: {structured}"
                if colored:
                    print(colored(message, "red"))
                else:
                    print(message)
            self.n_tests_executed += 1

        # Populate the test_results_df dataframe with all results found
        self.test_results_df = pd.DataFrame(self.results_dict)

        # Sum the the number of columns flagged per row
        self._calculate_final_scores()

        # Create the final dataframes of patterns and exceptions found
        self.patterns_df = pd.DataFrame(
            self.patterns_arr,
            columns=['Test ID', 'Column(s)', 'Description of Pattern', 'Display Information'])

        self.exceptions_summary_df = pd.DataFrame(
            self.results_summary_arr,
            columns=['Test ID', 'Column(s)', 'Description of Pattern', 'Number of Exceptions', 'Display Information'])

        # Add the Pattern Id column to patterns_df
        self.patterns_df['Pattern ID'] = list(range(len(self.patterns_df)))

        # Add the Issue Id column to exceptions_summary_df
        self.exceptions_summary_df['Issue ID'] = list(range(len(self.exceptions_summary_df)))

        # Display summary statistics
        self._output_stats()

        # Save safe versions of the patterns and exceptions found, to support restore_issues() if called
        self.safe_patterns_arr = self.patterns_arr.copy()
        self.safe_patterns_df = self.patterns_df.copy()
        self.safe_results_summary_arr = self.results_summary_arr.copy()
        self.safe_exceptions_summary_df = self.exceptions_summary_df.copy()
        self.safe_test_results_df = self.test_results_df.copy()
        self.safe_results_dict = self.results_dict.copy()
        self.safe_test_results_by_column_np = self.test_results_by_column_np.copy()

        # Initialize variables used by display_next()
        self.found_tests = self.get_test_ids_with_results()
        self.current_display_test = 0

