from __future__ import annotations

import copy
from collections.abc import Sequence

import matplotlib.pyplot as plt

from .analysis_cache_mixin import AnalysisCacheMixin
from .checker_utils import library_call
from .data_init_mixin import DataInitMixin

# Mixins
from .display_mixin import DisplayMixin
from .execution_mixin import ExecutionMixin
from .export_mixin import ExportMixin
from .plots_mixin import PlotsMixin
from .results_mixin import ResultsMixin
from .synth_data_mixin import SynthDataMixin

# Test implementation mixins
from .test_implementations import (
    BaseTestsMixin,
    BinaryTestsMixin,
    DateTestsMixin,
    MultiColumnTestsMixin,
    NumericTestsMixin,
    StringTestsMixin,
)


class DataConsistencyChecker(BaseTestsMixin, NumericTestsMixin, DateTestsMixin, StringTestsMixin, BinaryTestsMixin, MultiColumnTestsMixin, DisplayMixin, PlotsMixin, SynthDataMixin, ResultsMixin, AnalysisCacheMixin, DataInitMixin, ExecutionMixin, ExportMixin):
    """
    Automated data quality checker performing 164 tests to identify patterns and anomalies.

    This class examines tabular datasets for consistency patterns across single columns,
    pairs of columns, and larger column sets. It identifies both patterns and exceptions
    to those patterns, useful for EDA and interpretable outlier detection.
    """

    ##################################################################################################################
    # Tune the contamination rate
    ##################################################################################################################
    @library_call
    def test_contamination_level(
        self,
        contamination_levels_arr: Sequence[float] | None = None,
        execute_list: list[str] | None = None,
        exclude_list: list[str] | None = None,
        fast_only: bool = False,
        include_code_tests: bool = True,
        plot: bool = True,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Show how the findings depend on the contamination level, to help choose one.

        A pattern is only recognised when fewer rows than the contamination level break it, and exceptions are only
        found where a pattern is recognised. A small level finds only strong exceptions but may miss interesting
        patterns; a larger level finds more, with more noise. This runs the checks once per level, on a copy of the
        checker, so the results of the last `check_data_quality()` call are kept.

        Args:
            contamination_levels_arr: The levels to try, each a fraction of the rows (below 1) or a number of rows (an
                integer), as `freq_contamination_level` in `check_data_quality()`. A fraction smaller than one row
                counts as one row. Defaults to 0.0001, 0.0005, 0.001, 0.005, 0.01 and 0.05.
            execute_list: Only run these checks.
            exclude_list: Run every check except these.
            fast_only: Only run the fast checks.
            include_code_tests: Include the checks specific to code or ID values.
            plot: Plot the three counts against the level.

        Returns:
            tuple[list[int], list[int], list[int]]: One value per level in each list: the number of issues (patterns
                with exceptions), the number of rows flagged at least once, and the number of columns flagged at least
                once.

        Raises:
            ValueError: If no data was loaded with `init_data()`, or a level is neither a fraction below 1 nor a
                positive number of rows up to the number of rows loaded.
        """
        if contamination_levels_arr is None:
            contamination_levels_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        if self.orig_df is None or len(self.orig_df) == 0:
            raise ValueError("No data to check: call init_data() first.")
        for level in contamination_levels_arr:
            is_fraction = isinstance(level, float) and 0 < level < 1
            is_row_count = isinstance(level, int) and 1 <= level <= self.num_rows
            if not (is_fraction or is_row_count):
                raise ValueError(f"Invalid contamination level {level!r}: use a fraction of the rows below 1, or a "
                                 f"number of rows from 1 to {self.num_rows}.")

        # Run on a copy, so this checker keeps the results of its last run
        checker = copy.deepcopy(self)
        num_issues, num_rows_flagged, num_cols_flagged = [], [], []
        for level in contamination_levels_arr:
            checker.check_data_quality(
                execute_list=execute_list,
                exclude_list=exclude_list,
                fast_only=fast_only,
                include_code_tests=include_code_tests,
                freq_contamination_level=level if level * self.num_rows >= 1 or level >= 1 else 1,
            )
            scores = checker.test_results_by_column_np
            num_issues.append(0 if checker.exceptions_summary_df is None else len(checker.exceptions_summary_df))
            num_rows_flagged.append(int((scores.sum(axis=1) > 0).sum()))
            num_cols_flagged.append(int((scores.sum(axis=0) > 0).sum()))

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
            titles = ["Issues found", "Rows flagged at least once", "Columns flagged at least once"]
            for ax, counts, title in zip(axes, [num_issues, num_rows_flagged, num_cols_flagged], titles):
                ax.plot(contamination_levels_arr, counts, marker="o")
                ax.set_xscale("log")
                ax.set_xlabel("Contamination level")
                ax.set_title(title)
            fig.tight_layout()
            plt.show()
        return num_issues, num_rows_flagged, num_cols_flagged
