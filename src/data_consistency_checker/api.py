"""High-level one-shot analysis API."""

from __future__ import annotations

import pandas as pd

from .checker import DataConsistencyChecker
from .config import DataConsistencyConfig
from .report import DataConsistencyReport


def analyze(
    df: pd.DataFrame,
    config: DataConsistencyConfig | None = None,
) -> DataConsistencyReport:
    """Analyze a dataframe and return a structured report in one call."""
    resolved = DataConsistencyConfig() if config is None else config
    checker = DataConsistencyChecker(
        iqr_limit=resolved.iqr_limit,
        idr_limit=resolved.idr_limit,
        max_combinations=resolved.max_combinations,
        verbose=resolved.verbose,
    )
    checker.init_data(
        df,
        known_date_cols=list(resolved.known_date_cols) or None,
    )
    checker.check_data_quality(
        execute_list=list(resolved.execute_tests) or None,
        exclude_list=list(resolved.exclude_tests) or None,
        fast_only=resolved.fast_only,
        include_code_tests=resolved.include_code_tests,
        freq_contamination_level=resolved.freq_contamination_level,
        rare_contamination_level=resolved.rare_contamination_level,
        run_parallel=resolved.run_parallel,
        raise_on_error=resolved.raise_on_error,
    )
    return checker.get_report()


__all__ = ["analyze"]
