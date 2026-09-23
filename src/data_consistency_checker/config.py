"""Immutable configuration for reproducible DataConsistencyChecker analyses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConsistencyConfig:
    """Configuration for a one-shot data consistency analysis."""

    iqr_limit: float = 3.5
    idr_limit: float = 1.0
    max_combinations: int = 100_000
    verbose: int = 0
    known_date_cols: tuple[str, ...] = ()
    execute_tests: tuple[str, ...] = ()
    exclude_tests: tuple[str, ...] = ()
    fast_only: bool = False
    include_code_tests: bool = True
    freq_contamination_level: int | float = 0.005
    rare_contamination_level: int | float = 0.1
    run_parallel: bool = False
    raise_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate mutually exclusive and bounded configuration values."""
        if self.execute_tests and self.exclude_tests:
            raise ValueError("execute_tests and exclude_tests are mutually exclusive")
        if self.max_combinations <= 0:
            raise ValueError("max_combinations must be greater than zero")
        if self.verbose not in {-1, 0, 1, 2}:
            raise ValueError("verbose must be one of -1, 0, 1, or 2")
        if self.freq_contamination_level < 0:
            raise ValueError("freq_contamination_level must be non-negative")
        if self.rare_contamination_level < 0:
            raise ValueError("rare_contamination_level must be non-negative")


__all__ = ["DataConsistencyConfig"]
