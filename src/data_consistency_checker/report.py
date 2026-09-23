"""Structured, serializable result objects for DataConsistencyChecker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DataConsistencyReport:
    """Serializable snapshot of one checker analysis state."""

    n_rows: int
    n_columns: int
    executed_tests: tuple[str, ...]
    patterns: tuple[dict[str, Any], ...]
    exceptions: tuple[dict[str, Any], ...]
    row_scores: tuple[dict[str, Any], ...]
    execution_failures: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


__all__ = ["DataConsistencyReport"]
