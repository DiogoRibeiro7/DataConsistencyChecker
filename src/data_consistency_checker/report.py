"""Structured, serializable result objects for DataConsistencyChecker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON-safe primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass

    if value != value:
        return None
    return str(value)


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

    @classmethod
    def from_values(
        cls,
        *,
        n_rows: int,
        n_columns: int,
        executed_tests: tuple[str, ...],
        patterns: tuple[dict[str, Any], ...],
        exceptions: tuple[dict[str, Any], ...],
        row_scores: tuple[dict[str, Any], ...],
        execution_failures: tuple[dict[str, Any], ...],
    ) -> "DataConsistencyReport":
        """Build a report while normalizing nested values for JSON output."""
        return cls(
            n_rows=int(n_rows),
            n_columns=int(n_columns),
            executed_tests=tuple(str(test_id) for test_id in executed_tests),
            patterns=tuple(_json_safe(item) for item in patterns),
            exceptions=tuple(_json_safe(item) for item in exceptions),
            row_scores=tuple(_json_safe(item) for item in row_scores),
            execution_failures=tuple(
                _json_safe(item) for item in execution_failures
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""
        return _json_safe(asdict(self))


__all__ = ["DataConsistencyReport"]
