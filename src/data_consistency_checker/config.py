"""Immutable configuration for reproducible DataConsistencyChecker analyses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


_TUPLE_FIELDS = {"known_date_cols", "execute_tests", "exclude_tests"}


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration dictionary."""
        payload = asdict(self)
        for name in _TUPLE_FIELDS:
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DataConsistencyConfig":
        """Build a validated config from a dictionary-like object."""
        valid_fields = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - valid_fields)
        if unknown:
            raise ValueError(
                "Unknown configuration field(s): " + ", ".join(unknown)
            )

        normalized = dict(values)
        for name in _TUPLE_FIELDS:
            if name in normalized:
                value = normalized[name]
                if value is None:
                    normalized[name] = ()
                elif isinstance(value, str):
                    normalized[name] = (value,)
                else:
                    normalized[name] = tuple(str(item) for item in value)
        return cls(**normalized)

    @classmethod
    def from_file(cls, path: str | Path) -> "DataConsistencyConfig":
        """Load configuration from JSON or TOML."""
        config_path = Path(path)
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
            raw = raw.get("data_consistency_checker", raw)
        else:
            raise ValueError(
                f"Unsupported config format {suffix!r}. Use JSON or TOML."
            )

        if not isinstance(raw, dict):
            raise ValueError("Configuration file must contain an object/table")
        return cls.from_dict(raw)


__all__ = ["DataConsistencyConfig"]
