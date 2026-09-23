"""Tests for config serialization and file loading."""

from __future__ import annotations

import json

import pytest

from data_consistency_checker import DataConsistencyConfig


def test_config_round_trip_dict() -> None:
    config = DataConsistencyConfig(
        execute_tests=("MISSING_VALUES", "VERY_LARGE"),
        known_date_cols=("event_date",),
        verbose=-1,
    )

    restored = DataConsistencyConfig.from_dict(config.to_dict())

    assert restored == config


def test_config_loads_json(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "execute_tests": ["MISSING_VALUES"],
                "known_date_cols": ["event_date"],
                "verbose": -1,
            }
        ),
        encoding="utf-8",
    )

    config = DataConsistencyConfig.from_file(path)

    assert config.execute_tests == ("MISSING_VALUES",)
    assert config.known_date_cols == ("event_date",)
    assert config.verbose == -1


def test_config_loads_toml_table(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """[data_consistency_checker]
execute_tests = ["MISSING_VALUES"]
known_date_cols = ["event_date"]
verbose = -1
""",
        encoding="utf-8",
    )

    config = DataConsistencyConfig.from_file(path)

    assert config.execute_tests == ("MISSING_VALUES",)
    assert config.known_date_cols == ("event_date",)
    assert config.verbose == -1


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown configuration field"):
        DataConsistencyConfig.from_dict({"does_not_exist": True})


def test_config_rejects_unsupported_file_format(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("verbose: -1", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported config format"):
        DataConsistencyConfig.from_file(path)
