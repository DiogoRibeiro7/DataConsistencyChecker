"""Integration tests for the command-line interface."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data_consistency_checker.cli import main


def _write_csv(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "value": list(range(12)),
            "group": ["a", "b"] * 6,
        }
    ).to_csv(path, index=False)
    return path


def test_list_tests_outputs_implemented_ids(capsys) -> None:
    """The CLI exposes the live implemented test registry."""
    exit_code = main(["list-tests"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "MISSING_VALUES" in output


def test_list_tests_json_is_machine_readable(capsys) -> None:
    """JSON test listings can be consumed without text parsing."""
    exit_code = main(["list-tests", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "MISSING_VALUES" in payload


def test_check_writes_structured_json_report(tmp_path, capsys) -> None:
    """A CSV can be checked end-to-end and persisted as JSON."""
    input_path = _write_csv(tmp_path)
    output_path = tmp_path / "report.json"

    exit_code = main(
        [
            "check",
            str(input_path),
            "--tests",
            "MISSING_VALUES",
            "--output",
            str(output_path),
            "--verbose",
            "-1",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert payload["n_rows"] == 12
    assert payload["n_columns"] == 2
    assert payload["executed_tests"] == ["MISSING_VALUES"]
    assert "rows=12 columns=2 tests=1" in output


def test_check_rejects_unknown_test_id(tmp_path) -> None:
    """Invalid test IDs fail before the dataset analysis begins."""
    input_path = _write_csv(tmp_path)

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "check",
                str(input_path),
                "--tests",
                "NOT_A_TEST",
                "--verbose",
                "-1",
            ]
        )

    assert captured.value.code == 2


def test_check_rejects_unsupported_extension(tmp_path) -> None:
    """Unsupported input formats produce a normal argparse error exit."""
    path = tmp_path / "input.xlsx"
    path.write_text("not really a spreadsheet", encoding="utf-8")

    with pytest.raises(SystemExit) as captured:
        main(["check", str(path), "--verbose", "-1"])

    assert captured.value.code == 2
