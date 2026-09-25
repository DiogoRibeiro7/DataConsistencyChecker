"""Coverage tests for HTML export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker


def _checker_with_results() -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.num_rows = 10
    checker.patterns_df = pd.DataFrame(
        [
            {
                "Test ID": "PATTERN_TEST",
                "Column(s)": "a",
                "Description of Pattern": "Column a follows a pattern.",
                "Display Information": None,
                "Pattern ID": 0,
            }
        ]
    )
    checker.exceptions_summary_df = pd.DataFrame(
        [
            {
                "Test ID": "EXCEPTION_TEST",
                "Column(s)": "b",
                "Description of Pattern": "Column b follows a pattern, with exceptions.",
                "Number of Exceptions": 2,
                "Display Information": None,
                "Issue ID": 0,
            }
        ]
    )
    return checker


def test_gpt_export_html_retains_unsupported_stub(
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = DataConsistencyChecker(verbose=-1)

    checker.gpt_export_html()

    assert "not supported" in capsys.readouterr().out


def test_export_html_writes_patterns_and_exceptions(tmp_path: Path) -> None:
    checker = _checker_with_results()
    output_file = tmp_path / "report.html"

    checker.export_html(
        test_id_list=["PATTERN_TEST", "EXCEPTION_TEST"],
        output_file=str(output_file),
    )

    html = output_file.read_text(encoding="utf-8")
    assert "Data Consistency Check Results" in html
    assert "Pattern found (without exceptions)" in html
    assert "Column a follows a pattern." in html
    assert "A strong pattern, and exceptions to the pattern, were found." in html
    assert "Number of exceptions: 2" in html
    assert "20.0000% of rows" in html


def test_export_html_uses_test_catalog_when_ids_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _checker_with_results()
    output_file = tmp_path / "catalog-report.html"
    monkeypatch.setattr(
        checker,
        "get_test_list",
        lambda: ["PATTERN_TEST", "EXCEPTION_TEST"],
    )

    checker.export_html(output_file=str(output_file))

    html = output_file.read_text(encoding="utf-8")
    assert "PATTERN_TEST" in html
    assert "EXCEPTION_TEST" in html
