"""The examples in the documentation produce the results the documentation describes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker, DataConsistencyConfig, analyze
from data_consistency_checker.cli import main


def _orders() -> pd.DataFrame:
    """The order table of docs/getting-started/quickstart.md."""
    rows = np.arange(500)
    df = pd.DataFrame(
        {
            "price": (rows % 50) * 2.0 + 10,
            "quantity": rows % 7 + 1,
            "region": np.where(rows % 3 == 0, "north", "south"),
        }
    )
    df["total"] = df["price"] * df["quantity"]
    df.loc[42, "total"] = 1.0
    return df


@pytest.fixture(scope="module")
def checker() -> DataConsistencyChecker:
    dc = DataConsistencyChecker(verbose=-1)
    dc.init_data(_orders())
    dc.check_data_quality()
    return dc


def test_quickstart_summary_rows(checker) -> None:
    summary = checker.summarize_patterns_and_exceptions().set_index("Test ID")

    assert summary.loc["POSITIVE"].tolist() == [3, ""]
    assert summary.loc["UNUSUAL_ORDER_MAGNITUDE"].tolist() == ["", 1]
    assert summary.loc["MULTIPLE_OF_CONSTANT"].tolist() == [1, 1]


def test_quickstart_exceptions_all_involve_total(checker) -> None:
    exceptions = checker.get_exceptions_list()

    assert len(exceptions) == 5
    assert all("total" in columns for columns in exceptions["Column(s)"])
    assert '"price" AND "quantity" AND "total"' in set(
        exceptions.loc[exceptions["Test ID"] == "SIMILAR_TO_PRODUCT", "Column(s)"]
    )


def test_quickstart_scores_single_out_the_corrupted_row(checker) -> None:
    scores = checker.get_outlier_scores()

    assert scores[42] == 5
    assert set(scores[:42] + scores[43:]) == {0}
    assert len(checker.get_results_by_row_id(42)) == 5


def test_quickstart_one_shot_analysis_matches(checker) -> None:
    report = analyze(_orders(), DataConsistencyConfig(verbose=-1))

    assert len(report.exceptions) == len(checker.get_exceptions_list())
    assert report.to_dict()["n_rows"] == 500


def test_cli_summary_line_matches_the_docs(tmp_path, capsys) -> None:
    csv_path = tmp_path / "orders.csv"
    _orders().to_csv(csv_path, index=False)

    assert main(["check", str(csv_path), "--output", str(tmp_path / "report.json"), "--verbose", "-1"]) == 0

    assert capsys.readouterr().out.strip().splitlines()[-1] == (
        "rows=500 columns=4 tests=158 patterns=19 exceptions=5 failures=0"
    )
    assert (tmp_path / "report.json").exists()
