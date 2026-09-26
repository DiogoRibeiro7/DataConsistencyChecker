"""Behavioural tests for ``DataConsistencyChecker.display_detailed_results()``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker import display_mixin as display_module

EXECUTED = ["POSITIVE", "NEGATIVE", "COLUMN_ORDERED_ASC"]
QUIET = {"include_examples": False, "plot_results": False}


def _run(df: pd.DataFrame, execute_list=EXECUTED, **kwargs) -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    checker.check_data_quality(execute_list=execute_list, **kwargs)
    return checker


def _signed_frame() -> pd.DataFrame:
    rows = np.arange(200)
    df = pd.DataFrame(
        {
            "pos_all": rows + 1.0,
            "pos_most": rows + 2.0,
            "neg_all": -(rows + 1.0),
            "neg_most": -(rows + 3.0),
            "ordered": rows * 2.0,
        }
    )
    df.loc[7, "pos_most"] = -4.0
    df.loc[11, "neg_most"] = 5.0
    df.loc[40, "ordered"] = 1000.0
    return df


@pytest.fixture
def checker() -> DataConsistencyChecker:
    return _run(_signed_frame())


def _issue_id(checker: DataConsistencyChecker, test_id: str, column: str) -> int:
    summary = checker.exceptions_summary_df
    match = summary[(summary["Test ID"] == test_id) & (summary["Column(s)"] == column)]
    return int(match["Issue ID"].iloc[0])


def _pattern_id(checker: DataConsistencyChecker, test_id: str, column: str) -> int:
    patterns = checker.patterns_df
    match = patterns[(patterns["Test ID"] == test_id) & (patterns["Column(s)"] == column)]
    return int(match["Pattern ID"].iloc[0])


def _record_plots(checker: DataConsistencyChecker, monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_plots(test_id, cols, _columns_set, show_exceptions, **_kwargs):
        calls.append({"test_id": test_id, "cols": list(cols), "show_exceptions": show_exceptions})

    monkeypatch.setattr(checker, "_draw_results_plots", fake_plots)
    return calls


def _checker_with_findings(patterns=(), exceptions=()) -> DataConsistencyChecker:
    """Build a checker holding findings in the same frames ``check_data_quality()`` produces."""
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 3}))
    checker.patterns_df = pd.DataFrame(
        list(patterns), columns=["Test ID", "Column(s)", "Description of Pattern", "Display Information"]
    )
    checker.patterns_df["Pattern ID"] = list(range(len(checker.patterns_df)))
    checker.exceptions_summary_df = pd.DataFrame(
        list(exceptions),
        columns=["Test ID", "Column(s)", "Description of Pattern", "Number of Exceptions", "Display Information"],
    )
    checker.exceptions_summary_df["Issue ID"] = list(range(len(checker.exceptions_summary_df)))
    for _, column_set, *_ in patterns:
        checker.col_to_original_cols_dict[column_set] = [c.strip('"') for c in column_set.split(" AND ")]
    for test_id, column_set, *_ in exceptions:
        checker.col_to_original_cols_dict[checker.get_results_col_name(test_id, column_set)] = [
            c.strip('"') for c in column_set.split(" AND ")
        ]
    return checker


# ----------------------------------------------------------------------------------------------------------------------
# Early exits
# ----------------------------------------------------------------------------------------------------------------------


def test_reports_empty_dataset(capsys) -> None:
    DataConsistencyChecker(verbose=-1).display_detailed_results()

    assert "Empty dataset" in capsys.readouterr().out


def test_reports_when_nothing_was_found(capsys) -> None:
    rows = np.arange(50)
    checker = _run(pd.DataFrame({"x": rows + 1.0, "y": rows % 3 + 1.0}), execute_list=["NEGATIVE"])

    checker.display_detailed_results()

    assert "No patterns or exceptions to display." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, "8 patterns and exceptions were identified."),
        ({"show_exceptions": False}, "patterns were identified."),
        ({"show_patterns": False}, "4 issues were identified."),
    ],
)
def test_unfiltered_display_refuses_more_than_max_shown(checker, capsys, options, expected) -> None:
    checker.display_detailed_results(max_shown=1, **QUIET, **options)

    out = capsys.readouterr().out
    assert expected in out
    assert "beyond the limit to display all at once" in out
    assert "Columns(s):" not in out


def test_default_limit_depends_on_examples_and_plots(capsys) -> None:
    rows = np.arange(30)
    checker = _run(pd.DataFrame({f"c{i}": -(rows + i + 1.0) for i in range(60)}), execute_list=["NEGATIVE"])

    checker.display_detailed_results()
    assert "60 patterns and exceptions were identified" in capsys.readouterr().out

    checker.display_detailed_results(**QUIET)
    out = capsys.readouterr().out
    assert "beyond the limit" not in out
    assert out.count("Pattern found (without exceptions)") == 60


def test_rejects_rows_beyond_the_dataset(checker, capsys) -> None:
    checker.display_detailed_results(row_id_list=[500], **QUIET)

    out = capsys.readouterr().out
    assert "Row id specified was beyond the length of the dataframe" in out
    assert "Columns(s):" not in out


# ----------------------------------------------------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------------------------------------------------


def test_announces_requested_tests_and_reports_hidden_or_invalid_ones(checker, capsys) -> None:
    checker.display_detailed_results(test_id_list=["POSITIVE", "NEGATIVE", "NOT_A_TEST"], **QUIET)

    out = capsys.readouterr().out
    assert "Displaying results for tests:" in out
    assert "Not displaying patterns without exceptions for POSITIVE" in out
    assert "NOT_A_TEST is not a valid test ID" in out
    # POSITIVE is not in the short list, so only its exception is shown; NEGATIVE shows both.
    assert "Columns(s): pos_most" in out
    assert "Columns(s): pos_all" not in out
    assert "Columns(s): neg_all" in out
    assert "Columns(s): neg_most" in out


def test_column_filter_limits_findings_and_reports_invalid_columns(checker, capsys) -> None:
    checker.display_detailed_results(col_name_list=["neg_most", "nope"], **QUIET)

    out = capsys.readouterr().out
    assert "Displaying results for columns: ['neg_most', 'nope']" in out
    assert "nope is not a valid column name" in out
    assert "Columns(s): neg_most" in out
    for other in ["pos_all", "pos_most", "neg_all", "ordered"]:
        assert f"Columns(s): {other}" not in out


def test_row_filter_shows_only_exceptions_flagging_those_rows(checker, capsys) -> None:
    checker.display_detailed_results(row_id_list=[11], **QUIET)

    out = capsys.readouterr().out
    assert out.count("Issue ID:") == 1
    assert "Columns(s): neg_most" in out
    assert "Pattern found" not in out


def test_issue_filter_shows_only_the_selected_exception(checker, capsys) -> None:
    issue_id = _issue_id(checker, "COLUMN_ORDERED_ASC", "ordered")

    checker.display_detailed_results(issue_id_list=[issue_id], **QUIET)

    out = capsys.readouterr().out
    assert out.count("Issue ID:") == 1
    assert f"Issue ID: {issue_id}" in out
    assert "Columns(s): ordered" in out
    assert "Pattern found" not in out


def test_pattern_filter_shows_only_the_selected_pattern(checker, capsys) -> None:
    pattern_id = _pattern_id(checker, "NEGATIVE", "neg_all")

    checker.display_detailed_results(pattern_id_list=[pattern_id], **QUIET)

    out = capsys.readouterr().out
    assert out.count("Pattern found (without exceptions)") == 1
    assert "Columns(s): neg_all" in out
    assert "Issue ID" not in out


def test_skips_tests_excluded_from_the_run(capsys) -> None:
    rows = np.arange(60)
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(pd.DataFrame({"pos_all": rows + 1.0, "neg_all": -(rows + 1.0)}))
    checker.check_data_quality(exclude_list=["NEGATIVE"])

    checker.display_detailed_results(test_id_list=["NEGATIVE", "COLUMN_ORDERED_ASC"], **QUIET)

    lines = capsys.readouterr().out.splitlines()
    assert "COLUMN_ORDERED_ASC" in lines
    assert "NEGATIVE" not in lines


@pytest.mark.parametrize(
    ("options", "marker"),
    [
        ({"show_exceptions": False}, "Pattern found (without exceptions)"),
        ({"show_patterns": False}, "Issue ID:"),
    ],
)
def test_stops_after_max_shown_findings(checker, capsys, options, marker) -> None:
    checker.display_detailed_results(test_id_list=["NEGATIVE", "COLUMN_ORDERED_ASC"], max_shown=1, **QUIET, **options)

    out = capsys.readouterr().out
    assert out.count(marker) == 1
    assert "Showing the first 1 findings" in out


# ----------------------------------------------------------------------------------------------------------------------
# Headers
# ----------------------------------------------------------------------------------------------------------------------


def test_console_prints_each_test_header_once(checker, capsys) -> None:
    checker.display_detailed_results(test_id_list=["NEGATIVE", "COLUMN_ORDERED_ASC"], **QUIET)

    lines = capsys.readouterr().out.splitlines()
    assert lines.count("NEGATIVE") == 1
    assert lines.count("COLUMN_ORDERED_ASC") == 1
    assert any(line and set(line) == {"*"} for line in lines)


def test_single_test_display_omits_test_header(checker, capsys) -> None:
    checker.display_detailed_results(test_id_list=["NEGATIVE"], **QUIET)

    lines = capsys.readouterr().out.splitlines()
    assert "Columns(s): neg_all" in lines
    assert "NEGATIVE" not in lines
    assert not any(line and set(line) == {"*"} for line in lines)


def test_notebook_headers_are_rendered_as_markdown(checker, monkeypatch) -> None:
    displayed: list[object] = []
    monkeypatch.setattr(display_module, "is_notebook", lambda: True)
    monkeypatch.setattr(display_module, "display", displayed.append)

    checker.display_detailed_results(test_id_list=["NEGATIVE", "COLUMN_ORDERED_ASC"], **QUIET)

    markdown = [item.data for item in displayed]
    assert "### NEGATIVE" in markdown
    assert "### COLUMN_ORDERED_ASC" in markdown
    assert "### Columns(s): neg_all" in markdown
    assert "### Columns(s): neg_most" in markdown


# ----------------------------------------------------------------------------------------------------------------------
# Rendering patterns and exceptions
# ----------------------------------------------------------------------------------------------------------------------


def test_pattern_shows_description_examples_and_plot(checker, capsys, monkeypatch) -> None:
    plots = _record_plots(checker, monkeypatch)

    checker.display_detailed_results(test_id_list=["NEGATIVE"], show_exceptions=False)

    out = capsys.readouterr().out
    assert "Pattern found (without exceptions)" in out
    assert "Description: The column contains consistently zero or negative values" in out
    assert "Examples:" in out
    assert plots == [{"test_id": "NEGATIVE", "cols": ["neg_all"], "show_exceptions": False}]


def test_default_display_draws_a_plot_per_finding(checker, monkeypatch) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    try:
        # plot_results defaults to True: NEGATIVE has a pattern and an exception, COLUMN_ORDERED_ASC one and two.
        checker.display_detailed_results(test_id_list=["NEGATIVE", "COLUMN_ORDERED_ASC"], include_examples=False)
    finally:
        plt.close("all")

    assert len(shown) == 5


def test_exception_shows_examples_flagged_rows_and_plot(checker, capsys, monkeypatch) -> None:
    plots = _record_plots(checker, monkeypatch)

    checker.display_detailed_results(issue_id_list=[_issue_id(checker, "NEGATIVE", "neg_most")])

    out = capsys.readouterr().out
    assert "A strong pattern, and exceptions to the pattern, were found." in out
    assert "Number of exceptions: 1 (0.5000% of rows)" in out
    assert "Examples of values NOT flagged:" in out
    assert "Flagged values:" in out
    assert "Showing a flagged example" not in out
    assert plots == [{"test_id": "NEGATIVE", "cols": ["neg_most"], "show_exceptions": True}]


def test_exception_with_many_flagged_rows_labels_them_as_examples(capsys) -> None:
    rows = np.arange(200)
    df = pd.DataFrame({"many_neg": rows + 1.0, "other": rows % 9})
    df.loc[list(range(0, 120, 10)), "many_neg"] = -1.0
    checker = _run(df, execute_list=["POSITIVE"], freq_contamination_level=15)

    checker.display_detailed_results(plot_results=False)

    out = capsys.readouterr().out
    assert "Number of exceptions: 12" in out
    assert "Examples of flagged values:" in out
    assert "Flagged values:" not in out


def test_ordered_exception_shows_rows_around_a_flagged_row(checker, capsys) -> None:
    checker.display_detailed_results(
        issue_id_list=[_issue_id(checker, "COLUMN_ORDERED_ASC", "ordered")],
        plot_results=False,
    )

    out = capsys.readouterr().out
    assert "Examples of values NOT flagged (showing a consecutive set of rows):" in out
    assert "Showing a flagged example (row" in out
    assert "with 5 rows before and 5 rows after" in out


def test_exception_without_flagged_rows_skips_flagged_examples_and_plot(checker, capsys, monkeypatch) -> None:
    plots = _record_plots(checker, monkeypatch)
    checker.test_results_df[checker.get_results_col_name("NEGATIVE", "neg_most")] = False

    checker.display_detailed_results(issue_id_list=[_issue_id(checker, "NEGATIVE", "neg_most")])

    out = capsys.readouterr().out
    assert "Examples of values NOT flagged:" in out
    assert "Flagged values" not in out
    assert plots == []


@pytest.mark.parametrize(
    ("test_id", "headline"),
    [
        ("RARE_VALUES", "Unusual values were found."),
        ("NEGATIVE", "A strong pattern, and exceptions to the pattern, were found."),
    ],
)
def test_exception_headline_and_wrapped_description(capsys, test_id, headline) -> None:
    description = " ".join(["word"] * 40)  # 199 characters, wrapped to lines of at most 100
    checker = _checker_with_findings(exceptions=[(test_id, "a", description, 3, {})])

    checker.display_detailed_results(**QUIET)

    out = capsys.readouterr().out
    assert headline in out
    assert "Number of exceptions: 3 (15.0000% of rows)" in out
    assert description in " ".join(out.split())
    assert all(len(line) <= len("Description: ") + 100 for line in out.splitlines())


@pytest.mark.parametrize("test_id", ["DECISION_TREE_CLASSIFIER", "GROUPED_STRINGS"])
def test_structured_exception_descriptions_are_printed_verbatim(capsys, test_id) -> None:
    description = "first rule\n  second rule"
    checker = _checker_with_findings(exceptions=[(test_id, '"a" AND "b"', description, 2, {})])

    checker.display_detailed_results(**QUIET)

    lines = capsys.readouterr().out.splitlines()
    assert "Description:" in lines
    assert "first rule" in lines
    assert "  second rule" in lines


def test_decision_tree_pattern_description_is_printed_verbatim(capsys) -> None:
    description = "first rule\n  second rule"
    checker = _checker_with_findings(patterns=[("DECISION_TREE_CLASSIFIER", '"a" AND "b"', description, {})])

    checker.display_detailed_results(**QUIET)

    lines = capsys.readouterr().out.splitlines()
    assert "Pattern found (without exceptions)" in lines
    assert "Description:" in lines
    assert "  second rule" in lines


# ----------------------------------------------------------------------------------------------------------------------
# HTML export
# ----------------------------------------------------------------------------------------------------------------------


def test_save_to_disk_writes_html_report(checker, tmp_path, capsys) -> None:
    checker.display_detailed_results(
        test_id_list=["NEGATIVE", "COLUMN_ORDERED_ASC"],
        save_to_disk=True,
        output_folder=str(tmp_path),
        plot_results=False,
    )

    html = (tmp_path / "Data_consistency.html").read_text()
    assert checker.output_folder == str(tmp_path)
    assert html.startswith("<html>")
    assert "<h1>Data Consistency Check Results</h1>" in html
    assert "<H2>NEGATIVE</H2>" in html
    assert "<H2>COLUMN_ORDERED_ASC</H2>" in html
    assert "Columns(s): neg_all<br>" in html
    assert "Issue ID" in html
    assert "<table" in html
    assert html.rstrip().endswith("</html>")
    assert "Columns(s):" not in capsys.readouterr().out


def test_save_to_disk_embeds_result_plots_as_images(checker, tmp_path) -> None:
    checker.display_detailed_results(
        test_id_list=["NEGATIVE"],
        save_to_disk=True,
        output_folder=str(tmp_path),
        include_examples=False,
    )

    html = (tmp_path / "Data_consistency.html").read_text()
    images = sorted(p.name for p in tmp_path.glob("*.png"))
    assert images == ["output_0.png", "output_1.png"]
    for image in images:
        assert f"<img src={image}>" in html
    assert plt.get_fignums() == []


def test_save_to_disk_defaults_to_output_folder_in_working_directory(checker, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    checker.display_detailed_results(test_id_list=["NEGATIVE"], save_to_disk=True, **QUIET)

    report = tmp_path / "Output" / "Data_consistency.html"
    assert report.exists()
    assert "Columns(s): neg_most" in report.read_text()
