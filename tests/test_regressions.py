"""Regression tests for fixed defects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_consistency_checker import DataConsistencyChecker
from data_consistency_checker.test_implementations import numeric_tests_mixin


def _run(df: pd.DataFrame, execute_list: list[str], **kwargs) -> DataConsistencyChecker:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)
    checker.check_data_quality(execute_list=execute_list, **kwargs)
    return checker


def _checker_with_findings(exceptions) -> DataConsistencyChecker:
    """Build a checker holding exceptions in the same frames ``check_data_quality()`` produces."""
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 3}))
    checker.patterns_df = pd.DataFrame(
        columns=["Test ID", "Column(s)", "Description of Pattern", "Display Information", "Pattern ID"]
    )
    checker.exceptions_summary_df = pd.DataFrame(
        list(exceptions),
        columns=["Test ID", "Column(s)", "Description of Pattern", "Number of Exceptions", "Display Information"],
    )
    checker.exceptions_summary_df["Issue ID"] = list(range(len(checker.exceptions_summary_df)))
    for test_id, column_set, *_ in exceptions:
        checker.col_to_original_cols_dict[checker.get_results_col_name(test_id, column_set)] = [
            c.strip('"') for c in column_set.split(" AND ")
        ]
    return checker


def _signed_checker() -> DataConsistencyChecker:
    rows = np.arange(200)
    df = pd.DataFrame({"neg_all": -(rows + 1.0), "neg_most": -(rows + 3.0), "ordered": rows * 2.0})
    df.loc[11, "neg_most"] = 5.0
    return _run(df, ["NEGATIVE", "COLUMN_ORDERED_ASC"])


# ----------------------------------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------------------------------


def test_column_tends_desc_evaluates_date_columns() -> None:
    # The date branch used to assign one variable and test another, failing when a date column came first.
    checker = _run(pd.DataFrame({"when": pd.date_range("2020-01-01", periods=300, freq="D")[::-1]}),
                   ["COLUMN_TENDS_DESC"])

    assert checker.get_execution_failures() == []
    assert ("COLUMN_TENDS_DESC", "when") in set(zip(checker.patterns_df["Test ID"], checker.patterns_df["Column(s)"]))


def test_same_first_chars_reports_the_shared_prefix_length() -> None:
    rng = np.random.default_rng(0)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    words = ["".join(rng.choice(letters, 6)) for _ in range(300)]
    checker = _run(pd.DataFrame({"code": words, "code_ext": [w + "xyz" for w in words]}), ["SAME_FIRST_CHARS"])

    descriptions = checker.patterns_df["Description of Pattern"].tolist()
    assert descriptions == ['Columns "code" and "code_ext" consistently share the same first 6 characters']


def test_same_first_chars_skips_columns_dominated_by_one_first_letter() -> None:
    rng = np.random.default_rng(0)
    letters = np.array(list("bcdefghijklmnopqrstuvwxyz"))
    words = ["a" + "".join(rng.choice(letters, 5)) if i % 5 < 3 else "".join(rng.choice(letters, 6))
             for i in range(300)]
    checker = _run(pd.DataFrame({"code": words, "code_ext": [w + "xyz" for w in words]}), ["SAME_FIRST_CHARS"])

    assert len(checker.patterns_df) == 0
    assert len(checker.exceptions_summary_df) == 0


def test_matched_zero_missing_synthetic_data_always_contains_an_exception(monkeypatch) -> None:
    # With every random value 0, row 999 of the "most" column was left missing, so no exception was generated.
    monkeypatch.setattr(numeric_tests_mixin.random, "randint", lambda _low, _high: 0)

    synth = DataConsistencyChecker(verbose=-1).generate_synth_data(
        all_cols=False, execute_list=["MATCHED_ZERO_MISSING"])

    assert synth.loc[999, "matched zero miss rand_a"] == 0
    assert synth.loc[999, "matched zero miss most"] == 1


# ----------------------------------------------------------------------------------------------------------------------
# Execution options
# ----------------------------------------------------------------------------------------------------------------------


def test_run_parallel_warns_and_returns_the_sequential_results() -> None:
    rows = np.arange(100)
    df = pd.DataFrame({"neg": -(rows + 1.0), "other": rows % 7})
    sequential = _run(df, ["NEGATIVE"])

    with pytest.warns(RuntimeWarning, match="run_parallel"):
        parallel = _run(df, ["NEGATIVE"], run_parallel=True)

    columns = ["Test ID", "Column(s)", "Description of Pattern"]
    pd.testing.assert_frame_equal(parallel.patterns_df[columns], sequential.patterns_df[columns])
    assert len(parallel.patterns_df) == 1


def test_integer_contamination_level_of_one_row_is_applied() -> None:
    rows = np.arange(100)
    checker = _run(pd.DataFrame({"neg": -(rows + 1.0), "other": rows % 7}), ["NEGATIVE"], freq_contamination_level=1)

    assert checker.freq_contamination_level == 1


# ----------------------------------------------------------------------------------------------------------------------
# Display and HTML export
# ----------------------------------------------------------------------------------------------------------------------


def test_grouped_strings_description_is_written_to_the_html_report(tmp_path, capsys) -> None:
    checker = _checker_with_findings([("GROUPED_STRINGS", '"a" AND "b"', "grouped rule text", 2, {})])

    checker.display_detailed_results(save_to_disk=True, output_folder=str(tmp_path),
                                     include_examples=False, plot_results=False)

    assert "grouped rule text" in (tmp_path / "Data_consistency.html").read_text()
    assert "grouped rule text" not in capsys.readouterr().out


def test_html_column_header_line_break_is_well_formed(tmp_path) -> None:
    checker = _signed_checker()

    checker.display_detailed_results(test_id_list=["NEGATIVE"], save_to_disk=True, output_folder=str(tmp_path),
                                     include_examples=False, plot_results=False)

    html = (tmp_path / "Data_consistency.html").read_text()
    assert "-<br>" in html
    assert "<br\n" not in html.replace("\r\n", "\n")


@pytest.mark.parametrize(
    "options",
    [
        {"test_id_list": ["NEGATIVE", "COLUMN_ORDERED_ASC"], "max_shown": 1},
        {"row_id_list": [500]},
    ],
    ids=["max_shown_reached", "invalid_row_id"],
)
def test_html_report_is_completed_when_the_display_stops_early(tmp_path, options) -> None:
    checker = _signed_checker()

    checker.display_detailed_results(save_to_disk=True, output_folder=str(tmp_path),
                                     include_examples=False, plot_results=False, **options)

    assert (tmp_path / "Data_consistency.html").read_text().rstrip().endswith("</html>")


# ----------------------------------------------------------------------------------------------------------------------
# Example rows
# ----------------------------------------------------------------------------------------------------------------------


def test_binary_target_sample_includes_the_most_common_value_of_the_first_column() -> None:
    rows = np.arange(80)
    df = pd.DataFrame(
        {
            "first": np.where(rows < 20, rows, 7),  # 7 is the most common value, absent from the first rows
            "second": rows % 9 + 1.0,
            "flag": np.where(rows % 2 == 0, "yes", "no"),
        }
    )
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(df)

    sample = checker._get_sample_not_flagged("BINARY_MATCHES_SUM", '"first" AND "second" AND "flag"', is_patterns=True)

    # Up to n_examples // 4 rows with the most common value are taken for each value of the binary column
    assert (sample["first"] == 7).sum() == 4
    assert sample["flag"].value_counts().to_dict() == {"no": 5, "yes": 5}


# ----------------------------------------------------------------------------------------------------------------------
# Null handling and cached state
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("test_id", "add_nones"),
    [("FEW_WITHIN_RANGE", "one-row"), ("SAME_ALPHA_CHARS", "random"), ("SAME_ALPHA_CHARS", "in-sync")],
)
def test_checks_run_on_synthetic_data_with_missing_values(test_id, add_nones) -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(checker.generate_synth_data(all_cols=False, execute_list=[test_id], add_nones=add_nones))

    checker.check_data_quality(execute_list=[test_id])

    assert checker.get_execution_failures() == []


def test_word_counts_are_returned_from_the_cache() -> None:
    checker = DataConsistencyChecker(verbose=-1)
    checker.init_data(pd.DataFrame({"text": [f"w{i} v{i % 3}" for i in range(30)], "n": np.arange(30.0)}))

    first = checker.get_word_counts_dict()
    second = checker.get_word_counts_dict()

    assert isinstance(second, dict)
    assert second is first


def test_clear_results_reports_when_patterns_are_missing(capsys) -> None:
    checker = _signed_checker()
    checker.patterns_df = None

    checker.clear_results(test_id_list=["NEGATIVE"])

    assert "There are no results to clear" in capsys.readouterr().out
