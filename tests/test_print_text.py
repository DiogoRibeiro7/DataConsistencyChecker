"""print_text() formats text for an HTML file, a notebook or the console."""

from __future__ import annotations

import io
import os

from data_consistency_checker import checker_utils
from data_consistency_checker.checker_utils import print_text


def test_html_output_keeps_bold_spacing_and_line_breaks() -> None:
    f = io.StringIO()
    print_text("**Description**: a  b\nc", f)
    assert f.getvalue() == "<b>Description</b>:&nbsp;a&nbsp;&nbsp;b<br>c<br><br>" + os.linesep


def test_html_output_closes_headings() -> None:
    f = io.StringIO()
    print_text("## MISSING_VALUES", f)
    print_text("### Synthetic Data:", f)
    assert f.getvalue() == ("<H1>MISSING_VALUES</H1><br><br>" + os.linesep
                            + "<H2>Synthetic&nbsp;Data:</H2><br><br>" + os.linesep)


def test_notebook_output_is_markdown_keeping_spacing_and_line_breaks(monkeypatch) -> None:
    shown: list = []
    monkeypatch.setattr(checker_utils, "is_notebook", lambda: True)
    monkeypatch.setattr(checker_utils, "display", shown.append)
    print_text("## MISSING_VALUES")
    print_text("**Description**: a  b\nc")
    assert [m.data for m in shown] == ["## MISSING_VALUES", "**Description**:&nbsp;a&nbsp;&nbsp;b<br>c"]


def test_notebook_output_prints_decision_trees_as_they_are(monkeypatch, capsys) -> None:
    shown: list = []
    monkeypatch.setattr(checker_utils, "is_notebook", lambda: True)
    monkeypatch.setattr(checker_utils, "display", shown.append)
    tree = "predictable using a decision tree with the following rules: \n|--- a <= 1.5\n|   |--- class: 0"
    print_text(tree)
    assert shown == []
    assert capsys.readouterr().out == tree + "\n"


def test_console_output_drops_the_formatting(monkeypatch, capsys) -> None:
    monkeypatch.setattr(checker_utils, "is_notebook", lambda: False)
    print_text("## **Description**: a<br>b")
    assert capsys.readouterr().out == " Description: a\nb\n"
