"""MkDocs hook that renders the checks catalog from the live check registry.

The catalog page contains a placeholder that is replaced at build time, so the documented checks always match
the code: their IDs, descriptions, categories and flags come from the same definitions the checker uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

PLACEHOLDER = "<!-- checks-catalog -->"

CATEGORIES = [
    ("Any column type", "get_base_tests", "Single columns and pairs of columns of any type."),
    ("Numeric", "get_numeric_tests", "Numeric columns, alone or in pairs and larger sets."),
    ("Dates and times", "get_date_tests", "Date and time columns, and their relationships with other columns."),
    ("Strings", "get_string_tests", "String columns, including checks specific to code and ID values."),
    ("Binary", "get_binary_tests", "Columns with exactly two distinct values."),
    ("Sets of columns", "get_multi_column_tests", "Three or more columns, or checks across whole rows."),
]


def on_config(config, **kwargs):
    """Make the package importable from the repository's ``src`` directory."""
    src = Path(config.config_file_path).parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return config


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _catalog_markdown() -> str:
    from data_consistency_checker import DataConsistencyChecker, tests_definitions

    checker = DataConsistencyChecker(verbose=-1)
    catalog = {meta.test_id: meta for meta in checker.get_test_catalog()}
    parts = [
        f"The checker currently implements **{len(catalog)} checks**. "
        f"{sum(m.shortlist for m in catalog.values())} of them report patterns without exceptions by default "
        f"(the *short list*), {sum(m.fast for m in catalog.values())} are marked as fast, and "
        f"{sum(m.code for m in catalog.values())} are specific to code or ID values.",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| Short list | Patterns without exceptions are shown by default. Other checks' patterns need "
        "`show_short_list_only=False`; their exceptions are always reported. |",
        "| Fast | Included when running with `fast_only=True`. |",
        "| Code/ID | Specific to code or ID values; skipped with `include_code_tests=False`. |",
    ]
    for title, getter, blurb in CATEGORIES:
        ids = [test_id for test_id in getattr(tests_definitions, getter)(checker) if test_id in catalog]
        parts += ["", f"## {title}", "", f"{blurb} {len(ids)} checks.", "",
                  "| Check | Description | Short list | Fast | Code/ID |", "|---|---|:-:|:-:|:-:|"]
        for test_id in ids:
            meta = catalog[test_id]
            flags = ["&#10003;" if flag else "" for flag in (meta.shortlist, meta.fast, meta.code)]
            parts.append(f"| `{test_id}` | {_cell(meta.description)} | " + " | ".join(flags) + " |")
    return "\n".join(parts)


def on_page_markdown(markdown, page, **kwargs):
    """Replace the catalog placeholder with tables generated from the registry."""
    if PLACEHOLDER not in markdown:
        return markdown
    return markdown.replace(PLACEHOLDER, _catalog_markdown())
