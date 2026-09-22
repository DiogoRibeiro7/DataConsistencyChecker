"""Display mixin for DataConsistencyChecker.

This module contains helper methods related to displaying information
and statistics about the data. Splitting these out keeps the main
``check_data_consistency.py`` file more manageable.
"""

from __future__ import annotations

import os
from textwrap import wrap
from typing import Optional

import pandas as pd
from IPython.display import display, Markdown

from .checker_utils import is_notebook, print_text
from .test_registry import (
    TEST_DEFN_SHORT_DESC,
    TEST_DEFN_DESC,
    TEST_DEFN_IMPLEMENTED,
)


class DisplayMixin:
    """Mixin providing display helper methods."""

    # ------------------------------------------------------------------
    # Public helper methods about the tool itself
    # ------------------------------------------------------------------
    def get_test_list(self) -> list[str]:
        """Return the list of implemented test IDs."""
        return [x for x in self.test_dict.keys() if self.test_dict[x][TEST_DEFN_IMPLEMENTED]]

    def get_test_descriptions(self) -> dict:
        """Return a dictionary mapping test IDs to short descriptions."""
        return {
            x: self.test_dict[x][TEST_DEFN_DESC]
            for x in self.test_dict.keys()
            if self.test_dict[x][TEST_DEFN_IMPLEMENTED]
        }

    def print_test_descriptions(self, long_desc: bool = False, f: Optional[object] = None) -> None:
        """Print test descriptions.

        Args:
            long_desc: If ``True``, include the longer description from the
                test function docstring.
            f: Optional file handle. If provided, output is written to this file
                as HTML.
        """
        for test_id in self.test_dict.keys():
            text = self.test_dict[test_id][TEST_DEFN_DESC]
            if long_desc:
                doc_str = self.test_dict[test_id][TEST_DEFN_FUNC].__doc__
                if doc_str:
                    text += doc_str
                    text = " ".join(text.split())
            multiline_test_desc = wrap(text, 90)
            if f:
                f.write(f"{test_id+':':<30} {multiline_test_desc[0]}" + "<br>" + os.linesep)
            else:
                print(f"{test_id+':':<30} {multiline_test_desc[0]}")
            filler = "".join([" "] * 32)
            for line in multiline_test_desc[1:]:
                if f:
                    f.write(f"{filler} {line}" + "<br>" + os.linesep)
                else:
                    print(f"{filler} {line}")

    def get_patterns_shortlist(self) -> list[str]:
        """Return IDs of tests included in the short list."""
        return [x for x in self.test_dict.keys() if self.test_dict[x][TEST_DEFN_SHORTLIST]]

    def get_tests_for_codes(self) -> list[str]:
        """Return IDs of tests related to code/ID style values."""
        return [x for x in self.test_dict.keys() if self.test_dict[x][TEST_DEFN_CODE]]

    def demo_test(self, test_id: str, include_nulls: bool = False) -> None:
        """Demonstrate a single test on synthetic data.

        Args:
            test_id: ID of the test to run.
            include_nulls: Whether to generate several versions of the demo data
                containing different amounts of ``None`` values.
        """
        if test_id not in self.test_dict.keys():
            print(f"Error {test_id} is not a valid test")

        none_cases = ["none"]
        none_strs = [""]
        if include_nulls:
            none_cases = ["none", "one-row", "in-sync", "random", "80-percent"]
            none_strs = [
                "",
                "-- One row of Null values",
                "-- 50% Null values (same rows)",
                "-- 50% Null values (random locations)",
                "-- 80-percent Null Values",
            ]

        for none_idx, none_type in enumerate(none_cases):
            synth_df = self.generate_synth_data(all_cols=False, execute_list=[test_id], add_nones=none_type)
            print_text(f"## {test_id}")
            print_text(f"### Synthetic Data {none_strs[none_idx]}:")
            if is_notebook():
                display(synth_df)
            else:
                print(synth_df)

            self.verbose = 2
            self.init_data(synth_df)
            self.check_data_quality(execute_list=[test_id])
            self.display_detailed_results(show_short_list_only=False)

    # ------------------------------------------------------------------
    # Dataset statistics helpers
    # ------------------------------------------------------------------
    def display_columns_types_list(self) -> None:
        """Display lists of columns for each inferred type."""
        print()
        print_text("**String Columns**:")
        if len(self.string_cols) > 0:
            print_text(str(self.string_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

        print()
        print_text("**Numeric Columns**:")
        if len(self.numeric_cols) > 0:
            print_text(str(self.numeric_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

        print()
        print_text("**Binary Columns**:")
        if len(self.binary_cols) > 0:
            print_text(str(self.binary_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

        print()
        print_text("**Date/Time Columns**:")
        if len(self.date_cols) > 0:
            print_text(str(self.date_cols).replace("[", "").replace("]", ""))
        else:
            print_text("None")

    def display_columns_types_table(self) -> None:
        """Display the inferred types along with sample data."""
        var_types = []
        for col_name in self.orig_df.columns:
            if col_name in self.string_cols:
                var_types.append("String")
            elif col_name in self.binary_cols:
                var_types.append("Binary")
            elif col_name in self.date_cols:
                var_types.append("Date")
            elif col_name in self.numeric_cols:
                var_types.append("Numeric")
            else:
                var_types.append("Unused")
        df1 = pd.DataFrame([var_types], columns=self.orig_df.columns)
        df2 = self.orig_df.head(5).copy()
        display_df = pd.concat([df1, df2])

        print()
        print_text("Assigned column types and example rows:")
        if is_notebook():
            display(display_df)
        else:
            print(display_df)
