"""Display mixin for DataConsistencyChecker.

This module contains helper methods related to displaying information
and statistics about the data. Splitting these out keeps the main
``check_data_consistency.py`` file more manageable.
"""

from __future__ import annotations

import os
from textwrap import wrap
from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import display, Markdown

from .checker_utils import is_notebook, print_text, styling_flagged_rows
from .test_metadata import TestMetadata, metadata_from_definition


class DisplayMixin:
    """Mixin providing display helper methods."""

    # ------------------------------------------------------------------
    # Public helper methods about the tool itself
    # ------------------------------------------------------------------
    def get_test_list(self) -> list[str]:
        """Return the list of implemented test IDs."""
        return [test_id for test_id, definition in self.test_dict.items() if definition.implemented]

    def get_test_descriptions(self) -> dict:
        """Return a dictionary mapping test IDs to short descriptions."""
        return {
            test_id: definition.description
            for test_id, definition in self.test_dict.items()
            if definition.implemented
        }

    def get_test_catalog(self) -> list[TestMetadata]:
        """Return typed metadata for all implemented consistency checks."""
        return [
            metadata_from_definition(test_id, definition)
            for test_id, definition in self.test_dict.items()
            if definition.implemented
        ]

    def print_test_descriptions(self, long_desc: bool = False, f: Optional[object] = None) -> None:
        """Print test descriptions.

        Args:
            long_desc: If ``True``, include the longer description from the
                test function docstring.
            f: Optional file handle. If provided, output is written to this file
                as HTML.
        """
        for test_id in self.test_dict.keys():
            text = self.test_dict[test_id].description
            if long_desc:
                doc_str = self.test_dict[test_id].test_func.__doc__
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
        return [test_id for test_id, definition in self.test_dict.items() if definition.shortlist]

    def get_tests_for_codes(self) -> list[str]:
        """Return IDs of tests related to code/ID style values."""
        return [test_id for test_id, definition in self.test_dict.items() if definition.code]

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

    def _output_current_test(self, test_num, test_id):
        """
        Executed as tests run to allow monitoring progress.
        """
        if self.verbose <= 0:
            return
        if self.verbose == 1:
            print(f"Executing test {test_num:3}: {test_id:<30}")
            return

        # Many tests have sufficiently short descriptions, and so do not specify a separate short description.
        desc = self.test_dict[test_id].short_description
        if desc == "":
            desc = self.test_dict[test_id].description

        # Strip out any periods from short descriptions to be consistent.
        if desc[-1] == '.':
            desc = desc[:-1]

        if is_notebook():
            multiline_test_desc = wrap(desc, 70)
            print(f"Executing test {test_num:3}: {test_id}")
            print(f"  {multiline_test_desc[0]}")
            filler = ''.join([" "]*52)
            for line in multiline_test_desc[1:]:
               print(f'{filler} {line}')
        else:
            print(f"Executing test {test_num:3}: {test_id} \n  {desc}")


    def display_next(self):
        """
        This may be used where there are many results, and we wish to view detailed descriptions of all or most of
        these. This API calls display_detailed_results() for one test at a time, for each test that identified at least
        one pattern (with or without exceptions) during the last call to check_data_quality(). This allows, when working
        with notebooks, for output to be spread over multiple cells, which can make viewing it simpler. Note though,
        where many tests flag patterns, in most cases only a subset of these would be useful to examine in detail,
        though this varies for different projects.
        """

        if self.current_display_test >= len(self.found_tests):
            print("No further test results")
            return

        test_id = self.found_tests[self.current_display_test]

        if is_notebook():
            print(''.join(['.'] * 100))
            display(Markdown(f"### {test_id}"))
        else:
            print("\n\n\n")
            print(test_id)
        print_text(f"**Description**: {self.test_dict[test_id].description}")
        print()

        self.display_detailed_results(test_id_list=[test_id], max_shown=25)
        self.current_display_test += 1

    def display_least_flagged_rows(self, with_results=True, n_rows=10):
        """
        This displays the n_rows rows from the original data with the lowest scores. These are the rows with the least
        flagged issues. This may be called to provide context for the flagged rows. In rare cases, some returned rows
        may have non-zero scores, if all or most rows in the dataset are flagged at least once.

        with_results: bool
            If with_results is False, this displays a single dataframe showing the appropriate subset of the original
            data. If with_results is True, this displays a dataframe per original row, up to n_rows. For each, the
            original data is shown, along with all flagged issues, across all tests on all features.

        n_rows: int
            The maximum number of original rows to present.
        """

        sorted_df = self.test_results_df.sort_values('FINAL SCORE', ascending=True)
        if with_results:
            self._display_rows_with_tests(sorted_df, n_rows)
        else:
            df = self.orig_df.loc[sorted_df.index].head(n_rows).copy()
            df['FINAL SCORE'] = sorted_df['FINAL SCORE'][:len(df)]
            if is_notebook():
                display(df)
            else:
                print(df)

    def display_most_flagged_rows(self, with_results=True, n_rows=10):
        """
        This is similar to display_least_flagged_rows, but displays the rows with the most identified issues.

        with_results: bool
            If True, the flagged rows will be display in separate tables, with additional rows indicating which tests
            flagged which columns. If False, all displayed rows will be displayed in a single table; the flagged
            columns will be highlighted, but there will not be an indication of which tests flagged them.
        n_rows: int
            The maximum number of original rows to present.
        """

        if self.test_results_df is None or len(self.test_results_df) == 0:
            return None

        sorted_df = self.test_results_df.sort_values('FINAL SCORE', ascending=False)
        sorted_df.index = [x[0] if type(x) == tuple else x for x in sorted_df.index]
        if with_results:
            self._display_rows_with_tests(sorted_df, n_rows, check_score=True)
        else:
            row_idxs = np.array(sorted_df.index.to_list()).reshape(1, -1)[0]
            df = self.orig_df.loc[row_idxs].head(n_rows).copy()
            df['FINAL SCORE'] = sorted_df['FINAL SCORE'][:len(df)]
            df = df[df['FINAL SCORE'] > 0]
            if is_notebook():
                display(df.style.apply(styling_flagged_rows,
                                       flagged_cells=self.test_results_by_column_np,
                                       axis=None))
            else:
                print(df)


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
