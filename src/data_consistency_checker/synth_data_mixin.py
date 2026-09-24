"""Synthetic data generation mixin for DataConsistencyChecker.

This module contains helper methods extracted from the main
``check_data_consistency.py`` file. They generate synthetic datasets and
modify real datasets for demonstration purposes.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from .checker_utils import print_text


class SynthDataMixin:
    """Provide synthetic data utilities to :class:`DataConsistencyChecker`."""

    def generate_synth_data(
        self,
        all_cols: bool = False,
        execute_list: list[str] | None = None,
        exclude_list: list[str] | None = None,
        seed: int = 0,
        add_nones: str = "none",
    ) -> pd.DataFrame:
        """Generate a random dataset for demonstrating tests.

        Args:
            all_cols: When ``True`` create columns for every test. When ``False``
                only include columns for tests specified in ``execute_list``.
            execute_list: Optional list of test IDs to generate columns for.
            exclude_list: Optional list of test IDs **not** to generate columns
                for. ``execute_list`` and ``exclude_list`` cannot both be set.
            seed: Seed for the random number generators used during creation.
            add_nones: One of ``'none'``, ``'one-row'``, ``'in-sync'``,
                ``'random'`` or ``'80-percent'`` specifying how null values are
                inserted.

        Returns:
            The generated synthetic DataFrame.
        """
        assert add_nones in ["none", "one-row", "in-sync", "random", "80-percent"]

        if execute_list:
            for test_id in execute_list:
                if test_id not in self.get_test_list():
                    print_text(f"{test_id} is not a valid test ID. Unable to generate data.")
                    return pd.DataFrame()
        if exclude_list:
            for test_id in exclude_list:
                if test_id not in self.get_test_list():
                    print_text(f"{test_id} is not a valid test ID. Unable to generate data.")
                    return pd.DataFrame()

        self.synth_df = pd.DataFrame()
        for test_id in self.test_dict.keys():
            random.seed(seed)
            np.random.seed(seed)
            if all_cols or (
                (execute_list is None and exclude_list is None)
                or (execute_list and test_id in execute_list)
                or (exclude_list and test_id not in exclude_list)
            ):
                generator = self.test_dict[test_id].gen_func
                if generator is not None:
                    generator()

        if add_nones == "one-row":
            for col_name in self.synth_df.columns:
                self.synth_df.loc[0, col_name] = None
        elif add_nones == "in-sync":
            none_idxs = random.sample(range(self.num_synth_rows - 10), self.num_synth_rows // 2)
            for col_name in self.synth_df.columns:
                col_vals = self.synth_df[col_name].copy()
                col_vals.iloc[none_idxs] = None
                self.synth_df[col_name] = col_vals
        elif add_nones == "random":
            for col_name in self.synth_df.columns:
                none_idxs = random.sample(range(self.num_synth_rows - 10), self.num_synth_rows // 2)
                col_vals = self.synth_df[col_name].copy()
                col_vals.iloc[none_idxs] = None
                self.synth_df[col_name] = col_vals
        elif add_nones == "80-percent":
            none_idxs = random.sample(range(self.num_synth_rows - 10), int(self.num_synth_rows * 0.8))
            for col_name in self.synth_df.columns:
                col_vals = self.synth_df[col_name].copy()
                col_vals.iloc[none_idxs] = None
                self.synth_df[col_name] = col_vals

        return self.synth_df

    def _add_synthetic_column(self, col_name, col_values):
        """
        Add a column with the specified name and values to self.synth_df.
        This uses concat(), instead of simply adding columns, to avoid inefficiency issues.
        """
        self.synth_df = pd.concat([
            self.synth_df,
            pd.DataFrame({col_name: col_values})],
            axis=1)


    def modify_real_data(
        self, df: pd.DataFrame, num_modifications: int = 5
    ) -> Tuple[pd.DataFrame, List[Tuple[int, str]]]:
        """Introduce slight modifications to an existing dataset.

        Args:
            df: Real or synthetic dataset to modify.
            num_modifications: Number of cells to alter.

        Returns:
            A tuple with the modified DataFrame and a list describing the changed
            cells. Each list element is ``(row_index, column_name)``.
        """
        cell_list: List[Tuple[int, str]] = []
        for _ in range(num_modifications):
            row_index = random.randint(0, len(df) - 1)
            col_index = random.randint(0, len(df.columns) - 1)
            col_name = df.columns[col_index]

            if df.loc[row_index, col_name] is None:
                non_null_values = df[col_name].dropna()
                str_val = non_null_values.sample().values[0]
            else:
                str_val = str(df.loc[row_index, col_name]) + "9"
            df.loc[row_index, col_name] = str_val
            cell_list.append((row_index, col_name))

        return df, cell_list
