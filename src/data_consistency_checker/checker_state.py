"""Typed declaration of the state and cross-mixin API shared by the checker mixins.

``DataConsistencyChecker`` is assembled from many mixins that read each other's instance attributes and call each
other's methods. Every mixin inherits from :class:`CheckerState`, so a type checker sees one declaration of that
shared surface instead of reporting each use as a missing attribute.

:class:`CheckerState` has no runtime behaviour:

* The attribute declarations are annotations only. They create no class attributes, so an attribute that has not been
  assigned yet still raises ``AttributeError`` exactly as before. The values are assigned by the mixins, mostly in
  ``DataInitMixin.__init__()`` and ``DataInitMixin.init_data()``.
* The method declarations sit under ``if TYPE_CHECKING:`` and do not exist at runtime. They list only the methods that
  one mixin calls on another, with the exact signature of the implementing mixin. mypy checks each implementation
  against its declaration as an override, so an incompatible signature change that is not mirrored here is reported.

Attributes that the code only uses once they have been filled (``orig_df`` and ``test_results_by_column_np`` by
``init_data()``, ``synth_df`` by ``generate_synth_data()``) are declared with that type, not as optional.
``DataInitMixin`` assigns ``None`` to a few of them as a placeholder before they are filled; those assignments carry a
targeted ``type: ignore[assignment]``. Attributes that are legitimately ``None`` after initialisation and that the code
checks for ``None`` (for example ``patterns_df`` until ``check_data_quality()`` has run, or the lazily built analysis
caches) are declared optional, so mypy reports uses that do not handle ``None``.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .test_registry import TestDefinition


class CheckerState:
    """Instance state and cross-mixin methods shared by the ``DataConsistencyChecker`` mixins."""

    # Configuration, set in DataInitMixin.__init__()
    iqr_limit: float
    idr_limit: float
    max_combinations: int
    verbose: int
    DEBUG_MSG: bool

    # Test registry and the tests selected for the current check_data_quality() call
    test_dict: dict[str, TestDefinition]
    execute_list: list[str] | None
    exclude_list: list[str] | None
    execution_test_list: list[str]
    execution_times: dict[str, float] | None
    execution_failures: list[dict[str, Any]]
    n_tests_executed: int
    num_exceptions: int

    # Contamination levels. freq_contamination_level is a number of rows once check_data_quality() has set it.
    freq_contamination_level: float
    rare_contamination_level: float

    # Synthetic data, built by generate_synth_data()
    synth_df: pd.DataFrame
    num_synth_rows: int

    # The data examined and statistics about it, set by init_data()
    orig_df: pd.DataFrame
    num_rows: int
    num_valid_rows: dict[str, int]
    sample_df: pd.DataFrame
    binary_cols: list[str]
    numeric_cols: list[str]
    date_cols: list[str]
    string_cols: list[str]
    column_medians: dict[str, float]
    column_trimmed_means: dict[str, float]
    column_unique_vals: dict[str, list[Any]]  # Binary columns only; excludes None and NaN
    numeric_vals: dict[str, pd.Series]
    numeric_vals_filled: dict[str, pd.Series]
    sample_numeric_vals_filled: dict[str, pd.Series]
    # Set only when there are at least two numeric columns
    pearson_corr: pd.DataFrame
    spearman_corr: pd.DataFrame

    # Results. Each patterns_arr / results_summary_arr row is [test id, column(s), description, ...].
    patterns_arr: list[list[Any]]
    patterns_df: pd.DataFrame | None
    results_summary_arr: list[list[Any]]
    exceptions_summary_df: pd.DataFrame | None
    results_dict: dict[str, Any]
    test_results_df: pd.DataFrame | None
    test_results_by_column_np: npt.NDArray[np.float64]
    # Values are whatever column collection the test passed: a list, a tuple or a numpy array of column names
    col_to_original_cols_dict: dict[str, Collection[str]]
    single_test_summary_dict: dict[str, dict[str, Any]]
    single_test_summary_df: pd.DataFrame | None

    # Copies of the results saved by check_data_quality(), used by restore_issues()
    safe_patterns_arr: list[list[Any]]
    safe_patterns_df: pd.DataFrame | None
    safe_results_summary_arr: list[list[Any]]
    safe_exceptions_summary_df: pd.DataFrame | None
    safe_test_results_df: pd.DataFrame | None
    safe_results_dict: dict[str, Any]
    safe_test_results_by_column_np: npt.NDArray[np.float64] | None

    # Display and export state
    found_tests: list[str]
    current_display_test: int
    image_output_num: int
    output_folder: str | None

    # Analysis caches: None until first requested, then filled lazily by the AnalysisCacheMixin getters
    lower_limits_dict: dict[str, tuple[Any, Any, Any]] | None
    upper_limits_dict: dict[str, tuple[Any, Any, Any]] | None
    larger_pairs_dict: dict[tuple[str, str], pd.Series | None] | None
    larger_or_equal_pairs_dict: dict[tuple[str, str], pd.Series | None] | None
    larger_pairs_with_bool_dict: dict[tuple[str, str], bool] | None
    larger_or_equal_pairs_with_bool_dict: dict[tuple[str, str], bool] | None
    is_missing_dict: dict[str, pd.Series] | None
    sample_is_missing_dict: dict[str, pd.Series] | None
    num_missing_dict: dict[str, int] | None
    percentiles_dict: dict[str, pd.Series] | None
    nunique_dict: dict[str, int] | None
    count_most_freq_value_dict: dict[str, int] | None
    words_list_dict: dict[str, Any] | None  # Series.values of per-row word lists
    word_counts_dict: dict[str, list[int]] | None
    cols_same_bool_dict: dict[tuple[str, ...], bool] | None
    cols_same_count_dict: dict[tuple[str, ...], float] | None
    cols_pairs_both_null_dict: dict[tuple[str, ...], pd.Series] | None
    sample_cols_pairs_both_null_dict: dict[tuple[str, ...], pd.Series] | None
    col_pairs_either_null_bool_dict: dict[tuple[str, ...], bool] | None
    col_triples_all_null_bool_dict: dict[tuple[str, ...], bool] | None
    common_vals_dict: dict[str, list[Any]] | None

    if TYPE_CHECKING:
        # Methods called from a mixin other than the one implementing them. Signatures are copied verbatim from the
        # implementing mixin.

        # Implemented by AnalysisCacheMixin (analysis_cache_mixin.py)
        def get_similar_cols(
            self,
            full_cols_arr,
            include_self,
            lower_divisor,
            upper_multiplier,
            check_larger_false=False,
            check_larger_true=False,
        ): ...
        def get_limit_subset_sizes(self, full_cols_arr, similar_cols_dict, calc_size): ...
        def get_decision_tree_rules_as_categories(self, rules, categorical_features): ...
        def check_columns_same_scale_2(self, col_name_1, col_name_2, order=2): ...
        def check_columns_same_scale_3(self, col_name_1, col_name_2, col_name_3, order=2): ...
        def check_results_for_null(self, test_series, col_name, subset): ...
        def get_columns_iqr_upper_limit(self): ...
        def get_columns_iqr_lower_limit(self): ...
        def get_larger_pairs_dict(self, allow_equal, print_status=False): ...
        def get_larger_pairs_with_bool_dict(self): ...
        def get_is_missing_dict(self): ...
        def get_num_missing_dict(self): ...
        def get_sample_is_missing_dict(self): ...
        def get_percentiles_dict(self): ...
        def get_nunique_dict(self): ...
        def get_count_most_freq_value_dict(self): ...
        def get_words_list_dict(self): ...
        def get_word_counts_dict(self): ...
        def get_cols_same_bool_dict(self, force=False): ...
        def get_cols_same_count_dict(self): ...
        def get_col_pair_both_null_dict(self, force=False): ...
        def get_sample_col_pair_both_null_dict(self, force=False): ...
        def get_col_pairs_either_null_bool_dict(self, force=False): ...
        def get_col_triples_any_null_bool_dict(self): ...
        def get_common_values_dict(self): ...
        def _get_column_pairs_unique(self, force=False): ...
        def _get_binary_column_pairs_unique(self, same_vocabulary=True, force=False): ...
        def _get_numeric_column_pairs(self): ...
        def _get_numeric_column_pairs_unique(self, force=False): ...
        def _get_numeric_column_triples(self): ...
        def _get_numeric_column_triples_unique(self): ...
        def _get_string_column_pairs_unique(self, force=False): ...
        def _get_string_column_pairs(self): ...
        def _get_date_column_pairs_unique(self, force=False): ...

        # Implemented by DataInitMixin (data_init_mixin.py)
        def init_data(self, df: pd.DataFrame, known_date_cols: list[str] | None = None) -> None: ...

        # Implemented by DisplayMixin (display_mixin.py)
        def get_test_list(self) -> list[str]: ...
        def get_patterns_shortlist(self) -> list[str]: ...
        def get_tests_for_codes(self) -> list[str]: ...
        def _output_current_test(self, test_num, test_id): ...

        # Implemented by ExecutionMixin (execution_mixin.py)
        def check_data_quality(
            self,
            append_results: bool = False,
            execute_list: list[str] | None = None,
            exclude_list: list[str] | None = None,
            test_start_id: int = 0,
            fast_only: bool = False,
            include_code_tests: bool = True,
            freq_contamination_level: float = 0.005,
            rare_contamination_level: float = 0.1,
            run_parallel: bool = False,
            raise_on_error: bool = False,
        ) -> None: ...

        # Implemented by PlotsMixin (plots_mixin.py)
        def _draw_results_plots(self, test_id, cols, columns_set, show_exceptions, display_info, f): ...

        # Implemented by ResultsMixin (results_mixin.py)
        def get_results_col_name(self, test_id, col_name): ...
        def get_test_ids_with_results(self, include_patterns=True, include_exceptions=True): ...
        def get_patterns_list(
            self,
            test_exclude_list: list[str] | None = None,
            column_exclude_list: list[str] | None = None,
            show_short_list_only: bool = True,
        ) -> pd.DataFrame | None: ...
        def get_exceptions_list(self) -> pd.DataFrame | None: ...
        def summarize_patterns_by_test_and_feature(self, all_tests=False, heatmap=False): ...
        def summarize_exceptions_by_test_and_feature(self, all_tests=False, heatmap=False): ...
        def summarize_exceptions_by_test(self, heatmap=False): ...
        def summarize_patterns_and_exceptions(self, all_tests: bool = False, heatmap: bool = False) -> pd.DataFrame: ...
        def _get_condensed_col_list(self, test_id, col_name): ...
        def _get_rows_flagged(self, test_id, col_name): ...
        def _process_analysis_binary(
            self,
            test_id,
            original_cols,
            test_series,
            pattern_string,
            exception_str="",
            allow_patterns=True,
            display_info=None,
        ): ...
        def _process_analysis_counts(
            self,
            test_id,
            original_cols,
            test_series,
            pattern_string_1,
            pattern_string_2,
            allow_patterns=True,
            display_info=None,
        ): ...
        def _output_stats(self): ...
        def _calculate_final_scores(self) -> None: ...

        # Implemented by SynthDataMixin (synth_data_mixin.py)
        def generate_synth_data(
            self,
            all_cols: bool = False,
            execute_list: list[str] | None = None,
            exclude_list: list[str] | None = None,
            seed: int = 0,
            add_nones: str = "none",
        ) -> pd.DataFrame: ...
        def _add_synthetic_column(self, col_name, col_values): ...
