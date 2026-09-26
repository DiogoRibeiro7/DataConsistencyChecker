# DataConsistencyChecker

The main class. Create a checker, load data with `init_data()`, run the checks with `check_data_quality()`, then
use the methods below to examine the results.

::: data_consistency_checker.DataConsistencyChecker
    options:
      show_root_heading: true
      heading_level: 2
      members: false

## Loading data and running checks

::: data_consistency_checker.data_init_mixin.DataInitMixin.init_data
    options:
      heading_level: 3

::: data_consistency_checker.execution_mixin.ExecutionMixin.check_data_quality
    options:
      heading_level: 3

## Lists and summaries

::: data_consistency_checker.results_mixin.ResultsMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - get_patterns_list
        - get_exceptions_list
        - get_test_ids_with_results
        - get_results_by_row_id
        - get_exceptions
        - get_exceptions_by_column
        - summarize_patterns_and_exceptions
        - summarize_patterns_by_test
        - summarize_exceptions_by_test
        - summarize_patterns_by_test_and_feature
        - summarize_exceptions_by_test_and_feature
        - get_single_feature_tests_matrix

## Outlier scores

::: data_consistency_checker.results_mixin.ResultsMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - get_outlier_scores
        - get_outlier_score_summary

## Detailed display

::: data_consistency_checker.display_mixin.DisplayMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - display_detailed_results
        - display_next
        - display_most_flagged_rows
        - display_least_flagged_rows
        - display_columns_types_list
        - display_columns_types_table

## Plots and quick report

::: data_consistency_checker.plots_mixin.PlotsMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - quick_report
        - plot_final_scores_distribution_by_row
        - plot_final_scores_distribution_by_feature
        - plot_final_scores_distribution_by_test
        - plot_columns_vs_final_scores
        - check_data_quality_by_feature_pairs

## Curating results

::: data_consistency_checker.results_mixin.ResultsMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - clear_results
        - restore_results

## Reports and failures

::: data_consistency_checker.results_mixin.ResultsMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - get_report
        - get_execution_failures

## Information about the checks

::: data_consistency_checker.display_mixin.DisplayMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - get_test_list
        - get_test_catalog
        - get_test_descriptions
        - print_test_descriptions
        - get_patterns_shortlist
        - get_tests_for_codes
        - demo_test

## Synthetic data

::: data_consistency_checker.synth_data_mixin.SynthDataMixin
    options:
      show_root_heading: false
      show_bases: false
      heading_level: 3
      members:
        - generate_synth_data
        - modify_real_data
