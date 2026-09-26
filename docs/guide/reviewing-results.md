# Reviewing results

After `check_data_quality()`, the results can be listed, summarised, displayed in detail, exported or plotted.

## Lists

| Method | Returns |
|---|---|
| `get_patterns_list()` | A DataFrame of the patterns found without exceptions: check ID, columns, description and a `Pattern ID`. |
| `get_exceptions_list()` | A DataFrame of the patterns found with exceptions, with the number of exceptions and an `Issue ID`. |
| `get_test_ids_with_results()` | The IDs of the checks that found patterns and/or exceptions. |
| `get_results_by_row_id(row)` | The `(check, columns)` pairs that flag one row. |
| `get_exceptions()` | A DataFrame with one column per exception (True where a row is flagged) and the final scores. |
| `get_exceptions_by_column()` | A DataFrame shaped like the original data, with a score per cell. |

`get_patterns_list()` shows only the checks in the short list by default; pass `show_short_list_only=False` to
include all patterns. It also accepts `test_exclude_list` and `column_exclude_list`.

## Summaries

| Method | Summarises |
|---|---|
| `summarize_patterns_and_exceptions()` | For each check, the number of patterns found without and with exceptions. |
| `summarize_patterns_by_test()` | For each check, the number of columns with a pattern without exceptions. |
| `summarize_exceptions_by_test()` | For each check, the columns and rows flagged, and the number of issues. |
| `summarize_patterns_by_test_and_feature()` | A check × column grid, marking where patterns were found. |
| `summarize_exceptions_by_test_and_feature()` | A check × column grid, counting flagged rows. |
| `get_single_feature_tests_matrix()` | A check × column grid of the single-column checks' results, with `-` where a check did not run. |

Most summaries accept `heatmap=True` to also plot the table. `quick_report()` displays the main lists, summaries
and score plots in one call.

## Details

`display_detailed_results()` shows each finding with a description, example rows that follow the pattern, the
rows that break it and, for many checks, a plot. Without filters it can produce a lot of output, so narrow it
down:

```python
dc.display_detailed_results(test_id_list=["SIMILAR_TO_PRODUCT"])  # some checks
dc.display_detailed_results(col_name_list=["total"])              # findings involving some columns
dc.display_detailed_results(issue_id_list=[3])                    # some exceptions (Issue IDs)
dc.display_detailed_results(pattern_id_list=[0, 2])               # some patterns (Pattern IDs)
dc.display_detailed_results(row_id_list=[42])                     # exceptions flagging some rows
```

Options control what is shown: `show_patterns`, `show_exceptions`, `show_short_list_only`, `include_examples`,
`plot_results` and `max_shown`. When no filter is given and there are more findings than `max_shown` (by
default 200, halved when examples are included and again when plots are), the method explains how to narrow the
results instead of printing them all.

`display_next()` shows the findings of one check at a time, in execution order, which spreads long output over
several notebook cells.

## Rows

```python
dc.display_most_flagged_rows(n_rows=10)   # the most unusual rows, with the checks that flagged them
dc.display_least_flagged_rows(n_rows=10)  # the most typical rows, for context
```

With `with_results=True` (the default), each row is shown with the checks that flagged each of its columns;
with `with_results=False`, the rows are shown in one table.

## HTML reports

```python
dc.display_detailed_results(save_to_disk=True, output_folder="reports")
```

This writes `Data_consistency.html` into `output_folder` (an `Output` folder in the working directory by
default), with plots saved as PNG files beside it. Saving to disk allows many more findings than a notebook can
comfortably render.

## Score plots

| Method | Shows |
|---|---|
| `plot_final_scores_distribution_by_row()` | The distribution of outlier scores across rows. |
| `plot_final_scores_distribution_by_feature()` | The scores attributed to each column. |
| `plot_final_scores_distribution_by_test()` | The number of exceptions found by each check. |
| `plot_columns_vs_final_scores()` | How each column's values relate to the rows' scores. |
