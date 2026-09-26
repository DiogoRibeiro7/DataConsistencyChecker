# Curating results

Any outlier detector needs some tuning. With DataConsistencyChecker, rather than tuning many parameters and
re-running, the recommended approach is to start from the full set of findings and **remove the ones that are
not meaningful** for your data, for example relationships between columns that are not really comparable.
What remains is a set of findings you can stand behind and report.

## Clearing findings

```python
dc.clear_results(test_id_list=["LARGER_DIFF_RANGE"])  # all findings of some checks
dc.clear_results(col_name_list=["id"])                # all findings involving some columns
dc.clear_results(pattern_id_list=[4, 7])              # some patterns, by Pattern ID
dc.clear_results(issue_id_list=[2])                   # some exceptions, by Issue ID
dc.clear_results(clear_code_tests=True)               # findings of the code / ID checks
dc.clear_results(clear_all_patterns=True)             # every pattern (exceptions are kept)
dc.clear_results(clear_all_exceptions=True)           # every exception (patterns are kept)
```

Exactly one option may be given per call; call it repeatedly to clear more. The lists, summaries and displays
then reflect only the remaining findings. Raw, normalized and per-cell scores are recalculated too; see
[Scores after curating findings](outlier-scores.md#scores-after-curating-findings).

Call `get_report()` again after clearing findings to capture the updated results. Previously created reports
are immutable snapshots and keep their original findings and scores.

## Restoring

```python
dc.restore_results()
```

This undoes every `clear_results()` call, restoring the findings and scores of the last `check_data_quality()`.

The [Demo_Clear_Issues notebook](https://github.com/DiogoRibeiro7/DataConsistencyChecker/blob/main/Demo%20Notebooks/Demo_Clear_Issues.ipynb)
shows a full example.

## Other ways to reduce noise

- Skip the code / ID checks with `check_data_quality(include_code_tests=False)` when no column holds codes or
  IDs.
- Exclude checks that are not relevant with `exclude_list`, or run only the relevant ones with `execute_list`.
- Drop columns that should not be compared (such as identifiers) before calling `init_data()`.
- Lower the contamination level to keep only the strongest patterns.
