# Synthetic data and demos

Every check comes with a generator of synthetic columns designed to trigger it: typically one column set with a
pattern and no exceptions, and one with the pattern and a few exceptions. This is a quick way to see what a check
does, and it is what the test suite uses.

## Generating data

```python
dc = DataConsistencyChecker()
synth_df = dc.generate_synth_data(execute_list=["SIMILAR_TO_PRODUCT"])
```

| Parameter | Meaning |
|---|---|
| `all_cols` | Generate the columns for every check, not only those selected. |
| `execute_list`, `exclude_list` | Choose the checks whose columns are generated. |
| `seed` | Random seed, for reproducible data. |
| `add_nones` | Add missing values: `"none"`, `"one-row"`, `"in-sync"`, `"random"` or `"80-percent"`. |

The generated columns are named after the check, for example `positive all` (a pattern without exceptions) and
`positive most` (the same pattern with one exception) for `POSITIVE`.

`modify_real_data(df, num_modifications=5)` returns a copy of real data with a few values modified, together
with the list of modified `(row, column)` cells, which is useful for checking what the checker finds.

## Demonstrating a check

```python
dc.demo_test("SIMILAR_TO_PRODUCT")
dc.demo_test("SIMILAR_TO_PRODUCT", include_nulls=True)  # also with several kinds of missing values
```

`demo_test()` generates the synthetic data for one check, shows it, runs the check and displays the detailed
results.

## Describing the checks

```python
dc.get_test_descriptions()                 # {check ID: description}
dc.print_test_descriptions(long_desc=True) # descriptions, with each check's full documentation
```
