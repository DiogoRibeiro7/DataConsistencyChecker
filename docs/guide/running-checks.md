# Running checks

## Creating a checker

```python
from data_consistency_checker import DataConsistencyChecker

dc = DataConsistencyChecker(
    iqr_limit=3.5,             # inter-quartile range multiplier for very large / small values
    idr_limit=1.0,             # inter-decile range multiplier, used for strictly positive data
    max_combinations=100_000,  # cap on column combinations examined by a single check
    verbose=0,                 # -1 silent, 0 summary, 1 check names, 2 check descriptions
)
```

## Loading data

```python
dc.init_data(df, known_date_cols=None)
```

`init_data()` takes a pandas DataFrame and infers each column's type:

- **binary** — exactly two distinct values;
- **date** — datetime columns, plus text or numeric columns that parse as dates;
- **numeric** and **string** — the rest.

Columns with a single distinct value are ignored. Check the inferred types with `display_columns_types_list()`
or `display_columns_types_table()`. If date detection gets a column wrong, pass the date columns explicitly
with `known_date_cols`: those columns are converted to dates, and no other text or numeric column is. Columns
that already have a datetime type are always treated as dates.

!!! tip "Sort the data first"
    Several checks look at row order: increasing or decreasing values, values similar to the previous row,
    running sums and so on. If the data has a natural order (for example by time) but was loaded in another
    order, sort it before calling `init_data()`.

## Running the checks

```python
dc.check_data_quality(
    execute_list=None,               # run only these checks
    exclude_list=None,               # run every check except these
    fast_only=False,                 # run only checks marked as fast
    include_code_tests=True,         # include checks specific to code / ID values
    freq_contamination_level=0.005,  # fraction (or count) of rows allowed to break a pattern
    raise_on_error=False,            # stop at the first failing check instead of continuing
)
```

### Choosing checks

`execute_list` and `exclude_list` take check IDs from the [checks catalog](../checks.md), and cannot be used
together. `get_test_list()` returns every ID, and `get_test_catalog()` returns typed metadata for each check.

`fast_only=True` runs only the checks marked as fast. `include_code_tests=False` skips the checks that only make
sense for code or ID values, where individual characters are meaningful (`get_tests_for_codes()` lists them).

### Contamination level

`freq_contamination_level` is the maximum number of rows that may break a pattern for it to still be reported.
Give a fraction of the rows (the default is 0.5%) or an integer count of rows. See
[How it works](how-it-works.md#the-contamination-level).

### Limiting combinations

Checks on pairs or larger sets of columns can examine a very large number of combinations. A check skips itself
when it would examine more than `max_combinations`; with `verbose=1` or higher, it prints a message saying so.
Lower the limit for faster runs, or raise it to examine more combinations.

### Adding to or resuming a run

`append_results=True` adds the results of another `check_data_quality()` call to the existing ones instead of
replacing them, which is useful for building up results over several runs. `test_start_id` starts from a given
check number, to resume an interrupted run.

### When a check fails

A failing check does not stop the others by default. Its failure is kept as structured data:

```python
for failure in dc.get_execution_failures():
    print(failure["test_id"], failure["error"]["type"], failure["error"].get("cause"))
```

Each failure is an `OutlierDetectionError` that keeps the original exception as its cause. With
`raise_on_error=True`, the first failure is raised instead (with the original exception as `__cause__`), which
suits CI jobs where any failure should stop the run.

!!! note "`run_parallel`"
    `run_parallel=True` is not supported: checks store their results on the checker, so checks run in other
    processes could not report them back. Passing it issues a `RuntimeWarning` and runs the checks
    sequentially.
