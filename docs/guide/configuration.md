# Configuration and one-shot analysis

For scripts, pipelines and services, a whole analysis can be described by one immutable configuration object
and run in one call.

## `analyze()`

```python
from data_consistency_checker import DataConsistencyConfig, analyze

config = DataConsistencyConfig(
    execute_tests=("MISSING_VALUES", "VERY_LARGE"),
    known_date_cols=("event_date",),
    max_combinations=50_000,
    verbose=-1,
)

report = analyze(df, config)
payload = report.to_dict()
```

`analyze()` creates a checker, loads the data, runs the checks and returns a
[structured report](reports.md). Without a configuration it uses the defaults.

## `DataConsistencyConfig`

| Field | Default | Meaning |
|---|---|---|
| `iqr_limit` | `3.5` | Inter-quartile range multiplier for very large / small values. |
| `idr_limit` | `1.0` | Inter-decile range multiplier, used for strictly positive data. |
| `max_combinations` | `100_000` | Cap on the column combinations one check may examine. |
| `verbose` | `0` | Progress output: -1, 0, 1 or 2. |
| `known_date_cols` | `()` | Columns to treat as dates. |
| `execute_tests` | `()` | Run only these checks. |
| `exclude_tests` | `()` | Run every check except these. |
| `fast_only` | `False` | Run only the checks marked as fast. |
| `include_code_tests` | `True` | Include the checks specific to code / ID values. |
| `freq_contamination_level` | `0.005` | Fraction (or count) of rows allowed to break a pattern. |
| `rare_contamination_level` | `0.1` | Accepted for compatibility; not currently used by any check. |
| `run_parallel` | `False` | Not supported; runs sequentially with a warning. |
| `raise_on_error` | `False` | Stop at the first failing check instead of recording the failure. |

The configuration is validated on creation: `execute_tests` and `exclude_tests` are mutually exclusive,
`max_combinations` must be positive, `verbose` must be -1 to 2, and contamination levels must not be negative.

The lower-level `DataConsistencyChecker` API remains available for stateful work such as appending results or
resuming a run.

## Configuration files

A configuration round-trips through `to_dict()` / `from_dict()`, and can be loaded from JSON or TOML with
`DataConsistencyConfig.from_file(path)`. Unknown fields are rejected.

```toml
# checker.toml
[data_consistency_checker]
execute_tests = ["MISSING_VALUES", "VERY_LARGE"]
known_date_cols = ["event_date"]
max_combinations = 50000
verbose = -1
```

In TOML the settings may sit under a `[data_consistency_checker]` table (convenient inside a larger file) or at
the top level. JSON files use the same field names at the top level.

The same file can be used from Python, the [command line](cli.md) and CI:

```bash
data-consistency-checker check data.csv --config checker.toml --output report.json
```
