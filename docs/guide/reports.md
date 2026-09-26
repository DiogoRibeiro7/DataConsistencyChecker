# Structured reports

For pipelines, services, experiment tracking or stored results, use a report rather than the checker's internal
DataFrames.

```python
report = dc.get_report()   # or: report = analyze(df, config)
payload = report.to_dict()

import json
json.dumps(payload)        # always valid, strict JSON
```

A `DataConsistencyReport` is an immutable snapshot: using the checker further, or changing the dictionary
returned by `to_dict()`, does not affect it.

## Contents

| Field | Contents |
|---|---|
| `n_rows`, `n_columns` | The size of the analysed data. |
| `executed_tests` | The IDs of the checks that ran. |
| `patterns` | One entry per pattern without exceptions: `Test ID`, `Column(s)`, `Description of Pattern`, `Pattern ID`. |
| `exceptions` | One entry per pattern with exceptions: the same fields plus `Number of Exceptions`, with an `Issue ID`. |
| `row_scores` | One entry per row: `row_id`, `FINAL SCORE` and `NORMALIZED SCORE`. |
| `execution_failures` | Structured diagnostics for checks that failed, as returned by `get_execution_failures()`. |

An exception entry looks like this:

```json
{
  "Test ID": "SIMILAR_TO_PRODUCT",
  "Column(s)": "\"price\" AND \"quantity\" AND \"total\"",
  "Description of Pattern": "...",
  "Number of Exceptions": 1,
  "Issue ID": 3
}
```

`to_dict()` converts NumPy scalars to Python numbers, dates to ISO strings, and missing or non-finite values to
`null`, so the result can always be serialised with `json.dumps()`.
