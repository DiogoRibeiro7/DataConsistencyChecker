# Quickstart

This walk-through uses a small order table where `total` is always `price × quantity`, except in one row that
was corrupted.

```python
import numpy as np
import pandas as pd

from data_consistency_checker import DataConsistencyChecker

rows = np.arange(500)
df = pd.DataFrame({
    "price": (rows % 50) * 2.0 + 10,
    "quantity": rows % 7 + 1,
    "region": np.where(rows % 3 == 0, "north", "south"),
})
df["total"] = df["price"] * df["quantity"]
df.loc[42, "total"] = 1.0  # the corrupted row
```

## 1. Run the checks

```python
dc = DataConsistencyChecker()
dc.init_data(df)
dc.check_data_quality()
```

`init_data()` infers the type of each column (numeric, string, binary or date) and caches statistics used by
many checks. `check_data_quality()` then runs every check. Pass `verbose=-1` to the constructor to silence
progress output.

## 2. Get an overview

```python
dc.summarize_patterns_and_exceptions()
```

This returns one row per check that found something, with the number of patterns found without and with
exceptions. For this data it includes, among others:

| Test ID | Number Patterns without Exceptions | Number Patterns with Exceptions |
|---|---|---|
| `POSITIVE` | 3 | |
| `UNUSUAL_ORDER_MAGNITUDE` | | 1 |
| `MULTIPLE_OF_CONSTANT` | 1 | 1 |

The findings themselves are available as DataFrames:

```python
dc.get_patterns_list()    # patterns without exceptions
dc.get_exceptions_list()  # patterns with exceptions, with the number of rows that break them
```

Here `get_exceptions_list()` lists five exceptions, all involving `total`. The most informative is
`SIMILAR_TO_PRODUCT` on `"price" AND "quantity" AND "total"`: `total` is the product of `price` and `quantity`
in every row but one.

## 3. Look at the details

```python
dc.display_detailed_results(test_id_list=["SIMILAR_TO_PRODUCT"])
```

For each finding this shows a description, example rows that follow the pattern, the rows that break it and,
for many checks, a plot. Filter by check, column, row or finding ID to focus on what matters; see
[Reviewing results](../guide/reviewing-results.md).

## 4. Find the most unusual rows

```python
dc.display_most_flagged_rows()
dc.get_results_by_row_id(42)
```

Row 42 is flagged by five different checks, so it has an outlier score of 5, while every other row scores 0.
`get_results_by_row_id(42)` lists the five `(check, columns)` pairs that flag it.

## 5. Keep the results

```python
import json
from pathlib import Path

report = dc.get_report()        # a structured, JSON-safe snapshot
payload = report.to_dict()
Path("report.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

dc.display_detailed_results(save_to_disk=True, output_folder="reports")  # an HTML report
```

This saves `report.json` and `reports/Data_consistency.html`. Keep any PNG files written alongside the HTML
when sharing it. See [Structured reports](../guide/reports.md) for the JSON fields.

## The same analysis in one call

```python
from data_consistency_checker import DataConsistencyConfig, analyze

report = analyze(df, DataConsistencyConfig(verbose=-1))
```

To run the same example from the command line, first save the DataFrame:

```python
df.to_csv("orders.csv", index=False)
```

Then run this in a terminal from the directory containing `orders.csv`:

```bash
data-consistency-checker check orders.csv --output report.json
```

## Next steps

- [How it works](../guide/how-it-works.md) explains patterns, exceptions and contamination levels.
- [Running checks](../guide/running-checks.md) covers choosing checks and tuning the run.
- The [checks catalog](../checks.md) describes every check.
