# DataConsistencyChecker

**Interpretable data-quality checks, pattern discovery and outlier detection for tabular data.**

DataConsistencyChecker runs a large set of simple, explainable checks over a pandas DataFrame. Each check looks
for a *pattern* in one column, a pair of columns or a larger set of columns (for example "values are always
positive", "`total` is the product of `price` and `quantity`", "these two dates are always a week apart"), and
reports the rows that break it. Rows flagged by many checks are the most unusual rows in the dataset.

It works directly on categorical, numeric, string and date/time data: no encoding, binning, scaling or distance
metric is needed.

<div class="grid cards" markdown>

- **[Installation](getting-started/installation.md)** — install from GitHub or a clone.
- **[Quickstart](getting-started/quickstart.md)** — run the checks and read the results in a few lines.
- **[How it works](guide/how-it-works.md)** — patterns, exceptions, contamination and scores.
- **[Checks catalog](checks.md)** — every check, generated from the code.
- **[API reference](reference/checker.md)** — the full public API.
- **[Command line](guide/cli.md)** — check CSV, TSV and JSON files without writing Python.

</div>

## A first look

```python
from data_consistency_checker import DataConsistencyChecker

dc = DataConsistencyChecker()
dc.init_data(df)                 # any pandas DataFrame
dc.check_data_quality()          # run every check
dc.summarize_patterns_and_exceptions()
dc.display_detailed_results()    # each finding, with examples and plots
```

## Why use it

- **Interpretable** — every finding says which check, which columns and which rows, in plain language.
- **Broad** — checks cover single columns, pairs of columns, larger column sets and row order.
- **Mixed data types** — numeric, categorical, string and date/time columns are handled natively.
- **Transparent scoring** — a row's outlier score is the number of exceptions that flag it.
- **Useful for EDA too** — patterns without exceptions describe the data as much as the exceptions do.
- **Automation-friendly** — an immutable configuration, JSON-safe reports and a CLI for pipelines and CI.

## Two uses

**Exploratory data analysis.** Run the checks and review the patterns found, with or without exceptions. The
checks go well beyond distributions and correlations, so they complement other EDA tools.

**Interpretable outlier detection.** Rows flagged repeatedly, for different checks on different columns, are
reliably unusual. Unlike most detectors, you can see exactly why each row was flagged. The checks are
deliberately simple, so they do not replace multivariate detectors such as Isolation Forest or Local Outlier
Factor; they complement them.
