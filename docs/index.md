---
title: Home
description: Discover patterns in tabular data and understand the rows that break them with explainable Python checks.
hide:
  - navigation
  - toc
---

<div class="dcc-hero" markdown="1">

<p class="dcc-eyebrow">DataConsistencyChecker · Python</p>

# Understand what makes a row unusual.

Discover patterns in your tables, find the rows that break them, and see the checks behind every finding.
Interpretable data quality and outlier detection, directly on pandas DataFrames.

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Explore the API](reference/checker.md){ .md-button }

</div>

<div class="dcc-cards" markdown="1">

<div markdown="1">

## Discover relationships

Find regularities within columns, between columns and across rows. Work with numeric, string, categorical
and date/time data without encoding or scaling.

[Browse the checks](checks.md)

</div>

<div markdown="1">

## Explain each exception

Review the rows that break a pattern, with descriptions, examples and plots. Trace an outlier score back
to the checks and columns that contributed to it.

[Review your results](guide/reviewing-results.md)

</div>

<div markdown="1">

## Repeat the analysis

Use the same configuration in notebooks, Python scripts and the CLI. Export structured JSON or HTML reports
to share findings and inspect failed checks.

[Use structured reports](guide/reports.md)

</div>

</div>

## From a table to an explanation

DataConsistencyChecker looks for patterns such as "values are always positive", "`total` is the product of
`price` and `quantity`", or "these two dates are always a week apart". It reports both the relationships that
hold and the rows that break them, helping you explore a dataset and investigate possible data-quality issues.

Start with a CSV file or an existing DataFrame:

```python
import pandas as pd

from data_consistency_checker import DataConsistencyChecker

df = pd.read_csv("orders.csv")

dc = DataConsistencyChecker()
dc.init_data(df)
dc.check_data_quality()
dc.summarize_patterns_and_exceptions()
dc.display_detailed_results()  # findings, example rows and plots
```

The [quickstart](getting-started/quickstart.md) includes a complete sample dataset and walks through the results.
For file-based workflows, the [command-line guide](guide/cli.md) covers CSV, TSV and JSON input.

## Read the findings in context

A row's outlier score counts the patterns with exceptions that flag it. Repeated flags help prioritize review;
an unusual row is not necessarily an error. Every score can be traced back to individual findings.

Read [how the checks work](guide/how-it-works.md) for contamination thresholds and limitations, or explore
[practical examples](examples.md) for analysis and reporting workflows.

## Acknowledgements

DataConsistencyChecker is based on the original
[DataConsistencyChecker](https://github.com/Brett-Kennedy/DataConsistencyChecker) by
[Brett Kennedy](https://github.com/Brett-Kennedy), which introduced the approach and the checks. This project
continues that work as an installable package, with tests and documentation. To cite it, use the repository's
[`CITATION.cff`](https://github.com/DiogoRibeiro7/DataConsistencyChecker/blob/main/CITATION.cff).
