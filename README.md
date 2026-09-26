# DataConsistencyChecker

[![CI](https://github.com/DiogoRibeiro7/DataConsistencyChecker/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/DataConsistencyChecker/actions/workflows/python.yml)
[![Docs](https://github.com/DiogoRibeiro7/DataConsistencyChecker/actions/workflows/docs.yml/badge.svg?branch=main)](https://diogoribeiro7.github.io/DataConsistencyChecker/)
[![codecov](https://codecov.io/gh/DiogoRibeiro7/DataConsistencyChecker/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/DataConsistencyChecker)
![Python](https://img.shields.io/badge/python-3.10--3.14-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Interpretable data-quality checks, pattern discovery and outlier detection for tabular data.**

DataConsistencyChecker runs 158 simple, explainable checks over a pandas DataFrame. Each check looks for a
pattern in a column, a pair of columns or a larger set of columns (values always positive, one column the
product of two others, two dates always a week apart, and so on) and reports the rows that break it. Rows flagged
by many checks are the most unusual rows in the dataset, and every flag comes with an explanation.

📖 **Documentation: <https://diogoribeiro7.github.io/DataConsistencyChecker/>**

## Features

- **Interpretable** — every finding names the check, the columns and the rows, in plain language.
- **Broad** — single columns, pairs of columns, larger column sets and row order.
- **Mixed data** — numeric, categorical, string and date/time columns, with no encoding or binning.
- **Transparent scores** — a row's outlier score is the number of exceptions that flag it.
- **For EDA and outlier detection** — patterns without exceptions describe the data; exceptions point to
  unusual rows.
- **Automation-friendly** — immutable configuration, JSON-safe reports, a CLI and structured failure
  diagnostics.

## Installation

Python 3.10–3.14. The package is not on PyPI; install it from GitHub:

```bash
pip install "git+https://github.com/DiogoRibeiro7/DataConsistencyChecker.git"
```

or from a clone, with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/DiogoRibeiro7/DataConsistencyChecker.git
cd DataConsistencyChecker
poetry install
```

## Quickstart

```python
from data_consistency_checker import DataConsistencyChecker

dc = DataConsistencyChecker()
dc.init_data(df)                        # any pandas DataFrame
dc.check_data_quality()                 # run every check

dc.summarize_patterns_and_exceptions()  # what was found, per check
dc.get_exceptions_list()                # the patterns with exceptions
dc.display_detailed_results()           # each finding, with examples and plots
dc.display_most_flagged_rows()          # the most unusual rows, and why
```

One call, returning a JSON-safe report:

```python
from data_consistency_checker import DataConsistencyConfig, analyze

report = analyze(df, DataConsistencyConfig(verbose=-1))
payload = report.to_dict()
```

From the command line (CSV, TSV, JSON or JSONL):

```bash
data-consistency-checker check data.csv --output report.json
data-consistency-checker list-tests --details
```

See the [quickstart](https://diogoribeiro7.github.io/DataConsistencyChecker/getting-started/quickstart/) for a
complete worked example.

## How it works

Each check reports, for each column or set of columns it examines, one of three outcomes: a **pattern without
exceptions** (the property holds in every row), a **pattern with exceptions** (it holds in almost every row, and
the few rows that break it are flagged), or nothing. The *contamination level*, 0.5% of the rows by default,
sets how many rows may break a pattern for it to still count.

![A row flagged by three checks](docs/assets/images/outlier-row.jpg)

Findings can be reviewed as lists and summaries, displayed in detail with example rows and plots, exported as
HTML, pruned with `clear_results()`, and turned into per-row outlier scores. The
[user guide](https://diogoribeiro7.github.io/DataConsistencyChecker/guide/how-it-works/) explains each step, and
the [checks catalog](https://diogoribeiro7.github.io/DataConsistencyChecker/checks/) describes every check.

## Examples

The [`Demo Notebooks`](Demo%20Notebooks/) folder has worked examples on real datasets (California Housing,
Breast Cancer, Hypothyroid, OpenML datasets) and notebooks on specific features; see the
[examples page](https://diogoribeiro7.github.io/DataConsistencyChecker/examples/) for a guide.

## Development

```bash
poetry install
poetry run ruff check src
poetry run pytest
poetry install --with docs && poetry run mkdocs serve   # documentation preview
```

CI runs Ruff, mypy, the tests on Python 3.10–3.14 with coverage, a package build and a strict documentation
build; the documentation is published to GitHub Pages from `main`. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the full workflow, including how to add a check.

## License

[MIT](LICENSE)
