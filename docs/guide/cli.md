# Command line

Installing the package provides the `data-consistency-checker` command. `python -m data_consistency_checker`
is equivalent.

## `check`

```bash
data-consistency-checker check data.csv --output report.json
```

This loads a CSV, TSV, JSON or JSONL file (by extension), runs the checks, prints a one-line summary and, with
`--output`, writes the [structured report](reports.md) as JSON:

```text
rows=500 columns=4 tests=158 patterns=19 exceptions=5 failures=0
```

| Option | Meaning |
|---|---|
| `--config FILE` | Load settings from a JSON or TOML [configuration file](configuration.md#configuration-files). |
| `-o`, `--output FILE` | Write the structured report to this JSON file. |
| `--tests ID [ID ...]` | Run only these checks. |
| `--exclude-tests ID [ID ...]` | Run every check except these. |
| `--date-column COLUMN` | Treat this column as a date column. Repeat for several columns. |
| `--fast-only` | Run only the checks marked as fast. |
| `--max-combinations N` | Cap on the column combinations one check may examine. |
| `--raise-on-error` | Stop at the first failing check. |
| `-v`, `--verbose {-1,0,1,2}` | Progress output. |

`--tests` and `--exclude-tests` are mutually exclusive, and unknown check IDs are rejected. Options given on the
command line override the same settings from `--config`; settings not given keep the file's values.

## `list-tests`

```bash
data-consistency-checker list-tests            # check IDs
data-consistency-checker list-tests --details  # with descriptions and flags
data-consistency-checker list-tests --json     # machine-readable
```

## In CI

```bash
data-consistency-checker check data/latest.csv \
  --config checker.toml \
  --raise-on-error \
  --output artifacts/data-consistency.json
```

The report can then be inspected, archived or compared between runs.
