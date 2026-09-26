# Testing

```bash
poetry run pytest                                   # the whole suite
poetry run pytest tests/test_POSITIVE.py            # one module
poetry run pytest --cov=data_consistency_checker --cov-report=term-missing
```

## What the suite contains

- **Per-check modules** — `tests/test_<CHECK_ID>.py`, one per check. Each declares the columns of the check's
  synthetic data expected to show a pattern and to show exceptions, and uses the helpers in `tests/utils.py` to
  run the check and compare. Modules for checks that are not implemented are skipped by `tests/conftest.py`.
- **Behaviour tests** — for example `test_display_detailed_results.py`, `test_sample_not_flagged.py`,
  `test_cli.py`, `test_structured_report.py` and `test_regressions.py` (one test per fixed defect). They exercise
  the public API on small, purpose-built datasets.
- **Documentation examples** — `test_docs_examples.py` runs the examples of the documentation and checks the
  results they describe.

## Optional variants

Each per-check module also defines variants that are skipped by default, controlled by flags at the top of
`tests/utils.py`:

| Flag | Enables |
|---|---|
| `TEST_SYNTHETIC_NONES` | The synthetic tests with missing values (one row, in-sync, random and 80% missing). |
| `TEST_SYNTHETIC_ALL_COLUMNS` | The tests on the synthetic columns of every check at once. |
| `TEST_REAL` | The tests on real datasets from OpenML (downloaded, then cached in `dc_cache/`). |

Skipped variants are reported as such, with the flag to set in the reason, so the pass count only includes tests
that ran. Some variants with missing values currently fail; enabling them is useful when working on missing-value
handling.

## Continuous integration

Every pull request to `main` runs Ruff, mypy (not blocking), the tests with coverage on Python 3.10–3.14, a
package build with smoke tests of the imports and the CLI, and a strict documentation build. Codecov reports the
coverage of the changed lines.
