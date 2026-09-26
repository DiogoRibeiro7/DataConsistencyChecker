# Contributing to DataConsistencyChecker

Thank you for considering a contribution. This guide covers the development setup, the checks every change must
pass, and how to add a new consistency check.

## Development setup

Requirements: Python 3.10–3.14, [Poetry](https://python-poetry.org/) and Git.

```bash
git clone https://github.com/DiogoRibeiro7/DataConsistencyChecker.git
cd DataConsistencyChecker
poetry install                   # the package, its dependencies and the dev tools
poetry run pre-commit install    # optional: run Ruff and other hooks on each commit
```

The lock file is committed, so everyone uses the same dependency versions. After changing dependencies in
`pyproject.toml`, run `poetry lock` and commit both files.

## Checks every change must pass

These are the checks CI runs. Run them locally before opening a pull request:

```bash
poetry check --lock                                        # pyproject.toml and poetry.lock agree
poetry run ruff check src                                  # full rule set on the package
poetry run ruff check tests --select E9,F63,F7,F82         # correctness rules on the tests
poetry run pytest --cov=data_consistency_checker --cov-report=term-missing
poetry build                                               # the package builds
```

CI runs the tests on Python 3.10, 3.11, 3.12, 3.13 and 3.14, and Codecov reports the coverage of the lines each
pull request changes (*patch coverage*): new and changed code should be covered by tests. `poetry run mypy src`
also runs in CI; it is not yet blocking, but should not get worse.

If you change the documentation, also build it (see [Documentation](#documentation)).

## Project layout

```text
src/data_consistency_checker/
├── __init__.py              # public API: DataConsistencyChecker, analyze, DataConsistencyConfig, ...
├── checker.py               # DataConsistencyChecker, composed from the mixins below
├── checker_state.py         # typed declaration of the state shared by the mixins
├── data_init_mixin.py       # constructor and init_data(): column types and cached statistics
├── execution_mixin.py       # check_data_quality(): selecting and running the checks
├── results_mixin.py         # lists, summaries, outlier scores, reports, clear/restore
├── display_mixin.py         # detailed display, examples, most/least flagged rows
├── plots_mixin.py           # plots and quick_report()
├── export_mixin.py          # HTML export
├── synth_data_mixin.py      # synthetic data generation
├── analysis_cache_mixin.py  # statistics and column combinations shared by many checks
├── test_implementations/    # the checks themselves, one mixin per category
├── tests_definitions/       # the registry: ID, descriptions and flags of every check
├── test_registry.py         # TestDefinition (internal)
├── test_metadata.py         # TestMetadata (public)
├── api.py, config.py, report.py   # analyze(), DataConsistencyConfig, DataConsistencyReport
└── cli.py, __main__.py      # the data-consistency-checker command
tests/                       # pytest suite
docs/                        # MkDocs site
check_data_consistency.py    # legacy import shim
```

Use package-relative (or `data_consistency_checker.`) imports inside the package, and import
`DataConsistencyChecker` from `data_consistency_checker` in new examples.

## Adding a check

A check has three parts: a registry entry, an implementation with a synthetic-data generator, and a test module.

### 1. Register it

Add an entry to the `get_*_tests(checker)` function of the matching module in `tests_definitions/`:

| Module | For checks on |
|---|---|
| `base_tests.py` | single columns or pairs of any type |
| `numeric_tests.py` | numeric columns |
| `date_tests.py` | date and time columns |
| `string_tests.py` | string columns, including code / ID values |
| `binary_tests.py` | binary columns |
| `multi_column_tests.py` | three or more columns, or whole rows |

```python
'YOUR_CHECK_ID': _test(
    '',                                          # short description for progress output ('' to reuse the next)
    'Check if ...',                              # description, shown in the docs and CLI
    checker._check_your_check,
    checker._generate_your_check,
    False,  # shortlist: report patterns without exceptions by default
    True,   # implemented
    True,   # fast: quick even with many columns
    False,  # code: specific to code / ID values
),
```

### 2. Implement it

Add `_check_your_check(self, test_id)` and `_generate_your_check(self)` to the matching mixin in
`test_implementations/`. A typical check builds a boolean series, True where a row follows the pattern, and
hands it to `_process_analysis_binary()`, which records a pattern, a pattern with exceptions, or nothing,
according to the contamination level:

```python
def _check_your_check(self, test_id):
    for col_name in self.numeric_cols:
        test_series = self.orig_df[col_name].isna() | (self.numeric_vals_filled[col_name] >= 0)
        self._process_analysis_binary(
            test_id,
            [col_name],
            test_series,
            "The column contains consistently ...",
        )


def _generate_your_check(self):
    """Columns with the pattern, with and without an exception."""
    self._add_synthetic_column('your check all', [...])
    self._add_synthetic_column('your check most', [...])  # the same, with one row breaking the pattern
```

**Use a single leading underscore.** Name implementation methods `_check_*` / `_generate_*`, never
`__check_*`: Python renames double-underscore methods per class, so another mixin (or the registry) cannot
reach them.

Checks on combinations of columns should skip themselves, with a message when `verbose >= 1`, if there are more
combinations than `self.max_combinations`; the helpers in `analysis_cache_mixin.py` (such as
`_get_numeric_column_pairs_unique()`) return `None` for the pairs in that case.

### 3. Test it

Add `tests/test_YOUR_CHECK_ID.py`, following the existing per-check modules: it declares the columns of the
synthetic data expected to show the pattern and the exceptions, and the shared helpers in `tests/utils.py` run
the check on its synthetic data and compare the results.

```python
from utils import requires_synthetic, synth_test

test_id = "YOUR_CHECK_ID"

synth_patterns_cols = ["your check all"]
synth_exceptions_cols = ["your check most"]


@requires_synthetic
def test_synthetic_no_nulls():
    synth_test(test_id, "none", synth_patterns_cols, synth_exceptions_cols)
```

Add focused tests for any branch the synthetic data does not reach, and for missing values. Behaviour tests
should go through the public API where practical.

## Documentation

The documentation is built with MkDocs and published to GitHub Pages from `main`. The API reference is generated
from the docstrings (Google style), and the checks catalog from the registry.

```bash
poetry install --with docs
poetry run mkdocs serve           # preview at http://127.0.0.1:8000
poetry run mkdocs build --strict  # what CI runs
```

Keep the Material theme dark by default (slate, indigo, cyan), with a visible light-mode toggle. The landing
page uses the `dcc-hero` and three-column `dcc-cards` layouts in `docs/stylesheets/extra.css`; check layout,
contrast and navigation on desktop and mobile in both themes when changing them. Keep the design restrained
and the content specific to the package.

Use mkdocstrings for API signatures, Mermaid fences for workflow diagrams and LaTeX math for formulas.
The API reference and checks catalog must continue to reflect the source code. Builds are strict by default;
fix build warnings before submitting. The Docs workflow validates every pull request and publishes relevant
changes on `main` with the official configure, upload-artifact and deploy GitHub Pages Actions.

## Pull requests

- Keep each pull request focused on one change, and describe what it changes and how it was validated.
- Add or update tests with every behaviour change; a bug fix should come with a test that fails without it.
- Update the docstrings and the documentation when user-facing behaviour changes.
- Commit messages start with a short imperative summary line (for example `Fix SAME_FIRST_CHARS prefix length`),
  optionally followed by a body explaining why.
