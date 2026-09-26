# Architecture

`DataConsistencyChecker` is a single public class composed from focused mixins. Each mixin owns one
responsibility and shares state through the checker instance.

```text
DataConsistencyChecker
├── test implementations: Base, Numeric, Date, String, Binary, MultiColumn *TestsMixin
├── DisplayMixin         detailed display, examples, most / least flagged rows
├── PlotsMixin           plots and quick_report()
├── SynthDataMixin       synthetic data
├── ResultsMixin         lists, summaries, scores, reports, clear / restore
├── AnalysisCacheMixin   statistics and column combinations shared by many checks
├── DataInitMixin        constructor, init_data(), column types
├── ExecutionMixin       check_data_quality()
├── ExportMixin          HTML export
└── CheckerState         typed declaration of the shared state (annotations only)
```

## Flow of an analysis

1. **`init_data()`** (`DataInitMixin`) infers the type of each column, converts dates, takes a sample of the
   rows, and precomputes values used by many checks, such as numeric versions of the columns and their
   medians.
2. **`check_data_quality()`** (`ExecutionMixin`) selects the checks from the registry and runs each one. A
   failing check is recorded as a structured `OutlierDetectionError` and the others continue.
3. **Each check** (in `test_implementations/`) examines its columns or column sets, usually first on the sample,
   and passes a boolean series (True where a row follows the pattern) to `_process_analysis_binary()`
   (`ResultsMixin`). That decides, from the contamination level, whether it is a pattern, a pattern with
   exceptions or nothing, and records it.
4. **After the checks**, the results are assembled into the patterns and exceptions tables and the per-row
   results, from which the outlier scores are calculated.
5. **Results methods** (`ResultsMixin`, `DisplayMixin`, `PlotsMixin`, `ExportMixin`) read those tables.

Statistics used by several checks (missing-value masks, percentiles, column pairs that are identical or
ordered, and so on) are computed lazily and cached by `AnalysisCacheMixin`, so they are computed once per
dataset rather than once per check.

## The registry

`tests_definitions/` holds one module per category. Each maps check IDs to a `TestDefinition` with the
descriptions, the implementation and synthetic-data methods, and flags (short list, implemented, fast, code).
`get_test_catalog()` exposes the same information as public, immutable `TestMetadata` objects, which is also
how the [checks catalog](../checks.md) is generated.

## Typing the mixins

The mixins read and write state they do not define themselves (for example `self.orig_df`, set by
`DataInitMixin`), and call each other's methods. `CheckerState` declares that shared state and, for type
checkers only, the methods called across mixins, so mypy can check each mixin on its own. At run time it only
contributes annotations.

## The one-shot API

`analyze()` (`api.py`) wraps a checker for single-call use: an immutable `DataConsistencyConfig` (`config.py`)
describes the run, and `get_report()` returns an immutable, JSON-safe `DataConsistencyReport` (`report.py`). The
CLI (`cli.py`) is built on the same objects.
