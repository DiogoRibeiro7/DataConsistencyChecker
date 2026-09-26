# Performance

Small datasets are typically checked in under a minute. Very large ones, with hundreds of thousands of rows and
hundreds of columns, may take up to half an hour. To make runs faster:

- **Run only the fast checks** with `fast_only=True`.
- **Choose the checks**: `exclude_list` drops slow or irrelevant checks; `execute_list` runs only the relevant
  ones.
- **Lower `max_combinations`.** Checks on pairs or triples of columns examine many combinations: with 50 numeric
  columns there are 19,600 triples. A check skips itself when it would exceed the limit (100,000 by default).
  With `verbose=2`, the number of combinations each check needs is shown.
- **Skip the code / ID checks** with `include_code_tests=False` when no column holds codes or IDs. This also
  removes noise.
- **Drop columns** that are not worth checking before calling `init_data()`.
- **Sample the rows.** On a sample, patterns are still found reliably, although the exceptions found only cover
  the sampled rows.
- **Watch progress** with `verbose=1` or `2`, to decide whether to wait or narrow the run.
