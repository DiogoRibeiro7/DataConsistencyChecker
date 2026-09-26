# How it works

DataConsistencyChecker automates what a careful analyst does with a new dataset: look for regularities within
columns, between columns and across rows, and notice the values that do not fit.

## Checks, patterns and exceptions

The checker runs a set of [checks](../checks.md). Each one examines a single column, a pair of columns or a
larger set of columns, and each gives one of three outcomes for every column (or column set) it examines:

- **A pattern without exceptions** — the property holds for every row. For example, `total` is always the
  product of `price` and `quantity`.
- **A pattern with exceptions** — the property holds for almost every row. The few rows that break it are the
  *exceptions*, and each such finding is an *issue*.
- **No pattern** — the property does not hold often enough to be meaningful, and nothing is reported.

Take the check for the number of decimal digits in a numeric column. If every value has four decimal places,
that is a pattern without exceptions. If all but a handful have four and those few have eight, the handful are
exceptions: they may have been collected or processed differently. One such exception is rarely interesting on
its own. A row flagged by many checks, on different columns, almost certainly is.

Some checks have no notion of a pattern and only report exceptions, for example the checks for very large or
very small values.

## The contamination level

What separates "a pattern with exceptions" from "no pattern" is the **contamination level**: the maximum
number of rows that may break a pattern for it to still count as a pattern. It is set with
`freq_contamination_level` in `check_data_quality()`, either as a fraction of the rows (default `0.005`, that
is 0.5%) or as a row count.

With 10,000 rows and the default level, a check whose property holds in every row reports a pattern without
exceptions; if it fails in 1 to 50 rows, a pattern with those rows as exceptions; if it fails in more than 50,
nothing. Tuning this one parameter is usually all the tuning needed: raise it to see more patterns (and more
noise), lower it to keep only strong ones.

## The short list

Some patterns are interesting in themselves, even without exceptions, such as one column being the sum of
others. Others hold in most datasets and only matter when they have exceptions, such as a column being all
positive. Only checks in the **short list** report their patterns without exceptions by default; pass
`show_short_list_only=False` to see the rest. Exceptions are always reported, whatever the check. The
[checks catalog](../checks.md) shows which checks are in the short list.

## Outlier scores

Every exception flags some rows. A row's **outlier score** is simply the number of exceptions that flag it, so
the score is transparent: `get_results_by_row_id()` lists exactly which checks flagged a row and why. See
[Outlier scores](outlier-scores.md).

## Why run many checks together

- **Subtle outliers surface.** A row that deviates slightly in many ways is flagged many times, even when no
  single deviation stands out.
- **Shared work is amortised.** Many checks reuse the same statistics (percentiles, missing-value masks, column
  pairs), which are computed once per dataset.
- **Unflagged rows provide context.** Compare flagged rows with rows that follow the discovered patterns.
  A score of 0 means no active finding flags a row; it does not rule out problems the selected checks cannot
  detect.

## Limitations

- **False positives.** Some checks examine many column combinations, so some patterns will be coincidental or
  compare columns that are not really comparable. Every finding is explained, so these are easy to spot, and
  [`clear_results()`](curating-results.md) removes them.
- **Being unusual is not being wrong.** Frequently flagged rows are different from the rest of the dataset; they
  are not necessarily errors.
- **No multivariate distances.** By design, the checks are simple and interpretable. Detectors such as
  Isolation Forest or Local Outlier Factor find other kinds of outliers and are worth running alongside.
- **Thresholds are fixed.** Most checks have internal thresholds chosen to work well across many datasets.
  Different choices would find somewhat different patterns, so treat the output as the patterns found, not
  as the only patterns present.
