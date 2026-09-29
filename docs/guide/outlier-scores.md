# Outlier scores

Each pattern with exceptions flags a set of rows. A row's score is the number of these findings that flag it.

```python
dc.get_outlier_scores()                 # raw scores: one integer per row
dc.get_outlier_scores(normalized=True)  # scores divided by the number of active findings with exceptions
dc.get_outlier_score_summary()          # both, as a DataFrame with FINAL SCORE and NORMALIZED SCORE
```

## Raw scores

The raw score (`FINAL SCORE`) is a plain count: each pattern with exceptions that flags a row adds one point.
One check can contribute several points if it finds exceptions on different columns or column combinations.
`get_results_by_row_id(row)` lists the check and columns behind each point.

A score of 0 means no active finding flags the row. It does not establish that the row is correct: the result
depends on the checks selected, their thresholds and whether they completed successfully.

## Normalized scores

The normalized score divides each raw score by the number of active patterns with exceptions:

$$
s_i^{\mathrm{normalized}} =
\begin{cases}
\dfrac{s_i}{M}, & M > 0, \\
0, & M = 0.
\end{cases}
$$

Here $s_i$ is the raw score of row $i$, and $M$ is the number of active patterns with exceptions. The denominator
counts findings, not checks executed or rows flagged. When no patterns have exceptions, every normalized score
is zero.

It is always between 0 and 1. A value of 0.5 means the row was flagged by half of these findings. Normalization
provides a common range, but scores from different datasets or check selections can still reflect very
different patterns.

!!! warning "Not a probability"
    The normalized score is a relative frequency. It is not a calibrated probability, a p-value or a measure of
    statistical significance.

## Example: the order table

In the [quickstart](../getting-started/quickstart.md), five findings flag row 42 and no other row:

```python
dc.get_outlier_score_summary().iloc[[42]]
```

| Row | FINAL SCORE | NORMALIZED SCORE |
|---|---|---|
| 42 | 5 | 1.0 |

The normalized score is 1.0 because all five findings flag that row. The other 499 rows have raw and normalized
scores of zero. Row numbers are zero-based positions in the DataFrame passed to `init_data()`; see
[Reviewing results](reviewing-results.md#rows) for matching them to your input.

## Scores after curating findings

[`clear_results()`](curating-results.md) recalculates raw, normalized and per-cell scores from the remaining
findings. Removing a finding can reduce both a row's raw score and the normalization denominator, so its
normalized score may increase, decrease or stay the same. Clearing every exception sets all scores to zero;
clearing only patterns without exceptions leaves scores unchanged.

`restore_results()` restores the findings and scores saved after the last `check_data_quality()` call. Reports
already returned by `get_report()` remain snapshots; create a new report to include the curated scores.

## Where the scores come from

- `get_exceptions()` returns, for every row, which exceptions flag it, together with the final score.
- `get_exceptions_by_column()` spreads each exception's point evenly over the columns involved, giving a score
  per cell: this shows *where* in a row the problems are.
- `plot_final_scores_distribution_by_row()`, `..._by_feature()` and `..._by_test()` plot the scores, and
  `plot_columns_vs_final_scores()` looks for relationships between column values and scores.
