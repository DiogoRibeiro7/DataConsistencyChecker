# Outlier scores

Each exception flags a set of rows. A row's score is the number of exceptions that flag it.

```python
dc.get_outlier_scores()                 # raw scores: one integer per row
dc.get_outlier_scores(normalized=True)  # scores divided by the number of exceptions in the run
dc.get_outlier_score_summary()          # both, as a DataFrame with FINAL SCORE and NORMALIZED SCORE
```

## Raw scores

The raw score (`FINAL SCORE`) is a plain count: each pattern with exceptions that flags a row adds one point.
It is completely transparent: `get_results_by_row_id(row)` lists the checks and columns behind a row's score.

A row with a score of 0 was not flagged by any check, and can reasonably be considered typical of the dataset.

## Normalized scores

Raw scores depend on how many exceptions a run found, so they are hard to compare between runs or datasets. The
normalized score divides each raw score by the number of exception-bearing patterns in the run:

```text
normalized score = raw score / number of patterns with exceptions in the run
```

It is always between 0 and 1. A value of 0.5 means the row was flagged by half of the exceptions found.

!!! warning "Not a probability"
    The normalized score is a relative frequency. It is not a calibrated probability, a p-value or a measure of
    statistical significance.

## Where the scores come from

- `get_exceptions()` returns, for every row, which exceptions flag it, together with the final score.
- `get_exceptions_by_column()` spreads each exception's point evenly over the columns involved, giving a score
  per cell: this shows *where* in a row the problems are.
- `plot_final_scores_distribution_by_row()`, `..._by_feature()` and `..._by_test()` plot the scores, and
  `plot_columns_vs_final_scores()` looks for relationships between column values and scores.
