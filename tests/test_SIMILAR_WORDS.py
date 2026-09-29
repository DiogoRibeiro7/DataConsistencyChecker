from utils import (
    build_default_results,
    real_test,
    requires_real_data,
    requires_synthetic,
    requires_synthetic_all_columns,
    requires_synthetic_nones,
    synth_test,
    synth_test_all_cols,
)

test_id = "SIMILAR_WORDS"

synth_patterns_cols = ['"sim_words rand" AND "sim_words all"']
synth_exceptions_cols = ['"sim_words rand" AND "sim_words most"']

# With random or 80-percent nulls, only a quarter or a fifth of the rows have both values. Few of the rows whose word
# sets overlap less (a Jaccard similarity of 0.75) remain, fewer than the contamination level, so they are flagged
# as exceptions instead of the pattern being found with a lower similarity threshold.
synth_exceptions_cols_many_nulls = ['"sim_words rand" AND "sim_words all"'] + synth_exceptions_cols


@requires_real_data
def test_real():
    res = build_default_results()
    real_test(test_id, res)


@requires_synthetic
def test_synthetic_no_nulls():
    synth_test(test_id, "none", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_one_row_nulls():
    synth_test(test_id, "one-row", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_in_sync_nulls():
    synth_test(test_id, "in-sync", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_random_nulls():
    synth_test(test_id, "random", [], synth_exceptions_cols_many_nulls)


@requires_synthetic_nones
def test_synthetic_80_percent_nulls():
    synth_test(test_id, "80-percent", [], synth_exceptions_cols_many_nulls)


@requires_synthetic_all_columns
def test_synthetic_all_cols_no_nulls():
    synth_test_all_cols(test_id, "none", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_all_columns
def test_synthetic_all_cols_one_row_nulls():
    synth_test_all_cols(test_id, "one-row", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_all_columns
def test_synthetic_all_cols_in_sync_nulls():
    synth_test_all_cols(test_id, "in-sync", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_all_columns
def test_synthetic_all_cols_random_nulls():
    synth_test_all_cols(test_id, "random", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_all_columns
def test_synthetic_all_cols_80_percent_nulls():
    synth_test_all_cols(test_id, "80-percent", synth_patterns_cols, synth_exceptions_cols)
