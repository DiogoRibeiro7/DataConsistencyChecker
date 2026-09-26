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

test_id = "SMALL_GIVEN_PAIR"

synth_patterns_cols = []
synth_exceptions_cols = [
    '"small_given_pair rand_a" AND "small_given_pair rand_b" AND "small_given_pair most"',
    '"small_given_pair rand_a" AND "small_given_pair rand_b" AND "small_given_pair date_most"',
]


@requires_real_data
def test_real():
    res = build_default_results()
    res["SpeedDating"] = (0, 10497)  # we need to address this, as it's flagging way too many
    res["eucalyptus"] = (0, 4)
    res["adult"] = (0, 16)
    res["bank-marketing"] = (0, 16)
    real_test(test_id, res)


@requires_synthetic
def test_synthetic_no_nulls():
    synth_test(test_id, "none", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_one_row_nulls():
    synth_test(test_id, "one-row", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_in_sync_nulls():
    # With this many nulls, each pair of values in the two string columns has fewer than the 200 rows the check
    # needs, so it does not execute (as its docstring notes).
    synth_test(test_id, "in-sync", synth_patterns_cols, [])


@requires_synthetic_nones
def test_synthetic_random_nulls():
    # With this many nulls, each pair of values in the two string columns has fewer than the 200 rows the check
    # needs, so it does not execute (as its docstring notes).
    synth_test(test_id, "random", synth_patterns_cols, [])


@requires_synthetic_nones
def test_synthetic_80_percent_nulls():
    # With this many nulls, each pair of values in the two string columns has fewer than the 200 rows the check
    # needs, so it does not execute (as its docstring notes).
    synth_test(test_id, "80-percent", synth_patterns_cols, [])


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
