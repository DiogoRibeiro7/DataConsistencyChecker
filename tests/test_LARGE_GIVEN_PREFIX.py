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

test_id = "LARGE_GIVEN_PREFIX"

synth_patterns_cols = []  # None
synth_exceptions_cols = [
    '"large_given_prefix rand" AND "large_given_prefix most"',
    '"large_given_prefix rand" AND "large_given_prefix date_most"',
]


@requires_real_data
def test_real():
    res = build_default_results()
    res["musk"] = (
        [],
        [
            '"molecule_name" AND "f36"',
            '"molecule_name" AND "f47"',
        ],
    )
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
    synth_test(test_id, "random", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_80_percent_nulls():
    # With 80% of the rows null, the upper limit of "large_given_prefix most" as a whole, computed on the remaining
    # values, falls below its value of 290 in row 999, and this check does not flag values that are large for the
    # whole column.
    synth_test(test_id, "80-percent", synth_patterns_cols, [synth_exceptions_cols[1]])


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
    synth_test_all_cols(test_id, "random", 0, 0)


@requires_synthetic_all_columns
def test_synthetic_all_cols_80_percent_nulls():
    synth_test_all_cols(test_id, "80-percent", 0, 0)
