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

test_id = "UNIQUE_VALUES"

synth_patterns_cols = ["unique_vals all"]
synth_exceptions_cols = ["unique_vals most"]


@requires_real_data
def test_real():
    res = build_default_results()
    res["climate-model-simulation-crashes"] = (["V3"], 0)
    res["steel-plates-fault"] = (0, 2)  # real, but just a case of many rows. However, ints may be ID columns
    real_test(test_id, res)


@requires_synthetic
def test_synthetic_no_nulls():
    synth_test(test_id, "none", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_one_row_nulls():
    synth_test(
        test_id,
        "one-row",
        [],  # This test does not cover columns with Nulls
        [],
    )


@requires_synthetic_nones
def test_synthetic_in_sync_nulls():
    synth_test(test_id, "in-sync", [], [])


@requires_synthetic_nones
def test_synthetic_random_nulls():
    synth_test(test_id, "random", [], [])


@requires_synthetic_nones
def test_synthetic_80_percent_nulls():
    synth_test(test_id, "80-percent", [], [])


@requires_synthetic_all_columns
def test_synthetic_all_cols_no_nulls():
    synth_test_all_cols(test_id, "none", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_all_columns
def test_synthetic_all_cols_one_row_nulls():
    synth_test_all_cols(test_id, "one-row", [], [])


@requires_synthetic_all_columns
def test_synthetic_all_cols_in_sync_nulls():
    synth_test_all_cols(test_id, "in-sync", [], [])


@requires_synthetic_all_columns
def test_synthetic_all_cols_random_nulls():
    synth_test_all_cols(test_id, "random", [], [])


@requires_synthetic_all_columns
def test_synthetic_all_cols_80_percent_nulls():
    synth_test_all_cols(test_id, "80-percent", [], [])
