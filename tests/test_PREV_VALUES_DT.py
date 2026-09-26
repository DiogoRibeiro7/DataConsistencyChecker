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

test_id = "PREV_VALUES_DT"

synth_patterns_cols = ["pattern_history_df_str all_a", "pattern_history_df_num all_a"]
synth_exceptions_cols = [
    "pattern_history_df_str most_a",
    "pattern_history_df_str most_b",
    "pattern_history_df_num most_a",
]


@requires_real_data
def test_real():
    res = build_default_results()
    res["nursery"] = (
        ["finance", "social", "health"],  # the values cycle in this column
        [],
    )
    res["car-evaluation"] = (
        [
            "luggage_boot_size_small",
            "luggage_boot_size_med",
            "luggage_boot_size_big",
            "safety_low",
            "safety_med",
            "safety_high",
        ],
        [],
    )
    res["car"] = (["persons", "lug_boot", "safety"], [])
    real_test(test_id, res)


@requires_synthetic
def test_synthetic_no_nulls():
    synth_test(test_id, "none", synth_patterns_cols, synth_exceptions_cols)


@requires_synthetic_nones
def test_synthetic_one_row_nulls():
    synth_test(test_id, "one-row", synth_patterns_cols, synth_exceptions_cols)


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
