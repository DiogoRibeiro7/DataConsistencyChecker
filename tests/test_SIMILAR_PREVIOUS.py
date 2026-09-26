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

test_id = "SIMILAR_PREVIOUS"

synth_patterns_cols = ["sim_prev all"]
synth_exceptions_cols = ["sim_prev most", "sim_prev date_most"]


@requires_real_data
def test_real():
    res = build_default_results()
    res["analcatdata_authorship"] = (0, ["BookID"])
    res["SpeedDating"] = (["wave"], 0)
    res["eucalyptus"] = (0, ["Year"])
    res["har"] = (0, 66)
    res["JapaneseVowels"] = (0, ["utterance"])
    res["profb"] = (0, ["Week"])
    res["eeg-eye-state"] = (0, 10)
    res["climate-model-simulation-crashes"] = (0, ["V2"])
    res["mozilla4"] = (0, ["id"])
    res["electricity"] = (["date", "nswdemand"], ["day"])
    res["bank-marketing"] = (0, ["V10"])
    res["kr-vs-k"] = (0, ["V3"])
    res["volcanoes-a1"] = (0, ["V2"])
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
    synth_test(test_id, "80-percent", synth_patterns_cols, synth_exceptions_cols)


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
