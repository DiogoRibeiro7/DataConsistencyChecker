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

test_id = "RARE_DECIMALS"

synth_patterns_cols = ["rare_decimals all"]
synth_exceptions_cols = ["rare_decimals most"]


@requires_real_data
def test_real():
    res = build_default_results()
    res["isolet"] = (["f581", "f582", "f583"], [])
    res["bioresponse"] = (11, 207)
    res["SpeedDating"] = ([], 14)
    res["segment"] = ([], ["short-line-density-2"])
    res["jm1"] = ([], 10)
    res["profb"] = (["Pointspread"], [])
    res["higgs"] = (4, [])
    res["anneal"] = (["width"], [])
    res["ozone-level-8hr"] = (5, [])
    res["kc2"] = ([], 10)
    res["nomao"] = (8, 5)
    res["hypothyroid"] = ([], ["TT4", "FTI"])
    res["pc1"] = ([], 10)
    res["wine-quality-white"] = (["V6"], [])
    res["steel-plates-fault"] = (["V21"], [])
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
