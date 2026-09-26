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

test_id = "SIMILAR_TO_RATIO"

synth_patterns_cols = ['"similar_to_ratio rand_a" AND "similar_to_ratio rand_b" AND "similar_to_ratio all"']
synth_exceptions_cols = ['"similar_to_ratio rand_a" AND "similar_to_ratio rand_b" AND "similar_to_ratio most"']


@requires_real_data
def test_real():
    res = build_default_results()
    res["page-blocks"] = (
        [
            '"lenght" AND "height" AND "eccen"',
            '"area" AND "height" AND "lenght"',
            '"blackpix" AND "area" AND "p_black"',
            '"blackpix" AND "mean_tr" AND "wb_trans"',
            '"blackand" AND "area" AND "p_and"',
        ],
        [],
    )
    res["hypothyroid"] = ([], ['"TT4" AND "T4U" AND "FTI"'])
    res["baseball"] = (['"Hits" AND "At_bats" AND "Batting_average"'], 0)
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
