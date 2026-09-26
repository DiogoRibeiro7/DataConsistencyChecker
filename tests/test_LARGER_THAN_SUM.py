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

test_id = "LARGER_THAN_SUM"

synth_patterns_cols = ['"larger_sum rand_a" AND "larger_sum rand_b" AND "larger_sum all"']
synth_exceptions_cols = ['"larger_sum rand_a" AND "larger_sum rand_b" AND "larger_sum most"']


# This test can occasionally vary in the numbers of exceptions found, as it works on
# a sample, and in some cases, the values are only slightly above or below the sum.
@requires_real_data
def test_real():
    res = build_default_results()
    res["abalone"] = (
        [],
        [
            '"Shucked_weight" AND "Viscera_weight" AND "Whole_weight"',
            '"Shucked_weight" AND "Shell_weight" AND "Whole_weight"',
            '"Viscera_weight" AND "Shell_weight" AND "Whole_weight"',
        ],
    )
    res["vehicle"] = (36, 8)
    res["analcatdata_authorship"] = (3, 15)
    res["SpeedDating"] = (471, -400)
    res["segment"] = (2, 0)
    res["jm1"] = ([], ['"lOCode" AND "locCodeAndComment" AND "loc"', '"lOComment" AND "locCodeAndComment" AND "loc"'])
    res["adult"] = (1, ['"capitalgain" AND "capitalloss" AND "education-num"'])  # this one is weak
    res["anneal"] = (3, [])
    res["qsar-biodeg"] = (202, 345)
    res["wdbc"] = (1, ['"V5" AND "V10" AND "V29"'])
    res["ozone-level-8hr"] = (7, [])
    res["kc2"] = ([], 3)
    res["spambase"] = (366, -1680)
    res["baseball"] = (2, 2)
    res["mc1"] = (11, 5)
    res["pc1"] = ([], 13)
    res["cardiotocography"] = (103, 33)
    res["steel-plates-fault"] = (0, 2)
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
