from __future__ import annotations

# Mixins
from .display_mixin import DisplayMixin
from .plots_mixin import PlotsMixin
from .synth_data_mixin import SynthDataMixin
from .results_mixin import ResultsMixin
from .analysis_cache_mixin import AnalysisCacheMixin
from .data_init_mixin import DataInitMixin
from .execution_mixin import ExecutionMixin
from .export_mixin import ExportMixin

# Test implementation mixins
from .test_implementations import (
    BaseTestsMixin,
    NumericTestsMixin,
    DateTestsMixin,
    StringTestsMixin,
    BinaryTestsMixin,
    MultiColumnTestsMixin,
)


class DataConsistencyChecker(BaseTestsMixin, NumericTestsMixin, DateTestsMixin, StringTestsMixin, BinaryTestsMixin, MultiColumnTestsMixin, DisplayMixin, PlotsMixin, SynthDataMixin, ResultsMixin, AnalysisCacheMixin, DataInitMixin, ExecutionMixin, ExportMixin):
    """
    Automated data quality checker performing 164 tests to identify patterns and anomalies.

    This class examines tabular datasets for consistency patterns across single columns,
    pairs of columns, and larger column sets. It identifies both patterns and exceptions
    to those patterns, useful for EDA and interpretable outlier detection.
    """

    ##################################################################################################################
    # Tune the contamination rate
    ##################################################################################################################
    def test_contamination_level(self, contamination_levels_arr=None):
        """
        This may be used to help determine an appropriate contamination rate to set for the process. Patterns in the
        date will only be recognized by the tool as patterns if there are less than the specified contamination rate
        of exceptions. Exceptions to patterns can only be found if the patterns are first recognized.
        Setting the contamination rate to a small value will identify only strong exceptions, but
        may miss some interesting patterns in the data. Setting a higher value will expose more patterns, but will
        also generate some noise.

        This presents a set of 3 plots: the number of issues found, the number of rows flagged at least once, and the
        number of columns flagged at least once, based on the contamination rate. It also returns these counts as
        three arrays.
        """
        if contamination_levels_arr is None:
            contamination_levels_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        pass
