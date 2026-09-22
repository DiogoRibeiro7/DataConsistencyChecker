"""Backward-compatible import shim.

Prefer ``from data_consistency_checker import DataConsistencyChecker``.
This module remains temporarily so existing notebooks and user code continue
to work during the package-layout migration.
"""

from data_consistency_checker import DataConsistencyChecker

__all__ = ["DataConsistencyChecker"]
