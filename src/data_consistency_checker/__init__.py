"""DataConsistencyChecker public package API."""

from .api import analyze
from .checker import DataConsistencyChecker
from .config import DataConsistencyConfig
from .report import DataConsistencyReport

__all__ = ["DataConsistencyChecker", "DataConsistencyConfig", "DataConsistencyReport", "analyze"]

__version__ = "0.1.0"
