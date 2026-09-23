"""DataConsistencyChecker public package API."""

from importlib.metadata import PackageNotFoundError, version

from .api import analyze
from .checker import DataConsistencyChecker
from .config import DataConsistencyConfig
from .report import DataConsistencyReport
from .test_metadata import TestMetadata

__all__ = ["DataConsistencyChecker", "DataConsistencyConfig", "DataConsistencyReport", "TestMetadata", "analyze"]

try:
    __version__ = version("data-consistency-checker")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
