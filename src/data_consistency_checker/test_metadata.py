"""Typed metadata for DataConsistencyChecker test definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .test_registry import TestDefinition


@dataclass(frozen=True)
class TestMetadata:
    """Public metadata describing one consistency check."""

    test_id: str
    short_description: str
    description: str
    shortlist: bool
    implemented: bool
    fast: bool
    code: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


def metadata_from_definition(
    test_id: str,
    definition: TestDefinition,
) -> TestMetadata:
    """Convert an internal executable definition to public metadata."""
    return TestMetadata(
        test_id=test_id,
        short_description=definition.short_description,
        description=definition.description,
        shortlist=definition.shortlist,
        implemented=definition.implemented,
        fast=definition.fast,
        code=definition.code,
    )


__all__ = ["TestMetadata", "metadata_from_definition"]
