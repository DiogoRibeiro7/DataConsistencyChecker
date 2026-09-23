"""Typed metadata for DataConsistencyChecker test definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


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


def metadata_from_tuple(
    test_id: str,
    definition: tuple[
        str,
        str,
        Callable[..., Any],
        Callable[..., Any] | None,
        bool,
        bool,
        bool,
        bool,
    ],
) -> TestMetadata:
    """Convert an internal legacy tuple to typed public metadata."""
    return TestMetadata(
        test_id=test_id,
        short_description=definition[0],
        description=definition[1],
        shortlist=definition[4],
        implemented=definition[5],
        fast=definition[6],
        code=definition[7],
    )


__all__ = ["TestMetadata", "metadata_from_tuple"]
