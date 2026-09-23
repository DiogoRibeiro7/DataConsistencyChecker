"""Typed internal registry definitions for DataConsistencyChecker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


LegacyTestTuple = tuple[
    str,
    str,
    Callable[..., Any],
    Callable[..., Any] | None,
    bool,
    bool,
    bool,
    bool,
]


@dataclass(frozen=True)
class TestDefinition:
    """Internal executable definition for one consistency check."""

    short_description: str
    description: str
    test_func: Callable[..., Any]
    gen_func: Callable[..., Any] | None
    shortlist: bool
    implemented: bool
    fast: bool
    code: bool

    @classmethod
    def from_legacy_tuple(cls, value: LegacyTestTuple) -> "TestDefinition":
        """Convert a legacy 8-element tuple into named fields."""
        return cls(
            short_description=value[0],
            description=value[1],
            test_func=value[2],
            gen_func=value[3],
            shortlist=value[4],
            implemented=value[5],
            fast=value[6],
            code=value[7],
        )


def normalize_test_definitions(
    definitions: Mapping[str, LegacyTestTuple | TestDefinition],
) -> dict[str, TestDefinition]:
    """Normalize mixed legacy/typed definitions at the registry boundary."""
    return {
        test_id: (
            definition
            if isinstance(definition, TestDefinition)
            else TestDefinition.from_legacy_tuple(definition)
        )
        for test_id, definition in definitions.items()
    }


__all__ = ["LegacyTestTuple", "TestDefinition", "normalize_test_definitions"]
