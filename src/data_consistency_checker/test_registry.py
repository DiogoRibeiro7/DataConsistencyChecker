"""Typed internal registry definitions for DataConsistencyChecker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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


__all__ = ["TestDefinition"]
