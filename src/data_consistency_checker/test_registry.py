"""Typed internal registry definitions for DataConsistencyChecker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TestDefinition:
    """Internal executable definition for one consistency check."""

    __test__ = False  # Not a pytest test class, despite the name

    short_description: str
    description: str
    test_func: Callable[..., Any]
    gen_func: Callable[..., Any] | None
    shortlist: bool
    implemented: bool
    fast: bool
    code: bool


__all__ = ["TestDefinition"]
