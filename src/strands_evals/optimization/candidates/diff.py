"""Differences between two candidates."""

from __future__ import annotations

from ...types.optimization import Value
from ..declaration.space import Space
from .candidate import Candidate


def diff(space: Space, base: Candidate, other: Candidate) -> str:
    """A valid unified diff of text and file-backed values, on the real file paths.

    Args:
        space: Maps file-backed keys to their real paths.
        base: The candidate diffed against, usually the baseline.
        other: The candidate shown.
    """
    raise NotImplementedError


def changed_values(space: Space, base: Candidate, other: Candidate) -> dict[str, tuple[Value, Value]]:
    """Changed numbers and choices as `key -> (before, after)`.

    Args:
        space: Tells numbers and choices apart from text.
        base: The candidate compared against, usually the baseline.
        other: The candidate shown.

    Returns:
        Key -> (base value, other value) for every changed number or choice.
    """
    raise NotImplementedError


def changed_keys(base: Candidate, other: Candidate) -> list[str]:
    """Every key whose value differs, including keys present in only one of them.

    Args:
        base: The candidate compared against, e.g. the parent.
        other: The candidate shown.
    """
    raise NotImplementedError
