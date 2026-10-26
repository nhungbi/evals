"""A candidate: one typed point of the space, its content id, and its lineage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ...types.optimization import Value
from ..declaration.space import Space


def content_id(values: Mapping[str, Value]) -> str:
    """First 12 hex characters of the sha256 of the canonical JSON (sorted keys) of `values`."""
    raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Candidate:
    """Typed flat values plus lineage.

    `values` is stored as a read-only copy (`MappingProxyType`) and `id` is computed from it in
    `__post_init__`, so the content id can't go stale. Equality and hashing use `id`.
    `origin` is "baseline", the proposing block's name (`Block.name`), or "merge".
    """

    values: Mapping[str, Value]
    parent_ids: tuple[str, ...]
    origin: str
    rationale: str
    id: str = field(init=False)

    def __post_init__(self) -> None:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    @classmethod
    def baseline(cls, space: Space) -> Candidate:
        """Every value at its default (`space.complete({})`)."""
        raise NotImplementedError

    def child(self, space: Space, updates: Mapping[str, Value], *, origin: str, rationale: str) -> Candidate:
        """Copy these values, apply `updates`, complete, and validate; `origin` is the block name.

        Args:
            space: Completes and validates the new values.
            updates: The keys to change, from `Patch.updates`.
            origin: The proposing block's name.
            rationale: Why, from the patch.

        Raises:
            ValueError: The completed values fail `space.validate`.
        """
        raise NotImplementedError

    @classmethod
    def merged(
        cls,
        space: Space,
        values: Mapping[str, Value],
        parents: tuple[str, str],
        rationale: str,
    ) -> Candidate:
        """A merge of two frontier members.

        Args:
            space: Validates the merged values.
            values: The merged values, from `MergePlan.values`.
            parents: The two parent ids.
            rationale: Which variables came from which parent.

        Raises:
            ValueError: The merged values fail `space.validate`.
        """
        raise NotImplementedError
