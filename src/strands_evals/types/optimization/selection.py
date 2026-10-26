"""What selection strategies return."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .search import Value


class MergePlan(BaseModel):
    """A variable-wise merge of two frontier members since their latest common ancestor."""

    model_config = ConfigDict(frozen=True)

    first: str
    second: str
    ancestor: str
    values: dict[str, Value]
    from_first: tuple[str, ...]
    from_second: tuple[str, ...]


class SelectionStep(BaseModel):
    """The parent to propose on, or a merge to score instead of calling a method."""

    model_config = ConfigDict(frozen=True)

    parent_id: str
    merge: MergePlan | None = None
