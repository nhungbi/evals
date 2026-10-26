"""The declared space: flat typed keys, which are active, validation, and resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from typing_extensions import TypeAlias

from ...types.optimization import Value
from .files import FileText, JsonField, SkillLibrary
from .markers import Choice, Component, Float, Int, OneOf, TextValue
from .tools import ToolDescriptions

Marker: TypeAlias = (
    Int | Float | OneOf | TextValue | Component | Choice | FileText | JsonField | SkillLibrary | ToolDescriptions
)

ParamKind = Literal["int", "float", "one_of", "choice", "text"]


@dataclass(frozen=True)
class FileSource:
    """Where a file-backed text value lives."""

    path: Path
    pointer: str | None
    as_path: bool


@dataclass(frozen=True)
class ParamInfo:
    """One flat key of the space; `active_when` is `(key, value)` of the choice that enables it.

    `ge` / `le` are the inclusive bounds of an `Int` / `Float`, named as on the markers.
    """

    key: str
    name: str
    kind: ParamKind
    default: Value
    ge: float | None
    le: float | None
    log: bool
    options: tuple[Value, ...] | None
    size_cap: int | None
    edit_instructions: str | None
    active_when: tuple[str, str] | None
    source: FileSource | None


class Space:
    """Flattens markers into dotted keys and knows which keys are active under which values."""

    def __init__(self, markers: Mapping[str, Marker]):
        """Flatten the markers and probe every component with its defaults; raises `TypeError`."""
        raise NotImplementedError

    def params(self) -> list[ParamInfo]:
        raise NotImplementedError

    def param(self, key: str) -> ParamInfo:
        raise NotImplementedError

    def is_active(self, key: str, values: Mapping[str, Value]) -> bool:
        """Whether a key is active.

        Args:
            key: A flat key.
            values: The candidate's values.

        Returns:
            Whether `key` is active under the alternatives chosen in `values`.
        """
        raise NotImplementedError

    def complete(self, values: Mapping[str, Value]) -> dict[str, Value]:
        """Drop keys inactive under the chosen alternatives and fill missing active keys with defaults."""
        raise NotImplementedError

    def validate(self, values: Mapping[str, Value]) -> None:
        """Check bounds, size caps, formats, active keys, and that nothing is missing; raises `ValueError`."""
        raise NotImplementedError

    def resolve(self, name: str, values: Mapping[str, Value], workspace: Path | None) -> Any:
        """What the factory receives for one declared name: a fresh component, a value, or a path.

        Args:
            name: A declared name, not a flat key.
            values: The candidate's values.
            workspace: The candidate's copy, or None when no value reaches the agent as a path.
        """
        raise NotImplementedError
