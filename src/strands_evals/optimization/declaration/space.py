"""The declared space: flat typed keys, which are active, validation, and resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import TypeAlias

from ...types.optimization import Value
from .files import FileText, JsonField, SkillLibrary
from .markers import _UNSET, Choice, Component, Float, Int, OneOf, TextValue, _default_size_cap, _is_option, choice
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
    ge: float | None = None
    le: float | None = None
    log: bool = False
    options: tuple[Value, ...] | None = None
    size_cap: int | None = None
    edit_instructions: str | None = None
    active_when: tuple[str, str] | None = None
    source: FileSource | None = None


class Space:
    """Flattens markers into dotted keys and knows which keys are active under which values."""

    def __init__(self, markers: Mapping[str, Marker]):
        """Flatten the markers into typed keys; raises `TypeError`.

        Components were already probed by `component()`. A bare `component(...)` is a choice with one
        alternative. Every default must be a valid value of its own key, so the baseline always passes
        `validate`.

        Raises:
            TypeError: No markers, a name that can't be a key, an unknown marker type, a marker
                with no default, or a default that fails `validate` (e.g. text over its `size_cap`).
        """
        if not markers:
            raise TypeError("nothing to optimize: declare at least one marker")
        self._markers: dict[str, Marker] = {}
        self._params: dict[str, ParamInfo] = {}
        for name, marker in markers.items():
            if not isinstance(name, str) or not name or "." in name or "/" in name:
                raise TypeError(f"declared name {name!r} must be a non-empty string without '.' or '/'")
            if isinstance(marker, Component):
                marker = choice(marker)
            if not isinstance(marker, Marker):
                raise TypeError(f"{name}: {type(marker).__name__} is not a marker")
            self._markers[name] = marker
            for info in _flatten(name, marker):
                problem = _check_value(info, info.default)
                if problem is not None:
                    raise TypeError(f"invalid default: {problem}")
                self._params[info.key] = info

    def params(self) -> list[ParamInfo]:
        """Every key, including keys under alternatives that aren't the default, in declaration order."""
        return list(self._params.values())

    def param(self, key: str) -> ParamInfo:
        """The `ParamInfo` of one flat key; raises `KeyError` naming the known keys."""
        if key not in self._params:
            raise KeyError(f"{key!r} is not a key of this space; keys: {sorted(self._params)}")
        return self._params[key]

    def is_active(self, key: str, values: Mapping[str, Value]) -> bool:
        """Whether a key is active.

        A choice missing from `values` counts as its default. Conditions chain: a key under an
        alternative is active only if every enclosing choice selects it.

        Args:
            key: A flat key.
            values: The candidate's values.

        Returns:
            Whether `key` is active under the alternatives chosen in `values`.
        """
        info = self.param(key)
        while info.active_when is not None:
            condition_key, label = info.active_when
            condition = self._params[condition_key]
            if values.get(condition_key, condition.default) != label:
                return False
            info = condition
        return True

    def complete(self, values: Mapping[str, Value]) -> dict[str, Value]:
        """Drop keys inactive under the chosen alternatives and fill missing active keys with defaults.

        Keys the space doesn't know are kept, so `validate` reports them instead of a typo vanishing.
        """
        completed: dict[str, Value] = {}
        for key, info in self._params.items():
            # A choice precedes its sub-keys, so `completed` already holds the choices they depend on.
            if self.is_active(key, {**completed, **values}):
                completed[key] = values[key] if key in values else info.default
        completed.update({key: value for key, value in values.items() if key not in self._params})
        return completed

    def validate(self, values: Mapping[str, Value]) -> None:
        """Check types, bounds, size caps, active keys, and that nothing is missing; raises `ValueError`.

        Every problem is reported in one message.
        """
        problems: list[str] = []
        unknown = sorted(key for key in values if key not in self._params)
        if unknown:
            problems.append(f"not keys of this space: {unknown}")
        known = [key for key in values if key in self._params]
        inactive = sorted(key for key in known if not self.is_active(key, values))
        if inactive:
            problems.append(f"inactive under the chosen alternatives: {inactive}")
        missing = sorted(key for key in self._params if key not in values and self.is_active(key, values))
        if missing:
            problems.append(f"missing: {missing}")
        for key in known:
            problem = _check_value(self._params[key], values[key])
            if problem is not None:
                problems.append(problem)
        if problems:
            raise ValueError("invalid values: " + "; ".join(problems))

    def resolve(self, name: str, values: Mapping[str, Value], workspace: Path | None) -> Any:
        """What the factory receives for one declared name: a fresh component, a value, or a path.

        Components and tool copies are built fresh on every call; a component's fixed arguments are
        passed as is, so those objects are shared (see `component()`). `values` must already pass
        `validate` (so no key is missing); nothing is checked here.

        Args:
            name: A declared name, not a flat key.
            values: The candidate's values, already validated.
            workspace: The candidate's copy, or None when no value reaches the agent as a path.
        """
        if name not in self._markers:
            raise KeyError(f"{name!r} is not a declared name; declared: {list(self._markers)}")
        marker = self._markers[name]
        if isinstance(marker, Choice):
            label = cast(str, values[name])
            alternative = marker.alternatives[label]
            return alternative.build({arg: values[f"{name}.{label}.{arg}"] for arg in alternative.tunable})
        if isinstance(marker, ToolDescriptions):
            return marker.build({tool: cast(str, values[f"{name}.{tool}"]) for tool in marker.tools})
        return values[name]


def _flatten(name: str, marker: Marker) -> list[ParamInfo]:
    """The typed keys one declared marker expands into, a choice before its sub-keys."""
    if isinstance(marker, (Int, Float)):
        return [_number(name, name, marker, None)]
    if isinstance(marker, OneOf):
        default = marker.options[0] if marker.default is _UNSET else marker.default
        return [ParamInfo(name, name, "one_of", default, options=tuple(marker.options))]
    if isinstance(marker, TextValue):
        if marker.default is None:
            raise TypeError(f"{name}: TextValue has no default; {_NO_DEFAULT}")
        return [_text(name, name, marker.default, marker.size_cap, marker.edit_instructions)]
    if isinstance(marker, Choice):
        labels = tuple(marker.alternatives)
        infos = [ParamInfo(name, name, "choice", labels[0], options=labels)]
        for label, alternative in marker.alternatives.items():
            for arg, bound in alternative.tunable.items():
                infos.append(_number(f"{name}.{label}.{arg}", name, bound, (name, label)))
        return infos
    if isinstance(marker, ToolDescriptions):
        return [
            _text(f"{name}.{tool_name}", name, tool.tool_spec["description"], marker.size_cap, marker.edit_instructions)
            for tool_name, tool in marker.tools.items()
        ]
    raise TypeError(f"{name}: {type(marker).__name__} can't be declared yet")


_NO_DEFAULT = "pass default=, or under tunable give the wrapped parameter a default"


def _number(key: str, name: str, marker: Int | Float, active_when: tuple[str, str] | None) -> ParamInfo:
    if marker.default is None:
        raise TypeError(f"{key}: {type(marker).__name__} has no default; {_NO_DEFAULT}")
    kind: ParamKind = "int" if isinstance(marker, Int) else "float"
    return ParamInfo(
        key, name, kind, marker.default, ge=marker.ge, le=marker.le, log=marker.log, active_when=active_when
    )


def _text(key: str, name: str, default: str, size_cap: int | None, edit_instructions: str | None) -> ParamInfo:
    cap = _default_size_cap(default) if size_cap is None else size_cap
    return ParamInfo(key, name, "text", default, size_cap=cap, edit_instructions=edit_instructions)


def _check_value(info: ParamInfo, value: Any) -> str | None:
    """A one-line problem with one value, or None; text values are described by length only."""
    key = info.key
    if info.kind in ("int", "float"):
        expected = int if info.kind == "int" else float
        if type(value) is not expected:
            return f"{key}: expected {expected.__name__}, got {value!r}"
        if not (info.ge is not None and info.le is not None and info.ge <= value <= info.le):
            return f"{key}: {value!r} is outside [{info.ge!r}, {info.le!r}]"
        return None
    if info.kind in ("one_of", "choice"):
        if not _is_option(value, info.options or ()):
            return f"{key}: {value!r} is not one of {list(info.options or ())!r}"
        return None
    if not isinstance(value, str):
        return f"{key}: expected str, got {type(value).__name__}"
    if info.size_cap is not None and len(value) > info.size_cap:
        return f"{key}: {len(value)} characters, over the size_cap of {info.size_cap}"
    return None
