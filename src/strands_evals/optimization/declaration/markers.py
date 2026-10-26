"""Markers for plain values and components: what may change about an agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# `default=` not given: `tunable` fills in the wrapped parameter's own default.
_UNSET: Any = object()


class Int:
    """An integer argument with inclusive bounds; the default must lie in `[ge, le]`.

    Without `default=`, the wrapped function's parameter default is used. With neither (always the
    case under `build_task`'s `space=`, which has no function), declaration raises `TypeError`.
    """

    def __init__(self, ge: int, le: int, *, default: int = _UNSET, log: bool = False):
        raise NotImplementedError


class Float:
    """A float argument with inclusive bounds; the default must lie in `[ge, le]`.

    Without `default=`, the wrapped function's parameter default is used; with neither, declaration
    raises `TypeError`.
    """

    def __init__(self, ge: float, le: float, *, default: float = _UNSET, log: bool = False):
        raise NotImplementedError


class OneOf:
    """A choice among plain values.

    The default is `default=` if given, else the wrapped function's parameter default, else the first
    option. A default that isn't one of the options raises `TypeError` at declaration.
    """

    def __init__(self, *options: str | int | float | bool | None, default: Any = _UNSET):
        raise NotImplementedError


class TextValue:
    """A text knob; `size_cap` rejects prompt bloat (default: `_default_size_cap(default)`).

    Without `default=`, the wrapped function's parameter default is used; with neither, declaration
    raises `TypeError`.
    """

    def __init__(self, *, default: str = _UNSET, size_cap: int | None = None, edit_instructions: str | None = None):
        raise NotImplementedError


class Component:
    """One constructible alternative, returned by `component()`."""


class Choice:
    """A swap between components, returned by `choice()`; the first alternative is the baseline."""


def component(factory: Callable[..., Any], /, **kwargs: Any) -> Component:
    """Declare one constructible alternative; `Int`/`Float` kwargs are tunable, others pass through.

    `factory` is positional-only, so a constructor kwarg named `factory` can still be passed.
    """
    raise NotImplementedError


def choice(*alternatives: Component, labels: Sequence[str] | None = None) -> Choice:
    """Declare a swap between components.

    Labels default to class names. The same class twice without labels is a `TypeError` at
    declaration. Labels must be unique and contain no dots (they become key segments).
    """
    raise NotImplementedError


def _default_size_cap(baseline: str) -> int:
    """`max(2000, 3 * len(baseline))`; shared with `files.py`."""
    raise NotImplementedError
