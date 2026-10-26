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
        """Declare a tunable integer.

        Args:
            ge: Smallest allowed value.
            le: Largest allowed value.
            default: The baseline value; omit to use the wrapped parameter's default.
            log: Search on a log scale.

        Raises:
            TypeError: No default anywhere, or one outside `[ge, le]`.
        """
        raise NotImplementedError


class Float:
    """A float argument with inclusive bounds; the default must lie in `[ge, le]`.

    Without `default=`, the wrapped function's parameter default is used; with neither, declaration
    raises `TypeError`.
    """

    def __init__(self, ge: float, le: float, *, default: float = _UNSET, log: bool = False):
        """Declare a tunable float.

        Args:
            ge: Smallest allowed value.
            le: Largest allowed value.
            default: The baseline value; omit to use the wrapped parameter's default.
            log: Search on a log scale.

        Raises:
            TypeError: No default anywhere, or one outside `[ge, le]`.
        """
        raise NotImplementedError


class OneOf:
    """A choice among plain values.

    The default is `default=` if given, else the wrapped function's parameter default, else the first
    option. A default that isn't one of the options raises `TypeError` at declaration.
    """

    def __init__(self, *options: str | int | float | bool | None, default: Any = _UNSET):
        """Declare a choice among plain values.

        Args:
            *options: The plain values to choose from.
            default: The baseline value; omit to use the wrapped parameter's default, or the first option.

        Raises:
            TypeError: The default is not one of the options.
        """
        raise NotImplementedError


class TextValue:
    """A text knob; `size_cap` rejects prompt bloat (default: `_default_size_cap(default)`).

    Without `default=`, the wrapped function's parameter default is used; with neither, declaration
    raises `TypeError`.
    """

    def __init__(self, *, default: str = _UNSET, size_cap: int | None = None, edit_instructions: str | None = None):
        """Declare a tunable text.

        Args:
            default: The baseline text; omit to use the wrapped parameter's default.
            size_cap: Max characters; None means `max(2000, 3 * len(default))`.
            edit_instructions: Shown to the proposer next to this value.

        Raises:
            TypeError: No default anywhere.
        """
        raise NotImplementedError


class Component:
    """One constructible alternative, returned by `component()`."""


class Choice:
    """A swap between components, returned by `choice()`; the first alternative is the baseline."""


def component(factory: Callable[..., Any], /, **kwargs: Any) -> Component:
    """Declare one constructible alternative; `Int`/`Float` kwargs are tunable, others pass through.

    `factory` is positional-only, so a constructor kwarg named `factory` can still be passed.

    Args:
        factory: The class or callable to construct.
        **kwargs: Constructor arguments; `Int` / `Float` markers among them are tunable.

    Raises:
        TypeError: The constructor rejects its own defaults (probed at declaration).
    """
    raise NotImplementedError


def choice(*alternatives: Component, labels: Sequence[str] | None = None) -> Choice:
    """Declare a swap between components.

    Labels default to class names. The same class twice without labels is a `TypeError` at
    declaration. Labels must be unique and contain no dots (they become key segments).

    Args:
        *alternatives: From `component()`; the first is the baseline.
        labels: One name per alternative, used in keys; defaults to class names.

    Raises:
        TypeError: The same class twice without labels, or duplicate or dotted labels.
    """
    raise NotImplementedError


def _default_size_cap(baseline: str) -> int:
    """`max(2000, 3 * len(baseline))`; shared with `files.py`."""
    raise NotImplementedError
