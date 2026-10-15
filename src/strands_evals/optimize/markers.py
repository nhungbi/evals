"""The declaration markers: what may change about an agent.

Each marker declares one kind of knob on a construction function. `Float`/`Int` are bounded
numbers, `OneOf` picks a plain value, `TextValue` is a text knob, `component` wraps one
constructible alternative (its `Float`/`Int` kwargs are tunable, every other kwarg is fixed),
`choice` swaps between component alternatives, `Switch` is a structural on/off around another
declaration, and `SkillLibrary` is a curated set of skill files.

Markers carry declarations only. The typed parameter space they expand into lives in `space.py`,
and the two mounts (`@tunable` and `tunable(fn, inject=...)`) live in `factory.py`.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Float:
    """A tunable float constructor argument with an inclusive range.

    `default` is mandatory because it is the optimization baseline.
    """

    is_int = False

    def __init__(self, ge: float, le: float, *, default: float | None = None, log: bool = False):
        if default is None:
            raise ValueError("Float needs a default: it is the optimization baseline")
        if not (ge <= default <= le):
            raise ValueError(f"default {default} is outside [{ge}, {le}]")
        self.ge, self.le, self.default, self.log = ge, le, default, log

    def parse(self, value: Any) -> float:
        number = float(value)
        if not (self.ge <= number <= self.le):
            raise ValueError(f"{number} is outside the allowed range [{self.ge}, {self.le}]")
        return number


class Int(Float):
    """A tunable integer constructor argument with an inclusive range."""

    is_int = True

    def parse(self, value: Any) -> int:
        number = float(value)
        if number != int(number):
            raise ValueError(f"expected an integer, got {value}")
        return int(super().parse(number))


class OneOf:
    """A choice between plain values (strings, numbers, booleans). The first one is the baseline."""

    def __init__(self, *options: Any, default: Any | None = None):
        if not options:
            raise ValueError("OneOf needs at least one option")
        self.options = options
        self.default = options[0] if default is None else default
        if self.default not in options:
            raise ValueError(f"default {self.default!r} is not one of {list(options)}")

    def parse(self, value: Any) -> Any:
        for option in self.options:
            if option == value or str(option) == str(value).strip():
                return option
        raise ValueError(f"{value!r} is not one of {[str(option) for option in self.options]}")


class TextValue:
    """A tunable text parameter (a prompt, instructions).

    The size cap rejects prompt bloat; when unset, a default cap applies in the space.
    """

    def __init__(self, *, default: str, size_cap: int | None = None, edit_instructions: str | None = None):
        self.default = default
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions


class Component:
    """One constructible alternative: a factory plus its real constructor arguments.

    Kwargs whose value is `Float`/`Int` are tunable; every other kwarg is fixed and passes to the
    constructor untouched, so fixed and tunable arguments mix freely.
    """

    def __init__(self, factory: Callable[..., Any], **kwargs: Any):
        self.factory = factory
        self.name = getattr(factory, "__name__", str(factory))
        self.tunables = {k: v for k, v in kwargs.items() if isinstance(v, Float)}
        self.fixed = {k: v for k, v in kwargs.items() if not isinstance(v, Float)}

    def default_params(self) -> dict[str, Any]:
        return {k: v.default for k, v in self.tunables.items()}

    def parse_params(self, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params) - set(self.tunables)
        if unknown:
            raise ValueError(f"{self.name} has no tunable params {sorted(unknown)}; tunable: {sorted(self.tunables)}")
        return {k: self.tunables[k].parse(v) for k, v in params.items()}

    def build(self, params: dict[str, Any]) -> Any:
        """Construct the component with `params` over the defaults. A fresh instance per call."""
        return self.factory(**{**self.fixed, **self.default_params(), **self.parse_params(params)})


class Choice:
    """A swap between `Component` alternatives. The first one is the baseline."""

    def __init__(self, *alternatives: Component):
        if not alternatives:
            raise ValueError("choice needs at least one component")
        self.by_name = {c.name: c for c in alternatives}
        if len(self.by_name) != len(alternatives):
            raise ValueError("component alternatives must have distinct factory names")
        self.default = alternatives[0]


class Switch:
    """A structural on/off around another declaration (an optional plugin, an optional manager).

    When on, the inner declaration applies; when off, the parameter is passed as `off_value`
    (`None` by default, which most constructors treat as absent).
    """

    def __init__(self, inner: Any, *, default_on: bool = True, off_value: Any = None):
        if isinstance(inner, Component):
            inner = Choice(inner)
        if not isinstance(inner, (Choice, Float, TextValue, OneOf)):
            raise TypeError(f"Switch wraps a marker, got {type(inner).__name__}")
        self.inner = inner
        self.default_on = default_on
        self.off_value = off_value


class SkillLibrary:
    """A curated set of skill files the optimizer may add to, edit, or drop from.

    The candidate value is the JSON listing of member file names relative to `path`. Editing the
    library is the `SkillCuration` method's job; the declaration only names the directory.
    """

    def __init__(self, path: str, *, edit_instructions: str | None = None):
        self.path = path
        self.edit_instructions = edit_instructions


MARKER_TYPES = (Choice, Component, Float, TextValue, OneOf, Switch, SkillLibrary)


def component(factory: Callable[..., Any], **kwargs: Any) -> Component:
    """One constructible alternative: a native class (or factory) plus its real arguments."""
    return Component(factory, **kwargs)


def choice(*alternatives: Component) -> Choice:
    """A swap between `component` alternatives. The first one is the baseline."""
    return Choice(*alternatives)


def normalize_marker(name: str, marker: Any) -> Any:
    """Promote a bare `Component` to a one-alternative `Choice` and reject unknown marker types."""
    if isinstance(marker, Component):
        marker = Choice(marker)
    if not isinstance(marker, MARKER_TYPES):
        raise TypeError(
            f"{name}: expected choice/component/Float/Int/TextValue/OneOf/Switch/SkillLibrary, "
            f"got {type(marker).__name__}"
        )
    return marker


def probe_marker(name: str, marker: Any) -> None:
    """Build every component alternative once with its defaults, so an impossible declaration
    (a kwarg the constructor rejects) fails at declaration time instead of mid-run."""
    inner = marker.inner if isinstance(marker, Switch) else marker
    if not isinstance(inner, Choice):
        return
    for alternative in inner.by_name.values():
        try:
            alternative.build({})
        except Exception as error:
            raise TypeError(f"{name}: component {alternative.name} rejects its defaults: {error}") from error
