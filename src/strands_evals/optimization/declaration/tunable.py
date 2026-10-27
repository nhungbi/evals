"""The `tunable` mounts and the factory they return."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from ...types.optimization import Value
from .markers import _UNSET, Float, Int, OneOf, TextValue
from .space import Marker, Space

logger = logging.getLogger(__name__)

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


@dataclass(frozen=True)
class BuildContext:
    """What `Optimizer(build_task=...)` receives, once per candidate.

    Never saved. Fields may be added later; `build_task` functions keep working.

    Attributes:
        values: The candidate's typed flat values.
        resolve: `resolve(name)` returns what the factory path would pass for that declared name:
            a fresh component, a value, or a path inside the candidate's copy.
        workspace: The candidate's copy of the file-backed folders
            (`<run>/candidates/<id>/optimization_files/`), or None when the space has no
            file-backed values. Lets a config-defined agent load the whole edited file, e.g. the
            `agent.json` a `JsonField` points into.
    """

    values: Mapping[str, Value]
    resolve: Callable[[str], Any]
    workspace: Path | None


class _ResourceConflictError(TypeError):
    """A resource passed to a bound factory is named like a declared marker; a setup error, not a trial status."""


class TunableFactory:
    """A user's agent factory with declared tunable parameters.

    Built by `tunable`; wraps the user's function (name, docstring, signature) without changing it.
    """

    space: Space

    def __init__(self, fn: Callable[..., Any], markers: Mapping[str, Marker]):
        """Check the declared names against `fn` and build the space; raises `TypeError`."""
        if isinstance(fn, TunableFactory):
            raise TypeError(f"{fn._label} is already tunable; declare every marker in one tunable(...)")
        if not callable(fn):
            raise TypeError(f"tunable() needs a function, got {fn!r}")
        self._fn = fn
        self._signature = inspect.signature(fn)
        # Not every callable has a __name__ (functools.partial); used in error messages and logs.
        self._label: str = getattr(fn, "__name__", repr(fn))
        self._names = tuple(markers)
        parameters = self._signature.parameters
        self._accepted = frozenset(name for name, p in parameters.items() if p.kind in _KEYWORD_KINDS)
        self.space = Space(_fill_defaults(markers, self._label, parameters))
        self._baseline = self.space.complete({})
        # updated=(): copying fn.__dict__ could overwrite this factory's own attributes.
        functools.update_wrapper(self, fn, updated=())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build the baseline (every value at its default); never copies files.

        Arguments passed here, by position or by keyword, win over the declared values, as in a
        plain call of the function.
        """
        given = self._signature.bind_partial(*args, **kwargs).arguments
        resolved = {
            name: self.space.resolve(name, self._baseline, None)
            for name in self._names
            if name not in given and name not in kwargs  # a name caught by **kwargs is only in kwargs
        }
        return self._fn(*args, **kwargs, **resolved)

    def validate(self, values: Mapping[str, Value]) -> None:
        """Same as `self.space.validate(values)`; raises `ValueError`."""
        self.space.validate(values)

    def bind(self, values: Mapping[str, Value], *, workspace: Path | None = None) -> Callable[..., Any]:
        """Return a factory that builds fresh components with `values` bound on every call.

        Pass only the keys you change; the rest take their defaults. A key that doesn't apply under
        the chosen alternatives (e.g. a window size while the summarizing manager is chosen) raises
        `ValueError` instead of being ignored. Everything is checked here, before any agent is built.

        Files are copied only when a value reaches the agent as a path (`as_path=True`,
        `SkillLibrary`). The copy goes into `workspace` (made once, then reused), or, with
        `workspace=None`, into a temp folder made for this `bind()` call and deleted once the
        bound factory is garbage-collected.

        The bound factory takes resources as keyword arguments, e.g. `bound(sandbox=sb)`. Each is
        passed on only if the user's function names that parameter (`**kwargs` doesn't count). A
        resource named like a declared marker raises `_ResourceConflictError`.

        Args:
            values: The candidate's values.
            workspace: Where to write the candidate's file copy; None outside a run.

        Returns:
            A factory that builds fresh components on every call and takes resource kwargs.

        Raises:
            ValueError: The values are not a valid point of the space.
        """
        bound_values = {**self.space.complete(values), **values}
        self.space.validate(bound_values)

        def bound(**resources: Any) -> Any:
            conflicts = sorted(name for name in resources if name in self._names)
            if conflicts:
                raise _ResourceConflictError(
                    f"resources {conflicts} are named like declared markers of {self._label}; rename them"
                )
            passed = {name: value for name, value in resources.items() if name in self._accepted}
            dropped = sorted(set(resources) - set(passed))
            if dropped:
                logger.debug("factory=<%s>, resources=<%s> | factory does not declare them", self._label, dropped)
            resolved = {name: self.space.resolve(name, bound_values, workspace) for name in self._names}
            return self._fn(**resolved, **passed)

        return bound


@overload
def tunable(fn: Callable[..., Any], /, *, inject: Mapping[str, Marker]) -> TunableFactory: ...


@overload
def tunable(**markers: Marker) -> Callable[[Callable[..., Any]], TunableFactory]: ...


def tunable(
    fn: Callable[..., Any] | None = None, /, **kwargs: Any
) -> TunableFactory | Callable[[Callable[..., Any]], TunableFactory]:
    """Declare tunable parameters: `tunable(fn, inject={...})` or `@tunable(name=marker, ...)`.

    A marker naming no parameter of `fn` (unless `fn` takes `**kwargs`) raises `TypeError`.

    Each marker's baseline is its `default=` if given, which wins over the parameter's default.
    Otherwise it takes the parameter's own default, so `fn()` and the baseline build the same agent.
    With neither, `TypeError`, except that `OneOf` falls back to its first option. Under
    `Optimizer(build_task=..., space=...)` there is no function, so every marker but `OneOf` needs
    `default=`.

    Args:
        fn: The factory; omit when used as a decorator.
        **kwargs: `inject={name: marker}` when `fn` is given, otherwise the markers themselves.

    Returns:
        A `TunableFactory`, or a decorator returning one.

    Raises:
        TypeError: A marker names no parameter, or has no default anywhere.
    """
    if fn is not None:
        if set(kwargs) != {"inject"} or not isinstance(kwargs["inject"], Mapping):
            raise TypeError("tunable(fn, ...) takes only inject={name: marker}")
        return TunableFactory(fn, kwargs["inject"])
    if isinstance(kwargs.get("inject"), Mapping):
        raise TypeError("inject= needs the function: tunable(fn, inject={name: marker})")

    def decorate(function: Callable[..., Any]) -> TunableFactory:
        return TunableFactory(function, kwargs)

    return decorate


def _fill_defaults(
    markers: Mapping[str, Marker], fn_name: str, parameters: Mapping[str, inspect.Parameter]
) -> dict[str, Marker]:
    """Check each name against the function's parameters; a marker without `default=` takes the parameter's."""
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    filled: dict[str, Marker] = {}
    for name, marker in markers.items():
        parameter = parameters.get(name)
        if parameter is not None and parameter.kind not in _KEYWORD_KINDS:
            raise TypeError(f"{name}: {fn_name} takes {name!r} as {parameter.kind.description}, not by keyword")
        if parameter is None and not takes_kwargs:
            raise TypeError(f"{name}: {fn_name} has no parameter {name!r} and takes no **kwargs")
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            filled[name] = _with_parameter_default(name, marker, parameter.default)
        else:
            filled[name] = marker
    return filled


def _with_parameter_default(name: str, marker: Marker, default: Any) -> Marker:
    """`marker` with the parameter's `default` filled in, if the marker has none; raises `TypeError`."""
    if not isinstance(marker, (Int, Float, OneOf, TextValue)):
        return marker  # components, choices, and tools take no parameter default
    if isinstance(marker, OneOf):
        # OneOf marks "no default" with _UNSET, because None can be one of its options.
        if marker.default is not _UNSET:
            return marker
    elif marker.default is not None:
        return marker
    elif default is None:
        raise TypeError(f"{name}: the parameter's default is None; pass default= to {type(marker).__name__}")
    try:
        return marker._with_default(default)
    except TypeError as error:
        raise TypeError(f"{name}: the parameter's default {default!r} doesn't fit its marker: {error}") from error
