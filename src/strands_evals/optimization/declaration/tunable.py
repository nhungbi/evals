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

    A dataclass, not a record: it holds a callable and is never saved. New fields may be added
    later without breaking existing `build_task` functions.

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
        if not callable(fn):
            raise TypeError(f"tunable() needs a function, got {fn!r}")
        parameters = inspect.signature(fn).parameters
        self._fn = fn
        self._names = tuple(markers)
        self._accepted = frozenset(name for name, p in parameters.items() if p.kind in _KEYWORD_KINDS)
        self.space = Space(_fill_defaults(markers, fn, parameters))
        self._baseline = self.space.complete({})
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build the baseline (every value at its default); never copies files.

        Arguments passed here win over the declared values, as in a plain call of the function.
        """
        resolved = {name: self.space.resolve(name, self._baseline, None) for name in self._names if name not in kwargs}
        return self._fn(*args, **kwargs, **resolved)

    def validate(self, values: Mapping[str, Value]) -> None:
        """Same as `self.space.validate(values)`; raises `ValueError`."""
        self.space.validate(values)

    def bind(self, values: Mapping[str, Value], *, workspace: Path | None = None) -> Callable[..., Any]:
        """Return a factory that builds fresh components with `values` bound on every call.

        Missing keys take their defaults, but nothing is dropped: a key inactive under the chosen
        alternatives is an error, not silently ignored. Validated here, before any agent is built.

        Copies files only if an active value reaches the agent as a path (`as_path=True`,
        `SkillLibrary`): into `workspace` (cached), or with `workspace=None` into one temp folder per
        `bind()` call, deleted when the bound factory is garbage-collected. The bound factory takes
        resource kwargs, each passed only if the user's function declares it by name (`**kwargs`
        doesn't count); a kwarg named like a declared marker raises `_ResourceConflictError`.

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
                    f"resources {conflicts} are named like declared markers of {self.__name__}; rename them"
                )
            passed = {name: value for name, value in resources.items() if name in self._accepted}
            dropped = sorted(set(resources) - set(passed))
            if dropped:
                logger.debug("factory=<%s>, resources=<%s> | factory does not declare them", self.__name__, dropped)
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
    markers: Mapping[str, Marker], fn: Callable[..., Any], parameters: Mapping[str, inspect.Parameter]
) -> dict[str, Marker]:
    """Check each name against `fn`; a marker without `default=` takes the parameter's default."""
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    fn_name = getattr(fn, "__name__", repr(fn))
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
    if isinstance(marker, (Int, Float, TextValue)) and marker.default is None:
        if default is None:
            raise TypeError(f"{name}: the parameter's default is None; pass default= to {type(marker).__name__}")
    elif not (isinstance(marker, OneOf) and marker.default is _UNSET):
        return marker
    try:
        return marker._with_default(default)
    except TypeError as error:
        raise TypeError(f"{name}: the parameter's default {default!r} doesn't fit its marker: {error}") from error
