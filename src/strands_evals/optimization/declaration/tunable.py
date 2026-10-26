"""The `tunable` mounts and the factory they return."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from ...types.optimization import Value
from .space import Marker, Space


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
    """A user's agent factory with declared tunable parameters."""

    space: Space

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build the baseline (every value at its default); never copies files."""
        raise NotImplementedError

    def validate(self, values: Mapping[str, Value]) -> None:
        """Same as `self.space.validate(values)`; raises `ValueError`."""
        raise NotImplementedError

    def bind(self, values: Mapping[str, Value], *, workspace: Path | None = None) -> Callable[..., Any]:
        """Return a factory that builds fresh components with `values` bound on every call.

        Copies files only if an active value reaches the agent as a path (`as_path=True`,
        `SkillLibrary`): into `workspace` (cached), or with `workspace=None` into one temp folder per
        `bind()` call, deleted when the bound factory is garbage-collected. The bound factory takes
        resource kwargs, each passed only if the user's function declares it; a kwarg named like a
        declared marker raises `_ResourceConflictError`.
        """
        raise NotImplementedError


@overload
def tunable(fn: Callable[..., Any], /, *, inject: Mapping[str, Marker]) -> TunableFactory: ...


@overload
def tunable(**markers: Marker) -> Callable[[Callable[..., Any]], TunableFactory]: ...


def tunable(
    fn: Callable[..., Any] | None = None, /, **kwargs: Any
) -> TunableFactory | Callable[[Callable[..., Any]], TunableFactory]:
    """Declare tunable parameters: `tunable(fn, inject={...})` or `@tunable(name=marker, ...)`.

    A marker naming no parameter of `fn` (unless `fn` takes `**kwargs`) raises `TypeError`. A marker
    without `default=` takes the parameter's own default, so `fn()` and the baseline build the same
    agent; a marker with neither raises `TypeError`.
    """
    raise NotImplementedError
