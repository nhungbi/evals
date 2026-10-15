"""The two declaration mounts, and the factory object both of them return.

Mount 1, the decorator, for callers who co-locate declarations with their construction code. The
framework builds each marker's value and passes the finished object in:

```python
@tunable(conversation_manager=choice(
    component(SlidingWindowConversationManager, window_size=Int(4, 200, default=40)),
    component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)),
))
def create_agent(conversation_manager):
    return Agent(conversation_manager=conversation_manager, ...)

create_agent()                 # baseline agent, no optimizer involved
```

Mount 2, `inject=`, for callers who do not change their construction code at all. The factory
only needs to accept the named keyword, directly or through `**kwargs`:

```python
factory = tunable(create_harness, inject={
    "conversation_manager": choice(...),
    "effort": OneOf("low", "medium", "high"),
})
```

Both mounts return a `TunableFactory`. A bound factory builds fresh component instances on every
call (stateful objects are never shared across rollouts, and every component enters through the
agent's real constructor so its hooks register), and it records each object it builds into the
active trial, which is how usage metering works for single-agent and multi-agent rollouts alike.
"""

import inspect
import logging
from typing import Any, Callable, Mapping

from .execution import record_built
from .markers import normalize_marker, probe_marker
from .space import Space

logger = logging.getLogger(__name__)


class TunableFactory:
    """A construction function plus its declared search space.

    - Plain call builds the baseline: every declared parameter not supplied fills at its default.
    - `bind(candidate)` returns the factory with a candidate's typed values bound; markers resolve
      on every call of the bound function.
    - `build_task(rollout_fn)` adapts `rollout_fn(agent_factory, case)` to the optimizer's
      `build_task` contract. `Optimizer` calls this for you.
    """

    def __init__(self, fn: Callable[..., Any], markers: dict[str, Any]):
        if not markers:
            raise TypeError("tunable needs at least one marker: there is nothing to optimize otherwise")
        self._fn = fn
        self.markers = markers
        self._space = Space(markers=markers)
        self.__name__ = getattr(fn, "__name__", "tunable_factory")

    def space(self) -> Space:
        """The typed parameter space this factory declares (markers only, no file artifacts)."""
        return self._space

    def resolve(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Typed candidate values to constructed objects / parsed scalars, keyed by parameter name."""
        complete = self._space.complete(dict(values))
        return {name: self._space.resolve(name, complete) for name in self.markers}

    def bind(self, candidate: Any) -> Callable[..., Any]:
        """Return the factory with the candidate's values bound.

        Markers resolve on every call, so each invocation builds fresh component instances. Every
        object the bound call returns is recorded into the active trial for usage metering.
        """
        values = dict(candidate) if isinstance(candidate, Mapping) else dict(candidate.values)

        def bound(*args: Any, **kwargs: Any) -> Any:
            resolved = {name: value for name, value in self.resolve(values).items() if name not in kwargs}
            built = self._fn(*args, **kwargs, **resolved)
            record_built(built)
            return built

        return bound

    def from_candidate(self, candidate: Any, *args: Any, **kwargs: Any) -> Any:
        """Build in one call. Equivalent to `factory.bind(candidate)(*args, **kwargs)`."""
        return self.bind(candidate)(*args, **kwargs)

    def build_task(self, rollout_fn: Callable[..., Any]) -> Callable[[Any], Callable[[Any], Any]]:
        """Adapt `rollout_fn(agent_factory, case)` to the optimizer's `build_task` contract."""

        def build(candidate: Any) -> Callable[[Any], Any]:
            agent_factory = self.bind(candidate)
            if inspect.iscoroutinefunction(rollout_fn):

                async def async_task(case: Any) -> Any:
                    return await rollout_fn(agent_factory, case)

                return async_task
            return lambda case: rollout_fn(agent_factory, case)

        return build

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Plain call with baseline values for any declared parameter not supplied."""
        resolved = {name: value for name, value in self.resolve({}).items() if name not in kwargs}
        return self._fn(*args, **kwargs, **resolved)


def tunable(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    inject: Mapping[str, Any] | None = None,
    **markers: Any,
) -> Any:
    """Declare a construction function's tunable parameters.

    Two forms. `@tunable(name=marker, ...)` decorates your own factory; the framework builds each
    marker's value and passes it as the named parameter. `tunable(fn, inject={name: marker})`
    wraps an existing callable without touching it; the names must be keyword arguments the
    callable accepts, directly or via `**kwargs`.

    Raises:
        TypeError: On an unknown marker type, a marker name the callable cannot accept, a
            component that rejects its own defaults, or mixing the two forms.
    """
    if fn is None:
        if inject is not None:
            raise TypeError("inject requires the callable: tunable(fn, inject={...})")
        normalized = _normalized(markers)

        def wrap(inner: Callable[..., Any]) -> TunableFactory:
            parameters = set(inspect.signature(inner).parameters)
            missing = set(normalized) - parameters
            if missing:
                raise TypeError(f"@tunable declares {sorted(missing)}, but {inner.__name__} has no such parameters")
            return TunableFactory(inner, normalized)

        return wrap
    if markers:
        raise TypeError("pass markers as decorator kwargs or through inject, not both")
    if inject is None:
        raise TypeError("tunable(fn) needs inject={name: marker}; the bare form was removed")
    normalized = _normalized(inject)
    parameters = inspect.signature(fn).parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if not accepts_kwargs:
        allowed = {
            name
            for name, parameter in parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        unknown = set(normalized) - allowed
        if unknown:
            raise TypeError(f"{getattr(fn, '__name__', fn)} accepts neither **kwargs nor {sorted(unknown)}")
    return TunableFactory(fn, normalized)


def _normalized(markers: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, marker in markers.items():
        normalized[name] = normalize_marker(name, marker)
        probe_marker(name, normalized[name])
    return normalized
