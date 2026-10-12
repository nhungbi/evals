"""The component-optimization front door: one declaration vocabulary, two ways to mount it.

The vocabulary names native strands classes and their real constructor arguments. `component`
wraps one constructible alternative (`Float`/`Int` kwargs are tunable, plain kwargs are fixed),
`choice` swaps between alternatives (the first is the baseline), `OneOf` picks a plain value,
`TextValue` is a text knob. Every alternative is probed (built once with its defaults) at
declaration time, so an impossible declaration fails before any rollout.

Mount 1, the decorator, for callers who co-locate declarations with their construction code. The
framework builds each component and passes the finished object in:

```python
@tunable(conversation_manager=choice(
    component(SlidingWindowConversationManager, window_size=Int(4, 200, default=40)),
    component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)),
))
def create_agent(sandbox, conversation_manager):
    return Agent(sandbox=sandbox, conversation_manager=conversation_manager, ...)

create_agent(sandbox)                 # baseline agent, no optimizer involved
```

Mount 2, `inject=`, for callers who do not change their construction code at all. The factory
only needs to accept the named keyword (directly or through `**kwargs`); `create_harness` and a
thin `def build_agent(**overrides)` wrapper both qualify:

```python
factory = tunable(create_harness, inject={
    "conversation_manager": choice(...),
    "effort": OneOf("low", "medium", "high"),
})
```

Both mounts return a `TunableFactory` that plugs into `Optimizer(experiment, factory=..., run=...)`.
"""

import inspect

from typing_extensions import Any, Callable, Mapping

from .declarative import Choice, Component, Float, Int, OneOf, TextValue, TunableFactory, probe_markers
from .declarative import tunable as _decorator_tunable

__all__ = [
    "Choice",
    "Component",
    "Float",
    "Int",
    "OneOf",
    "TextValue",
    "TunableFactory",
    "choice",
    "component",
    "tunable",
]

_MARKER_TYPES = (Choice, Component, Float, TextValue, OneOf)


def component(factory: Callable[..., Any], **kwargs: Any) -> Component:
    """One constructible alternative: a native class (or factory) plus its real arguments.

    `Float`/`Int` kwargs are tunable; plain kwargs are fixed and pass through unchanged.
    """
    return Component(factory, **kwargs)


def choice(*alternatives: Component) -> Choice:
    """A swap between `component` alternatives. The first one is the baseline."""
    return Choice(*alternatives)


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
    callable accepts (directly or via `**kwargs`).

    Raises:
        TypeError: On an unknown marker type, a marker name the callable cannot accept, a
            component that rejects its own defaults, or mixing the two forms.
    """
    if fn is None:
        if inject is not None:
            raise TypeError("inject requires the callable: tunable(fn, inject={...})")
        return _decorator_tunable(**markers)
    if markers:
        raise TypeError("pass markers as decorator kwargs or through inject, not both")
    if inject is None:
        raise TypeError("tunable(fn) needs inject={name: marker}; the bare form was removed")
    return _injected(fn, inject)


def _injected(fn: Callable[..., Any], inject: Mapping[str, Any]) -> TunableFactory:
    """Build a `TunableFactory` whose markers bind to `fn` keyword arguments by name."""
    normalized: dict[str, Any] = {}
    for name, marker in inject.items():
        if not isinstance(marker, _MARKER_TYPES):
            raise TypeError(f"{name}: expected choice/component/Float/Int/TextValue/OneOf, got {type(marker).__name__}")
        normalized[name] = Choice(marker) if isinstance(marker, Component) else marker
        probe_markers({name: normalized[name]})

    parameters = inspect.signature(fn).parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if not accepts_kwargs:
        unknown = set(normalized) - {
            name
            for name, parameter in parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        if unknown:
            raise TypeError(f"{getattr(fn, '__name__', fn)} accepts neither **kwargs nor {sorted(unknown)}")
    return TunableFactory(fn, normalized)
