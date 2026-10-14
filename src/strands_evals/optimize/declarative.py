"""Canonical declaration vocabulary for component optimization, plus the `@tunable` frontend.

`Component`, `Choice`, `Float`, `Int`, and `TextValue` describe what may change; `@tunable`
mounts them for callers who own agent construction (declarations at the construction site, the
framework builds the component and passes it in). A plugin that mounted the same vocabulary on
already-built agents was removed: swapping a component after `Agent.__init__` leaves the old
component's hooks registered, so structural changes only happen through construction.

The `@tunable` shape follows hypothesis' `@given`: decorator kwargs name function parameters and
carry strategy objects.

```python
@tunable(
    conversation_manager=Choice(
        Component(SlidingWindowConversationManager, window_size=Int(4, 200, default=40)),
        Component(SummarizingConversationManager,
                  summary_ratio=Float(0.1, 0.8, default=0.3),
                  preserve_recent_messages=Int(4, 16, default=10)),
    ),
    sop=TextValue(default="be helpful", size_cap=14_000),
)
def create_agent(case_context, conversation_manager, sop):
    return Agent(..., conversation_manager=conversation_manager, system_prompt=sop)

create_agent.artifacts()                 # Mapping[str, Artifact] for the Optimizer
agent = create_agent.from_candidate(candidate, case_context)   # fresh agent per rollout
```
"""

import inspect
import json
import logging
from collections.abc import Mapping

from typing_extensions import Any, Callable

from .artifacts import Artifact, _BaseArtifact

logger = logging.getLogger(__name__)


class Float:
    """A tunable float constructor argument with an inclusive range."""

    is_int = False

    def __init__(self, ge: float, le: float, *, default: float | None = None):
        if default is None:
            raise ValueError("Float needs a default: it is the optimization baseline")
        if not (ge <= default <= le):
            raise ValueError(f"default {default} is outside [{ge}, {le}]")
        self.ge, self.le, self.default = ge, le, default

    def parse(self, value: Any) -> float:
        number = float(value)
        if not (self.ge <= number <= self.le):
            raise ValueError(f"{number} is outside the allowed range [{self.ge}, {self.le}]")
        return number

    def to_artifact(self) -> "Artifact":
        """Adapt this marker to the optimizable `Artifact` protocol (used by `ArtifactSpace`)."""
        return _to_artifact(self)


class Int(Float):
    """A tunable integer constructor argument with an inclusive range."""

    is_int = True

    def parse(self, value: Any) -> int:
        number = float(value)
        if number != int(number):
            raise ValueError(f"expected an integer, got {value}")
        return int(super().parse(number))


class Component:
    """One constructible alternative: a factory plus its arguments.

    Marked arguments (`Float`/`Int`) are tunable; plain values pass through unchanged, so fixed
    and tunable constructor arguments mix freely.
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
        return self.factory(**{**self.fixed, **self.default_params(), **self.parse_params(params)})

    def to_artifact(self) -> "Artifact":
        """Adapt this marker to the optimizable `Artifact` protocol (used by `ArtifactSpace`)."""
        return _to_artifact(self)


class Choice:
    """A swap between `Component` alternatives. The first one is the baseline."""

    def __init__(self, *alternatives: Component):
        if not alternatives:
            raise ValueError("Choice needs at least one Component")
        self.by_name = {c.name: c for c in alternatives}
        if len(self.by_name) != len(alternatives):
            raise ValueError("Component alternatives must have distinct factory names")
        self.default = alternatives[0]

    def default_value(self) -> str:
        return json.dumps({"type": self.default.name, "params": self.default.default_params()})

    def parse(self, value: str) -> tuple[Component, dict[str, Any]]:
        spec = json.loads(value)
        kind = spec.get("type")
        if kind not in self.by_name:
            raise ValueError(f"unknown component type {kind!r}; one of {sorted(self.by_name)}")
        component = self.by_name[kind]
        return component, component.parse_params(spec.get("params") or {})

    def build(self, value: str) -> Any:
        component, params = self.parse(value)
        return component.build(params)

    def to_artifact(self) -> "Artifact":
        """Adapt this marker to the optimizable `Artifact` protocol (used by `ArtifactSpace`)."""
        return _to_artifact(self)


class TextValue:
    """A tunable text parameter (a prompt)."""

    def __init__(self, *, default: str, size_cap: int | None = None, edit_instructions: str | None = None):
        self.default = default
        self.size_cap = size_cap
        self.edit_instructions = edit_instructions

    def to_artifact(self) -> "Artifact":
        """Adapt this marker to the optimizable `Artifact` protocol (used by `ArtifactSpace`)."""
        return _to_artifact(self)


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
            if str(option) == str(value).strip():
                return option
        raise ValueError(f"{value!r} is not one of {[str(option) for option in self.options]}")

    def to_artifact(self) -> "Artifact":
        """Adapt this marker to the optimizable `Artifact` protocol (used by `ArtifactSpace`)."""
        return _to_artifact(self)


def _to_artifact(marker: Any) -> "_DeclaredArtifact":
    if isinstance(marker, Component):
        marker = Choice(marker)
    return _DeclaredArtifact("", marker)


class _DeclaredArtifact(_BaseArtifact):
    """One declared parameter as an optimizable `Artifact`. Validation delegates to the marker,
    and for `Choice`/`Component` the real constructor is probed, so impossible values die at
    proposal time."""

    def __init__(self, name: str, marker: Any):
        edit_instructions = _instructions_for(name, marker)
        super().__init__(edit_instructions=edit_instructions, size_cap=getattr(marker, "size_cap", None))
        self.name = name
        self.marker = marker

    def read(self) -> str:
        if isinstance(self.marker, Choice):
            return self.marker.default_value()
        return str(self.marker.default)

    def apply(self, value: str, workspace) -> None:
        return None  # runtime artifact: delivered through the factory call

    def display_path(self) -> str:
        return f"<tunable:{self.name}>"

    def validate(self, value: str) -> None:
        super().validate(value)
        if isinstance(self.marker, Choice):
            component, params = self.marker.parse(value)
            try:
                component.build(params)
            except Exception as e:  # noqa: BLE001 - any constructor failure must reject the edit
                raise ValueError(f"component {component.name} rejected params {params}: {e}") from e
        elif isinstance(self.marker, (Float, OneOf)):
            self.marker.parse(value)


def _instructions_for(name: str, marker: Any) -> str:
    if isinstance(marker, Choice):
        shapes = []
        for component in marker.by_name.values():
            params = ", ".join(
                f'"{k}": <{"int" if v.is_int else "number"} {v.ge}-{v.le}>' for k, v in component.tunables.items()
            )
            shapes.append(f'{{"type": "{component.name}", "params": {{{params}}}}}')
        return "JSON component spec, exactly one of: " + " OR ".join(shapes) + ". Output valid JSON only."
    if isinstance(marker, Float):
        kind = "integer" if marker.is_int else "number"
        return f"A {kind} between {marker.ge} and {marker.le}. Output only the number."
    if isinstance(marker, OneOf):
        listed = ", ".join(str(option) for option in marker.options)
        return f"One of: {listed}. Output only the value."
    return marker.edit_instructions or "Text value."


class TunableFactory:
    """What `@tunable` returns: the original factory plus the optimization surface."""

    def __init__(self, fn: Callable[..., Any], markers: dict[str, Any]):
        self._fn = fn
        self.markers = markers
        self.__name__ = getattr(fn, "__name__", "tunable_factory")

    def artifacts(self) -> dict[str, Artifact]:
        return {name: _DeclaredArtifact(name, marker) for name, marker in self.markers.items()}

    def resolve(self, candidate_values: dict[str, str]) -> dict[str, Any]:
        """Candidate strings to constructed objects / parsed scalars, keyed by parameter name."""
        resolved: dict[str, Any] = {}
        for name, marker in self.markers.items():
            raw = candidate_values.get(name)
            if raw is None:
                raw = _DeclaredArtifact(name, marker).read()
            if isinstance(marker, Choice):
                resolved[name] = marker.build(raw)  # fresh component instance per call
            elif isinstance(marker, (Float, OneOf)):
                resolved[name] = marker.parse(raw)
            else:
                resolved[name] = raw
        return resolved

    def from_candidate(self, candidate, *args: Any, **kwargs: Any) -> Any:
        """Call the factory with the candidate's values resolved into real objects."""
        values = dict(candidate) if isinstance(candidate, Mapping) else candidate.values
        return self._fn(*args, **kwargs, **self.resolve(values))

    def bind(self, candidate: Any) -> Callable[..., Any]:
        """Return the factory with the candidate's values bound. Markers resolve on every call,
        so each invocation builds fresh component instances."""
        values = dict(candidate) if isinstance(candidate, Mapping) else dict(candidate.values)

        def bound(*args: Any, **kwargs: Any) -> Any:
            return self._fn(*args, **kwargs, **self.resolve(values))

        return bound

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
        baseline = {name: None for name in self.markers if name not in kwargs}
        resolved = self.resolve({})
        for name in baseline:
            kwargs[name] = resolved[name]
        return self._fn(*args, **kwargs)


def probe_markers(markers: Mapping[str, Any]) -> None:
    """Build every component alternative once with its defaults, so an impossible declaration
    (a kwarg the constructor rejects) fails at declaration time instead of mid-run."""
    for name, marker in markers.items():
        if not isinstance(marker, Choice):
            continue
        for alternative in marker.by_name.values():
            try:
                alternative.build({})
            except Exception as error:
                raise TypeError(f"{name}: component {alternative.name} rejects its defaults: {error}") from error


def tunable(**markers: Any) -> Callable[[Callable[..., Any]], TunableFactory]:
    """Declare a construction function's tunable parameters (hypothesis-style decorator)."""
    for name, marker in markers.items():
        if not isinstance(marker, (Choice, Float, TextValue, Component, OneOf)):
            raise TypeError(f"{name}: expected choice/component/Float/Int/TextValue/OneOf, got {type(marker).__name__}")
        if isinstance(marker, Component):
            markers[name] = Choice(marker)  # a single Component is a one-alternative swap
        probe_markers({name: markers[name]})

    def wrap(fn: Callable[..., Any]) -> TunableFactory:
        signature_params = set(inspect.signature(fn).parameters)
        missing = set(markers) - signature_params
        if missing:
            raise TypeError(f"@tunable declares {sorted(missing)}, but {fn.__name__} has no such parameters")
        return TunableFactory(fn, markers)

    return wrap
