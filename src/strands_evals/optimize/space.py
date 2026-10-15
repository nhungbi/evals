"""The typed parameter space: every declared knob as a flat, typed, conditional parameter.

A candidate is canonically a flat typed mapping. Keys are dotted paths: a `choice` contributes
its own key (the chosen alternative's name) plus one key per tunable constructor argument of each
alternative (`manager.SummarizingConversationManager.summary_ratio`), active only when that
alternative is chosen. Numeric search methods and effect analysis read the typed values directly;
the content hash and the diff are derived views (`candidate.py`).

The space is built from declaration markers (`markers.py`) and file artifacts (`artifacts.py`)
together, so one run can tune a numeric knob and rewrite a skill file.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from .artifacts import Artifact, ArtifactSpace, _default_size_cap
from .markers import Choice, Float, OneOf, SkillLibrary, Switch, TextValue, normalize_marker, probe_marker

logger = logging.getLogger(__name__)

ParamKind = Literal["float", "int", "one_of", "choice", "switch", "text", "skills"]

#: Kinds proposed by text reflection; everything else is numeric/categorical search territory.
TEXT_KINDS = ("text", "skills")

_SWITCH_ON = "on"
_SWITCH_OFF = "off"


@dataclass(frozen=True)
class ParamInfo:
    """One typed parameter of the space.

    Attributes:
        key: The flat dotted key this parameter has in candidate values.
        kind: What the value is: a bounded number, a categorical pick, text, or a skill listing.
        default: The baseline value.
        low: Inclusive lower bound (`float`/`int` kinds).
        high: Inclusive upper bound (`float`/`int` kinds).
        options: The allowed values (`one_of`, `choice`, `switch` kinds).
        active_when: `(key, value)` condition under which this parameter exists, `None` for
            always-active parameters. Conditions chain: the condition key may itself carry one.
        log: Whether a numeric parameter should be searched on a log scale.
        size_cap: Maximum character count for `text` kinds.
        edit_instructions: What a text proposer is told about editing this parameter.
    """

    key: str
    kind: ParamKind
    default: Any
    low: float | None = None
    high: float | None = None
    options: tuple[Any, ...] = ()
    active_when: tuple[str, Any] | None = None
    log: bool = False
    size_cap: int | None = None
    edit_instructions: str | None = None


@dataclass
class _Entry:
    """One declared name: its marker or artifact, plus the parameters it expanded into."""

    name: str
    marker: Any = None
    artifact: Artifact | None = None
    params: list[ParamInfo] = field(default_factory=list)


class Space:
    """Every knob of one optimization run: markers and file artifacts under one typed vocabulary.

    Args:
        markers: Declared names to markers, usually a `TunableFactory`'s declarations.
        artifacts: Named file/text artifacts (`FileArtifact`, `JsonFieldArtifact`, ...), as a
            mapping or an `ArtifactSpace`. Merged with the markers under one namespace.
        root: Workspace root copied into each candidate workspace before artifact values apply.
            Taken from `artifacts` when that is an `ArtifactSpace` with a root.
    """

    def __init__(
        self,
        markers: Mapping[str, Any] | None = None,
        artifacts: ArtifactSpace | Mapping[str, Artifact] | None = None,
        *,
        root: str | Path | None = None,
    ):
        self._entries: dict[str, _Entry] = {}
        self._infos: dict[str, ParamInfo] = {}
        self.artifact_space: ArtifactSpace | None = None

        for name, marker in dict(markers or {}).items():
            marker = normalize_marker(name, marker)
            probe_marker(name, marker)
            entry = _Entry(name=name, marker=marker, params=_expand(name, marker))
            self._register(entry)

        if artifacts is not None:
            self.artifact_space = artifacts if isinstance(artifacts, ArtifactSpace) else ArtifactSpace(artifacts)
            for name in self.artifact_space.names():
                artifact = self.artifact_space[name]
                info = ParamInfo(
                    key=name,
                    kind="text",
                    default=artifact.read(),
                    size_cap=artifact.size_cap,
                    edit_instructions=artifact.edit_instructions,
                )
                self._register(_Entry(name=name, artifact=artifact, params=[info]))
        if root is not None and self.artifact_space is None:
            raise ValueError("root only applies to file artifacts; pass artifacts with it")
        if not self._entries:
            raise ValueError("the space is empty: declare at least one marker or artifact")

    def _register(self, entry: _Entry) -> None:
        if entry.name in self._entries:
            raise ValueError(f"{entry.name!r} is declared twice (as marker and artifact, or in both mounts)")
        self._entries[entry.name] = entry
        for info in entry.params:
            self._infos[info.key] = info

    # -- introspection -----------------------------------------------------------------

    def names(self) -> list[str]:
        """The declared top-level names (one per marker or artifact)."""
        return list(self._entries)

    def params(self) -> list[ParamInfo]:
        """Every parameter of the space, including ones under non-default alternatives."""
        return list(self._infos.values())

    def param(self, key: str) -> ParamInfo:
        return self._infos[key]

    def __contains__(self, key: str) -> bool:
        return key in self._infos

    def entry_name(self, key: str) -> str:
        """The declared name a parameter key belongs to (`manager.X.ratio` belongs to `manager`)."""
        return key.split(".", 1)[0]

    def display_path(self, key: str) -> str:
        """The path a diff shows for this parameter."""
        entry = self._entries.get(self.entry_name(key))
        if entry is not None and entry.artifact is not None:
            return entry.artifact.display_path()
        return f"<tunable:{key}>"

    # -- values ------------------------------------------------------------------------

    def defaults(self) -> dict[str, Any]:
        """The baseline candidate: every active-by-default parameter at its default."""
        return self.complete({})

    def is_active(self, key: str, values: Mapping[str, Any]) -> bool:
        """Whether `key` exists under the choices `values` makes. Conditions chain upward."""
        info = self._infos[key]
        while info.active_when is not None:
            condition_key, condition_value = info.active_when
            condition = self._infos[condition_key]
            if values.get(condition_key, condition.default) != condition_value:
                return False
            info = condition
        return True

    def active(self, values: Mapping[str, Any]) -> list[ParamInfo]:
        """The parameters that exist under the choices `values` makes."""
        return [info for key, info in self._infos.items() if self.is_active(key, values)]

    def complete(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Fill missing values from the defaults, following the choices the values make."""
        out: dict[str, Any] = {}
        for info in self._infos.values():
            merged = {**out, **values}
            if self.is_active(info.key, merged):
                out[info.key] = values.get(info.key, info.default)
        return out

    def validate(self, values: Mapping[str, Any]) -> None:
        """Every value is a known active parameter inside its domain, and none is missing.

        Raises:
            ValueError: On an unknown key, a value outside its domain, an inactive key for the
                choices made, or a missing active parameter.
        """
        unknown = sorted(set(values) - set(self._infos))
        if unknown:
            raise ValueError(f"not part of the search space: {unknown}")
        inactive = sorted(key for key in values if not self.is_active(key, values))
        if inactive:
            raise ValueError(f"not active under the chosen alternatives: {inactive}")
        active = {info.key for info in self.active(values)}
        missing = sorted(active - set(values))
        if missing:
            raise ValueError(f"missing values: {missing}")
        for key in values:
            self.validate_value(key, values[key])

    def validate_value(self, key: str, value: Any) -> None:
        """Check one value against its parameter's domain. Raises `ValueError` when invalid."""
        info = self._infos[key]
        if info.kind in ("float", "int"):
            number = float(value)
            if info.kind == "int" and number != int(number):
                raise ValueError(f"{key}: expected an integer, got {value!r}")
            if not (info.low <= number <= info.high):  # type: ignore[operator]
                raise ValueError(f"{key}: {value!r} is outside the allowed range [{info.low}, {info.high}]")
        elif info.kind in ("one_of", "choice", "switch"):
            if value not in info.options and str(value) not in [str(option) for option in info.options]:
                raise ValueError(f"{key}: {value!r} is not one of {list(info.options)}")
        elif info.kind == "text":
            text = str(value)
            cap = info.size_cap if info.size_cap is not None else _default_size_cap(str(info.default))
            if len(text) > cap:
                raise ValueError(f"{key}: value is {len(text)} characters, over the {cap} character cap")
            entry = self._entries.get(self.entry_name(key))
            if entry is not None and entry.artifact is not None:
                entry.artifact.validate(text)
        elif info.kind == "skills":
            json.loads(str(value))  # the listing must stay valid JSON

    # -- construction ------------------------------------------------------------------

    def resolve(self, name: str, values: Mapping[str, Any]) -> Any:
        """Build the real value for one declared marker name under `values`.

        A `choice` constructs the chosen component fresh (so stateful objects are never shared
        across rollouts); a `Switch` that is off yields its `off_value`; scalars and text pass
        through parsed.
        """
        entry = self._entries[name]
        marker = entry.marker
        if marker is None:
            raise ValueError(f"{name} is a file artifact, not a constructor argument")
        complete = self.complete(dict(values))
        return _resolve_marker(name, marker, complete)

    # -- workspaces --------------------------------------------------------------------

    def materialize(self, values: Mapping[str, Any], workspace: Path) -> Path:
        """Write the candidate's artifact values into a workspace copy and return its path."""
        if self.artifact_space is None:
            workspace.mkdir(parents=True, exist_ok=True)
            return workspace
        text_values = {name: str(values[name]) for name in self.artifact_space.names() if name in values}
        return self.artifact_space.materialize(text_values, workspace)


def _expand(name: str, marker: Any, active_when: tuple[str, Any] | None = None) -> list[ParamInfo]:
    """Expand one marker into its flat typed parameters."""
    if isinstance(marker, Float):
        return [
            ParamInfo(
                key=name,
                kind="int" if marker.is_int else "float",
                default=marker.default,
                low=marker.ge,
                high=marker.le,
                log=marker.log,
                active_when=active_when,
            )
        ]
    if isinstance(marker, OneOf):
        return [
            ParamInfo(
                key=name,
                kind="one_of",
                default=marker.default,
                options=tuple(marker.options),
                active_when=active_when,
                edit_instructions="One of: " + ", ".join(str(option) for option in marker.options),
            )
        ]
    if isinstance(marker, TextValue):
        return [
            ParamInfo(
                key=name,
                kind="text",
                default=marker.default,
                size_cap=marker.size_cap,
                edit_instructions=marker.edit_instructions,
                active_when=active_when,
            )
        ]
    if isinstance(marker, Choice):
        infos = [
            ParamInfo(
                key=name,
                kind="choice",
                default=marker.default.name,
                options=tuple(marker.by_name),
                active_when=active_when,
                edit_instructions="One of: " + ", ".join(marker.by_name),
            )
        ]
        for alternative in marker.by_name.values():
            for param_name, bound in alternative.tunables.items():
                infos.extend(
                    _expand(f"{name}.{alternative.name}.{param_name}", bound, active_when=(name, alternative.name))
                )
        return infos
    if isinstance(marker, Switch):
        infos = [
            ParamInfo(
                key=name,
                kind="switch",
                default=_SWITCH_ON if marker.default_on else _SWITCH_OFF,
                options=(_SWITCH_ON, _SWITCH_OFF),
                active_when=active_when,
            )
        ]
        infos.extend(_expand(f"{name}.{_SWITCH_ON}", marker.inner, active_when=(name, _SWITCH_ON)))
        return infos
    if isinstance(marker, SkillLibrary):
        members = sorted(str(p.relative_to(marker.path)) for p in Path(marker.path).rglob("SKILL.md"))
        return [
            ParamInfo(
                key=name,
                kind="skills",
                default=json.dumps(members),
                edit_instructions=marker.edit_instructions,
                active_when=active_when,
            )
        ]
    raise TypeError(f"{name}: cannot expand marker {type(marker).__name__}")


def _resolve_marker(key: str, marker: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(marker, Choice):
        chosen_name = str(values[key])
        if chosen_name not in marker.by_name:
            raise ValueError(f"{key}: unknown component type {chosen_name!r}; one of {sorted(marker.by_name)}")
        alternative = marker.by_name[chosen_name]
        params = {
            param_name: values[f"{key}.{chosen_name}.{param_name}"]
            for param_name in alternative.tunables
            if f"{key}.{chosen_name}.{param_name}" in values
        }
        return alternative.build(params)
    if isinstance(marker, Switch):
        if values[key] != _SWITCH_ON:
            return marker.off_value
        return _resolve_marker(f"{key}.{_SWITCH_ON}", marker.inner, values)
    if isinstance(marker, (Float, OneOf)):
        return marker.parse(values[key])
    if isinstance(marker, TextValue):
        return str(values[key])
    if isinstance(marker, SkillLibrary):
        return marker.path
    raise TypeError(f"{key}: cannot resolve marker {type(marker).__name__}")
