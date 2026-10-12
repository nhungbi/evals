"""Artifact adapters: the named text values an optimizer is allowed to edit.

An `Artifact` names one optimizable text value and knows how to read its baseline and write a
proposed value into a per-candidate workspace. The customer's original files are never written.

Three adapters cover the common agent surface:

- `TextArtifact`: an in-memory string (an `instructions` parameter, a tool description).
- `FileArtifact`: a file (`SKILL.md`, `AGENTS.md`, a policy file). Writes go to the workspace copy.
- `JsonFieldArtifact`: a JSON-pointer field inside a config file, so one field can be edited while
  the rest of the config stays intact.

An `ArtifactSpace` is the named collection the proposer may touch.
"""

import copy
import json
import shutil
from pathlib import Path

import yaml
from typing_extensions import Any, Mapping, Protocol, runtime_checkable

from .formulas.formula import Formula

_DEFAULT_SIZE_CAP_FLOOR = 2_000
_DEFAULT_SIZE_CAP_MULTIPLIER = 3


def _default_size_cap(baseline: str) -> int:
    """Cap proposed values at a multiple of the baseline size.

    Prompt bloat is the best-documented failure mode of naive prompt optimization, so every
    artifact carries a hard cap. The floor keeps short baselines editable at all.
    """
    return max(_DEFAULT_SIZE_CAP_FLOOR, _DEFAULT_SIZE_CAP_MULTIPLIER * len(baseline))


def _parse_json_pointer(pointer: str) -> list[str]:
    """Split an RFC 6901 JSON pointer such as `/instructions` into reference tokens."""
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def _resolve_pointer(document: Any, tokens: list[str]) -> Any:
    node = document
    for token in tokens:
        if isinstance(node, list):
            node = node[int(token)]
        else:
            node = node[token]
    return node


def _set_pointer(document: Any, tokens: list[str], value: Any) -> None:
    parent = _resolve_pointer(document, tokens[:-1]) if len(tokens) > 1 else document
    last = tokens[-1]
    if isinstance(parent, list):
        parent[int(last)] = value
    else:
        parent[last] = value


def _validate_frontmatter(value: str) -> None:
    """Require a well-formed YAML frontmatter block when the value opens with one."""
    if not value.startswith("---\n"):
        return
    end = value.find("\n---", 4)
    if end < 0:
        raise ValueError("frontmatter block is not closed with '---'")
    yaml.safe_load(value[4:end])


@runtime_checkable
class Artifact(Protocol):
    """One optimizable text value: a baseline to read and a workspace location to write."""

    name: str
    edit_instructions: str | None
    size_cap: int | None

    def read(self) -> str:
        """Return the baseline value of this artifact."""
        ...

    def apply(self, value: str, workspace: Path) -> None:
        """Write `value` into the candidate workspace. Never touches the original."""
        ...

    def display_path(self) -> str:
        """The path used in rendered diffs."""
        ...

    def validate(self, value: str) -> None:
        """Raise `ValueError` when `value` violates this artifact's format constraints."""
        ...


class _BaseArtifact:
    """Shared guardrails: size cap and name bookkeeping."""

    def __init__(self, *, edit_instructions: str | None = None, size_cap: int | None = None):
        self.name = ""  # assigned once, by the ArtifactSpace that owns this artifact
        self.edit_instructions = edit_instructions
        self.size_cap = size_cap

    def _effective_size_cap(self) -> int:
        return self.size_cap if self.size_cap is not None else _default_size_cap(self.read())

    def read(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def validate(self, value: str) -> None:
        cap = self._effective_size_cap()
        if len(value) > cap:
            raise ValueError(f"value is {len(value)} characters, over the {cap} character cap")


class _PathArtifact(_BaseArtifact):
    """Shared path resolution for artifacts backed by a file."""

    def __init__(self, path: str | Path, *, edit_instructions: str | None = None, size_cap: int | None = None):
        super().__init__(edit_instructions=edit_instructions, size_cap=size_cap)
        self.path = Path(path)
        self._root: Path | None = None

    def bind_root(self, root: Path) -> None:
        """Resolve this artifact's relative path against the owning space's `root`.

        Called by `ArtifactSpace`. Rebinding to a different root raises, because it means the
        same artifact instance was registered in two spaces and reads would silently move.
        """
        if self._root is not None and self._root != root:
            raise ValueError(
                f"artifact {self.name or self.path} is already bound to root {self._root}, cannot rebind to {root}"
            )
        self._root = root

    def _source_path(self) -> Path:
        if self.path.is_absolute() or self._root is None:
            return self.path
        return self._root / self.path

    def _relative_path(self) -> Path:
        if not self.path.is_absolute():
            return self.path
        if self._root is not None and self.path.is_relative_to(self._root):
            return self.path.relative_to(self._root)
        return Path(self.path.name)


class TextArtifact(_BaseArtifact):
    """An in-memory string with no file behind it.

    The optimized value flows through `Candidate.values`; the caller's `agent_factory` reads it
    from there (for example to pass as a tool `description`). `apply` therefore writes nothing.
    """

    def __init__(self, value: str, *, edit_instructions: str | None = None, size_cap: int | None = None):
        super().__init__(edit_instructions=edit_instructions, size_cap=size_cap)
        self._value = value

    def read(self) -> str:
        return self._value

    def apply(self, value: str, workspace: Path) -> None:
        return None

    def display_path(self) -> str:
        return f"<{self.name}>"


class FileArtifact(_PathArtifact):
    """A file whose full text is optimizable, such as a `SKILL.md` or `AGENTS.md`.

    `path` is resolved against the space's `root` when one is set. `apply` writes the value to the
    same relative path inside the candidate workspace.
    """

    def read(self) -> str:
        return self._source_path().read_text()

    def apply(self, value: str, workspace: Path) -> None:
        target = workspace / self._relative_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)

    def display_path(self) -> str:
        return str(self._relative_path())

    def validate(self, value: str) -> None:
        super().validate(value)
        if self._source_path().suffix == ".md" and self.read().startswith("---\n"):
            _validate_frontmatter(value)


class JsonFieldArtifact(_PathArtifact):
    """One field inside a JSON config file, addressed by an RFC 6901 pointer.

    `read` returns the field as text: the raw string when the field holds a string, otherwise its
    JSON encoding. `apply` rewrites only that field in the workspace copy of the file.
    """

    def __init__(
        self,
        path: str | Path,
        pointer: str,
        *,
        edit_instructions: str | None = None,
        size_cap: int | None = None,
    ):
        super().__init__(path, edit_instructions=edit_instructions, size_cap=size_cap)
        self.pointer = pointer
        self._tokens = _parse_json_pointer(pointer)

    def _baseline_is_string(self) -> bool:
        document = json.loads(self._source_path().read_text())
        return isinstance(_resolve_pointer(document, self._tokens), str)

    def read(self) -> str:
        document = json.loads(self._source_path().read_text())
        field = _resolve_pointer(document, self._tokens)
        return field if isinstance(field, str) else json.dumps(field, indent=2)

    def apply(self, value: str, workspace: Path) -> None:
        target = workspace / self._relative_path()
        source = target if target.exists() else self._source_path()
        document = json.loads(source.read_text())
        _set_pointer(document, self._tokens, value if self._baseline_is_string() else json.loads(value))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2) + "\n")

    def display_path(self) -> str:
        return f"{self._relative_path()}#{self.pointer}"

    def validate(self, value: str) -> None:
        super().validate(value)
        if not self._baseline_is_string():
            try:
                json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"field {self.pointer} holds JSON, and the proposed value is not valid JSON") from e


class ArtifactSpace:
    """The named collection of artifacts the proposer is allowed to touch.

    Args:
        artifacts: The optimizable artifacts, keyed by the name the proposer will use. Names
            that are not Python identifiers are only expressible through this mapping form.
        root: Optional directory copied wholesale into each candidate workspace before artifact
            values are applied, so relative-path artifacts and their sibling files (a config next
            to its skills directory) travel together.
        **named: Artifacts as keyword arguments, merged with `artifacts`.

    An artifact instance belongs to one space: registering it again under a different name, or
    under a root that moves its path, raises instead of silently rebinding.

    Example:
        space = ArtifactSpace(
            root=Path("my-agent"),
            instructions=JsonFieldArtifact("agent.json", "/instructions"),
            triage_skill=FileArtifact(".agent/skills/triage/SKILL.md"),
        )
    """

    def __init__(
        self,
        artifacts: Mapping[str, Artifact] | None = None,
        *,
        root: str | Path | None = None,
        **named: Artifact,
    ):
        if isinstance(artifacts, (str, Path)):
            raise TypeError("the first argument is the artifacts mapping; pass the workspace directory as root=...")
        merged = dict(artifacts or {})
        overlap = merged.keys() & named.keys()
        if overlap:
            raise ValueError(f"artifacts given twice, as mapping and keyword: {sorted(overlap)}")
        merged.update(named)
        if not merged:
            raise ValueError("ArtifactSpace needs at least one artifact")
        self.root = Path(root) if root is not None else None
        self.artifacts: dict[str, Artifact] = {}
        for name, artifact in merged.items():
            adapt = getattr(artifact, "to_artifact", None)
            if callable(adapt):  # declaration markers (Choice/Component/Float/Int/TextValue)
                artifact = adapt()
            if artifact.name and artifact.name != name:
                raise ValueError(f"artifact is already named {artifact.name!r}, cannot register it as {name!r}")
            artifact.name = name
            if self.root is not None and isinstance(artifact, _PathArtifact):
                artifact.bind_root(self.root)
            self.artifacts[name] = artifact

    def __getitem__(self, name: str) -> Artifact:
        return self.artifacts[name]

    def __contains__(self, name: str) -> bool:
        return name in self.artifacts

    def names(self) -> list[str]:
        return list(self.artifacts)

    def baseline_values(self) -> dict[str, str]:
        """Read every artifact's current value: the seed candidate."""
        return {name: artifact.read() for name, artifact in self.artifacts.items()}

    def materialize(self, values: dict[str, str], workspace: Path) -> Path:
        """Copy `root` (when set) into `workspace` and apply `values` on top of it."""
        workspace.mkdir(parents=True, exist_ok=True)
        if self.root is not None:
            shutil.copytree(self.root, workspace, dirs_exist_ok=True)
        for name, artifact in self.artifacts.items():
            artifact.apply(values[name], workspace)
        return workspace

    def validate(self, name: str, value: str) -> None:
        """Guardrail check for one proposed edit. Raises `ValueError` when it must be rejected."""
        if name not in self.artifacts:
            raise ValueError(f"artifact {name!r} is not in the optimizable space {self.names()}")
        self.artifacts[name].validate(value)


class FormulaArtifact(_BaseArtifact):
    """Adapts one tunable parameter of a `Formula` into an optimizable artifact.

    The Formula stays the runtime application mechanism; this wrapper exposes one of its tunable
    params as a named, guarded, diffable value in the artifact space. String params pass through
    as-is; dict or list params are serialized as JSON for the proposer and parsed back on apply.
    The task factory reads `candidate[name]` and calls `apply_to_formula` before building the
    agent, so like `TextArtifact` this is a runtime artifact and `apply` writes nothing to disk.
    """

    def __init__(
        self,
        formula: Formula,
        param_key: str,
        *,
        edit_instructions: str | None = None,
        size_cap: int | None = None,
    ):
        super().__init__(edit_instructions=edit_instructions, size_cap=size_cap)
        self.formula = formula
        self.param_key = param_key
        self._baseline_is_str = isinstance(formula.get_tunable_params()[param_key], str)

    def read(self) -> str:
        value = self.formula.get_tunable_params()[self.param_key]
        return value if isinstance(value, str) else json.dumps(value, indent=1)

    def apply(self, value: str, workspace: Path) -> None:
        return None

    def display_path(self) -> str:
        return f"<formula:{self.param_key}>"

    def validate(self, value: str) -> None:
        super().validate(value)
        parsed = self.to_params(value)
        # Probe the Formula's own validation on a copy, so a strict Formula rejects bad edits here.
        probe = copy.deepcopy(self.formula)
        probe.update_params({self.param_key: parsed})

    def to_params(self, value: str) -> Any:
        """Parse a candidate value back into the Formula's param shape."""
        if self._baseline_is_str:
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"{self.param_key} holds structured data, and the value is not valid JSON") from e

    def apply_to_formula(self, value: str) -> None:
        """Push a candidate value into the live Formula (the task factory calls this per build)."""
        self.formula.update_params({self.param_key: self.to_params(value)})
