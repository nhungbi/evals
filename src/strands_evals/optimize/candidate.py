"""Immutable, content-addressed candidates over typed values.

A `Candidate` is one point in the search: a flat typed mapping of parameter key to value, plus
its lineage (parent id and the rationale that produced it). The typed mapping is canonical; the
content id (hash of its canonical JSON) and the diff are derived views, so caching, resume, and
review all key off the same values that numeric search reads.
"""

import difflib
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .space import Space

_ID_LENGTH = 12


def content_id(values: Mapping[str, Any]) -> str:
    """The candidate id: a hash of the canonical JSON serialization of the typed values."""
    canonical = json.dumps(dict(values), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:_ID_LENGTH]


class Candidate:
    """A typed snapshot of every parameter value, identified by the hash of its content.

    Attributes:
        values: Parameter key to typed value (number, string, or option). Immutable after
            construction.
        parent_id: Id of the candidate this one was mutated from, `None` for the baseline.
        rationale: The written reason for the edit that produced this candidate.
        id: Content hash. Two candidates with identical values share an id.
    """

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        parent_id: str | None = None,
        rationale: str | None = None,
        space: "Space | None" = None,
        baseline: "Candidate | None" = None,
        workspace_dir: Path | None = None,
    ):
        self.values: dict[str, Any] = dict(values)
        self.parent_id = parent_id
        self.rationale = rationale
        self.id = content_id(self.values)
        # Search context, injected by the optimizer so `materialize()` and `diff()` work bare.
        self._space = space
        self._baseline = baseline
        self._workspace_dir = workspace_dir

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    @property
    def baseline(self) -> "Candidate | None":
        """The run's baseline candidate this one is diffed against, when known."""
        return self._baseline

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Candidate) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Candidate(id={self.id!r}, parent_id={self.parent_id!r}, keys={sorted(self.values)})"

    def with_context(
        self,
        *,
        space: "Space | None" = None,
        baseline: "Candidate | None" = None,
        workspace_dir: Path | None = None,
    ) -> "Candidate":
        """Return a copy with search context attached. Values, id, and lineage are unchanged."""
        return Candidate(
            self.values,
            parent_id=self.parent_id,
            rationale=self.rationale,
            space=space or self._space,
            baseline=baseline or self._baseline,
            workspace_dir=workspace_dir or self._workspace_dir,
        )

    def mutate(self, updates: Mapping[str, Any], *, rationale: str | None = None) -> "Candidate":
        """Return a child candidate with the given parameter values replaced.

        Values made inactive by a changed choice are dropped, and values the new choice needs are
        filled at their defaults, so a child is always a complete, valid point of the space.
        """
        if self._space is None:
            merged = {**self.values, **updates}
        else:
            kept = {k: v for k, v in self.values.items() if self._space.is_active(k, {**self.values, **updates})}
            merged = self._space.complete({**kept, **updates})
        return Candidate(
            merged,
            parent_id=self.id,
            rationale=rationale,
            space=self._space,
            baseline=self._baseline or self,
        )

    def materialize(self, workspace: Path | None = None) -> Path:
        """Write this candidate's artifact values into a workspace directory and return its path."""
        if self._space is None:
            raise ValueError("candidate has no Space; pass one via the optimizer or Candidate(space=...)")
        destination = workspace or self._workspace_dir
        if destination is None:
            raise ValueError("no workspace directory available; pass `workspace` explicitly")
        return self._space.materialize(self.values, Path(destination))

    def diff(self, baseline: "Candidate | None" = None) -> str:
        """Render a reviewable diff of this candidate against `baseline`.

        Text parameters render as a unified diff; numbers and choices render as one-line
        before/after pairs under the same file-style headers.
        """
        other = baseline or self._baseline
        if other is None:
            raise ValueError("no baseline candidate to diff against; pass one explicitly")
        parts: list[str] = []
        for key in sorted(set(self.values) | set(other.values)):
            old = other.values.get(key)
            new = self.values.get(key)
            if old == new:
                continue
            path = self._space.display_path(key) if self._space and key in self._space else key
            kind = self._space.param(key).kind if self._space and key in self._space else "text"
            if kind in ("text", "skills"):
                old_lines = _diff_lines(str(old) if old is not None else "")
                new_lines = _diff_lines(str(new) if new is not None else "")
                parts.append(
                    "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"))
                )
            else:
                old_text = f"{key} = {old!r}" if key in other.values else f"{key} (absent under those choices)"
                new_text = f"{key} = {new!r}" if key in self.values else f"{key} (absent under these choices)"
                parts.append(f"--- a/{path}\n+++ b/{path}\n-{old_text}\n+{new_text}\n")
        return "\n".join(part for part in parts if part)


def _diff_lines(text: str) -> list[str]:
    # difflib emits no separator after a line that lacks a trailing newline, which glues the
    # removed and added text together for single-line values; normalize every line.
    return [line if line.endswith("\n") else line + "\n" for line in text.splitlines()]
