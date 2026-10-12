"""Immutable, content-addressed snapshots of artifact values.

A `Candidate` is one point in the search: the full set of artifact values plus its lineage (parent
id and the proposal rationale that produced it). Its id is a hash of the content, so identical
content always gets an identical id, and caching, resume, and audit key off that for free.
"""

import difflib
import hashlib
from pathlib import Path

from typing_extensions import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .artifacts import ArtifactSpace

_ID_LENGTH = 12


def _content_id(values: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(b"\x00")
        digest.update(values[name].encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:_ID_LENGTH]


class Candidate:
    """A snapshot of all artifact values, identified by the hash of its content.

    Attributes:
        values: Artifact name to value. Treated as immutable after construction.
        parent_id: Id of the candidate this one was mutated from, `None` for the seed.
        rationale: The proposer's written reason for the edit that produced this candidate.
        id: Content hash. Two candidates with identical values share an id.
    """

    def __init__(
        self,
        values: Mapping[str, str],
        *,
        parent_id: str | None = None,
        rationale: str | None = None,
        space: "ArtifactSpace | None" = None,
        baseline: "Candidate | None" = None,
        workspace_dir: Path | None = None,
    ):
        self.values: dict[str, str] = dict(values)
        self.parent_id = parent_id
        self.rationale = rationale
        self.id = _content_id(self.values)
        # Search context, injected by the optimizer so `materialize()` and `diff()` work bare.
        self._space = space
        self._baseline = baseline
        self._workspace_dir = workspace_dir

    def __getitem__(self, name: str) -> str:
        return self.values[name]

    @property
    def baseline(self) -> "Candidate | None":
        """The run's seed candidate this one is diffed against, when known."""
        return self._baseline

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Candidate) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Candidate(id={self.id!r}, parent_id={self.parent_id!r}, artifacts={sorted(self.values)})"

    def with_context(
        self,
        *,
        space: "ArtifactSpace | None" = None,
        baseline: "Candidate | None" = None,
        workspace_dir: Path | None = None,
    ) -> "Candidate":
        """Return a copy with search context attached. Values, id, and lineage are unchanged.

        The optimizer uses this to hand out candidates whose `materialize()` and `diff()` work
        bare; a `Candidate` itself is never mutated after construction.
        """
        return Candidate(
            self.values,
            parent_id=self.parent_id,
            rationale=self.rationale,
            space=space or self._space,
            baseline=baseline or self._baseline,
            workspace_dir=workspace_dir or self._workspace_dir,
        )

    def materialize(self, workspace: Path | None = None) -> Path:
        """Write this candidate's values into a workspace directory and return its path.

        Args:
            workspace: Destination directory. Defaults to the directory the optimizer assigned
                (`<run_dir>/candidates/<id>/artifacts`).
        """
        if self._space is None:
            raise ValueError("candidate has no ArtifactSpace; pass one via the optimizer or Candidate(space=...)")
        destination = workspace or self._workspace_dir
        if destination is None:
            raise ValueError("no workspace directory available; pass `workspace` explicitly")
        return self._space.materialize(self.values, Path(destination))

    def diff(self, baseline: "Candidate | None" = None) -> str:
        """Render a unified diff of this candidate against `baseline`.

        Args:
            baseline: Candidate to diff against. Defaults to the run's seed candidate.

        Returns:
            A unified diff across every changed artifact, empty string when nothing changed.
        """
        other = baseline or self._baseline
        if other is None:
            raise ValueError("no baseline candidate to diff against; pass one explicitly")
        parts: list[str] = []
        for name in sorted(self.values):
            old = other.values.get(name, "")
            new = self.values[name]
            if old == new:
                continue
            path = self._space[name].display_path() if self._space and name in self._space else name
            # difflib emits no separator after a line that lacks a trailing newline, which glues
            # the removed and added text together for single-line values; normalize both sides.
            old_lines = [line if line.endswith("\n") else line + "\n" for line in old.splitlines()]
            new_lines = [line if line.endswith("\n") else line + "\n" for line in new.splitlines()]
            lines = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
            parts.append("".join(lines))
        return "\n".join(parts)

    def mutate(self, artifact_name: str, new_value: str, *, rationale: str | None = None) -> "Candidate":
        """Return a child candidate with one artifact value replaced."""
        values = dict(self.values)
        values[artifact_name] = new_value
        return Candidate(
            values,
            parent_id=self.id,
            rationale=rationale,
            space=self._space,
            baseline=self._baseline or self,
        )
