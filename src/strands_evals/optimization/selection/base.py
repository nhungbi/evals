"""The `Selection` base class and its context: picks the parent, never the champion."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ...types.optimization import SelectionStep, Value


@dataclass(frozen=True)
class SelectionContext:
    """The pool as selection sees it: selection-split scores only.

    A dataclass, not a record: it holds the RNG and callables, and is never saved.

    Attributes:
        scores: Candidate id -> case name -> selection-split score.
        values: Candidate id -> key -> value.
        parents: Candidate id -> parent ids.
        champion_id: The current champion.
        selection_cases: Every selection-split case name.
        rng: Seeded per iteration.
        merges_done: Merges so far in this run.
        merged_pairs: Pairs already merged, each tried once.
        validate: `space.validate`.
        name_of: Returns a key's declared name (the merge unit).
    """

    scores: Mapping[str, Mapping[str, float]]
    values: Mapping[str, Mapping[str, Value]]
    parents: Mapping[str, tuple[str, ...]]
    champion_id: str
    selection_cases: Sequence[str]
    rng: random.Random
    merges_done: int
    merged_pairs: frozenset[frozenset[str]]
    validate: Callable[[Mapping[str, Value]], None]
    name_of: Callable[[str], str]


class Selection(ABC):
    """Decides the parent each iteration (or a merge) and which proposals join the pool.

    Not to be confused with `Search`, which decides which method changes which variables.
    An ABC because users subclass it (like `Method`); internal seams such as `TrialCache` are
    `Protocol`s.

    Attributes:
        name: Stable strategy name, stored in `run.json` with `config()` and compared on resume.
    """

    name: ClassVar[str]

    @abstractmethod
    def choose(self, ctx: SelectionContext) -> SelectionStep:
        """The parent, or a merge; an id outside the pool makes the optimizer raise `ValueError`."""

    @abstractmethod
    def admits(self, candidate_id: str, scores: Mapping[str, float], ctx: SelectionContext) -> bool:
        """Whether the candidate joins the pool; it already passed constraints and coverage.

        Which cases it leads doesn't depend on the strategy, so the optimizer computes that itself
        (`cases_led`).
        """

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json` and compared on resume."""
        return {}
