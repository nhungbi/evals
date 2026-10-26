"""The `Method` base class and its request: proposes values, never decides what is accepted."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ...types.optimization import Observation, Patch, ReflectionRecord
from ..candidates.candidate import Candidate
from ..declaration.space import ParamKind, Space


@dataclass(frozen=True)
class ProposalRequest:
    """Everything a method may see when proposing; never holdout data.

    A dataclass, not a record: it holds the space, the parent, and a callback, and is never saved.
    `log(event, **fields)` writes a lineage event; a method writes only `screened` and
    `proposer_error` (any other name raises `ValueError`). `attempt` is the redraw number within this
    iteration and block (0 = first ask).
    """

    space: Space
    keys: Sequence[str]
    parent: Candidate
    records: Sequence[ReflectionRecord]
    n: int
    iteration: int
    seed: int
    attempt: int
    log: Callable[..., None]


class Method(ABC):
    """Proposes patches on a parent and learns from how each one did.

    `kinds` are the variable types it can propose. `uses_records=True` means the method reads the
    parent's train-split records, so the parent is scored on the train split before `propose` if it
    has none. An ABC because users subclass it.
    """

    kinds: ClassVar[frozenset[ParamKind]]
    uses_records: ClassVar[bool] = False

    @abstractmethod
    async def propose(self, request: ProposalRequest) -> list[Patch]:
        """Return up to `request.n` patches; `[]` means nothing useful this time (a redraw cause).

        Raising means the method itself failed and ends the run with `stop_reason="proposer_error"`.
        """

    @abstractmethod
    def observe(self, observation: Observation) -> None:
        """Called exactly once per patch this method returned."""

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json` and compared on resume.

        Override it so a changed setting is caught on resume; the default `{}` checks nothing.
        """
        return {}
