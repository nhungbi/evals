"""Per-iteration result records, read by `on_iteration`, resume, the schedule, and the results API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .search import ObservationOutcome


class ProposalOutcome(BaseModel):
    """One proposal of one iteration.

    `number` is the display number (None for an invalid patch); `block` is the proposing block's name;
    `status` uses the same vocabulary as `Observation.outcome`; `train_score` is the objective it was
    observed with (`Observation.objective`).
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str | None
    number: int | None
    parent_id: str
    block: str
    changed_keys: list[str]
    rationale: str
    status: ObservationOutcome
    train_score: float | None


class IterationRecord(BaseModel):
    """One iteration. `round`, `role`, and `blocks` let `Schedule.next` rebuild where a `Nested` run is.

    `round` is the round path, outermost `Nested` first (`Step.round`); None outside any `Nested`.
    """

    model_config = ConfigDict(frozen=True)

    iteration: int
    parent_id: str
    round: tuple[int, ...] | None
    role: Literal["opener", "inner"]
    blocks: list[str]
    proposals: list[ProposalOutcome]
    rollouts_spent: int
