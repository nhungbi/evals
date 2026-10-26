"""Per-iteration result records, read by `on_iteration`, resume, and the results API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .scores import AcceptanceVerdict
from .search import ObservationOutcome


class ProposalOutcome(BaseModel):
    """One proposal of one iteration.

    `holdout_score` and `verdict` are None unless the proposal was the iteration's challenger.
    `block` is the proposing block's name (None for a merge); `status` uses the same vocabulary as
    `Observation.outcome`.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str | None
    parent_ids: tuple[str, ...]
    block: str | None
    changed_keys: list[str]
    rationale: str
    status: ObservationOutcome
    selection_score: float | None
    holdout_score: float | None
    verdict: AcceptanceVerdict | None


class IterationRecord(BaseModel):
    """One iteration; `parent_ids` holds one parent, or two for a merge."""

    model_config = ConfigDict(frozen=True)

    iteration: int
    champion_id: str
    parent_ids: tuple[str, ...]
    step: Literal["propose", "merge"]
    proposals: list[ProposalOutcome]
    rollouts_spent: int
