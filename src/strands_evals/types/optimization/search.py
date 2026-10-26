"""What search methods return and are told."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .trials import Usage

Value = str | int | float | bool | None

ObservationOutcome = Literal["accepted", "rejected", "admitted", "dominated", "infeasible", "unscored", "duplicate"]


class Patch(BaseModel):
    """One proposed change to the parent: only the keys to change, not a full candidate.

    `state` is the method's private memory, handed back in `observe` and on resume; it must be
    JSON-serializable. `usage` is the proposer's own model cost, reported in `search_cost` and
    never counted against the rollout budget. No proposer name: the optimizer knows which block
    asked, and records it as the candidate's `origin`.
    """

    model_config = ConfigDict(frozen=True)

    updates: dict[str, Value]
    rationale: str
    state: Any = None
    usage: Usage | None = None


class ReflectionRecord(BaseModel):
    """One of the parent's worst train-split cases, as shown to a proposer."""

    model_config = ConfigDict(frozen=True)

    case: str
    score: float
    input: str
    output: str
    feedback: list[str]
    trajectory: str | None


class Observation(BaseModel):
    """Feedback after a patch was handled; told exactly once per patch.

    `scalar` is the selection-split objective, None if never measured. `reason` comes from
    selection-split data only; gate outcomes say only "rejected at the holdout gate".
    """

    model_config = ConfigDict(frozen=True)

    patch: Patch
    scalar: float | None
    outcome: ObservationOutcome
    reason: str
