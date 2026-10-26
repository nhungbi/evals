"""Scores, splits, and verdicts for the optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Splits(BaseModel):
    """The run's one train / selection / holdout split; every split holds at least one case."""

    model_config = ConfigDict(frozen=True)

    train: tuple[str, ...]
    selection: tuple[str, ...]
    holdout: tuple[str, ...]


class CandidateScores(BaseModel):
    """One candidate's scores on one split.

    `per_case` is the mean of that case's measured per-run scalars; a run's scalar is measured only
    when its status is completed. `per_metric` holds per-metric means (usage over completed and
    agent_failure runs), used by constraints.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    per_case: dict[str, float]
    per_metric: dict[str, dict[str, float]]
    per_run: dict[str, list[float | None]]
    failures: int

    def mean(self, cases: Sequence[str]) -> float | None:
        """Mean per-case score over the given cases that were measured; None if none were."""
        raise NotImplementedError

    def subset(self, cases: Sequence[str]) -> dict[str, float]:
        """Per-case scores restricted to the given cases that were measured."""
        raise NotImplementedError


class SplitScores(BaseModel):
    """One `CandidateScores` per split, never mixed; a split not scored yet is empty."""

    model_config = ConfigDict(frozen=True)

    train: CandidateScores
    selection: CandidateScores
    holdout: CandidateScores


class PairedComparison(BaseModel):
    """Paired bootstrap result on the cases both candidates measured."""

    model_config = ConfigDict(frozen=True)

    mean_diff: float
    ci_low: float
    ci_high: float
    n_cases: int


class AcceptanceVerdict(BaseModel):
    """The holdout gate's decision for one challenger."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    reason: str
    comparison: PairedComparison | None
    violations: list[str]
    coverage: float


class Admission(BaseModel):
    """Admission on the selection split.

    `infeasible`: a constraint or coverage failed. `dominated`: the selection strategy said no.
    `cases_led` is None unless admitted; computed by `admit` for every strategy, possibly `[]`.
    """

    model_config = ConfigDict(frozen=True)

    outcome: Literal["admitted", "infeasible", "dominated"]
    reason: str
    violations: list[str]
    cases_led: list[str] | None
