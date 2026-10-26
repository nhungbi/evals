"""Objectives, constraints, and turning trials into scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...evaluators.evaluator import Evaluator
from ...types.optimization import Splits, SplitScores, TrialRecord
from .metrics import MetricRegistry


class MetricExpr:
    """A linear combination of metrics, e.g. `0.8*metric("Helpfulness") - 0.0001*metric("tokens")`.

    Supports `+`/`-` with another expression, `*`/`/` by a number, and unary `-`; anything else
    raises `TypeError`. Per case, N/A evaluator terms drop and the remaining evaluator weights are
    rescaled to the same total |weight|; usage terms are never rescaled. All evaluator terms N/A, or
    any usage term unmeasured, leaves the case unmeasured.
    """

    terms: tuple[tuple[str, float], ...]

    def __init__(self, terms: tuple[tuple[str, float], ...]):
        """`terms` are (metric name as written, weight) pairs; built by `metric()` and the operators."""
        raise NotImplementedError

    def __add__(self, other: MetricExpr) -> MetricExpr:
        raise NotImplementedError

    def __sub__(self, other: MetricExpr) -> MetricExpr:
        raise NotImplementedError

    def __mul__(self, weight: float) -> MetricExpr:
        raise NotImplementedError

    def __rmul__(self, weight: float) -> MetricExpr:
        raise NotImplementedError

    def __truediv__(self, divisor: float) -> MetricExpr:
        raise NotImplementedError

    def __neg__(self) -> MetricExpr:
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"0.8*Helpfulness - 0.0001*tokens"`; stored in the record."""
        raise NotImplementedError


@dataclass(frozen=True)
class Objective:
    """What to optimize, returned by `Maximize` / `Minimize`."""

    expr: MetricExpr
    direction: Literal["maximize", "minimize"]

    def __str__(self) -> str:
        """E.g. `"Maximize(0.8*Helpfulness - 0.0001*tokens)"`; stored in `run.json`."""
        raise NotImplementedError


@dataclass(frozen=True)
class Constraint:
    """A bound that can only reject, returned by `AtMost` / `AtLeast`."""

    expr: MetricExpr
    op: Literal["<=", ">="]
    bound: float

    def __str__(self) -> str:
        """E.g. `"tokens <= 4000"`; stored in `run.json`."""
        raise NotImplementedError


def metric(name: str | Evaluator | type[Evaluator]) -> MetricExpr:
    """An evaluator score or usage metric.

    An evaluator instance resolves via `get_name()` (its `name=` or class name); a class via
    `get_type_name()`. Names are checked against the run's evaluators when the `Optimizer` is built.
    """
    raise NotImplementedError


def Maximize(target: str | MetricExpr) -> Objective:
    raise NotImplementedError


def Minimize(target: str | MetricExpr) -> Objective:
    raise NotImplementedError


def AtMost(target: str | MetricExpr, bound: float) -> Constraint:
    """A constraint that rejects when the mean exceeds `bound`.

    Args:
        target: A metric name or a `metric(...)` expression.
        bound: The largest allowed mean.
    """
    raise NotImplementedError


def AtLeast(target: str | MetricExpr, bound: float) -> Constraint:
    """A constraint that rejects when the mean falls below `bound`.

    Args:
        target: A metric name or a `metric(...)` expression.
        bound: The smallest allowed mean.
    """
    raise NotImplementedError


class Criteria:
    """The objective plus the constraints; names resolve through the registry at construction."""

    def __init__(self, objective: Objective | None, constraints: Sequence[Constraint], registry: MetricRegistry):
        """`objective=None` means maximize the mean of every evaluator's score.

        Args:
            objective: What to optimize; None means the mean of every evaluator's score.
            constraints: Bounds that can only reject.
            registry: Resolves the metric names.

        Raises:
            ValueError: A metric name is unknown or ambiguous.
        """
        raise NotImplementedError

    def scalar(self, metrics: Mapping[str, float]) -> float | None:
        """The objective for one case or run, always higher-is-better (`Minimize` negates)."""
        raise NotImplementedError

    def violations(self, per_case: Mapping[str, Mapping[str, float]]) -> list[str]:
        """One message per violated constraint; a constraint never measured counts as violated.

        `per_case` maps case name -> metric name -> value.
        """
        raise NotImplementedError


def score(candidate_id: str, records: Sequence[TrialRecord], criteria: Criteria, splits: Splits) -> SplitScores:
    """Group records by split and average per case.

    A run's scalar is measured only when its status is completed, so a crash never scores.

    Args:
        candidate_id: The candidate.
        records: Its trial records on any splits.
        criteria: Computes each run's scalar.
        splits: The run's splits.

    Returns:
        One `CandidateScores` per split; a split not scored yet is empty.
    """
    raise NotImplementedError
