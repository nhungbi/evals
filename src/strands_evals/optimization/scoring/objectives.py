"""Objectives, constraints, and turning trials into scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...evaluators.evaluator import Evaluator
from ...types.optimization import Splits, SplitScores, TrialRecord
from .metrics import MetricRegistry

MetricTarget = str | Evaluator | type[Evaluator]


@dataclass(frozen=True, init=False)
class _Direction:
    """One metric to maximize or minimize; `weight` and `scale` are read only inside `WeightedSum`."""

    metric: str
    weight: float
    scale: float | None

    def __init__(self, target: MetricTarget, /, *, weight: float = 1.0, scale: float | None = None):
        """Name the metric; an evaluator instance resolves via `get_name()`, a class via `get_type_name()`.

        Names are checked against the run's evaluators when the `Optimizer` is built.

        Args:
            target: A metric name, an evaluator class, or an evaluator instance.
            weight: Relative importance inside `WeightedSum`.
            scale: The size of a meaningful change, so metrics in different units combine; None means
                1.0 for an evaluator score. A usage metric inside `WeightedSum` needs it.

        Raises:
            ValueError: `weight` or `scale` is not > 0.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"Minimize(tokens, scale=1000)"`; stored in `run.json`."""
        raise NotImplementedError


class Maximize(_Direction):
    """Maximize a metric, e.g. `Maximize("Helpfulness")`."""


class Minimize(_Direction):
    """Minimize a metric, e.g. `Minimize("tokens")`."""


@dataclass(frozen=True, init=False)
class WeightedSum:
    """Several objectives as one number: per case, the weighted mean of each oriented value / scale.

    `WeightedSum(Maximize("Helpfulness"), Minimize("tokens", scale=1000))` makes 1,000 tokens worth
    as much as the whole 0-1 Helpfulness range. Per case, N/A evaluator terms drop and the remaining
    evaluator weights are rescaled to the same total; usage terms are never rescaled. All evaluator
    terms N/A, or any usage term unmeasured, leaves the case unmeasured.
    """

    objectives: tuple[Maximize | Minimize, ...]

    def __init__(self, *objectives: Maximize | Minimize):
        """Combine objectives.

        Args:
            *objectives: At least one `Maximize` or `Minimize`.

        Raises:
            TypeError: No objectives, or one that isn't a `Maximize` / `Minimize`.
            ValueError: The same metric twice, or a usage metric without `scale=`.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"WeightedSum(Maximize(Helpfulness), Minimize(tokens, scale=1000))"`; stored in `run.json`."""
        raise NotImplementedError


Objective = Maximize | Minimize | WeightedSum


@dataclass(frozen=True, init=False)
class _Bound:
    """A bound on one metric's mean that can only reject."""

    metric: str
    bound: float

    def __init__(self, target: MetricTarget, bound: float, /):
        """Name the metric as in `Maximize`.

        Args:
            target: A metric name, an evaluator class, or an evaluator instance.
            bound: The limit on the mean.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """E.g. `"tokens <= 4000"`; stored in `run.json`."""
        raise NotImplementedError


class AtMost(_Bound):
    """Rejects when the metric's mean exceeds `bound`, e.g. `AtMost("tokens", 4000)`."""


class AtLeast(_Bound):
    """Rejects when the metric's mean falls below `bound`, e.g. `AtLeast("Harmfulness", 0.95)`."""


Constraint = AtMost | AtLeast


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
        """The objective for one case or run, always higher-is-better (`Minimize` negates).

        A lone `Maximize` / `Minimize` gives the metric's value as is; `WeightedSum` its weighted mean.
        """
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
