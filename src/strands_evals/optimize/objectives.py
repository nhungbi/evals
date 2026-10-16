"""The scoring layer: objectives and constraints over per-case metrics.

A metric is an evaluator's score (named by the evaluator), a usage metric from the execution
layer (`tokens`, `seconds`, `model_calls`, ...), or a `Derived` metric computed from others. The
case is the unit everywhere: repetitions average within a case first, and the gate compares
paired per-case scalars.

When no objective is given, the objective is: maximize the mean of all evaluator scores, which
keeps the zero-ceremony path identical to a plain eval run. Combiners that reduce several
objectives to one scalar (`WeightedSum`, and later `Chebyshev` / `Lexicographic`) carry the full
acceptance guarantee; `Pareto` is a reporting mode and does not gate.

One invariant holds without exception: a constraint can only reject a candidate, never accept
one. Violations are checked on the selection split at admission (so a violator never joins the
pool) and on the held-out split at the gate, the same cases as the objective there.
"""

import logging
from dataclasses import dataclass
from statistics import fmean
from typing import Callable, Mapping

from .execution import USAGE_METRICS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Objective:
    """One goal: a metric name and the direction that counts as better."""

    metric: str
    direction: int  # +1 maximize, -1 minimize

    def scalar(self, case_metrics: Mapping[str, float]) -> float | None:
        value = case_metrics.get(self.metric)
        return None if value is None else self.direction * value


def Maximize(metric: str) -> Objective:  # noqa: N802 - public API reads as a declaration
    """Maximize one metric: an evaluator's name, or a usage metric name."""
    return Objective(metric=metric, direction=1)


def Minimize(metric: str) -> Objective:  # noqa: N802 - public API reads as a declaration
    """Minimize one metric, for example `Minimize("tokens")`."""
    return Objective(metric=metric, direction=-1)


class Combine:
    """How several objectives reduce to what the gate tests."""

    def __init__(self, *objectives: Objective):
        if not objectives:
            raise ValueError("a combiner needs at least one objective")
        self.objectives = objectives

    def scalar(self, case_metrics: Mapping[str, float]) -> float | None:
        raise NotImplementedError

    def metric_names(self) -> list[str]:
        return [objective.metric for objective in self.objectives]


class WeightedSum(Combine):
    """One scalar: the weighted mean of each objective's directed value.

    The gate's full guarantee applies, because the paired test runs on this scalar. Weights
    default to 1.0 each; pick them so the objectives are on comparable scales. A metric not
    measured for a case (for example a not-applicable evaluator on that case) drops out of that
    case's scalar and the remaining weights renormalize, so mixed case pools where different
    evaluators apply to different cases stay comparable; a case where nothing was measured is
    excluded entirely.
    """

    def __init__(self, *objectives: Objective, weights: list[float] | None = None):
        super().__init__(*objectives)
        self.weights = list(weights) if weights is not None else [1.0] * len(objectives)
        if len(self.weights) != len(objectives):
            raise ValueError("weights and objectives must have the same length")

    def scalar(self, case_metrics: Mapping[str, float]) -> float | None:
        total = 0.0
        weight_sum = 0.0
        for weight, objective in zip(self.weights, self.objectives, strict=True):
            value = objective.scalar(case_metrics)
            if value is None:
                continue
            total += weight * value
            weight_sum += weight
        if weight_sum == 0.0:
            return None
        return total / weight_sum


class Chebyshev(Combine):
    """The weakest objective decides (max-min fairness over improvements). Not implemented yet."""

    def scalar(self, case_metrics: Mapping[str, float]) -> float | None:
        raise NotImplementedError("Chebyshev combining lands after v1; use WeightedSum or a single objective")


class Lexicographic(Combine):
    """The first objective decides, ties broken by the next. Not implemented yet."""

    def scalar(self, case_metrics: Mapping[str, float]) -> float | None:
        raise NotImplementedError("Lexicographic combining lands after v1; use WeightedSum or a single objective")


class Pareto(Combine):
    """Jointly optimized objectives, reported as a set. Not implemented yet.

    `Pareto` is a reporting mode by design: the gate tests one scalar, so the final Pareto set is
    presented for human choice rather than auto-accepted.
    """

    def scalar(self, case_metrics: Mapping[str, float]) -> float | None:
        raise NotImplementedError("Pareto reporting lands after v1; use WeightedSum or a single objective")


class Derived:
    """A metric computed from a trial's other metrics. Not implemented yet."""

    def __init__(self, name: str, fn: Callable[[Mapping[str, float]], float]):
        raise NotImplementedError("Derived metrics land after v1; reference evaluator or usage metrics directly")


@dataclass(frozen=True)
class Constraint:
    """A bound on one metric's mean over a split's cases. A violated constraint rejects, nothing more."""

    metric: str
    bound: float
    direction: int  # +1 means the metric must stay at most `bound`; -1 means at least

    def violation(self, mean_value: float) -> float:
        """How far over the line the candidate is: positive means violated."""
        return self.direction * (mean_value - self.bound)

    def describe(self) -> str:
        op = "<=" if self.direction > 0 else ">="
        return f"{self.metric} {op} {self.bound}"


def AtMost(metric: str, bound: float) -> Constraint:  # noqa: N802 - public API reads as a declaration
    """The metric's mean must stay at or below `bound`, e.g. `AtMost("tokens", 4000)`."""
    return Constraint(metric=metric, bound=bound, direction=1)


def AtLeast(metric: str, bound: float) -> Constraint:  # noqa: N802 - public API reads as a declaration
    """The metric's mean must stay at or above `bound`, e.g. `AtLeast("Safety", 0.95)`."""
    return Constraint(metric=metric, bound=bound, direction=-1)


class Projection:
    """From per-case metrics to the one scalar the search and the gate consume.

    Args:
        objectives: One `Objective`, a `Combine` of several, or `None` for the default objective
            (maximize the mean of all evaluator scores).
        constraints: Bounds checked on the selection split at admission and on the held-out split
            at the gate. They only ever reject.
        evaluator_names: The experiment's evaluator names, used to validate metric references and
            to compute the default objective.

    An evaluator can be named with or without its `Evaluator` suffix: `Maximize("Helpfulness")`
    reads `HelpfulnessEvaluator`'s scores. A short name that is itself another evaluator's name,
    or a usage metric, keeps its own meaning.
    """

    def __init__(
        self,
        objectives: Objective | Combine | None,
        constraints: tuple[Constraint, ...] | list[Constraint] = (),
        *,
        evaluator_names: list[str] | None = None,
    ):
        self.objectives = objectives
        self.constraints = tuple(constraints)
        self.evaluator_names = list(evaluator_names or [])
        reserved = set(self.evaluator_names) | set(USAGE_METRICS)
        self._aliases = {
            name.removesuffix("Evaluator"): name
            for name in self.evaluator_names
            if name.endswith("Evaluator") and name != "Evaluator" and name.removesuffix("Evaluator") not in reserved
        }
        known = reserved | set(self._aliases)
        for metric in self.metric_names() + [c.metric for c in self.constraints]:
            if known and metric not in known:
                raise ValueError(
                    f"unknown metric {metric!r}; evaluators: {self.evaluator_names}, usage: {list(USAGE_METRICS)}"
                )

    def metric_names(self) -> list[str]:
        if self.objectives is None:
            return list(self.evaluator_names)
        if isinstance(self.objectives, Combine):
            return self.objectives.metric_names()
        return [self.objectives.metric]

    def _with_aliases(self, case_metrics: Mapping[str, float]) -> Mapping[str, float]:
        """`case_metrics` with each evaluator's short name added next to its full name."""
        aliased = {alias: case_metrics[name] for alias, name in self._aliases.items() if name in case_metrics}
        return {**case_metrics, **aliased} if aliased else case_metrics

    def scalar(self, case_metrics: Mapping[str, float]) -> float | None:
        """The gate scalar for one case; `None` when a needed metric was not measured."""
        if self.objectives is None:
            values = [case_metrics[name] for name in self.evaluator_names if name in case_metrics]
            return fmean(values) if values else None
        return self.objectives.scalar(self._with_aliases(case_metrics))

    def check_constraints(self, split_metrics: Mapping[str, Mapping[str, float]]) -> list[str]:
        """Evaluate every constraint on one split's per-case metrics; the violated ones, described.

        Args:
            split_metrics: Case name to that case's metric values, for the cases of one split
                (selection at admission, holdout at the gate).
        """
        violated: list[str] = []
        aliased = [self._with_aliases(metrics) for metrics in split_metrics.values()]
        for constraint in self.constraints:
            values = [metrics[constraint.metric] for metrics in aliased if constraint.metric in metrics]
            if not values:
                violated.append(f"{constraint.describe()}: metric never measured")
                continue
            mean_value = fmean(values)
            amount = constraint.violation(mean_value)
            if amount > 0:
                violated.append(f"{constraint.describe()}: mean {mean_value:.4g} violates by {amount:.4g}")
        return violated
