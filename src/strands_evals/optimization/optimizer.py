"""The optimizer loop: the only module that sees every layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..experiment import Experiment
from ..types.optimization import IterationRecord, Value
from .acceptance.budget import Budget
from .acceptance.gate import Gate
from .declaration.space import Marker
from .declaration.tunable import TunableFactory
from .execution.rollout import EnvironmentFn, RolloutFn, TaskFn
from .result import OptimizationResult
from .scoring.objectives import Constraint, Objective
from .search.blocks import Search
from .selection.base import Selection


class Optimizer:
    """Searches for changes to a declared agent and keeps only those the holdout gate accepts.

    Pass either `factory=` (a `TunableFactory`) or `build_task=` with `space=`. `environment=`
    and `rollout_fn=` work only with `factory=`. `objective=None` maximizes the mean of every
    evaluator's score, as a plain eval run reports it. `retries` caps the infrastructure re-run
    passes per candidate. Construction warns when a TPE block is combined with `CaseFrontier`.
    """

    def __init__(
        self,
        experiment: Experiment,
        *,
        factory: TunableFactory | None = None,
        rollout_fn: RolloutFn | None = None,
        build_task: Callable[[Mapping[str, Value], Callable[[str], Any]], TaskFn] | None = None,
        space: Mapping[str, Marker] | None = None,
        objective: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        search: Search | None = None,
        selection: Selection | None = None,
        gate: Gate | None = None,
        budget: Budget | None = None,
        environment: EnvironmentFn | None = None,
        proposals_per_iteration: int = 3,
        max_workers: int = 4,
        retries: int = 2,
        seed: int = 0,
        base_dir: str | Path | None = None,
        on_iteration: Callable[[IterationRecord], None] | None = None,
    ):
        """`base_dir=None` puts runs under `.strands-evals/optimization/runs/`, relative to the working directory."""
        raise NotImplementedError

    def run(self) -> OptimizationResult:
        raise NotImplementedError

    async def run_async(self) -> OptimizationResult:
        raise NotImplementedError

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        experiment: Experiment,
        factory: TunableFactory | None = None,
        rollout_fn: RolloutFn | None = None,
        build_task: Callable[[Mapping[str, Value], Callable[[str], Any]], TaskFn] | None = None,
        space: Mapping[str, Marker] | None = None,
        objective: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        search: Search | None = None,
        selection: Selection | None = None,
        environment: EnvironmentFn | None = None,
        budget: Budget | None = None,
        max_workers: int | None = None,
        on_iteration: Callable[[IterationRecord], None] | None = None,
    ) -> Optimizer:
        """Reopen a run directory; every saved trial is reused as is.

        Pass again what `run.json` can't store: the experiment, the agent (`factory=`, or
        `build_task=` with `space=`), `rollout_fn`, `environment`, the objective, constraints,
        search, and selection. They must match the run, else `ValueError` naming the field. The
        gate, seed, `proposals_per_iteration`, and `retries` are read from `run.json`. Only
        `budget`, `max_workers`, and `on_iteration` may differ (`None` keeps the saved value);
        resuming with a larger `Budget` is how to extend a run.

        If you changed cases, evaluators, judge settings, or agent code, start a new run instead:
        resuming would mix results from the old and new setups.
        """
        raise NotImplementedError
