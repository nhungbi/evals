"""The optimizer loop: the only module that sees every layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ..experiment import Experiment
from ..types.optimization import IterationRecord
from .acceptance.budget import Budget
from .acceptance.gate import Gate
from .declaration.space import Marker
from .declaration.tunable import BuildContext, TunableFactory
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

    `build_task(ctx)` is called once per candidate and returns the `Experiment` task, `task(case)`.
    Build the agent inside `task`, never in `build_task` itself: an agent shared across cases and
    runs carries conversation state from one trial into the next (invariant 7). The factory path
    enforces this; `build_task` can't.
    """

    def __init__(
        self,
        experiment: Experiment,
        *,
        factory: TunableFactory | None = None,
        rollout_fn: RolloutFn | None = None,
        build_task: Callable[[BuildContext], TaskFn] | None = None,
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
        """`base_dir=None` puts runs under `.strands-evals/optimization/runs/`, relative to the working directory.

        Args:
            experiment: Supplies the cases and evaluators; it is never run itself.
            factory: The `@tunable` agent factory. Pass this or `build_task`.
            rollout_fn: Runs one case with a fresh agent; defaults to building the agent and invoking it
                on `case.input`.
            build_task: Returns the task for one candidate, from a `BuildContext`; needs `space`.
            space: Declared name -> marker, for `build_task`.
            objective: What to maximize or minimize.
            constraints: Bounds that can only reject.
            search: Which method changes which variables; None means `Search()`.
            selection: How the parent is chosen; None means `Greedy()`.
            gate: Acceptance and split settings; None means `Gate()`.
            budget: Rollout and iteration caps; None means `Budget()`.
            environment: Per-attempt resources passed to the factory by name.
            proposals_per_iteration: Patches each proposing block returns per iteration.
            max_workers: Parallel trials within one candidate.
            retries: Max infrastructure re-run passes per candidate.
            seed: Seeds the splits, sampling, bootstrap, and TPE.
            base_dir: Where run directories go.
            on_iteration: Called with each finished iteration.

        Raises:
            TypeError: Conflicting arguments, e.g. `factory` and `build_task` together.
            ValueError: No evaluators and no objective, a split with no cases, or a budget below the
                baseline's cost.
        """
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
        build_task: Callable[[BuildContext], TaskFn] | None = None,
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

        Args:
            run_dir: The run directory to reopen.
            experiment: As in `__init__`; must match the run.
            factory: As in `__init__`.
            rollout_fn: As in `__init__`.
            build_task: As in `__init__`.
            space: As in `__init__`; its keys must match the run.
            objective: As in `__init__`; must match the run.
            constraints: As in `__init__`; must match the run.
            search: As in `__init__`; must match the run.
            selection: As in `__init__`; must match the run.
            environment: As in `__init__`.
            budget: A new budget, e.g. larger to extend the run; None keeps the saved one.
            max_workers: None keeps the saved value.
            on_iteration: Called only for iterations run after the resume.

        Raises:
            ValueError: The configuration differs from `run.json`, naming the field.
        """
        raise NotImplementedError
