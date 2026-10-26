"""Scoring one candidate on cases through `Experiment`, with caching and the re-run pass."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ...case import Case
from ...evaluators.evaluator import Evaluator
from ...types.evaluation_report import EvaluationReport
from ...types.optimization import SplitName, TrialRecord, TrialStatus
from .rollout import TaskFn
from .traces import TraceCapture


@dataclass
class TrialLogEntry:
    """What the task wrapper recorded for one trial."""

    status: TrialStatus
    error_type: str | None
    error: str | None
    attempts: int
    trace_id: int
    attempt_span_id: int


class TrialLog:
    """One run's trial status, written by `instrument`, keyed by case name.

    `setup_error` holds a resource named like a declared marker; `TrialRunner.run` raises it.
    """

    setup_error: TypeError | None

    def record(self, case: str, entry: TrialLogEntry) -> None:
        """Store one trial's entry; called by `instrument`.

        Args:
            case: The case name.
            entry: What the wrapper recorded for that case's trial.
        """
        raise NotImplementedError

    def records(self) -> dict[str, TrialLogEntry]:
        raise NotImplementedError


class TrialCache(Protocol):
    """Saved trials and reports; implemented by `record.RunDirectory`."""

    def load(self, candidate_id: str, run_index: int, case: str) -> TrialRecord | None:
        """Return a saved trial.

        Args:
            candidate_id: The candidate.
            run_index: Which of `runs_per_candidate` runs.
            case: The case name.

        Returns:
            The saved trial, or None if it was never saved.
        """

    def save(self, record: TrialRecord) -> None: ...

    def save_report(
        self, candidate_id: str, split: SplitName, run_index: int, pass_index: int, report: EvaluationReport
    ) -> None:
        """`pass_index` 0 = first pass, k = k-th re-run pass (at most `retries`).

        Args:
            candidate_id: The candidate.
            split: The split the report covers.
            run_index: Which of `runs_per_candidate` runs.
            pass_index: 0 for the first pass, k for the k-th re-run pass.
            report: `Experiment`'s report for that pass.
        """
        ...


class TrialRunner:
    """Runs a candidate's uncached trials through a fresh `Experiment` per call.

    `retries` caps the infrastructure re-run passes per call (D7); `Experiment` keeps its own throttling
    retries inside each attempt.
    """

    def __init__(
        self,
        cases: Sequence[Case],
        evaluators: Sequence[Evaluator],
        traces: TraceCapture,
        cache: TrialCache,
        *,
        split_of: Mapping[str, SplitName],
        max_workers: int,
        retries: int,
    ):
        """Build the runner once per run.

        Args:
            cases: Every case of the run.
            evaluators: The experiment's evaluators.
            traces: The run's trace capture, installed before any `Experiment` is built.
            cache: Saved trials and reports (the run directory).
            split_of: Case name -> its split, given once so a record's split can't be wrong.
            max_workers: Parallel trials within one candidate.
            retries: Max infrastructure re-run passes per call.
        """
        raise NotImplementedError

    def pending(self, candidate_id: str, cases: Sequence[str], *, runs: int) -> int:
        """Number of (case, run) trials with no saved record: the real cost of the first passes.

        Args:
            candidate_id: The candidate.
            cases: The case names to score.
            runs: Runs per case.
        """
        raise NotImplementedError

    async def run(
        self,
        candidate_id: str,
        task: TaskFn,
        cases: Sequence[str],
        *,
        runs: int,
        allowance: int,
    ) -> tuple[list[TrialRecord], int]:
        """Score every run, then re-run infrastructure failures; returns (records, rollouts spent).

        Returns one record per requested (case, run), cached ones included, so `score()` sees the whole
        candidate.

        `task` is not yet instrumented (`make_task(...)` or the `build_task` path's task); each run wraps
        it with `instrument(task, TrialLog(), traces)`. After every run's first pass, up to `retries`
        re-run passes each retry the trials still `infrastructure_invalid`. `allowance` is the budget left
        (`BudgetTracker.remaining`) and caps only the re-run passes; trials left when it runs out stay
        `infrastructure_invalid`. A judge-only
        infrastructure failure re-runs just the judges on the saved output, for 0 rollouts. Each trial is
        saved as soon as it finishes.

        Args:
            candidate_id: The candidate.
            task: The candidate's task, not yet instrumented.
            cases: The case names to score.
            runs: Runs per case (`runs_per_candidate`).
            allowance: Rollouts left in the budget; caps only the re-run passes.

        Returns:
            One record per requested (case, run), cached ones included, and the rollouts spent.

        Raises:
            TypeError: A resource is named like a declared marker.
        """
        raise NotImplementedError
