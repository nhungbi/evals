"""What a run returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..types.evaluation_report import EvaluationReport
from ..types.optimization import IterationRecord, SplitName, TestReport, Usage, ValidationReport, Value
from .candidate import Candidate
from .declaration.space import Space

Grain = Literal["iteration", "proposal", "case"]


class CandidateView:
    """A candidate as the results show it."""

    def __init__(self, candidate: Candidate, baseline: Candidate, space: Space, number: int):
        """Built by the optimizer; users never construct one.

        Args:
            candidate: The candidate shown.
            baseline: The run's baseline, for diffs.
            space: Maps file-backed keys to their real paths.
            number: Its display number; `summary()` shows "candidate 3 (3f9a1c0b7e2d)".
        """
        raise NotImplementedError

    @property
    def id(self) -> str:
        """The content id, a hash of the values; the key everywhere."""
        raise NotImplementedError

    @property
    def number(self) -> int:
        """The display number (0 = baseline, then proposal order); for people only."""
        raise NotImplementedError

    @property
    def values(self) -> dict[str, Value]:
        """A copy of the candidate's typed values."""
        raise NotImplementedError

    def diff(self) -> str:
        """Unified diff of text and file-backed values against the baseline, on the real paths."""
        raise NotImplementedError

    def changed_values(self) -> dict[str, tuple[Value, Value]]:
        """Changed numbers and choices against the baseline."""
        raise NotImplementedError


@dataclass
class OptimizationResult:
    """What a run returns.

    Attributes:
        best: The champion, `validation.champion_id` (the baseline when nothing feasible beats it).
        validation: How the champion was picked on validation: every finalist's objective and estimates, and
            why any was infeasible.
        test: The champion against the baseline on test, report only; None without test cases or when it
            couldn't be afforded (the reason is in the `test` event).
        effects: `effects.json`: per metric and option from the search fit, or empty with the reason.
        stop_reason: Why the search stopped: `budget` (what's left after the end-phase reserves can't score
            one more candidate), `iterations` (the cap), `exhausted` (every block produced no candidates on
            its latest turn, or the schedule had nothing left), or `proposer_error` (a method raised). The
            end phase runs after any of them.
        agent_cost: The agent's own usage per split (each trial's last attempt): train = the search,
            validation and test = the end phase.
        search_cost: Total proposer usage.
        judge_cost: Total judge usage across trials, earlier attempts included.
        retry_cost: The agent's usage in earlier attempts (`TrialRecord.earlier`).
    """

    run_id: str
    run_dir: Path
    baseline: CandidateView
    best: CandidateView
    validation: ValidationReport
    test: TestReport | None
    effects: dict[str, Any]
    iterations: list[IterationRecord]
    rollouts_spent: int
    stop_reason: Literal["budget", "iterations", "exhausted", "proposer_error"]
    agent_cost: dict[SplitName, Usage]
    search_cost: Usage
    judge_cost: Usage
    retry_cost: Usage

    def summary(self) -> str:
        raise NotImplementedError

    def to_rows(self, grain: Grain = "proposal") -> list[dict[str, Any]]:
        """The run as flat rows, built from the event log, so a resumed run's rows cover the whole run.

        Every row of one grain has the same keys; a value that doesn't apply is None. Per-metric columns
        are named `<prefix>.<metric>`, e.g. `train.HelpfulnessEvaluator`.

        Args:
            grain: What one row is: `proposal` (every proposal, the baseline included, with its label at
                its step and its rank under the last search fit), `iteration` (one search step), or
                `case` (one trial: candidate x case x run, on every split it was scored on).
        """
        raise NotImplementedError

    def to_json(self, grain: Grain = "proposal") -> str:
        raise NotImplementedError

    def to_file(self, path: str | Path, grain: Grain = "proposal") -> None:
        """Write `to_json(grain)` to `path`.

        Args:
            path: The JSON file to write.
            grain: One row per iteration, proposal, or case.
        """
        raise NotImplementedError

    def candidate(self, candidate_id: str) -> CandidateView:
        """Any saved candidate by id; unknown ids raise `KeyError`."""
        raise NotImplementedError

    def reports(self, candidate_id: str) -> list[EvaluationReport]:
        raise NotImplementedError
