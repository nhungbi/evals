"""The optimizer loop: Pareto-frontier search over candidates with a hard rollout budget.

Cases split three ways. The proposer reads failures from the training split, the frontier ranks
candidates on the selection split, and the acceptance gate judges on the held-out split, so no
split both chooses and judges. Every admitted candidate joins a pool; each iteration samples one
frontier member as the parent (see `frontier.py`), so specialists that uniquely solve some cases
keep being explored. Admission is loose (the candidate must land on the frontier); the champion,
the candidate a run ships, only changes when a challenger clears the strict paired-bootstrap gate
on the held-out split.

The optimizer does not run agents itself. The caller supplies either a `@tunable` agent factory
plus a `rollout_fn(make_agent, case)` function, or a `build_task` factory that materializes a candidate's
artifacts and returns a task callable in the exact shape `Experiment.run_evaluations_async`
already accepts. Either way, scoring reuses the caller's experiment unchanged, including its
retry policy and worker pool.

Each candidate is scored with k repeated runs, because a single run of an LLM judge over an LLM
agent is too noisy to rank candidates. Task results cache per `(candidate, case, run index)`, and
scored reports persist in the run directory, so resuming a crashed run re-scores nothing.
"""

import json
import logging
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from strands.models.model import Model
from typing_extensions import Any, Callable, Literal, Mapping

from ..case import Case
from ..experiment import Experiment
from ..local_file_task_result_store import LocalFileTaskResultStore
from ..types.evaluation_report import EvaluationReport
from .acceptance import AcceptanceVerdict, paired_bootstrap_compare
from .artifacts import Artifact, ArtifactSpace
from .candidate import Candidate
from .declarative import TunableFactory
from .frontier import pareto_front, sample_parent
from .proposer import Proposer, build_reflection_records
from .runs import RunDirectory

logger = logging.getLogger(__name__)

_ERROR_REASON_PREFIXES = ("An error occurred:", "Evaluator error:")

# Below this many held-out cases, judge noise buys too many false accepts (measured at 8-10%
# false positives with 5 cases in the gate calibration experiments).
_MIN_HOLDOUT_FOR_QUIET_GATE = 5

ProposalStatus = Literal["proposed", "unscored", "scored", "admitted", "dominated", "accepted"]


@dataclass(frozen=True)
class Budget:
    """The single authoritative budget knob. The loop is never unbounded.

    Attributes:
        max_rollouts: Hard cap on task executions across the whole run. One rollout is one case
            executed once; k repeated runs of n cases cost k*n rollouts per candidate.
        max_iterations: Optional cap on proposer iterations, on top of the rollout cap.
    """

    max_rollouts: int = 500
    max_iterations: int | None = None


@dataclass(frozen=True)
class Gate:
    """Acceptance-gate configuration: how much evidence an edit needs before it is kept.

    Attributes:
        runs_per_candidate: k repeated scoring runs per candidate. The default of 3 suits
            LLM-judge scoring; 1 suits verifiable rewards.
        holdout_fraction: Fraction of cases held out for the champion gate. Neither the proposer
            nor the frontier ever reads them.
        selection_fraction: Fraction of cases the frontier ranks candidates on. When too few
            cases remain to carve out a separate selection split, the training split doubles as
            the selection split and a warning is logged.
        margin: Minimum mean held-out improvement a challenger must clear to become champion.
        confidence: Confidence level for the bootstrap interval in the champion gate.
        min_coverage: Minimum fraction of cases a candidate must have scores for, against its
            parent's selection cases for admission and against the champion's held-out cases
            for the gate. Below the floor the candidate is dominated or rejected, so an edit
            that crashes the agent on the hardest cases cannot win by being compared only on the
            cases it survived.
        frontier_eps: How far below a case's best score still counts as leading that case.
            Zero means only exact ties share the lead.
        challenger_holdout_only: Score challengers on the selection and held-out splits only.
            Admission reads only the selection split and the gate only the holdout, so this
            skips the training cases for every candidate that is never used as a parent. A
            candidate is topped up on the training split (its other cases hit the per-case
            cache) the first time it is sampled as a parent or becomes champion.
    """

    runs_per_candidate: int = 3
    holdout_fraction: float = 0.3
    selection_fraction: float = 0.3
    margin: float = 0.0
    confidence: float = 0.95
    min_coverage: float = 0.9
    frontier_eps: float = 0.0
    challenger_holdout_only: bool = False


@dataclass
class CandidateScore:
    """Per-case scores for one candidate, aggregated over k repeated runs."""

    candidate_id: str
    per_case: dict[str, float] = field(default_factory=dict)
    per_case_runs: dict[str, list[float]] = field(default_factory=dict)
    case_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    error_rollouts: int = 0

    def mean(self, case_names: list[str] | None = None) -> float:
        """Mean per-case score over `case_names` (all scored cases when omitted)."""
        names = [n for n in (case_names or list(self.per_case)) if n in self.per_case]
        if not names:
            return 0.0
        return sum(self.per_case[n] for n in names) / len(names)

    def subset(self, case_names: list[str]) -> dict[str, float]:
        return {n: self.per_case[n] for n in case_names if n in self.per_case}


@dataclass
class ProposalOutcome:
    """What happened to one proposal in one iteration.

    `status` is the proposal's final state: `unscored` means the budget ran out before it could
    be scored, `dominated` means it was scored but did not land on the frontier, `admitted`
    means it joined the pool (and may be sampled as a parent), and `accepted` means it was
    admitted and also cleared the gate to become champion. An admitted proposal that challenged
    the champion and lost keeps `admitted`, with the losing `verdict` attached.
    """

    candidate_id: str
    artifact_name: str
    rationale: str
    diff: str
    parent_id: str = ""
    train_score: float | None = None
    selection_score: float | None = None
    holdout_score: float | None = None
    cases_led: list[str] = field(default_factory=list)
    verdict: AcceptanceVerdict | None = None
    status: ProposalStatus = "proposed"
    error_rollouts: int = 0
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """One loop iteration, in the order things happened. Fed to `on_iteration` as it completes.

    The `incumbent_*` fields describe the champion at the start of the iteration. `parent_id` is
    the frontier member the iteration's proposals were edited from, and `frontier` maps each
    frontier member to the number of selection cases it led when the parent was sampled.
    """

    iteration: int
    incumbent_id: str
    incumbent_train_score: float
    incumbent_holdout_score: float
    incumbent_values: dict[str, str] = field(default_factory=dict)
    parent_id: str = ""
    frontier: dict[str, int] = field(default_factory=dict)
    proposals: list[ProposalOutcome] = field(default_factory=list)
    accepted_candidate_id: str | None = None
    rollouts_spent: int = 0


@dataclass
class OptimizationResult:
    """The answer to the developer's two questions: did it get better, and what changed.

    `best` is the champion: the candidate that last cleared the gate. `frontier` maps each final
    frontier member to the selection cases it leads, for inspecting the specialists the search
    kept. `accepted_count` counts champion changes and `rejected_count` counts lost champion
    challenges; `admitted_count` and `dominated_count` count frontier admissions.
    """

    run_id: str
    run_dir: Path
    baseline: Candidate
    best: Candidate
    baseline_holdout_score: float
    best_holdout_score: float
    iterations: list[IterationRecord]
    lineage: list[dict[str, Any]]
    rollouts_spent: int
    accepted_count: int
    rejected_count: int
    admitted_count: int
    dominated_count: int
    frontier: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        delta = self.best_holdout_score - self.baseline_holdout_score
        return (
            f"baseline {self.baseline_holdout_score:.2f} -> best {self.best_holdout_score:.2f} "
            f"on the held-out split ({delta:+.2f})\n"
            f"{self.accepted_count} edits accepted, {self.rejected_count} rejected, "
            f"{self.admitted_count} admitted to the frontier, {self.dominated_count} dominated | "
            f"{self.rollouts_spent} rollouts | run_id {self.run_id}"
        )

    def reports(self, candidate_id: str) -> dict[str, EvaluationReport]:
        """The `EvaluationReport`s stored while scoring `candidate_id`, keyed by report name.

        Keys are the report file stems (`run0`, `run1.holdout`, ...): one report per scoring run,
        tagged with the case subset it covered. Each value is the exact `EvaluationReport` the
        experiment returned for that run, loaded back from the run directory, so everything the
        evals SDK puts in a report is available: per-case scores, the `evaluator` tag on every
        row, reasons, `test_passes`, and `detailed_results`.
        """
        scores_dir = RunDirectory(self.run_dir).candidate_dir(candidate_id) / "scores"
        return {
            path.name.removesuffix(".report.json"): EvaluationReport.model_validate_json(path.read_text())
            for path in sorted(scores_dir.glob("run*.report.json"))
        }

    def to_rows(self, grain: Literal["iteration", "proposal", "case"] = "proposal") -> list[dict[str, Any]]:
        """Flatten the run into JSON-serializable rows, one dict per row.

        `grain="iteration"` gives one row per loop iteration: the champion's id, scores, and
        applied values, the sampled parent, the frontier size, plus rollout spend.
        `grain="proposal"` gives one row per proposal: its parent, per-split scores, the cases
        it leads on the frontier, the gate verdict (when it challenged the champion), and the
        candidate's applied values. `grain="case"` gives one
        row per stored report row: exactly what each evaluator returned for each case in each
        scoring run (score, `evaluator` tag, reason, `test_pass`), joined with the candidate's
        iteration and status. Nothing is averaged; aggregate however your analysis needs
        (for example `df.groupby(["iteration", "evaluator"]).score.mean()`).

        Every value is a plain string or number, so the rows serialize as-is; component values
        are JSON strings in the candidate's own encoding.
        """
        if grain == "iteration":
            rows: list[dict[str, Any]] = []
            cumulative = 0
            for record in self.iterations:
                cumulative += record.rollouts_spent
                rows.append(
                    {
                        "iteration": record.iteration,
                        "incumbent_id": record.incumbent_id,
                        "incumbent_train_score": record.incumbent_train_score,
                        "incumbent_holdout_score": record.incumbent_holdout_score,
                        "incumbent_values": dict(record.incumbent_values),
                        "parent_id": record.parent_id,
                        "frontier_size": len(record.frontier),
                        "proposals_count": len(record.proposals),
                        "accepted_candidate_id": record.accepted_candidate_id,
                        "rollouts_spent": record.rollouts_spent,
                        "rollouts_cumulative": cumulative,
                    }
                )
            return rows
        if grain == "proposal":
            rows = []
            for record in self.iterations:
                for outcome in record.proposals:
                    verdict = outcome.verdict
                    rows.append(
                        {
                            "iteration": record.iteration,
                            "candidate_id": outcome.candidate_id,
                            "parent_id": outcome.parent_id,
                            "artifact_name": outcome.artifact_name,
                            "status": outcome.status,
                            "train_score": outcome.train_score,
                            "selection_score": outcome.selection_score,
                            "holdout_score": outcome.holdout_score,
                            "cases_led": list(outcome.cases_led),
                            "mean_diff": verdict.mean_diff if verdict else None,
                            "ci_low": verdict.ci_low if verdict else None,
                            "ci_high": verdict.ci_high if verdict else None,
                            "margin": verdict.margin if verdict else None,
                            "n_cases": verdict.n_cases if verdict else None,
                            "verdict_reason": verdict.reason if verdict else None,
                            "error_rollouts": outcome.error_rollouts,
                            "rationale": outcome.rationale,
                            "values": dict(outcome.values),
                            "diff": outcome.diff,
                        }
                    )
            return rows
        if grain == "case":
            candidates: list[tuple[int, str, str]] = [(0, self.baseline.id, "baseline")]
            for record in self.iterations:
                for outcome in record.proposals:
                    candidates.append((record.iteration, outcome.candidate_id, outcome.status))
            rows = []
            for iteration, candidate_id, status in candidates:
                for report_name, report in self.reports(candidate_id).items():
                    for i, case_row in enumerate(report.cases):
                        detailed = report.detailed_results[i] if i < len(report.detailed_results) else []
                        rows.append(
                            {
                                "iteration": iteration,
                                "candidate_id": candidate_id,
                                "status": status,
                                "report": report_name,
                                "case_name": case_row.get("name"),
                                "evaluator": case_row.get("evaluator"),
                                "score": report.scores[i] if i < len(report.scores) else None,
                                "test_pass": report.test_passes[i] if i < len(report.test_passes) else None,
                                "applicable": EvaluationReport.is_applicable(detailed),
                                "reason": report.reasons[i] if i < len(report.reasons) else "",
                            }
                        )
            return rows
        raise ValueError(f"unknown grain {grain!r}; expected 'iteration', 'proposal', or 'case'")

    def to_json(
        self,
        grain: Literal["iteration", "proposal", "case"] = "proposal",
        path: str | Path | None = None,
        indent: int = 2,
    ) -> str:
        """Serialize `to_rows(grain)` to a JSON string, writing it to `path` when given."""
        payload = json.dumps(self.to_rows(grain), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(payload)
        return payload


class Optimizer:
    """Pareto-frontier optimizer over an artifact space, scored by an `Experiment`.

    Args:
        experiment: The cases and evaluators to optimize against. Every case needs a unique name.
        build_task: Builds a task callable from a candidate. The callable is handed straight to
            `Experiment.run_evaluations_async`. Called once per scoring run, so it must be cheap
            and must not carry state between calls. Not needed when `factory` is given.
        artifacts: The artifacts the proposer may edit: an `ArtifactSpace`, or a plain mapping of
            name to `Artifact` when no workspace `root` is needed. Not needed when `factory` is
            given.
        factory: A `tunable` construction function (decorator or `inject=` form). Its markers
            become the artifacts, and each rollout builds a fresh agent with the candidate's
            values. Requires `rollout_fn`.
        rollout_fn: Called as `rollout_fn(agent_factory, case)` for each rollout, where `agent_factory` is
            construction function with the candidate's values bound. Returns what an `Experiment`
            task returns. May be async.
        budget: Rollout and iteration caps. A conservative default applies when unset.
        gate: Acceptance-gate configuration (repeated runs, selection and holdout splits,
            margin, confidence, coverage floor, frontier tolerance). Defaults suit LLM-judge
            scoring.
        proposer: Edit proposer. Mutually exclusive with `proposer_model`.
        proposer_model: Model for the default proposer, when `proposer` is not given.
        proposals_per_iteration: How many edits the proposer is asked for each iteration. Only
            applies to the default proposer; a caller-supplied `proposer` carries its own.
        max_workers: Parallel workers for `run_evaluations_async`.
        base_dir: Where run directories are created. Defaults to `.strands-evals/optimize/runs`.
        run_dir: An existing run directory to resume. Cached scores are reused, the case split
            is reloaded, the pool is rebuilt from the lineage's admissions, and the champion
            continues from the `best` pointer. The caller must re-supply the same experiment and
            artifacts the run was created with.
        on_iteration: Optional callback invoked with each completed `IterationRecord`.
        seed: Seed for the case split, parent sampling, and bootstrap, recorded in `run.json`.
    """

    def __init__(
        self,
        experiment: Experiment,
        build_task: Callable[[Candidate], Callable[[Case], Any]] | None = None,
        artifacts: ArtifactSpace | Mapping[str, Artifact] | None = None,
        *,
        factory: TunableFactory | None = None,
        rollout_fn: Callable[[Callable[..., Any], Case], Any] | None = None,
        budget: Budget | None = None,
        gate: Gate | None = None,
        proposer: Proposer | None = None,
        proposer_model: Model | str | None = None,
        proposals_per_iteration: int = 3,
        max_workers: int = 4,
        base_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        on_iteration: Callable[[IterationRecord], None] | None = None,
        seed: int = 0,
    ):
        if factory is not None:
            if build_task is not None or artifacts is not None:
                raise ValueError("pass factory and rollout_fn, or build_task and artifacts, not both")
            if rollout_fn is None:
                raise ValueError("factory needs rollout_fn: the function that drives one rollout")
            build_task = factory.build_task(rollout_fn)
            artifacts = factory.artifacts()
        elif build_task is None or artifacts is None:
            raise ValueError("pass build_task and artifacts, or factory and rollout_fn")
        self.experiment = experiment
        self.build_task = build_task
        self.space = artifacts if isinstance(artifacts, ArtifactSpace) else ArtifactSpace(artifacts)
        self.budget = budget or Budget()
        self.gate = gate or Gate()
        if proposer is not None and proposer_model is not None:
            raise ValueError("pass either proposer or proposer_model, not both")
        self.proposer = proposer or Proposer(model=proposer_model, proposals_per_iteration=proposals_per_iteration)
        self.max_workers = max_workers
        self.on_iteration = on_iteration
        self.seed = seed

        names = [case.name for case in experiment.cases]
        if any(name is None for name in names):
            raise ValueError("every case needs a name: the optimizer keys caches and splits on case names")
        if len(set(names)) != len(names):
            raise ValueError("case names must be unique")
        if len(names) < 2:
            raise ValueError("optimization needs at least 2 cases (one for training, one held out)")
        self._case_names: list[str] = list(names)  # type: ignore[arg-type]

        if run_dir is not None:
            self.run_directory = RunDirectory.load(run_dir)
            config = self.run_directory.read_config()
            self.holdout_names: list[str] = config["holdout_cases"]
            self.train_names: list[str] = config["train_cases"]
            # Runs recorded before the selection split existed rank the frontier on train.
            self.selection_names: list[str] = config.get("selection_cases", self.train_names)
            self._validate_resume(config)
            self.run_directory.log_event("resume", budget=asdict(self.budget), gate=asdict(self.gate))
        else:
            self.train_names, self.selection_names, self.holdout_names = self._split_cases()
            self.run_directory = RunDirectory.create(
                base_dir,
                config={
                    "budget": asdict(self.budget),
                    "gate": asdict(self.gate),
                    "seed": seed,
                    "train_cases": self.train_names,
                    "selection_cases": self.selection_names,
                    "holdout_cases": self.holdout_names,
                    "artifacts": {name: self.space[name].display_path() for name in self.space.names()},
                },
            )
        if len(self.holdout_names) < _MIN_HOLDOUT_FOR_QUIET_GATE:
            logger.warning(
                "holdout_cases=<%d> | small holdouts inflate false-positive accepts, %d or more recommended",
                len(self.holdout_names),
                _MIN_HOLDOUT_FOR_QUIET_GATE,
            )

        self._scores: dict[str, CandidateScore] = {}
        self._rollouts_spent = self._replay_spend()

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        experiment: Experiment,
        build_task: Callable[[Candidate], Callable[[Case], Any]] | None = None,
        artifacts: ArtifactSpace | Mapping[str, Artifact] | None = None,
        **kwargs: Any,
    ) -> "Optimizer":
        """Reopen a run directory and continue it. Already-scored candidates are cache hits.

        The experiment, task factory, and artifacts are code, so the caller re-supplies them;
        everything recorded in the run directory (split, budget spend, incumbent, scores) is
        reloaded, and the supplied cases and artifact names must match what was recorded.
        """
        return cls(experiment, build_task, artifacts, run_dir=run_dir, **kwargs)

    async def run(self) -> OptimizationResult:
        """Drive the loop until the budget is exhausted, and return the champion found."""
        baseline = self._seed_candidate()
        self.run_directory.save_values(baseline.id, baseline.values)
        self.run_directory.write_manifest(
            baseline.id, parent_id=None, rationale="baseline", values=baseline.values, status="proposed"
        )
        if self.run_directory.get_best() is None:
            self.run_directory.set_best(baseline.id)
            self.run_directory.log_event("propose", candidate_id=baseline.id, parent_id=None, rationale="baseline")

        baseline_score = await self._score_candidate(baseline)
        pool = await self._reload_pool(baseline)
        champion = pool.get(self.run_directory.get_best() or baseline.id, baseline)

        # Counters and iteration numbering replay from the lineage, so a resumed run continues
        # the sequence instead of restarting at iteration 1 and under-reporting totals.
        prior_events = self.run_directory.load_events()
        iterations: list[IterationRecord] = []
        accepted_count = sum(1 for e in prior_events if e["event"] == "accept")
        rejected_count = sum(1 for e in prior_events if e["event"] == "reject")
        admitted_count = sum(1 for e in prior_events if e["event"] == "admit")
        dominated_count = sum(1 for e in prior_events if e["event"] == "dominated")
        iteration = max((e.get("iteration", 0) for e in prior_events), default=0)

        while True:
            iteration += 1
            if self.budget.max_iterations is not None and iteration > self.budget.max_iterations:
                logger.info("iteration=<%d> | iteration cap reached, stopping", iteration)
                break
            if not self._can_afford(self._cost_per_candidate()):
                logger.info("rollouts_spent=<%d> | budget exhausted, stopping", self._rollouts_spent)
                break

            # Selection: sample one frontier member, weighted by the selection cases it leads.
            front = self._frontier(pool)
            parent = pool[sample_parent(front, self._parent_rng(iteration))] if front else champion
            if not await self._ensure_fully_scored(parent):
                logger.info("candidate=<%s> | budget cannot cover the parent's training cases, stopping", parent.id)
                break
            parent_score = self._scores[parent.id]
            champion_score = self._scores[champion.id]
            self.run_directory.log_event(
                "select",
                iteration=iteration,
                parent_id=parent.id,
                frontier={candidate_id: sorted(cases) for candidate_id, cases in front.items()},
            )

            record = IterationRecord(
                iteration=iteration,
                incumbent_id=champion.id,
                incumbent_train_score=champion_score.mean(self.train_names),
                incumbent_holdout_score=champion_score.mean(self.holdout_names),
                incumbent_values=dict(champion.values),
                parent_id=parent.id,
                frontier={candidate_id: len(cases) for candidate_id, cases in front.items()},
            )

            records = build_reflection_records(
                parent_score.case_rows,
                parent_score.subset(self.train_names),
            )
            proposals = await self._propose_with_retry(parent, records, iteration)
            if not proposals:
                logger.info("iteration=<%d> | proposer returned no valid edits, stopping", iteration)
                break

            outcomes: list[ProposalOutcome] = []
            candidates_by_id: dict[str, Candidate] = {}
            for proposal in proposals:
                candidate = parent.mutate(proposal.artifact_name, proposal.new_value, rationale=proposal.rationale)
                if candidate.id in candidates_by_id or candidate.id in pool:
                    logger.info("candidate=<%s> | duplicate proposal content, dropped", candidate.id)
                    continue
                candidates_by_id[candidate.id] = candidate
                self.run_directory.save_values(candidate.id, candidate.values)
                self.run_directory.write_manifest(
                    candidate.id,
                    parent_id=parent.id,
                    rationale=proposal.rationale,
                    values=candidate.values,
                    status="proposed",
                )
                self.run_directory.log_event(
                    "propose",
                    candidate_id=candidate.id,
                    parent_id=parent.id,
                    artifact=proposal.artifact_name,
                    rationale=proposal.rationale,
                    iteration=iteration,
                )
                outcome = ProposalOutcome(
                    candidate_id=candidate.id,
                    artifact_name=proposal.artifact_name,
                    rationale=proposal.rationale,
                    diff=candidate.diff(parent),
                    parent_id=parent.id,
                    values=dict(candidate.values),
                )
                if not self._can_afford(self._cost_per_candidate()) and candidate.id not in self._scores:
                    outcome.status = "unscored"
                    self.run_directory.set_status(candidate.id, "unscored")
                    outcomes.append(outcome)
                    continue
                score = await self._score_candidate(self._with_context(candidate), self._challenger_cases())
                outcome.train_score = None if self.gate.challenger_holdout_only else score.mean(self.train_names)
                outcome.selection_score = score.mean(self.selection_names)
                outcome.holdout_score = score.mean(self.holdout_names)
                outcome.error_rollouts = score.error_rollouts
                outcome.status = "scored"
                self.run_directory.set_status(candidate.id, "scored")
                outcomes.append(outcome)

            # Admission (loose): a candidate joins the pool when it lands on the frontier. The
            # best-scoring candidates are tried first, so they win exact ties within the batch.
            scored = [o for o in outcomes if o.status == "scored"]
            scored.sort(key=lambda o: o.selection_score or 0.0, reverse=True)
            admitted: list[ProposalOutcome] = []
            for outcome in scored:
                cases_led = self._admission_cases(outcome.candidate_id, parent_score, pool)
                if cases_led is None:
                    outcome.status = "dominated"
                    dominated_count += 1
                    self.run_directory.set_status(outcome.candidate_id, "dominated")
                    self.run_directory.log_event(
                        "dominated", candidate_id=outcome.candidate_id, parent_id=parent.id, iteration=iteration
                    )
                    continue
                outcome.status = "admitted"
                outcome.cases_led = cases_led
                admitted_count += 1
                pool[outcome.candidate_id] = self._with_context(candidates_by_id[outcome.candidate_id])
                admitted.append(outcome)
                self.run_directory.set_status(outcome.candidate_id, "admitted")
                self.run_directory.log_event(
                    "admit",
                    candidate_id=outcome.candidate_id,
                    parent_id=parent.id,
                    iteration=iteration,
                    cases_led=cases_led,
                )

            # Champion gate (strict): only the strongest admission that beats the champion on the
            # selection split challenges it, which caps how often the holdout is consulted.
            champion_selection = champion_score.mean(self.selection_names)
            challengers = [o for o in admitted if (o.selection_score or 0.0) > champion_selection]
            if challengers:
                outcome = challengers[0]
                challenger_score = self._scores[outcome.candidate_id]
                verdict = paired_bootstrap_compare(
                    challenger_score.subset(self.holdout_names),
                    champion_score.subset(self.holdout_names),
                    margin=self.gate.margin,
                    confidence=self.gate.confidence,
                    min_coverage=self.gate.min_coverage,
                    seed=self.seed,
                )
                outcome.verdict = verdict
                event = "accept" if verdict.accepted else "reject"
                self.run_directory.log_event(
                    event,
                    candidate_id=outcome.candidate_id,
                    champion_id=champion.id,
                    iteration=iteration,
                    mean_diff=verdict.mean_diff,
                    ci_low=verdict.ci_low,
                    ci_high=verdict.ci_high,
                    reason=verdict.reason,
                )
                if verdict.accepted:
                    outcome.status = "accepted"
                    accepted_count += 1
                    self.run_directory.set_status(outcome.candidate_id, "accepted")
                    self.run_directory.set_best(outcome.candidate_id)
                    champion = pool[outcome.candidate_id]
                    # Top the new champion up to the full case set so the iteration records and
                    # the result report its training score; its other cases hit the cache.
                    await self._ensure_fully_scored(champion)
                    record.accepted_candidate_id = outcome.candidate_id
                else:
                    rejected_count += 1

            record.proposals = outcomes
            record.rollouts_spent = self._rollouts_spent
            iterations.append(record)
            if self.on_iteration is not None:
                self.on_iteration(record)

        best_id = self.run_directory.get_best() or baseline.id
        best = pool.get(best_id) or self._candidate_from_values(best_id, parent=baseline)
        best_score = self._scores.get(best_id, baseline_score)
        return OptimizationResult(
            run_id=self.run_directory.run_id,
            run_dir=self.run_directory.root,
            baseline=baseline,
            best=best,
            baseline_holdout_score=baseline_score.mean(self.holdout_names),
            best_holdout_score=best_score.mean(self.holdout_names),
            iterations=iterations,
            lineage=self.run_directory.load_events(),
            rollouts_spent=self._rollouts_spent,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            admitted_count=admitted_count,
            dominated_count=dominated_count,
            frontier={candidate_id: sorted(cases) for candidate_id, cases in self._frontier(pool).items()},
        )

    # -- selection ---------------------------------------------------------------------

    def _frontier(self, pool: Mapping[str, Candidate]) -> dict[str, frozenset[str]]:
        """The pool's Pareto frontier on the selection split, in pool order."""
        return pareto_front(
            {candidate_id: self._scores[candidate_id].subset(self.selection_names) for candidate_id in pool},
            self.selection_names,
            self.gate.frontier_eps,
        )

    def _parent_rng(self, iteration: int) -> random.Random:
        # Seeded per iteration (a string seed hashes deterministically), so a resumed run samples
        # the same parent an uninterrupted run would have at that iteration.
        return random.Random(f"{self.seed}:{iteration}")

    def _admission_cases(
        self, candidate_id: str, parent_score: CandidateScore, pool: Mapping[str, Candidate]
    ) -> list[str] | None:
        """The selection cases a candidate would lead if admitted, or `None` when it is dominated.

        The candidate is placed after every pool member, so an exact tie on every case it leads
        goes to the existing member and a no-op edit is never admitted. A candidate that lost
        coverage against its parent (crashed on its selection cases) is dominated outright.
        """
        candidate_cases = self._scores[candidate_id].subset(self.selection_names)
        parent_cases = parent_score.subset(self.selection_names)
        covered = len(set(candidate_cases) & set(parent_cases))
        if parent_cases and covered < self.gate.min_coverage * len(parent_cases):
            return None
        trial = {member_id: self._scores[member_id].subset(self.selection_names) for member_id in pool}
        trial[candidate_id] = candidate_cases
        front = pareto_front(trial, self.selection_names, self.gate.frontier_eps)
        if candidate_id not in front:
            return None
        return sorted(front[candidate_id])

    async def _ensure_fully_scored(self, candidate: Candidate) -> bool:
        """Top a candidate up to the full case set, returning False when the budget cannot.

        Only candidates scored on the selection and held-out splits alone need this
        (`challenger_holdout_only`); the cases they were already scored on hit the cache.
        """
        if self.run_directory.report_path(candidate.id, 0, "full").exists():
            await self._score_candidate(candidate)
            return True
        cached = self._scores.get(candidate.id)
        missing = [n for n in self._case_names if cached is None or n not in cached.per_case]
        if not self._can_afford(self.gate.runs_per_candidate * len(missing)):
            return False
        await self._score_candidate(candidate)
        return True

    def _challenger_cases(self) -> list[str] | None:
        """The cases a challenger is scored on: every case, or selection plus holdout only."""
        if not self.gate.challenger_holdout_only:
            return None
        return sorted(set(self.selection_names) | set(self.holdout_names))

    async def _propose_with_retry(self, incumbent: Candidate, records: list[dict[str, Any]], iteration: int):
        """Call the proposer with transient-failure retries.

        A dropped stream or timeout in the proposer's LLM call must not crash a run that has
        hundreds of cached rollouts behind it: retry, and on repeated failure record the error in
        the lineage and stop gracefully so the run stays resumable.
        """
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await self.proposer.propose_async(self.space, incumbent, records)
            except Exception as e:  # noqa: BLE001 - any proposer-call failure is handled the same way
                last_error = e
                logger.warning("attempt=<%d>, error=<%s> | proposer call failed, retrying", attempt + 1, e)
        self.run_directory.log_event("proposer_error", iteration=iteration, error=str(last_error), attempts=3)
        logger.error("iteration=<%d>, error=<%s> | proposer failed after retries, stopping run", iteration, last_error)
        return []

    # -- scoring -----------------------------------------------------------------------

    async def _score_candidate(self, candidate: Candidate, case_names: list[str] | None = None) -> CandidateScore:
        """Score a candidate with k repeated runs, reusing every cached report.

        `case_names` restricts scoring to a case subset (selection plus holdout for a holdout-only
        challenger); `None` scores the full set. Rollouts are counted per case actually executed,
        so a case whose result is already in the per-case store (for example the selection and
        holdout cases of a challenger being topped up to the full set) costs nothing.
        """
        names = case_names if case_names is not None else self._case_names
        tag = self._report_tag(case_names)

        cached = self._scores.get(candidate.id)
        if cached is not None and all(n in cached.per_case for n in names):
            return cached

        experiment = self.experiment if case_names is None else self._subset_experiment(names)
        reports: list[EvaluationReport] = []
        for run_index in range(self.gate.runs_per_candidate):
            report_path = self.run_directory.report_path(candidate.id, run_index, tag)
            if report_path.exists():
                reports.append(EvaluationReport.model_validate_json(report_path.read_text()))
                continue
            store = LocalFileTaskResultStore(self.run_directory.scores_dir(candidate.id, run_index))
            executed = sum(1 for n in names if store.load(n) is None)
            task = self.build_task(self._with_context(candidate))
            report = await experiment.run_evaluations_async(
                task, max_workers=self.max_workers, evaluation_data_store=store
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report.model_dump_json(indent=2))
            self._rollouts_spent += executed
            self.run_directory.log_event(
                "score",
                candidate_id=candidate.id,
                run_index=run_index,
                tag=tag,
                rollouts=executed,
                overall_score=report.overall_score,
            )
            reports.append(report)

        score = self._aggregate(candidate.id, reports)
        if cached is not None:
            # Merge a subset score into an existing one (for example topping a challenger up to
            # the full set after acceptance) so the current best carries train and holdout both.
            score.per_case = {**cached.per_case, **score.per_case}
            score.per_case_runs = {**cached.per_case_runs, **score.per_case_runs}
            score.case_rows = {**cached.case_rows, **score.case_rows}
            score.error_rollouts += cached.error_rollouts
        self._scores[candidate.id] = score
        return score

    def _subset_experiment(self, case_names: list[str]) -> Experiment:
        # Rebuilt from cases and evaluators only: any future Experiment configuration beyond
        # those two fields would be dropped here and needs this helper updated.
        wanted = set(case_names)
        cases = [c for c in self.experiment.cases if c.name in wanted]
        return Experiment(cases=cases, evaluators=self.experiment.evaluators)

    def _aggregate(self, candidate_id: str, reports: list[EvaluationReport]) -> CandidateScore:
        """Reduce k reports to per-case means.

        Error rows (a crashed rollout or a crashed judge) are excluded and counted rather than
        scored 0.0, and rows with nothing to judge are dropped the way
        `EvaluationReport.calculate_overall_score` drops them. A crashed rollout must not look
        like a bad prompt.

        `case_rows` deliberately keeps only the most recent run's rows per case: the reflection
        records show the proposer one representative behavior, not k near-duplicates.
        """
        score = CandidateScore(candidate_id=candidate_id)
        run_case_scores: dict[str, list[float]] = {}
        for report in reports:
            by_case: dict[str, list[float]] = {}
            rows_by_case: dict[str, list[dict[str, Any]]] = {}
            for i, case_row in enumerate(report.cases):
                name = case_row.get("name")
                if name is None:
                    continue
                reason = report.reasons[i] if i < len(report.reasons) else ""
                detailed = report.detailed_results[i] if i < len(report.detailed_results) else []
                if reason.startswith(_ERROR_REASON_PREFIXES):
                    score.error_rollouts += 1
                    continue
                if detailed and not EvaluationReport.is_applicable(detailed):
                    continue
                by_case.setdefault(name, []).append(report.scores[i])
                rows_by_case.setdefault(name, []).append(
                    {
                        "evaluator": case_row.get("evaluator"),
                        "score": report.scores[i],
                        "reason": reason,
                        "input": case_row.get("input"),
                        "actual_output": case_row.get("actual_output"),
                    }
                )
            for name, values in by_case.items():
                run_case_scores.setdefault(name, []).append(statistics.fmean(values))
                score.case_rows[name] = rows_by_case[name]
        for name, values in run_case_scores.items():
            score.per_case_runs[name] = values
            score.per_case[name] = statistics.fmean(values)
        return score

    # -- helpers -----------------------------------------------------------------------

    def _seed_candidate(self) -> Candidate:
        base = Candidate(self.space.baseline_values(), rationale="baseline")
        return base.with_context(
            space=self.space, baseline=base, workspace_dir=self.run_directory.workspace_dir(base.id)
        )

    def _with_context(self, candidate: Candidate, *, baseline: Candidate | None = None) -> Candidate:
        """Attach the run's search context to a candidate, returning a bound copy."""
        return candidate.with_context(
            space=self.space, baseline=baseline, workspace_dir=self.run_directory.workspace_dir(candidate.id)
        )

    def _candidate_from_values(
        self, candidate_id: str, *, parent: Candidate, rationale: str | None = None
    ) -> Candidate:
        values = self.run_directory.load_values(candidate_id)
        if values is None:
            raise FileNotFoundError(f"candidate {candidate_id} has no stored values in {self.run_directory.root}")
        manifest = self.run_directory.read_manifest(candidate_id) or {}
        candidate = Candidate(values, parent_id=manifest.get("parent_id", parent.id), rationale=rationale)
        return self._with_context(candidate, baseline=parent.baseline or parent)

    async def _reload_pool(self, baseline: Candidate) -> dict[str, Candidate]:
        """Rebuild the pool, in admission order, from the lineage. A fresh run's pool is the baseline.

        Every pool member's reports are already on disk, so re-scoring them is a pure cache
        replay. `accept` events count as admissions too, so the champion of a run recorded before
        frontier admissions existed is still a pool member.
        """
        pool: dict[str, Candidate] = {baseline.id: baseline}
        for event in self.run_directory.load_events():
            candidate_id = event.get("candidate_id")
            if event.get("event") not in ("admit", "accept") or candidate_id is None or candidate_id in pool:
                continue
            candidate = self._candidate_from_values(candidate_id, parent=baseline)
            if self.run_directory.report_path(candidate_id, 0, "full").exists():
                await self._score_candidate(candidate)
            else:
                await self._score_candidate(candidate, self._challenger_cases())
            pool[candidate_id] = candidate
        best_id = self.run_directory.get_best()
        if best_id is not None and best_id not in pool:
            candidate = self._candidate_from_values(best_id, parent=baseline)
            await self._score_candidate(candidate)
            pool[best_id] = candidate
        return pool

    def _validate_resume(self, config: dict[str, Any]) -> None:
        """Fail fast when the caller resumes with different cases or artifacts than recorded.

        Without this, a mismatched pairing fails obscurely later: holdout names missing from the
        new experiment would make the gate compare nothing and reject everything.
        """
        recorded_cases = (
            set(config["train_cases"]) | set(config.get("selection_cases", [])) | set(config["holdout_cases"])
        )
        current_cases = set(self._case_names)
        if recorded_cases != current_cases:
            raise ValueError(
                f"resumed experiment does not match the recorded run: "
                f"missing cases {sorted(recorded_cases - current_cases)}, "
                f"new cases {sorted(current_cases - recorded_cases)}. "
                f"Changed cases need a fresh run (the train/holdout split would be stale)."
            )
        recorded_artifacts = set(config.get("artifacts", {}))
        current_artifacts = set(self.space.names())
        if recorded_artifacts != current_artifacts:
            raise ValueError(
                f"resumed artifacts do not match the recorded run: "
                f"recorded {sorted(recorded_artifacts)}, supplied {sorted(current_artifacts)}"
            )

    def _cost_per_candidate(self) -> int:
        challenger_cases = self._challenger_cases()
        cases = len(challenger_cases) if challenger_cases is not None else len(self._case_names)
        return self.gate.runs_per_candidate * cases

    def _report_tag(self, case_names: list[str] | None) -> str:
        """Name the case subset a scoring run covered, so subset reports never shadow full ones."""
        if case_names is None:
            return "full"
        if sorted(case_names) == sorted(self.holdout_names):
            return "holdout"
        if set(case_names) == set(self.selection_names) | set(self.holdout_names):
            return "challenge"
        return "subset"

    def _can_afford(self, rollouts: int) -> bool:
        return self._rollouts_spent + rollouts <= self.budget.max_rollouts

    def _replay_spend(self) -> int:
        return sum(
            event.get("rollouts", 0) for event in self.run_directory.load_events() if event.get("event") == "score"
        )

    def _split_cases(self) -> tuple[list[str], list[str], list[str]]:
        """Split cases into train, selection, and holdout, returning them in that order.

        The holdout is carved first, so a given seed holds out the same cases however the
        selection fraction is set. At least one training case always remains; when no selection
        case fits beside it, the training split doubles as the selection split.
        """
        names = list(self._case_names)
        rng = random.Random(self.seed)
        rng.shuffle(names)
        total = len(names)
        holdout_count = min(total - 1, max(1, round(self.gate.holdout_fraction * total)))
        selection_count = min(total - holdout_count - 1, max(0, round(self.gate.selection_fraction * total)))
        holdout = sorted(names[:holdout_count])
        selection = sorted(names[holdout_count : holdout_count + selection_count])
        train = sorted(names[holdout_count + selection_count :])
        if not selection:
            logger.warning(
                "cases=<%d> | too few cases for a separate selection split, the frontier ranks on train",
                total,
            )
            selection = list(train)
        return train, selection, holdout
