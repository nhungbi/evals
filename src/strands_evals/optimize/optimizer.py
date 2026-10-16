"""The optimizer loop: exploration chooses the parent, the gate chooses the champion.

Every admitted candidate stays in a pool. Each iteration, the case frontier (the pool members
that are best on at least one selection case) supplies a parent: either two frontier members are
merged variable by variable, or one is sampled, weighted by the cases it leads, and the
schedule's block proposes typed changes to it. Each proposal is built fresh and scored with k
repeated runs through the caller's own `Experiment` (usage metered per trial), and the scoring
layer reduces per-case metrics to one objective scalar.

Acceptance then has two steps. Admission (loose, on the selection split): a proposal joins the
pool when it breaks no constraint and would land on the case frontier. The champion gate (strict,
on the holdout): the one admitted proposal with the best selection mean, if that mean beats the
champion's, must meet every constraint on the holdout and clear the paired bootstrap test. Every
outcome is fed back to the proposing method as its selection-split scalar; only the gate reads
the holdout.

The loop never patches a built agent, never skips the gate, and never runs unbounded.
"""

import asyncio
import json
import logging
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from ..case import Case
from ..experiment import Experiment
from ..local_file_task_result_store import LocalFileTaskResultStore
from ..types.evaluation_report import EvaluationReport
from .artifacts import Artifact, ArtifactSpace
from .candidate import Candidate
from .execution import TrialLedger, default_rollout, instrumented
from .factory import TunableFactory
from .frontier import pareto_front, sample_parent
from .gate import AcceptanceVerdict, Budget, Gate, paired_bootstrap_compare
from .merge import MergePlan, find_merge_pair
from .methods import Observation, Patch, build_reflection_records
from .objectives import Combine, Constraint, Objective, Projection
from .runs import RunDirectory
from .search import Block, Search, default_search
from .space import Space
from .trajectories import render_trajectory

logger = logging.getLogger(__name__)

_ERROR_REASON_PREFIXES = ("An error occurred:", "Evaluator error:")

# Below this many held-out cases, judge noise buys too many false accepts (measured at 8-10%
# false positives with 5 cases in the gate calibration experiments).
_MIN_HOLDOUT_FOR_QUIET_GATE = 5

# Selection has no configuration in v1. A case is led only by exact ties for its best value, and
# a run takes at most this many merge steps (each pair of frontier members is merged once).
_FRONTIER_EPS = 0.0
_MAX_MERGES = 5

ProposalStatus = Literal["proposed", "unscored", "scored", "infeasible", "dominated", "admitted", "accepted"]


@dataclass
class CandidateScore:
    """One candidate's measurements: per-case objective scalars and per-case metric values.

    `per_case` holds the projection's scalar (what the frontier and the gate compare).
    `per_metric` holds every metric's per-case value: each evaluator's score plus the usage
    metrics from the trial ledger.
    """

    candidate_id: str
    per_case: dict[str, float] = field(default_factory=dict)
    per_metric: dict[str, dict[str, float]] = field(default_factory=dict)
    per_case_runs: dict[str, list[float]] = field(default_factory=dict)
    case_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    error_rollouts: int = 0
    trial_statuses: dict[str, int] = field(default_factory=dict)

    def mean(self, case_names: list[str] | None = None) -> float:
        """Mean per-case scalar over `case_names` (all scored cases when omitted)."""
        names = [n for n in (case_names or list(self.per_case)) if n in self.per_case]
        if not names:
            return 0.0
        return sum(self.per_case[n] for n in names) / len(names)

    def subset(self, case_names: list[str]) -> dict[str, float]:
        return {n: self.per_case[n] for n in case_names if n in self.per_case}

    def metrics_subset(self, case_names: list[str]) -> dict[str, dict[str, float]]:
        """Per-case metric values restricted to `case_names`, keyed case -> metric -> value."""
        out: dict[str, dict[str, float]] = {}
        for metric, by_case in self.per_metric.items():
            for name in case_names:
                if name in by_case:
                    out.setdefault(name, {})[metric] = by_case[name]
        return out


@dataclass
class ProposalOutcome:
    """What happened to one proposed candidate in one iteration.

    `status` is the final state: `unscored` means the budget ran out first; `infeasible` that it
    broke a constraint on the selection split (see `constraint_violations`); `dominated` that it
    was scored but did not land on the case frontier; `admitted` that it joined the pool, so it
    can be a parent (it may have challenged the champion and lost: see `verdict` and
    `constraint_violations`); `accepted` that it also cleared the gate. `parent_ids` lists every
    parent: one for a method's proposal, two for a merge.
    """

    candidate_id: str
    block: str
    method: str
    changed_keys: list[str]
    rationale: str
    diff: str
    parent_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    train_score: float | None = None
    selection_score: float | None = None
    holdout_score: float | None = None
    cases_led: list[str] = field(default_factory=list)
    verdict: AcceptanceVerdict | None = None
    constraint_violations: list[str] = field(default_factory=list)
    status: ProposalStatus = "proposed"
    error_rollouts: int = 0
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """One loop iteration, in the order things happened. Fed to `on_iteration` as it completes.

    The `incumbent_*` fields describe the champion at the start of the iteration. `step` is
    `propose` when the schedule's blocks proposed on a sampled parent and `merge` when two
    frontier members were merged instead. `frontier` maps each frontier member to the number of
    selection cases it led at selection. `rollouts_spent` is this iteration's spend.
    """

    iteration: int
    incumbent_id: str
    incumbent_train_score: float
    incumbent_holdout_score: float
    incumbent_values: dict[str, Any] = field(default_factory=dict)
    step: Literal["propose", "merge"] = "propose"
    parent_id: str = ""
    frontier: dict[str, int] = field(default_factory=dict)
    blocks: list[str] = field(default_factory=list)
    proposals: list[ProposalOutcome] = field(default_factory=list)
    accepted_candidate_id: str | None = None
    rollouts_spent: int = 0


@dataclass
class OptimizationResult:
    """The answer to the developer's two questions: did it get better, and what changed.

    `best` is the champion: the candidate that last cleared the gate. `frontier` maps each final
    case-frontier member to the selection cases it leads. `accepted_count` counts champion
    changes and `rejected_count` lost champion challenges (gate or holdout constraints);
    `admitted_count`, `dominated_count`, and `infeasible_count` count admission outcomes, and
    `merge_count` counts merge steps.
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
    admitted_count: int = 0
    dominated_count: int = 0
    infeasible_count: int = 0
    merge_count: int = 0
    frontier: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        delta = self.best_holdout_score - self.baseline_holdout_score
        return (
            f"baseline {self.baseline_holdout_score:.2f} -> best {self.best_holdout_score:.2f} "
            f"on the held-out split ({delta:+.2f})\n"
            f"{self.accepted_count} edits accepted, {self.rejected_count} rejected, "
            f"{self.admitted_count} admitted to the pool, {self.dominated_count} dominated, "
            f"{self.infeasible_count} infeasible, {self.merge_count} merges | "
            f"{self.rollouts_spent} rollouts | run_id {self.run_id}"
        )

    def reports(self, candidate_id: str) -> dict[str, EvaluationReport]:
        """The `EvaluationReport`s stored while scoring `candidate_id`, keyed by report name."""
        scores_dir = RunDirectory(self.run_dir).candidate_dir(candidate_id) / "scores"
        return {
            path.name.removesuffix(".report.json"): EvaluationReport.model_validate_json(path.read_text())
            for path in sorted(scores_dir.glob("run*.report.json"))
        }

    def to_rows(self, grain: Literal["iteration", "proposal", "case"] = "proposal") -> list[dict[str, Any]]:
        """Flatten the run into JSON-serializable rows, one dict per row.

        `grain="iteration"` gives the convergence view, `grain="proposal"` the admission and
        gate view (one row per proposed candidate with its parents, per-split scores, cases led,
        verdict, constraint violations, and applied typed values), and `grain="case"` every
        stored report row joined with candidate lineage.
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
                        "step": record.step,
                        "parent_id": record.parent_id,
                        "frontier_size": len(record.frontier),
                        "blocks": list(record.blocks),
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
                            "parent_ids": list(outcome.parent_ids),
                            "block": outcome.block,
                            "method": outcome.method,
                            "changed_keys": list(outcome.changed_keys),
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
                            "constraint_violations": list(outcome.constraint_violations),
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


@dataclass
class _Draft:
    """One proposed candidate before registration: where it came from and what to diff it against."""

    candidate: Candidate
    parent_ids: list[str]
    block: str
    method: str
    rationale: str
    diff_base: Candidate
    patch: Patch | None = None


class Optimizer:
    """The loop: declared space in, gated improvements out.

    Args:
        experiment: The cases and evaluators to optimize against. Every case needs a unique name.
        factory: A `tunable` construction function (either mount). Its markers become the typed
            space, and each rollout builds a fresh agent with the candidate's values bound.
        rollout_fn: Called as `rollout_fn(agent_factory, case)` per rollout. May be async. When
            omitted, the default rollout builds one agent and invokes it once with the case's
            input, so the simplest run writes no rollout code. Return
            `{"output": ..., "trajectory": ...}` to show the proposer what the agent did.
        artifacts: File/text artifacts merged into the space next to the factory's markers (an
            `ArtifactSpace` or a plain mapping). Rollouts that must read a materialized workspace
            use the `build_task` form instead.
        objectives: `None` (the default: maximize the mean of all evaluator scores), one
            `Maximize`/`Minimize`, or a scalar combiner (`WeightedSum`). The case frontier and
            the gate both compare this scalar.
        constraints: `AtMost`/`AtLeast` bounds. Checked at admission on the selection split (a
            violator never joins the pool) and at the gate on the holdout. Constraints never
            accept anything.
        search: `Search(blocks=..., schedule=...)`. When omitted, one default block per variable
            type (text to `Reflection`, numbers and choices to `TPE`) under an `Alternate`
            schedule. The blocks always propose on the parent the case frontier supplies.
        gate: Acceptance configuration. Always on; see `Gate`.
        budget: Rollout and iteration caps. A conservative default cap applies when unset.
        proposals_per_iteration: How many candidates each proposing block is asked for.
        retries: How many extra times a trial reruns after an infrastructure failure.
        max_workers: Parallel workers for `run_evaluations_async`.
        base_dir: Where run directories are created. Defaults to `.strands-evals/optimize/runs`.
        run_dir: An existing run directory to resume (what `Optimizer.resume` wraps).
        on_iteration: Optional callback invoked with each completed `IterationRecord`.
        seed: Seed for the case split, parent sampling, and the bootstrap, recorded in
            `run.json`.
        build_task: Advanced execution path: builds a task callable from a candidate directly
            (container fleets, AgentCore Runtime, workspace-reading rollouts). Requires
            `artifacts` and excludes `factory`/`rollout_fn`.
    """

    def __init__(
        self,
        experiment: Experiment,
        factory: TunableFactory | None = None,
        *,
        rollout_fn: Callable[[Callable[..., Any], Case], Any] | None = None,
        artifacts: ArtifactSpace | Mapping[str, Artifact] | None = None,
        objectives: Objective | Combine | None = None,
        constraints: tuple[Constraint, ...] | list[Constraint] = (),
        search: Search | None = None,
        gate: Gate | None = None,
        budget: Budget | None = None,
        proposals_per_iteration: int = 3,
        retries: int = 2,
        max_workers: int = 4,
        base_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        on_iteration: Callable[[IterationRecord], None] | None = None,
        seed: int = 0,
        build_task: Callable[[Candidate], Callable[[Case], Any]] | None = None,
    ):
        if factory is not None and not isinstance(factory, TunableFactory):
            raise TypeError(
                "factory must be a tunable factory: decorate it with @tunable(...) or wrap it with "
                "tunable(fn, inject={...})"
            )
        if build_task is not None:
            if factory is not None or rollout_fn is not None:
                raise ValueError("pass factory (with optional rollout_fn), or build_task with artifacts, not both")
            if artifacts is None:
                raise ValueError("build_task needs artifacts: the space the proposer may edit")
            self.build_task = build_task
            self.space = Space(artifacts=artifacts)
        elif factory is not None:
            self.build_task = factory.build_task(rollout_fn or default_rollout)
            self.space = Space(markers=factory.markers, artifacts=artifacts)
        else:
            raise ValueError("pass a tunable factory, or build_task with artifacts")

        self.experiment = experiment
        self.budget = budget or Budget()
        self.gate = gate or Gate()
        self.retries = retries
        self.proposals_per_iteration = proposals_per_iteration
        self.max_workers = max_workers
        self.on_iteration = on_iteration
        self.seed = seed

        evaluator_names = [_evaluator_name(evaluator) for evaluator in experiment.evaluators]
        self.projection = Projection(objectives, constraints, evaluator_names=evaluator_names)
        self.search = search or default_search(self.space)
        self._blocks: list[tuple[Block, list[str]]] = self.search.resolved(self.space)
        if not self._blocks:
            raise ValueError("the search has no blocks: nothing can be proposed")

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
                    "space": {info.key: info.kind for info in self.space.params()},
                    "objectives": self.projection.metric_names() if objectives is not None else "evaluator-mean",
                    "constraints": [constraint.describe() for constraint in self.projection.constraints],
                    "blocks": {block.name: keys for block, keys in self._blocks},
                },
            )
        if len(self.holdout_names) < _MIN_HOLDOUT_FOR_QUIET_GATE:
            logger.warning(
                "holdout_cases=<%d> | small holdouts inflate false-positive accepts, %d or more recommended",
                len(self.holdout_names),
                _MIN_HOLDOUT_FOR_QUIET_GATE,
            )

        self._scores: dict[str, CandidateScore] = {}
        # Parent ids of every pool member, for finding merge ancestors; rebuilt on resume.
        self._parents: dict[str, tuple[str, ...]] = {}
        self._merged_pairs: set[frozenset[str]] = {
            frozenset(event["parent_ids"])
            for event in self.run_directory.load_events()
            if event.get("event") == "merge"
        }
        self._rollouts_spent = self._replay_spend()

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        experiment: Experiment,
        factory: TunableFactory | None = None,
        **kwargs: Any,
    ) -> "Optimizer":
        """Reopen a run directory and continue it. Already-scored candidates are cache hits."""
        return cls(experiment, factory, run_dir=run_dir, **kwargs)

    def run_sync(self) -> OptimizationResult:
        """Synchronous convenience wrapper around `run()`, for scripts and notebooks."""
        return asyncio.run(self.run())

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
        champion_score = self._scores[champion.id]

        # Counters and iteration numbering replay from the lineage, so a resumed run continues
        # the sequence instead of restarting at iteration 1 and under-reporting totals.
        prior_events = self.run_directory.load_events()
        iterations: list[IterationRecord] = []
        accepted_count = sum(1 for e in prior_events if e["event"] == "accept")
        rejected_count = sum(1 for e in prior_events if e["event"] in ("reject", "constraint_reject"))
        admitted_count = sum(1 for e in prior_events if e["event"] == "admit")
        dominated_count = sum(1 for e in prior_events if e["event"] == "dominated")
        infeasible_count = sum(1 for e in prior_events if e["event"] == "infeasible")
        merge_count = sum(1 for e in prior_events if e["event"] == "merge")
        iteration = max((e.get("iteration", 0) for e in prior_events), default=0)

        while True:
            iteration += 1
            if self.budget.max_iterations is not None and iteration > self.budget.max_iterations:
                logger.info("iteration=<%d> | iteration cap reached, stopping", iteration)
                break
            if not self._can_afford(self._cost_per_candidate()):
                logger.info("rollouts_spent=<%d> | budget exhausted, stopping", self._rollouts_spent)
                break
            spent_before = self._rollouts_spent

            # Selection: merge two complementary frontier members when an untried pair exists and
            # the run has merges left; otherwise sample one member, weighted by the cases it leads.
            front = self._frontier(pool)
            plan = self._merge_plan(front, pool, merge_count)
            if plan is not None:
                parent = pool[plan.first]
            else:
                parent = pool[sample_parent(front, self._parent_rng(iteration))] if front else champion
                if not await self._ensure_fully_scored(parent):
                    logger.info("candidate=<%s> | budget cannot cover the parent's training cases, stopping", parent.id)
                    break
            parent_score = self._scores[parent.id]
            step: Literal["propose", "merge"] = "merge" if plan is not None else "propose"
            self.run_directory.log_event(
                "select",
                iteration=iteration,
                step=step,
                parent_id=parent.id,
                frontier={candidate_id: sorted(cases) for candidate_id, cases in front.items()},
            )
            record = IterationRecord(
                iteration=iteration,
                incumbent_id=champion.id,
                incumbent_train_score=champion_score.mean(self.train_names),
                incumbent_holdout_score=champion_score.mean(self.holdout_names),
                incumbent_values=dict(champion.values),
                step=step,
                parent_id=parent.id,
                frontier={candidate_id: len(cases) for candidate_id, cases in front.items()},
            )

            # Proposal: the merged candidate, or the moving blocks' patches of the parent.
            drafts: list[_Draft] = []
            if plan is not None:
                merge_count += 1
                self._merged_pairs.add(frozenset((plan.first, plan.second)))
                self.run_directory.log_event(
                    "merge",
                    iteration=iteration,
                    parent_ids=[plan.first, plan.second],
                    ancestor_id=plan.ancestor,
                    from_first=list(plan.from_first),
                    from_second=list(plan.from_second),
                )
                rationale = (
                    f"merge of {plan.first} ({', '.join(plan.from_first)}) and "
                    f"{plan.second} ({', '.join(plan.from_second)}) against ancestor {plan.ancestor}"
                )
                merged = Candidate(
                    plan.values, parent_id=plan.first, rationale=rationale, space=self.space, baseline=baseline
                )
                drafts.append(
                    _Draft(merged, [plan.first, plan.second], "merge", "merge", rationale, pool[plan.ancestor])
                )
            else:
                moving = self.search.schedule.blocks_for(iteration, [block for block, _ in self._blocks])
                record.blocks = [block.name for block in moving]
                records = build_reflection_records(parent_score.case_rows, parent_score.subset(self.train_names))
                patches = await self._propose(moving, parent, records, iteration)
                if not patches:
                    logger.info("iteration=<%d> | no block produced a valid proposal, stopping", iteration)
                    break
                for patch in patches:
                    candidate = parent.mutate(patch.updates, rationale=patch.rationale)
                    block_name = getattr(patch, "block", "")
                    drafts.append(
                        _Draft(candidate, [parent.id], block_name, patch.method, patch.rationale, parent, patch)
                    )

            outcomes, candidates_by_id = self._register(drafts, pool, iteration)
            await self._score_proposals(outcomes, candidates_by_id)

            # Admission (loose): a feasible proposal joins the pool when it lands on the case
            # frontier. The best selection means go first, so they win exact ties within the batch.
            scored = [o for o in outcomes if o.status == "scored"]
            scored.sort(key=lambda o: o.selection_score or 0.0, reverse=True)
            admitted: list[ProposalOutcome] = []
            for outcome in scored:
                score = self._scores[outcome.candidate_id]
                violations = self.projection.check_constraints(score.metrics_subset(self.selection_names))
                if violations:
                    outcome.status = "infeasible"
                    outcome.constraint_violations = violations
                    infeasible_count += 1
                    self.run_directory.set_status(outcome.candidate_id, "infeasible")
                    self.run_directory.log_event(
                        "infeasible", candidate_id=outcome.candidate_id, iteration=iteration, violations=violations
                    )
                    continue
                cases_led = self._admission_cases(outcome.candidate_id, parent_score, pool)
                if cases_led is None:
                    outcome.status = "dominated"
                    dominated_count += 1
                    self.run_directory.set_status(outcome.candidate_id, "dominated")
                    self.run_directory.log_event(
                        "dominated", candidate_id=outcome.candidate_id, parent_id=outcome.parent_id, iteration=iteration
                    )
                    continue
                outcome.status = "admitted"
                outcome.cases_led = cases_led
                admitted_count += 1
                pool[outcome.candidate_id] = self._with_context(candidates_by_id[outcome.candidate_id])
                self._parents[outcome.candidate_id] = tuple(outcome.parent_ids)
                admitted.append(outcome)
                self.run_directory.set_status(outcome.candidate_id, "admitted")
                self.run_directory.log_event(
                    "admit",
                    candidate_id=outcome.candidate_id,
                    parent_id=outcome.parent_id,
                    parent_ids=outcome.parent_ids,
                    iteration=iteration,
                    cases_led=cases_led,
                )

            # Champion gate (strict): only the strongest admission whose selection mean beats the
            # champion's challenges it, so the holdout is consulted at most once per iteration.
            champion_selection = champion_score.mean(self.selection_names)
            challengers = [o for o in admitted if (o.selection_score or 0.0) > champion_selection]
            if challengers:
                outcome = challengers[0]
                challenger_score = self._scores[outcome.candidate_id]
                violations = self.projection.check_constraints(challenger_score.metrics_subset(self.holdout_names))
                if violations:
                    outcome.constraint_violations = violations
                    rejected_count += 1
                    self.run_directory.log_event(
                        "constraint_reject",
                        candidate_id=outcome.candidate_id,
                        champion_id=champion.id,
                        iteration=iteration,
                        violations=violations,
                    )
                else:
                    verdict = paired_bootstrap_compare(
                        challenger_score.subset(self.holdout_names),
                        champion_score.subset(self.holdout_names),
                        margin=self.gate.margin,
                        confidence=self.gate.confidence,
                        min_coverage=self.gate.min_coverage,
                        seed=self.seed,
                    )
                    outcome.verdict = verdict
                    self.run_directory.log_event(
                        "accept" if verdict.accepted else "reject",
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
                        # Top the new champion up to the full case set so the records report its
                        # training score; its other cases hit the cache.
                        if not await self._ensure_fully_scored(champion):
                            logger.warning(
                                "candidate=<%s> | budget cannot cover the new champion's training cases, "
                                "its training score covers only the cases already scored",
                                champion.id,
                            )
                        champion_score = self._scores[champion.id]
                        record.accepted_candidate_id = outcome.candidate_id
                    else:
                        rejected_count += 1

            # Every patch is told its outcome exactly once, including duplicates (told the existing
            # candidate's result) and drafts the space rejected (told nothing was measured).
            outcomes_by_id = {o.candidate_id: o for o in outcomes}
            for draft in drafts:
                if draft.patch is not None:
                    self._observe(draft.patch, draft.candidate.id, outcomes_by_id.get(draft.candidate.id))

            record.proposals = outcomes
            record.rollouts_spent = self._rollouts_spent - spent_before
            iterations.append(record)
            if self.on_iteration is not None:
                self.on_iteration(record)

        best_id = self.run_directory.get_best() or baseline.id
        best = pool.get(best_id) or self._candidate_from_values(best_id, baseline=baseline)
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
            infeasible_count=infeasible_count,
            merge_count=merge_count,
            frontier={candidate_id: sorted(cases) for candidate_id, cases in self._frontier(pool).items()},
        )

    # -- selection -----------------------------------------------------------------------

    def _frontier(self, pool: Mapping[str, Candidate]) -> dict[str, frozenset[str]]:
        """The pool's case frontier on the selection split, in pool order."""
        return pareto_front(
            {candidate_id: self._scores[candidate_id].subset(self.selection_names) for candidate_id in pool},
            self.selection_names,
            _FRONTIER_EPS,
        )

    def _merge_plan(
        self, front: Mapping[str, frozenset[str]], pool: Mapping[str, Candidate], merges_done: int
    ) -> MergePlan | None:
        """The next merge to run, or `None` when merging is off, used up, or no pair qualifies.

        A planned merge whose values the space rejects is marked tried and skipped, so the run
        never retries an impossible combination.
        """
        if merges_done >= _MAX_MERGES or len(front) < 2:
            return None
        while True:
            plan = find_merge_pair(
                front,
                {candidate_id: candidate.values for candidate_id, candidate in pool.items()},
                self._parents,
                self._merged_pairs,
                group_of=self.space.entry_name,
            )
            if plan is None:
                return None
            try:
                self.space.validate(plan.values)
                return plan
            except ValueError as error:
                logger.warning(
                    "first=<%s>, second=<%s>, error=<%s> | merged values rejected by the space",
                    plan.first,
                    plan.second,
                    error,
                )
                self._merged_pairs.add(frozenset((plan.first, plan.second)))

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
        front = pareto_front(trial, self.selection_names, _FRONTIER_EPS)
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
        """The cases a proposal is scored on: every case, or selection plus holdout only."""
        if not self.gate.challenger_holdout_only:
            return None
        return sorted(set(self.selection_names) | set(self.holdout_names))

    # -- proposing -----------------------------------------------------------------------

    async def _propose(
        self, moving: list[Block], parent: Candidate, records: list[dict[str, Any]], iteration: int
    ) -> list[Patch]:
        """Ask every moving block's method for patches of `parent`, with transient-failure retries."""
        keys_by_block = {block.name: keys for block, keys in self._blocks}
        patches: list[Patch] = []
        for block in moving:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    proposed = await block.method.propose(
                        self.space, keys_by_block[block.name], parent, records, self.proposals_per_iteration
                    )
                    for patch in proposed:
                        patch.method = patch.method or type(block.method).__name__
                        patch.block = block.name  # type: ignore[attr-defined]
                    patches.extend(proposed)
                    last_error = None
                    break
                except (NotImplementedError, ImportError):
                    raise
                except Exception as error:  # noqa: BLE001 - any proposer-call failure retries the same way
                    last_error = error
                    logger.warning(
                        "block=<%s>, attempt=<%d>, error=<%s> | propose failed, retrying",
                        block.name,
                        attempt + 1,
                        error,
                    )
            if last_error is not None:
                self.run_directory.log_event(
                    "proposer_error", iteration=iteration, block=block.name, error=str(last_error), attempts=3
                )
                logger.error("block=<%s>, error=<%s> | propose failed after retries", block.name, last_error)
        return patches

    def _register(
        self, drafts: list[_Draft], pool: Mapping[str, Candidate], iteration: int
    ) -> tuple[list[ProposalOutcome], dict[str, Candidate]]:
        """Validate, deduplicate, and record each draft. Drafts already in the pool are dropped."""
        outcomes: list[ProposalOutcome] = []
        candidates_by_id: dict[str, Candidate] = {}
        for draft in drafts:
            candidate = draft.candidate
            try:
                self.space.validate(candidate.values)
            except ValueError as error:
                logger.warning("candidate=<%s>, error=<%s> | proposal rejected by the space", candidate.id, error)
                continue
            if candidate.id in candidates_by_id or candidate.id in pool:
                logger.info("candidate=<%s> | duplicate proposal content, dropped", candidate.id)
                continue
            candidates_by_id[candidate.id] = candidate
            changed = sorted(
                key
                for key in set(candidate.values) | set(draft.diff_base.values)
                if candidate.values.get(key) != draft.diff_base.values.get(key)
            )
            self.run_directory.save_values(candidate.id, candidate.values)
            self.run_directory.write_manifest(
                candidate.id,
                parent_id=draft.parent_ids[0],
                parent_ids=draft.parent_ids if len(draft.parent_ids) > 1 else None,
                rationale=draft.rationale,
                values=candidate.values,
                status="proposed",
            )
            self.run_directory.log_event(
                "propose",
                candidate_id=candidate.id,
                parent_id=draft.parent_ids[0],
                parent_ids=draft.parent_ids,
                block=draft.block,
                changed=changed,
                rationale=draft.rationale,
                iteration=iteration,
            )
            outcomes.append(
                ProposalOutcome(
                    candidate_id=candidate.id,
                    block=draft.block,
                    method=draft.method,
                    changed_keys=changed,
                    rationale=draft.rationale,
                    diff=candidate.diff(draft.diff_base),
                    parent_id=draft.parent_ids[0],
                    parent_ids=list(draft.parent_ids),
                    values=dict(candidate.values),
                )
            )
        return outcomes, candidates_by_id

    async def _score_proposals(self, outcomes: list[ProposalOutcome], candidates_by_id: dict[str, Candidate]) -> None:
        for outcome in outcomes:
            if not self._can_afford(self._cost_per_candidate()) and outcome.candidate_id not in self._scores:
                outcome.status = "unscored"
                self.run_directory.set_status(outcome.candidate_id, "unscored")
                continue
            candidate = self._with_context(candidates_by_id[outcome.candidate_id])
            score = await self._score_candidate(candidate, self._challenger_cases())
            outcome.train_score = None if self.gate.challenger_holdout_only else score.mean(self.train_names)
            outcome.selection_score = score.mean(self.selection_names)
            outcome.holdout_score = score.mean(self.holdout_names)
            outcome.error_rollouts = score.error_rollouts
            outcome.status = "scored"
            self.run_directory.set_status(outcome.candidate_id, "scored")

    def _observe(self, patch: Patch, candidate_id: str, outcome: ProposalOutcome | None) -> None:
        """Feed one patch's result back to its block's method, as the selection-split scalar.

        `outcome` is `None` when the draft was never registered: a duplicate of a scored
        candidate (told that candidate's result) or a draft the space rejected (told nothing was
        measured).
        """
        method = self._method_for(patch)
        if method is None:
            return
        score = self._scores.get(candidate_id)
        measured = score is not None and bool(score.subset(self.selection_names))
        scalar = score.mean(self.selection_names) if score is not None and measured else None
        verdict = outcome.verdict if outcome is not None else None
        if verdict is not None:
            reason = verdict.reason
        elif outcome is not None and outcome.constraint_violations:
            reason = "; ".join(outcome.constraint_violations)
        elif outcome is not None:
            reason = outcome.status
        else:
            reason = "duplicate of a measured candidate" if measured else "rejected by the space"
        method.observe(
            Observation(
                patch=patch,
                scalar=scalar,
                accepted=verdict.accepted if verdict is not None else None,
                crashed=score is not None and not measured,
                reason=reason,
            )
        )

    def _method_for(self, patch: Patch) -> Any:
        for block, _ in self._blocks:
            if block.name == getattr(patch, "block", None):
                return block.method
        return None

    # -- scoring -----------------------------------------------------------------------

    async def _score_candidate(self, candidate: Candidate, case_names: list[str] | None = None) -> CandidateScore:
        """Score a candidate with k repeated runs, reusing every cached report.

        Evaluator scores come from the experiment's reports; usage metrics come from the trial
        ledger. Both land per case in `per_metric`, and the projection reduces them to the
        objective scalar in `per_case`.
        """
        names = case_names if case_names is not None else self._case_names
        tag = self._report_tag(case_names)

        cached = self._scores.get(candidate.id)
        if cached is not None and all(n in cached.per_case for n in names):
            return cached

        experiment = self.experiment if case_names is None else self._subset_experiment(names)
        ledger = TrialLedger()
        reports: list[EvaluationReport] = []
        fresh_usage = False
        for run_index in range(self.gate.runs_per_candidate):
            report_path = self.run_directory.report_path(candidate.id, run_index, tag)
            if report_path.exists():
                reports.append(EvaluationReport.model_validate_json(report_path.read_text()))
                continue
            store = LocalFileTaskResultStore(self.run_directory.scores_dir(candidate.id, run_index))
            executed = sum(1 for n in names if store.load(n) is None)
            task = instrumented(self.build_task(candidate), ledger, run_index, retries=self.retries)
            fresh_usage = True
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

        usage = ledger.per_case_usage() if fresh_usage else (self.run_directory.load_usage(candidate.id) or {})
        if fresh_usage:
            previous = self.run_directory.load_usage(candidate.id) or {}
            usage = {**previous, **usage}
            self.run_directory.save_usage(candidate.id, usage)

        score = self._aggregate(candidate.id, reports, usage)
        score.trial_statuses = ledger.status_counts()
        if cached is not None:
            score.per_case = {**cached.per_case, **score.per_case}
            score.per_case_runs = {**cached.per_case_runs, **score.per_case_runs}
            score.case_rows = {**cached.case_rows, **score.case_rows}
            for metric, by_case in cached.per_metric.items():
                score.per_metric[metric] = {**by_case, **score.per_metric.get(metric, {})}
            score.error_rollouts += cached.error_rollouts
        self._scores[candidate.id] = score
        return score

    def _subset_experiment(self, case_names: list[str]) -> Experiment:
        # Rebuilt from cases and evaluators only: any future Experiment configuration beyond
        # those two fields would be dropped here and needs this helper updated.
        wanted = set(case_names)
        cases = [c for c in self.experiment.cases if c.name in wanted]
        return Experiment(cases=cases, evaluators=self.experiment.evaluators)

    def _aggregate(
        self, candidate_id: str, reports: list[EvaluationReport], usage: dict[str, dict[str, float]]
    ) -> CandidateScore:
        """Reduce k reports plus the usage ledger to per-case metrics and objective scalars.

        Error rows (a crashed rollout or a crashed judge) are excluded and counted rather than
        scored 0.0: a crashed rollout must not look like a bad prompt. `case_rows` keeps only the
        most recent run's rows per case, so reflection sees one representative behavior; a row's
        `trajectory` is the task's returned trajectory rendered as capped text, once per case, so
        the rows never hold a full `Session`.
        """
        score = CandidateScore(candidate_id=candidate_id)
        metric_runs: dict[str, dict[str, list[float]]] = {}
        for report in reports:
            by_case_metric: dict[str, dict[str, list[float]]] = {}
            rows_by_case: dict[str, list[dict[str, Any]]] = {}
            rendered: dict[str, str] = {}
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
                evaluator = case_row.get("evaluator") or "evaluator"
                by_case_metric.setdefault(name, {}).setdefault(evaluator, []).append(report.scores[i])
                if name not in rendered:
                    rendered[name] = render_trajectory(case_row.get("actual_trajectory"))
                rows_by_case.setdefault(name, []).append(
                    {
                        "evaluator": evaluator,
                        "score": report.scores[i],
                        "reason": reason,
                        "input": case_row.get("input"),
                        "actual_output": case_row.get("actual_output"),
                        "trajectory": rendered[name],
                    }
                )
            for name, by_metric in by_case_metric.items():
                for metric, values in by_metric.items():
                    metric_runs.setdefault(metric, {}).setdefault(name, []).append(statistics.fmean(values))
                score.case_rows[name] = rows_by_case[name]
        for metric, by_case in metric_runs.items():
            score.per_metric[metric] = {name: statistics.fmean(values) for name, values in by_case.items()}
        for name, metrics in usage.items():
            for metric, value in metrics.items():
                score.per_metric.setdefault(metric, {})[name] = value

        cases = {name for by_case in score.per_metric.values() for name in by_case}
        for name in cases:
            case_metrics = {metric: by_case[name] for metric, by_case in score.per_metric.items() if name in by_case}
            scalar = self.projection.scalar(case_metrics)
            if scalar is not None:
                score.per_case[name] = scalar
                score.per_case_runs[name] = [scalar]
        return score

    # -- helpers -----------------------------------------------------------------------

    def _seed_candidate(self) -> Candidate:
        base = Candidate(self.space.defaults(), rationale="baseline")
        return base.with_context(
            space=self.space, baseline=base, workspace_dir=self.run_directory.workspace_dir(base.id)
        )

    def _with_context(self, candidate: Candidate, *, baseline: Candidate | None = None) -> Candidate:
        return candidate.with_context(
            space=self.space, baseline=baseline, workspace_dir=self.run_directory.workspace_dir(candidate.id)
        )

    def _candidate_from_values(self, candidate_id: str, *, baseline: Candidate) -> Candidate:
        """Rebuild a stored candidate, with the primary parent its manifest recorded."""
        values = self.run_directory.load_values(candidate_id)
        if values is None:
            raise FileNotFoundError(f"candidate {candidate_id} has no stored values in {self.run_directory.root}")
        manifest = self.run_directory.read_manifest(candidate_id) or {}
        candidate = Candidate(values, parent_id=manifest.get("parent_id"), rationale=manifest.get("rationale"))
        return self._with_context(candidate, baseline=baseline)

    async def _reload_pool(self, baseline: Candidate) -> dict[str, Candidate]:
        """Rebuild the pool, in admission order, from the lineage. A fresh run's pool is the baseline.

        Every pool member's reports are already on disk, so re-scoring them is a pure cache
        replay. `accept` events count as admissions too, so the champion is always a pool member.
        """
        pool: dict[str, Candidate] = {baseline.id: baseline}
        self._parents = {baseline.id: ()}
        member_ids = [
            event["candidate_id"]
            for event in self.run_directory.load_events()
            if event.get("event") in ("admit", "accept") and event.get("candidate_id")
        ]
        best_id = self.run_directory.get_best()
        if best_id is not None:
            member_ids.append(best_id)
        for candidate_id in member_ids:
            if candidate_id in pool:
                continue
            candidate = self._candidate_from_values(candidate_id, baseline=baseline)
            if self.run_directory.report_path(candidate_id, 0, "full").exists():
                await self._score_candidate(candidate)
            else:
                await self._score_candidate(candidate, self._challenger_cases())
            pool[candidate_id] = candidate
            self._parents[candidate_id] = self._recorded_parents(candidate_id)
        return pool

    def _recorded_parents(self, candidate_id: str) -> tuple[str, ...]:
        """A candidate's parent ids as its manifest recorded them."""
        manifest = self.run_directory.read_manifest(candidate_id) or {}
        if manifest.get("parent_ids"):
            return tuple(manifest["parent_ids"])
        return (manifest["parent_id"],) if manifest.get("parent_id") else ()

    def _validate_resume(self, config: dict[str, Any]) -> None:
        recorded_cases = (
            set(config["train_cases"]) | set(config.get("selection_cases", [])) | set(config["holdout_cases"])
        )
        current_cases = set(self._case_names)
        if recorded_cases != current_cases:
            raise ValueError(
                f"resumed experiment does not match the recorded run: "
                f"missing cases {sorted(recorded_cases - current_cases)}, "
                f"new cases {sorted(current_cases - recorded_cases)}. "
                f"Changed cases need a fresh run (the case split would be stale)."
            )
        recorded_space = set(config.get("space", {}))
        current_space = {info.key for info in self.space.params()}
        if recorded_space != current_space:
            raise ValueError(
                f"resumed space does not match the recorded run: "
                f"recorded {sorted(recorded_space)}, supplied {sorted(current_space)}"
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


def _evaluator_name(evaluator: Any) -> str:
    get_name = getattr(evaluator, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return type(evaluator).__name__
