"""End-to-end loop tests with a scripted proposer and a deterministic evaluator.

The task answers correctly only when the candidate's `instructions` contain the magic phrase, so
an edit that adds the phrase measurably improves the score and an unrelated edit changes nothing.
No LLM is called anywhere.
"""

import json

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators.deterministic import Equals
from strands_evals.optimize import (
    ArtifactSpace,
    Budget,
    Gate,
    Optimizer,
    TextArtifact,
)
from strands_evals.optimize.proposer import Proposal, Proposer

MAGIC = "Always answer 42."


def make_cases(n=6):
    return [Case[str, str](name=f"case_{i}", input=f"question {i}", expected_output="42") for i in range(n)]


def make_experiment(n=6):
    return Experiment(cases=make_cases(n), evaluators=[Equals()])


def build_task(candidate):
    def task(case):
        answer = "42" if MAGIC in candidate["instructions"] else "0"
        return {"output": answer}

    return task


class ScriptedProposer(Proposer):
    """Returns pre-scripted proposal batches instead of calling an LLM."""

    def __init__(self, batches, proposals_per_iteration=3):
        super().__init__(proposals_per_iteration=proposals_per_iteration)
        self.batches = list(batches)
        self.calls = 0

    def propose(self, space, incumbent, records):
        self.calls += 1
        if not self.batches:
            return []
        return self.filter_valid(space, incumbent, self.batches.pop(0))


def make_optimizer(tmp_path, batches, *, cases=6, budget=None, gate=None, **kwargs):
    return Optimizer(
        make_experiment(cases),
        build_task,
        ArtifactSpace(instructions=TextArtifact("You are unhelpful.")),
        budget=budget or Budget(max_rollouts=200),
        gate=gate or Gate(runs_per_candidate=2),
        proposer=ScriptedProposer(batches),
        base_dir=tmp_path,
        max_workers=2,
        **kwargs,
    )


async def test_loop_accepts_an_improving_edit(tmp_path):
    seen_iterations = []
    optimizer = make_optimizer(
        tmp_path,
        [[Proposal(artifact_name="instructions", new_value=MAGIC, rationale="the agent never answers 42")]],
        on_iteration=seen_iterations.append,
    )

    result = await optimizer.run()

    assert result.baseline_holdout_score == 0.0
    assert result.best_holdout_score == 1.0
    assert result.accepted_count == 1
    assert MAGIC in result.best.values["instructions"]
    assert "+Always answer 42." in result.best.diff()
    assert result.summary().startswith("baseline 0.00 -> best 1.00")

    assert [e["event"] for e in result.lineage if e["event"] == "accept"] == ["accept"]
    assert optimizer.run_directory.get_best() == result.best.id
    assert optimizer.run_directory.read_manifest(result.best.id)["status"] == "accepted"

    assert len(seen_iterations) == 1
    accepted = [p for p in seen_iterations[0].proposals if p.status == "accepted"]
    assert len(accepted) == 1
    assert accepted[0].verdict is not None and accepted[0].verdict.accepted


async def test_result_exports_serializable_rows_with_applied_values(tmp_path):
    optimizer = make_optimizer(
        tmp_path,
        [[Proposal(artifact_name="instructions", new_value=MAGIC, rationale="the agent never answers 42")]],
    )
    result = await optimizer.run()

    iteration_rows = result.to_rows("iteration")
    assert iteration_rows[0]["iteration"] == 1
    assert iteration_rows[0]["incumbent_values"]["instructions"]
    assert iteration_rows[0]["rollouts_cumulative"] == result.rollouts_spent

    proposal_rows = result.to_rows("proposal")
    accepted = [r for r in proposal_rows if r["status"] == "accepted"]
    assert len(accepted) == 1
    assert MAGIC in accepted[0]["values"]["instructions"]
    assert accepted[0]["parent_id"] == result.baseline.id
    assert accepted[0]["ci_low"] is not None and accepted[0]["verdict_reason"]

    out = tmp_path / "report.json"
    payload = result.to_json("proposal", path=out)
    assert json.loads(payload) == json.loads(out.read_text()) == proposal_rows

    case_rows = result.to_rows("case")
    stored = result.reports(result.baseline.id)
    assert stored and all(name.startswith("run") for name in stored)
    baseline_rows = [r for r in case_rows if r["candidate_id"] == result.baseline.id]
    first_report = next(iter(stored.values()))
    assert baseline_rows[0]["score"] == first_report.scores[0]
    assert baseline_rows[0]["evaluator"] == first_report.cases[0].get("evaluator")
    assert baseline_rows[0]["status"] == "baseline" and baseline_rows[0]["iteration"] == 0
    assert {r["status"] for r in case_rows} >= {"baseline", "accepted"}
    json.dumps(case_rows)  # every row stays JSON-serializable

    with pytest.raises(ValueError, match="unknown grain"):
        result.to_rows("run")


async def test_loop_rejects_an_edit_that_changes_nothing_measurable(tmp_path):
    optimizer = make_optimizer(
        tmp_path,
        [[Proposal(artifact_name="instructions", new_value="You are extra unhelpful.", rationale="reworded")]],
    )

    result = await optimizer.run()

    assert result.accepted_count == 0
    assert result.rejected_count == 1
    assert result.best.id == result.baseline.id
    assert result.best_holdout_score == result.baseline_holdout_score
    reject_events = [e for e in result.lineage if e["event"] == "reject"]
    assert len(reject_events) == 1
    assert "tie" in reject_events[0]["reason"]


async def test_budget_stops_before_first_iteration(tmp_path):
    # Baseline alone costs 6 cases * 2 runs = 12 rollouts; nothing is left for a candidate.
    optimizer = make_optimizer(
        tmp_path,
        [[Proposal(artifact_name="instructions", new_value=MAGIC, rationale="fix")]],
        budget=Budget(max_rollouts=12),
    )

    result = await optimizer.run()

    assert result.rollouts_spent == 12
    assert result.iterations == []
    assert result.best.id == result.baseline.id
    assert optimizer.proposer.calls == 0


async def test_max_iterations_cap(tmp_path):
    batches = [[Proposal(artifact_name="instructions", new_value=f"variant {i}", rationale="try")] for i in range(5)]
    optimizer = make_optimizer(tmp_path, batches, budget=Budget(max_rollouts=10_000, max_iterations=2))

    result = await optimizer.run()

    assert len(result.iterations) == 2


async def test_resume_replays_scores_without_new_rollouts(tmp_path):
    first = make_optimizer(
        tmp_path,
        [[Proposal(artifact_name="instructions", new_value=MAGIC, rationale="fix")]],
    )
    first_result = await first.run()
    spent = first_result.rollouts_spent
    assert spent > 0

    resumed = Optimizer.resume(
        first.run_directory.root,
        experiment=make_experiment(),
        build_task=build_task,
        artifacts=ArtifactSpace(instructions=TextArtifact("You are unhelpful.")),
        gate=Gate(runs_per_candidate=2),
        proposer=ScriptedProposer([]),
        max_workers=2,
    )
    resumed_result = await resumed.run()

    # The incumbent picks up from the accepted candidate, and no rollout re-runs.
    assert resumed_result.best.id == first_result.best.id
    assert resumed_result.best_holdout_score == 1.0
    assert resumed_result.rollouts_spent == spent
    score_events = [e for e in resumed.run_directory.load_events() if e["event"] == "score"]
    assert len(score_events) == len([e for e in first_result.lineage if e["event"] == "score"])
    # Counters replay from lineage: the accept from the first segment is still reported.
    assert resumed_result.accepted_count == 1
    assert resumed_result.rejected_count == 0


async def test_identical_reproposal_is_a_cache_hit(tmp_path):
    proposal = Proposal(artifact_name="instructions", new_value="A different bad prompt.", rationale="retry")
    optimizer = make_optimizer(tmp_path, [[proposal], [proposal]], budget=Budget(max_rollouts=10_000))

    result = await optimizer.run()

    # Two iterations proposed identical content; the second scoring was a content-address hit.
    assert len(result.iterations) == 2
    score_events = [e for e in result.lineage if e["event"] == "score"]
    # Baseline (2 runs) + candidate (2 runs) = 4 score events; the re-proposal adds none.
    assert len(score_events) == 4


async def test_error_rollouts_are_excluded_not_scored_zero(tmp_path):
    def crashing_factory(candidate):
        def task(case):
            if case.name == "case_0":
                raise RuntimeError("boom")
            return {"output": "42" if MAGIC in candidate["instructions"] else "0"}

        return task

    optimizer = Optimizer(
        make_experiment(),
        crashing_factory,
        ArtifactSpace(instructions=TextArtifact(MAGIC)),
        budget=Budget(max_rollouts=12),
        gate=Gate(runs_per_candidate=2),
        proposer=ScriptedProposer([]),
        base_dir=tmp_path,
        max_workers=2,
    )
    result = await optimizer.run()

    # case_0 crashed every run: it is missing from the per-case scores, not scored 0.0,
    # so the baseline holdout mean stays 1.0 wherever case_0 would have landed.
    baseline_score = optimizer._scores[result.baseline.id]
    assert "case_0" not in baseline_score.per_case
    assert baseline_score.error_rollouts > 0
    assert all(value == 1.0 for value in baseline_score.per_case.values())


def test_optimizer_rejects_unnamed_or_duplicate_cases(tmp_path):
    space = ArtifactSpace(instructions=TextArtifact("x"))

    unnamed = Experiment(cases=[Case[str, str](input="q")], evaluators=[Equals()])
    with pytest.raises(ValueError, match="needs a name"):
        Optimizer(unnamed, build_task, space, base_dir=tmp_path)

    duplicated = Experiment(
        cases=[Case[str, str](name="a", input="q"), Case[str, str](name="a", input="r")],
        evaluators=[Equals()],
    )
    with pytest.raises(ValueError, match="unique"):
        Optimizer(duplicated, build_task, space, base_dir=tmp_path)


async def test_proposer_crash_stops_gracefully_and_stays_resumable(tmp_path):
    class CrashingProposer(ScriptedProposer):
        def propose(self, space, incumbent, records):
            raise RuntimeError("stream read timed out")

    optimizer = make_optimizer(tmp_path, [])
    optimizer.proposer = CrashingProposer([])

    result = await optimizer.run()

    # The run ends with the incumbent intact instead of raising, and the failure is on record.
    assert result.best.id == result.baseline.id
    error_events = [e for e in result.lineage if e["event"] == "proposer_error"]
    assert len(error_events) == 1
    assert "timed out" in error_events[0]["error"]


async def test_cheap_challenger_scoring_spends_fewer_rollouts(tmp_path):
    # Three scored proposals in one iteration: the winner is accepted (and topped up to the full
    # set), the other two are rejected and never pay the train-case cost. That is where cheap
    # scoring saves, so the run must come in under the full-scoring cost for the same candidates.
    proposals = [
        Proposal(artifact_name="instructions", new_value=MAGIC, rationale="fix"),
        Proposal(artifact_name="instructions", new_value="still unhelpful A", rationale="a"),
        Proposal(artifact_name="instructions", new_value="still unhelpful B", rationale="b"),
    ]
    optimizer = make_optimizer(
        tmp_path,
        [proposals],
        budget=Budget(max_rollouts=10_000, max_iterations=1),
        gate=Gate(runs_per_candidate=2, challenger_holdout_only=True),
    )
    result = await optimizer.run()

    assert result.accepted_count == 1
    assert result.best_holdout_score == 1.0

    events = [e for e in result.lineage if e["event"] == "score"]
    holdout_n = len(optimizer.holdout_names)
    train_n = len(optimizer.train_names)
    total = holdout_n + train_n
    k = optimizer.gate.runs_per_candidate
    # Rejected challengers were scored on the holdout only; proven by holdout-tagged events.
    holdout_events = [e for e in events if e.get("tag") == "holdout"]
    assert holdout_events and all(e["rollouts"] == holdout_n for e in holdout_events)
    # The accepted challenger's top-up re-runs only the train cases (holdout hit the cache).
    assert [e for e in events if e.get("tag") == "full" and e["rollouts"] == train_n]
    # Full-scoring would cost baseline + 3 challengers all on the full set.
    full_path_cost = total * k * 4
    assert result.rollouts_spent < full_path_cost


async def test_duplicate_proposals_collapse_to_one_candidate(tmp_path):
    # Two proposals with identical content are one content-addressed candidate. Without the
    # dedupe, the duplicate outcome would overwrite the accepted manifest with "rejected".
    proposals = [
        Proposal(artifact_name="instructions", new_value=MAGIC, rationale="fix"),
        Proposal(artifact_name="instructions", new_value=MAGIC, rationale="same fix, reworded rationale"),
    ]
    optimizer = make_optimizer(tmp_path, [proposals], budget=Budget(max_rollouts=10_000, max_iterations=1))

    result = await optimizer.run()

    assert result.accepted_count == 1
    assert len(result.iterations[0].proposals) == 1
    assert optimizer.run_directory.read_manifest(result.best.id)["status"] == "accepted"


async def test_uncompared_proposals_are_skipped_not_rejected(tmp_path):
    # Once one candidate is accepted, the rest of the iteration's scored candidates never face
    # the gate. They must be recorded as skipped, with a lineage event, so in-run counters and
    # lineage replay agree.
    proposals = [
        Proposal(artifact_name="instructions", new_value=MAGIC, rationale="fix"),
        Proposal(artifact_name="instructions", new_value="still unhelpful", rationale="noise"),
    ]
    optimizer = make_optimizer(tmp_path, [proposals], budget=Budget(max_rollouts=10_000, max_iterations=1))

    result = await optimizer.run()

    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert result.skipped_count == 1
    statuses = sorted(p.status for p in result.iterations[0].proposals)
    assert statuses == ["accepted", "skipped"]
    skip_events = [e for e in result.lineage if e["event"] == "skip"]
    assert len(skip_events) == 1
    skipped = [p for p in result.iterations[0].proposals if p.status == "skipped"][0]
    assert skipped.verdict is None
    assert optimizer.run_directory.read_manifest(skipped.candidate_id)["status"] == "skipped"

    # A resumed run replays the same totals from the lineage.
    resumed = Optimizer.resume(
        optimizer.run_directory.root,
        experiment=make_experiment(),
        build_task=build_task,
        artifacts=ArtifactSpace(instructions=TextArtifact("You are unhelpful.")),
        gate=Gate(runs_per_candidate=2),
        proposer=ScriptedProposer([]),
        max_workers=2,
    )
    resumed_result = await resumed.run()
    assert resumed_result.accepted_count == 1
    assert resumed_result.rejected_count == 0
    assert resumed_result.skipped_count == 1


async def test_low_holdout_coverage_rejects(tmp_path):
    # An edit that crashes the agent on held-out cases must not win by being compared only on
    # the cases it survived.
    crash_cases = set()

    def crashy_build_task(candidate):
        def task(case):
            if "CRASH" in candidate["instructions"] and case.name in crash_cases:
                raise RuntimeError("boom")
            return {"output": "42" if MAGIC in candidate["instructions"] else "0"}

        return task

    proposal = Proposal(artifact_name="instructions", new_value=MAGIC + " CRASH", rationale="risky fix")
    optimizer = Optimizer(
        make_experiment(),
        crashy_build_task,
        ArtifactSpace(instructions=TextArtifact("You are unhelpful.")),
        budget=Budget(max_rollouts=10_000, max_iterations=1),
        gate=Gate(runs_per_candidate=2),
        proposer=ScriptedProposer([[proposal]]),
        base_dir=tmp_path,
        max_workers=2,
    )
    crash_cases.add(optimizer.holdout_names[0])

    result = await optimizer.run()

    assert result.accepted_count == 0
    reject_events = [e for e in result.lineage if e["event"] == "reject"]
    assert len(reject_events) == 1
    assert "coverage floor" in reject_events[0]["reason"]
    challenger = result.iterations[0].proposals[0]
    assert challenger.error_rollouts > 0


async def test_resume_rejects_mismatched_cases_and_artifacts(tmp_path):
    import pytest

    first = make_optimizer(tmp_path, [])
    await first.run()

    with pytest.raises(ValueError, match="does not match the recorded run"):
        Optimizer.resume(
            first.run_directory.root,
            experiment=make_experiment(4),
            build_task=build_task,
            artifacts=ArtifactSpace(instructions=TextArtifact("You are unhelpful.")),
            proposer=ScriptedProposer([]),
        )

    with pytest.raises(ValueError, match="artifacts do not match"):
        Optimizer.resume(
            first.run_directory.root,
            experiment=make_experiment(),
            build_task=build_task,
            artifacts=ArtifactSpace(renamed=TextArtifact("You are unhelpful.")),
            proposer=ScriptedProposer([]),
        )

    # A matching resume is recorded in the lineage.
    resumed = Optimizer.resume(
        first.run_directory.root,
        experiment=make_experiment(),
        build_task=build_task,
        artifacts=ArtifactSpace(instructions=TextArtifact("You are unhelpful.")),
        proposer=ScriptedProposer([]),
    )
    assert [e for e in resumed.run_directory.load_events() if e["event"] == "resume"]


def test_proposer_and_proposer_model_are_mutually_exclusive(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="not both"):
        Optimizer(
            make_experiment(),
            build_task,
            ArtifactSpace(instructions=TextArtifact("x")),
            proposer=ScriptedProposer([]),
            proposer_model="some-model",
            base_dir=tmp_path,
        )


async def test_plain_mapping_accepted_as_artifacts(tmp_path):
    optimizer = Optimizer(
        make_experiment(),
        build_task,
        {"instructions": TextArtifact("You are unhelpful.")},
        budget=Budget(max_rollouts=200),
        gate=Gate(runs_per_candidate=2),
        proposer=ScriptedProposer([[Proposal(artifact_name="instructions", new_value=MAGIC, rationale="fix")]]),
        base_dir=tmp_path,
        max_workers=2,
    )
    result = await optimizer.run()
    assert result.accepted_count == 1


async def test_cheap_challenger_scoring_rejects_like_full_path(tmp_path):
    # A no-op edit ties on the holdout and must be rejected under cheap scoring too.
    optimizer = make_optimizer(
        tmp_path,
        [[Proposal(artifact_name="instructions", new_value="You are still unhelpful.", rationale="reworded")]],
        gate=Gate(runs_per_candidate=2, challenger_holdout_only=True),
    )
    result = await optimizer.run()
    assert result.accepted_count == 0
    assert result.best.id == result.baseline.id
