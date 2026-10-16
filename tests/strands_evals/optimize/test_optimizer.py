"""End-to-end loop tests with a scripted method and a deterministic evaluator. No LLM is called.

Twenty cases in two groups: a group-A case is answered correctly only when knob `a` holds
`FIX_A`, a group-B case only when knob `b` holds `FIX_B`. A fix to one knob is a specialist that
leads only its group's selection cases, which is what the case frontier and merge are for.
"""

import logging

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators.deterministic import Equals
from strands_evals.optimize import (
    ArtifactSpace,
    AtMost,
    Block,
    Budget,
    CaseFrontier,
    Gate,
    Greedy,
    Method,
    Optimizer,
    Patch,
    Search,
    TextArtifact,
)

FIX_A = "always answer A questions"
FIX_B = "always answer B questions"
SEED = 2  # with 10 + 10 cases this seed puts both groups in every split


def make_cases():
    return [
        Case[str, str](name=f"{g}{i}", input=f"question {i}", expected_output="42", metadata={"group": g})
        for g in ("a", "b")
        for i in range(10)
    ]


ALWAYS_WRONG: set[str] = set()  # case names answered wrongly whatever the candidate


def build_task(candidate):
    def task(case):
        fixed = candidate["a"] == FIX_A if case.metadata["group"] == "a" else candidate["b"] == FIX_B
        fixed = fixed and case.name not in ALWAYS_WRONG
        answer = "42" if fixed else "0"
        return {
            "output": answer,
            "trajectory": [
                {"role": "user", "content": [{"text": case.input}]},
                {"role": "assistant", "content": [{"text": answer}]},
            ],
        }

    return task


class ScriptedMethod(Method):
    """Returns pre-scripted patch batches and records what it was given and told."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.parents = []
        self.records = []
        self.observations = []

    async def propose(self, space, keys, parent, records, n_proposals):
        self.parents.append(parent.id)
        self.records.append(records)
        if not self.batches:
            return []
        return [Patch(updates=dict(updates), rationale="scripted") for updates in self.batches.pop(0)]

    def observe(self, observation):
        self.observations.append(observation)


def make_optimizer(tmp_path, batches, *, run_dir=None, max_iterations=2, constraints=(), gate=None, selection=None):
    method = ScriptedMethod(batches)
    optimizer = Optimizer(
        Experiment(cases=make_cases(), evaluators=[Equals()]),
        build_task=build_task,
        artifacts=ArtifactSpace(a=TextArtifact("base a"), b=TextArtifact("base b")),
        search=Search(blocks=[Block(method, "a", "b")]),
        selection=selection or CaseFrontier(),  # these tests exercise the frontier and merge
        constraints=list(constraints),
        gate=gate or Gate(runs_per_candidate=1, holdout_fraction=0.4),
        budget=Budget(max_rollouts=1_000, max_iterations=max_iterations),
        base_dir=tmp_path,
        run_dir=run_dir,
        seed=SEED,
        max_workers=2,
    )
    return optimizer, method


SPECIALISTS = [[{"a": FIX_A}, {"b": FIX_B}]]


def test_cases_split_into_disjoint_train_selection_and_holdout(tmp_path):
    optimizer, _ = make_optimizer(tmp_path, [])
    train, selection, holdout = (
        set(optimizer.train_names),
        set(optimizer.selection_names),
        set(optimizer.holdout_names),
    )

    assert not (train & selection) and not (train & holdout) and not (selection & holdout)
    assert len(holdout) == 8 and len(selection) == 6 and len(train) == 6
    for split in (selection, holdout):
        assert {name[0] for name in split} == {"a", "b"}


def test_too_few_cases_rank_the_frontier_on_train(tmp_path, caplog):
    experiment = Experiment(cases=make_cases()[:2], evaluators=[Equals()])
    with caplog.at_level(logging.WARNING):
        optimizer = Optimizer(
            experiment,
            build_task=build_task,
            artifacts=ArtifactSpace(a=TextArtifact("base a"), b=TextArtifact("base b")),
            search=Search(blocks=[Block(ScriptedMethod([]), "a", "b")]),
            base_dir=tmp_path,
        )
    assert optimizer.selection_names == optimizer.train_names
    assert "frontier ranks on train" in caplog.text


async def test_specialists_join_the_frontier_and_their_merge_becomes_champion(tmp_path):
    optimizer, method = make_optimizer(tmp_path, SPECIALISTS)

    result = await optimizer.run()

    first, second = result.iterations
    assert first.step == "propose"
    assert [p.status for p in first.proposals] == ["accepted", "admitted"]
    assert second.step == "merge"
    merged = second.proposals[0]
    assert merged.status == "accepted"
    assert len(merged.parent_ids) == 2 and merged.method == "merge"
    assert result.best.values == {"a": FIX_A, "b": FIX_B}
    assert result.best_holdout_score == 1.0 and result.baseline_holdout_score == 0.0
    assert result.merge_count == 1 and result.accepted_count == 2
    assert list(result.frontier) == [result.best.id]
    # The merge step calls no method: the method proposed once, on the baseline.
    assert method.parents == [result.baseline.id]


async def test_greedy_proposes_on_the_champion_instead_of_merging_specialists(tmp_path):
    optimizer, method = make_optimizer(tmp_path, SPECIALISTS + [[{"b": FIX_B}]], selection=Greedy())

    result = await optimizer.run()

    first, second = result.iterations
    assert [p.status for p in first.proposals] == ["accepted", "admitted"]
    assert second.step == "propose" and result.merge_count == 0
    assert method.parents == [result.baseline.id, first.accepted_candidate_id]
    assert result.best.values == {"a": FIX_A, "b": FIX_B}  # same answer, reached by editing the champion


async def test_methods_are_told_the_selection_scalar_never_the_holdout(tmp_path, monkeypatch):
    optimizer, method = make_optimizer(tmp_path, SPECIALISTS, max_iterations=1)
    # Make the holdout disagree with the selection split, so the two means differ.
    monkeypatch.setattr(f"{__name__}.ALWAYS_WRONG", {optimizer.holdout_names[0]})

    result = await optimizer.run()

    outcomes = result.iterations[0].proposals
    assert len(method.observations) == 2
    for observation, outcome in zip(method.observations, outcomes, strict=True):
        assert observation.patch.updates.items() <= outcome.values.items()
        assert observation.scalar == outcome.selection_score
    assert any(o.scalar != p.holdout_score for o, p in zip(method.observations, outcomes, strict=True))
    # One challenger per iteration: exactly one patch faced the gate, the other never did.
    assert [o.accepted is None for o in method.observations].count(True) == 1


async def test_a_constraint_broken_on_selection_keeps_the_proposal_out_of_the_pool(tmp_path):
    optimizer, method = make_optimizer(tmp_path, SPECIALISTS, max_iterations=1, constraints=[AtMost("tokens", -1)])

    result = await optimizer.run()

    assert [p.status for p in result.iterations[0].proposals] == ["infeasible", "infeasible"]
    assert all("tokens <= -1" in p.constraint_violations[0] for p in result.iterations[0].proposals)
    assert result.best.id == result.baseline.id and result.infeasible_count == 2
    assert all(o.accepted is None for o in method.observations)


async def test_reflection_records_carry_the_returned_trajectory(tmp_path):
    optimizer, method = make_optimizer(tmp_path, [[{"a": FIX_A}]], max_iterations=1)

    await optimizer.run()

    record = method.records[0][0]
    assert record["Trajectory"].splitlines() == ["[user] question " + record["case"][1:], "[assistant] 0"]


async def test_resume_rebuilds_the_pool_and_never_repeats_a_merge(tmp_path):
    first, _ = make_optimizer(tmp_path, SPECIALISTS, max_iterations=1)
    await first.run()
    run_dir = first.run_directory.root

    resumed, method = make_optimizer(tmp_path, [], run_dir=run_dir, max_iterations=3)
    result = await resumed.run()

    steps = [record.step for record in result.iterations]
    assert steps[0] == "merge"  # the pool came back from the lineage, so the pair was mergeable
    assert result.best.values == {"a": FIX_A, "b": FIX_B}
    assert result.merge_count == 1  # iteration 3 proposes instead of merging the same pair again
    assert len(method.parents) == 1


def test_resume_rejects_a_changed_case_set(tmp_path):
    first, _ = make_optimizer(tmp_path, [])
    with pytest.raises(ValueError, match="does not match the recorded run"):
        Optimizer(
            Experiment(cases=make_cases()[:-1], evaluators=[Equals()]),
            build_task=build_task,
            artifacts=ArtifactSpace(a=TextArtifact("base a"), b=TextArtifact("base b")),
            search=Search(blocks=[Block(ScriptedMethod([]), "a", "b")]),
            run_dir=first.run_directory.root,
        )


async def test_iteration_rows_report_per_iteration_spend(tmp_path):
    optimizer, _ = make_optimizer(tmp_path, SPECIALISTS)

    result = await optimizer.run()

    rows = result.to_rows("iteration")
    assert [row["step"] for row in rows] == ["propose", "merge"]
    assert rows[-1]["rollouts_cumulative"] == sum(r.rollouts_spent for r in result.iterations)
    assert rows[-1]["rollouts_cumulative"] + 20 == result.rollouts_spent  # plus the baseline's 20
