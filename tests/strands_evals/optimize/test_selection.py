"""Selection is switchable: the same scripted run under `CaseFrontier` and `Greedy`. No LLM is called."""

import random

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.optimize import (
    ArtifactSpace,
    Block,
    Budget,
    CaseFrontier,
    Gate,
    Greedy,
    Method,
    Optimizer,
    Patch,
    Search,
    Selection,
    SelectionContext,
    SelectionStep,
    TextArtifact,
)
from strands_evals.types.evaluation import EvaluationOutput

SEED = 2  # with 10 + 10 cases this seed puts both groups in every split

# Per-group score of each value of knob `v`. The specialists each beat the baseline on one group
# and lose badly on the other; the generalist is a little better than the baseline everywhere,
# so it is better on average than every candidate yet best on no single case.
SCORES = {
    "base": {"a": 0.5, "b": 0.5},
    "spec_a": {"a": 0.6, "b": 0.0},
    "spec_b": {"a": 0.0, "b": 0.6},
    "general": {"a": 0.55, "b": 0.55},
}


class OutputScore(Evaluator):
    """Scores a case with the number the task returned."""

    def evaluate(self, evaluation_case):
        return [EvaluationOutput(score=float(evaluation_case.actual_output), test_pass=True, reason="scored")]

    async def evaluate_async(self, evaluation_case):
        return self.evaluate(evaluation_case)


def build_task(candidate):
    return lambda case: {"output": str(SCORES[candidate["v"]][case.metadata["group"]])}


class ScriptedMethod(Method):
    def __init__(self, batches):
        self.batches = list(batches)
        self.parents = []

    async def propose(self, space, keys, parent, records, n_proposals):
        self.parents.append(parent.id)
        return [Patch(updates=dict(u), rationale="scripted") for u in (self.batches.pop(0) if self.batches else [])]


def make_optimizer(tmp_path, selection, batches, *, run_dir=None, max_iterations=2):
    method = ScriptedMethod(batches)
    optimizer = Optimizer(
        Experiment(
            cases=[Case(name=f"{g}{i}", input="q", metadata={"group": g}) for g in "ab" for i in range(10)],
            evaluators=[OutputScore()],
        ),
        build_task=build_task,
        artifacts=ArtifactSpace(v=TextArtifact("base")),
        search=Search(blocks=[Block(method, "v")]),
        selection=selection,
        gate=Gate(runs_per_candidate=1, holdout_fraction=0.4),
        budget=Budget(max_rollouts=1_000, max_iterations=max_iterations),
        base_dir=tmp_path,
        run_dir=run_dir,
        seed=SEED,
    )
    return optimizer, method


SCRIPT = [[{"v": "spec_a"}, {"v": "spec_b"}], [{"v": "general"}]]


async def test_frontier_keeps_out_a_generalist_that_leads_no_case(tmp_path):
    optimizer, _ = make_optimizer(tmp_path, CaseFrontier(), SCRIPT)

    result = await optimizer.run()

    first, second = result.iterations
    assert [p.status for p in first.proposals] == ["admitted", "admitted"]  # each leads its own group
    assert second.parent_id in {p.candidate_id for p in first.proposals}  # a specialist, not the champion
    assert [p.status for p in second.proposals] == ["dominated"]
    assert result.best.id == result.baseline.id


async def test_greedy_builds_on_the_champion_and_lets_the_generalist_challenge(tmp_path):
    optimizer, method = make_optimizer(tmp_path, Greedy(), SCRIPT)

    result = await optimizer.run()

    first, second = result.iterations
    assert [p.status for p in first.proposals] == ["admitted", "admitted"]
    assert method.parents == [result.baseline.id, result.baseline.id]  # the champion never moved in iteration 1
    assert [p.status for p in second.proposals] == ["accepted"]
    assert result.best.values == {"v": "general"}
    assert result.merge_count == 0


async def test_resume_rejects_a_changed_selection(tmp_path):
    first, _ = make_optimizer(tmp_path, CaseFrontier(), SCRIPT, max_iterations=1)
    await first.run()

    with pytest.raises(ValueError, match="resumed selection does not match"):
        make_optimizer(tmp_path, Greedy(), [], run_dir=first.run_directory.root)


def test_greedy_is_the_default_and_a_non_selection_is_rejected(tmp_path):
    optimizer, _ = make_optimizer(tmp_path, None, [])
    assert isinstance(optimizer.selection, Greedy)

    with pytest.raises(TypeError, match="takes a Selection"):
        make_optimizer(tmp_path, "greedy", [])


# -- the strategy interface, without the loop ------------------------------------------------------


def make_context(scores, values, parents, *, champion_id="base", validate=None, merged_pairs=frozenset()):
    return SelectionContext(
        scores=scores,
        values=values,
        parents=parents,
        selection_cases=["a0", "b0"],
        champion_id=champion_id,
        rng=random.Random(0),
        merged_pairs=merged_pairs,
        validate=validate or (lambda values: None),
    )


SPECIALIST_POOL = {
    "scores": {"base": {"a0": 0.0, "b0": 0.0}, "sa": {"a0": 1.0, "b0": 0.0}, "sb": {"a0": 0.0, "b0": 1.0}},
    "values": {"base": {"x": "0", "y": "0"}, "sa": {"x": "1", "y": "0"}, "sb": {"x": "0", "y": "1"}},
    "parents": {"base": (), "sa": ("base",), "sb": ("base",)},
}


def test_greedy_chooses_the_champion_and_admits_everything():
    ctx = make_context(**SPECIALIST_POOL, champion_id="sa")

    step = Greedy().choose(ctx)

    assert (step.parent_id, step.merge) == ("sa", None)
    assert Greedy().admits("new", {"a0": 0.0, "b0": 0.0}, ctx) == []


def test_case_frontier_merges_two_specialists():
    step = CaseFrontier().choose(make_context(**SPECIALIST_POOL))

    assert step.merge is not None and step.parent_id == step.merge.first == "sa"
    assert step.merge.values == {"x": "1", "y": "1"}


def test_case_frontier_reports_a_merge_the_space_rejects_and_samples_instead():
    def reject(values):
        raise ValueError("not a valid point")

    step = CaseFrontier().choose(make_context(**SPECIALIST_POOL, validate=reject))

    assert step.merge is None and step.parent_id in {"sa", "sb"}
    assert step.unmergeable == (frozenset({"sa", "sb"}),)


def test_case_frontier_admits_only_frontier_landings():
    ctx = make_context(**SPECIALIST_POOL)

    assert CaseFrontier().admits("tie", {"a0": 1.0, "b0": 0.0}, ctx) is None  # ties sa: the older member keeps it
    assert CaseFrontier().admits("both", {"a0": 1.0, "b0": 1.0}, ctx) == ["a0", "b0"]


async def test_a_selection_naming_a_non_member_fails_clearly(tmp_path):
    class Lost(Selection):
        name = "lost"

        def choose(self, ctx):
            return SelectionStep(parent_id="not-in-the-pool")

        def admits(self, candidate_id, candidate_scores, ctx):
            return []

    optimizer, _ = make_optimizer(tmp_path, Lost(), SCRIPT)

    with pytest.raises(ValueError, match=r"Lost.choose named \['not-in-the-pool'\]"):
        await optimizer.run()
