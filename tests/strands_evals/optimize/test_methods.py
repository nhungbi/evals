import importlib.util

import pytest

from strands_evals.optimize import Candidate, Float, Int, Observation, Space, TextValue, build_reflection_records
from strands_evals.optimize.methods import TPE

needs_optuna = pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="optional extra: search")


def make_space():
    return Space(
        markers={
            "prompt": TextValue(default="base"),
            "window": Int(4, 64, default=8),
            "ratio": Float(0.1, 0.9, default=0.5),
        }
    )


@needs_optuna
async def test_tpe_keeps_one_study_per_context_of_the_other_variables():
    space = make_space()
    tpe = TPE(seed=0)
    parent_a = Candidate(space.defaults(), space=space)
    parent_b = Candidate({**space.defaults(), "prompt": "edited"}, space=space)

    first = await tpe.propose(space, ["window", "ratio"], parent_a, [], 2)
    again = await tpe.propose(space, ["window", "ratio"], parent_a, [], 1)
    other = await tpe.propose(space, ["window", "ratio"], parent_b, [], 1)

    assert len(tpe._studies) == 2  # same prompt shares a study; a different prompt gets its own
    assert first[0].draw[0] == again[0].draw[0] != other[0].draw[0]
    assert set(first[0].updates) == {"window", "ratio"}


@needs_optuna
async def test_tpe_observe_tells_the_drawing_study_and_fails_unscored_trials():
    space = make_space()
    tpe = TPE(seed=0)
    parent = Candidate(space.defaults(), space=space)
    told, unscored = await tpe.propose(space, ["window"], parent, [], 2)

    tpe.observe(Observation(patch=told, scalar=0.75, accepted=None))
    tpe.observe(Observation(patch=unscored, scalar=None, accepted=None))

    study = tpe._studies[told.draw[0]]
    states = sorted(trial.state.name for trial in study.trials)
    assert states == ["COMPLETE", "FAIL"]
    assert study.best_value == 0.75


def test_reflection_records_carry_a_trajectory_only_when_present():
    row = {"evaluator": "Equals", "score": 0.0, "reason": "wrong", "input": "q", "actual_output": "a"}
    rows = {"traced": [{**row, "trajectory": "[user] q\n[assistant] a"}], "plain": [{**row, "trajectory": ""}]}

    records = {r["case"]: r for r in build_reflection_records(rows, {"traced": 0.0, "plain": 0.0})}

    assert records["traced"]["Trajectory"] == "[user] q\n[assistant] a"
    assert "Trajectory" not in records["plain"]
