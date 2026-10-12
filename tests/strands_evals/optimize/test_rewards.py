from strands_evals.optimize import RewardFunction, RewardFunctionEvaluator
from strands_evals.types.evaluation import EvaluationData


class ExactMatchReward(RewardFunction):
    def __call__(self, **fields):
        match = str(fields.get("actual_output") or "").strip() == str(fields.get("expected_output") or "").strip()
        return {"reward_value": 1.0 if match else 0.0, "reason": "match" if match else "mismatch"}


def _data(actual, expected):
    return EvaluationData(input="q", actual_output=actual, expected_output=expected, name="c")


def test_reward_evaluator_passes_on_match():
    ev = RewardFunctionEvaluator(ExactMatchReward())
    out = ev.evaluate(_data("42", "42"))[0]
    assert out.score == 1.0 and out.test_pass and out.reason == "match"


def test_reward_evaluator_fails_on_mismatch():
    ev = RewardFunctionEvaluator(ExactMatchReward())
    out = ev.evaluate(_data("41", "42"))[0]
    assert out.score == 0.0 and not out.test_pass


def test_reward_evaluator_threshold():
    class Half(RewardFunction):
        def __call__(self, **fields):
            return {"reward_value": 0.5}

    strict = RewardFunctionEvaluator(Half(), pass_threshold=1.0)
    lenient = RewardFunctionEvaluator(Half(), pass_threshold=0.5)
    assert not strict.evaluate(_data("x", "y"))[0].test_pass
    assert lenient.evaluate(_data("x", "y"))[0].test_pass


def test_reward_evaluator_default_reason():
    class Bare(RewardFunction):
        def __call__(self, **fields):
            return {"reward_value": 0.0}

    out = RewardFunctionEvaluator(Bare()).evaluate(_data("x", "y"))[0]
    assert "reward_value=0.0" in out.reason


def test_reward_evaluator_passes_conversation_fields():
    """A conversation-level reward reads the multi-turn record, not just the final output."""

    class GoalMentionedReward(RewardFunction):
        def __call__(self, *, actual_interactions=None, **_):
            messages = [m for turn in (actual_interactions or []) for m in turn.get("messages", [])]
            mentioned = any("TICKET-7" in str(m) for m in messages)
            return {"reward_value": 1.0 if mentioned else 0.0}

    interactions = [
        {"node_name": "user", "messages": ["My ticket is TICKET-7."]},
        {"node_name": "agent", "messages": ["Noted."]},
        {"node_name": "user", "messages": ["Which ticket did I mention?"]},
        {"node_name": "agent", "messages": ["TICKET-7."]},
    ]
    data = EvaluationData(input="q", actual_output="TICKET-7.", name="c", actual_interactions=interactions)
    assert RewardFunctionEvaluator(GoalMentionedReward()).evaluate(data)[0].score == 1.0

    empty = EvaluationData(input="q", actual_output="?", name="c", actual_interactions=[])
    assert RewardFunctionEvaluator(GoalMentionedReward()).evaluate(empty)[0].score == 0.0


def test_reward_evaluator_forwards_all_rollout_fields():
    seen: dict = {}

    class Capture(RewardFunction):
        def __call__(self, **fields):
            seen.update(fields)
            return {"reward_value": 1.0}

    data = EvaluationData(
        input="q",
        actual_output="a",
        expected_output="a",
        expected_assertion="agent recalls the ticket id",
        actual_trajectory=["tool_a"],
        expected_trajectory=["tool_a"],
        actual_interactions=[{"node_name": "agent", "messages": ["a"]}],
        expected_interactions=[{"node_name": "agent", "messages": ["a"]}],
        metadata={"k": "v"},
        name="c",
    )
    RewardFunctionEvaluator(Capture()).evaluate(data)
    expected_keys = {
        "input",
        "actual_output",
        "expected_output",
        "expected_assertion",
        "actual_trajectory",
        "expected_trajectory",
        "actual_interactions",
        "expected_interactions",
        "metadata",
    }
    assert expected_keys == set(seen)
    assert seen["actual_trajectory"] == ["tool_a"]
    assert seen["expected_assertion"] == "agent recalls the ticket id"
