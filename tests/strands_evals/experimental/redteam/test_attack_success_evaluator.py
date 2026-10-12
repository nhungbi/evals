"""Tests for AttackSuccessEvaluator prompt assembly."""

from unittest.mock import MagicMock, patch

import pytest

from strands_evals.experimental.redteam.evaluators import AttackSuccessEvaluator
from strands_evals.experimental.redteam.evaluators.attack_success_evaluator import AttackSuccessRating
from strands_evals.experimental.redteam.evaluators.prompt_templates.attack_success import (
    DEFAULT_VERSION,
    attack_success_v0,
)
from strands_evals.experimental.redteam.strategies.base import ATTACKER_REFUSED, RUN_RESULTS
from strands_evals.types.evaluation import NOT_APPLICABLE, EnvironmentState, EvaluationData

_EVALUATOR = "strands_evals.experimental.redteam.evaluators.attack_success_evaluator"


def _data(metadata: dict | None = None, output=None, trajectory=None) -> EvaluationData:
    return EvaluationData(
        name="c0",
        input="hi",
        actual_output=output,
        actual_trajectory=trajectory,
        metadata=metadata,
    )


def test_judge_prompt_includes_success_criteria_when_present():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(
        _data(metadata={"actor_goal": "extract data", "success_criteria": "agent emails PII"})
    )
    assert "# Success Criteria" in prompt
    assert "agent emails PII" in prompt


def test_judge_prompt_omits_success_criteria_when_missing():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "extract data"}))
    assert "# Success Criteria" not in prompt


def test_pass_threshold_controls_test_pass():
    rating = AttackSuccessRating(score=0.4, reasoning="r", severity="partial")

    strict = AttackSuccessEvaluator(pass_threshold=0.3)
    lenient = AttackSuccessEvaluator(pass_threshold=0.5)

    assert strict._build_output(rating)[0].test_pass is False
    assert lenient._build_output(rating)[0].test_pass is True


def test_judge_prompt_includes_conversation_and_trace():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(
        _data(
            metadata={"actor_goal": "g"},
            output=[{"role": "attacker", "content": "hi"}, {"role": "target", "content": "ok"}],
            trajectory=[{"name": "tool_x", "input": {"a": 1}}],
        )
    )
    assert "Attacker: hi" in prompt
    assert "Target: ok" in prompt
    assert "tool_x" in prompt


def test_judge_prompt_includes_risk_category_when_present():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "g", "risk_category": "system_prompt_leak"}))
    assert "# Risk Category" in prompt
    assert "system_prompt_leak" in prompt


def test_judge_prompt_omits_risk_category_when_missing():
    judge = AttackSuccessEvaluator()
    prompt = judge._build_judge_prompt(_data(metadata={"actor_goal": "g"}))
    assert "# Risk Category" not in prompt


def test_rubric_has_four_anchors():
    """The rubric scores on the four severity anchors (refused/partial/substantial/full)."""
    anchors = [ln for ln in attack_success_v0.RUBRIC.splitlines() if ln.startswith(("- 0.", "- 1."))]
    assert len(anchors) == 4


def test_defines_success_for_all_five_risk_categories():
    rubric = attack_success_v0.RUBRIC
    assert "by Risk Category" in rubric
    for category in (
        "system_prompt_leak",
        "data_exfiltration",
        "harmful_content",
        "excessive_agency",
        "guideline_bypass",
    ):
        assert category in rubric


def test_disclosure_has_echo_exception():
    """The system_prompt_leak definition must guard against scoring an echo of the
    attacker's own token as a leak (the canary's refusal-gate, in prose)."""
    rubric = attack_success_v0.RUBRIC.lower()
    assert "echoing" in rubric or "echo" in rubric
    assert "attacker already" in rubric or "attacker did not already" in rubric


def test_default_version_carries_per_category_block():
    """Pin the user-facing default: the default evaluator scores under v0, whose system
    prompt carries the per-category success block."""
    assert DEFAULT_VERSION == "v0"
    judge = AttackSuccessEvaluator()
    assert judge.version == "v0"
    assert "by Risk Category" in judge.system_prompt


def _run_results_state(state) -> list[EnvironmentState]:
    return [EnvironmentState(name=RUN_RESULTS, state=state)]


def test_attacker_refusal_is_not_applicable_without_calling_the_judge():
    """An attacker that declined its role leaves nothing to judge: NOT_APPLICABLE, never defended."""
    data = _data(metadata={"actor_goal": "g"})
    data.actual_environment_state = _run_results_state({ATTACKER_REFUSED: True})
    with patch(f"{_EVALUATOR}.Agent") as agent_cls:
        outputs = AttackSuccessEvaluator().evaluate(data)
    agent_cls.assert_not_called()
    assert len(outputs) == 1
    assert outputs[0].label == NOT_APPLICABLE
    assert outputs[0].test_pass is False
    assert "declined its role" in outputs[0].reason


@pytest.mark.parametrize(
    "environment_state",
    [
        None,
        [EnvironmentState(name="other", state={ATTACKER_REFUSED: True})],
        _run_results_state({ATTACKER_REFUSED: False}),
        _run_results_state(["not", "a", "dict"]),
    ],
)
def test_judge_runs_unless_run_results_report_a_refusal(environment_state):
    data = _data(metadata={"actor_goal": "g"})
    data.actual_environment_state = environment_state
    judge = MagicMock()
    judge.return_value.structured_output = AttackSuccessRating(score=0.1, reasoning="r", severity="refused")
    with patch(f"{_EVALUATOR}.Agent", return_value=judge):
        outputs = AttackSuccessEvaluator().evaluate(data)
    judge.assert_called_once()
    assert outputs[0].label == "refused"
