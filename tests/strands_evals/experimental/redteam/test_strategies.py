"""Tests for AttackStrategy implementations."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.experimental import redteam
from strands_evals.experimental.redteam import strategies, task
from strands_evals.experimental.redteam.case import RedTeamCase
from strands_evals.experimental.redteam.strategies import (
    BUILTIN_STRATEGIES,
    RUN_RESULTS,
    BadLikertJudgeStrategy,
    CrescendoStrategy,
    GoatStrategy,
    PairStrategy,
    PromptStrategy,
    SequentialBreakStrategy,
    base,
)
from strands_evals.experimental.redteam.strategies._common import ATTACKER_REFUSAL_JUDGE_SYSTEM_PROMPT
from strands_evals.experimental.redteam.strategies.base import AttackerRefusalRating, AttackRunResult, AttackStrategy
from strands_evals.experimental.redteam.strategies.target_session import TargetCheckpoint
from strands_evals.experimental.redteam.types import AttackGoal, RedTeamConfig

_BASE = "strands_evals.experimental.redteam.strategies.base"


class _FakeSession:
    """A TargetSession for forward-only strategy tests.

    PromptStrategy never backtracks, but the fake implements the full protocol
    surface (snapshot/restore too) so it stays a valid TargetSession and can't
    silently diverge from the contract the strategy is handed.
    """

    def __init__(self, reply_fn):
        self._reply_fn = reply_fn
        self.trace: list[dict] = []

    def invoke(self, message):
        return self._reply_fn(message)

    def reset(self):
        self.trace.clear()

    def snapshot(self):
        return TargetCheckpoint(agent_snapshot=None, trace_len=len(self.trace))

    def restore(self, checkpoint):
        del self.trace[checkpoint.trace_len :]


def _case(name: str = "c0") -> RedTeamCase:
    return RedTeamCase(
        name=name,
        input="hello",
        config=RedTeamConfig(attack_goal=AttackGoal(risk_category="prompt_injection", actor_goal="goal")),
    )


def _prompt_strategy(template: str = "p {max_turns}", **kwargs) -> PromptStrategy:
    """A PromptStrategy whose attacker refusal check is stubbed to "not a refusal" (no LLM)."""
    strategy = PromptStrategy("gradual_escalation", template, **kwargs)
    strategy._is_attacker_refusal = MagicMock(return_value=False)  # type: ignore[method-assign]
    return strategy


def test_redteam_config_is_strategy_agnostic():
    """RedTeamConfig no longer carries strategy/template; cases are strategy-agnostic."""
    config = RedTeamConfig(attack_goal=AttackGoal(risk_category="prompt_injection", actor_goal="g"))
    assert not hasattr(config, "strategy")
    assert not hasattr(config, "system_prompt_template")
    assert "strategy" not in RedTeamConfig.model_fields
    assert "system_prompt_template" not in RedTeamConfig.model_fields


def test_attack_run_result_to_environment_state_keeps_all_metadata():
    pruned = [{"role": "attacker", "content": "a"}, {"role": "target", "content": "no"}]
    result = AttackRunResult(
        conversation=[{"role": "attacker", "content": "hi"}],
        strategy_succeeded=True,
        strategy_score=0.9,
        metadata={"turns_used": 3, "backtracks": 1, "reasoning_trace": [{"o": "x"}]},
        pruned_branches=pruned,
    )
    state = result.to_environment_state()

    assert state.name == RUN_RESULTS == "redteam_run_results"
    # Strategy-specific keys (e.g. GOAT's reasoning_trace) pass through alongside the report's fields.
    assert state.state == {
        "turns_used": 3,
        "backtracks": 1,
        "reasoning_trace": [{"o": "x"}],
        "pruned_branches": pruned,
    }


def test_attack_run_result_to_environment_state_omits_unrecorded_keys():
    state = AttackRunResult(conversation=[], metadata={"turns_used": 2}).to_environment_state()
    assert state.state == {"turns_used": 2, "pruned_branches": []}


def test_prompt_strategy_label_defaults_to_name():
    strategy = BUILTIN_STRATEGIES["gradual_escalation"]
    assert strategy.label == strategy.name == "gradual_escalation"


def test_prompt_strategy_label_override():
    strategy = PromptStrategy("gradual_escalation", "tmpl {max_turns}", label="grad-5")
    assert strategy.name == "gradual_escalation"
    assert strategy.label == "grad-5"


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_prompt_strategy_run_attack_drives_loop(mock_simulator_cls):
    """PromptStrategy.run_attack drives the ActorSimulator loop via the injected call_target."""
    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, True, False]
    act_result = MagicMock()
    act_result.structured_output.message = "follow-up attack"
    mock_sim.act.return_value = act_result
    mock_simulator_cls.return_value = mock_sim

    strategy = _prompt_strategy("prompt {actor_profile} {max_turns}")
    session = _FakeSession(lambda _m: "target reply")

    result = strategy.run_attack(_case(), session, max_turns=7)

    assert isinstance(result, AttackRunResult)
    assert len(result.conversation) == 4
    assert result.conversation[0]["role"] == "attacker"
    assert result.conversation[1]["content"] == "target reply"
    # max_turns placeholder substituted into the simulator's system prompt
    template = mock_simulator_cls.call_args.kwargs["system_prompt_template"]
    assert "{max_turns}" not in template and "7" in template


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_prompt_strategy_attacker_refusal_ends_case_without_sending_it(mock_simulator_cls):
    """A simulator message that declines the attacker role stops the attack before reaching the target."""
    mock_sim = MagicMock()
    mock_sim.has_next.return_value = True
    act_result = MagicMock()
    act_result.structured_output.message = "I can't play this role"
    mock_sim.act.return_value = act_result
    mock_simulator_cls.return_value = mock_sim
    strategy = _prompt_strategy()
    strategy._is_attacker_refusal = MagicMock(return_value=True)  # type: ignore[method-assign]
    sent: list[str] = []

    result = strategy.run_attack(_case(), _FakeSession(lambda m: sent.append(m) or "target reply"), max_turns=7)

    assert sent == ["hello"]  # only the case input; the refusal was never sent
    assert result.conversation == [
        {"role": "attacker", "content": "hello"},
        {"role": "target", "content": "target reply"},
        {"role": "attacker_refused", "content": "I can't play this role"},  # kept for debugging
    ]
    assert result.metadata["attacker_refused"] is True
    assert result.metadata["turns_used"] == 1
    strategy._is_attacker_refusal.assert_called_once_with("I can't play this role", None)


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_prompt_strategy_ctor_max_turns_caps_below_ceiling(mock_simulator_cls):
    """The strategy's own max_turns wins when smaller than the injected ceiling."""
    mock_sim = MagicMock()
    mock_sim.has_next.return_value = False
    mock_simulator_cls.return_value = mock_sim

    strategy = PromptStrategy("gradual_escalation", "p {max_turns}", max_turns=3)
    strategy.run_attack(_case(), _FakeSession(lambda _m: "r"), max_turns=50)  # ceiling 50

    assert mock_simulator_cls.call_args.kwargs["max_turns"] == 3


def test_max_allowed_turns_importable_from_every_location():
    """The constant moved to strategies/base.py; the old task.py import and the package roots still work."""
    assert redteam.MAX_ALLOWED_TURNS is strategies.MAX_ALLOWED_TURNS is task.MAX_ALLOWED_TURNS is base.MAX_ALLOWED_TURNS
    assert base.MAX_ALLOWED_TURNS == 50


@pytest.mark.parametrize(
    "strategy_cls",
    [
        AttackStrategy,
        BadLikertJudgeStrategy,
        CrescendoStrategy,
        GoatStrategy,
        PairStrategy,
        PromptStrategy,
        SequentialBreakStrategy,
    ],
)
def test_run_attack_max_turns_defaults_to_ceiling(strategy_cls):
    """A custom task can omit max_turns on every strategy."""
    default = inspect.signature(strategy_cls.run_attack).parameters["max_turns"].default
    assert default == base.MAX_ALLOWED_TURNS


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_run_attack_without_max_turns_runs_under_ceiling(mock_simulator_cls):
    mock_simulator_cls.return_value.has_next.return_value = False

    strategy = PromptStrategy("gradual_escalation", "p {max_turns}", max_turns=100)
    strategy.run_attack(_case(), _FakeSession(lambda _m: "r"))

    assert mock_simulator_cls.call_args.kwargs["max_turns"] == base.MAX_ALLOWED_TURNS


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_run_attack_explicit_max_turns_above_ceiling_is_honored(mock_simulator_cls):
    """Only the built-in task enforces MAX_ALLOWED_TURNS; a direct caller's explicit value is not clamped."""
    mock_simulator_cls.return_value.has_next.return_value = False

    strategy = PromptStrategy("gradual_escalation", "p {max_turns}", max_turns=100)
    strategy.run_attack(_case(), _FakeSession(lambda _m: "r"), max_turns=80)

    assert mock_simulator_cls.call_args.kwargs["max_turns"] == 80


@patch("strands_evals.experimental.redteam.strategies.prompt_strategy.ActorSimulator")
def test_prompt_strategy_run_attack_handles_target_exception(mock_simulator_cls):
    mock_sim = MagicMock()
    mock_sim.has_next.side_effect = [True, False]
    mock_simulator_cls.return_value = mock_sim

    def broken(_msg):
        raise RuntimeError("boom")

    result = _prompt_strategy().run_attack(_case(), _FakeSession(broken), max_turns=3)
    assert "[Error: boom]" in result.conversation[1]["content"]


def _refusal_judge_returning(rating):
    judge = MagicMock()
    judge.return_value.structured_output = rating
    return judge


@pytest.mark.parametrize("is_refusal", [True, False])
def test_attacker_refusal_check_returns_the_judge_verdict(is_refusal):
    judge = _refusal_judge_returning(AttackerRefusalRating(is_refusal=is_refusal))
    with patch(f"{_BASE}.Agent", return_value=judge):
        assert PromptStrategy("s", "t")._is_attacker_refusal("some message", None) is is_refusal


def test_attacker_refusal_check_judges_every_message_with_only_the_message():
    """No phrase prefilter: a refusal with no stock phrase still reaches the judge."""
    message = "This asks me to help bypass safety measures, so let's discuss something else."
    judge = _refusal_judge_returning(AttackerRefusalRating(is_refusal=True))
    with patch(f"{_BASE}.Agent", return_value=judge) as agent_cls:
        assert PromptStrategy("s", "t")._is_attacker_refusal(message, "m") is True
    agent_cls.assert_called_once_with(
        model="m", system_prompt=ATTACKER_REFUSAL_JUDGE_SYSTEM_PROMPT, callback_handler=None
    )
    assert judge.call_args.args[0] == f"# Attacker Message\n{message}"
    assert judge.call_args.kwargs["structured_output_model"] is AttackerRefusalRating


def test_attacker_refusal_check_parse_failure_returns_none():
    """A judge that returns no structured output gives no verdict, so the caller can count the failure."""
    with patch(f"{_BASE}.Agent", return_value=_refusal_judge_returning(None)):
        assert PromptStrategy("s", "t")._is_attacker_refusal("some message", None) is None


def test_attacker_refusal_check_judge_error_returns_none():
    """A judge that raises (throttle, structured-output failure) gives no verdict instead of aborting the case."""
    judge = MagicMock(side_effect=RuntimeError("throttled"))
    with patch(f"{_BASE}.Agent", return_value=judge):
        assert PromptStrategy("s", "t")._is_attacker_refusal("some message", None) is None


@pytest.mark.parametrize(
    ("verdict", "stopped", "stats", "conversation"),
    [
        (False, False, {"attacker_refused": False, "refusal_judge_failures": 0}, []),
        (None, False, {"attacker_refused": False, "refusal_judge_failures": 1}, []),
        (
            True,
            True,
            {"attacker_refused": True, "refusal_judge_failures": 0},
            [{"role": "attacker_refused", "content": "msg"}],
        ),
    ],
)
def test_attacker_refused_turn_records_the_verdict(verdict, stopped, stats, conversation):
    strategy = PromptStrategy("s", "t")
    strategy._is_attacker_refusal = MagicMock(return_value=verdict)  # type: ignore[method-assign]
    run_stats = strategy._new_refusal_stats()
    run_conversation: list[dict] = []
    assert strategy._attacker_refused_turn("msg", "m", run_conversation, run_stats) is stopped
    assert run_stats == stats
    assert run_conversation == conversation
    strategy._is_attacker_refusal.assert_called_once_with("msg", "m")
