"""The unified component vocabulary and its two mounts: the decorator and `inject=`."""

import pytest
from strands import Agent
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
from strands.hooks import BeforeModelCallEvent

from strands_evals import Case, Experiment
from strands_evals.evaluators.deterministic import Equals
from strands_evals.optimize import (
    Budget,
    Float,
    Gate,
    Int,
    OneOf,
    Optimizer,
    TextValue,
    choice,
    component,
    tunable,
)
from strands_evals.optimize.proposer import Proposal

from .test_optimizer import ScriptedProposer

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # construction only, never invoked
SOP = "You review pull requests."

CONV_SPACE = choice(
    component(SlidingWindowConversationManager, window_size=Int(4, 200, default=40), per_turn=True),
    component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)),
)
SUMMARIZING_SPEC = '{"type": "SummarizingConversationManager", "params": {"summary_ratio": 0.5}}'


@tunable(
    conversation_manager=CONV_SPACE,
    sop=TextValue(default=SOP, size_cap=200),
    effort=OneOf("low", "medium", "high"),
)
def create_agent(name, conversation_manager, sop, effort):
    agent = Agent(model=MODEL, system_prompt=sop, conversation_manager=conversation_manager, name=name)
    agent.state.set("effort", effort)
    return agent


def test_markers_become_artifacts_with_baselines_and_instructions():
    artifacts = create_agent.artifacts()

    assert set(artifacts) == {"conversation_manager", "sop", "effort"}
    assert artifacts["sop"].read() == SOP
    assert artifacts["effort"].read() == "low"
    assert '"type": "SlidingWindowConversationManager"' in artifacts["conversation_manager"].read()
    assert artifacts["effort"].edit_instructions == "One of: low, medium, high. Output only the value."
    assert "window_size" in artifacts["conversation_manager"].edit_instructions


def test_bare_call_is_the_baseline_system():
    agent = create_agent("baseline")

    assert type(agent.conversation_manager) is SlidingWindowConversationManager
    assert agent.conversation_manager.window_size == 40
    assert agent.state.get("effort") == "low"


def test_bound_candidate_builds_the_swapped_component_at_construction():
    agent_factory = create_agent.bind({"conversation_manager": SUMMARIZING_SPEC, "effort": "high"})

    agent = agent_factory("candidate")
    live = agent.conversation_manager
    # Private SDK registry: the only way to see which object each hook callback belongs to.
    owners = [entry.callback.__self__ for entry in agent.hooks._registered_callbacks[BeforeModelCallEvent]]

    assert type(live) is SummarizingConversationManager
    assert live.summary_ratio == 0.5
    assert agent.state.get("effort") == "high"
    assert any(owner is live for owner in owners)
    assert not any(isinstance(owner, SlidingWindowConversationManager) for owner in owners)


def test_each_bound_call_builds_fresh_components():
    agent_factory = create_agent.bind({"conversation_manager": SUMMARIZING_SPEC})

    assert agent_factory("one").conversation_manager is not agent_factory("two").conversation_manager


def test_validation_rejects_values_outside_the_declaration():
    artifacts = create_agent.artifacts()

    artifacts["effort"].validate("medium")
    artifacts["conversation_manager"].validate(SUMMARIZING_SPEC)
    for name, value in [
        ("effort", "max"),
        ("conversation_manager", '{"type": "NoSuchManager", "params": {}}'),
        ("conversation_manager", '{"type": "SlidingWindowConversationManager", "params": {"window_size": 500}}'),
        ("sop", "x" * 201),
    ]:
        with pytest.raises(ValueError):
            artifacts[name].validate(value)


def test_impossible_component_defaults_fail_at_declaration_time():
    with pytest.raises(TypeError, match="rejects its defaults"):
        tunable(conversation_manager=component(SlidingWindowConversationManager, no_such_kwarg=1))(lambda cm: cm)


def test_inject_wraps_an_untouched_callable_by_keyword():
    def build_agent(**overrides):
        return Agent(model=MODEL, system_prompt=SOP, **overrides)

    factory = tunable(build_agent, inject={"conversation_manager": CONV_SPACE})
    agent = factory.bind({"conversation_manager": SUMMARIZING_SPEC})()

    assert type(agent.conversation_manager) is SummarizingConversationManager
    assert type(factory()).__name__ == "Agent"  # bare call still works, baseline values


def test_inject_rejects_names_the_callable_cannot_accept():
    def fixed_agent(name):
        return name

    with pytest.raises(TypeError, match="accepts neither"):
        tunable(fixed_agent, inject={"conversation_manager": CONV_SPACE})
    with pytest.raises(TypeError, match="expected choice/component"):
        tunable(fixed_agent, inject={"name": "not-a-marker"})


def test_tunable_rejects_mixed_and_bare_forms():
    with pytest.raises(TypeError, match="not both"):
        tunable(lambda: None, inject={"effort": OneOf("low")}, effort=OneOf("low"))
    with pytest.raises(TypeError, match="bare form was removed"):
        tunable(lambda: None)
    with pytest.raises(TypeError, match="requires the callable"):
        tunable(inject={"effort": OneOf("low")})


@tunable(style=OneOf("vague", "precise"))
def answer_factory(style):
    return "42" if style == "precise" else "0"


def make_experiment():
    cases = [Case[str, str](name=f"case_{index}", input=f"q{index}", expected_output="42") for index in range(6)]
    return Experiment(cases=cases, evaluators=[Equals()])


@pytest.mark.parametrize("use_async_run", [False, True])
async def test_optimizer_runs_a_factory_with_a_run_function(tmp_path, use_async_run):
    def run(agent_factory, case):
        return {"output": agent_factory()}

    async def async_run(agent_factory, case):
        return {"output": agent_factory()}

    optimizer = Optimizer(
        make_experiment(),
        factory=answer_factory,
        run=async_run if use_async_run else run,
        budget=Budget(max_rollouts=200),
        gate=Gate(runs_per_candidate=1),
        proposer=ScriptedProposer([[Proposal(artifact_name="style", new_value="precise", rationale="be exact")]]),
        base_dir=tmp_path,
        max_workers=2,
    )

    result = await optimizer.run()

    assert result.baseline_holdout_score == 0.0
    assert result.best_holdout_score == 1.0
    assert result.best.values == {"style": "precise"}
    assert "+precise" in result.best.diff()


def test_optimizer_requires_one_complete_form(tmp_path):
    experiment = make_experiment()

    with pytest.raises(ValueError, match="factory needs run"):
        Optimizer(experiment, factory=answer_factory, base_dir=tmp_path)
    with pytest.raises(ValueError, match="not both"):
        Optimizer(experiment, lambda candidate: None, factory=answer_factory, run=lambda f, c: None, base_dir=tmp_path)
    with pytest.raises(ValueError, match="pass build_task and artifacts"):
        Optimizer(experiment, base_dir=tmp_path)
