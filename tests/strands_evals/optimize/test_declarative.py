"""The decided design: one declaration vocabulary, mounted through construction.

`@tunable` declarations must expose the right baselines, enforce validation, and produce the
declared components on a real strands Agent. These tests pin that behavior.
"""

import json

import pytest
from strands import Agent
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

from strands_evals.optimize import (
    Choice,
    Component,
    Float,
    Int,
    TextValue,
    tunable,
)

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # construction only, never invoked

CONV_SPACE = Choice(
    Component(SlidingWindowConversationManager, window_size=Int(4, 200, default=40)),
    Component(
        SummarizingConversationManager,
        summary_ratio=Float(0.1, 0.8, default=0.3),
        preserve_recent_messages=Int(4, 16, default=10),
    ),
)

SWAP_VALUE = json.dumps(
    {"type": "SummarizingConversationManager", "params": {"summary_ratio": 0.5, "preserve_recent_messages": 4}}
)


@tunable(conversation_manager=CONV_SPACE, sop=TextValue(default="be helpful", size_cap=2000))
def create_agent(conversation_manager, sop):
    return Agent(model=MODEL, system_prompt=sop, conversation_manager=conversation_manager, callback_handler=None)


def test_decorator_builds_baseline_and_candidate_agents():
    baseline_agent = create_agent()
    assert isinstance(baseline_agent.conversation_manager, SlidingWindowConversationManager)
    assert baseline_agent.conversation_manager.window_size == 40

    agent = create_agent.from_candidate({"conversation_manager": SWAP_VALUE, "sop": "be terse"})
    assert isinstance(agent.conversation_manager, SummarizingConversationManager)
    assert agent.conversation_manager.summary_ratio == 0.5
    assert agent.conversation_manager.preserve_recent_messages == 4
    assert agent.system_prompt == "be terse"

    # fresh component per call: stateful managers never shared
    again = create_agent.from_candidate({"conversation_manager": SWAP_VALUE})
    assert again.conversation_manager is not agent.conversation_manager


def test_decorator_artifacts_validate_like_the_loop_expects():
    artifacts = create_agent.artifacts()
    spec = json.loads(artifacts["conversation_manager"].read())
    assert spec == {"type": "SlidingWindowConversationManager", "params": {"window_size": 40}}

    artifacts["conversation_manager"].validate(SWAP_VALUE)
    with pytest.raises(ValueError, match="unknown component type"):
        artifacts["conversation_manager"].validate(json.dumps({"type": "Nope", "params": {}}))
    with pytest.raises(ValueError, match="no tunable params"):
        artifacts["conversation_manager"].validate(
            json.dumps({"type": "SummarizingConversationManager", "params": {"no_such": 1}})
        )
    with pytest.raises(ValueError, match="outside the allowed range"):
        artifacts["conversation_manager"].validate(
            json.dumps({"type": "SummarizingConversationManager", "params": {"summary_ratio": 7.0}})
        )
    # fixed+tunable mixing: fixed kwargs flow into construction untouched
    fixed = Choice(Component(SummarizingConversationManager, pin_first=2, summary_ratio=Float(0.1, 0.8, default=0.3)))
    manager = fixed.build(json.dumps({"type": "SummarizingConversationManager", "params": {}}))
    assert manager.pin_first == 2


def test_tunable_rejects_markers_without_matching_parameters():
    with pytest.raises(TypeError, match="no such parameters"):

        @tunable(nonexistent=TextValue(default="x"))
        def factory():  # pragma: no cover - never called
            return None


def test_swap_builds_the_declared_component_through_construction():
    """A candidate swap produces the declared component with its params, built at construction."""
    agent = create_agent.from_candidate({"conversation_manager": SWAP_VALUE})
    assert isinstance(agent.conversation_manager, SummarizingConversationManager)
    assert agent.conversation_manager.summary_ratio == 0.5
    assert agent.conversation_manager.preserve_recent_messages == 4


def test_markers_live_inside_an_artifact_space_next_to_file_artifacts(tmp_path):
    from strands_evals.optimize import ArtifactSpace, FileArtifact

    (tmp_path / "sop.md").write_text("review carefully")
    space = ArtifactSpace(
        root=tmp_path,
        sop=FileArtifact("sop.md"),
        conversation=CONV_SPACE,
        temperature=Float(0.0, 1.0, default=0.7),
    )

    baselines = space.baseline_values()
    assert baselines["sop"] == "review carefully"
    assert json.loads(baselines["conversation"])["type"] == "SlidingWindowConversationManager"
    assert baselines["temperature"] == "0.7"

    space.validate("conversation", SWAP_VALUE)  # same guardrails as anywhere else
    with pytest.raises(ValueError, match="outside the allowed range"):
        space.validate("temperature", "3.0")

    manager = CONV_SPACE.build(SWAP_VALUE)  # build_task uses the marker it kept
    assert isinstance(manager, SummarizingConversationManager)
