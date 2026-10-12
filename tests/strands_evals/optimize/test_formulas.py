import pytest

from strands_evals.optimize import (
    ArtifactSpace,
    Candidate,
    ContextExpansionFormula,
    FormulaArtifact,
    SystemPromptFormula,
    ToolDescriptionFormula,
)


def test_system_prompt_formula_roundtrip():
    f = SystemPromptFormula(system_prompt="be helpful")
    assert f.get_tunable_params() == {"system_prompt": "be helpful"}
    f.update_params({"system_prompt": "be expert"})
    assert f.process({})["system_prompt"] == "be expert"


def test_tool_description_formula_sparse_overrides():
    f = ToolDescriptionFormula(base={"search": "Search.", "click": "Click."})
    assert f.get_tunable_params() == {"tool_descriptions": {}}
    f.update_params({"tool_descriptions": {"search": "Search the catalog by keyword."}})
    assert f.get_tunable_params() == {"tool_descriptions": {"search": "Search the catalog by keyword."}}
    assert f.effective() == {"search": "Search the catalog by keyword.", "click": "Click."}


def test_tool_description_formula_rejects_unknown_and_oversized():
    f = ToolDescriptionFormula(base={"search": "Search."}, max_chars=20)
    f.update_params({"tool_descriptions": {"nope": "x"}})  # unknown name dropped
    f.update_params({"tool_descriptions": {"search": "x" * 50}})  # over cap dropped
    assert f.overrides == {}
    strict = ToolDescriptionFormula(base={"search": "Search."}, strict=True)
    with pytest.raises(ValueError, match="not in the toolset"):
        strict.update_params({"tool_descriptions": {"nope": "x"}})


def test_context_expansion_formula_first_turn_only():
    f = ContextExpansionFormula(instruction="Consider security.")
    one_turn = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
    out = f.process(one_turn)
    assert "Consider security." in out["messages"][0]["content"][-1]["text"]
    two_turns = {
        "messages": [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}]},
            {"role": "user", "content": [{"text": "again"}]},
        ]
    }
    assert f.process(two_turns) == two_turns  # not the first turn, untouched


def test_formula_requires_trigger():
    from strands_evals.optimize import Formula

    class NoTrigger(Formula):
        def process(self, context, **kwargs):
            return context

        def get_tunable_params(self):
            return {}

        def update_params(self, params):
            return None

    with pytest.raises(ValueError, match="trigger"):
        NoTrigger("x", [])


def test_formula_artifact_bridges_into_space():
    f = ToolDescriptionFormula(base={"search": "Search."})
    space = ArtifactSpace(search=FormulaArtifact(f, "tool_descriptions", size_cap=800))
    assert space.baseline_values() == {"search": "{}"}

    space.validate("search", '{"search": "Search the catalog by keyword."}')
    space["search"].apply_to_formula('{"search": "Search the catalog by keyword."}')
    assert f.effective()["search"] == "Search the catalog by keyword."


def test_formula_artifact_validate_runs_formula_validation():
    f = ToolDescriptionFormula(base={"search": "Search."}, strict=True)
    artifact = FormulaArtifact(f, "tool_descriptions")
    artifact.name = "search"
    with pytest.raises(ValueError):
        artifact.validate('{"unknown": "x"}')  # strict formula rejects, so the probe raises


def test_formula_artifact_rejects_non_json_for_structured_param():
    f = ToolDescriptionFormula(base={"search": "Search."})
    artifact = FormulaArtifact(f, "tool_descriptions")
    artifact.name = "search"
    with pytest.raises(ValueError, match="not valid JSON"):
        artifact.validate("not json")


def test_formula_artifact_candidate_diff_is_clean():
    f = ToolDescriptionFormula(base={"search": "Search."})
    space = ArtifactSpace(search=FormulaArtifact(f, "tool_descriptions", size_cap=800))
    baseline = Candidate(space.baseline_values(), space=space)
    child = baseline.mutate("search", '{"search": "Search by keyword."}')
    assert "Search by keyword." in child.diff(baseline)
