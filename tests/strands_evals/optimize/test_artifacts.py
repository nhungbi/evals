import json

import pytest

from strands_evals.optimize import ArtifactSpace, FileArtifact, JsonFieldArtifact, TextArtifact


def test_text_artifact_reads_value_and_apply_writes_nothing(tmp_path):
    artifact = TextArtifact("be helpful")
    artifact.name = "instructions"

    assert artifact.read() == "be helpful"
    artifact.apply("new value", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_file_artifact_applies_into_workspace_not_source(tmp_path):
    source_root = tmp_path / "agent"
    source_root.mkdir()
    (source_root / "SKILL.md").write_text("old body")
    workspace = tmp_path / "ws"

    space = ArtifactSpace(root=source_root, skill=FileArtifact("SKILL.md"))
    space.materialize({"skill": "new body"}, workspace)

    assert (workspace / "SKILL.md").read_text() == "new body"
    assert (source_root / "SKILL.md").read_text() == "old body"


def test_file_artifact_frontmatter_guardrail(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text("---\nname: triage\n---\nbody")
    artifact = FileArtifact(path)
    artifact.name = "skill"

    artifact.validate("---\nname: better\n---\nnew body")
    with pytest.raises(ValueError, match="frontmatter"):
        artifact.validate("---\nname: broken\nnew body with no closing fence")


def test_size_cap_guardrail(tmp_path):
    artifact = TextArtifact("short", size_cap=10)
    artifact.name = "instructions"

    artifact.validate("ten chars.")
    with pytest.raises(ValueError, match="cap"):
        artifact.validate("x" * 11)


def test_json_field_artifact_roundtrip(tmp_path):
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"instructions": "old", "model": "m1"}))
    workspace = tmp_path / "ws"

    space = ArtifactSpace(root=tmp_path, instructions=JsonFieldArtifact("agent.json", "/instructions"))
    assert space.baseline_values() == {"instructions": "old"}

    space.materialize({"instructions": "new"}, workspace)
    written = json.loads((workspace / "agent.json").read_text())
    assert written == {"instructions": "new", "model": "m1"}
    assert json.loads(config.read_text())["instructions"] == "old"


def test_json_field_artifact_non_string_field_requires_valid_json(tmp_path):
    config = tmp_path / "agent.json"
    config.write_text(json.dumps({"tools": ["a", "b"]}))
    artifact = JsonFieldArtifact(config, "/tools")
    artifact.name = "tools"

    assert json.loads(artifact.read()) == ["a", "b"]
    artifact.validate('["a", "b", "c"]')
    with pytest.raises(ValueError, match="not valid JSON"):
        artifact.validate("not json at all")


def test_space_rejects_unknown_artifact(tmp_path):
    space = ArtifactSpace(instructions=TextArtifact("x"))
    with pytest.raises(ValueError, match="not in the optimizable space"):
        space.validate("nope", "value")


def test_space_requires_at_least_one_artifact():
    with pytest.raises(ValueError, match="at least one artifact"):
        ArtifactSpace()


def test_space_accepts_plain_mapping_and_merges_kwargs():
    space = ArtifactSpace({"sop file": TextArtifact("a")}, extra=TextArtifact("b"))

    assert sorted(space.names()) == ["extra", "sop file"]
    assert space["sop file"].name == "sop file"


def test_space_rejects_duplicate_mapping_and_kwarg_names():
    with pytest.raises(ValueError, match="given twice"):
        ArtifactSpace({"a": TextArtifact("x")}, a=TextArtifact("y"))


def test_space_rejects_positional_root():
    with pytest.raises(TypeError, match="root="):
        ArtifactSpace("my-agent", a=TextArtifact("x"))


def test_artifact_cannot_be_registered_under_two_names():
    artifact = TextArtifact("x")
    ArtifactSpace(a=artifact)

    with pytest.raises(ValueError, match="already named"):
        ArtifactSpace(b=artifact)


def test_file_artifact_cannot_rebind_to_a_different_root(tmp_path):
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "f.md").write_text("x")
    artifact = FileArtifact("f.md")
    ArtifactSpace(root=first_root, f=artifact)

    with pytest.raises(ValueError, match="already bound"):
        ArtifactSpace(root=second_root, f=artifact)
