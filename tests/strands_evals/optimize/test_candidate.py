from strands_evals.optimize import ArtifactSpace, Candidate, TextArtifact


def test_identical_content_gets_identical_id():
    a = Candidate({"instructions": "x", "skill": "y"})
    b = Candidate({"skill": "y", "instructions": "x"}, parent_id="something", rationale="different lineage")

    assert a.id == b.id
    assert a == b


def test_different_content_gets_different_id():
    assert Candidate({"instructions": "x"}).id != Candidate({"instructions": "y"}).id


def test_mutate_records_lineage():
    parent = Candidate({"instructions": "old"})
    child = parent.mutate("instructions", "new", rationale="because")

    assert child.parent_id == parent.id
    assert child.rationale == "because"
    assert child.values == {"instructions": "new"}
    assert parent.values == {"instructions": "old"}


def test_diff_renders_only_changed_artifacts():
    space = ArtifactSpace(instructions=TextArtifact("old line\n"), skill=TextArtifact("same\n"))
    baseline = Candidate(space.baseline_values(), space=space)
    child = baseline.mutate("instructions", "new line\n")

    diff = child.diff(baseline)
    assert "-old line" in diff
    assert "+new line" in diff
    assert "same" not in diff


def test_diff_of_identical_candidates_is_empty():
    baseline = Candidate({"instructions": "x"})
    assert Candidate({"instructions": "x"}).diff(baseline) == ""


def test_materialize_uses_space(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "SKILL.md").write_text("old")

    from strands_evals.optimize import FileArtifact

    space = ArtifactSpace(root=source, skill=FileArtifact("SKILL.md"))
    candidate = Candidate({"skill": "new"}, space=space, workspace_dir=tmp_path / "ws")

    workspace = candidate.materialize()
    assert (workspace / "SKILL.md").read_text() == "new"


def test_with_context_returns_bound_copy_and_leaves_original_untouched(tmp_path):
    import pytest

    from strands_evals.optimize import ArtifactSpace, TextArtifact

    space = ArtifactSpace(instructions=TextArtifact("be helpful"))
    bare = Candidate(space.baseline_values())

    bound = bare.with_context(space=space, baseline=bare, workspace_dir=tmp_path / "ws")

    assert bound.id == bare.id
    assert bound.baseline is bare
    assert bare.baseline is None
    bound.materialize()
    with pytest.raises(ValueError):
        bare.materialize()
