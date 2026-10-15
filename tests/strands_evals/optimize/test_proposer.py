from strands_evals.optimize import ArtifactSpace, Candidate, TextArtifact, build_reflection_records
from strands_evals.optimize.proposer import Proposal, Proposer


def _space_and_incumbent(**artifacts):
    space = ArtifactSpace(**(artifacts or {"instructions": TextArtifact("be helpful")}))
    incumbent = Candidate(space.baseline_values(), space=space)
    return space, incumbent


def test_build_reflection_records_picks_worst_cases_first():
    rows = {
        name: [
            {"evaluator": "Equals", "score": score, "reason": f"reason for {name}", "input": "q", "actual_output": "a"}
        ]
        for name, score in [("good", 0.9), ("bad", 0.1), ("mid", 0.5)]
    }
    scores = {"good": 0.9, "bad": 0.1, "mid": 0.5}

    records = build_reflection_records(rows, scores, max_records=2)

    assert [r["case"] for r in records] == ["bad", "mid"]
    assert records[0]["Feedback"] == "- Equals (score 0.1): reason for bad"
    assert records[0]["Inputs"] == "q"
    assert records[0]["Generated Outputs"] == "a"


def test_build_reflection_records_skips_cases_without_rows():
    records = build_reflection_records({}, {"orphan": 0.0})
    assert records == []


def test_build_reflection_records_carries_a_trajectory_only_when_present():
    row = {"evaluator": "Equals", "score": 0.0, "reason": "wrong", "input": "q", "actual_output": "a"}
    rows = {"traced": [{**row, "trajectory": "[user] q\n[assistant] a"}], "plain": [{**row, "trajectory": ""}]}

    records = {r["case"]: r for r in build_reflection_records(rows, {"traced": 0.0, "plain": 0.0})}

    assert records["traced"]["Trajectory"] == "[user] q\n[assistant] a"
    assert "Trajectory" not in records["plain"]


def test_default_prompt_describes_the_trajectory():
    assert "Trajectory" in Proposer().system_prompt


def test_filter_valid_drops_unknown_artifact():
    space, incumbent = _space_and_incumbent()
    proposer = Proposer()

    valid = proposer.filter_valid(space, incumbent, [Proposal(artifact_name="nope", new_value="x", rationale="r")])

    assert valid == []


def test_filter_valid_drops_over_cap_value():
    space, incumbent = _space_and_incumbent(instructions=TextArtifact("short", size_cap=10))
    proposer = Proposer()

    valid = proposer.filter_valid(
        space, incumbent, [Proposal(artifact_name="instructions", new_value="x" * 11, rationale="r")]
    )

    assert valid == []


def test_filter_valid_drops_no_op_edit():
    space, incumbent = _space_and_incumbent()
    proposer = Proposer()

    valid = proposer.filter_valid(
        space, incumbent, [Proposal(artifact_name="instructions", new_value="be helpful", rationale="r")]
    )

    assert valid == []


def test_filter_valid_keeps_good_proposal():
    space, incumbent = _space_and_incumbent()
    proposer = Proposer()
    proposal = Proposal(artifact_name="instructions", new_value="be more helpful", rationale="r")

    assert proposer.filter_valid(space, incumbent, [proposal]) == [proposal]


def test_filter_valid_repairs_decorated_artifact_name():
    space, incumbent = _space_and_incumbent()
    proposer = Proposer()

    valid = proposer.filter_valid(
        space,
        incumbent,
        [Proposal(artifact_name="instructions (task-reviewer.sop.md)", new_value="be better", rationale="r")],
    )

    assert len(valid) == 1
    assert valid[0].artifact_name == "instructions"


def test_filter_valid_does_not_repair_ambiguous_name():
    from strands_evals.optimize import ArtifactSpace, Candidate, TextArtifact

    space = ArtifactSpace(sop=TextArtifact("a"), sop_extra=TextArtifact("b"))
    incumbent = Candidate(space.baseline_values(), space=space)
    proposer = Proposer()

    valid = proposer.filter_valid(
        space, incumbent, [Proposal(artifact_name="sop extras", new_value="x", rationale="r")]
    )

    assert valid == []


def test_filter_valid_truncates_to_proposals_per_iteration():
    space, incumbent = _space_and_incumbent()
    proposer = Proposer(proposals_per_iteration=2)
    proposals = [Proposal(artifact_name="instructions", new_value=f"edit {i}", rationale="r") for i in range(5)]

    valid = proposer.filter_valid(space, incumbent, proposals)

    assert len(valid) == 2
    assert [p.new_value for p in valid] == ["edit 0", "edit 1"]


def test_render_artifacts_extends_fence_past_backticks_in_value():
    space, incumbent = _space_and_incumbent(instructions=TextArtifact("code:\n```python\nx = 1\n```\ndone"))

    block = Proposer._render_artifacts(space, incumbent)

    assert "````\ncode:" in block
    assert block.rstrip().endswith("````")


def test_render_artifacts_states_the_size_limit():
    space, incumbent = _space_and_incumbent(instructions=TextArtifact("be helpful", size_cap=500))

    block = Proposer._render_artifacts(space, incumbent)

    assert "Size limit: 500 characters" in block
