from strands_evals.optimize import ArtifactSpace, Candidate, ContrastiveReflectionProposer, TextArtifact
from strands_evals.optimize.proposer import Proposal


def test_contrastive_proposer_empty_records_returns_nothing():
    space = ArtifactSpace(sop=TextArtifact("review the PR"))
    incumbent = Candidate(space.baseline_values(), space=space)
    assert ContrastiveReflectionProposer().propose(space, incumbent, []) == []


def test_contrastive_proposer_reuses_guardrails():
    # It subclasses Proposer, so filter_valid still rejects out-of-space / over-cap / no-op edits.
    space = ArtifactSpace(sop=TextArtifact("review the PR", size_cap=40))
    incumbent = Candidate(space.baseline_values(), space=space)
    proposer = ContrastiveReflectionProposer()
    good = Proposal(artifact_name="sop", new_value="review the PR carefully", rationale="r")
    over_cap = Proposal(artifact_name="sop", new_value="x" * 60, rationale="r")
    unknown = Proposal(artifact_name="nope", new_value="y", rationale="r")
    assert proposer.filter_valid(space, incumbent, [good, over_cap, unknown]) == [good]


def test_contrastive_proposer_uses_contrastive_prompt():
    assert "contrastive" in ContrastiveReflectionProposer().system_prompt.lower()
    assert "WHEN" in ContrastiveReflectionProposer().system_prompt
