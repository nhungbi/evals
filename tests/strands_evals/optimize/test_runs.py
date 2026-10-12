import pytest

from strands_evals.optimize import RunDirectory


def test_create_and_load_roundtrip(tmp_path):
    run = RunDirectory.create(tmp_path, config={"seed": 7})

    loaded = RunDirectory.load(run.root)
    assert loaded.run_id == run.run_id
    assert loaded.read_config() == {"schema_version": 1, "seed": 7}


def test_load_rejects_non_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunDirectory.load(tmp_path)


def test_manifest_and_status(tmp_path):
    run = RunDirectory.create(tmp_path, config={})
    run.write_manifest("abc123", parent_id=None, rationale="baseline", values={"a": "v"}, status="proposed")

    manifest = run.read_manifest("abc123")
    assert manifest["parent_id"] is None
    assert manifest["status"] == "proposed"
    assert "a" in manifest["artifact_hashes"]

    run.set_status("abc123", "accepted")
    assert run.read_manifest("abc123")["status"] == "accepted"

    with pytest.raises(ValueError):
        run.set_status("abc123", "bogus")


def test_lineage_is_append_only_and_ordered(tmp_path):
    run = RunDirectory.create(tmp_path, config={})
    run.log_event("propose", candidate_id="a")
    run.log_event("score", candidate_id="a", rollouts=6)
    run.log_event("accept", candidate_id="a")

    events = run.load_events()
    assert [e["event"] for e in events] == ["propose", "score", "accept"]
    assert events[1]["rollouts"] == 6


def test_best_pointer(tmp_path):
    run = RunDirectory.create(tmp_path, config={})
    assert run.get_best() is None
    run.set_best("abc123")
    assert run.get_best() == "abc123"
    run.set_best("def456")
    assert run.get_best() == "def456"


def test_values_roundtrip(tmp_path):
    run = RunDirectory.create(tmp_path, config={})
    assert run.load_values("abc") is None
    run.save_values("abc", {"instructions": "text"})
    assert run.load_values("abc") == {"instructions": "text"}
