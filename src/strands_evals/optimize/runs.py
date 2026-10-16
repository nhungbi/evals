"""Run persistence: the append-only directory a single optimization run lives in.

Layout:

    <base>/<run_id>/
        run.json                 # config, budget, train/selection/holdout split, space, blocks
        candidates/<cand_id>/
            manifest.json        # parent id(s), rationale, per-value hashes, status
            values.json          # the typed values
            usage.json           # per-case usage metrics
            artifacts/           # materialized workspace for this candidate
            scores/run<i>/       # one task-result cache directory per repeated run
            scores/run<i>.report.json   # the scored EvaluationReport for that run
        lineage.jsonl            # append-only event log: propose, score, select, merge, admit,
                                 # dominated, infeasible, accept, reject, constraint_reject
        best                     # pointer file naming the current champion

Rejected candidates keep their content and scores, resume replays `lineage.jsonl` (the pool is
rebuilt from its `admit` and `accept` events), and rollback is the `best` pointer simply not
moving.
"""

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

from typing_extensions import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path(".strands-evals/optimize/runs")

SCHEMA_VERSION = 1

# Lifecycle of a candidate in the run directory. `unscored` means the budget ran out first;
# `infeasible` that it broke a constraint on the selection split; `dominated` that it was scored
# but did not land on the case frontier; `admitted` that it joined the pool (it may have
# challenged the champion and lost); `accepted` that it also cleared the gate to become champion.
CANDIDATE_STATUSES = ("proposed", "unscored", "scored", "infeasible", "dominated", "admitted", "accepted")


class RunDirectory:
    """Filesystem handle for one optimization run."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.run_id = self.root.name

    @classmethod
    def create(cls, base_dir: str | Path | None = None, *, config: dict[str, Any]) -> "RunDirectory":
        """Create a fresh run directory with a new run id and write `run.json`."""
        run_id = uuid.uuid4().hex[:6]
        run = cls(Path(base_dir or DEFAULT_BASE_DIR) / run_id)
        run.root.mkdir(parents=True, exist_ok=False)
        (run.root / "candidates").mkdir()
        run.write_config({"schema_version": SCHEMA_VERSION, **config})
        return run

    @classmethod
    def load(cls, root: str | Path) -> "RunDirectory":
        """Open an existing run directory, for resume or audit."""
        run = cls(root)
        if not (run.root / "run.json").exists():
            raise FileNotFoundError(f"{run.root} is not an optimization run directory (no run.json)")
        return run

    # -- run.json --------------------------------------------------------------------

    def write_config(self, config: dict[str, Any]) -> None:
        (self.root / "run.json").write_text(json.dumps(config, indent=2) + "\n")

    def read_config(self) -> dict[str, Any]:
        return json.loads((self.root / "run.json").read_text())

    # -- candidates ------------------------------------------------------------------

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.root / "candidates" / candidate_id

    def workspace_dir(self, candidate_id: str) -> Path:
        return self.candidate_dir(candidate_id) / "artifacts"

    def scores_dir(self, candidate_id: str, run_index: int) -> Path:
        return self.candidate_dir(candidate_id) / "scores" / f"run{run_index}"

    def report_path(self, candidate_id: str, run_index: int, tag: str = "full") -> Path:
        """Report file for one scoring run. `tag` names the case subset scored (for example
        `full` or `holdout`), so a candidate scored on the holdout only and later re-scored on
        the full set keep separate report files instead of one shadowing the other."""
        suffix = "" if tag == "full" else f".{tag}"
        return self.candidate_dir(candidate_id) / "scores" / f"run{run_index}{suffix}.report.json"

    def write_manifest(
        self,
        candidate_id: str,
        *,
        parent_id: str | None,
        rationale: str | None,
        values: dict[str, Any],
        status: str,
        parent_ids: list[str] | None = None,
    ) -> None:
        """Write a candidate's manifest. `parent_ids` is recorded only for merges (two parents)."""
        if status not in CANDIDATE_STATUSES:
            raise ValueError(f"status must be one of {CANDIDATE_STATUSES}, got {status!r}")
        directory = self.candidate_dir(candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            **({"parent_ids": list(parent_ids)} if parent_ids else {}),
            "rationale": rationale,
            "value_hashes": {
                name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                for name, value in sorted(values.items())
            },
            "status": status,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def read_manifest(self, candidate_id: str) -> dict[str, Any] | None:
        path = self.candidate_dir(candidate_id) / "manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set_status(self, candidate_id: str, status: str) -> None:
        manifest = self.read_manifest(candidate_id)
        if manifest is None:
            raise FileNotFoundError(f"candidate {candidate_id} has no manifest")
        if status not in CANDIDATE_STATUSES:
            raise ValueError(f"status must be one of {CANDIDATE_STATUSES}, got {status!r}")
        manifest["status"] = status
        (self.candidate_dir(candidate_id) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def save_values(self, candidate_id: str, values: dict[str, Any]) -> None:
        """Persist the typed parameter values so resume can rebuild the candidate exactly."""
        directory = self.candidate_dir(candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "values.json").write_text(json.dumps(values, indent=2) + "\n")

    def load_values(self, candidate_id: str) -> dict[str, Any] | None:
        path = self.candidate_dir(candidate_id) / "values.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save_usage(self, candidate_id: str, usage: dict[str, dict[str, float]]) -> None:
        """Persist per-case usage metrics, so a resumed run keeps its cost measurements."""
        directory = self.candidate_dir(candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")

    def load_usage(self, candidate_id: str) -> dict[str, dict[str, float]] | None:
        path = self.candidate_dir(candidate_id) / "usage.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # -- lineage ---------------------------------------------------------------------

    def log_event(self, event: str, **fields: Any) -> None:
        """Append one event to `lineage.jsonl`."""
        record = {"time": time.time(), "event": event, **fields}
        with (self.root / "lineage.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("run_id=<%s>, event=<%s> | lineage event recorded", self.run_id, event)

    def load_events(self) -> list[dict[str, Any]]:
        path = self.root / "lineage.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # -- incumbent pointer -----------------------------------------------------------

    def set_best(self, candidate_id: str) -> None:
        (self.root / "best").write_text(candidate_id + "\n")

    def get_best(self) -> str | None:
        path = self.root / "best"
        if not path.exists():
            return None
        return path.read_text().strip()
