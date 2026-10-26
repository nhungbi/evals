"""The append-only `lineage.jsonl` event log."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

EventName = Literal[
    "config",
    "propose",
    "score",
    "select",
    "merge",
    "admit",
    "dominated",
    "infeasible",
    "duplicate",
    "unscored",
    "accept",
    "reject",
    "proposer_error",
    "resume",
    "stop",
    "observe",
    "screened",
    "iteration_end",
]


class Lineage:
    """One JSON event per line; the first line is the run's config.

    Every event has `event`, `iteration` (0 for setup and the baseline), and `time` (ISO 8601, for
    reading only; resume never uses it). Records nest whole under the key shown, via `model_dump`.

    - `config`: the `run.json` contents, including `schema_version`.
    - `propose`: `candidate_id`, `parent_ids`, `origin`, `changed_keys`, `rationale`.
    - `score`: `candidate_id`, `split`, `case_names`, `runs`, `rollouts` (spent by this call),
      `per_run` (case name -> each run's scalar), `statuses` (status -> trial count).
    - `select`: `step` (`SelectionStep`).
    - `merge`: `plan` (`MergePlan`), `candidate_id`, `scored` (False when the merged values existed).
    - `admit`, `dominated`: `candidate_id`, `admission` (`Admission`).
    - `infeasible`: at admission, `candidate_id` and `admission`; for a patch that failed validation
      or touched keys outside its block, `block`, `attempt`, `updates`, `reason` (no candidate).
    - `duplicate`: `block`, `attempt`, `candidate_id` (the existing candidate).
    - `unscored`: `candidate_id`, `pending`, `remaining`.
    - `accept`, `reject`: `candidate_id`, `champion_id` (before the decision), `verdict`
      (`AcceptanceVerdict`).
    - `proposer_error`: `block`, `attempt`, `error_type`, `error`.
    - `screened`: `block`, `key`, `reason`.
    - `resume`: `spent`, `budget`.
    - `stop`: `reason` (a `stop_reason`, or `"setup"` when the baseline measured nothing), `message`.
    - `observe`: `block`, `candidate_id` (None for an invalid patch), `observation` (`Observation`).
    - `iteration_end`: `record` (`IterationRecord`), `exhausted` (blocks with no candidates this turn).

    Resume reads: `spent` = sum of `score.rollouts`; pool = baseline + `admit`; champion = the last
    `accept`, else the baseline; candidates and parents from `propose` and `merge`; merges from
    `merge`; turn, exhausted blocks, and iterations from `iteration_end`; method state and
    `search_cost` from `observe`. Only events up to the last `iteration_end` are replayed, except
    `score`, which always counts toward `spent`.
    """

    def __init__(self, path: Path):
        raise NotImplementedError

    def append(self, event: EventName, **fields: Any) -> None:
        """Append one event as a JSON line.

        Args:
            event: The event name.
            **fields: JSON-serializable event fields.
        """
        raise NotImplementedError

    def read(self) -> list[dict[str, Any]]:
        """Every event, dropping a truncated last line."""
        raise NotImplementedError
