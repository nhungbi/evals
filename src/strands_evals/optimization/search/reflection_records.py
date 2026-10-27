"""The parent's worst and best train-split cases, rendered for a proposer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...types.optimization import CandidateScores, ReflectionRecord, TrialRecord


def build_records(
    records: Sequence[TrialRecord],
    scores: CandidateScores,
    train_cases: Sequence[str],
    *,
    max_records: int = 8,
) -> list[ReflectionRecord]:
    """The parent's worst train cases as `failing` and its best as `passing`, worst first.

    Cases are ranked by the parent's per-case objective (ties in `train_cases` order). With
    `k = min(max_records, cases ranked)`, the worst `ceil(k / 2)` are `failing` and the best
    `floor(k / 2)` are `passing`; no case is in both. Ranking, not a pass threshold, so it works for
    any objective, usage terms included. A case with no completed run (every run an
    `agent_failure`) has no score and counts as worst, ahead of every scored case, never `passing`,
    with `score=None` and the error as feedback. `infrastructure_invalid` and `verifier_invalid` runs
    say nothing about the agent and are skipped.

    Args:
        records: The parent's trial records, filtered to split == "train".
        scores: The parent's train-split scores (`SplitScores.train`).
        train_cases: Every train case name.
        max_records: How many cases to show in total, split between the two groups.
    """
    raise NotImplementedError


def _render_trajectory(trajectory: Any, *, max_chars: int = 5000) -> str:
    """A capped text view of a trajectory, keeping head and tail.

    Args:
        trajectory: A `Session` or the rollout's own list; read messages by role, never by class.
        max_chars: Cap on the rendered text, keeping head and tail.
    """
    raise NotImplementedError
