"""The parent's worst train-split cases, rendered for a proposer."""

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
    """The `max_records` worst train cases; `scores` is the parent's `SplitScores.train`.

    Args:
        records: The parent's trial records, filtered to split == "train".
        scores: The parent's train-split scores (`SplitScores.train`).
        train_cases: Every train case name.
        max_records: How many of the worst cases to keep.
    """
    raise NotImplementedError


def _render_trajectory(trajectory: Any, *, max_chars: int = 5000) -> str:
    """A capped text view of a trajectory, keeping head and tail.

    Args:
        trajectory: A `Session` or the rollout's own list; read messages by role, never by class.
        max_chars: Cap on the rendered text, keeping head and tail.
    """
    raise NotImplementedError
