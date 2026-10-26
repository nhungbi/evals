"""CaseFrontier: sample parents from the per-case frontier, or merge two specialists."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import ClassVar

from ...types.optimization import SelectionStep
from .base import Selection, SelectionContext

MAX_MERGES = 5  # merges scored per run, across all pairs


class CaseFrontier(Selection):
    """Parent sampled from the frontier weighted by cases led; admits only proposals on the frontier.

    Merges two frontier members that changed disjoint variables since their latest common ancestor;
    each pair once, at most `MAX_MERGES` per run. A pair whose merged values fail validation is skipped.
    """

    name: ClassVar[str] = "case_frontier"

    def choose(self, ctx: SelectionContext) -> SelectionStep:
        raise NotImplementedError

    def admits(self, candidate_id: str, scores: Mapping[str, float], ctx: SelectionContext) -> bool:
        raise NotImplementedError


def case_frontier(scores: Mapping[str, Mapping[str, float]], cases: Sequence[str]) -> dict[str, frozenset[str]]:
    """Members and the cases each leads; exact ties share, older members win ties to join.

    `scores` maps candidate id -> case name -> score; the result maps member id -> cases it leads.

    Args:
        scores: Candidate id -> case name -> selection-split score.
        cases: Every selection case name.
    """
    raise NotImplementedError


def cases_led(candidate_id: str, scores: Mapping[str, Mapping[str, float]], cases: Sequence[str]) -> list[str]:
    """The cases where `candidate_id` has the best score in `scores` (exact ties share); possibly `[]`.

    `scores` maps candidate id -> case name -> score and includes the candidate. Used by `admit`
    for every strategy.

    Args:
        candidate_id: The candidate.
        scores: Candidate id -> case name -> score, including the candidate.
        cases: Every selection case name.
    """
    raise NotImplementedError


def sample_parent(front: Mapping[str, frozenset[str]], rng: random.Random) -> str:
    """Sample a member weighted by the number of cases it leads.

    Args:
        front: Member id -> the cases it leads.
        rng: The iteration's seeded RNG.
    """
    raise NotImplementedError
