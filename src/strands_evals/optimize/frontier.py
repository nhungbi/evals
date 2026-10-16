"""Pareto-frontier parent selection, the selection strategy from GEPA.

Every admitted candidate stays in a pool. A candidate is on the frontier when it is the best (within
`eps`) on at least one selection case and no other pool member leads a strict superset of its
cases. Each iteration samples one frontier member as the parent, weighted by how many cases it
leads, so a specialist that uniquely solves a rare case keeps being explored even when its average
is mediocre. Always building on the current best stalls after its first improvement; in GEPA's
ablation (arXiv:2507.19457, Table 3) Pareto selection gained +12.44 against +6.05 for that greedy
strategy.

Ties between identical lead sets go to the earlier pool member, so a no-op edit that scores
exactly like its parent never joins the frontier.
"""

import random

from typing_extensions import Mapping, Sequence


def case_leaders(
    scores: Mapping[str, Mapping[str, float]],
    cases: Sequence[str],
    eps: float = 0.0,
) -> dict[str, frozenset[str]]:
    """Map each candidate to the cases on which it is within `eps` of the best score.

    Args:
        scores: Candidate id to its per-case scores. A case missing from a candidate's scores
            (every rollout errored) never counts as led by that candidate.
        cases: The selection cases to compare on.
        eps: Tolerance below the best score that still counts as leading.

    Returns:
        Candidate id to the frozen set of cases it leads, in `scores` order.
    """
    led: dict[str, set[str]] = {candidate_id: set() for candidate_id in scores}
    for case in cases:
        present = [per_case[case] for per_case in scores.values() if case in per_case]
        if not present:
            continue
        best = max(present)
        for candidate_id, per_case in scores.items():
            if case in per_case and per_case[case] >= best - eps:
                led[candidate_id].add(case)
    return {candidate_id: frozenset(cases_led) for candidate_id, cases_led in led.items()}


def pareto_front(
    scores: Mapping[str, Mapping[str, float]],
    cases: Sequence[str],
    eps: float = 0.0,
) -> dict[str, frozenset[str]]:
    """The non-dominated candidates and the cases each one leads.

    A candidate is dropped when it leads no case, when another candidate leads a strict superset
    of its cases, or when an earlier candidate (in `scores` order) leads exactly the same cases.

    Args:
        scores: Candidate id to its per-case scores, in pool order (oldest first).
        cases: The selection cases to compare on.
        eps: Tolerance below the best score that still counts as leading.

    Returns:
        Frontier member id to the cases it leads, in pool order.
    """
    led = case_leaders(scores, cases, eps)
    front: dict[str, frozenset[str]] = {}
    seen: set[frozenset[str]] = set()
    for candidate_id, cases_led in led.items():
        if not cases_led or cases_led in seen:
            continue
        if any(cases_led < other for other_id, other in led.items() if other_id != candidate_id):
            continue
        seen.add(cases_led)
        front[candidate_id] = cases_led
    return front


def sample_parent(front: Mapping[str, frozenset[str]], rng: random.Random) -> str:
    """Sample one frontier member, weighted by the number of cases it leads.

    Raises:
        ValueError: When the frontier is empty.
    """
    if not front:
        raise ValueError("cannot sample a parent from an empty frontier")
    candidate_ids = list(front)
    return rng.choices(candidate_ids, weights=[len(front[c]) for c in candidate_ids], k=1)[0]
