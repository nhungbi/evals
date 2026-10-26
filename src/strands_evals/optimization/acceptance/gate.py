"""The holdout gate: the only thing that changes the champion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...types.optimization import AcceptanceVerdict, CandidateScores
from ..scoring.objectives import Criteria


@dataclass(frozen=True, kw_only=True)
class Gate:
    """Acceptance settings; there is no way to turn the gate off.

    Accept only if the challenger's mean holdout improvement exceeds `margin` and the bootstrap CI's
    lower bound is > 0. The false-accept rate per gate test is `(1 - confidence) / 2`
    (0.95 -> 2.5%), not `1 - confidence`.
    """

    runs_per_candidate: int = 3
    holdout_fraction: float = 0.3
    selection_fraction: float = 0.3
    margin: float = 0.0
    confidence: float = 0.95
    min_coverage: float = 0.9


def decide(
    gate: Gate,
    challenger: CandidateScores,
    champion: CandidateScores,
    holdout: Sequence[str],
    criteria: Criteria,
    *,
    seed: int | str,
) -> AcceptanceVerdict:
    """Ordered checks on the holdout: coverage, constraints, tie, margin, CI. Ties reject.

    `seed` is a str from the optimizer (`f"{run seed}:{iteration}:gate"`), so each iteration's bootstrap
    differs but replays on resume; `random.Random` seeds str values the same way in every process,
    unlike `hash()`.

    Args:
        gate: Supplies `margin`, `confidence`, and `min_coverage`.
        challenger: The challenger's holdout scores (`SplitScores.holdout`).
        champion: The champion's holdout scores, cached from when it was scored.
        holdout: Every holdout case name.
        criteria: Checks the holdout constraints.
        seed: Bootstrap seed, a str from the optimizer.

    Returns:
        The verdict; a rejection carries the first failing check as its reason.
    """
    raise NotImplementedError


def covers(candidate: CandidateScores, parent: CandidateScores, cases: Sequence[str], min_coverage: float) -> bool:
    """Whether the candidate measured at least `min_coverage` of the cases the parent measured.

    Args:
        candidate: The candidate's scores on one split.
        parent: The scores it is compared with: the parent at admission, the champion at the gate.
        cases: The split's case names.
        min_coverage: Fraction of the parent's measured cases the candidate must also have measured.
    """
    raise NotImplementedError
