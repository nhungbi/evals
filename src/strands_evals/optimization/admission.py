"""Admission on the selection split: constraints, then coverage, then the selection strategy."""

from __future__ import annotations

from ..types.optimization import Admission, CandidateScores
from .scoring.objectives import Criteria
from .selection.base import Selection, SelectionContext


def admit(
    candidate: CandidateScores,
    parent: CandidateScores,
    *,
    criteria: Criteria,
    selection: Selection,
    ctx: SelectionContext,
    min_coverage: float,
) -> Admission:
    """Selection-split scores only, never the holdout.

    Fixed order: constraints (`criteria.violations`), coverage (`acceptance.gate.covers`, with
    `min_coverage` = `gate.min_coverage`), then `selection.admits`. The first two always run, so a
    custom `Selection` can't skip them. `Admission.cases_led` comes from `selection.frontier.cases_led`
    over `ctx.scores` plus the candidate, whatever the strategy.

    Args:
        candidate: The candidate's selection-split scores.
        parent: The parent's selection-split scores; for a merge, the union of both parents' cases.
        criteria: Checks the constraints on the selection split.
        selection: The strategy, asked last.
        ctx: The pool as selection sees it; supplies the selection case names.
        min_coverage: `gate.min_coverage`.

    Returns:
        `admitted`, `infeasible` (a constraint or coverage failed), or `dominated` (the strategy said no).
    """
    raise NotImplementedError
