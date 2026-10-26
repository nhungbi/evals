"""Admission on the selection split: constraints, then coverage, then the selection strategy."""

from __future__ import annotations

from collections.abc import Sequence

from ..types.optimization import Admission, CandidateScores
from .scoring.objectives import Criteria
from .selection.base import Selection, SelectionContext


def admit(
    candidate: CandidateScores,
    parent: CandidateScores,
    *,
    selection_cases: Sequence[str],
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
    """
    raise NotImplementedError
