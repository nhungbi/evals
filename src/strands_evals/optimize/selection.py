"""Selection: how each iteration picks the candidate to build on. The gate alone picks the champion.

A `Selection` owns two decisions and nothing else:

- `choose`: which pool member is this iteration's parent, and whether the iteration merges two
  members instead of calling the search methods.
- `admits`: whether a scored, feasible proposal joins the pool, and which selection cases it leads.

The loop executes the step and records it, and everything downstream (constraints, the coverage
floor, the one-challenger rule, the paired gate, `observe`) is the same for every selection, so
two runs that differ only in selection are a clean comparison of exploration strategies. A
selection only ever reads the selection split; the holdout belongs to the gate.

Built-ins: `Greedy` (the default: always build on the champion) and `CaseFrontier` (sample a
frontier member weighted by the selection cases it leads, and merge complementary specialists).
Pass one as `Optimizer(..., selection=...)`.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .frontier import pareto_front, sample_parent
from .merge import MergePlan, find_merge_pair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionContext:
    """What a selection may read about the run. Built by the loop each iteration.

    Attributes:
        scores: Every pool member's per-case objective scalars on the selection split, in pool
            order (oldest first).
        values: Every pool member's typed values, in pool order.
        parents: Candidate id to its parent ids, for every pool member.
        selection_cases: The selection split's case names.
        champion_id: The current champion.
        rng: Seeded per iteration, so a resumed run makes the same choice.
        merged_pairs: Pairs already merged, or found unmergeable, in this run.
        merges_done: How many merge steps the run has taken.
        validate: Raises `ValueError` when values are not a valid point of the space.
        group_of: Maps a parameter key to the declared name it moves with.
    """

    scores: Mapping[str, Mapping[str, float]]
    values: Mapping[str, Mapping[str, Any]]
    parents: Mapping[str, Sequence[str]]
    selection_cases: Sequence[str]
    champion_id: str
    rng: random.Random
    merged_pairs: frozenset[frozenset[str]] = frozenset()
    merges_done: int = 0
    validate: Callable[[Mapping[str, Any]], None] = lambda values: None
    group_of: Callable[[str], str] = lambda key: key


@dataclass(frozen=True)
class SelectionStep:
    """One iteration's choice.

    Attributes:
        parent_id: The pool member the iteration builds on. For a merge, the merge's first parent.
        merge: The planned merge, or `None` to let the schedule's blocks propose on the parent.
        unmergeable: Pairs found impossible while choosing (the space rejected the merged values).
            The loop records them so they are never planned again.
    """

    parent_id: str
    merge: MergePlan | None = None
    unmergeable: tuple[frozenset[str], ...] = ()


class Selection:
    """How each iteration picks its parent and which proposals join the pool.

    Attributes:
        name: Recorded in `run.json`; a resumed run must use the same selection.
    """

    name = ""

    def choose(self, ctx: SelectionContext) -> SelectionStep:
        """This iteration's parent, and optionally a merge to run instead of proposing."""
        raise NotImplementedError

    def admits(
        self, candidate_id: str, candidate_scores: Mapping[str, float], ctx: SelectionContext
    ) -> list[str] | None:
        """Whether a scored, feasible proposal joins the pool.

        Returns:
            The selection cases it leads (recorded on the `admit` event, possibly empty), or
            `None` when it is not admitted, in which case it is recorded as dominated.
        """
        raise NotImplementedError


class CaseFrontier(Selection):
    """Sample a frontier member weighted by the selection cases it leads; merge specialists.

    A pool member is on the frontier when it is best (exact ties share the lead) on
    at least one selection case and no other member leads a strict superset of its cases; ties
    between identical lead sets go to the older member, so a no-op edit never joins. When two
    frontier members changed different variables since their latest common ancestor, the
    iteration merges them instead of proposing; each pair is merged once, and a run takes at
    most `MAX_MERGES` merge steps. A proposal is admitted only when it lands on the frontier.
    """

    name = "frontier"
    MAX_MERGES = 5
    EPS = 0.0

    def frontier(self, ctx: SelectionContext) -> dict[str, frozenset[str]]:
        """The pool's case frontier on the selection split, in pool order."""
        return pareto_front(ctx.scores, ctx.selection_cases, self.EPS)

    def choose(self, ctx: SelectionContext) -> SelectionStep:
        front = self.frontier(ctx)
        unmergeable: list[frozenset[str]] = []
        if ctx.merges_done < self.MAX_MERGES and len(front) >= 2:
            tried = set(ctx.merged_pairs)
            while (plan := find_merge_pair(front, ctx.values, ctx.parents, tried, group_of=ctx.group_of)) is not None:
                try:
                    ctx.validate(plan.values)
                    return SelectionStep(parent_id=plan.first, merge=plan, unmergeable=tuple(unmergeable))
                except ValueError as error:
                    logger.warning(
                        "first=<%s>, second=<%s>, error=<%s> | merged values rejected by the space",
                        plan.first,
                        plan.second,
                        error,
                    )
                    pair = frozenset((plan.first, plan.second))
                    tried.add(pair)
                    unmergeable.append(pair)
        parent_id = sample_parent(front, ctx.rng) if front else ctx.champion_id
        return SelectionStep(parent_id=parent_id, unmergeable=tuple(unmergeable))

    def admits(
        self, candidate_id: str, candidate_scores: Mapping[str, float], ctx: SelectionContext
    ) -> list[str] | None:
        # The candidate goes after every pool member, so an exact tie on every case it leads
        # goes to the existing member.
        trial = {**ctx.scores, candidate_id: candidate_scores}
        front = pareto_front(trial, ctx.selection_cases, self.EPS)
        return sorted(front[candidate_id]) if candidate_id in front else None


class Greedy(Selection):
    """Always build on the champion: hill-climbing from the current best. No merges. The default.

    Every feasible proposal that covers its parent's selection cases joins the pool, so a
    generalist that beats the champion on average can challenge it even when it leads no single
    case. Pool members other than the champion are never parents.
    """

    name = "greedy"

    def choose(self, ctx: SelectionContext) -> SelectionStep:
        return SelectionStep(parent_id=ctx.champion_id)

    def admits(
        self, candidate_id: str, candidate_scores: Mapping[str, float], ctx: SelectionContext
    ) -> list[str] | None:
        return []
