"""Variable-wise merge of two case-frontier members, the crossover step from GEPA.

Two frontier members lead different cases by construction (neither leads a superset of the
other's), so each may hold a fix the other lacks. When both descend from a common ancestor and
changed different variables relative to it, the merge takes every variable from whichever side
changed it. No method is called; the merged candidate is scored, admitted, and gated like any
other.

Variables merge by declared name, not by key: a choice and every key under its alternatives
(`manager`, `manager.SummarizingConversationManager.summary_ratio`) move as one unit, so a merge
never pairs one side's choice with the other side's settings for it.

A pair is not mergeable when both sides changed the same variable to different values, when one
side contributes nothing the other lacks, or when the merged values already exist in the pool.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class MergePlan:
    """One planned merge.

    Attributes:
        first: The frontier member listed first in pool order. It is the merged candidate's
            primary parent.
        second: The other frontier member.
        ancestor: The latest common ancestor both are diffed against.
        values: The merged typed values.
        from_first: Declared names taken from `first` (changed by `first` relative to `ancestor`).
        from_second: Declared names taken from `second`.
    """

    first: str
    second: str
    ancestor: str
    values: dict[str, Any]
    from_first: tuple[str, ...]
    from_second: tuple[str, ...]


def lineage(candidate_id: str, parents: Mapping[str, Sequence[str]]) -> set[str]:
    """The candidate and every ancestor reachable through `parents`."""
    seen: set[str] = set()
    stack = [candidate_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


def common_ancestor(
    first: str,
    second: str,
    parents: Mapping[str, Sequence[str]],
    order: Sequence[str],
) -> str | None:
    """The latest (in `order`) candidate both descend from, or `None` when they share none.

    A candidate counts as its own ancestor, so when one side descends from the other the result
    is the older side, and the merge then has nothing to add from it.
    """
    shared = lineage(first, parents) & lineage(second, parents)
    if not shared:
        return None
    position = {candidate_id: index for index, candidate_id in enumerate(order)}
    return max(shared, key=lambda candidate_id: position.get(candidate_id, -1))


def merge_values(
    ancestor: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    group_of: Callable[[str], str] = lambda key: key,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]] | None:
    """Merge two candidates group by group against their common ancestor.

    Args:
        ancestor: The common ancestor's values.
        first: One side's values.
        second: The other side's values.
        group_of: Maps a key to the group it moves with (the space's declared name).

    Returns:
        The merged values and the groups taken from each side, or `None` when both sides changed
        the same group differently or when either side contributes no change.
    """

    def by_group(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for key, value in values.items():
            groups.setdefault(group_of(key), {})[key] = value
        return groups

    base, left, right = by_group(ancestor), by_group(first), by_group(second)
    merged: dict[str, Any] = {}
    from_first: list[str] = []
    from_second: list[str] = []
    for group in sorted(set(base) | set(left) | set(right)):
        old, a, b = base.get(group, {}), left.get(group, {}), right.get(group, {})
        if a != old and b != old and a != b:
            return None
        if a != old:
            merged.update(a)
            from_first.append(group)
        elif b != old:
            merged.update(b)
            from_second.append(group)
        else:
            merged.update(old)
    if not from_first or not from_second:
        return None
    return merged, tuple(from_first), tuple(from_second)


def find_merge_pair(
    front: Mapping[str, frozenset[str]],
    pool_values: Mapping[str, Mapping[str, Any]],
    parents: Mapping[str, Sequence[str]],
    tried: set[frozenset[str]],
    group_of: Callable[[str], str] = lambda key: key,
) -> MergePlan | None:
    """Pick the frontier pair to merge next, or `None` when no untried pair is mergeable.

    Pairs covering the most selection cases between them are tried first; ties go to pairs
    earlier in pool order. `pool_values` iterates in pool order (oldest first).

    Args:
        front: Frontier member id to the selection cases it leads, in pool order.
        pool_values: Every pool member's values, in pool order.
        parents: Candidate id to its parent ids.
        tried: Pairs already merged in this run.
        group_of: Maps a key to the group it moves with (the space's declared name).
    """
    order = list(pool_values)
    members = list(front)
    pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1 :]]
    pairs.sort(key=lambda pair: (-len(front[pair[0]] | front[pair[1]]), members.index(pair[0]), members.index(pair[1])))
    existing = [dict(values) for values in pool_values.values()]
    for first, second in pairs:
        if frozenset((first, second)) in tried:
            continue
        ancestor = common_ancestor(first, second, parents, order)
        if ancestor is None or ancestor not in pool_values:
            continue
        result = merge_values(pool_values[ancestor], pool_values[first], pool_values[second], group_of)
        if result is None:
            continue
        merged, from_first, from_second = result
        if merged in existing:
            continue
        return MergePlan(first, second, ancestor, merged, from_first, from_second)
    return None
