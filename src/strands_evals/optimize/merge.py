"""Knob-wise merge of two frontier members, the crossover step from GEPA.

Two frontier members lead different cases by construction (neither leads a superset of the
other's), so each may hold a fix the other lacks. When both descend from a common ancestor and
changed different knobs relative to it, the merge takes every knob from whichever side changed
it. No LLM call is involved; the merged candidate is scored, admitted, and gated like any other.

A pair is not mergeable when both sides changed the same knob to different values (resolving
that needs a text merge), when one side contributes nothing the other lacks, or when the merged
values already exist in the pool.
"""

from dataclasses import dataclass

from typing_extensions import Mapping, Sequence


@dataclass(frozen=True)
class MergePlan:
    """One planned merge.

    Attributes:
        first: The frontier member listed first in pool order. It is the merged candidate's
            primary parent.
        second: The other frontier member.
        ancestor: The latest common ancestor both are diffed against.
        values: The merged knob values.
        from_first: Knobs taken from `first` (changed by `first` relative to `ancestor`).
        from_second: Knobs taken from `second`.
    """

    first: str
    second: str
    ancestor: str
    values: dict[str, str]
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
    ancestor: Mapping[str, str],
    first: Mapping[str, str],
    second: Mapping[str, str],
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]] | None:
    """Merge two candidates knob by knob against their common ancestor.

    Returns:
        The merged values and the knobs taken from each side, or `None` when both sides changed
        the same knob to different values or when either side contributes no change.
    """
    merged = dict(ancestor)
    from_first: list[str] = []
    from_second: list[str] = []
    for knob in sorted(set(ancestor) | set(first) | set(second)):
        base, left, right = ancestor.get(knob), first.get(knob), second.get(knob)
        if left != base and right != base and left != right:
            return None
        if left != base and left is not None:
            merged[knob] = left
            from_first.append(knob)
        elif right != base and right is not None:
            merged[knob] = right
            from_second.append(knob)
    if not from_first or not from_second:
        return None
    return merged, tuple(from_first), tuple(from_second)


def find_merge_pair(
    front: Mapping[str, frozenset[str]],
    pool_values: Mapping[str, Mapping[str, str]],
    parents: Mapping[str, Sequence[str]],
    tried: set[frozenset[str]],
) -> MergePlan | None:
    """Pick the frontier pair to merge next, or `None` when no untried pair is mergeable.

    Pairs covering the most selection cases between them are tried first; ties go to pairs
    earlier in pool order. `pool_values` iterates in pool order (oldest first).

    Args:
        front: Frontier member id to the selection cases it leads, in pool order.
        pool_values: Every pool member's knob values, in pool order.
        parents: Candidate id to its parent ids.
        tried: Pairs already merged in this run.
    """
    order = list(pool_values)
    members = list(front)
    pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1 :]]
    pairs.sort(key=lambda pair: (-len(front[pair[0]] | front[pair[1]]), members.index(pair[0]), members.index(pair[1])))
    existing = list(pool_values.values())
    for first, second in pairs:
        if frozenset((first, second)) in tried:
            continue
        ancestor = common_ancestor(first, second, parents, order)
        if ancestor is None or ancestor not in pool_values:
            continue
        result = merge_values(pool_values[ancestor], pool_values[first], pool_values[second])
        if result is None:
            continue
        merged, from_first, from_second = result
        if any(merged == dict(values) for values in existing):
            continue
        return MergePlan(first, second, ancestor, merged, from_first, from_second)
    return None
