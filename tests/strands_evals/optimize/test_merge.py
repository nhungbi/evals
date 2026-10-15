from strands_evals.optimize.merge import common_ancestor, find_merge_pair, lineage, merge_values

BASE = {"prompt": "p0", "tool": "t0", "window": "40"}


def test_merge_takes_each_knob_from_the_side_that_changed_it():
    first = {**BASE, "prompt": "p1"}
    second = {**BASE, "tool": "t1"}

    merged, from_first, from_second = merge_values(BASE, first, second)

    assert merged == {"prompt": "p1", "tool": "t1", "window": "40"}
    assert (from_first, from_second) == (("prompt",), ("tool",))


def test_conflicting_edits_to_the_same_knob_do_not_merge():
    assert merge_values(BASE, {**BASE, "prompt": "p1"}, {**BASE, "prompt": "p2", "tool": "t1"}) is None


def test_identical_edits_to_the_same_knob_merge():
    first = {**BASE, "prompt": "p1", "window": "64"}
    second = {**BASE, "prompt": "p1", "tool": "t1"}

    merged, _, from_second = merge_values(BASE, first, second)

    assert merged == {"prompt": "p1", "tool": "t1", "window": "64"}
    assert from_second == ("tool",)


def test_a_side_with_nothing_new_does_not_merge():
    assert merge_values(BASE, {**BASE, "prompt": "p1"}, dict(BASE)) is None


def test_common_ancestor_is_the_latest_shared_one():
    parents = {"base": (), "a": ("base",), "a1": ("a",), "a2": ("a",), "m": ("a1", "b")}
    order = ["base", "a", "b", "a1", "a2", "m"]
    parents["b"] = ("base",)

    assert lineage("m", parents) == {"m", "a1", "a", "b", "base"}
    assert common_ancestor("a1", "a2", parents, order) == "a"
    assert common_ancestor("a1", "b", parents, order) == "base"
    assert common_ancestor("a", "a1", parents, order) == "a"  # one descends from the other


def test_find_merge_pair_prefers_widest_coverage_and_skips_tried_pairs():
    values = {
        "base": dict(BASE),
        "a": {**BASE, "prompt": "p1"},
        "b": {**BASE, "tool": "t1"},
        "c": {**BASE, "window": "64"},
    }
    parents = {"base": (), "a": ("base",), "b": ("base",), "c": ("base",)}
    front = {"a": frozenset({"c1", "c2"}), "b": frozenset({"c3", "c4"}), "c": frozenset({"c5"})}

    plan = find_merge_pair(front, values, parents, tried=set())
    assert (plan.first, plan.second, plan.ancestor) == ("a", "b", "base")
    assert plan.values == {"prompt": "p1", "tool": "t1", "window": "40"}

    retry = find_merge_pair(front, values, parents, tried={frozenset(("a", "b"))})
    assert (retry.first, retry.second) == ("a", "c")


def test_find_merge_pair_skips_merges_already_in_the_pool():
    values = {
        "base": dict(BASE),
        "a": {**BASE, "prompt": "p1"},
        "b": {**BASE, "tool": "t1"},
        "ab": {**BASE, "prompt": "p1", "tool": "t1"},
    }
    parents = {"base": (), "a": ("base",), "b": ("base",), "ab": ("a", "b")}
    front = {"a": frozenset({"c1"}), "b": frozenset({"c2"})}

    assert find_merge_pair(front, values, parents, tried=set()) is None
