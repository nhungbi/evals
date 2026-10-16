from strands_evals.optimize.merge import common_ancestor, find_merge_pair, merge_values


def group(key):
    return key.split(".", 1)[0]


def test_merge_takes_each_group_from_the_side_that_changed_it():
    ancestor = {"prompt": "base", "manager": "Sliding", "manager.Sliding.window": 6}
    first = {"prompt": "better", "manager": "Sliding", "manager.Sliding.window": 6}
    second = {"prompt": "base", "manager": "Summarizing", "manager.Summarizing.ratio": 0.5}

    merged, from_first, from_second = merge_values(ancestor, first, second, group)

    assert merged == {"prompt": "better", "manager": "Summarizing", "manager.Summarizing.ratio": 0.5}
    assert from_first == ("prompt",)
    assert from_second == ("manager",)


def test_a_choice_and_its_settings_move_as_one_unit():
    # Keyed merging would pair one side's choice with the other side's stale setting for it.
    ancestor = {"manager": "Sliding", "manager.Sliding.window": 6}
    first = {"manager": "Sliding", "manager.Sliding.window": 40}
    second = {"manager": "Summarizing", "manager.Summarizing.ratio": 0.5}

    assert merge_values(ancestor, first, second, group) is None


def test_merge_refuses_when_one_side_adds_nothing():
    ancestor = {"a": "x", "b": "y"}
    assert merge_values(ancestor, {"a": "x2", "b": "y"}, dict(ancestor), group) is None


def test_common_ancestor_is_the_latest_shared_one():
    parents = {"base": (), "p": ("base",), "a": ("p",), "b": ("p",)}
    assert common_ancestor("a", "b", parents, ["base", "p", "a", "b"]) == "p"


def test_find_merge_pair_skips_tried_pairs_and_existing_values():
    pool = {"base": {"a": "x", "b": "y"}, "A": {"a": "x2", "b": "y"}, "B": {"a": "x", "b": "y2"}}
    parents = {"base": (), "A": ("base",), "B": ("base",)}
    front = {"A": frozenset({"c1"}), "B": frozenset({"c2"})}

    plan = find_merge_pair(front, pool, parents, set(), group)
    assert plan is not None and plan.values == {"a": "x2", "b": "y2"} and plan.ancestor == "base"

    assert find_merge_pair(front, pool, parents, {frozenset({"A", "B"})}, group) is None
    assert find_merge_pair(front, {**pool, "AB": {"a": "x2", "b": "y2"}}, parents, set(), group) is None
