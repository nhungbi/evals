import random

import pytest

from strands_evals.optimize.frontier import case_leaders, pareto_front, sample_parent

CASES = ["c1", "c2", "c3", "c4"]


def test_case_leaders_share_ties_and_respect_eps():
    scores = {"a": {"c1": 1.0, "c2": 0.5}, "b": {"c1": 0.95, "c2": 0.5}}

    assert case_leaders(scores, ["c1", "c2"]) == {"a": frozenset({"c1", "c2"}), "b": frozenset({"c2"})}
    assert case_leaders(scores, ["c1", "c2"], eps=0.1) == {"a": frozenset({"c1", "c2"}), "b": frozenset({"c1", "c2"})}


def test_missing_case_scores_never_lead():
    scores = {"a": {"c1": 0.0}, "b": {}}

    assert case_leaders(scores, ["c1", "c2"]) == {"a": frozenset({"c1"}), "b": frozenset()}


def test_pareto_front_keeps_specialists_and_drops_dominated():
    # The README example: D has the best average but leads only a case A also leads.
    scores = {
        "A": {"c1": 0.9, "c2": 0.5, "c3": 0.6, "c4": 0.4},
        "B": {"c1": 0.7, "c2": 0.8, "c3": 0.6, "c4": 0.3},
        "C": {"c1": 0.6, "c2": 0.4, "c3": 0.5, "c4": 0.9},
        "D": {"c1": 0.8, "c2": 0.7, "c3": 0.6, "c4": 0.4},
    }

    front = pareto_front(scores, CASES)

    assert front == {"A": frozenset({"c1", "c3"}), "B": frozenset({"c2", "c3"}), "C": frozenset({"c4"})}


def test_identical_lead_sets_keep_the_earliest_member():
    scores = {"parent": {"c1": 1.0, "c2": 0.0}, "clone": {"c1": 1.0, "c2": 0.0}}

    assert pareto_front(scores, ["c1", "c2"]) == {"parent": frozenset({"c1", "c2"})}


def test_a_superset_leader_dominates_the_members_it_combines():
    scores = {"a": {"c1": 1.0, "c2": 0.0}, "b": {"c1": 0.0, "c2": 1.0}, "both": {"c1": 1.0, "c2": 1.0}}

    assert pareto_front(scores, ["c1", "c2"]) == {"both": frozenset({"c1", "c2"})}


def test_sample_parent_is_weighted_by_cases_led_and_seeded():
    front = {"wide": frozenset({"c1", "c2", "c3"}), "narrow": frozenset({"c4"})}

    draws = [sample_parent(front, random.Random(f"0:{i}")) for i in range(2000)]

    assert 0.70 < draws.count("wide") / len(draws) < 0.80
    assert sample_parent(front, random.Random("0:7")) == sample_parent(front, random.Random("0:7"))


def test_sample_parent_rejects_an_empty_frontier():
    with pytest.raises(ValueError, match="empty frontier"):
        sample_parent({}, random.Random(0))
