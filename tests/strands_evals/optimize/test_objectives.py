import pytest

from strands_evals.optimize import AtLeast, Maximize, Projection, WeightedSum


def test_an_evaluator_can_be_named_without_its_suffix():
    projection = Projection(
        Maximize("Helpfulness"),
        [AtLeast("Safety", 0.9)],
        evaluator_names=["HelpfulnessEvaluator", "SafetyEvaluator"],
    )

    assert projection.scalar({"HelpfulnessEvaluator": 0.7, "SafetyEvaluator": 1.0}) == 0.7
    violations = projection.check_constraints({"c1": {"HelpfulnessEvaluator": 0.7, "SafetyEvaluator": 0.5}})
    assert violations == ["Safety >= 0.9: mean 0.5 violates by 0.4"]


def test_full_names_keep_working_and_combine_with_short_ones():
    projection = Projection(
        WeightedSum(Maximize("HelpfulnessEvaluator"), Maximize("Correctness"), weights=[3.0, 1.0]),
        evaluator_names=["HelpfulnessEvaluator", "CorrectnessEvaluator"],
    )

    assert projection.scalar({"HelpfulnessEvaluator": 1.0, "CorrectnessEvaluator": 0.0}) == 0.75


def test_a_short_name_that_is_another_metric_keeps_its_own_meaning():
    # `tokens` is a usage metric, so `tokensEvaluator` gets no short alias.
    projection = Projection(Maximize("tokens"), evaluator_names=["tokensEvaluator"])
    assert projection.scalar({"tokens": 5.0, "tokensEvaluator": 1.0}) == 5.0


def test_an_unknown_metric_still_fails_at_construction():
    with pytest.raises(ValueError, match="unknown metric 'Helpful'"):
        Projection(Maximize("Helpful"), evaluator_names=["HelpfulnessEvaluator"])
