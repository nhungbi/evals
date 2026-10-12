from strands_evals.optimize import paired_bootstrap_compare


def test_clear_improvement_is_accepted():
    incumbent = {f"case_{i}": 0.2 for i in range(10)}
    challenger = {f"case_{i}": 0.9 for i in range(10)}

    verdict = paired_bootstrap_compare(challenger, incumbent)

    assert verdict.accepted
    assert verdict.mean_diff > 0.5
    assert verdict.ci_low > 0
    assert verdict.n_cases == 10


def test_tie_rejects():
    scores = {f"case_{i}": 0.5 for i in range(10)}

    verdict = paired_bootstrap_compare(dict(scores), dict(scores))

    assert not verdict.accepted
    assert "tie" in verdict.reason


def test_noisy_mixed_result_rejects():
    # Half the cases better, half worse by the same amount: mean ~0, CI spans zero.
    incumbent = {f"case_{i}": 0.5 for i in range(10)}
    challenger = {f"case_{i}": 0.5 + (0.3 if i % 2 else -0.3) for i in range(10)}

    verdict = paired_bootstrap_compare(challenger, incumbent)

    assert not verdict.accepted


def test_margin_must_be_cleared():
    incumbent = {f"case_{i}": 0.50 for i in range(10)}
    challenger = {f"case_{i}": 0.52 for i in range(10)}

    assert paired_bootstrap_compare(challenger, incumbent, margin=0.05).accepted is False
    assert paired_bootstrap_compare(challenger, incumbent, margin=0.01).accepted is True


def test_regression_rejects():
    incumbent = {f"case_{i}": 0.9 for i in range(10)}
    challenger = {f"case_{i}": 0.2 for i in range(10)}

    verdict = paired_bootstrap_compare(challenger, incumbent)

    assert not verdict.accepted
    assert verdict.mean_diff < 0


def test_no_overlapping_cases_rejects():
    verdict = paired_bootstrap_compare({"a": 1.0}, {"b": 0.0})

    assert not verdict.accepted
    assert verdict.n_cases == 0


def test_missing_cases_are_dropped_not_imputed():
    incumbent = {"a": 0.0, "b": 0.0, "c": 0.0}
    challenger = {"a": 1.0, "b": 1.0}  # case c errored on every challenger rollout

    verdict = paired_bootstrap_compare(challenger, incumbent)

    assert verdict.n_cases == 2


def test_verdict_is_deterministic_for_a_seed():
    incumbent = {f"case_{i}": 0.4 + 0.01 * i for i in range(8)}
    challenger = {f"case_{i}": 0.6 + 0.02 * i for i in range(8)}

    first = paired_bootstrap_compare(challenger, incumbent, seed=42)
    second = paired_bootstrap_compare(challenger, incumbent, seed=42)

    assert first == second


def test_low_challenger_coverage_rejects():
    incumbent = {f"case_{i}": 0.2 for i in range(10)}
    challenger = {f"case_{i}": 1.0 for i in range(8)}  # cases 8 and 9 errored on every rollout

    verdict = paired_bootstrap_compare(challenger, incumbent)

    assert not verdict.accepted
    assert "coverage floor" in verdict.reason


def test_coverage_floor_is_configurable():
    incumbent = {f"case_{i}": 0.2 for i in range(10)}
    challenger = {f"case_{i}": 1.0 for i in range(8)}

    verdict = paired_bootstrap_compare(challenger, incumbent, min_coverage=0.7)

    assert verdict.accepted
