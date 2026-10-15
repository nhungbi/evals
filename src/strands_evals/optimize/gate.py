"""The acceptance layer: the gate, the budget, and the paired statistical comparison.

LLM judges disagree with themselves between identical runs, so accepting an edit off one score
turns the search into a random walk. The gate is deliberately conservative and has no bypass:

- Scores come from k repeated runs per candidate, averaged per case.
- The comparison is paired per case on the held-out split.
- A challenger wins only when its mean paired difference clears the caller's margin and the
  bootstrap confidence interval of that difference stays above zero.
- Ties and unclear results reject; the incumbent stays.

The gate tests one scalar (the projection's per-case value), so a single objective and the
scalar combiners carry the full guarantee. Verifiable rewards configure the same gate cheaper
(`runs_per_candidate=1`); nothing turns it off.
"""

import random
from dataclasses import dataclass
from typing import Mapping

_DEFAULT_RESAMPLES = 2_000


@dataclass(frozen=True)
class Budget:
    """The authoritative spend cap, denominated in what costs money. The loop is never unbounded.

    Attributes:
        max_rollouts: Hard cap on task executions across the whole run. One rollout is one case
            executed once; k repeated runs of n cases cost k*n rollouts per candidate.
        max_iterations: Optional cap on loop iterations, on top of the rollout cap.
    """

    max_rollouts: int = 500
    max_iterations: int | None = None


@dataclass(frozen=True)
class Gate:
    """Acceptance-gate configuration: how much evidence an edit needs before it is kept.

    Attributes:
        runs_per_candidate: k repeated scoring runs per candidate. The default of 3 suits
            LLM-judge scoring; 1 suits verifiable rewards.
        holdout_fraction: Fraction of cases held out for acceptance. The proposer only sees
            failures from the training split.
        margin: Minimum mean held-out improvement an edit must clear to be accepted.
        confidence: Confidence level for the bootstrap interval in acceptance.
        min_coverage: Minimum fraction of the incumbent's held-out cases a challenger must have
            scores for. Below the floor the challenger is rejected, so an edit that crashes the
            agent on the hardest cases cannot win by being compared only on the cases it
            survived.
        challenger_holdout_only: Score challengers on the held-out split only, cutting the
            dominant per-challenger cost. An accepted challenger is topped up on the training
            split afterward (its holdout cases hit the per-case cache).
    """

    runs_per_candidate: int = 3
    holdout_fraction: float = 0.3
    margin: float = 0.0
    confidence: float = 0.95
    min_coverage: float = 0.9
    challenger_holdout_only: bool = False


@dataclass(frozen=True)
class AcceptanceVerdict:
    """The outcome of one challenger-versus-incumbent comparison.

    Attributes:
        accepted: Whether the challenger replaces the incumbent.
        mean_diff: Mean paired per-case difference (challenger minus incumbent).
        ci_low: Lower bound of the bootstrap confidence interval on `mean_diff`.
        ci_high: Upper bound of the same interval.
        margin: The improvement the caller demanded.
        n_cases: How many held-out cases were compared.
        reason: One sentence saying why the verdict came out this way.
    """

    accepted: bool
    mean_diff: float
    ci_low: float
    ci_high: float
    margin: float
    n_cases: int
    reason: str


def paired_bootstrap_compare(
    challenger_scores: Mapping[str, float],
    incumbent_scores: Mapping[str, float],
    *,
    margin: float = 0.0,
    confidence: float = 0.95,
    min_coverage: float = 0.9,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> AcceptanceVerdict:
    """Compare per-case scores pairwise and decide acceptance.

    Args:
        challenger_scores: Case name to the challenger's mean score on that case.
        incumbent_scores: Case name to the incumbent's mean score on that case.
        margin: Minimum mean improvement demanded. Zero still rejects ties, because the
            confidence interval must clear zero as well.
        confidence: Confidence level of the bootstrap interval on the mean difference.
        min_coverage: Minimum fraction of the incumbent's scored cases the challenger must also
            have scores for. Cases missing from either side (every rollout errored) are dropped
            from the comparison rather than imputed, so without this floor a challenger that
            crashes the agent on the hardest cases would be compared only on the cases it
            survived. Coverage below the floor rejects.
        n_resamples: Bootstrap resamples.
        seed: Seed for the resampling, so verdicts are reproducible from the run directory.

    Returns:
        An `AcceptanceVerdict`.
    """
    shared = sorted(set(challenger_scores) & set(incumbent_scores))
    if not shared:
        return AcceptanceVerdict(
            accepted=False,
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            margin=margin,
            n_cases=0,
            reason="no overlapping held-out cases to compare",
        )
    if len(shared) < min_coverage * len(incumbent_scores):
        return AcceptanceVerdict(
            accepted=False,
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            margin=margin,
            n_cases=len(shared),
            reason=(
                f"challenger has scores for only {len(shared)} of the incumbent's "
                f"{len(incumbent_scores)} held-out cases, below the {min_coverage:.0%} coverage floor"
            ),
        )

    diffs = [challenger_scores[name] - incumbent_scores[name] for name in shared]
    mean_diff = sum(diffs) / len(diffs)

    if all(d == 0.0 for d in diffs):
        return AcceptanceVerdict(
            accepted=False,
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            margin=margin,
            n_cases=len(shared),
            reason="identical scores on every held-out case (tie rejects)",
        )

    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        resampled_means.append(sum(sample) / len(sample))
    resampled_means.sort()
    alpha = 1.0 - confidence
    ci_low = _percentile(resampled_means, alpha / 2)
    ci_high = _percentile(resampled_means, 1.0 - alpha / 2)

    if mean_diff <= margin:
        reason = f"mean improvement {mean_diff:+.4f} did not clear the margin {margin:+.4f}"
        accepted = False
    elif ci_low <= 0.0:
        reason = f"improvement {mean_diff:+.4f} is not distinguishable from noise (CI low {ci_low:+.4f} <= 0)"
        accepted = False
    else:
        reason = f"improvement {mean_diff:+.4f} cleared margin {margin:+.4f} with CI [{ci_low:+.4f}, {ci_high:+.4f}]"
        accepted = True

    return AcceptanceVerdict(
        accepted=accepted,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        margin=margin,
        n_cases=len(shared),
        reason=reason,
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]
