"""The rollout budget: one rollout = one case executed once."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Budget:
    """`max_rollouts` always applies; `max_iterations=None` means no iteration cap."""

    max_rollouts: int = 500
    max_iterations: int | None = None


class BudgetTracker:
    """Counts spent rollouts so real rollouts never exceed `max_rollouts`."""

    remaining: int

    def __init__(self, budget: Budget, spent: int = 0):
        """Start counting from `spent`.

        Args:
            budget: The run's limits.
            spent: Rollouts already spent; on resume, summed from the `score` events.
        """
        raise NotImplementedError

    def can_afford(self, rollouts: int) -> bool:
        """Callers pass `TrialRunner.pending(...)`, never `cases x runs`."""
        raise NotImplementedError

    def spend(self, rollouts: int) -> None:
        raise NotImplementedError
