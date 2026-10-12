"""Reward functions as evaluators.

Graduated from strands-labs/harness-optimizer (Apache-2.0). A `RewardFunction` scores one rollout
to a single float, the per-rollout scalar shape RL trainers such as VERL and OpenRLHF consume.
`RewardFunctionEvaluator` wraps one as a strands-evals evaluator so a verifiable reward (tests
pass, exact match) flows through the same `Experiment` pipeline, caching, and acceptance gate as
every judge, just without a written reason unless the reward supplies one.

The rollout is passed as keyword arguments mirroring `EvaluationData`, including the trajectory
and the multi-turn interaction record, so a reward can score a whole conversation (for example
one driven by `ActorSimulator`). This module does no weight training. This contract is the
scoring surface a weight-training integration consumes from outside, so one scoring core serves
both this optimizer and a future trainer.
"""

from abc import ABC, abstractmethod

from typing_extensions import Any

from ..evaluators.evaluator import Evaluator
from ..types.evaluation import EvaluationData, EvaluationOutput


class RewardFunction(ABC):
    """Score a rollout to `{"reward_value": float, "reason": optional str}`.

    The call receives the rollout fields as keyword arguments: `input`, `actual_output`,
    `expected_output`, `expected_assertion`, `actual_trajectory`, `expected_trajectory`,
    `actual_interactions`, `expected_interactions`, and `metadata`. A single-turn verifiable
    reward reads `actual_output` against `expected_output`. A conversation-level reward reads
    `actual_interactions` or `actual_trajectory`. Implementations should read the fields they
    need and absorb the rest with `**_: Any`, so new fields never break an existing reward.
    """

    @abstractmethod
    def __call__(self, **rollout_fields: Any) -> dict[str, Any]: ...


class RewardFunctionEvaluator(Evaluator):
    """Adapts a `RewardFunction` to the `Evaluator` contract."""

    def __init__(self, reward: RewardFunction, pass_threshold: float = 1.0, name: str | None = None):
        super().__init__(name=name)
        self.reward = reward
        self.pass_threshold = pass_threshold

    def evaluate(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        result = self.reward(
            input=evaluation_case.input,
            actual_output=evaluation_case.actual_output,
            expected_output=evaluation_case.expected_output,
            expected_assertion=evaluation_case.expected_assertion,
            actual_trajectory=evaluation_case.actual_trajectory,
            expected_trajectory=evaluation_case.expected_trajectory,
            actual_interactions=evaluation_case.actual_interactions,
            expected_interactions=evaluation_case.expected_interactions,
            metadata=evaluation_case.metadata or {},
        )
        score = float(result.get("reward_value", 0.0))
        return [
            EvaluationOutput(
                score=score,
                test_pass=score >= self.pass_threshold,
                reason=str(result.get("reason", "")) or f"reward_value={score}",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)
