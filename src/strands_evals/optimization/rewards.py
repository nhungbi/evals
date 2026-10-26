"""Reward functions shared with RL trainers, and their evaluator adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import NotRequired, TypedDict

from ..evaluators.evaluator import Evaluator
from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT


class RewardResult(TypedDict):
    reward_value: float
    reason: NotRequired[str]


class RewardFunction(ABC):
    """A reward over rollout fields (mirroring `EvaluationData`); absorb extra fields with `**_`."""

    @abstractmethod
    def __call__(self, **rollout_fields: Any) -> RewardResult: ...


class RewardFunctionEvaluator(Evaluator[InputT, OutputT]):
    """Adapts a `RewardFunction` into an `Evaluator`; the reason falls back to `f"reward_value={score}"`.

    Named after the reward's class unless `name=` is given, so two different rewards need no `name=`;
    two instances of the same reward class do. Refer to it in an objective by that name or with
    `metric(instance)`; `metric(RewardFunctionEvaluator)` names no particular reward.
    """

    def __init__(self, reward: RewardFunction, *, pass_threshold: float = 1.0, name: str | None = None):
        """Wrap `reward` as an evaluator.

        Args:
            reward: The reward function to adapt.
            pass_threshold: `test_pass` is `reward_value >= pass_threshold`.
            name: The evaluator name; defaults to the reward's class name, e.g. `"ExactMatchReward"`.
        """
        raise NotImplementedError

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Call the reward with the case's fields; `test_pass` = `reward_value >= pass_threshold`."""
        raise NotImplementedError

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        raise NotImplementedError
