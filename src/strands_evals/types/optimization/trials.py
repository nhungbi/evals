"""Trial records for the optimizer: one case executed once, its status and usage."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..evaluation import EvaluationOutput
from ..trace import Session

SplitName = Literal["train", "selection", "holdout"]


class TrialStatus(str, Enum):
    """What happened in a trial, decided from what happened, never inferred from a score.

    `VERIFIER_INVALID` means every evaluator failed, with at least one evaluator. A trial with
    zero evaluators (usage-only objective) is `COMPLETED`.
    """

    COMPLETED = "completed"
    AGENT_FAILURE = "agent_failure"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    VERIFIER_INVALID = "verifier_invalid"


class Usage(BaseModel):
    """Model usage summed over `chat` spans (never `invoke_agent` spans, which carry running totals).

    For a trial: the last attempt's agent spans, judge spans excluded. The same type holds
    `judge_usage`, `search_cost`, and `judge_cost`.

    Field names differ from `types.trace.TokenUsage` on purpose: they are the user-facing metric names
    (`USAGE_METRICS`), and `input_tokens` here includes cache tokens, which `TokenUsage.input_tokens`
    doesn't on every provider.

    Attributes:
        tokens: Everything the model processed, including cache (span `total_tokens`).
        input_tokens: Full prompt including cache, computed as `total_tokens - output_tokens`.
        output_tokens: Everything generated: text, tool-call arguments, reasoning.
        cache_read_tokens: Part of `input_tokens`; None when the provider never reported it
            (unmeasured, not 0).
        cache_write_tokens: Part of `input_tokens`; None when the provider never reported it
            (unmeasured, not 0).
        model_calls: Number of `chat` spans.
        seconds: Duration of the `rollout` span (environment setup and teardown excluded).
    """

    model_config = ConfigDict(frozen=True)

    tokens: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    model_calls: int
    seconds: float


class TrialRecord(BaseModel):
    """One case x one run of one candidate.

    Stores no objective scalar: scoring computes each run's scalar from `evaluations` and `usage`.
    Each evaluator name is in exactly one of `evaluations` and `evaluator_errors`.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    case_name: str
    run_index: int
    split: SplitName
    status: TrialStatus
    error_type: str | None
    error: str | None
    attempts: int
    output: Any
    trajectory: list[Any] | Session | None
    interactions: list | None
    environment_state: list | None
    evaluations: dict[str, EvaluationOutput]
    evaluator_errors: dict[str, str]
    usage: Usage
    judge_usage: Usage
