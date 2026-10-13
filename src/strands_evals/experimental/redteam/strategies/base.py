"""Base class for attack strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ....types import EnvironmentState
from ._common import ATTACKER_REFUSAL_JUDGE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from ..case import RedTeamCase
    from .target_session import TargetSession

logger = logging.getLogger(__name__)

# Name of the task-output `environment_state` entry `RedTeamReport` reads run stats from.
RUN_RESULTS = "redteam_run_results"

# Run-stats key a strategy sets when its attacker model declined its role. The target was never fully
# tested, so `AttackSuccessEvaluator` and `RedTeamReport` treat the attack as inconclusive, never defended.
ATTACKER_REFUSED = "attacker_refused"

# Conversation role of the refused attacker message a strategy keeps for debugging; it was never sent to
# the target, so it is the last entry and has no target reply.
ATTACKER_REFUSED_ROLE = "attacker_refused"


class AttackerRefusalRating(BaseModel):
    """Structured judge output: did the attacker decline its task?"""

    is_refusal: bool = Field(description="True if the attacker declined its task instead of writing the message")


@dataclass
class AttackRunResult:
    """Result of one `AttackStrategy.run_attack` execution.

    `strategy_succeeded` / `strategy_score` are observability only; the authoritative verdict comes from
    `AttackSuccessEvaluator` over the trace.

    Attributes:
        conversation: Ordered attacker/target turns; mapped to `output`. When the attacker declined its role,
            a final `ATTACKER_REFUSED_ROLE` entry holds the refused message, which was never sent.
        strategy_succeeded: The strategy's own early-stop signal, not the verdict.
        strategy_score: The strategy's internal stop-decision score, if any.
        metadata: Free-form run details (e.g. `turns_used`, `backtracks`).
        pruned_branches: Refused (attacker, target) pairs dropped during backtracking; empty for append-only
            strategies.
    """

    conversation: list[dict[str, Any]]
    strategy_succeeded: bool = False
    strategy_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pruned_branches: list[dict[str, Any]] = field(default_factory=list)

    def to_environment_state(self) -> EnvironmentState:
        """Return the `RUN_RESULTS` entry a task puts in its `environment_state` for `RedTeamReport` to read.

        The state holds the strategy's `metadata` plus `pruned_branches`.
        """
        return EnvironmentState(name=RUN_RESULTS, state={**self.metadata, "pruned_branches": self.pruned_branches})


class AttackStrategy(ABC):
    """Base class for red team attack strategies.

    The task runner injects a `TargetSession` into `run_attack`; strategies drive it via `invoke` (and
    `snapshot`/`restore` if they backtrack).
    """

    def __init__(self, *, label: str | None = None) -> None:
        """Initialize the strategy.

        Args:
            label: Instance identifier for cross-product case naming and report grouping. Defaults to `name`.
                Pass distinct labels to compare two instances of the same strategy in one experiment.
        """
        self._label = label

    @property
    @abstractmethod
    def name(self) -> str:
        """Type identifier for the strategy (e.g. `"crescendo"`)."""
        ...

    @property
    def label(self) -> str:
        """Instance identifier; defaults to `name`."""
        return self._label or self.name

    @abstractmethod
    def run_attack(
        self,
        case: RedTeamCase,
        target_session: TargetSession,
        *,
        max_turns: int,
        model: Model | str | None = None,
        **kwargs: Any,
    ) -> AttackRunResult:
        """Run the multi-turn attack and return its result.

        Parallel safety: a single strategy instance is shared across every case in an experiment,
        including parallel ones. Implementations MUST keep per-case state in `run_attack` locals
        (or the supplied `target_session`); writing to `self` mid-run is cross-talk between
        concurrent cases. If a strategy needs to track state across turns within one case, use
        local variables. If it needs to reset something on `self` between cases, do it in
        `reset()`, not in `run_attack`.

        Args:
            case: The red team case carrying the attack goal.
            target_session: Session for invoking the target, snapshotting/restoring its state, and reading
                its tool-use `trace`.
            max_turns: Experiment-level ceiling. A strategy with its own `max_turns` should run
                `min(self._max_turns, max_turns)`.
            model: Model for any strategy-internal LLM calls; ctor model takes precedence.
            **kwargs: Reserved for forward compatibility.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Per-case reset hook; no-op by default. Override only if `self` holds mutable state."""

    def _is_attacker_refusal(self, message: str, model: Model | str | None) -> bool:
        """Return True if the attacker's `message` declines its role, judged by a fresh LLM judge.

        Strategies with an attacker model call this on each message before sending it. When it fires, they
        set `ATTACKER_REFUSED` in their run stats and keep the message as a final `ATTACKER_REFUSED_ROLE`
        conversation entry instead of sending it: sent to the target, a refusal would read as a defended
        target. Every message is judged, since a phrase prefilter would let a refusal without a
        known phrase through; only the message is sent, so the call stays small. A judge that fails keeps
        the turn rather than abort the case and lose the turns already collected.

        Args:
            message: The attacker's message, before it is sent to the target.
            model: Model for the judge.
        """
        judge = Agent(model=model, system_prompt=ATTACKER_REFUSAL_JUDGE_SYSTEM_PROMPT, callback_handler=None)
        try:
            result = judge(f"# Attacker Message\n{message}", structured_output_model=AttackerRefusalRating)
        except Exception as e:
            logger.warning("error=<%s> | refusal judge failed | keeping turn", e)
            return False
        rating = result.structured_output
        if rating is None:
            logger.warning("reason=<parse_failure> | refusal judge produced no structured output | keeping turn")
            return False
        return bool(rating.is_refusal)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the strategy's static config.

        The base persists `strategy_type` and `label`; subclasses override to add their own ctor fields.
        """
        out: dict[str, Any] = {"strategy_type": type(self).__name__}
        if self._label is not None:
            out["label"] = self._label
        return out

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        custom_strategies: list[type[AttackStrategy]] | None = None,
    ) -> AttackStrategy:
        """Reconstruct a strategy from its `to_dict` payload.

        Resolves `strategy_type` against the built-in subclasses plus any `custom_strategies` the caller passes.
        """
        # Lazy import: subclasses inherit from AttackStrategy, so importing them at module top
        # would cycle (base -> strategies/__init__ -> crescendo -> base).
        from . import (
            BadLikertJudgeStrategy,
            CrescendoStrategy,
            GoatStrategy,
            PairStrategy,
            PromptStrategy,
            SequentialBreakStrategy,
        )

        registry: dict[str, type[AttackStrategy]] = {
            BadLikertJudgeStrategy.__name__: BadLikertJudgeStrategy,
            CrescendoStrategy.__name__: CrescendoStrategy,
            GoatStrategy.__name__: GoatStrategy,
            PairStrategy.__name__: PairStrategy,
            PromptStrategy.__name__: PromptStrategy,
            SequentialBreakStrategy.__name__: SequentialBreakStrategy,
        }
        for custom in custom_strategies or []:
            registry[custom.__name__] = custom

        payload = dict(data)
        type_name = payload.pop("strategy_type", None)
        if type_name is None:
            raise ValueError("Strategy dict is missing required key 'strategy_type'.")
        if type_name not in registry:
            raise ValueError(f"Cannot find strategy {type_name!r}. Pass `custom_strategies=[...]` to register it.")
        target_cls = registry[type_name]
        return target_cls(**payload)
