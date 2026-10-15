"""The execution layer: run one candidate on one case and record what happened honestly.

Three jobs live here. The rollout protocol with its provided default, so the simplest run writes
no rollout code. Usage metering: the bound factory records every agent it builds during a trial
(`record_built`), and the trial sums usage across the recorded agents when it ends, so cost
objectives and constraints work in single-agent and multi-agent rollouts alike. And the trial
status taxonomy: a trial's status is decided by what happened, never inferred from a score, so a
crash never silently becomes a zero.
"""

import inspect
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Any, Callable

from ..utils import is_throttling_error

logger = logging.getLogger(__name__)

#: The usage metric names objectives and constraints may reference, e.g. `AtMost("tokens", 4000)`.
USAGE_METRICS = ("input_tokens", "output_tokens", "tokens", "seconds", "model_calls")

_BUILT_AGENTS: ContextVar[list[Any] | None] = ContextVar("strands_evals_optimize_built_agents", default=None)


def record_built(obj: Any) -> None:
    """Record one object a bound factory built, into the active trial's usage scope.

    Called by `TunableFactory.bind`'s bound callable. Outside a trial this is a no-op, so plain
    factory calls in production code cost nothing.
    """
    agents = _BUILT_AGENTS.get()
    if agents is not None:
        agents.append(obj)


async def default_rollout(agent_factory: Callable[..., Any], case: Any) -> dict[str, Any]:
    """The default rollout: build one agent and invoke it once with the case's input.

    A custom rollout exists for anything that is more than one call: a simulated-user
    conversation, a multi-agent system, a benchmark's own harness.
    """
    agent = agent_factory()
    return {"output": str(await agent.invoke_async(case.input))}


class TrialStatus(str, Enum):
    """What one trial (one case, one repetition) measured.

    `COMPLETED` and `AGENT_FAILURE` measured something (usage counts for both; a failed agent's
    scores are excluded from means and the failure is counted). `INFRASTRUCTURE_INVALID` and
    `VERIFIER_INVALID` measured nothing: infrastructure failures retry, and a trial where every
    evaluator failed is no evidence about the candidate.
    """

    COMPLETED = "completed"
    AGENT_FAILURE = "agent_failure"
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"
    VERIFIER_INVALID = "verifier_invalid"


def classify(error: BaseException) -> TrialStatus:
    """The status of a trial whose task raised `error`: infrastructure, or the agent itself."""
    if is_throttling_error(error):
        return TrialStatus.INFRASTRUCTURE_INVALID
    try:
        from botocore.exceptions import (
            ConnectTimeoutError,
            EndpointConnectionError,
            NoCredentialsError,
            ReadTimeoutError,
        )

        if isinstance(error, (ConnectTimeoutError, EndpointConnectionError, NoCredentialsError, ReadTimeoutError)):
            return TrialStatus.INFRASTRUCTURE_INVALID
    except ImportError:  # pragma: no cover - botocore ships with strands' Bedrock model
        pass
    return TrialStatus.AGENT_FAILURE


def usage_of(agents: list[Any], seconds: float) -> dict[str, float]:
    """Sum usage across the agents one trial built. Metric names are `USAGE_METRICS`."""
    input_tokens = 0.0
    output_tokens = 0.0
    model_calls = 0.0
    for agent in agents:
        metrics = getattr(agent, "event_loop_metrics", None)
        used = dict(getattr(metrics, "accumulated_usage", {}) or {})
        input_tokens += float(used.get("inputTokens", used.get("input_tokens", 0)) or 0)
        output_tokens += float(used.get("outputTokens", used.get("output_tokens", 0)) or 0)
        model_calls += float(getattr(metrics, "cycle_count", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": input_tokens + output_tokens,
        "seconds": seconds,
        "model_calls": model_calls,
    }


@dataclass
class TrialRecord:
    """One trial's outcome: its status and the usage it cost."""

    case_name: str
    run_index: int
    status: TrialStatus
    usage: dict[str, float] = field(default_factory=dict)


class TrialLedger:
    """Every trial of one candidate: statuses and usage, aggregated per case.

    The ledger is the usage side of scoring: evaluator scores come from the experiment's reports,
    usage metrics come from here, and the scoring layer joins them per case.
    """

    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def record(self, case_name: str, run_index: int, status: TrialStatus, usage: dict[str, float]) -> None:
        self.records.append(TrialRecord(case_name=case_name, run_index=run_index, status=status, usage=usage))
        logger.debug(
            "case=<%s>, run=<%d>, status=<%s>, tokens=<%s> | trial recorded",
            case_name,
            run_index,
            status.value,
            usage.get("tokens"),
        )

    def per_case_usage(self) -> dict[str, dict[str, float]]:
        """Each case's usage metrics, averaged over the trials that measured something."""
        by_case: dict[str, list[dict[str, float]]] = {}
        for record in self.records:
            if record.status in (TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE) and record.usage:
                by_case.setdefault(record.case_name, []).append(record.usage)
        return {
            case: {metric: fmean([u.get(metric, 0.0) for u in usages]) for metric in USAGE_METRICS}
            for case, usages in by_case.items()
        }

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.status.value] = counts.get(record.status.value, 0) + 1
        return counts


def instrumented(
    task: Callable[[Any], Any],
    ledger: TrialLedger,
    run_index: int,
    *,
    retries: int = 2,
) -> Callable[[Any], Any]:
    """Wrap one scoring run's task with trial semantics.

    Each call opens a usage scope (the bound factory records the agents it builds into it), times
    the trial, classifies any exception, retries infrastructure failures up to `retries` more
    times, and records the trial in the ledger. Agent failures re-raise so the experiment records
    the error row, which the scoring layer then excludes rather than scoring 0.0.
    """

    async def wrapped(case: Any) -> Any:
        name = getattr(case, "name", None) or ""
        for attempt in range(retries + 1):
            agents: list[Any] = []
            token = _BUILT_AGENTS.set(agents)
            started = time.monotonic()
            try:
                result = task(case)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as error:
                usage = usage_of(agents, time.monotonic() - started)
                status = classify(error)
                if status is TrialStatus.INFRASTRUCTURE_INVALID and attempt < retries:
                    logger.warning(
                        "case=<%s>, attempt=<%d>, error=<%s> | infrastructure failure, trial retries",
                        name,
                        attempt,
                        error,
                    )
                    continue
                ledger.record(name, run_index, status, usage)
                raise
            else:
                ledger.record(name, run_index, TrialStatus.COMPLETED, usage_of(agents, time.monotonic() - started))
                return result
            finally:
                _BUILT_AGENTS.reset(token)
        raise RuntimeError("unreachable")  # pragma: no cover - the loop always returns or raises

    return wrapped
