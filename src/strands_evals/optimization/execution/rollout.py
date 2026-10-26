"""The default rollout and the task wrappers `Experiment` drives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import TYPE_CHECKING, Any

from ...case import Case
from .traces import TraceCapture

if TYPE_CHECKING:
    from .trials import TrialLog

TaskFn = Callable[[Case], Any]
RolloutFn = Callable[[Callable[..., Any], Case], Any]
EnvironmentFn = Callable[
    [Case],
    Mapping[str, Any] | AbstractContextManager[Mapping[str, Any]] | AbstractAsyncContextManager[Mapping[str, Any]],
]


async def default_rollout(agent_factory: Callable[..., Any], case: Case) -> dict[str, Any]:
    """Build an agent and return `{"output": str(await agent.invoke_async(case.input))}`.

    Args:
        agent_factory: Builds a fresh agent; call it with no arguments.
        case: The case to run.
    """
    raise NotImplementedError


def make_task(
    agent_factory: Callable[..., Any],
    rollout_fn: RolloutFn,
    environment: EnvironmentFn | None = None,
) -> TaskFn:
    """Wrap a rollout as a task: per-attempt environment, then `rollout_fn` inside a `rollout` span.

    `rollout_fn` receives `functools.partial(agent_factory, **resources)`. An error creating or
    entering the environment is `infrastructure_invalid`; an error exiting it only warns.

    Args:
        agent_factory: The candidate's bound factory.
        rollout_fn: Runs one case with a fresh agent.
        environment: Called once per attempt; returns resources or a context manager yielding them.

    Returns:
        `task(case)`, not yet instrumented.
    """
    raise NotImplementedError


def instrument(task: TaskFn, log: TrialLog, traces: TraceCapture) -> TaskFn:
    """Open one span per attempt, classify and re-raise errors, and attach the `Session` as trajectory.

    Called by `TrialRunner.run`, once per run with a fresh `TrialLog`. A `_ResourceConflictError` is
    stored as `log.setup_error`, not classified. A rollout-returned `trajectory` that is not None wins;
    a returned dict without one gets a copy with the `Session` added, never the dict itself changed;
    a non-dict return becomes `{"output": value, "trajectory": session}`. No agent spans -> trajectory None.

    Args:
        task: The task from `make_task` or `build_task`.
        log: A fresh log for this run.
        traces: The run's trace capture.

    Returns:
        The wrapped task that `Experiment` calls.
    """
    raise NotImplementedError
