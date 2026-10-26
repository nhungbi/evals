"""Trial status from the real exception object."""

from __future__ import annotations

from ...types.optimization import TrialStatus


def classify(error: BaseException) -> TrialStatus:
    """`INFRASTRUCTURE_INVALID` for throttling, network, and credential errors; `AGENT_FAILURE` otherwise.

    Only these two come from an exception. `make_task` marks an environment setup error
    `INFRASTRUCTURE_INVALID` whatever its type, and `TrialRunner` sets `VERIFIER_INVALID` when every
    evaluator is in `evaluator_errors`.
    """
    raise NotImplementedError
