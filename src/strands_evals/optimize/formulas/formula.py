"""Formula: a runtime hook that exposes tunable text as optimizable parameters.

A `Formula` attaches to a live agent and processes its context on a trigger event, while exposing
the text it controls as tunable parameters. This module graduates the Formula concept from
strands-labs/harness-optimizer (Apache-2.0) into strands-evals so the optimizer can tune runtime
surfaces (context expansion, tool descriptions) alongside static text artifacts, with no
dependency on that package. `FormulaArtifact` (in `artifacts.py`) adapts any Formula to the
optimizable Artifact protocol.
"""

from abc import ABC, abstractmethod

from typing_extensions import Any


class Formula(ABC):
    """A named, tunable runtime processor of agent context.

    Subclasses define what context to process (`process`), what text is tunable
    (`get_tunable_params` / `update_params`), and when to run (`trigger_timings`, a list of
    strands hook event types or framework-agnostic strings).
    """

    def __init__(self, name: str, trigger_timings: list[Any]):
        if not trigger_timings:
            raise ValueError("trigger_timings must not be empty: a Formula needs at least one trigger")
        self.name = name
        self.trigger_timings = trigger_timings

    @abstractmethod
    def process(self, context: dict, **kwargs: Any) -> dict:
        """Return the context updates this formula applies (for example a new `system_prompt`)."""
        ...

    @abstractmethod
    def get_tunable_params(self) -> dict:
        """Return the current tunable parameters, keyed by name."""
        ...

    @abstractmethod
    def update_params(self, params: dict) -> None:
        """Apply new values for tunable parameters."""
        ...

    def can_process(self, context: dict) -> bool:
        """Whether this formula should run for `context`. Defaults to always."""
        return True
