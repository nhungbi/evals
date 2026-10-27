"""Reflection: an LLM contrasts the parent's failing and passing train cases and proposes a text edit."""

from __future__ import annotations

from typing import Any, ClassVar

from strands.models.model import Model

from ...types.optimization import Observation, Patch
from ..declaration.space import ParamKind
from .method import Method, ProposalRequest


class Reflection(Method):
    """Proposes complete new text values by contrasting the parent's failing and passing train cases.

    Seeing passing cases next to failing ones lets it find what separates them even when the
    evaluators write thin or no reasons. Still one structured-output call with no tools.

    The LLM call is retried 3 times; if the last error is infrastructure it raises, otherwise
    unparseable output returns `[]`. Guardrails run in code after the call, including the leakage
    screen (drop a value that adds a shown case's identifier-like name, or a verbatim run of >= 40
    characters from a shown record of either group, that the parent's value didn't contain). The last
    `history` attempts from `observe()` are shown in the prompt ranked best first by selection scalar
    (unmeasured last), each with its new values capped head and tail; 0 turns that off.
    """

    kinds: ClassVar[frozenset[ParamKind]] = frozenset({"text"})
    uses_records: ClassVar[bool] = True

    def __init__(
        self,
        model: Model | str | None = None,
        *,
        system_prompt: str | None = None,
        max_records: int = 8,
        history: int = 5,
    ):
        """Configure the proposer agent.

        Args:
            model: The proposer's model; None means the repo's default Bedrock model.
            system_prompt: Replaces the default prompt.
            max_records: Train cases shown per proposal, split between failing and passing.
            history: Own past attempts shown in the prompt; 0 turns that off.
        """
        raise NotImplementedError

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """Model id, `max_records`, `history`, and the sha256 of the system prompt."""
        raise NotImplementedError
