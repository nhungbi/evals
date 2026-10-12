"""ContextExpansionFormula: append an instruction to the first user message.

Graduated from strands-labs/harness-optimizer (Apache-2.0). The tunable parameter is
`instruction`; it is injected into turn 0 only.
"""

import logging

from typing_extensions import Any

from .formula import Formula

logger = logging.getLogger(__name__)


class ContextExpansionFormula(Formula):
    """Adds a tunable `instruction` to the first user message."""

    def __init__(self, instruction: str | None = None):
        super().__init__("context_expansion_formula", ["BeforeModelCallEvent"])
        self.instruction = instruction

    def process(self, context: dict, **kwargs: Any) -> dict:
        if not self.instruction:
            logger.warning("instruction is empty | skipping processing")
            return context
        messages = context.get("messages", [])
        user_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "user"]
        if len(user_indices) != 1:
            return context  # only the first turn
        index = user_indices[0]
        message = dict(messages[index])
        content = list(message.get("content", []))
        content.append({"text": f"\n\n{self.instruction}"})
        message["content"] = content
        updated = list(messages)
        updated[index] = message
        return {"messages": updated}

    def get_tunable_params(self) -> dict:
        return {"instruction": self.instruction}

    def update_params(self, params: dict) -> None:
        if "instruction" in params:
            self.instruction = params["instruction"]
