"""SystemPromptFormula: a system prompt string as a tunable parameter.

Graduated from strands-labs/harness-optimizer (Apache-2.0).
"""

import logging

from typing_extensions import Any

from .formula import Formula

logger = logging.getLogger(__name__)


class SystemPromptFormula(Formula):
    """Manages a system prompt as the single tunable parameter `system_prompt`."""

    def __init__(self, system_prompt: str | None = None):
        super().__init__("system_prompt_formula", ["BeforeInvocationEvent"])
        self.system_prompt = system_prompt

    def process(self, context: dict, **kwargs: Any) -> dict:
        if not self.system_prompt:
            logger.warning("system_prompt is empty | skipping processing")
            return context
        return {"system_prompt": self.system_prompt}

    def get_tunable_params(self) -> dict:
        return {"system_prompt": self.system_prompt}

    def update_params(self, params: dict) -> None:
        if "system_prompt" in params:
            self.system_prompt = params["system_prompt"]
