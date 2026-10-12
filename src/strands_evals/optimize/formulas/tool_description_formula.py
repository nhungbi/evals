"""ToolDescriptionFormula: optimize an agent's tool descriptions.

Graduated from strands-labs/harness-optimizer (Apache-2.0). A tool description is the text the
model reads when deciding whether to call a tool, how to fill its inputs, and how to read its
result. The formula keeps a full `base` map (one entry per tool, never mutated) and a sparse set
of `overrides` (only the tools an optimizer edited). The overrides are the tunable parameter and
the only thing delivered, so an unedited tool always keeps its own description. This is the
runtime surface the PoC bake-off found highest-leverage on a mature agent.
"""

import logging

from .formula import Formula

logger = logging.getLogger(__name__)

TOOL_DESCRIPTIONS_KEY = "tool_descriptions"
DEFAULT_MAX_CHARS = 800  # Bedrock rejects longer; past this a description belongs in a skill.


class ToolDescriptionFormula(Formula):
    """Formula whose tunable parameter is a sparse set of tool-description overrides.

    Args:
        base: Full `{tool_name: description}` for the agent's toolset. An empty base means the
            toolset is unknown, so override names cannot be checked and each is accepted.
        overrides: Overrides already in force (a previous run's output).
        strict: When `True`, an override for an unknown name or one over `max_chars` raises
            instead of being dropped with a warning.
        max_chars: Upper bound on one description. Defaults to 800.
    """

    def __init__(
        self,
        base: dict[str, str],
        overrides: dict[str, str] | None = None,
        strict: bool = False,
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        super().__init__("tool_description_formula", ["BeforeInvocationEvent"])
        self.strict = strict
        self.max_chars = max_chars
        self.base: dict[str, str] = {str(k): str(v) for k, v in (base or {}).items()}
        self.overrides: dict[str, str] = {}
        if not self.base:
            logger.info("tool_description_formula | empty base, override names cannot be checked")
        if overrides:
            self._merge(overrides)

    def process(self, context: dict, **kwargs) -> dict:
        if not self.overrides:
            return context
        return {TOOL_DESCRIPTIONS_KEY: dict(self.overrides)}

    def get_tunable_params(self) -> dict:
        return {TOOL_DESCRIPTIONS_KEY: dict(self.overrides)}

    def update_params(self, params: dict) -> None:
        edits = params.get(TOOL_DESCRIPTIONS_KEY)
        if edits is None:
            return
        if not isinstance(edits, dict):
            raise ValueError(f"{TOOL_DESCRIPTIONS_KEY} must be a mapping, got {type(edits).__name__}")
        accepted = self._merge(edits)
        logger.info("accepted=<%d>, in_force=<%d> | tool description edits applied", len(accepted), len(self.overrides))

    def effective(self) -> dict[str, str]:
        """`base` with `overrides` applied: what the agent actually sees."""
        out = dict(self.base)
        out.update(self.overrides)
        return out

    def replace_overrides(self, overrides: dict | None) -> None:
        """Set the overrides wholesale (checkpoint restore), validated like edits."""
        self.overrides = {}
        if overrides:
            self._merge(overrides)

    def _merge(self, edits: dict) -> list[str]:
        accepted: list[str] = []
        for name, text in edits.items():
            name = str(name)
            text = "" if text is None else str(text)
            problem = self._problem(name, text)
            if problem == "empty":
                logger.warning("name=<%s> | tool description is empty, keeping current text", name)
                continue
            if problem:
                message = f"rejecting tool description for {name!r}: {problem}"
                if self.strict:
                    raise ValueError(message)
                logger.warning("%s", message)
                continue
            self.overrides[name] = text
            accepted.append(name)
        return accepted

    def _problem(self, name: str, text: str) -> str:
        if not text.strip():
            return "empty"
        if self.base and name not in self.base:
            return f"not in the toolset (known: {sorted(self.base)})"
        if len(text) > self.max_chars:
            return f"{len(text)} chars, over the {self.max_chars}-char limit"
        return ""
