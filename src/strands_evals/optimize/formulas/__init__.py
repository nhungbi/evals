"""Formulas: runtime hooks whose tunable text the optimizer can edit.

Graduated from strands-labs/harness-optimizer (Apache-2.0). Attach a Formula to an agent at
build time and wrap it with `FormulaArtifact` to make its tunable parameter optimizable under
the acceptance gate.
"""

from .context_expansion_formula import ContextExpansionFormula
from .formula import Formula
from .system_prompt_formula import SystemPromptFormula
from .tool_description_formula import TOOL_DESCRIPTIONS_KEY, ToolDescriptionFormula

__all__ = [
    "TOOL_DESCRIPTIONS_KEY",
    "ContextExpansionFormula",
    "Formula",
    "SystemPromptFormula",
    "ToolDescriptionFormula",
]
