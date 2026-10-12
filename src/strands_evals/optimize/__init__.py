"""Agent optimization: improve an agent's text artifacts against an `Experiment`.

The optimizer rewrites named artifacts (instructions, `SKILL.md` files, tool descriptions, config
fields), scores each candidate with the caller's existing cases and evaluators, and outputs the
best candidate as a reviewable diff. This module does no weight training. The `RewardFunction`
scoring contract is shared with trainers that do, which consume it from outside this package.

See `README.md` in this directory for scope, cost warnings, and usage.
"""

from .acceptance import AcceptanceVerdict, paired_bootstrap_compare
from .artifacts import Artifact, ArtifactSpace, FileArtifact, FormulaArtifact, JsonFieldArtifact, TextArtifact
from .candidate import Candidate
from .contrastive import ContrastiveReflectionProposer
from .declarative import Choice, Component, Float, Int, OneOf, TextValue, TunableFactory
from .factory import choice, component, tunable
from .formulas import ContextExpansionFormula, Formula, SystemPromptFormula, ToolDescriptionFormula
from .optimizer import (
    Budget,
    CandidateScore,
    Gate,
    IterationRecord,
    OptimizationResult,
    Optimizer,
    ProposalOutcome,
    ProposalStatus,
)
from .proposer import Proposal, Proposer, build_reflection_records
from .rewards import RewardFunction, RewardFunctionEvaluator
from .runs import RunDirectory

__all__ = [
    "AcceptanceVerdict",
    "Artifact",
    "ArtifactSpace",
    "Budget",
    "Candidate",
    "Component",
    "Float",
    "Int",
    "Choice",
    "CandidateScore",
    "ContextExpansionFormula",
    "ContrastiveReflectionProposer",
    "FileArtifact",
    "Formula",
    "FormulaArtifact",
    "Gate",
    "IterationRecord",
    "JsonFieldArtifact",
    "OptimizationResult",
    "OneOf",
    "Optimizer",
    "Proposal",
    "ProposalOutcome",
    "ProposalStatus",
    "Proposer",
    "RewardFunction",
    "RewardFunctionEvaluator",
    "RunDirectory",
    "SystemPromptFormula",
    "TextArtifact",
    "TextValue",
    "TunableFactory",
    "ToolDescriptionFormula",
    "build_reflection_records",
    "choice",
    "component",
    "paired_bootstrap_compare",
    "tunable",
]
