"""Agent optimization: declare the space, run the loop, review what survived the gate.

The developer declares what may change about their agent (prompts, components, numeric knobs,
skills, files on disk), hands the optimizer their existing `Experiment`, and gets back changes
that provably helped: a reviewable diff for text, the winning typed values for knobs. Inside, a
search layer proposes (LLM reflection for text, Bayesian search for numbers and choices), an
execution layer builds agents fresh and meters usage, a scoring layer reduces metrics to one
scalar under optional objectives and constraints, and a mandatory statistical acceptance gate
decides. This module does no weight training; the `RewardFunction` scoring contract is shared
with trainers that do, which consume it from outside this package.

See `README.md` in this directory for scope, cost warnings, and usage.
"""

from .artifacts import Artifact, ArtifactSpace, FileArtifact, FormulaArtifact, JsonFieldArtifact, TextArtifact
from .candidate import Candidate
from .execution import USAGE_METRICS, TrialLedger, TrialStatus, default_rollout
from .factory import TunableFactory, tunable
from .formulas import ContextExpansionFormula, Formula, SystemPromptFormula, ToolDescriptionFormula
from .gate import AcceptanceVerdict, Budget, Gate, paired_bootstrap_compare
from .markers import Choice, Component, Float, Int, OneOf, SkillLibrary, Switch, TextValue, choice, component
from .methods import (
    TPE,
    ContrastiveReflection,
    ContrastiveReflectionProposer,
    Method,
    Observation,
    Patch,
    Reflection,
    SkillCuration,
    build_reflection_records,
)
from .objectives import (
    AtLeast,
    AtMost,
    Chebyshev,
    Combine,
    Constraint,
    Derived,
    Lexicographic,
    Maximize,
    Minimize,
    Objective,
    Pareto,
    Projection,
    WeightedSum,
)
from .optimizer import (
    CandidateScore,
    IterationRecord,
    OptimizationResult,
    Optimizer,
    ProposalOutcome,
    ProposalStatus,
)
from .rewards import RewardFunction, RewardFunctionEvaluator
from .runs import RunDirectory
from .search import Alternate, Block, Joint, Nested, Schedule, Search
from .space import ParamInfo, Space

__all__ = [
    "AcceptanceVerdict",
    "Alternate",
    "Artifact",
    "ArtifactSpace",
    "AtLeast",
    "AtMost",
    "Block",
    "Budget",
    "Candidate",
    "CandidateScore",
    "Chebyshev",
    "Choice",
    "Combine",
    "Component",
    "Constraint",
    "ContextExpansionFormula",
    "ContrastiveReflection",
    "ContrastiveReflectionProposer",
    "Derived",
    "FileArtifact",
    "Float",
    "Formula",
    "FormulaArtifact",
    "Gate",
    "Int",
    "IterationRecord",
    "Joint",
    "JsonFieldArtifact",
    "Lexicographic",
    "Maximize",
    "Method",
    "Minimize",
    "Nested",
    "Objective",
    "Observation",
    "OneOf",
    "OptimizationResult",
    "Optimizer",
    "ParamInfo",
    "Pareto",
    "Patch",
    "Projection",
    "ProposalOutcome",
    "ProposalStatus",
    "Reflection",
    "RewardFunction",
    "RewardFunctionEvaluator",
    "RunDirectory",
    "Schedule",
    "Search",
    "SkillCuration",
    "SkillLibrary",
    "Space",
    "Switch",
    "SystemPromptFormula",
    "TPE",
    "TextArtifact",
    "TextValue",
    "ToolDescriptionFormula",
    "TrialLedger",
    "TrialStatus",
    "TunableFactory",
    "USAGE_METRICS",
    "WeightedSum",
    "build_reflection_records",
    "choice",
    "component",
    "default_rollout",
    "paired_bootstrap_compare",
    "tunable",
]
