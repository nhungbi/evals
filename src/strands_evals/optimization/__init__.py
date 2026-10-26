"""Eval-driven optimization of agent prompts, components, and numeric knobs."""

from ..types.optimization import (
    AcceptanceVerdict,
    IterationRecord,
    MergePlan,
    Observation,
    ObservationOutcome,
    PairedComparison,
    Patch,
    ProposalOutcome,
    ReflectionRecord,
    SelectionStep,
    TrialStatus,
    Usage,
    Value,
)
from .acceptance.budget import Budget
from .acceptance.gate import Gate
from .candidates.candidate import Candidate
from .declaration.files import FileText, JsonField, SkillLibrary
from .declaration.markers import Choice, Component, Float, Int, OneOf, TextValue, choice, component
from .declaration.space import FileSource, Marker, ParamInfo, ParamKind, Space
from .declaration.tools import ToolDescriptions, tool_descriptions
from .declaration.tunable import TunableFactory, tunable
from .execution.rollout import EnvironmentFn, RolloutFn, TaskFn
from .optimizer import Optimizer
from .result import CandidateView, Grain, OptimizationResult
from .rewards import RewardFunction, RewardFunctionEvaluator, RewardResult
from .scoring.objectives import AtLeast, AtMost, Constraint, Maximize, MetricExpr, Minimize, Objective, metric
from .search.blocks import Block, Search
from .search.method import Method, ProposalRequest
from .search.reflection import Reflection
from .search.schedules import Alternate, Joint, Schedule
from .search.tpe import TPE
from .selection.base import Selection, SelectionContext
from .selection.frontier import CaseFrontier
from .selection.greedy import Greedy

__all__ = [
    "AcceptanceVerdict",
    "Alternate",
    "AtLeast",
    "AtMost",
    "Block",
    "Budget",
    "Candidate",
    "CandidateView",
    "CaseFrontier",
    "Choice",
    "Component",
    "Constraint",
    "EnvironmentFn",
    "FileSource",
    "FileText",
    "Float",
    "Gate",
    "Grain",
    "Greedy",
    "Int",
    "IterationRecord",
    "Joint",
    "JsonField",
    "Marker",
    "Maximize",
    "MergePlan",
    "Method",
    "MetricExpr",
    "Minimize",
    "Objective",
    "Observation",
    "ObservationOutcome",
    "OneOf",
    "OptimizationResult",
    "Optimizer",
    "PairedComparison",
    "ParamInfo",
    "ParamKind",
    "Patch",
    "ProposalOutcome",
    "ProposalRequest",
    "Reflection",
    "ReflectionRecord",
    "RewardFunction",
    "RewardFunctionEvaluator",
    "RewardResult",
    "RolloutFn",
    "Schedule",
    "Search",
    "Selection",
    "SelectionContext",
    "SelectionStep",
    "SkillLibrary",
    "Space",
    "TPE",
    "TaskFn",
    "TextValue",
    "ToolDescriptions",
    "TrialStatus",
    "TunableFactory",
    "Usage",
    "Value",
    "choice",
    "component",
    "metric",
    "tool_descriptions",
    "tunable",
]
