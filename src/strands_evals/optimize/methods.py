"""The search methods: how a candidate is proposed, one algorithm per variable type.

A `Method` proposes typed value updates for the variables of its block, on the parent the case
frontier sampled (the rest of the parent's values hold), and is told every outcome through
`observe`. The value it is told is the candidate's selection-split scalar, never the holdout: the
holdout belongs to the gate alone, so a method cannot learn to fit the cases it is judged on. The
method layer only proposes; acceptance belongs to the gate.

Built-ins: `Reflection` (an LLM reads the parent's training-split failures, the evaluators'
written reasons, and the rollout trajectory when the rollout returned one, and proposes a text
edit), `ContrastiveReflection` (mines winning versus losing cases when evaluators write no
reasons), `TPE` (Optuna's tree-structured Parzen estimator over numbers and choices, an optional
`search` extra), and `SkillCuration` (edits a `SkillLibrary`; lands after v1).
"""

import asyncio
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from .artifacts import _default_size_cap
from .candidate import Candidate, content_id
from .prompt_templates.proposer import contrastive_v1, proposer_v1
from .space import TEXT_KINDS, Space

logger = logging.getLogger(__name__)

DEFAULT_MAX_RECORDS = 8
_MAX_OUTPUT_CHARS = 4_000
_MAX_FEEDBACK_CHARS = 3_000

_OPTUNA_HINT = (
    'TPE needs optuna. Install the search extra: pip install "strands-evals[search]". '
    "To stay dependency-free, assign these variables to Reflection() in an explicit Block instead."
)


@dataclass
class Patch:
    """One proposed candidate change: typed value updates plus the reason for them.

    `draw` is the proposing method's private handle (for example an Optuna trial), carried back
    to the method in `observe`.
    """

    updates: dict[str, Any]
    rationale: str
    method: str = ""
    draw: Any = field(default=None, repr=False, compare=False)


@dataclass
class Observation:
    """What the loop tells the proposing method about one of its patches.

    Attributes:
        patch: The patch this outcome belongs to.
        scalar: The candidate's mean objective scalar on the selection split (never the
            holdout), `None` when it was never scored.
        accepted: The gate verdict, `None` when the candidate never faced the gate.
        crashed: Whether scoring failed outright (constraint rejections are not crashes).
        reason: What happened: admission, the constraint check, or the gate's written reason.
    """

    patch: Patch
    scalar: float | None
    accepted: bool | None
    crashed: bool = False
    reason: str = ""


class Method:
    """One search algorithm: proposes typed updates for its block's variables.

    Attributes:
        kinds: The parameter kinds this method knows how to propose. The default block
            assignment routes each variable to a method whose `kinds` include it.
    """

    kinds: tuple[str, ...] = ()

    async def propose(
        self,
        space: Space,
        keys: list[str],
        parent: Candidate,
        records: list[dict[str, Any]],
        n_proposals: int,
    ) -> list[Patch]:
        """Propose up to `n_proposals` patches over `keys` on `parent`, given its training failures."""
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        """Told the outcome of every patch this method proposed. Default: nothing to update."""
        return None


class _ProposalModel(BaseModel):
    """One structured edit returned by the reflection LLM."""

    artifact_name: str = Field(description="Exact name of the variable to edit, from the provided list.")
    new_value: str = Field(description="The complete new value, not a patch.")
    rationale: str = Field(description="One paragraph: the diagnosed failure cause and why this edit fixes it.")


class _ProposalBatch(BaseModel):
    proposals: list[_ProposalModel]


def build_reflection_records(
    case_rows: dict[str, list[dict[str, Any]]],
    per_case_scores: dict[str, float],
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Build one reflection record per worst-scoring case, in the gepa teacher-prompt shape.

    Args:
        case_rows: Case name to that case's evaluator rows (`evaluator`, `reason`, `input`,
            `actual_output`, and optionally `trajectory`, the rendered trajectory text).
        per_case_scores: Case name to its aggregate score, used to pick the worst cases.
        max_records: How many cases to include, lowest score first.

    Returns:
        Records shaped `{"case", "score", "Inputs", "Generated Outputs", "Feedback"}`, plus
        `"Trajectory"` when the case's rows carry a non-empty one.
    """
    worst = sorted(per_case_scores, key=lambda name: per_case_scores[name])[:max_records]
    records = []
    for name in worst:
        rows = case_rows.get(name, [])
        if not rows:
            continue
        first = rows[0]
        feedback = "\n".join(
            f"- {row.get('evaluator', 'evaluator')} (score {row.get('score', '')}): {row.get('reason', '')}"
            for row in rows
            if row.get("reason")
        )
        record = {
            "case": name,
            "score": round(per_case_scores[name], 4),
            "Inputs": _as_text(first.get("input")),
            "Generated Outputs": _as_text(first.get("actual_output"))[:_MAX_OUTPUT_CHARS],
            "Feedback": feedback[:_MAX_FEEDBACK_CHARS],
        }
        if first.get("trajectory"):
            record["Trajectory"] = first["trajectory"]
        records.append(record)
    return records


class Reflection(Method):
    """An LLM reads the failing cases and the evaluators' written reasons and proposes edits.

    The input is deliberately rich, because textual feedback is this method's entire advantage
    over scalar search: the parent's worst training cases, the agent's actual outputs, every
    evaluator's written reason, and the rollout trajectory when the rollout returned one.
    Guardrails run in code after the call, since a prompt cannot enforce anything: an edit may
    only touch the block's variables, text values have size caps, and numeric or categorical
    edits must parse into the variable's domain.
    """

    kinds = ("text", "skills", "one_of", "choice", "switch", "float", "int")
    template = proposer_v1
    require_records = False

    def __init__(self, model: Model | str | None = None, *, system_prompt: str | None = None):
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else self.template.SYSTEM_PROMPT

    async def propose(
        self,
        space: Space,
        keys: list[str],
        parent: Candidate,
        records: list[dict[str, Any]],
        n_proposals: int,
    ) -> list[Patch]:
        if self.require_records and not records:
            return []
        active_keys = [key for key in keys if key in parent.values]
        prompt = self.template.format_user_prompt(
            artifacts_block=self._render_variables(space, active_keys, parent),
            records_block=json.dumps(records, indent=2, ensure_ascii=False),
            n_proposals=n_proposals,
            artifact_names=active_keys,
        )
        batch = await asyncio.to_thread(self._call, prompt)
        return self._filter_valid(space, active_keys, parent, batch.proposals, n_proposals)

    def _call(self, prompt: str) -> _ProposalBatch:
        agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = agent(prompt, structured_output_model=_ProposalBatch)
        return result.structured_output  # type: ignore[return-value]

    def _filter_valid(
        self,
        space: Space,
        keys: list[str],
        parent: Candidate,
        proposals: list[_ProposalModel],
        n_proposals: int,
    ) -> list[Patch]:
        """Drop proposals that violate a guardrail or change nothing, logging each rejection."""
        patches: list[Patch] = []
        for proposal in proposals:
            key = _repair_key(proposal.artifact_name, keys)
            if key not in keys:
                logger.warning(
                    "key=<%s> | proposal names a variable outside the block, dropped", proposal.artifact_name
                )
                continue
            try:
                value = _typed_value(space, key, proposal.new_value)
                space.validate_value(key, value)
            except (ValueError, KeyError) as error:
                logger.warning("key=<%s>, error=<%s> | proposal rejected by guardrail", key, error)
                continue
            if parent.values.get(key) == value:
                logger.info("key=<%s> | proposal changes nothing, dropped", key)
                continue
            patches.append(Patch(updates={key: value}, rationale=proposal.rationale, method=type(self).__name__))
        if len(patches) > n_proposals:
            logger.info("returned=<%d>, kept=<%d> | more edits than asked, truncating", len(patches), n_proposals)
            patches = patches[:n_proposals]
        return patches

    @staticmethod
    def _render_variables(space: Space, keys: list[str], parent: Candidate) -> str:
        blocks = []
        for key in keys:
            info = space.param(key)
            value = str(parent.values[key])
            note = f"\nEdit instructions: {info.edit_instructions}" if info.edit_instructions else ""
            limit = ""
            if info.kind in TEXT_KINDS:
                cap = info.size_cap if info.size_cap is not None else _default_size_cap(str(info.default))
                limit = (
                    f"\nSize limit: {cap} characters (the value below is {len(value)}; over-limit edits are rejected)"
                )
            elif info.kind in ("float", "int"):
                limit = f"\nAllowed range: {info.low} to {info.high}. Output only the number."
            # A value containing a ``` run would close the fence early; use a longer fence.
            fence = "```"
            while fence in value:
                fence += "`"
            blocks.append(
                f"### artifact_name: {key}\nPath: {space.display_path(key)}{note}{limit}\n{fence}\n{value}\n{fence}"
            )
        return "\n\n".join(blocks)


class ContrastiveReflection(Reflection):
    """Reflect contrastively on failing versus passing cases and append WHEN/DO insights.

    Pick it over `Reflection` when the eval suite's written reasons are thin, or when failures
    live in trajectories rather than in any single reason. It skips the LLM call when there are
    no reflection records, because contrastive reflection has nothing to compare without
    failures.
    """

    template = contrastive_v1
    require_records = True


class TPE(Method):
    """Optuna's tree-structured Parzen estimator over numbers, choices, and switches.

    Bayesian numeric search: it models which regions of the block's variables produced good
    selection-split scalars and proposes the next values there. Needs the `search` extra
    (`optuna`); without it, constructing a candidate proposal raises with the install command.
    `n_startup_trials` is how many candidates a study sees before the model kicks in (proposals
    are random until then); Optuna's default of 10 would make a short run random search, so the
    default here is 4.

    The parent changes from iteration to iteration (the case frontier samples it), and a value
    that is good under one parent's prompt can be bad under another's. So TPE keeps one study per
    context, the parent's values of every variable outside the block, and evidence from different
    contexts is never mixed.
    """

    kinds = ("float", "int", "one_of", "choice", "switch")

    def __init__(self, seed: int | None = None, n_startup_trials: int = 4, **options: Any):
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        self.options = options
        self._studies: dict[str, Any] = {}

    def _optuna_study(self, context: str) -> Any:
        import optuna  # optional extra: search

        if context not in self._studies:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                sampler = optuna.samplers.TPESampler(
                    seed=self.seed,
                    n_startup_trials=self.n_startup_trials,
                    **{"multivariate": True, "group": True, **self.options},
                )
            self._studies[context] = optuna.create_study(direction="maximize", sampler=sampler)
        return self._studies[context]

    async def propose(
        self,
        space: Space,
        keys: list[str],
        parent: Candidate,
        records: list[dict[str, Any]],
        n_proposals: int,
    ) -> list[Patch]:
        context = content_id({key: value for key, value in parent.values.items() if key not in keys})
        try:
            study = self._optuna_study(context)
        except ImportError as error:
            raise ImportError(_OPTUNA_HINT) from error

        patches: list[Patch] = []
        # Shallow keys first, so a choice is suggested before the parameters under it.
        ordered = sorted(keys, key=lambda key: key.count("."))
        for _ in range(n_proposals):
            trial = study.ask()
            updates: dict[str, Any] = {}
            for key in ordered:
                info = space.param(key)
                if not space.is_active(key, {**parent.values, **updates}):
                    continue
                if info.kind in ("one_of", "choice", "switch"):
                    updates[key] = trial.suggest_categorical(key, list(info.options))
                elif info.kind == "int":
                    updates[key] = trial.suggest_int(key, int(info.low), int(info.high), log=info.log)
                elif info.kind == "float":
                    updates[key] = trial.suggest_float(key, info.low, info.high, log=info.log)
            if not updates:
                study.tell(trial, state=_optuna_fail())
                continue
            summary = ", ".join(f"{key}={value!r}" for key, value in updates.items())
            patches.append(
                Patch(updates=updates, rationale=f"TPE proposal: {summary}", method="TPE", draw=(context, trial))
            )
        return patches

    def observe(self, observation: Observation) -> None:
        """Tell the context's study the selection scalar; an unscored candidate fails its trial."""
        draw = observation.patch.draw
        if draw is None:
            return
        context, trial = draw
        study = self._studies.get(context)
        if study is None:
            return
        if observation.scalar is None or observation.crashed:
            study.tell(trial, state=_optuna_fail())
            return
        study.tell(trial, observation.scalar)


def _optuna_fail() -> Any:
    import optuna  # optional extra: search

    return optuna.trial.TrialState.FAIL


class SkillCuration(Method):
    """Curate a `SkillLibrary`: add, rewrite, or drop skill files. Not implemented yet."""

    kinds = ("skills",)

    async def propose(
        self,
        space: Space,
        keys: list[str],
        parent: Candidate,
        records: list[dict[str, Any]],
        n_proposals: int,
    ) -> list[Patch]:
        raise NotImplementedError("SkillCuration lands after v1; route skill variables to Reflection() meanwhile")


def _repair_key(name: str, valid_keys: list[str]) -> str:
    """Strip a parenthesized decoration from a variable name, e.g. `sop (task-reviewer.sop.md)`."""
    if name in valid_keys:
        return name
    stripped = name.strip()
    matches = [valid for valid in valid_keys if stripped == valid or stripped.startswith(f"{valid} (")]
    if len(matches) == 1:
        return matches[0]
    return name


def _typed_value(space: Space, key: str, raw: str) -> Any:
    """Parse the LLM's string into the parameter's typed domain."""
    info = space.param(key)
    if info.kind == "int":
        return int(float(raw.strip()))
    if info.kind == "float":
        return float(raw.strip())
    if info.kind in ("one_of", "choice", "switch"):
        for option in info.options:
            if option == raw or str(option) == raw.strip():
                return option
        return raw.strip()
    return raw


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


# Legacy aliases from the pre-merge module: the old `proposer=` plug point mapped to methods.
Proposer = Reflection
ContrastiveReflectionProposer = ContrastiveReflection
