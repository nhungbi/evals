"""Reflection-based edit proposal.

The proposer is one LLM call whose input is deliberately rich, because textual feedback is this
method's entire advantage over scalar-reward optimization: the worst-scoring cases, the agent's
actual outputs, and every evaluator's written `reason`. The records follow the field-tested shape
from the gepa library's teacher prompts: `Inputs`, `Generated Outputs`, `Feedback`, plus
`Trajectory` when the task returned one (rendered by `trajectories.render_trajectory`).

Guardrails are enforced in code after the call, since a prompt cannot enforce anything: an edit
may only touch artifacts in the space, every artifact has a size cap, and file and config
artifacts must pass format validity checks.
"""

import asyncio
import json
import logging

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Any, cast

from .artifacts import ArtifactSpace, _default_size_cap
from .candidate import Candidate
from .prompt_templates.proposer import proposer_v1

logger = logging.getLogger(__name__)

DEFAULT_MAX_RECORDS = 8
_MAX_OUTPUT_CHARS = 4_000
_MAX_FEEDBACK_CHARS = 3_000


class Proposal(BaseModel):
    """One structured edit returned by the proposer LLM."""

    artifact_name: str = Field(description="Exact name of the artifact to edit, from the provided list.")
    new_value: str = Field(description="The complete new value for the artifact, not a patch.")
    rationale: str = Field(description="One paragraph: the diagnosed failure cause and why this edit fixes it.")


class ProposalBatch(BaseModel):
    """Structured output wrapper for the proposer call."""

    proposals: list[Proposal]


def build_reflection_records(
    case_rows: dict[str, list[dict[str, Any]]],
    per_case_scores: dict[str, float],
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    """Build one reflection record per worst-scoring case.

    Args:
        case_rows: Case name to that case's evaluator rows. Each row needs `evaluator`,
            `reason`, and the case's `input` / `actual_output` fields from the report, and may
            carry `trajectory`, the rendered trajectory text.
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


class Proposer:
    """LLM edit proposer with code-enforced guardrails.

    Subclasses select a different prompt by overriding `template` (a module exporting
    `SYSTEM_PROMPT` and `format_user_prompt`) and may set `require_records = True` to skip the
    call when there are no reflection records to learn from.
    """

    template = proposer_v1
    require_records = False

    def __init__(
        self,
        model: Model | str | None = None,
        *,
        proposals_per_iteration: int = 3,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.proposals_per_iteration = proposals_per_iteration
        self.system_prompt = system_prompt if system_prompt is not None else self.template.SYSTEM_PROMPT

    def propose(
        self,
        space: ArtifactSpace,
        incumbent: Candidate,
        records: list[dict[str, Any]],
    ) -> list[Proposal]:
        """Ask the LLM for edits and return only the ones that pass every guardrail."""
        if self.require_records and not records:
            return []
        agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = agent(self._build_prompt(space, incumbent, records), structured_output_model=ProposalBatch)
        batch = cast(ProposalBatch, result.structured_output)
        return self.filter_valid(space, incumbent, batch.proposals)

    async def propose_async(
        self,
        space: ArtifactSpace,
        incumbent: Candidate,
        records: list[dict[str, Any]],
    ) -> list[Proposal]:
        """Async entry point; the optimizer loop calls this one.

        The default delegates to `propose` in a worker thread, so a custom proposer only needs
        to override `propose`. Override this too when the model can be called natively async.
        """
        return await asyncio.to_thread(self.propose, space, incumbent, records)

    def _build_prompt(self, space: ArtifactSpace, incumbent: Candidate, records: list[dict[str, Any]]) -> str:
        return self.template.format_user_prompt(
            artifacts_block=self._render_artifacts(space, incumbent),
            records_block=json.dumps(records, indent=2, ensure_ascii=False),
            n_proposals=self.proposals_per_iteration,
            artifact_names=space.names(),
        )

    def filter_valid(
        self,
        space: ArtifactSpace,
        incumbent: Candidate,
        proposals: list[Proposal],
    ) -> list[Proposal]:
        """Drop proposals that violate a guardrail or change nothing, logging each rejection.

        An `artifact_name` that is not an exact space name but starts with exactly one of them
        (an LLM echoing decoration like `sop (task-reviewer.sop.md)`) is repaired to that name
        rather than discarded. The result is truncated to `proposals_per_iteration`: a model that
        returns more edits than asked must not multiply the iteration's scoring cost.
        """
        valid = []
        for proposal in proposals:
            name = _repair_artifact_name(proposal.artifact_name, space.names())
            if name != proposal.artifact_name:
                logger.info(
                    "artifact=<%s>, repaired=<%s> | proposal artifact name normalized", proposal.artifact_name, name
                )
                proposal = proposal.model_copy(update={"artifact_name": name})
            try:
                space.validate(proposal.artifact_name, proposal.new_value)
            except ValueError as e:
                logger.warning("artifact=<%s>, error=<%s> | proposal rejected by guardrail", proposal.artifact_name, e)
                continue
            if incumbent.values.get(proposal.artifact_name) == proposal.new_value:
                logger.info("artifact=<%s> | proposal changes nothing, dropped", proposal.artifact_name)
                continue
            valid.append(proposal)
        if len(valid) > self.proposals_per_iteration:
            logger.info(
                "returned=<%d>, kept=<%d> | proposer returned more edits than asked, truncating",
                len(valid),
                self.proposals_per_iteration,
            )
            valid = valid[: self.proposals_per_iteration]
        return valid

    @staticmethod
    def _render_artifacts(space: ArtifactSpace, incumbent: Candidate) -> str:
        blocks = []
        for name in space.names():
            artifact = space[name]
            note = f"\nEdit instructions: {artifact.edit_instructions}" if artifact.edit_instructions else ""
            value = incumbent.values[name]
            # State the size budget: the cap is enforced in code, and a proposer that does not
            # know the number wastes proposals on over-cap rewrites.
            cap = artifact.size_cap if artifact.size_cap is not None else _default_size_cap(artifact.read())
            limit = f"\nSize limit: {cap} characters (the value below is {len(value)}; over-limit edits are rejected)"
            # An artifact containing a ``` run would close the fence early; use a longer fence.
            fence = "```"
            while fence in value:
                fence += "`"
            blocks.append(
                f"### artifact_name: {name}\nPath: {artifact.display_path()}{note}{limit}\n{fence}\n{value}\n{fence}"
            )
        return "\n\n".join(blocks)


def _repair_artifact_name(name: str, valid_names: list[str]) -> str:
    """Strip a parenthesized decoration from an artifact name, e.g. `sop (task-reviewer.sop.md)`.

    Only that exact shape is repaired; anything else stays as-is and fails the space guardrail,
    because guessing at other shapes risks applying an edit to the wrong artifact.
    """
    if name in valid_names:
        return name
    stripped = name.strip()
    matches = [valid for valid in valid_names if stripped == valid or stripped.startswith(f"{valid} (")]
    if len(matches) == 1:
        return matches[0]
    return name


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
