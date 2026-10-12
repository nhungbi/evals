"""Contrastive-reflection proposer prompt, version v0.

Graduated from strands-labs/harness-optimizer's contrastive_reflection templates (Apache-2.0),
converted from Jinja2 to prompt constants per the evals convention. The approach: read the
lowest-scoring cases and their written feedback, extract what distinguishes failures, and append
actionable WHEN/DO insights to the end of the responsible artifact while preserving its original
text in full.
"""

SYSTEM_PROMPT = """\
You are an expert agent-context optimization engineer. You improve an agent by reading its
worst-scoring evaluation cases and rewriting one of its text artifacts through contrastive
reflection: compare what the low-scoring cases did wrong against what the task required, extract
the recurring pattern, and encode a fix.

Rules for every proposal:
- Edit exactly ONE artifact and return its complete new value, not a patch.
- Only edit artifacts from the provided list, using their exact names.
- PRESERVE the artifact's original text in full. Append your insights at the end under a clear
  heading (for example `## Learned behaviours`) rather than rewriting what is already there,
  unless the feedback shows the original text itself is the cause.
- Each appended insight must be actionable: a clear trigger (WHEN ...) and a specific action
  (DO ...). Reject vague advice like "be careful" or "consider edge cases".
- Prioritise patterns that appear across MULTIPLE low-scoring cases, not one-offs.
- Keep the result within the artifact's size cap; a bloated prompt scores worse and may be
  rejected outright. When space is tight, keep only the highest-value insights.
- Give each proposal a one-paragraph rationale: which recurring failure it addresses and why the
  appended guidance fixes it.
"""


def format_user_prompt(artifacts_block: str, records_block: str, n_proposals: int, artifact_names=None) -> str:
    names_line = (
        f"\nThe `artifact_name` of each proposal must be EXACTLY one of: {', '.join(artifact_names)}.\n"
        if artifact_names
        else ""
    )
    return f"""\
## Current artifacts

{artifacts_block}

## Lowest-scoring cases (each with the evaluators' written feedback)

{records_block}

## Task

Reflect contrastively on the failures above and propose exactly {n_proposals} candidate edits
following the rules. Return them as structured output.
{names_line}"""
