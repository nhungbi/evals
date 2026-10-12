"""ContrastiveReflectionProposer: an alternative proposer graduated from harness-optimizer.

Graduated from strands-labs/harness-optimizer's ContrastiveReflectionOptimizer (Apache-2.0),
re-homed behind the optimizer's `proposer=` plug point and reduced to the loop's needs: no
shell-tool agent, no temp trace files, no Jinja2. It reflects on the lowest-scoring cases and
appends actionable WHEN/DO insights to the responsible artifact, preserving the original text.
Pick it over the default reason-driven proposer when the eval suite's written reasons are thin
or when the agent's failures live in trajectories rather than in any single evaluator reason.
"""

from .prompt_templates.proposer import contrastive_v0
from .proposer import Proposer


class ContrastiveReflectionProposer(Proposer):
    """Reflect contrastively on failing cases and append learned behaviours to one artifact.

    Same call path as the base `Proposer` with a different prompt, and it skips the LLM call
    when there are no reflection records, because contrastive reflection has nothing to compare
    without failures.
    """

    template = contrastive_v0
    require_records = True
