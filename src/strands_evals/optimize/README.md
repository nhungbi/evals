# Optimize

Improve an agent by rewriting its text artifacts — instructions, `SKILL.md` files, tool descriptions, config fields — against the cases and evaluators you already have. The loop builds the agent with a proposed edit, runs your `Experiment` against it, and has an LLM read the failures to propose the next edit. An edit is kept only when it beats the current version on held-out cases across repeated runs. The output is a diff a human reviews and commits. This module does no weight training. The `RewardFunction` scoring contract is shared with trainers that do, which consume it from outside this package.

**Stability**: public API. Breaking changes go through a deprecation cycle with a `DeprecationWarning` for at least one minor version.

## Cost

Price a run before you start. The loop's spend in rollouts (one rollout = one case executed once):

```
rollouts = (1 + iterations * proposals_per_iteration) * cases * runs_per_candidate
```

With defaults (`Gate(runs_per_candidate=3)`, `proposals_per_iteration=3`) and 20 cases, each iteration costs 180 rollouts and the baseline costs 60. A `Budget(max_rollouts=2000)` therefore buys about 10 iterations. On top of every rollout: one judge call per LLM evaluator per case, plus one proposer call per iteration. Judge calls usually dominate the bill — `judge_calls = rollouts * llm_evaluators`.

The budget is a hard stop. If you set nothing, a conservative default cap (500 rollouts) applies.

## Usage

```python
from strands_evals import Case, Experiment
from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.optimize import (
    ArtifactSpace, Budget, FileArtifact, Gate, JsonFieldArtifact, Optimizer,
)

experiment = Experiment(cases=cases, evaluators=[GoalSuccessRateEvaluator()])

space = ArtifactSpace(
    root="my-agent",  # copied into each candidate workspace
    instructions=JsonFieldArtifact("agent.json", "/instructions"),
    triage_skill=FileArtifact(".agent/skills/triage/SKILL.md"),
)

def build_task(candidate):
    workspace = candidate.materialize()  # applies the candidate's values to a workspace copy

    def task(case):
        agent = build_my_agent(workspace)  # however you build your agent from config
        return {"output": str(agent(case.input))}

    return task

optimizer = Optimizer(experiment, build_task, space, budget=Budget(max_rollouts=2000))
result = await optimizer.run()

print(result.summary())
print(result.best.diff())
```

When you don't need a workspace `root`, a plain mapping works in place of an `ArtifactSpace`: `Optimizer(experiment, build_task, {"instructions": TextArtifact("...")})`.

Every case needs a unique `name`: the optimizer keys its caches and its train/holdout split on case names.

## Reading the result

- `result.summary()` — baseline vs best score on the held-out split, accept/reject/skip counts, spend.
- `result.best.diff()` — the deliverable: a unified diff against your baseline artifacts.
- `result.iterations` — per-iteration records: each proposal's rationale, diff, scores, verdict, and status (`accepted`, `rejected`, `skipped` when an earlier candidate was accepted the same iteration, or `unscored` when the budget ran out).
- `result.lineage` — the raw append-only event log (propose / score / accept / reject / skip / resume).

The module never edits your files. Apply the diff yourself after review, and commit it with the run id in the message.

## Required closing step

Before you merge an accepted diff, run one fresh evaluation on cases the optimizer never saw. The acceptance gate defends against judge noise, but repeated tests against the same held-out split can still overfit it slowly. A final check on unseen cases is the honest number.

## Resuming a run

Runs persist under `.strands-evals/optimize/runs/<run_id>/` (append-only: config, per-candidate manifests, workspaces, scores, `lineage.jsonl`, and the `best` pointer). A crashed or budget-stopped run continues with every already-scored candidate a cache hit:

```python
optimizer = Optimizer.resume(
    ".strands-evals/optimize/runs/9f3ac1",
    experiment=experiment, build_task=build_task, artifacts=space,
)
result = await optimizer.run()
```

Resume validates that the supplied cases and artifact names match the recorded run and refuses otherwise; changed cases need a fresh run, because the recorded train/holdout split would be stale.

The run directory keeps every candidate's workspace and scores for the life of the run. Budget the disk accordingly, and delete run directories you no longer need.

## How acceptance works

Gate settings live on `Gate` (`runs_per_candidate`, `holdout_fraction`, `margin`, `confidence`, `min_coverage`). Cases split into a training split (the proposer sees its failures) and a held-out split (acceptance is judged only here). Each candidate is scored `runs_per_candidate` times, per-case scores are averaged, and a challenger replaces the incumbent only when the paired per-case improvement on the held-out split clears your `margin` and its bootstrap confidence interval stays above zero. Ties reject. Crashed rollouts are excluded from scores and counted, not scored 0.0, and a challenger missing scores for more than `min_coverage` of the incumbent's held-out cases is rejected outright, so an edit cannot win by crashing the hard cases. Below 5 held-out cases the optimizer logs a warning: small holdouts measurably inflate false-positive accepts.

With many cases and LLM judges, challengers can be scored on the holdout only (`Gate(challenger_holdout_only=True)`); an accepted challenger is topped up on the training split afterward, with its holdout scores served from cache.

## Runtime surfaces, alternative proposer, and reward functions

These capabilities were graduated from strands-labs/harness-optimizer (Apache-2.0) and are native here, with no dependency on that package.

- **Formulas** (`SystemPromptFormula`, `ToolDescriptionFormula`, `ContextExpansionFormula`) are runtime hooks whose tunable text the optimizer can edit. Wrap one with `FormulaArtifact(formula, param_key)` to make its parameter an optimizable artifact, and mix it freely with `FileArtifact` / `JsonFieldArtifact` in the same space.
- **`ContrastiveReflectionProposer`** is an alternative to the default reason-driven proposer. It reflects on the lowest-scoring cases and appends actionable WHEN/DO insights to the responsible artifact. Pass it as `proposer=` when the eval suite's written reasons are thin, or when failures live in trajectories rather than in any single reason.
- **`RewardFunction` + `RewardFunctionEvaluator`** let a bare scalar reward (tests pass, exact match) score through the same `Experiment` and acceptance gate as any judge. Use `Gate(runs_per_candidate=1)` for a verifiable reward; keep the default (3) for judge scoring.

### Wiring a Formula into your agent

The optimizer treats a `FormulaArtifact` like any other artifact; it does not attach the Formula to your agent. Your `build_task` pushes the candidate's value into the Formula and reads the effective text when building the agent:

```python
from strands_evals.optimize import (
    ArtifactSpace, ContrastiveReflectionProposer, FormulaArtifact, ToolDescriptionFormula,
)

tools = ToolDescriptionFormula(base={"search": "Search tickets."})
space = ArtifactSpace(
    instructions=JsonFieldArtifact("agent.json", "/instructions"),
    search_desc=FormulaArtifact(tools, "tool_descriptions", size_cap=800),
)

def build_task(candidate):
    space["search_desc"].apply_to_formula(candidate["search_desc"])
    descriptions = tools.effective()  # base with the candidate's overrides applied

    def task(case):
        agent = build_my_agent(tool_descriptions=descriptions)
        return {"output": str(agent(case.input))}

    return task

optimizer = Optimizer(experiment, build_task, space, proposer=ContrastiveReflectionProposer())
```
