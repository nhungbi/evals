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
- `result.to_rows(grain)` — the run flattened into plot-ready rows (below).
- `result.to_json(grain, path="run.json")` — the same rows serialized, written to `path` when given.
- `result.reports(candidate_id)` — the stored `EvaluationReport`s for one candidate, the evals SDK's own report objects.
- `result.iterations` / `result.lineage` — the structured records and the raw append-only event log, when the rows are not enough.

### Rows for graphs

`grain="iteration"` gives one row per loop iteration — the convergence view:

| column | meaning | typical graph |
|---|---|---|
| `iteration` | 1-based loop index | x-axis |
| `incumbent_train_score` / `incumbent_holdout_score` | the current best candidate's mean scores on each split | convergence curve; plot both to watch the train/holdout gap (overfitting) |
| `rollouts_cumulative` | total rollouts spent so far | cost-vs-quality curve (`holdout` against this instead of iteration) |
| `rollouts_spent` | rollouts this iteration | spend per iteration bars |
| `proposals_count`, `accepted_candidate_id` | how many edits were tried, and which (if any) won | accept markers on the convergence curve |
| `incumbent_id`, `incumbent_values` | the winning candidate and its full artifact values | annotations / joining rows to candidates |

`grain="proposal"` gives one row per proposed edit — the gate's view:

| column | meaning | typical graph |
|---|---|---|
| `holdout_score`, `train_score` | the challenger's split means (`train_score` is null under `challenger_holdout_only`) | scatter per iteration, colored by `status` |
| `mean_diff`, `ci_low`, `ci_high`, `margin` | the paired-bootstrap verdict against the incumbent | the gate plot: `mean_diff` with CI error bars per proposal; accepted = CI entirely above the margin |
| `status` | `accepted` / `rejected` / `skipped` (never compared: an earlier edit won the iteration) / `unscored` (budget) | color / facet |
| `artifact_name` | which artifact the edit touched | per-artifact accept-rate bars (where the leverage is) |
| `n_cases`, `error_rollouts` | held-out cases compared, and rollouts that crashed | gate-health panel |
| `candidate_id`, `parent_id`, `rationale`, `values`, `diff`, `verdict_reason` | lineage and the edit itself | tooltips / drill-down |

```python
import pandas as pd
import matplotlib.pyplot as plt

iters = pd.DataFrame(result.to_rows("iteration"))
props = pd.DataFrame(result.to_rows("proposal"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# 1. convergence: holdout score vs cumulative rollouts, accepts marked
ax1.plot(iters.rollouts_cumulative, iters.incumbent_holdout_score, marker="o", label="holdout")
ax1.plot(iters.rollouts_cumulative, iters.incumbent_train_score, marker=".", alpha=0.5, label="train")
accepted = iters[iters.accepted_candidate_id.notna()]
ax1.scatter(accepted.rollouts_cumulative, accepted.incumbent_holdout_score, marker="*", s=160, zorder=3)
ax1.set_xlabel("rollouts spent"); ax1.set_ylabel("score"); ax1.legend(); ax1.set_title("convergence")

# 2. the gate: every proposal's paired improvement with its confidence interval
scored = props[props.mean_diff.notna()].reset_index()
colors = scored.status.map({"accepted": "tab:green", "rejected": "tab:red"})
ax2.errorbar(scored.index, scored.mean_diff,
             yerr=[scored.mean_diff - scored.ci_low, scored.ci_high - scored.mean_diff],
             fmt="none", ecolor=colors)
ax2.scatter(scored.index, scored.mean_diff, c=colors)
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_xlabel("proposal"); ax2.set_ylabel("paired improvement"); ax2.set_title("acceptance gate")
```

`grain="case"` gives one row per stored report row: exactly what each evaluator returned for each case in each scoring run, nothing averaged. Columns are the report's own fields (`case_name`, `evaluator`, `score`, `test_pass`, `applicable`, `reason`) joined with `iteration`, `candidate_id`, `status`, and `report` (the scoring-run file, e.g. `run0` or `run1.holdout`). Aggregate it however your analysis needs:

```python
cases = pd.DataFrame(result.to_rows("case"))
incumbents = cases.merge(iters[["iteration", "incumbent_id"]],
                         left_on="candidate_id", right_on="incumbent_id")
per_evaluator = (incumbents[incumbents.applicable]
                 .groupby(["iteration", "evaluator"]).score.mean().unstack())
per_evaluator.plot(subplots=True, layout=(2, 4), figsize=(12, 5), ylim=(0, 1), legend=False,
                   title=per_evaluator.columns.tolist())
```

The source objects behind those rows are also available directly: `result.reports(candidate_id)` returns the exact `EvaluationReport`s stored while scoring that candidate (keyed `run0`, `run1.holdout`, ...), so everything the evals SDK puts in a report — including `detailed_results` and `display()` — is one call away.

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
