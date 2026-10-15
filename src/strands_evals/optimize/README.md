# Optimize

Improve an agent by rewriting its text artifacts — instructions, `SKILL.md` files, tool descriptions, config fields — against the cases and evaluators you already have. The loop builds the agent with a proposed edit, runs your `Experiment` against it, and has an LLM read the failures to propose the next edit. Every edit that is the best on at least one case stays on a Pareto frontier the search keeps building from, and an edit only becomes the result (the champion) when it beats the current champion on held-out cases across repeated runs. The output is a diff a human reviews and commits. This module does no weight training. The `RewardFunction` scoring contract is shared with trainers that do, which consume it from outside this package.

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

Every case needs a unique `name`: the optimizer keys its caches, its case split, and its frontier on case names.

## Reading the result

- `result.summary()` — baseline vs champion score on the held-out split, accept/reject, admitted/dominated, and merge counts, spend.
- `result.best.diff()` — the deliverable: the champion as a unified diff against your baseline artifacts.
- `result.frontier` — the final frontier: each member's candidate id and the selection cases it leads.
- `result.to_rows(grain)` — the run flattened into plot-ready rows (below).
- `result.to_json(grain, path="run.json")` — the same rows serialized, written to `path` when given.
- `result.reports(candidate_id)` — the stored `EvaluationReport`s for one candidate, the evals SDK's own report objects.
- `result.iterations` / `result.lineage` — the structured records and the raw append-only event log, when the rows are not enough.

### Rows for graphs

`grain="iteration"` gives one row per loop iteration — the convergence view:

| column | meaning | typical graph |
|---|---|---|
| `iteration` | 1-based loop index | x-axis |
| `incumbent_train_score` / `incumbent_holdout_score` | the champion's mean scores on each split at the start of the iteration | convergence curve; plot both to watch the train/holdout gap (overfitting) |
| `rollouts_cumulative` | total rollouts spent so far | cost-vs-quality curve (`holdout` against this instead of iteration) |
| `rollouts_spent` | rollouts this iteration | spend per iteration bars |
| `proposals_count`, `accepted_candidate_id` | how many edits were tried, and which (if any) became champion | accept markers on the convergence curve |
| `incumbent_id`, `incumbent_values` | the champion and its full artifact values | annotations / joining rows to candidates |
| `step`, `parent_id`, `frontier_size` | `propose` or `merge`; the frontier member this iteration's edits came from (a merge's first parent); and how many members the frontier had | exploration view: how often the search left the champion's line, and where merges happened |

`grain="proposal"` gives one row per proposed edit — the frontier's and the gate's view:

| column | meaning | typical graph |
|---|---|---|
| `train_score`, `selection_score`, `holdout_score` | the candidate's split means (`train_score` is null under `challenger_holdout_only`) | scatter per iteration, colored by `status` |
| `cases_led` | the selection cases the candidate led when it was admitted (empty when dominated) | specialist view: which cases each edit uniquely helps |
| `mean_diff`, `ci_low`, `ci_high`, `margin` | the paired-bootstrap verdict against the champion; null unless the candidate challenged it | the gate plot: `mean_diff` with CI error bars per challenge; accepted = CI entirely above the margin |
| `status` | `accepted` (admitted and became champion) / `admitted` (joined the frontier; lost or never made a champion challenge) / `dominated` (scored, not on the frontier) / `unscored` (budget) | color / facet |
| `artifact_name` | which artifact the edit touched | per-artifact accept-rate bars (where the leverage is) |
| `n_cases`, `error_rollouts` | held-out cases compared, and rollouts that crashed | gate-health panel |
| `candidate_id`, `parent_id`, `parent_ids`, `rationale`, `values`, `diff`, `verdict_reason` | lineage (`parent_ids` has two entries for a merge) and the edit itself | tooltips / drill-down |

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
colors = scored.status.eq("accepted").map({True: "tab:green", False: "tab:red"})
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

Resume validates that the supplied cases and artifact names match the recorded run and refuses otherwise; changed cases need a fresh run, because the recorded case split would be stale. The pool is rebuilt from the lineage's admissions, and each iteration's parent is sampled from a seed derived from the run seed and the iteration number, so a resumed run picks the same parents an uninterrupted run would have.

The run directory keeps every candidate's workspace and scores for the life of the run. Budget the disk accordingly, and delete run directories you no longer need.

## How the search works

Cases split three ways (settings on `Gate`): a training split (the proposer reads its failures), a selection split (`selection_fraction`, the frontier ranks candidates here), and a held-out split (`holdout_fraction`, only the champion gate reads it). No split both chooses and judges. With too few cases for a separate selection split, the training split doubles as it and a warning is logged.

Each iteration:

1. **Selection.** When two frontier members changed different knobs since their latest common ancestor, and that pair has not been merged yet, the iteration is a **merge step**: the two are combined knob by knob (each knob from whichever side changed it) with no proposer call, and the merged candidate goes through steps 3–5 like any other. `Gate(max_merges=5)` caps merge steps per run; `0` disables them. A pair that changed the same knob differently is never merged. Otherwise, the frontier is every candidate in the pool that is the best (within `frontier_eps`) on at least one selection case and is not dominated: no other candidate leads a strict superset of its cases, and on identical lead sets the earlier candidate wins. One frontier member is sampled as the parent, weighted by how many cases it leads, so a specialist that uniquely solves some cases keeps being explored even when its average is mediocre.
2. **Proposal.** The proposer reads the parent's worst training cases and returns edits of the parent.
3. **Scoring.** Each edit is scored `runs_per_candidate` times and per-case scores are averaged. Crashed rollouts are excluded and counted, not scored 0.0.
4. **Admission (loose).** An edit joins the pool when it would land on the frontier and has scores for at least `min_coverage` of its parent's selection cases. A no-op edit ties its parent everywhere and is dominated.
5. **Champion gate (strict).** The strongest admitted edit challenges the champion only when its selection mean is higher. It becomes champion only when the paired per-case improvement on the held-out split clears your `margin` and its bootstrap confidence interval stays above zero. Ties reject, and a challenger missing scores for more than `min_coverage` of the champion's held-out cases is rejected outright, so an edit cannot win by crashing the hard cases. Below 5 held-out cases the optimizer logs a warning: small holdouts measurably inflate false-positive accepts.

With many cases and LLM judges, challengers can be scored on the selection and held-out splits only (`Gate(challenger_holdout_only=True)`); a candidate is topped up on the training split the first time it is sampled as a parent or becomes champion, with its other scores served from cache.

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
