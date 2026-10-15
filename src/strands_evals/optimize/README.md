# Optimize

Declare what may change about your agent, run the loop against the cases and evaluators you already have, and review what survived the gate. The deliverable is a reviewable diff for text changes plus the winning typed values for numeric knobs. A human applies them. This module does no weight training; the `RewardFunction` scoring contract is shared with trainers that do, which consume it from outside this package.

**Branch status**: this is the merged-API shape branch (`feat/optimize-unified`). The design is specified in the merged API document; the test suite is intentionally absent here and returns before release.

## The layers

| Layer | File | Job |
|---|---|---|
| Declaration | `markers.py`, `factory.py`, `artifacts.py` | Turn a construction function into a search space without changing how it is called |
| Space | `space.py`, `candidate.py` | The flat typed parameter vocabulary; candidates are typed mappings, hashed and diffed as derived views |
| Execution | `execution.py` | Run one candidate on one case: default rollout, trial statuses, usage metering |
| Scoring | `objectives.py` | Objectives and constraints over per-case metrics; constraints only ever reject |
| Search | `methods.py`, `search.py` | Blocks, methods (`Reflection`, `TPE`), and schedules; this layer proposes and never decides |
| Acceptance | `gate.py` | The mandatory statistical gate and the rollout-denominated budget |
| Loop | `optimizer.py`, `runs.py` | The iteration loop, run persistence, resume, results |

## Usage

The simplest run writes no rollout code and no search code:

```python
from strands_evals import Case, Experiment
from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.optimize import Optimizer, TextValue, tunable

experiment = Experiment(cases=cases, evaluators=[GoalSuccessRateEvaluator()])

@tunable(system_prompt=TextValue(default="You answer support tickets concisely."))
def create_agent(system_prompt):
    return Agent(model=MODEL, system_prompt=system_prompt)

optimizer = Optimizer(experiment, create_agent)
result = await optimizer.run()        # or optimizer.run_sync() in a script
print(result.summary())
print(result.best.diff())
```

Name an objective and bound the cost:

```python
from strands_evals.optimize import AtMost, Budget, Maximize

optimizer = Optimizer(
    experiment,
    create_agent,
    objectives=Maximize("Helpfulness"),
    constraints=[AtMost("tokens", 4000)],
    budget=Budget(max_rollouts=2000),
)
```

Everything explicit: an existing function wrapped without touching it, components to swap, a custom rollout, and a custom search:

```python
from strands_evals.optimize import Block, Float, Gate, Int, Reflection, Search, TPE, choice, component, tunable

factory = tunable(create_harness, inject={
    "instructions": TextValue(default=INSTRUCTIONS, size_cap=14_000),
    "conversation_manager": choice(
        component(SlidingWindowConversationManager, window_size=Int(4, 200, default=40)),
        component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)),
    ),
})

async def rollout(agent_factory, case):
    agent = agent_factory(context_manager=False)
    return {"output": str(await agent.invoke_async(case.input))}

optimizer = Optimizer(
    experiment, factory, rollout_fn=rollout,
    search=Search(blocks=[Block(TPE(), "conversation_manager"), Block(Reflection(), "instructions")]),
    gate=Gate(runs_per_candidate=3),
)
```

Every case needs a unique `name`: the optimizer keys its caches and its train/holdout split on case names.

## Candidates are typed

A candidate is a flat typed mapping with dotted keys; a key under a choice alternative exists only when that alternative is chosen:

```python
{
    "conversation_manager": "SummarizingConversationManager",
    "conversation_manager.SummarizingConversationManager.summary_ratio": 0.5,
    "instructions": "Review tickets. Cite the ticket id.",
}
```

The content hash (dedupe, caching, resume) and the diff are derived from this mapping. `result.best.values` is the mapping; `result.best.diff()` renders text parameters as unified diffs and scalar parameters as one-line before/after pairs.

## Search methods

Numbers and choices default to `TPE` (Optuna's tree-structured Parzen estimator); text defaults to `Reflection` (an LLM reads the training split's failures and the evaluators' written reasons). The default schedule is `Alternate`: one block per iteration, so an accepted candidate changed exactly one block's variables. Every method is told every gated outcome through `observe`, so a Bayesian method updates on gated truth rather than raw means.

`TPE` needs the `search` extra: `pip install "strands-evals[search]"`. A space with numeric or choice variables and no optuna fails at construction with that command; assign those variables to `Reflection()` in an explicit `Block` to stay dependency-free.

## Cost

Price a run before you start: `rollouts = (1 + iterations * proposals_per_iteration) * cases * runs_per_candidate`, and each rollout also pays one judge call per LLM evaluator. The budget is a hard stop; if you set nothing, a conservative default cap (500 rollouts) applies. Usage is metered per trial (`input_tokens`, `output_tokens`, `tokens`, `seconds`, `model_calls`), summed across every agent the bound factory builds in that trial, so `AtMost("tokens", ...)` works in single-agent and multi-agent rollouts alike.

## How acceptance works

Gate settings live on `Gate` (`runs_per_candidate`, `holdout_fraction`, `margin`, `confidence`, `min_coverage`, `challenger_holdout_only`). Cases split into a training split (reflection sees its failures) and a held-out split (acceptance is judged only here). Each candidate is scored `runs_per_candidate` times, per-case metrics reduce to one scalar through the objectives, constraints reject violators on the holdout, and a challenger replaces the incumbent only when the paired per-case improvement clears the margin and its bootstrap confidence interval stays above zero. Ties reject. Crashed trials are excluded and counted (`TrialStatus`), never scored 0.0, and a challenger missing too many holdout cases is rejected on coverage. The gate always runs and has no bypass; verifiable rewards configure it cheaper with `Gate(runs_per_candidate=1)`.

Before you merge an accepted diff, run one fresh evaluation on cases the optimizer never saw. The gate defends against judge noise, but repeated tests against the same held-out split can still overfit it slowly.

## Reading the result

- `result.summary()`, `result.best.diff()`, `result.best.values`
- `result.to_rows(grain)` / `result.to_json(grain, path=...)` with `grain` of `"iteration"`, `"proposal"`, or `"case"`; proposal rows carry the gate verdict, constraint violations, and the candidate's full typed values
- `result.reports(candidate_id)` for the stored `EvaluationReport`s
- Runs persist under `.strands-evals/optimize/runs/<run_id>/` (append-only). `Optimizer.resume(run_dir, experiment=..., factory=...)` continues a crashed or budget-stopped run with every already-scored candidate a cache hit.

## Not implemented yet (lands after v1, same record format)

`Chebyshev` / `Lexicographic` / `Pareto` combiners, `Derived` metrics, the `Nested` schedule, `SkillCuration`, and the effects model all raise `NotImplementedError` with a pointer to what to use meanwhile.
