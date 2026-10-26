import json

import pytest
from pydantic import ValidationError

from strands_evals.types.optimization import MergePlan, Observation, Patch, SelectionStep


def _canonical(values):
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize(
    "value",
    [1, 1.0, 0.5, True, False, "1", None],
    ids=["int", "integral-float", "float", "true", "false", "str", "none"],
)
def test_patch_values_keep_their_exact_type_through_json(value):
    patch = Patch(updates={"key": value}, rationale="")

    restored = Patch.model_validate_json(patch.model_dump_json())

    assert type(restored.updates["key"]) is type(value)
    assert restored.updates["key"] == value


def test_canonical_json_of_values_is_unchanged_by_a_round_trip():
    updates = {"temperature": 0.5, "max_turns": 3, "use_cache": True, "prompt": "Be brief.", "budget": 2.0}
    patch = Patch(updates=updates, rationale="")

    restored = Patch.model_validate_json(patch.model_dump_json())

    assert _canonical(restored.updates) == _canonical(updates)


def test_observation_round_trips_through_json():
    observation = Observation(
        patch=Patch(updates={"max_turns": 3}, rationale="fewer turns", state={"trial": 4}),
        scalar=0.75,
        outcome="admitted",
        reason="not this iteration's best",
    )

    restored = Observation.model_validate_json(observation.model_dump_json())

    assert restored == observation


def test_observation_rejects_an_unknown_outcome():
    with pytest.raises(ValidationError):
        Observation(patch=Patch(updates={}, rationale=""), scalar=None, outcome="won", reason="")


def test_selection_step_with_merge_round_trips_through_json():
    step = SelectionStep(
        parent_id="aaa",
        merge=MergePlan(
            first="aaa",
            second="bbb",
            ancestor="ccc",
            values={"system_prompt": "Be brief.", "temperature": 0.2},
            from_first=("system_prompt",),
            from_second=("temperature",),
        ),
    )

    restored = SelectionStep.model_validate_json(step.model_dump_json())

    assert restored == step
    assert isinstance(restored.merge.from_first, tuple)


def test_records_are_frozen():
    patch = Patch(updates={"max_turns": 3}, rationale="")

    with pytest.raises(ValidationError):
        patch.rationale = "changed"
