import pytest
from strands import tool

from strands_evals.optimization import Float, Int, OneOf, Space, TextValue, choice, component, tool_descriptions


class Window:
    def __init__(self, size: int = 40):
        self.size = size


class Summary:
    def __init__(self, ratio: float = 0.3, keep: int = 10):
        self.ratio = ratio
        self.keep = keep


@tool
def lookup(query: str) -> str:
    """Look up a record."""
    return query


def make_space(**extra):
    return Space(
        {
            "manager": choice(component(Window, size=Int(1, 100)), component(Summary, ratio=Float(0.1, 0.9))),
            "temperature": Float(0.0, 1.0, default=0.7),
            "effort": OneOf("low", "high"),
            "prompt": TextValue(default="You help.", size_cap=50),
            **extra,
        }
    )


BASELINE = {
    "manager": "Window",
    "manager.Window.size": 40,
    "temperature": 0.7,
    "effort": "low",
    "prompt": "You help.",
}


# --- flattening ---


def test_params_flatten_in_declaration_order_with_a_choice_before_its_sub_keys():
    space = make_space()
    assert [p.key for p in space.params()] == [
        "manager",
        "manager.Window.size",
        "manager.Summary.ratio",
        "temperature",
        "effort",
        "prompt",
    ]


def test_param_infos_carry_type_bounds_and_activation():
    space = make_space()
    manager = space.param("manager")
    assert (manager.kind, manager.default, manager.options) == ("choice", "Window", ("Window", "Summary"))
    ratio = space.param("manager.Summary.ratio")
    assert (ratio.kind, ratio.name, ratio.default, ratio.ge, ratio.le) == ("float", "manager", 0.3, 0.1, 0.9)
    assert ratio.active_when == ("manager", "Summary")
    assert space.param("effort").options == ("low", "high")
    assert space.param("prompt").size_cap == 50
    assert space.param("prompt").source is None


def test_text_without_size_cap_gets_the_derived_cap():
    space = Space({"prompt": TextValue(default="x" * 1000)})
    assert space.param("prompt").size_cap == 3000


def test_a_bare_component_is_a_choice_with_one_alternative():
    space = Space({"manager": component(Window, size=Int(1, 100))})
    assert space.param("manager").options == ("Window",)
    assert space.complete({}) == {"manager": "Window", "manager.Window.size": 40}


def test_tool_descriptions_flatten_to_one_text_key_per_tool():
    space = Space({"tools": tool_descriptions(lookup)})
    info = space.param("tools.lookup")
    assert (info.kind, info.name, info.default, info.size_cap) == ("text", "tools", "Look up a record.", 800)


def test_param_names_the_known_keys_on_a_typo():
    with pytest.raises(KeyError, match="temperature"):
        make_space().param("temprature")


# --- declaration errors ---


@pytest.mark.parametrize(
    "markers",
    [
        {},
        {"a.b": Int(1, 2, default=1)},
        {"a/b": Int(1, 2, default=1)},
        {"": Int(1, 2, default=1)},
        {"x": 3},
        {"x": Window()},
    ],
)
def test_bad_declarations_fail_at_construction(markers):
    with pytest.raises(TypeError):
        Space(markers)


@pytest.mark.parametrize("marker", [Int(1, 5), Float(0.0, 1.0), TextValue()])
def test_a_marker_without_a_default_fails_outside_tunable(marker):
    with pytest.raises(TypeError, match="no default"):
        Space({"x": marker})


def test_one_of_without_a_default_takes_the_first_option():
    assert Space({"x": OneOf("a", "b")}).param("x").default == "a"


def test_a_default_longer_than_its_size_cap_fails_at_declaration():
    with pytest.raises(TypeError, match="invalid default: prompt: 11 characters, over the size_cap of 10"):
        Space({"prompt": TextValue(default="x" * 11, size_cap=10)})


# --- activity and completion ---


def test_sub_keys_are_active_only_under_their_alternative():
    space = make_space()
    assert space.is_active("manager.Window.size", {})
    assert not space.is_active("manager.Summary.ratio", {})
    assert space.is_active("manager.Summary.ratio", {"manager": "Summary"})
    assert not space.is_active("manager.Window.size", {"manager": "Summary"})


def test_complete_fills_defaults_for_the_baseline():
    assert make_space().complete({}) == BASELINE


def test_complete_follows_a_changed_choice():
    completed = make_space().complete({**BASELINE, "manager": "Summary"})
    assert "manager.Window.size" not in completed
    assert completed["manager.Summary.ratio"] == 0.3


def test_complete_keeps_unknown_keys_so_validate_reports_them():
    completed = make_space().complete({"temprature": 0.2})
    assert completed["temprature"] == 0.2


# --- validation ---


def test_the_baseline_is_valid():
    make_space().validate(BASELINE)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"temperature": 1.5}, "outside"),
        ({"temperature": 1}, "expected float"),
        ({"manager.Window.size": 40.0}, "expected int"),
        ({"manager.Window.size": True}, "expected int"),
        ({"manager.Window.size": 0}, "outside"),
        ({"effort": "medium"}, "not one of"),
        ({"manager": "Missing"}, "not one of"),
        ({"prompt": "x" * 51}, "over the size_cap"),
        ({"prompt": 3}, "expected str"),
        ({"typo": 1}, "not keys of this space"),
        ({"manager.Summary.ratio": 0.5}, "inactive"),
    ],
)
def test_invalid_values_fail_validation(change, message):
    with pytest.raises(ValueError, match=message):
        make_space().validate({**BASELINE, **change})


def test_a_missing_active_key_fails_validation():
    values = {key: value for key, value in BASELINE.items() if key != "manager.Window.size"}
    with pytest.raises(ValueError, match=r"missing: \['manager.Window.size'\]"):
        make_space().validate(values)


def test_validation_reports_every_problem_at_once():
    with pytest.raises(ValueError) as raised:
        make_space().validate({**BASELINE, "temperature": 2.0, "effort": "medium", "typo": 1})
    message = str(raised.value)
    assert "temperature" in message and "effort" in message and "typo" in message


def test_one_of_validation_compares_types():
    space = Space({"x": OneOf(1, 2)})
    space.validate({"x": 1})
    with pytest.raises(ValueError):
        space.validate({"x": True})


# --- resolution ---


def test_resolve_builds_a_fresh_component_every_call():
    space = make_space()
    values = {**BASELINE, "manager": "Summary", "manager.Summary.ratio": 0.5}
    first = space.resolve("manager", values, None)
    second = space.resolve("manager", values, None)
    assert isinstance(first, Summary) and first.ratio == 0.5
    assert first is not second


def test_resolve_passes_a_components_fixed_arguments_as_is():
    # Documented in component(): fixed arguments are shared across builds, never copied.
    history = []
    space = Space({"manager": component(Summary, ratio=Float(0.1, 0.9), keep=history)})
    baseline = space.complete({})
    first = space.resolve("manager", baseline, None)
    second = space.resolve("manager", baseline, None)
    assert first is not second
    assert first.keep is history and second.keep is history


def test_resolve_passes_plain_values_through():
    space = make_space()
    assert space.resolve("temperature", {**BASELINE, "temperature": 0.2}, None) == 0.2
    assert space.resolve("effort", BASELINE, None) == "low"
    assert space.resolve("manager", BASELINE, None).size == 40


def test_resolve_returns_tool_copies_with_the_candidates_descriptions():
    space = Space({"tools": tool_descriptions(lookup)})
    built = space.resolve("tools", {"tools.lookup": "Find it."}, None)
    assert built[0].tool_spec["description"] == "Find it."
    assert lookup.tool_spec["description"] == "Look up a record."


def test_resolve_takes_declared_names_not_flat_keys():
    with pytest.raises(KeyError):
        make_space().resolve("manager.Window.size", BASELINE, None)
