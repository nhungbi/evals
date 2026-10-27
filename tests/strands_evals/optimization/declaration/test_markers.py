import pytest
from strands import tool

from strands_evals.optimization import Float, Int, OneOf, TextValue, choice, component, tool_descriptions
from strands_evals.optimization.declaration.markers import _default_size_cap


class Manager:
    def __init__(self, window_size: int = 40, ratio: float = 0.5, factory: str = "x"):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.ratio = ratio
        self.factory = factory


class OtherManager:
    def __init__(self, depth: int = 2):
        self.depth = depth


def make_manager(window_size: int = 8) -> Manager:
    return Manager(window_size=window_size)


# --- Int / Float ---


def test_int_keeps_bounds_and_default():
    marker = Int(1, 10, default=3, log=True)
    assert (marker.ge, marker.le, marker.default, marker.log) == (1, 10, 3, True)


def test_int_default_is_optional():
    assert Int(1, 10).default is None


@pytest.mark.parametrize(
    "make",
    [
        lambda: Int(1, 10, default=11),
        lambda: Int(1, 10, default=0),
        lambda: Int(10, 1),
        lambda: Int(0, 10, log=True),
        lambda: Int(1.5, 10),  # type: ignore[arg-type]
        lambda: Int(1, 10, default=True),
        lambda: Int(1, 10, default=2.0),  # type: ignore[arg-type]
    ],
)
def test_int_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


def test_float_stores_floats_so_an_integer_default_hashes_like_a_proposed_float():
    marker = Float(0, 2, default=1)
    assert marker.default == 1.0 and type(marker.default) is float
    assert type(marker.ge) is float and type(marker.le) is float


@pytest.mark.parametrize(
    "make",
    [
        lambda: Float(0.0, 1.0, default=1.5),
        lambda: Float(1.0, 0.0),
        lambda: Float(-1.0, 1.0, log=True),
        lambda: Float(0.0, 1.0, default=False),
        lambda: Float("0", 1.0),  # type: ignore[arg-type]
        lambda: Float(float("nan"), 1.0),  # NaN compares False, so ge > le alone wouldn't catch it
        lambda: Float(0.0, float("inf")),
        lambda: Float(0.0, 1.0, default=float("nan")),
    ],
)
def test_float_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


# --- OneOf ---


def test_one_of_default_is_unset_until_filled():
    marker = OneOf("low", "high")
    assert marker.options == ("low", "high")
    assert marker._with_default("high").default == "high"


def test_one_of_accepts_none_as_an_option_and_default():
    assert OneOf(None, "x", default=None).default is None


@pytest.mark.parametrize(
    "make",
    [
        lambda: OneOf(),
        lambda: OneOf("a", "a"),
        lambda: OneOf("a", ["b"]),  # type: ignore[arg-type]
        lambda: OneOf("a", "b", default="c"),
        lambda: OneOf(1, 2, default=True),  # True == 1, but it is not the option 1
    ],
)
def test_one_of_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


def test_one_of_tells_true_from_one():
    marker = OneOf(1, True)
    assert marker.options == (1, True)


# --- TextValue ---


def test_text_value_keeps_its_settings():
    marker = TextValue(default="hi", size_cap=10, edit_instructions="be brief")
    assert (marker.default, marker.size_cap, marker.edit_instructions) == ("hi", 10, "be brief")


@pytest.mark.parametrize(
    "make",
    [
        lambda: TextValue(default=3),  # type: ignore[arg-type]
        lambda: TextValue(default="x", size_cap=0),
        lambda: TextValue(default="x", size_cap=True),
    ],
)
def test_text_value_rejects_bad_declarations(make):
    with pytest.raises(TypeError):
        make()


def test_default_size_cap_is_at_least_2000_and_three_times_the_baseline():
    assert _default_size_cap("x" * 10) == 2000
    assert _default_size_cap("x" * 1000) == 3000


# --- component ---


def test_component_splits_tunable_and_fixed_kwargs():
    declared = component(Manager, window_size=Int(1, 100, default=20), factory="kept")
    assert declared.label == "Manager"
    assert set(declared.tunable) == {"window_size"}
    assert declared.fixed == {"factory": "kept"}


def test_component_takes_a_missing_default_from_the_constructor():
    declared = component(Manager, window_size=Int(1, 100), ratio=Float(0.0, 1.0))
    assert declared.tunable["window_size"].default == 40
    assert declared.tunable["ratio"].default == 0.5


def test_component_accepts_a_plain_function():
    declared = component(make_manager, window_size=Int(1, 100))
    assert declared.label == "make_manager"
    assert declared.tunable["window_size"].default == 8


def test_component_builds_fresh_instances():
    declared = component(Manager, window_size=Int(1, 100, default=20))
    first, second = declared.build({"window_size": 5}), declared.build({"window_size": 5})
    assert first is not second
    assert first.window_size == 5


def test_component_whose_constructor_rejects_its_defaults_fails_at_declaration():
    with pytest.raises(TypeError, match="rejects its own defaults") as raised:
        component(Manager, window_size=Int(0, 100, default=0))
    assert isinstance(raised.value.__cause__, ValueError)


def test_component_with_an_unknown_kwarg_fails_at_declaration():
    with pytest.raises(TypeError, match="rejects its own defaults"):
        component(Manager, no_such_arg=1)


def test_component_without_a_default_anywhere_fails():
    with pytest.raises(TypeError, match="has no default"):
        component(Manager, missing=Int(1, 5))


def test_component_constructor_default_outside_the_bounds_fails():
    with pytest.raises(TypeError, match="doesn't fit"):
        component(Manager, window_size=Int(1, 10))


@tool
def lookup(query: str) -> str:
    """Look up a record."""
    return query


@pytest.mark.parametrize(
    "marker",
    [
        OneOf("a", "b"),
        TextValue(default="x"),
        tool_descriptions(lookup),
        choice(component(OtherManager)),
    ],
)
def test_component_rejects_markers_other_than_int_and_float(marker):
    with pytest.raises(TypeError, match="only Int and Float"):
        component(Manager, factory=marker)


def test_component_rejects_a_non_callable():
    with pytest.raises(TypeError):
        component("Manager")  # type: ignore[arg-type]


# --- choice ---


def test_choice_labels_default_to_class_names_in_order():
    declared = choice(component(Manager), component(OtherManager))
    assert list(declared.alternatives) == ["Manager", "OtherManager"]


def test_choice_with_the_same_class_twice_needs_labels():
    with pytest.raises(TypeError, match="labels"):
        choice(component(Manager, ratio=0.1), component(Manager, ratio=0.9))
    declared = choice(component(Manager, ratio=0.1), component(Manager, ratio=0.9), labels=["low", "high"])
    assert list(declared.alternatives) == ["low", "high"]


@pytest.mark.parametrize("labels", [["a"], ["a", "a"], ["a", "b.c"], ["a", "b/c"], ["a", ""]])
def test_choice_rejects_bad_labels(labels):
    with pytest.raises(TypeError):
        choice(component(Manager), component(OtherManager), labels=labels)


def test_choice_rejects_non_components():
    with pytest.raises(TypeError):
        choice()
    with pytest.raises(TypeError):
        choice(Manager)  # type: ignore[arg-type]
