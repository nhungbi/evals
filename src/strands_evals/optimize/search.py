"""The search layer: which block of variables moves next, proposed by which method.

A `Block` groups variables under one `Method`. A `Schedule` says which blocks move in each
iteration while the rest hold at the parent's values. When no search is given, every variable
lands in a default block for its type (text to `Reflection`, numbers and choices to `TPE`) and
the schedule is `Alternate`, one block per iteration, so the gate's attribution of an accepted
edit stays clean. Most users never touch this layer.
"""

import logging
from typing import Sequence

from .methods import TPE, Method, Reflection
from .space import TEXT_KINDS, Space

logger = logging.getLogger(__name__)

_OPTUNA_INIT_HINT = (
    "this space has numeric or choice variables, and their default search method (TPE) needs optuna. "
    'Install the search extra: pip install "strands-evals[search]". To stay dependency-free, pass '
    "search=Search(blocks=[Block(Reflection(), ...)]) assigning those variables explicitly."
)


class Block:
    """One method and the variables it proposes.

    Variables are named by parameter key, or by a declared top-level name, which stands for every
    parameter under it (`Block(TPE(), "conversation_manager")` covers the choice and each
    alternative's tunable arguments).
    """

    def __init__(self, method: Method, *variables: str, name: str | None = None):
        if not variables:
            raise ValueError("a Block needs at least one variable")
        self.method = method
        self.variables = list(variables)
        self.name = name or f"{type(method).__name__.lower()}({', '.join(variables)})"

    def resolve_keys(self, space: Space) -> list[str]:
        """Expand declared names into parameter keys and check every key exists in the space."""
        keys: list[str] = []
        for variable in self.variables:
            expanded = [
                info.key for info in space.params() if info.key == variable or space.entry_name(info.key) == variable
            ]
            if not expanded:
                raise ValueError(f"block {self.name!r}: {variable!r} is not in the space ({space.names()})")
            keys.extend(key for key in expanded if key not in keys)
        unsupported = [key for key in keys if self.method.kinds and space.param(key).kind not in self.method.kinds]
        if unsupported:
            raise ValueError(
                f"block {self.name!r}: {type(self.method).__name__} does not propose "
                f"{[f'{key} ({space.param(key).kind})' for key in unsupported]}"
            )
        return keys


class Schedule:
    """Which blocks move in each iteration. Subclasses implement `blocks_for`."""

    def blocks_for(self, iteration: int, blocks: list[Block]) -> list[Block]:
        """The blocks allowed to propose in `iteration` (1-based)."""
        raise NotImplementedError


class Alternate(Schedule):
    """One block at a time, in turn (block-coordinate search). The default schedule.

    Keeping one block per iteration keeps the gate's attribution clean: an accepted candidate
    changed exactly one block's variables.
    """

    def blocks_for(self, iteration: int, blocks: list[Block]) -> list[Block]:
        return [blocks[(iteration - 1) % len(blocks)]]


class Joint(Schedule):
    """Every block proposes in every iteration. Faster per iteration, muddier attribution."""

    def blocks_for(self, iteration: int, blocks: list[Block]) -> list[Block]:
        return list(blocks)


class Nested(Schedule):
    """Rounds: an outer structural choice, an inner search under it. Not implemented yet.

    The intended semantics: each round proposes the outer block, the inner blocks search under it
    for `inner_iterations`, and the outer method is judged by the round's best, so a structural
    choice is compared at its tuned best rather than at its untuned first draw.
    """

    def __init__(self, *, outer: str, inner: Schedule | None = None, inner_iterations: int = 4):
        raise NotImplementedError("Nested schedules land after v1; use Alternate (default) or Joint")


class Search:
    """How the run searches: its blocks and its schedule.

    When omitted on the optimizer, `default_search` builds one block per variable type and
    alternates between them.
    """

    def __init__(self, blocks: Sequence[Block] = (), schedule: Schedule | None = None):
        wrong = [block for block in blocks if not isinstance(block, Block)]
        if wrong:
            raise TypeError(f"Search(blocks=...) takes Block objects, not {wrong}")
        self.blocks = list(blocks)
        self.schedule = schedule or Alternate()

    def resolved(self, space: Space) -> list[tuple[Block, list[str]]]:
        """Each block with its expanded parameter keys, validated against the space."""
        resolved = [(block, block.resolve_keys(space)) for block in self.blocks]
        seen: dict[str, str] = {}
        for block, keys in resolved:
            for key in keys:
                if key in seen:
                    raise ValueError(f"{key!r} is in two blocks: {seen[key]!r} and {block.name!r}")
                seen[key] = block.name
        return resolved


def default_search(space: Space) -> Search:
    """One block per variable type: text to `Reflection`, numbers and choices to `TPE`.

    Raises:
        ImportError: When the space has numeric or choice variables and optuna is not installed.
            Install the `search` extra, or assign those variables to `Reflection` explicitly.
    """
    text_keys = [info.key for info in space.params() if info.kind in TEXT_KINDS]
    numeric_keys = [info.key for info in space.params() if info.kind not in TEXT_KINDS]
    blocks: list[Block] = []
    if text_keys:
        blocks.append(Block(Reflection(), *text_keys, name="texts"))
    if numeric_keys:
        try:
            import optuna  # noqa: F401  # optional extra: search, checked here to fail at init
        except ImportError as error:
            raise ImportError(_OPTUNA_INIT_HINT) from error
        blocks.append(Block(TPE(), *numeric_keys, name="numerics"))
    return Search(blocks=blocks, schedule=Alternate())
