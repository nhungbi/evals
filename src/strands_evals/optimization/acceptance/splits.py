"""The run's train / selection / holdout split."""

from __future__ import annotations

from collections.abc import Sequence

from ...types.optimization import Splits


def split_cases(names: Sequence[str], *, holdout_fraction: float, selection_fraction: float, seed: int) -> Splits:
    """Shuffle with `random.Random(seed)` and carve the holdout first.

    Fewer than one case in any split raises `ValueError` naming the shortfall. Train never doubles
    as selection.

    Args:
        names: Every case name; names are unique.
        holdout_fraction: Share of cases held out. Carved first, so a seed holds out the same cases
            whatever `selection_fraction` is.
        selection_fraction: Share of cases for the selection split, carved from the rest.
        seed: The run seed.

    Returns:
        The three splits; every case is in exactly one.

    Raises:
        ValueError: A split would get no cases, e.g. "need >= 3 cases".
    """
    raise NotImplementedError
