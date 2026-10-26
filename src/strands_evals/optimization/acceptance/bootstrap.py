"""Paired bootstrap comparison, kept separate so it can be swapped later."""

from __future__ import annotations

from collections.abc import Mapping

from ...types.optimization import PairedComparison


def paired_bootstrap(
    challenger: Mapping[str, float],
    champion: Mapping[str, float],
    *,
    level: float,
    seed: int | str,
    resamples: int = 2000,
) -> PairedComparison:
    """Two-sided CI at `level` of the mean per-case difference, over the cases both measured.

    Args:
        challenger: Case name -> the challenger's holdout score.
        champion: Case name -> the champion's holdout score.
        level: Two-sided CI level (the gate's `confidence`).
        seed: Seeds `random.Random`; the optimizer passes `f"{seed}:{iteration}:gate"`.
        resamples: Number of bootstrap resamples.

    Returns:
        The mean per-case difference (challenger - champion), its CI, and the number of shared cases.
    """
    raise NotImplementedError
