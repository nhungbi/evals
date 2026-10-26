"""TPE over numbers and choices, one study per context (optional extra: search)."""

from __future__ import annotations

from typing import Any, ClassVar

from ...types.optimization import Observation, Patch
from ..declaration.space import ParamKind
from .method import Method, ProposalRequest


class TPE(Method):
    """Optuna TPE with one study per context (the values outside its block); studies maximize.

    `seed=None` derives the seed from the run seed. The sampler is reseeded before each ask from
    (seed, iteration, attempt, context), and studies are rebuilt with `add_trial` on resume.
    Unmeasured outcomes are told as `TrialState.FAIL`. A duplicate's trial is closed with the existing
    candidate's scalar (FAIL if that was never measured), so the study learns the point is taken and
    no trial stays open.
    """

    kinds: ClassVar[frozenset[ParamKind]] = frozenset({"int", "float", "one_of", "choice"})

    def __init__(self, *, seed: int | None = None, n_startup_trials: int = 4):
        """Configure the sampler; studies are created per context on first use.

        Args:
            seed: None derives it from the run seed.
            n_startup_trials: Random trials before TPE models the results.
        """
        raise NotImplementedError

    async def propose(self, request: ProposalRequest) -> list[Patch]:
        raise NotImplementedError

    def observe(self, observation: Observation) -> None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """`seed` and `n_startup_trials`."""
        raise NotImplementedError
