"""Red team case type."""

from pydantic import PrivateAttr, model_validator
from typing_extensions import Self

from ...case import Case
from ...types import InputT, OutputT
from .strategies.base import AttackStrategy
from .types import RedTeamConfig


class RedTeamCase(Case[InputT, OutputT]):
    """Case carrying a typed RedTeamConfig. AttackGoal fields are mirrored into metadata."""

    config: RedTeamConfig
    # Runtime only: a private attribute isn't serialized, and `RedTeamExperiment` sets it on every run.
    _strategy: AttackStrategy | None = PrivateAttr(default=None)

    @property
    def strategy(self) -> AttackStrategy:
        """The attack strategy for this case x strategy variant.

        `RedTeamExperiment` attaches it to each expanded variant; base cases have none.

        Raises:
            ValueError: If no strategy is attached.
        """
        if self._strategy is None:
            raise ValueError(
                f"RedTeamCase {self.name!r} has no strategy; run it through RedTeamExperiment "
                "with attack_strategies set."
            )
        return self._strategy

    @strategy.setter
    def strategy(self, value: AttackStrategy) -> None:
        self._strategy = value

    @model_validator(mode="after")
    def _sync_metadata_from_config(self) -> Self:
        dump = dict(self.config.attack_goal.model_dump())
        if self.metadata is None:
            self.metadata = dump
        else:
            for key, value in dump.items():
                self.metadata.setdefault(key, value)
        return self
