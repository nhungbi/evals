"""Deprecated alias for `strands_evals.redteam`."""

import importlib
import pkgutil
import sys
import warnings

from strands_evals import redteam as _redteam
from strands_evals.redteam import *  # noqa: F403
from strands_evals.redteam import __all__  # noqa: F401

warnings.warn(
    "`strands_evals.experimental.redteam` is deprecated and will be removed in a future release. "
    "Import from `strands_evals.redteam` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias every submodule so old deep imports (`strands_evals.experimental.redteam.strategies.base`) and
# `unittest.mock.patch` targets resolve to the same module objects as the new path.
for _info in pkgutil.walk_packages(_redteam.__path__, prefix=f"{_redteam.__name__}."):
    _module = importlib.import_module(_info.name)
    _alias = _info.name.replace(_redteam.__name__, __name__, 1)
    sys.modules[_alias] = _module
    _parent, _, _child = _alias.rpartition(".")
    setattr(sys.modules[_parent], _child, _module)
