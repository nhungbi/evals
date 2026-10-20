import importlib
import sys
from unittest.mock import patch

import pytest

from strands_evals import redteam
from strands_evals.redteam.generators import adversarial
from strands_evals.redteam.strategies import base

_OLD = "strands_evals.experimental.redteam"


@pytest.fixture
def old_module():
    for name in [n for n in sys.modules if n == _OLD or n.startswith(f"{_OLD}.")]:
        del sys.modules[name]
    with pytest.warns(DeprecationWarning, match="strands_evals.redteam"):
        return importlib.import_module(_OLD)


def test_old_path_warns_and_reexports_public_api(old_module):
    assert old_module.__all__ == redteam.__all__
    for name in redteam.__all__:
        assert getattr(old_module, name) is getattr(redteam, name)


def test_old_deep_import_is_same_module(old_module):
    assert importlib.import_module(f"{_OLD}.strategies.base") is base
    assert old_module.strategies.base is base


def test_old_patch_target_patches_new_module(old_module):
    with patch(f"{_OLD}.generators.adversarial.Agent") as mock_agent:
        assert adversarial.Agent is mock_agent
