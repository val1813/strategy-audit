"""共享 fixture。合成面板工厂在 synth.py（可被测试直接 import）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synth import equal_weight, lookahead_weight, make_prices, month_ends  # noqa: E402


@pytest.fixture(scope="session")
def px():
    return make_prices()


@pytest.fixture(scope="session")
def reb(px):
    return month_ends(px)


@pytest.fixture(scope="session")
def wm_clean(px, reb):
    return equal_weight(px, reb)


@pytest.fixture(scope="session")
def wm_dirty(px, reb):
    return lookahead_weight(px, reb)
