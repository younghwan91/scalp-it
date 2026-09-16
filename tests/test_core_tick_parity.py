"""리서치 스크립트의 호가단위·왕복비용 복사본 → krx-quant-core 정본 교체 검증.

스크립트들은 패키지가 아니고(최상위에서 바로 도는 것도 있다) 일부는 simnode 전용
의존성(numba·lightgbm)을 import 한다. 그래서 모듈을 통째로 import 하지 않고, 파일을
AST 로 읽어 **교체된 함수·상수 정의만** 떼어 실행한다. 교체 전 원본 구현은 아래
``_orig_*`` 로 본문을 그대로 옮겨 두고, 조밀한 가격 격자에서 한 값도 다르지 않은지 본다.

잠긴(research/runs/*/PREREG.lock) 전진 판정 연구가 쓰는 파일은 대상이 아니다 —
``scripts/flow_83/features.py``·``tiger_86/engine.py``·``bandit_88/rewards.py``·
``rl_85/dqn.py`` 는 손대지 않았다.
"""

from __future__ import annotations

import ast
import math
import random
import re
from datetime import date
from pathlib import Path

import pytest
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.market.codes import Market

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


# --------------------------------------------------------------------- 원본(교체 전 본문 그대로)

def _orig_tick(p):
    # scripts/orderflow_80/of_scratch.py:tick · of_strength.py:tick · of_build.py:_tick
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if p < lim:
            return t
    return 1000


def _orig_stock_tick(p: float) -> float:
    # scripts/etf_strength_81.py:stock_tick
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if p < lim:
            return float(t)
    return 1000.0


# --------------------------------------------------------------------- AST 추출

def _extract(path: Path, names: set[str]) -> dict:
    """``path`` 에서 top-level import 와 ``names`` 의 def/대입만 실행한 네임스페이스.

    krx_quant_core·datetime import 만 살린다(스크립트의 무거운 의존성은 건너뛴다).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {
            "krx_quant_core", "datetime", "__future__"
        }:
            if node.module != "__future__":
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names or isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            keep.append(node)
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    return ns


TICK_FUNCS = [
    ("orderflow_80/of_scratch.py", "tick", _orig_tick),
    ("orderflow_80/of_strength.py", "tick", _orig_tick),
    ("orderflow_80/of_build.py", "_tick", _orig_tick),
    ("etf_strength_81.py", "stock_tick", _orig_stock_tick),
]


def _price_grid() -> list[float]:
    bands = (2000, 5000, 20000, 50000, 200000, 500000)
    prices: set[float] = set(range(1, 2_000_001, 97))
    for b in bands:
        for d in (-1, -0.5, -0.01, 0, 0.01, 0.5, 1):
            prices.add(b + d)
    prices.update({0, -1, -0.5, 0.5, 2_000_000})
    rng = random.Random(0)
    prices.update(rng.uniform(0, 2_000_000) for _ in range(50_000))
    return sorted(prices)


PRICES = _price_grid()


@pytest.mark.parametrize(("rel", "name", "orig"), TICK_FUNCS)
def test_tick_replacement_matches_original_everywhere(rel, name, orig):
    fn = _extract(SCRIPTS / rel, {name})[name]
    for p in PRICES:
        got, want = fn(p), orig(p)
        assert got == want and type(got) is type(want), (rel, p, got, want)


@pytest.mark.parametrize(("rel", "name", "orig"), TICK_FUNCS)
def test_tick_replacement_keeps_nan_and_numpy_behaviour(rel, name, orig):
    np = pytest.importorskip("numpy")
    fn = _extract(SCRIPTS / rel, {name})[name]
    assert fn(math.nan) == orig(math.nan)
    assert fn(math.inf) == orig(math.inf)
    for p in (np.float64(1999.0), np.float64(49999.5), np.int64(200000), np.float32(4999.9)):
        assert fn(p) == orig(p)


@pytest.mark.parametrize(("rel", "name", "_orig"), TICK_FUNCS)
def test_tick_replacement_uses_core(rel, name, _orig):
    src = (SCRIPTS / rel).read_text(encoding="utf-8")
    assert "from krx_quant_core.market.ticks import tick_size_int" in src
    assert "(2000, 1), (5000, 5)" not in src


def test_etf_tick_is_kept():
    ns = _extract(SCRIPTS / "etf_strength_81.py", {"etf_tick"})
    assert ns["etf_tick"](1999) == 1.0 and ns["etf_tick"](2000) == 5.0


# --------------------------------------------------------------------- 왕복비용

COST_FILES = [
    "orderflow_80/of_long.py",
    "orderflow_80/of_sim.py",
    "orderflow_80/of_scratch.py",
    "orderflow_80/of_strength.py",
    "orderflow_80/of_build.py",
    "orderflow_80/of_scratch_report.py",
    "rl_82/dataset.py",
]


def test_core_round_trip_cost_2026_is_23bp():
    for m in (Market.KOSDAQ, Market.KOSPI):
        assert round_trip_cost(date(2026, 1, 2), m) == 0.0023


@pytest.mark.parametrize("rel", COST_FILES)
def test_research_cost_constant_comes_from_core(rel):
    ns = _extract(SCRIPTS / rel, {"COST"})
    assert ns["COST"] == 0.0023


@pytest.mark.parametrize("rel", [*COST_FILES, "rl_82/lgbm.py"])
def test_no_cost_literal_left(rel):
    src = (SCRIPTS / rel).read_text(encoding="utf-8")
    assert not re.search(r"\b0\.0023\b", src), rel


def test_lgbm_takes_cost_from_dataset():
    src = (SCRIPTS / "rl_82" / "lgbm.py").read_text(encoding="utf-8")
    assert re.search(r"from dataset import .*\bCOST\b", src)
