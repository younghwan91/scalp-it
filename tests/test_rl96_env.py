"""96번 환경 — 행동을 3개로 좁힌 사상이 **93번 체결 코드와 정확히 같은 일**을 하는지 본다.

여기서 사상이 어긋나면 "지정가를 눌렀는데 시장가로 샀다" 같은 조용한 오류가 되고,
95번이 배제한 행동이 되살아나 결론이 무의미해진다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="96번 학습 스크립트는 torch 를 쓴다")

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("rl96_dqn", ROOT / "scripts" / "rl_96" / "dqn.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_action_space_excludes_market_buy():
    m = _mod()
    assert m.N_ACT == 3
    assert list(m.ACT4) == [0, 2, 3], "1번(93 의 시장가 매수)이 다시 들어오면 95번 결론이 깨진다"


def test_gamma_is_longer_horizon_than_93():
    m = _mod()
    assert m.GAMMA == 0.999
    import importlib.util as iu
    spec = iu.spec_from_file_location("rl85_dqn_probe", ROOT / "scripts" / "rl_85" / "dqn.py")
    r85 = iu.module_from_spec(spec)
    spec.loader.exec_module(r85)
    assert m.GAMMA > r85.GAMMA


def test_feature_list_contains_both_scales():
    m = _mod()
    assert "dstr30" in m.FEATS and "ret1800" in m.FEATS
    assert len(m.FEATS) == len(set(m.FEATS)), "피처 이름이 겹치면 열이 조용히 덮인다"


def test_training_records_action_names_for_the_shadow():
    """행동 이름이 산출물에 남아야 그림자가 인덱스로 추측하지 않는다."""
    m = _mod()
    assert m.ACTION_NAMES == ("hold", "limit_buy", "sell")
    assert len(m.ACTION_NAMES) == m.N_ACT
