"""106번 VI 재개 탐색 — 복원 규칙과 라벨 산술을 손으로 아는 값으로 고정한다.

복원이 공백 하나를 더 잡거나 놓치면 표본이 바뀌고, 라벨이 한 초 밀리면 결론이 뒤집힌다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parents[1] / "scripts" / "explore_106"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"explore106_{name}", _DIR / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V = _load("vi_reconstruct")
L = _load("labels")

T9 = 9 * 3600


def _ticks(code: str, secs, prices):
    return pd.DataFrame({"code": code, "sec": np.asarray(secs, np.int64),
                         "seq": np.arange(len(secs)), "price": np.asarray(prices, float)})


def _quotes(code: str, secs):
    return pd.DataFrame({"code": code, "sec": np.asarray(secs, np.int64)})


def _liquid_background(code="BG", lo=T9, hi=T9 + 22800):
    """다른 종목이 매초 체결 — 시장 전체 틱 유입이 정상임을 뜻한다."""
    s = np.arange(lo, hi)
    return _ticks(code, s, np.full(len(s), 50000.0))


# ---------------------------------------------------------------- VI 복원

def test_gap_with_jump_and_frozen_quotes_is_vi():
    # 10:00:00 까지 1초 간격 체결, 130초 공백, 10:02:10 에 +2% 로 재개
    pre = np.arange(T9 + 3000, T9 + 3600)
    post = np.arange(T9 + 3600 + 130, T9 + 3600 + 130 + 300)
    secs = np.concatenate([pre, post])
    prices = np.concatenate([np.full(len(pre), 10000.0), np.full(len(post), 10200.0)])
    tk = pd.concat([_ticks("A", secs, prices), _liquid_background()], ignore_index=True)
    qt = _quotes("A", np.arange(T9 + 3000, T9 + 3600 + 430))   # 단일가 접수 중에도 매초 호가 갱신
    ev = V.reconstruct(tk, qt)
    ev = ev[ev.code == "A"]
    assert len(ev) == 1
    r = ev.iloc[0]
    assert r.sec_prev == T9 + 3599 and r.sec_resume == T9 + 3730 and r.gap == 131
    assert abs(r.jump - 0.02) < 1e-9
    assert r.n_quote_gap == 129 and r.keep and r.direction == 1


def test_short_gap_is_not_vi():
    pre = np.arange(T9 + 3000, T9 + 3600)
    post = np.arange(T9 + 3600 + 50, T9 + 3600 + 350)
    secs = np.concatenate([pre, post])
    prices = np.concatenate([np.full(len(pre), 10000.0), np.full(len(post), 10500.0)])
    tk = pd.concat([_ticks("A", secs, prices), _liquid_background()], ignore_index=True)
    ev = V.reconstruct(tk, _quotes("A", secs))
    assert len(ev[ev.code == "A"]) == 0, "110초 미만 공백은 후보도 아니다"


def test_gap_without_price_move_is_liquidity_gap():
    pre = np.arange(T9 + 3000, T9 + 3600)
    post = np.arange(T9 + 3600 + 130, T9 + 3600 + 430)
    secs = np.concatenate([pre, post])
    prices = np.full(len(secs), 10000.0)                      # 변화 0
    tk = pd.concat([_ticks("A", secs, prices), _liquid_background()], ignore_index=True)
    ev = V.reconstruct(tk, _quotes("A", np.arange(T9 + 3000, T9 + 3600 + 430)))
    a = ev[ev.code == "A"]
    assert len(a) == 1 and not a.iloc[0].keep and "가격" in a.iloc[0].reason


def test_gap_with_frozen_quotes_is_liquidity_gap():
    # 가격이 뛰어도 공백 중 호가가 안 움직였으면 단일가 접수가 아니라 그냥 조용했던 것이다
    pre = np.arange(T9 + 3000, T9 + 3600)
    post = np.arange(T9 + 3600 + 130, T9 + 3600 + 430)
    secs = np.concatenate([pre, post])
    prices = np.concatenate([np.full(len(pre), 10000.0), np.full(len(post), 10200.0)])
    tk = pd.concat([_ticks("A", secs, prices), _liquid_background()], ignore_index=True)
    qt = _quotes("A", np.concatenate([pre, [T9 + 3600, T9 + 3650], post]))   # 공백 중 2번
    ev = V.reconstruct(tk, qt)
    a = ev[ev.code == "A"]
    assert len(a) == 1 and not a.iloc[0].keep and "호가 정지" in a.iloc[0].reason
    assert a.iloc[0].n_quote_gap == 1                      # 시작 2초 안의 1건은 안 센다


def test_premove_alone_qualifies():
    # 재개가 같은 가격이어도 공백 직전 60초에 +3.5% 올랐으면 VI 로 본다(동적 VI 근사)
    pre_a = np.arange(T9 + 3000, T9 + 3540)
    pre_b = np.arange(T9 + 3540, T9 + 3600)
    post = np.arange(T9 + 3600 + 125, T9 + 3600 + 425)
    secs = np.concatenate([pre_a, pre_b, post])
    prices = np.concatenate([np.full(len(pre_a), 10000.0), np.full(len(pre_b), 10350.0),
                             np.full(len(post), 10350.0)])
    tk = pd.concat([_ticks("A", secs, prices), _liquid_background()], ignore_index=True)
    ev = V.reconstruct(tk, _quotes("A", np.arange(T9 + 3000, T9 + 3600 + 425)))
    a = ev[ev.code == "A"]
    assert len(a) == 1 and a.iloc[0].keep and abs(a.iloc[0].pre_move - 0.035) < 1e-9


def test_market_wide_outage_is_not_vi():
    # 모든 종목이 같은 구간에 멈췄다 → 수집 중단이지 VI 가 아니다
    pre = np.arange(T9 + 3000, T9 + 3600)
    post = np.arange(T9 + 3600 + 130, T9 + 3600 + 430)
    secs = np.concatenate([pre, post])
    pa = np.concatenate([np.full(len(pre), 10000.0), np.full(len(post), 10200.0)])
    bg = _ticks("BG", secs, np.full(len(secs), 50000.0))      # 배경 종목도 같은 공백
    tk = pd.concat([_ticks("A", secs, pa), bg], ignore_index=True)
    ev = V.reconstruct(tk, _quotes("A", secs))
    a = ev[ev.code == "A"]
    assert len(a) == 1 and not a.iloc[0].keep and "시장" in a.iloc[0].reason


def test_gap_outside_session_is_ignored():
    # 15:20 마감 단일가 공백은 연속 세션 밖
    pre = np.arange(T9 + 22000, T9 + 22800)
    post = np.array([T9 + 22800 + 600])
    secs = np.concatenate([pre, post])
    prices = np.concatenate([np.full(len(pre), 10000.0), [10300.0]])
    tk = pd.concat([_ticks("A", secs, prices), _liquid_background(hi=T9 + 22800 + 601)],
                   ignore_index=True)
    ev = V.reconstruct(tk, _quotes("A", secs))
    assert len(ev[ev.code == "A"]) == 0


def test_direction_down():
    pre = np.arange(T9 + 3000, T9 + 3600)
    post = np.arange(T9 + 3600 + 130, T9 + 3600 + 430)
    secs = np.concatenate([pre, post])
    prices = np.concatenate([np.full(len(pre), 10000.0), np.full(len(post), 9700.0)])
    tk = pd.concat([_ticks("A", secs, prices), _liquid_background()], ignore_index=True)
    ev = V.reconstruct(tk, _quotes("A", secs))
    assert ev[ev.code == "A"].iloc[0].direction == -1


# ---------------------------------------------------------------- 라벨

def test_taker_fixed_enters_t0_plus_1_exits_t0_plus_1_plus_h():
    #        t:  0    1     2     3     4     5
    bid = np.array([100, 100, 110, 130, 100, 100], float)
    ask = np.array([101, 102, 111, 131, 101, 101], float)
    r = L.taker_fixed(bid, ask, h=2)              # t=0 → 진입 ask[1]=102, 청산 bid[3]=130
    assert abs(r[0] - (130 / 102 - 1 - L.COST)) < 1e-12
    assert np.isnan(r[3]) and np.isnan(r[4]) and np.isnan(r[5]), "뒤가 모자라면 NaN"


def test_taker_fixed_cost_once_and_no_oracle():
    bid = np.array([100, 100, 100, 100, 999, 100], float)   # 큰 값이 창 안에 있어도 정해진 초에만 판다
    ask = bid.copy()
    r = L.taker_fixed(bid, ask, h=4)                         # t=0 → 청산 bid[5]=100
    assert abs(r[0] + L.COST) < 1e-12


def test_maker_fixed_requires_fill_and_exits_h_after_fill():
    bid = np.array([100.0] * 30)
    bid[5] = 99.0                     # t=3 지정가 bid[4]=100 → t=5 뚫림 체결
    bid[9] = 104.0                    # 체결 뒤 h=4 초 → bid[9]
    e = L.fill_second(bid)
    r = L.maker_fixed(bid, e, h=4)
    assert e[3] == 5
    assert abs(r[3] - (104 / 100 - 1 - L.COST)) < 1e-12
    assert np.isnan(r[20]), "안 뚫리면 미체결 = NaN"


def test_oracle_reused_from_95():
    assert L.oracle is L.C.oracle and L.maker_pessimistic is L.C.maker_pessimistic
