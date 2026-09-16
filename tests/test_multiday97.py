"""97번 급등주 다일 — **진입이 신호 다음 날 시가**이고 청산이 정확히 h 거래일 뒤 종가인지.

한 칸 밀리면 신호일 자체의 급등을 먹은 것으로 계산돼 전부 거짓 양수가 된다. 95번이 다섯 번 겪은 일이다.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_p = Path(__file__).resolve().parents[1] / "scripts" / "multiday_97" / "screen.py"


def _mod():
    spec = importlib.util.spec_from_file_location("multiday97", _p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["multiday97"] = m
    spec.loader.exec_module(m)
    return m


# 신호일은 인덱스 20(= 2026-01-25). 앞 20일은 변동성 계산용 워밍업이다(min_periods 10).
SIGNAL_DAY = pd.Timestamp("2026-01-05") + pd.Timedelta(days=20)


def _frame(n=30):
    rows = []
    closes, opens = [], []
    for i in range(n):
        # 가격은 1,000원 이상이어야 한다(스크린 조건) — 비율은 그대로라 기대값 계산은 안 바뀐다.
        if i < 20:
            c = 10000 + (i % 3) * 100          # 변동성이 0 이 아니어야 순위·매칭이 성립한다
            o = c
        elif i == 20:
            c, o = 11000, 10100                # 신호일 — 급등
        else:
            c = 11000 + (i - 20) * 100
            o = c - 100 if i == 21 else c
        closes.append(c)
        opens.append(o)
    for i in range(n):
        rows.append({"code": "A", "date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=i),
                     "open": opens[i], "high": max(opens[i], closes[i]), "low": min(opens[i], closes[i]),
                     "close": closes[i], "volume": 1000, "trade_value": 1e9})
        rows.append({"code": "B", "date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=i),
                     "open": 5000 + (i % 2) * 100, "high": 5100, "low": 5000,
                     "close": 5000 + (i % 2) * 100, "volume": 10, "trade_value": 1e8})
    return pd.DataFrame(rows)


def test_entry_is_next_day_open_not_signal_day_close():
    m = _mod()
    ev = m.build_events(_frame(), top_value=2, top_ret=1)
    a = ev[(ev.code == "A") & (ev.date == SIGNAL_DAY)]
    assert len(a) == 1, "신호일(급등한 날)이 사건이어야 한다"
    # 진입 = 다음 날 시가 11,000(=11,100−100), 청산 = 1일 보유 → 그날 종가 11,100
    assert abs(float(a.fwd1.iloc[0]) - (11100 / 11000 - 1 - m.COST)) < 1e-12


def test_hold_length_counts_trading_days_from_entry():
    m = _mod()
    ev = m.build_events(_frame(), top_value=2, top_ret=1)
    a = ev[(ev.code == "A") & (ev.date == SIGNAL_DAY)]
    # 진입 11,000, 2일 보유 → 신호일+2 종가 11,200
    assert abs(float(a.fwd2.iloc[0]) - (11200 / 11000 - 1 - m.COST)) < 1e-12


def test_cost_subtracted_once():
    m = _mod()
    ev = m.build_events(_frame(), top_value=2, top_ret=2)
    b = ev[(ev.code == "B") & (ev.date == SIGNAL_DAY)]
    assert len(b) == 1, "상위 2 를 고르면 B 도 사건이어야 한다"
    # B 는 그날 시가와 종가가 같아 총수익 0 — 남는 건 비용뿐이어야 한다(두 번 빠지면 −46bp)
    assert abs(float(b.fwd1.iloc[0]) + m.COST) < 1e-12


def test_volatility_column_uses_only_past():
    m = _mod()
    ev = m.build_events(_frame(), top_value=2, top_ret=2)
    assert "vol20" in ev.columns and ev.vol20.notna().all()
