"""105번 개장 동시호가 탐색 — 라벨·전일 피처·유니버스의 **순수 함수**(DB 없음, 테스트 대상).

왜 따로 뗐나: 97번에서 순위를 당일 거래대금으로 매겨 부호가 뒤집혔다(+71.9 → −28.0bp). 그래서
"전일 값만 쓰는가 · 비용은 한 번만 빼는가 · 창이 09:00 open → 지정 분 close 인가"를 합성 데이터로
먼저 못 박고, run.py 는 DB 에서 읽어 이 함수들에 넘기기만 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAX_BP = 20.0                     # 매도 증권거래세·농특세 (krx_quant_core.costs 2026)
FEE_BP = 3.0                      # 왕복 수수료 0.015% × 2
# daytrade-it 실측 스프레드(97 §4b). 단일가는 스프레드가 없는 게 아니라 한 번에 지불한다.
OPEN_SPREAD_BP = {"거래소": 19.1, "코스닥": 30.7}     # 개장 동시호가
CONT_SPREAD_BP = {"거래소": 14.0, "코스닥": 19.1}     # 연속 세션(매도 쪽)
HOLDS = (1, 5, 15)
TOP_VALUE = 1300
MIN_PRICE = 1000
MAX_GAP_DAYS = 7                  # 전일 행이 이보다 오래됐으면(거래정지 등) 전일 값으로 안 본다
RET_TAIL = 0.10                   # 전일 수익률 상위/하위 10%


def cost_bp(market: str) -> float:
    """종목 한 건의 왕복 비용(bp): 세금 + 수수료 + 개장 동시호가 스프레드 절반 + 연속세션 스프레드 절반."""
    return TAX_BP + FEE_BP + OPEN_SPREAD_BP[market] / 2 + CONT_SPREAD_BP[market] / 2


def prior_day_features(daily: pd.DataFrame) -> pd.DataFrame:
    """일봉 → (code, date) 행마다 **전일** 종가·고가·거래대금·수익률·20일 변동성.

    date 는 거래 당일(D)이고 p_* 는 전부 D−1 행에서 온다. 20일 변동성도 D−1 까지의 수익률로만 계산한다.
    전일 행과 달력 7일 넘게 비면 버린다(그 "전일"은 전일이 아니다).
    """
    d = daily.sort_values(["code", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    g = d.groupby("code", sort=False)
    d["ret1"] = g["close"].pct_change()
    d["vol20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=10).std())
    g = d.groupby("code", sort=False)
    out = pd.DataFrame({
        "code": d["code"], "date": d["date"],
        "p_date": g["date"].shift(1),
        "p_close": g["close"].shift(1), "p_high": g["high"].shift(1),
        "p_tv": g["trade_value"].shift(1), "p_ret": g["ret1"].shift(1),
        "p_vol20": g["vol20"].shift(1),
    })
    out = out.dropna(subset=["p_date", "p_close", "p_tv"])
    out = out[(out["date"] - out["p_date"]).dt.days <= MAX_GAP_DAYS]
    return out.reset_index(drop=True)


def day_last_close(minute_eod: pd.DataFrame) -> pd.DataFrame:
    """장 막판 분봉(≤15:30) → (code, date) 당 **원주가** 종가(15:30 단일가 행, 없으면 그 전 마지막 행).

    왜: `daily_bars.close` 는 분할 소급 수정주가이고 분봉은 원주가다. 분할 종목 31개에서 10배·5배·2배 어긋나
    거짓 ±90% 갭이 된다(107 번에서 셀 하나가 −151bp 로 찍힘). 분봉끼리 비교해야 같은 단위다.
    """
    m = minute_eod.copy()
    m["ts"] = pd.to_datetime(m["ts"])
    m = m[m["ts"].dt.time <= pd.Timestamp("15:30").time()].sort_values(["code", "ts"])
    m["date"] = m["ts"].dt.normalize()
    out = m.groupby(["code", "date"], sort=False)["close"].last().rename("close_raw").reset_index()
    return out


def attach_raw_prev_close(feats: pd.DataFrame, last_closes: pd.DataFrame) -> pd.DataFrame:
    """전일(p_date)의 원주가 종가를 p_close_raw 로 붙인다. 전일 분봉이 없으면 그 행은 없다."""
    lc = last_closes.rename(columns={"date": "p_date", "close_raw": "p_close_raw"})
    lc = lc.assign(p_date=pd.to_datetime(lc["p_date"]))
    return feats.merge(lc[["code", "p_date", "p_close_raw"]], on=["code", "p_date"], how="inner")


def _q4(s: pd.Series) -> pd.Series:
    """날짜 안 4분위(0~3). 동률은 순위로 깨서 빈 분위가 안 생기게 한다."""
    r = s.rank(method="first")
    return pd.cut(r, 4, labels=False) if len(s) >= 4 else pd.Series(0, index=s.index)


def build_universe(feats: pd.DataFrame) -> pd.DataFrame:
    """전일 종가 ≥ 1,000 · 전일 거래대금 상위 1,300. 그 안에서 전일 수익률 분위(판정 셀)와 매칭 층을 붙인다.

    ret_cell: 'top10' / 'bot10' / 'mid'. vol_q·tv_q: 날짜 안 4분위(매치드 널 층). 전부 전일 값으로만 만든다.
    1,000원 필터는 원주가(p_close_raw)로 — 수정주가는 분할 종목에서 단위가 다르다. 수익률·고가=종가·거래대금은
    일봉 내부 비율이라 수정주가끼리 써도 된다.
    """
    u = feats[feats["p_close_raw"] >= MIN_PRICE].copy()
    u["v_rank"] = u.groupby("date")["p_tv"].rank(ascending=False, method="first")
    u = u[u["v_rank"] <= TOP_VALUE].copy()
    u = u.dropna(subset=["p_ret", "p_vol20"])     # 변동성 층이 없으면 널 대조도 없다 — 상장 10일 미만은 뺀다
    pct = u.groupby("date")["p_ret"].rank(pct=True, method="first")
    u["ret_cell"] = np.where(pct > 1 - RET_TAIL, "top10", np.where(pct <= RET_TAIL, "bot10", "mid"))
    u["vol_q"] = u.groupby("date")["p_vol20"].transform(_q4)
    u["tv_q"] = u.groupby("date")["p_tv"].transform(_q4)
    u["limit_up"] = (u["p_close"] == u["p_high"]) & (u["p_ret"] >= 0.28)
    return u.reset_index(drop=True)


def horizon_closes(minute: pd.DataFrame) -> pd.DataFrame:
    """09:00~09:15 분봉 → (code, date) 당 open0(09:00 행 open)과 c1·c5·c15(해당 분 이하 마지막 close).

    09:00 행이 없으면(개장 체결 없음) 그 종목·날은 없다. 09:05 가 비면 그 전 마지막 분의 close 를 쓴다 —
    "09:05 에 시장가로 팔면 받는 가격"의 보수적 근사다(그 분에 체결이 없었다는 건 호가가 움직이지 않았다는 뜻이다).
    """
    m = minute.copy()
    m["ts"] = pd.to_datetime(m["ts"])
    m["date"] = m["ts"].dt.normalize()
    m["minute"] = m["ts"].dt.hour * 60 + m["ts"].dt.minute - 9 * 60
    m = m[(m["minute"] >= 0) & (m["minute"] <= max(HOLDS))].sort_values(["code", "date", "minute"])
    first = m[m["minute"] == 0][["code", "date", "open"]].rename(columns={"open": "open0"})
    out = first
    for h in HOLDS:
        last = (m[m["minute"] <= h].groupby(["code", "date"], sort=False)["close"].last()
                .rename(f"c{h}").reset_index())
        out = out.merge(last, on=["code", "date"], how="left")
    return out.reset_index(drop=True)


def label(events: pd.DataFrame) -> pd.DataFrame:
    """r{h} = c{h}/open0 − 1 − 비용(종목별, 한 번). gap 은 보고용(진입 조건 아님), 원주가끼리(분봉/분봉)."""
    e = events.copy()
    c = e["cost_bp"] / 1e4
    for h in HOLDS:
        e[f"r{h}"] = e[f"c{h}"] / e["open0"] - 1 - c
    e["gap"] = e["open0"] / e["p_close_raw"] - 1
    return e


def build_events(daily: pd.DataFrame, minute: pd.DataFrame, markets: pd.DataFrame,
                 minute_eod: pd.DataFrame, bad_days: set[str] | None = None) -> pd.DataFrame:
    """일봉 + 개장 분봉 + 마감 분봉 + 시장 → 라벨 붙은 사건 목록(유니버스 전체 = 매치드 널의 풀)."""
    u = build_universe(attach_raw_prev_close(prior_day_features(daily), day_last_close(minute_eod)))
    hc = horizon_closes(minute)
    e = u.merge(hc, on=["code", "date"], how="inner")
    e = e.merge(markets[["code", "market"]], on="code", how="inner")
    e["cost_bp"] = e["market"].map(cost_bp)
    if bad_days:
        e = e[~e["date"].dt.strftime("%Y-%m-%d").isin(bad_days)]
    return label(e).reset_index(drop=True)
