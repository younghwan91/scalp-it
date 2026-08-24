"""저변동성 팩터 포트폴리오 선정 — 배포 스펙(31번 채택본) 한 곳 정의.

31번 다중팩터 배치에서 채택된 저변동성 팩터를 실전 포트폴리오로 굴리는 공용 로직이다.
:mod:`scalp_it.cli_lowvol_portfolio`(오늘자 롱숏 표·CSV)와 :mod:`scalp_it.cli_book_report`
(PEAD+저변동 결합 대시보드)가 **같은 함수**(:func:`select_lowvol_portfolio`)를 쓴다 —
배포 스펙이 한 곳에서만 정의되도록.

    신호      -std(직전 60거래일 일수익)  → Q10 = 저변동(롱), Q1 = 고변동(숏)
    유니버스  리밸일 직전 20거래일 평균 거래대금 >= 5000(백만원=50억), 스팩·리츠 제외
    포트폴리오 폴드(리밸 시점)내 횡단면 데실. 롱=Q10·숏=Q1 각 동일가중 (시장중립)
    진입      리밸일 익일 시가        보유  +21거래일(월 리밸)      비용  왕복 68bp+숏차입

31번: 50억서 롱숏 net +2.34%/월(t=3.66), 500억서도 net +3.07%(t=2.77). 엣지는 저변동
롱 + 고변동(복권주) 숏. 32번 결합 검증: PEAD 와 역변동성 결합 시 샤프 1.39→1.57.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: 배포 스펙 확정 규칙 — 바꾸면 31번 검증 자격이 사라진다.
RULE_VERSION = "v1-2026-08-24"
DEPLOY_LIQ = 5000.0   # 진입 직전 20거래일 평균 거래대금 하한 (백만원 = 50억)
LIQ_WINDOW = 20
VOL_WINDOW = 60       # 일수익 표준편차 룩백
HOLD_DAYS = 21        # 보유 거래일 (월 리밸)
N_DECILES = 10
MIN_NAMES = 10        # 유니버스 하한 (데실당 1+)
COST_ROUND_TRIP = 0.0068
BORROW_21D = 0.03 * 21 / 252  # 숏차입 연 3% × 21/252

#: 31번 채택 기대치 (50억, 월 롱숏 net).
EXPECTED = {"net_longshort": 0.0234, "t": 3.66}


@dataclass
class LowvolPortfolio:
    """최신 리밸 시점의 확정 롱숏(저변동 롱 / 고변동 숏)."""

    period: str                 # 리밸 월 'YYYY-MM'
    asof_date: pd.Timestamp     # 신호 계산 기준일(리밸일)
    longs: pd.DataFrame
    shorts: pd.DataFrame


def _spac_reit(meta: pd.DataFrame) -> set[str]:
    if meta.empty or "name" not in meta.columns:
        return set()
    mask = meta["name"].str.contains("스팩|기업인수목적|리츠", na=False)
    return set(meta.loc[mask, "code"].astype(str))


def select_lowvol_portfolio(
    adj: pd.DataFrame,
    meta: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    liq_floor: float = DEPLOY_LIQ,
) -> LowvolPortfolio | None:
    """리밸일(=asof 이하 최신 거래일) 기준 저변동 롱(Q10)·고변동 숏(Q1)을 확정한다.

    형성 가능 = 리밸일에 실제 거래 중이고, 60거래일 변동성·20거래일 유동성을 계산할
    이력이 있으며 유니버스가 ``MIN_NAMES`` 이상. 그런 시점이 없으면 ``None``.
    """
    if adj.empty:
        return None
    df = adj.copy()
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= asof].sort_values(["code", "date"])
    if df.empty:
        return None
    rebal = df["date"].max()  # 신호 계산 기준일

    excl = _spac_reit(meta)
    rows = []
    for code, g in df.groupby("code", sort=False):
        if code in excl:
            continue
        if g["date"].iloc[-1] != rebal:  # 리밸일에 거래하지 않은 종목 제외
            continue
        cl = g["close"].to_numpy(dtype=float)
        tv = g["trade_value"].to_numpy(dtype=float)
        if len(cl) < VOL_WINDOW + 1:
            continue
        rets = cl[1:] / cl[:-1] - 1.0
        vol = float(np.nanstd(rets[-VOL_WINDOW:], ddof=1))
        if len(tv) < LIQ_WINDOW or not np.isfinite(vol) or vol <= 0:
            continue
        liq = float(np.nanmean(tv[-LIQ_WINDOW:]))
        if not (np.isfinite(liq) and liq >= liq_floor):
            continue
        rows.append({"code": code, "vol60": vol, "liq": liq,
                     "close": float(cl[-1])})
    if len(rows) < MIN_NAMES:
        return None
    cur = pd.DataFrame(rows)
    # 신호 = -vol (Q10 = 저변동 = 롱)
    r = (-cur["vol60"]).rank(method="first")
    cur["decile"] = pd.qcut(r, N_DECILES, labels=range(1, N_DECILES + 1)).astype(int)
    cur["vol_ann"] = cur["vol60"] * np.sqrt(252)

    name_map = dict(zip(meta["code"].astype(str), meta["name"])) if len(meta) else {}
    mkt_map = (dict(zip(meta["code"].astype(str), meta["market"]))
               if "market" in meta.columns else {})
    sec_map = (dict(zip(meta["code"].astype(str), meta["sector"]))
               if "sector" in meta.columns else {})
    cur["name"] = cur["code"].map(name_map).fillna("")
    cur["market"] = cur["code"].map(mkt_map)
    cur["sector"] = cur["code"].map(sec_map)

    longs = cur[cur["decile"] == N_DECILES].sort_values("vol60", ascending=True)
    shorts = cur[cur["decile"] == 1].sort_values("vol60", ascending=False)
    return LowvolPortfolio(
        period=rebal.strftime("%Y-%m"),
        asof_date=rebal,
        longs=longs.reset_index(drop=True),
        shorts=shorts.reset_index(drop=True),
    )


def build_frame(portfolio: LowvolPortfolio) -> pd.DataFrame:
    """롱숏을 하나의 매매 지시 프레임으로. 비중 = 다리 내 동일가중."""
    rows = []
    for side, frame in (("롱", portfolio.longs), ("숏", portfolio.shorts)):
        n = len(frame)
        weight = 100.0 / n if n else 0.0
        for x in frame.itertuples():
            rows.append({
                "side": side,
                "code": str(x.code),
                "name": x.name,
                "market": "" if pd.isna(x.market) else x.market,
                "decile": int(x.decile),
                "vol_ann_pct": round(float(x.vol_ann) * 100, 1),
                "close": round(float(x.close), 1),
                "weight_pct": round(weight, 3),
                "liq_eok": round(float(x.liq) / 100, 0),
            })
    return pd.DataFrame(rows)
