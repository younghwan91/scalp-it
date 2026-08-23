"""신규주 유니버스 검증 — 사전등록 `docs/research/manju/15-newlisting-preregistration.md` 그대로.

  유니버스  상장일 = `daily_bars` 첫 거래일(≥ 2016-10-01). 경과일 d = 일봉 행 순서. d < 60.
            스팩·리츠·ETP 이름 제외. (A) 상장 후 D−1 까지 일평균 거래대금 ≥ 50억 ·
            (B) 트리거 분까지 당일 누적 거래대금 ≥ 50억.
  (A) 신규주 돌파  d ≥ 1. 돌파선 = 상장일~D−1 일봉 고가 최대(원주가). 분봉 고가가 처음 넘는 분
            (strict >). 09:00 봉 시가가 이미 위면 갭돌파 → 없음. 트리거 09:01~14:59.
            진입 = 트리거 분 종가. 진입 분 종가 ≥ 전일종가×1.295 (잠김) 제외·보고.
            주 규칙 `손절2` (−2%/종가) · 비교 `스승` (+2%/−2%/종가, 같은 분 손절 우선).
  (B) 신규주 상따  d ≥ 0. 기준선 = 전일종가 (D0 는 09:00 봉 시가). 분봉 고가가 기준선×1.25 를
            처음 넘는 분. 09:00 봉 고가가 이미 위면 없음. 09:01~14:59. 누적 거래대금 ≥ 50억.
            진입 = 트리거 분 종가. 잠김(≥ 기준선×1.295) 제외·보고 (D0 는 2023-06-26 이전 상장만).
            주 규칙 익일 시가 · 비교 당일 종가.
  피처 8    d_listing · ret_vs_d0_open · d0_close_pos · touch_minute · gap_open · cum_tv
            · cum_tv_ratio_hist · fi_net_prev. 추가·변경 금지.
  폴드      달력 분기. 유효 폴드 = 이벤트 ≥ 20건. 유효 폴드 ≥ 3 & 합 ≥ 80 미만이면 "판정 불가".
  채택      1 net34>0 · 2 유효 폴드 중 음수 ≤ 1 · 3 상위1% 제거 net34>0 · 4 독립 net34>0
            · 5 net55>0 · 6 (A)만: 상한가 마감 제외 net34>0.
  AUC       n ≥ 150 일 때만. GBM(max_depth=2, n_estimators=100), 시간순 2폴드.
            1단계 실패 & AUC ≥ 0.60 & n ≥ 150 → 2단계(폴드 안 상위 1/3). 아니면 끝.
  참고      (B) 11년 일봉 근사 (진입 = 기준선×1.25 가정 → 익일 시가). 판정 아님.

문서 외 파라미터·변형은 돌리지 않는다. 재사용: `newlisting_events.classify/compute`(유니버스·
일봉 이벤트), `verify_breakout._slot/describe/_row` (슬롯·통계 표), `limitup_timing.limit_up_price`.

문서가 모호한 지점 — 택한 (보수) 해석:
  A. 손절·목표 판정은 트리거 분 **다음 봉부터** 15:19 까지. 15:30 봉은 종가 청산가로만 (09번 A).
  B. 일 단위 제외 (09번 C 변형): 개장 봉 = **그날 첫 봉**. 첫 봉 시가가 `daily_bars` 시가와 0.2% 안에서
     일치해야 한다 — 개장 VI(시가가 전일종가 ±10% 밖)로 첫 체결이 09:02 인 날도 첫 봉 시가 = 시가이므로
     포함한다(09번 C 는 09:00 봉 부재를 전부 버렸지만 신규주 (B) 는 그런 날이 일봉 이벤트의 15% 라
     버리면 강한 갭 날만 빠지는 편향이 생긴다; 문서는 "VI 구간 = 봉 없음" 이라고만 했다). 첫 봉 시가가
     일봉 시가와 다르면 시가를 모르므로 제외·보고. 개장 봉 에서의 돌파는 트리거 아님(09:00 봉 규칙 동일).
     15:30 봉 없음 / 15:30 종가 vs `daily_bars` 종가 0.2% 초과 → 제외·보고.
  C. 수정/원주가 비율 드리프트 (전일→당일 0.2% 초과) 는 (A)·(B) 모두 그 종목·일자 제외 (문서는 (A)
     돌파선 항목에 적었지만 (B) 기준선 전일종가도 같은 오염을 받으므로 보수 적용).
  D. (B) 트리거 "처음 넘는" = strict > (문서 (A) 와 일봉 집계 `newlisting_events` 와 동일).
     고가 == 기준선×1.25 정확히 같은 봉 수는 따로 보고한다.
  E. 독립 이벤트의 기준 집합 = 일봉 주 이벤트 ∪ 분봉 이벤트 (09번 F). d 차이 > 5 면 독립.
  F. `fi_net_prev` = (foreign_ + institution) / acc_trde_qty 의 D−1 행 (`supply_demand`).
     결측·D0 → 0 (문서: 결측 0 채움). `ret_vs_d0_open` 의 상장일 시초가 = `daily_bars` D0 open(원주가).
  G. AUC 2폴드 = 날짜 정렬 앞 절반 학습 → 뒤 절반 검증. AUC 는 검증 폴드 1개 값.
  H. (B) 주 규칙 익일 시가가 없는 건(표본 마지막 날) 은 수익 계산에서 제외·보고.
  I. 당일 상한가 마감 = 15:30 종가 ≥ limit_up_price(기준선) (호가단위 내림). D0 2023-06-26 이후
     상장은 판정 불가 → 라벨 결측, "고가 = 종가" 대리 보고.

실행:
    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/verify_newlisting.py --out /path/newlist_result.md
DB 는 읽기만 한다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limitup_timing import limit_up_price
from newlisting_events import (
    D0_RULE_CHANGE,
    INDEP_GAP,
    LIMIT_RATIO,
    LISTING_CUTOFF,
    N_DAYS,
    SANGTTA_RATIO,
    TV_FLOOR_MN,
    _load_env,
    classify,
    compute,
)
from verify_breakout import (
    AUCTION_SLOT,
    CONT_LAST,
    COSTS,
    GBM,
    HDR,
    MINUTE_TV_DIV,
    SCALE_TOL,
    STOP,
    TARGET,
    TRIG_HI,
    TRIG_LO,
    _row,
    _slot,
    describe,
)

from scalp_it.adapters.db import connect
from scalp_it.intraday.loader import load_minute_bars

# ---- 사전 고정 (문서와 동일) -----------------------------------------------------------
FEATURES = ["d_listing", "ret_vs_d0_open", "d0_close_pos", "touch_minute", "gap_open",
            "cum_tv", "cum_tv_ratio_hist", "fi_net_prev"]
START, END = "2025-08-01", "2026-08-21"
COST, COST_HI = COSTS["net34"], COSTS["net55"]
FOLD_MIN_EVENTS = 20
MIN_VALID_FOLDS = 3
MIN_VALID_SUM = 80
MAX_NEG_FOLDS = 1
AUC_MIN_N = 150
AUC_GATE = 0.60
TOP_FRAC = 1 / 3
ADJ_DRIFT_TOL = 0.002
TV_500 = 50_000                      # 5차 겹침 확인용 (백만원)
DAILY_RECENT_FROM = "2025-05-01"     # 수정주가·수급은 분봉 구간 + 여유만 읽는다


# ====================================================================================
# 로드 · 일봉 준비
# ====================================================================================
def load_all(conn):
    t0 = time.time()
    first = pd.read_sql("select code, min(date) as first_date from daily_bars group by code", conn)
    first["code"] = first["code"].str.strip()
    cand = first[first["first_date"] >= LISTING_CUTOFF]
    codes = tuple(cand["code"].tolist())
    daily = pd.read_sql(
        "select code, date, open, high, low, close, volume, trade_value from daily_bars "
        "where code in %(codes)s order by code, date", conn, params={"codes": codes})
    daily["code"] = daily["code"].str.strip()
    stocks = pd.read_sql("select code, name, market from stocks", conn)
    stocks["code"] = stocks["code"].str.strip()
    delisted = pd.read_sql("select code, name from delisted_stocks", conn)
    delisted["code"] = delisted["code"].str.strip()
    adj = pd.read_sql(
        "select code, date, close from daily_bars_adjusted where code in %(codes)s and date >= %(s)s",
        conn, params={"codes": codes, "s": DAILY_RECENT_FROM})
    adj["code"] = adj["code"].str.strip()
    sd = pd.read_sql(
        "select code, date, foreign_, institution, acc_trde_qty from supply_demand "
        "where code in %(codes)s and date >= %(s)s", conn, params={"codes": codes, "s": DAILY_RECENT_FROM})
    sd["code"] = sd["code"].str.strip()
    print(f"일봉 {len(daily):,} 행 · 수정주가 {len(adj):,} · 수급 {len(sd):,} ({time.time()-t0:.0f}s)",
          file=sys.stderr)
    return first, daily, stocks, delisted, adj, sd


def prepare_daily(first, daily, stocks, delisted, adj, sd):
    """`newlisting_events.compute` 로 유니버스·일봉 이벤트를 만들고 드리프트·피처 3·8·익일 시가를 붙인다."""
    f = classify(first, stocks, delisted)
    keep = set(f.loc[f["keep"], "code"])
    w = compute(daily, keep)                       # d < N_DAYS 행만. 원주가.
    w["date"] = pd.to_datetime(w["date"])
    w = w.sort_values(["code", "date"]).reset_index(drop=True)
    # 익일 시가 (B 주 규칙) — 같은 종목 다음 일봉 행. d=59 의 익일은 d=60 이라 w 밖 → 원본에서
    d = daily.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["n_open"] = d.groupby("code", sort=False)["open"].shift(-1)
    w = w.merge(d[["code", "date", "n_open"]], on=["code", "date"], how="left")
    # D0 종가 위치 (피처 3) — D0 행 값을 종목 전체에 전파. D0 이벤트 자신은 미래 정보 → 0
    rng = (w["high"] - w["low"]).replace(0, np.nan)
    w["cp"] = ((w["close"] - w["low"]) / rng).fillna(0.5)
    w["d0_close_pos_stock"] = w.groupby("code", sort=False)["cp"].transform("first")
    w["d0_close_pos"] = np.where(w["d"] == 0, 0.0, w["d0_close_pos_stock"])
    # 수정/원주가 드리프트 (해석 C)
    a = adj.copy()
    a["date"] = pd.to_datetime(a["date"])
    w = w.merge(a.rename(columns={"close": "close_adj"}), on=["code", "date"], how="left")
    w["factor"] = w["close"] / w["close_adj"]
    w["factor_prev"] = w.groupby("code", sort=False)["factor"].shift(1)
    w["adj_drift"] = ((w["factor"] / w["factor_prev"] - 1).abs() > ADJ_DRIFT_TOL).fillna(False)
    # 수급 피처 8 — D−1 행 (해석 F)
    s = sd.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["fi_net"] = (s["foreign_"].astype(float) + s["institution"].astype(float)) / \
        s["acc_trde_qty"].astype(float).replace(0, np.nan)
    w = w.merge(s[["code", "date", "fi_net"]], on=["code", "date"], how="left")
    w["fi_net_prev_raw"] = w.groupby("code", sort=False)["fi_net"].shift(1)
    w["fi_net_prev"] = np.where(w["d"] == 0, 0.0, w["fi_net_prev_raw"].fillna(0.0))
    w["fi_missing"] = (w["d"] >= 1) & w["fi_net_prev_raw"].isna()
    return f, w


# ====================================================================================
# 분봉 → 이벤트
# ====================================================================================
def _new_cnt() -> dict:
    return {k: 0 for k in (
        "univ_days_with_bars", "drop_open_mismatch", "open_vi_days", "drop_no1530", "drop_scale", "drop_drift",
        "A_cand_days", "A_gap", "A_no_break", "A_trig_0900", "A_trig_1500plus", "A_locked",
        "B_cand_days", "B_open_above", "B_no_break", "B_eq_touch", "B_trig_0900", "B_trig_1500plus",
        "B_cumtv_short", "B_locked", "B_no_next_open", "B_eq_only")}


def events_from_batch(bars: pd.DataFrame, key: pd.DataFrame, cnt: dict, why: dict) -> tuple[list, list]:
    """유니버스 (종목,일자) 마다 (A)·(B) 이벤트를 만든다. `why[(code,date)] = {"A": 사유, "B": 사유}`
    는 일봉 주 이벤트가 분봉 단계에서 빠진 이유를 되짚기 위한 기록이다."""
    b = bars.copy()
    b["date"] = b["ts"].dt.normalize()
    b["mclk"] = b["ts"].dt.hour * 60 + b["ts"].dt.minute - 540
    b["slot"] = _slot(b)
    b = b[b["slot"] >= 0]
    ub = b.merge(key, on=["code", "date"], how="inner")
    if ub.empty:
        return [], []
    ub = ub.sort_values(["code", "date", "slot"]).reset_index(drop=True)
    ev_a, ev_b = [], []
    for (code, dt), grp in ub.groupby(["code", "date"], sort=False):
        cnt["univ_days_with_bars"] += 1
        row = grp.iloc[0]
        slot = grp["slot"].to_numpy()
        wd = why.setdefault((code, dt), {})
        if abs(float(grp["open"].iloc[0]) / float(row["open_day"]) - 1) > SCALE_TOL:
            cnt["drop_open_mismatch"] += 1
            wd["A"] = wd["B"] = "첫 봉 시가 ≠ 일봉 시가"
            continue
        cnt["open_vi_days"] += int(slot[0] != 0)      # 개장 VI 등으로 첫 봉이 09:00 이 아닌 날 (포함)
        auc = grp[grp["slot"] == AUCTION_SLOT]
        if auc.empty:
            cnt["drop_no1530"] += 1
            wd["A"] = wd["B"] = "15:30 봉 없음"
            continue
        close1530 = float(auc["close"].iloc[0])
        if abs(close1530 / float(row["close_day"]) - 1) > SCALE_TOL:
            cnt["drop_scale"] += 1
            wd["A"] = wd["B"] = "원주가 불일치"
            continue
        if bool(row["adj_drift"]):
            cnt["drop_drift"] += 1
            wd["A"] = wd["B"] = "수정주가 드리프트"
            continue
        cont = grp[grp["slot"] <= CONT_LAST].reset_index(drop=True)
        hi = cont["high"].to_numpy(dtype=float)
        lo = cont["low"].to_numpy(dtype=float)
        cl = cont["close"].to_numpy(dtype=float)
        mclk = cont["mclk"].to_numpy()
        cslot = cont["slot"].to_numpy()
        cum_tv = np.cumsum(cont["trade_value"].to_numpy(dtype=float)) / MINUTE_TV_DIV
        open0 = float(cont["open"].iloc[0])
        d = int(row["d"])
        prev_close = float(row["prev_close"]) if d >= 1 else np.nan
        base = {
            "code": code, "date": dt, "d": d, "listing_date": pd.Timestamp(row["listing_date"]),
            "d0_open": float(row["d0_open"]), "open0": open0, "close1530": close1530,
            "prev_close": prev_close, "adv_hist": float(row["adv_hist"]) if d >= 1 else np.nan,
            "d0_close_pos": float(row["d0_close_pos"]), "fi_net_prev": float(row["fi_net_prev"]),
            "fi_missing": bool(row["fi_missing"]), "tv_day": float(row["tv_day"]),
            "n_open": float(row["n_open"]) if pd.notna(row["n_open"]) else np.nan,
            "A_daily": bool(row["A"]), "B_daily": bool(row["B"]),
        }

        # ---------- (A) 신규주 돌파
        if d >= 1 and pd.notna(row["prior_max_high"]) and row["adv_hist"] >= TV_FLOOR_MN:
            cnt["A_cand_days"] += 1
            line = float(row["prior_max_high"])
            if open0 > line:
                cnt["A_gap"] += 1
                wd["A"] = "갭돌파"
            else:
                idx = np.flatnonzero(hi > line)
                if len(idx) == 0:
                    cnt["A_no_break"] += 1
                    wd["A"] = "분봉 미돌파"
                else:
                    i = int(idx[0])
                    if i == 0 or cslot[i] < TRIG_LO:
                        cnt["A_trig_0900"] += 1
                        wd["A"] = "개장 봉 트리거"
                    elif mclk[i] > TRIG_HI:
                        cnt["A_trig_1500plus"] += 1
                        wd["A"] = "15:00 이후 트리거"
                    else:
                        entry = cl[i]
                        if entry >= prev_close * LIMIT_RATIO:
                            cnt["A_locked"] += 1
                            wd["A"] = "상한가 잠김"
                        else:
                            wd["A"] = "이벤트"
                            after_lo, after_hi = lo[i + 1:], hi[i + 1:]
                            sp = np.flatnonzero(after_lo <= entry * STOP)
                            tp = np.flatnonzero(after_hi >= entry * TARGET)
                            has_s, has_t = len(sp) > 0, len(tp) > 0
                            r_close = close1530 / entry - 1
                            ret_stop2 = STOP - 1 if has_s else r_close
                            tgt_first = has_t and (not has_s or tp[0] < sp[0])
                            ret_gmh = (TARGET - 1) if tgt_first else ((STOP - 1) if has_s else r_close)
                            lim = limit_up_price(prev_close)
                            ev_a.append({**base, "line": line, "entry": entry, "touch_minute": int(mclk[i]),
                                         "cum_tv": float(cum_tv[i]), "ret_stop2": ret_stop2,
                                         "ret_gmh": ret_gmh, "ret_none": r_close,
                                         "label": int(tgt_first), "hit_stop": int(has_s), "hit_tgt": int(has_t),
                                         "is_limit_close": int(close1530 >= lim - 1e-6),
                                         "limit_undet": False, "close_eq_high": int(close1530 >= hi.max())})

        # ---------- (B) 신규주 상따
        ref = open0 if d == 0 else prev_close
        cnt["B_cand_days"] += 1
        if hi[0] > ref * SANGTTA_RATIO:
            cnt["B_open_above"] += 1
            wd["B"] = "개장 봉 이미 위"
        else:
            eq = bool(np.any(np.isclose(hi, ref * SANGTTA_RATIO)))
            cnt["B_eq_touch"] += int(eq)
            idx = np.flatnonzero(hi > ref * SANGTTA_RATIO)
            if len(idx) == 0:
                cnt["B_no_break"] += 1
                cnt["B_eq_only"] += int(eq)
                wd["B"] = "분봉 미돌파 (== 만 닿음)" if eq else "분봉 미돌파"
            else:
                i = int(idx[0])
                if i == 0 or cslot[i] < TRIG_LO:
                    cnt["B_trig_0900"] += 1
                    wd["B"] = "개장 봉 트리거"
                elif mclk[i] > TRIG_HI:
                    cnt["B_trig_1500plus"] += 1
                    wd["B"] = "15:00 이후 트리거"
                elif cum_tv[i] < TV_FLOOR_MN:
                    cnt["B_cumtv_short"] += 1
                    wd["B"] = "누적 50억 미달"
                else:
                    entry = cl[i]
                    lock_applies = d >= 1 or base["listing_date"] < pd.Timestamp(D0_RULE_CHANGE)
                    if lock_applies and entry >= ref * LIMIT_RATIO:
                        cnt["B_locked"] += 1
                        wd["B"] = "상한가 잠김"
                    else:
                        wd["B"] = "이벤트"
                        undet = (d == 0) and base["listing_date"] >= pd.Timestamp(D0_RULE_CHANGE)
                        if undet:
                            is_lim = np.nan
                        else:
                            is_lim = float(close1530 >= limit_up_price(ref) - 1e-6)
                        if np.isnan(base["n_open"]):
                            cnt["B_no_next_open"] += 1
                            wd["B"] = "익일 시가 없음"
                            continue
                        ev_b.append({**base, "ref": ref, "entry": entry, "touch_minute": int(mclk[i]),
                                     "cum_tv": float(cum_tv[i]),
                                     "ret_next": base["n_open"] / entry - 1,
                                     "ret_close": close1530 / entry - 1,
                                     "label": is_lim, "is_limit_close": is_lim,
                                     "limit_undet": bool(undet), "close_eq_high": int(close1530 >= hi.max())})
    return ev_a, ev_b


def finalize(ev: pd.DataFrame, key: pd.DataFrame, daily_flag: str) -> pd.DataFrame:
    if ev.empty:
        return ev
    ev = ev.sort_values(["code", "date"]).reset_index(drop=True)
    # 피처
    ev["d_listing"] = ev["d"].astype(float)
    ev["ret_vs_d0_open"] = ev["entry"] / ev["d0_open"] - 1
    ev["gap_open"] = np.where(ev["d"] == 0, 0.0, ev["open0"] / ev["prev_close"] - 1)
    ev["cum_tv_ratio_hist"] = np.where(ev["d"] == 0, 0.0, ev["cum_tv"] / ev["adv_hist"])
    ev["cum_tv_ratio_hist"] = pd.Series(ev["cum_tv_ratio_hist"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 독립 (해석 E): 기준 집합 = 일봉 주 이벤트 ∪ 분봉 이벤트, 같은 종목 d 차 > INDEP_GAP
    ref = pd.concat([key.loc[key[daily_flag], ["code", "d"]], ev[["code", "d"]]]).drop_duplicates()
    ref = ref.sort_values(["code", "d"])
    prev_d = {}
    for c, grp in ref.groupby("code"):
        ds = grp["d"].to_numpy()
        for j, dd in enumerate(ds):
            prev_d[(c, int(dd))] = int(ds[j - 1]) if j > 0 else None
    ev["indep"] = [int(prev_d.get((c, int(dd))) is None or dd - prev_d[(c, int(dd))] > INDEP_GAP)
                   for c, dd in zip(ev["code"], ev["d"])]
    ev["month"] = ev["date"].dt.strftime("%Y-%m")
    ev["quarter"] = ev["date"].dt.to_period("Q").astype(str)
    ev["seg"] = np.where(ev["d"] == 0, "D0", np.where(ev["d"] <= 20, "D1~D20", f"D21~D{N_DAYS-1}"))
    return ev


# ====================================================================================
# 통계·판정
# ====================================================================================
def fold_table(ev: pd.DataFrame, col: str) -> tuple[list[str], list[str], int]:
    lines = ["| 분기 | n | 평균 | net34 | 승률 | 유효(≥20) | 음수 |", "|---|---:|---:|---:|---:|---|---|"]
    valid, neg = [], 0
    for q, gq in ev.groupby("quarter"):
        r = gq[col].dropna()
        ok = len(r) >= FOLD_MIN_EVENTS
        is_neg = ok and (r.mean() - COST) < 0
        if ok:
            valid.append(q)
        neg += int(is_neg)
        lines.append(f"| {q} | {len(r):,} | {r.mean():+.2%} | {r.mean()-COST:+.2%} | {(r>0).mean():.1%} | "
                     f"{'O' if ok else '—'} | {'음수' if is_neg else ''} |")
    return lines, valid, neg


def conditions(ev: pd.DataFrame, col: str, with_limit_cond: bool) -> tuple[dict, list[str], int]:
    r = ev[col].dropna()
    _, valid, neg = fold_table(ev, col)
    ind = ev.loc[ev["indep"] == 1, col].dropna()
    cond = {
        "1 net34>0": r.mean() - COST > 0,
        f"2 음수 유효폴드<={MAX_NEG_FOLDS}": neg <= MAX_NEG_FOLDS,
        "3 상위1%제거 net34>0": r[r <= r.quantile(0.99)].mean() - COST > 0,
        "4 독립 net34>0": (ind.mean() - COST > 0) if len(ind) else False,
        "5 net55>0": r.mean() - COST_HI > 0,
    }
    if with_limit_cond:
        nl = ev.loc[ev["is_limit_close"] != 1, col].dropna()
        cond["6 상한가마감 제외 net34>0"] = (nl.mean() - COST > 0) if len(nl) else False
    return cond, valid, neg


def _cond_line(cond: dict, valid: list[str], neg: int) -> str:
    return ("  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items())
            + f"  (음수 유효폴드 {neg}/{len(valid)}) → **{sum(cond.values())}/{len(cond)}**")


def decidable(ev: pd.DataFrame) -> tuple[bool, list[str], int]:
    q = ev.groupby("quarter").size()
    valid = [k for k, v in q.items() if v >= FOLD_MIN_EVENTS]
    tot = int(sum(q[k] for k in valid))
    return (len(valid) >= MIN_VALID_FOLDS and tot >= MIN_VALID_SUM), valid, tot


def auc_block(ev: pd.DataFrame, out: list[str], col: str) -> tuple[float, pd.DataFrame | None]:
    P = out.append
    x = ev.dropna(subset=["label"]).sort_values(["date", "code"]).reset_index(drop=True)
    P(f"### 예측 가능성 — GBM(max_depth=2, n_estimators=100), 시간순 2폴드 · 라벨 있는 n = {len(x):,} "
      f"(라벨 결측 {len(ev)-len(x):,})\n")
    if len(x) < AUC_MIN_N:
        P(f"n {len(x):,} < {AUC_MIN_N} → **AUC 미계산** (사전등록: 최소 n 150). 2단계 없음.\n")
        return np.nan, None
    x[FEATURES] = x[FEATURES].fillna(0.0)
    half = len(x) // 2
    tr, te = np.arange(half), np.arange(half, len(x))
    y = x.loc[te, "label"].astype(int)
    m = GradientBoostingClassifier(**GBM)
    m.fit(x.loc[tr, FEATURES], x.loc[tr, "label"].astype(int))
    p = m.predict_proba(x.loc[te, FEATURES])[:, 1]
    auc = roc_auc_score(y, p) if y.nunique() > 1 else np.nan
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
    P(f"- train {len(tr):,} ({x.loc[tr,'date'].min().date()}~{x.loc[tr,'date'].max().date()}) / "
      f"test {len(te):,} ({x.loc[te,'date'].min().date()}~{x.loc[te,'date'].max().date()})  "
      f"AUC **{auc:.3f}**  양성률 train {x.loc[tr,'label'].mean():.1%} test {y.mean():.1%}")
    P("- 중요도: " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    xo = x.loc[te].copy()
    xo["pred"] = p
    xo["q"] = pd.qcut(xo["pred"].rank(method="first"), min(5, len(xo)), labels=False) + 1
    P("- 예측 5분위(OOS, 1=낮음): 양성률 / 주 규칙 평균 / net34 / n")
    for q, gq in xo.groupby("q"):
        r = gq[col].dropna()
        P(f"  - Q{q}: {gq['label'].mean():.1%} / {r.mean():+.2%} / {r.mean()-COST:+.2%} / {len(gq)}")
    P("")
    return float(auc), xo


def report_technique(ev: pd.DataFrame, tag: str, col: str, cmp_col: str, cmp_name: str,
                     with_limit_cond: bool, cnt: dict, out: list[str]) -> str:
    P = out.append
    P(f"## {tag}\n")
    if ev.empty:
        P("이벤트 0건.\n")
        return "판정 불가 (표본 부족)"
    ok, valid, tot = decidable(ev)
    P("### 0단계 — 보고\n")
    per_day = ev.groupby("date").size()
    P(f"- **분봉 이벤트 n = {len(ev):,}** / {ev['code'].nunique()} 종목 / {ev['date'].nunique()} 거래일 · "
      f"하루 최대 {per_day.max()} · 독립 {int(ev['indep'].sum()):,}")
    P(f"- 일봉 주 이벤트와 같은 (종목,일자): {int(ev[('A_daily' if tag.startswith('(A)') else 'B_daily')].sum()):,}")
    seg = ev.groupby("seg").agg(n=("code", "size"), codes=("code", "nunique"),
                                lim=("is_limit_close", lambda s: int((s == 1).sum())),
                                und=("limit_undet", "sum"), ceh=("close_eq_high", "sum"))
    P("\n| 구간 | n | 종목 | 상한가 마감 | 판정 불가(D0) | 고가=종가(D0 대리) |\n|---|---:|---:|---:|---:|---:|")
    for s, r in seg.iterrows():
        P(f"| {s} | {r['n']} | {r['codes']} | {r['lim']} | {int(r['und'])} | {int(r['ceh']) if r['und'] else '—'} |")
    P(f"| 합계 | {len(ev)} | {ev['code'].nunique()} | {int((ev['is_limit_close']==1).sum())} | "
      f"{int(ev['limit_undet'].sum())} | |")
    P("\n| 월 | n | 종목 |\n|---|---:|---:|")
    for mth, gm in ev.groupby("month"):
        P(f"| {mth} | {len(gm)} | {gm['code'].nunique()} |")
    P("")
    n_lim = int((ev["is_limit_close"] == 1).sum())
    contrib = ev.loc[ev["is_limit_close"] == 1, col].sum() / len(ev)
    P(f"- 당일 상한가 마감: **{n_lim}** 건 ({n_lim/len(ev):.1%}) · 주 규칙 기여 {contrib:+.2%}p "
      f"(전체 평균 {ev[col].mean():+.2%} 중)")
    if tag.startswith("(A)"):
        P(f"- 상한가 잠김(진입 불가) 제외 {cnt['A_locked']} · 손절 도달 {ev['hit_stop'].mean():.1%} · "
          f"+2% 도달 {ev['hit_tgt'].mean():.1%} · +2% 먼저 {ev['label'].mean():.1%}")
    else:
        ov = ev[(ev["d"] >= 21) & (ev["tv_day"] >= TV_500)]
        P(f"- 상한가 잠김 제외 {cnt['B_locked']} · 누적 50억 미달 제외 {cnt['B_cumtv_short']} · "
          f"익일 시가 없음 제외 {cnt['B_no_next_open']} · 고가 == 기준선×1.25 정확히 같은 봉 있는 날 "
          f"{cnt['B_eq_touch']} (그중 == 만 닿고 넘지 못해 이벤트 아님 {cnt['B_eq_only']}; 해석 D)")
        P(f"- D21~59 중 당일 거래대금 ≥ 500억 (5차 표본 겹침): **{len(ov)}** 건 / {ov['code'].nunique()} 종목")
    P(f"- `fi_net_prev` 결측(0 채움, d≥1): {int(ev['fi_missing'].sum())}")
    P("\n피처 요약:\n\n| 피처 | 평균 | 중앙 | min | max |\n|---|---:|---:|---:|---:|")
    for f in FEATURES:
        s = ev[f]
        P(f"| {f} | {s.mean():.4g} | {s.median():.4g} | {s.min():.4g} | {s.max():.4g} |")
    P("")
    P(f"**판정 가능 여부:** 유효 분기 폴드(≥{FOLD_MIN_EVENTS}건) {len(valid)}개 {valid} · 합계 {tot} → "
      + ("**판정 가능**" if ok else f"**판정 불가 (표본 부족: 유효 폴드 ≥{MIN_VALID_FOLDS} & 합 ≥{MIN_VALID_SUM} 미달)**") + "\n")

    # ---- 1단계
    P("### 1단계 — 전부 진입 (주 규칙)\n")
    P(HDR)
    d_all = describe(ev[col])
    P(_row("전체", d_all))
    for s in ("D0", "D1~D20", f"D21~D{N_DAYS-1}"):
        sub = ev.loc[ev["seg"] == s, col]
        if len(sub):
            P(_row(s, describe(sub)))
    P(_row("독립 이벤트만", describe(ev.loc[ev["indep"] == 1, col])))
    P(_row("상한가 마감 제외", describe(ev.loc[ev["is_limit_close"] != 1, col])))
    P(_row("상한가 마감만", describe(ev.loc[ev["is_limit_close"] == 1, col])))
    P("")
    P(f"비용 4단계 (전체 평균): 0bp {d_all['mean']:+.2%} · 21bp {d_all['net21']:+.2%} · "
      f"34bp {d_all['net34']:+.2%} · 55bp {d_all['net55']:+.2%} · min {d_all['mn']:+.1%} max {d_all['mx']:+.1%}\n")
    lines, valid_q, neg = fold_table(ev, col)
    P("분기 폴드 (주 규칙):\n")
    out.extend(lines)
    P("")
    cond, _, neg = conditions(ev, col, with_limit_cond)
    stage1_pass = all(cond.values())
    P("**채택 조건 (주 규칙):** " + _cond_line(cond, valid_q, neg) + "\n")
    P(f"### 비교 1회 — {cmp_name} (판정 아님)\n")
    P(HDR)
    P(_row(cmp_name, describe(ev[cmp_col])))
    P(_row(f"{cmp_name} 상한가 마감 제외", describe(ev.loc[ev["is_limit_close"] != 1, cmp_col])))
    ccond, _, cneg = conditions(ev, cmp_col, with_limit_cond)
    P("")
    P(f"- {cmp_name} 조건(참고): " + _cond_line(ccond, valid_q, cneg) + "\n")

    auc, xo = auc_block(ev, out, col)

    # ---- 판정
    P("### 판정 (기계적)\n")
    if not ok:
        verdict = "판정 불가 (표본 부족)"
        P(f"유효 폴드 {len(valid)} · 합 {tot} → **{verdict}**. 조건 결과는 기록만 "
          f"({sum(cond.values())}/{len(cond)}). 전진 수집 후 같은 정의로 재측정.\n")
        return verdict
    if stage1_pass:
        verdict = "채택 (전진 기록 착수)"
        P(f"1단계 조건 전부 통과 → **{verdict}**. 2단계 생략.\n")
        return verdict
    P(f"1단계 **미달** ({sum(cond.values())}/{len(cond)}). ")
    if xo is None or not (auc >= AUC_GATE):
        verdict = "기각"
        P(f"AUC {'미계산' if np.isnan(auc) else f'{auc:.3f}'} → 2단계 없음 → **{verdict}**.\n")
        return verdict
    P(f"AUC {auc:.3f} ≥ {AUC_GATE} → 2단계.\n")
    P("### 2단계 — 선별 진입 (검증 폴드 안 예측 상위 1/3)\n")
    sel = xo.sort_values("pred", ascending=False).head(int(len(xo) * TOP_FRAC))
    P(HDR)
    P(_row("상위 1/3", describe(sel[col])))
    P(_row("상위 1/3 독립만", describe(sel.loc[sel["indep"] == 1, col])))
    P(_row("상위 1/3 상한가 마감 제외", describe(sel.loc[sel["is_limit_close"] != 1, col])))
    top1 = xo.sort_values(["date", "pred"], ascending=[True, False]).groupby("date").head(1)
    P(_row("참고: 하루 상위 1건", describe(top1[col])))
    P("")
    lines2, valid2, neg2 = fold_table(sel, col)
    out.extend(lines2)
    cond2, _, neg2 = conditions(sel, col, with_limit_cond)
    P("\n**2단계 채택 조건 (상위 1/3):** " + _cond_line(cond2, valid2, neg2) + "\n")
    verdict = "채택 (2단계, 전진 기록 착수)" if all(cond2.values()) else "기각"
    P(f"2단계 → **{verdict}**.\n")
    return verdict


def daily_reference_B(w: pd.DataFrame, out: list[str]) -> None:
    """(B) 11년 일봉 근사 — 진입 = 기준선×1.25 가정, 청산 = 익일 시가. 판정 아님."""
    P = out.append
    x = w[w["B"] & w["n_open"].notna()].copy()
    x["ret"] = x["n_open"] / (x["B_ref"] * SANGTTA_RATIO) - 1
    P("## (B) 참고 — 11년 일봉 근사 (진입 = 기준선×1.25 가정 → 익일 시가, 판정 아님)\n")
    d = describe(x["ret"])
    P(HDR)
    P(_row(f"전체 {x['date'].min().date()}~{x['date'].max().date()}", d))
    P(_row("D0 제외", describe(x.loc[x["d"] >= 1, "ret"])))
    P(_row("상한가 마감 제외(판정 가능분)", describe(x.loc[~x["limit_close"].astype(bool), "ret"])))
    P("\n| 연도 | n | 평균 | net34 | 승률 |\n|---|---:|---:|---:|---:|")
    neg = 0
    for y, gy in x.groupby("year"):
        r = gy["ret"]
        neg += int(r.mean() - COST < 0)
        P(f"| {y} | {len(r)} | {r.mean():+.2%} | {r.mean()-COST:+.2%} | {(r>0).mean():.1%} |")
    P(f"\n음수 연도 {neg}/{x['year'].nunique()}\n")


# ====================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--batch", type=int, default=15)
    args = ap.parse_args()
    t0 = time.time()
    _load_env()

    if args.cache and os.path.exists(args.cache):
        blob = pd.read_pickle(args.cache)
        ev_a, ev_b, w, cnt, meta = blob["ev_a"], blob["ev_b"], blob["w"], blob["cnt"], blob["meta"]
    else:
        c = connect()
        first, daily, stocks, delisted, adj, sd = load_all(c)
        f, w = prepare_daily(first, daily, stocks, delisted, adj, sd)
        key = w[(w["date"] >= pd.Timestamp(START)) & (w["date"] <= pd.Timestamp(END))].copy()
        key = key.rename(columns={"open": "open_day", "high": "high_day", "low": "low_day",
                                  "close": "close_day", "volume": "vol_day", "trade_value": "tv_day"})
        codes = sorted(key["code"].unique())
        cur = c.cursor()
        cur.execute("select distinct code from minute_bars where ts >= %s", (START,))
        mb_codes = {str(r[0]).strip() for r in cur.fetchall()}
        cur.close()
        codes_mb = [x for x in codes if x in mb_codes]
        print(f"유니버스 (종목,일자) {len(key):,} · 종목 {len(codes)} (분봉 있음 {len(codes_mb)}) · "
              f"일봉 주 이벤트 (A) {int(key['A'].sum())} (B) {int(key['B'].sum())} ({time.time()-t0:.0f}s)",
              file=sys.stderr)
        cnt = _new_cnt()
        why: dict = {}
        pa, pb = [], []
        nb = (len(codes_mb) + args.batch - 1) // args.batch
        for i in range(0, len(codes_mb), args.batch):
            chunk = codes_mb[i:i + args.batch]
            bars = load_minute_bars(c, START, END, codes=chunk)
            a, b = events_from_batch(bars, key[key["code"].isin(chunk)], cnt, why)
            pa.extend(a)
            pb.extend(b)
            print(f"  배치 {i//args.batch+1}/{nb}: {len(bars):,}행 → A {len(pa)} B {len(pb)} "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)
            del bars
        c.close()
        ev_a = finalize(pd.DataFrame(pa), key, "A")
        ev_b = finalize(pd.DataFrame(pb), key, "B")
        lost = {}
        for t in ("A", "B"):
            kk = key[key[t]]
            rs = [why.get((cd, dt), {}).get(t, "분봉 없음 (종목·일자)") for cd, dt in zip(kk["code"], kk["date"])]
            lost[t] = pd.Series(rs).value_counts().to_dict()
        meta = {"n_univ_days": len(key), "n_codes": len(codes), "n_codes_mb": len(codes_mb), "lost": lost,
                "A_daily": int(key["A"].sum()), "B_daily": int(key["B"].sum()),
                "A_daily_codes_nomb": int(key.loc[key["A"] & ~key["code"].isin(mb_codes)].shape[0]),
                "B_daily_codes_nomb": int(key.loc[key["B"] & ~key["code"].isin(mb_codes)].shape[0]),
                "run_at": time.strftime("%Y-%m-%d %H:%M"), "w_keep_codes": int(f["keep"].sum())}
        if args.cache:
            pd.to_pickle({"ev_a": ev_a, "ev_b": ev_b, "w": w, "cnt": cnt, "meta": meta}, args.cache)

    out: list[str] = []
    P = out.append
    P("# 신규주 검증 — 사전등록 15 · **확정 측정** (분봉 커버리지 99%)\n")
    P(f"실행 {meta['run_at']} · 기간 {START}~{END} · 상장일 ≥ {LISTING_CUTOFF} · N={N_DAYS} · "
      f"신규주 유니버스 {meta['w_keep_codes']} 종목\n")
    P("## 표본\n")
    P("| 항목 | 값 |\n|---|---:|")
    P(f"| 유니버스 (종목,일자) d<{N_DAYS}, 최근 1년 | {meta['n_univ_days']:,} / {meta['n_codes']} 종목 |")
    P(f"| 그중 `minute_bars` 에 있는 종목 | {meta['n_codes_mb']} |")
    P(f"| 일봉 주 이벤트 (A) / (B) (재계산) | {meta['A_daily']} / {meta['B_daily']} |")
    P(f"| 일봉 주 이벤트 중 분봉 종목 없음 (A) / (B) | {meta['A_daily_codes_nomb']} / {meta['B_daily_codes_nomb']} |")
    P(f"| 유니버스 일자 중 분봉 있는 날 | {cnt['univ_days_with_bars']:,} |")
    P(f"| 일 단위 제외: 첫 봉 시가 ≠ 일봉 시가 / 15:30 봉 없음 / 원주가 불일치 / 수정주가 드리프트 | "
      f"{cnt['drop_open_mismatch']} / {cnt['drop_no1530']} / {cnt['drop_scale']} / {cnt['drop_drift']} |")
    P(f"| 첫 봉이 09:00 이 아닌 날 (개장 VI 등 — 첫 봉을 개장 봉으로 보고 포함, 해석 B) | {cnt['open_vi_days']} |")
    P(f"| (A) 후보일 (d≥1·adv≥50억) → 갭돌파 제외 / 분봉 미돌파 / 개장 봉 트리거 / 15:00 이후 / 잠김 | "
      f"{cnt['A_cand_days']} → {cnt['A_gap']} / {cnt['A_no_break']} / {cnt['A_trig_0900']} / "
      f"{cnt['A_trig_1500plus']} / {cnt['A_locked']} |")
    P(f"| (B) 후보일 → 개장 봉 이미 위 / 미돌파 / 개장 봉 트리거 / 15:00 이후 / 누적 50억 미달 / 잠김 / 익일 시가 없음 | "
      f"{cnt['B_cand_days']} → {cnt['B_open_above']} / {cnt['B_no_break']} / {cnt['B_trig_0900']} / "
      f"{cnt['B_trig_1500plus']} / {cnt['B_cumtv_short']} / {cnt['B_locked']} / {cnt['B_no_next_open']} |")
    P(f"| **분봉 이벤트 (A) / (B)** | **{len(ev_a)} / {len(ev_b)}** |")
    P("")
    P("일봉 주 이벤트 → 분봉 단계 결과 (사유별):\n")
    P("| 사유 | (A) | (B) |\n|---|---:|---:|")
    keys = sorted(set(meta["lost"]["A"]) | set(meta["lost"]["B"]), key=lambda k: (k != "이벤트", k))
    for k in keys:
        P(f"| {k} | {meta['lost']['A'].get(k, 0)} | {meta['lost']['B'].get(k, 0)} |")
    P("")
    va = report_technique(ev_a, "(A) 신규주 돌파 — 주 규칙 `손절2`", "ret_stop2", "ret_gmh",
                          "스승 +2/−2/종가", True, cnt, out)
    vb = report_technique(ev_b, "(B) 신규주 상따 — 주 규칙 익일 시가", "ret_next", "ret_close",
                          "당일 종가 청산", False, cnt, out)
    daily_reference_B(w, out)
    P("## 최종 판정\n")
    P(f"- (A) 신규주 돌파: **{va}**")
    P(f"- (B) 신규주 상따: **{vb}**")
    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"→ {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
