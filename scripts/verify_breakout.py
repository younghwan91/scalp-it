"""돌파 매매 검증 — 사전등록 `docs/research/manju/09-breakout-preregistration.md` 그대로.

  이벤트  직전 250거래일 최고가(수정주가, D−1) 를 원주가로 환산한 돌파선을
          분봉 고가가 **처음** 넘는 그 분 (strict >). 09:00 시가가 이미 위면 그날은 없음.
          트리거 09:01~14:59 만. 진입 = 트리거 분 종가.
          진입 분 종가 ≥ 전일종가 × 1.295 (상한가 잠김) 이면 제외, 건수 보고.
  유니버스  직전 20일 평균 거래대금 ≥ 50억(D−1) · 스팩 제외 · 250거래일 이력.
  청산 3종  주 규칙 `손절2`: 진입 후 분봉 저가 ≤ 진입가×0.98 → 그 가격 / 아니면 당일 종가(15:30).
            비교 1 `스승`: +2%/−2%/종가, 같은 분이면 손절 우선.
            비교 2 `무관리`: 종가만.
  피처 8    touch_minute · ret_at_trigger · cum_tv_ratio · burst5 · base_days · base_range20
            · mkt_ret · round_in_path. 추가·변경 금지.
  판정      0단계 보고 → 1단계 전부 진입(채택 5조건) + AUC → (실패 & AUC ≥ 0.60 일 때만) 2단계.

문서 외 파라미터·변형·조합은 돌리지 않는다. 재사용: `breakout_events.compute`(일봉 이벤트·
유니버스), `limitup_timing.limit_up_price`(호가단위 상한가), `scalp_it` 로더.

문서가 모호한 지점 — 택한 (보수) 해석:
  A. 손절·목표 판정은 트리거 분 **다음 봉부터** 15:19 까지의 접속매매 봉으로만 한다. 트리거 분의
     저가는 진입(그 분 종가) 이전에 찍힌 값이라 손절 판정에 쓸 수 없다. 15:30 단일가 봉은 손절
     판정에 쓰지 않고 종가 청산가로만 쓴다 (그 봉 종가가 손절선 아래면 손절가가 아니라 종가로
     나간다 — 더 낮은 값).
  B. "전일종가" 는 일봉 수정주가 전일종가 × (당일 원주가/수정주가 비율) 을 정수로 반올림한 값
     (권리락 당일은 기준가가 된다). 돌파선도 같은 비율로 환산한다 (문서 식).
  C. 15:30 봉이 없거나 15:30 봉 종가가 `daily_bars` 원주가 종가와 0.2% 넘게 어긋나는
     (종목,일자) 는 제외하고 건수를 보고한다 — 분봉이 수정주가인 종목·거래정지 등. 개장 봉(09:00,
     특수일 10:00)이 없는 날도 시가를 알 수 없으므로 제외·보고.
  D. `burst5` 는 트리거 봉에서 5봉 앞 봉의 종가. 그 봉이 없으면(09:01~09:05 트리거) 전일종가(B).
  E. `mkt_ret` 의 "분봉 보유 유니버스" = 이번 실행에서 읽은 분봉 종목 전부(거래대금·스팩 필터
     없음). 종목별 전일 마지막 봉 종가 대비 트리거 분 종가(직전 봉 전진 채움) 의 동일가중 평균.
  F. "독립 이벤트" 의 기준 집합 = 일봉 주 이벤트(장중 돌파 ∧ 유니버스) ∪ 분봉 이벤트. 같은
     종목에서 직전 5거래일(그 종목 일봉 행 기준) 안에 기준 집합 이벤트가 없으면 독립.
  G. 월별 폴드는 **이번 표본에서** 이벤트 ≥ 100 건인 달만 센다. 음수 월 = 34bp 차감 후 평균 < 0.
  H. 당일 상한가 마감 = 15:30 종가 ≥ `limit_up_price(전일종가)` (호가단위 내림). 분리 보고만.
  I. AUC 라벨 = "+2% 가 −2% 보다 먼저 닿음" — 스승 규칙과 같은 판정(같은 분이면 손절 → 0).
     시간순 3폴드(날짜 정렬 3등분, 앞 폴드 학습 → 다음 폴드 검증, 폴드 1·2 AUC 평균).
  J. 2단계 "폴드 안에서 상위 1/3" = 검증 폴드(1·2) 각각에서 예측 점수 상위 floor(n/3).
     5조건의 월별 폴드는 1단계에서 정한 달 집합 중 선별 이벤트가 있는 달로 센다.

실행:
    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/verify_breakout.py --cache /path/ev.pkl --out /path/report.md
DB 는 읽기만 한다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from breakout_events import INDEP_GAP, LOOKBACK
from breakout_events import compute as daily_compute
from limitup_timing import limit_up_price

from scalp_it.adapters.db import connect, load_ohlcv
from scalp_it.intraday.loader import available_codes, load_minute_bars

# ---- 사전 고정 (문서와 동일) -----------------------------------------------------------
FEATURES = ["touch_minute", "ret_at_trigger", "cum_tv_ratio", "burst5",
            "base_days", "base_range20", "mkt_ret", "round_in_path"]
START, END = "2025-08-01", "2026-08-21"
DAILY_LEAD = "2024-06-01"            # 250거래일 선행
STOP, TARGET = 0.98, 1.02
COSTS = {"net0": 0.0, "net21": 0.0021, "net34": 0.0034, "net55": 0.0055}
COST, COST_HI = COSTS["net34"], COSTS["net55"]
LIMIT_AT_ENTRY = 1.295
TRIG_LO, TRIG_HI = 1, 359            # 개장 봉 제외(슬롯 ≥ 1) ~ 14:59 (09:00 기준 경과 분 ≤ 359)
CONT_LAST = 379                      # 15:19 마지막 접속매매 봉
AUCTION_SLOT = 380                   # 15:30 단일가 봉
N_SLOTS = 381
MINUTE_TV_DIV = 1e6                  # minute_bars.trade_value 원 → daily 백만원
SCALE_TOL = 0.002                    # 15:30 봉 종가 vs daily_bars 원주가 종가 허용 괴리
FOLD_MIN_EVENTS = 100
MAX_NEG_MONTHS = 3
AUC_GATE = 0.60
TOP_FRAC = 1 / 3
GBM = {"max_depth": 2, "n_estimators": 100, "random_state": 0}


# ====================================================================================
# 일봉 준비
# ====================================================================================
def prepare_daily(daily: pd.DataFrame, raw_close: pd.DataFrame, spac: set[str]) -> pd.DataFrame:
    """유니버스 (종목,일자) 마다 돌파선·환산계수·피처 5·6·독립성 계산용 열을 만든다.

    `breakout_events.compute` 를 그대로 써서 돌파선(prior_high)·adv20·갭/장중 구분·유니버스를
    얻고, 여기서는 전일종가·base_days·base_range20·원주가 환산만 더한다.
    """
    d = daily.sort_values(["code", "date"]).reset_index(drop=True)
    w = daily_compute(d.assign(date=d["date"].dt.date))
    w["date"] = pd.to_datetime(w["date"])
    # compute() 는 START~END 로 잘라 돌려준다. 이력 열은 전체 d 에서 계산해 붙인다.
    g = d.groupby("code", sort=False)
    d["prev_close_adj"] = g["close"].shift(1)
    hi20 = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    lo20 = g["low"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).min())
    d["base_range20"] = (hi20 - lo20) / lo20
    d["rn"] = g.cumcount()
    # base_days — 직전 250일 최고가를 만든 날(동률이면 가장 최근)로부터 경과 거래일
    d["base_days"] = np.nan
    codes_arr = d["code"].to_numpy()
    high_arr = d["high"].to_numpy(dtype=float)
    bd = np.full(len(d), np.nan)
    starts = np.r_[0, np.flatnonzero(codes_arr[1:] != codes_arr[:-1]) + 1, len(d)]
    in_range = (d["date"] >= pd.Timestamp(START)).to_numpy()
    for s, e in pairwise(starts):
        for i in range(s + LOOKBACK, e):
            if not in_range[i]:
                continue
            win = high_arr[i - LOOKBACK:i]
            j = LOOKBACK - 1 - int(np.argmax(win[::-1]))   # 동률 → 가장 최근
            bd[i] = LOOKBACK - j
    d["base_days"] = bd
    extra = d[["code", "date", "prev_close_adj", "base_range20", "rn", "base_days", "close"]]
    w = w.merge(extra.rename(columns={"close": "close_adj"}), on=["code", "date"], how="left")
    w = w.merge(raw_close.rename(columns={"close": "close_raw"}), on=["code", "date"], how="left")
    w = w.merge(raw_close.rename(columns={"close": "prev_close_rawtbl"}).assign(
        date=lambda x: x["date"]),  # 자리만; 아래서 shift
        on=["code", "date"], how="left")
    w = w.sort_values(["code", "date"]).reset_index(drop=True)
    w["prev_close_rawtbl"] = w.groupby("code", sort=False)["close_raw"].shift(1)
    w["factor"] = w["close_raw"] / w["close_adj"]
    w["line_raw"] = w["prior_high"] * w["factor"]
    w["prev_close_raw"] = np.round(w["prev_close_adj"] * w["factor"])
    w["univ"] = (w["has_hist"] & w["in_univ"] & ~w["code"].isin(spac)
                 & w["prior_high"].notna() & w["factor"].notna() & w["prev_close_raw"].notna())
    w["ev_daily"] = w["univ"] & w["ev_intraday"]          # 일봉 주 이벤트 (문서 표의 5,823)
    return w


def _key_frame(w: pd.DataFrame) -> pd.DataFrame:
    cols = ["code", "date", "line_raw", "prev_close_raw", "prev_close_rawtbl", "adv20",
            "base_days", "base_range20", "close_raw", "rn", "ev_daily", "factor"]
    return w.loc[w["univ"], cols].reset_index(drop=True)


# ====================================================================================
# 분봉 배치 처리
# ====================================================================================
def _slot(b: pd.DataFrame) -> np.ndarray:
    """봉 → 슬롯. 접속매매 0..379 (개장 봉 = 0), 종가 단일가 = 380, 그 외(15:35 시간외 등) = −1.

    KRX 특수일 처리: 수능일(10:00 개장·16:30 마감)·연초 개장일(10:00 개장·15:30 마감) 은 그날
    종목들의 첫 봉·마지막 봉 최빈 분으로 개장·마감 분을 정한다 (개장 540|600, 마감 930|990 두 값뿐).
    """
    m = (b["ts"].dt.hour * 60 + b["ts"].dt.minute).to_numpy()
    per = pd.DataFrame({"date": b["date"].to_numpy(), "code": b["code"].to_numpy(), "m": m}) \
        .groupby(["date", "code"])["m"].agg(["min", "max"]).reset_index()
    fm = per.groupby("date")["min"].agg(lambda s: s.mode().iloc[0])
    lm = per.groupby("date")["max"].agg(lambda s: s.mode().iloc[0])
    open_m = b["date"].map(pd.Series(np.where(fm >= 600, 600, 540), index=fm.index)).to_numpy()
    close_m = b["date"].map(pd.Series(np.where(lm >= 990, 990, 930), index=lm.index)).to_numpy()
    rel = m - open_m
    cont = (rel >= 0) & (m <= close_m - 11)
    return np.where(cont, rel, np.where(m == close_m, AUCTION_SLOT, -1))


def accumulate_market(b: pd.DataFrame, acc: dict) -> None:
    """분봉 보유 유니버스 동일가중 등락률(전일 마지막 봉 종가 → 각 분) 의 합·개수를 누적한다."""
    last = b.groupby(["code", "date"], sort=True)["close"].last().reset_index()
    last["prev"] = last.groupby("code", sort=False)["close"].shift(1)
    last = last.dropna(subset=["prev"]).reset_index(drop=True)
    last = last[last["prev"] > 0]
    if last.empty:
        return
    last["row"] = np.arange(len(last))
    bb = b.merge(last[["code", "date", "row"]], on=["code", "date"], how="inner")
    mat = np.full((len(last), N_SLOTS), np.nan)
    mat[bb["row"].to_numpy(), bb["slot"].to_numpy()] = bb["close"].to_numpy(dtype=float)
    idx = np.where(~np.isnan(mat), np.arange(N_SLOTS)[None, :], 0)
    idx = np.maximum.accumulate(idx, axis=1)
    filled = mat[np.arange(len(last))[:, None], idx]
    ret = filled / last["prev"].to_numpy()[:, None] - 1
    di = last["date"].map(acc["date_idx"])
    ok = di.notna().to_numpy()
    di = di[ok].astype(int).to_numpy()
    ret = ret[ok]
    np.add.at(acc["sum"], di, np.nan_to_num(ret))
    np.add.at(acc["cnt"], di, (~np.isnan(ret)).astype(np.int64))


def _round_in_path(entry: np.ndarray) -> np.ndarray:
    """1유효숫자 라운드피겨(10^k × 1~9) 가 (entry, entry×1.02] 에 있는가."""
    out = np.zeros(len(entry), dtype=int)
    for i, e in enumerate(entry):
        if not np.isfinite(e) or e <= 0:
            continue
        hi = e * TARGET
        k = math.floor(math.log10(hi))
        for kk in (k - 1, k):
            base = 10.0 ** kk
            for dgt in range(1, 10):
                rf = dgt * base
                if e < rf <= hi + 1e-9:
                    out[i] = 1
                    break
            if out[i]:
                break
    return out


def events_from_batch(bars: pd.DataFrame, key: pd.DataFrame, acc: dict, cnt: dict) -> pd.DataFrame:
    b = bars.copy()
    b["date"] = b["ts"].dt.normalize()
    b["mclk"] = b["ts"].dt.hour * 60 + b["ts"].dt.minute - 540     # 09:00 기준 경과 분 (시계)
    b["slot"] = _slot(b)
    b = b[b["slot"] >= 0]
    accumulate_market(b, acc)

    ub = b.merge(key, on=["code", "date"], how="inner")
    if ub.empty:
        return pd.DataFrame()
    ub = ub.sort_values(["code", "date", "slot"]).reset_index(drop=True)
    ub["gid"] = ub.groupby(["code", "date"], sort=False).ngroup()
    ub["pos"] = ub.groupby("gid", sort=False).cumcount()
    g = ub.groupby("gid", sort=False)
    info = g.agg(code=("code", "first"), date=("date", "first"), first_slot=("slot", "first"),
                 open0=("open", "first"), line_raw=("line_raw", "first"),
                 prev_close_raw=("prev_close_raw", "first"), close_raw=("close_raw", "first"),
                 ev_daily=("ev_daily", "first"))
    c1530 = ub.loc[ub["slot"] == AUCTION_SLOT].drop_duplicates("gid").set_index("gid")["close"]
    info["close1530"] = c1530.reindex(info.index)
    cnt["univ_days_with_bars"] += len(info)
    cnt["daily_main_with_bars"] += int(info["ev_daily"].sum())

    ok = info["first_slot"] == 0
    cnt["drop_no0900"] += int((~ok).sum())
    info = info[ok]
    ok = info["close1530"].notna()
    cnt["drop_no1530"] += int((~ok).sum())
    info = info[ok]
    ok = (info["close1530"] / info["close_raw"] - 1).abs() <= SCALE_TOL
    cnt["drop_scale"] += int((~ok).sum())
    info = info[ok]
    gap = info["open0"] > info["line_raw"]
    cnt["gap_break"] += int(gap.sum())
    info = info[~gap]
    cnt["daily_main_after_dayfilters"] += int(info["ev_daily"].sum())

    ub = ub[ub["gid"].isin(info.index)]
    hits = ub[ub["high"] > ub["line_raw"]].groupby("gid", sort=False).head(1)
    cnt["intraday_break_any_minute"] += len(hits)
    cnt["daily_main_no_minute_break"] += int(info.loc[~info.index.isin(hits["gid"]), "ev_daily"].sum())
    early = hits["slot"] < TRIG_LO            # 개장 봉(시가 동시호가 포함)
    late = hits["mclk"] > TRIG_HI             # 15:00 이후 (시계 기준)
    cnt["trig_0900"] += int(early.sum())
    cnt["trig_1500plus"] += int(late.sum())
    hits = hits[~early & ~late]
    locked = hits["close"] >= hits["prev_close_raw"] * LIMIT_AT_ENTRY
    cnt["limit_at_entry"] += int(locked.sum())
    hits = hits[~locked].copy()
    if hits.empty:
        return pd.DataFrame()

    ev = hits.rename(columns={"close": "entry", "mclk": "touch_minute", "slot": "trig_slot",
                              "pos": "trig_pos"})
    ev["cum_tv"] = ub.groupby("gid", sort=False)["trade_value"].cumsum().loc[ev.index] / MINUTE_TV_DIV
    ev = ev[["gid", "code", "date", "entry", "touch_minute", "trig_slot", "trig_pos", "cum_tv", "line_raw",
             "prev_close_raw", "prev_close_rawtbl", "adv20", "base_days", "base_range20",
             "close_raw", "rn", "ev_daily", "factor"]].set_index("gid")
    ev["close1530"] = info["close1530"].reindex(ev.index)

    # --- 청산: 트리거 다음 봉부터 15:19 까지 (해석 A)
    x = ub[ub["gid"].isin(ev.index)].merge(
        ev[["entry", "trig_pos"]], left_on="gid", right_index=True)
    after = (x["pos"] > x["trig_pos"]) & (x["slot"] <= CONT_LAST)
    stop_pos = x[after & (x["low"] <= x["entry"] * STOP)].groupby("gid")["pos"].min()
    tgt_pos = x[after & (x["high"] >= x["entry"] * TARGET)].groupby("gid")["pos"].min()
    ev["stop_pos"] = stop_pos.reindex(ev.index)
    ev["tgt_pos"] = tgt_pos.reindex(ev.index)
    # burst5 — 5봉 앞 종가, 없으면 전일종가 (해석 D)
    b5 = x[x["pos"] == x["trig_pos"] - 5].set_index("gid")["close"]
    ev["ref5"] = b5.reindex(ev.index)
    ev["ref5"] = ev["ref5"].fillna(ev["prev_close_raw"])

    r_close = ev["close1530"] / ev["entry"] - 1
    has_stop = ev["stop_pos"].notna()
    has_tgt = ev["tgt_pos"].notna()
    ev["ret_stop2"] = np.where(has_stop, STOP - 1, r_close)
    tgt_first = has_tgt & (~has_stop | (ev["tgt_pos"] < ev["stop_pos"]))
    ev["ret_gmh"] = np.where(has_stop & ~tgt_first, STOP - 1,
                             np.where(tgt_first, TARGET - 1, r_close))
    ev["ret_none"] = r_close
    ev["label_tgt_first"] = tgt_first.astype(int)
    ev["hit_stop"] = has_stop.astype(int)
    ev["hit_target"] = has_tgt.astype(int)
    # --- 피처
    ev["ret_at_trigger"] = ev["entry"] / ev["prev_close_raw"] - 1
    ev["cum_tv_ratio"] = ev["cum_tv"] / ev["adv20"]
    ev["burst5"] = ev["entry"] / ev["ref5"] - 1
    ev["round_in_path"] = _round_in_path(ev["entry"].to_numpy(dtype=float))
    # --- 상한가 마감 (해석 H)
    lim = np.array([limit_up_price(p) for p in ev["prev_close_raw"].to_numpy(dtype=float)])
    ev["is_limit_close"] = (ev["close1530"].to_numpy(dtype=float) >= lim - 1e-6).astype(int)
    ev["prev_mismatch"] = ((ev["prev_close_raw"] / ev["prev_close_rawtbl"] - 1).abs() > SCALE_TOL)
    return ev.reset_index(drop=True)


# ====================================================================================
# 후처리: mkt_ret · 독립 이벤트
# ====================================================================================
def finalize(ev: pd.DataFrame, acc: dict, key: pd.DataFrame) -> pd.DataFrame:
    di = ev["date"].map(acc["date_idx"]).astype(int).to_numpy()
    si = ev["trig_slot"].astype(int).to_numpy()
    s, c = acc["sum"][di, si], acc["cnt"][di, si]
    ev["mkt_ret"] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    ev["mkt_n"] = c
    # 독립 이벤트 (해석 F)
    ref = pd.concat([key.loc[key["ev_daily"], ["code", "rn"]], ev[["code", "rn"]]]) \
        .drop_duplicates().sort_values("rn", kind="stable").reset_index(drop=True)
    ref["rn_ref"] = ref["rn"]
    e = ev[["code", "rn"]].reset_index().sort_values("rn", kind="stable")
    m = pd.merge_asof(e, ref[["code", "rn", "rn_ref"]], on="rn", by="code",
                      allow_exact_matches=False, direction="backward")
    m = m.set_index("index").reindex(ev.index)
    gap = ev["rn"] - m["rn_ref"]
    ev["indep"] = (gap.isna() | (gap > INDEP_GAP)).astype(int)
    ev["month"] = ev["date"].dt.strftime("%Y-%m")
    return ev


# ====================================================================================
# 통계·보고
# ====================================================================================
def describe(r: pd.Series) -> dict:
    r = r.dropna()
    w, l = r[r > 0], r[r <= 0]
    pl = abs(w.mean() / l.mean()) if len(w) and len(l) and l.mean() != 0 else np.nan
    top = r[r <= r.quantile(0.99)] if len(r) else r
    return {"n": len(r), "mean": r.mean(), "net21": r.mean() - COSTS["net21"],
            "net34": r.mean() - COST, "net55": r.mean() - COST_HI, "median": r.median(),
            "win": (r > 0).mean(), "pl": pl, "p5": r.quantile(.05), "p25": r.quantile(.25),
            "p75": r.quantile(.75), "p95": r.quantile(.95), "mn": r.min(), "mx": r.max(),
            "trim1_net34": top.mean() - COST}


def _row(tag: str, d: dict) -> str:
    if d["n"] == 0:
        return f"| {tag} | 0 | | | | | | | | | |"
    return (f"| {tag} | {d['n']:,} | {d['mean']:+.2%} | {d['net34']:+.2%} | {d['net55']:+.2%} | "
            f"{d['median']:+.2%} | {d['win']:.1%} | {d['pl']:.2f} | {d['p5']:+.1%} / {d['p25']:+.1%} / "
            f"{d['p75']:+.1%} / {d['p95']:+.1%} | {d['trim1_net34']:+.2%} |")


HDR = ("| 구분 | n | 평균 | net34 | net55 | 중앙 | 승률 | 손익비 | p5/p25/p75/p95 | 상위1%제거 net34 |\n"
       "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")


def monthly_table(ev: pd.DataFrame, col: str, months: list[str]) -> tuple[list[str], int]:
    lines = ["| 월 | n | 평균 | net34 | 승률 | 폴드 |", "|---|---:|---:|---:|---:|---|"]
    neg = 0
    for m, gm in ev.groupby("month"):
        r = gm[col]
        inf = m in months
        is_neg = inf and (r.mean() - COST) < 0
        neg += int(is_neg)
        lines.append(f"| {m} | {len(r):,} | {r.mean():+.2%} | {r.mean()-COST:+.2%} | "
                     f"{(r>0).mean():.1%} | {'O' if inf else '—'}{' 음수' if is_neg else ''} |")
    return lines, neg


def five_conditions(ev: pd.DataFrame, col: str, months: list[str]) -> tuple[dict, int]:
    r = ev[col]
    _, neg = monthly_table(ev, col, months)
    ind = ev.loc[ev["indep"] == 1, col]
    cond = {
        "1 net34>0": r.mean() - COST > 0,
        f"2 음수월<={MAX_NEG_MONTHS}": neg <= MAX_NEG_MONTHS,
        "3 상위1%제거 net34>0": r[r <= r.quantile(0.99)].mean() - COST > 0,
        "4 독립 net34>0": (ind.mean() - COST > 0) if len(ind) else False,
        "5 net55>0": r.mean() - COST_HI > 0,
    }
    return cond, neg


def _cond_line(cond: dict, neg: int, n_months: int) -> str:
    return ("  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items())
            + f"  (음수 월 {neg}/{n_months}) → **{sum(cond.values())}/5**")


def analyze(ev: pd.DataFrame, meta: dict, out: list[str]) -> None:
    P = out.append
    n_daily = meta["n_daily_main"]
    n_cov = meta["cnt"]["daily_main_with_bars"]
    P(f"# 돌파 매매 검증 — 사전등록 09 · **예비 (표본 {n_cov/n_daily:.0%})**\n")
    P(f"실행 {meta['run_at']} · 기간 {START}~{END} · 분봉 종목 {meta['n_codes_loaded']} "
      f"(DB 종목 {meta['n_codes_db']}) · 캐시 {meta.get('cache') or '없음'}\n")
    P("## 표본\n")
    P("| 항목 | 값 |\n|---|---:|")
    P(f"| 일봉 주 이벤트 (장중 돌파 ∧ 유니버스, 재계산) | {n_daily:,} 건 / {meta['n_daily_codes']:,} 종목 |")
    P(f"| 그중 분봉이 있는 (종목,일자) | {n_cov:,} 건 (**{n_cov/n_daily:.1%}**) |")
    c = meta["cnt"]
    P(f"| 일 단위 제외: 개장 봉 없음 / 종가 봉 없음 / 원주가 스케일 불일치 | "
      f"{c['drop_no0900']:,} / {c['drop_no1530']:,} / {c['drop_scale']:,} |")
    P(f"| 갭돌파 (09:00 시가 > 돌파선) — 제외 | {c['gap_break']:,} |")
    P(f"| 분봉으로 돌파선을 넘은 날 (시간대 무관) | {c['intraday_break_any_minute']:,} |")
    P(f"| 일봉은 주 이벤트인데 분봉 고가가 돌파선을 안 넘은 날 (환산 오차) | {c['daily_main_no_minute_break']:,} |")
    P(f"| 트리거가 개장 봉 — 제외 / 15:00 이후 — 제외 | {c['trig_0900']:,} / {c['trig_1500plus']:,} |")
    P(f"| 진입 분 종가 ≥ 전일종가×1.295 (상한가 잠김) — 제외 | {c['limit_at_entry']:,} |")
    P(f"| **분봉 이벤트 (최종)** | **{len(ev):,} 건 / {ev['code'].nunique():,} 종목 / "
      f"{ev['date'].nunique()} 거래일** |")
    P(f"| 전일종가 환산값 vs daily_bars 원주가 전일종가 0.2% 초과 불일치 | {int(ev['prev_mismatch'].sum()):,} |")
    P(f"| mkt_ret 계산 종목 수 (트리거 분 기준, 중앙값) | {int(ev['mkt_n'].median())} |")
    P("")
    if len(ev) < 200:
        P("표본 부족(<200). 종료.")
        return
    per_day = ev.groupby("date").size()
    P(f"하루 건수: 중앙값 {per_day.median():.0f} · p90 {per_day.quantile(.9):.0f} · 최대 {per_day.max()}\n")

    # ---- 0단계
    P("## 0단계 — 보고만\n")
    months_all = ev.groupby("month").size()
    P("| 월 | 이벤트 | 종목 |\n|---|---:|---:|")
    for m, gm in ev.groupby("month"):
        P(f"| {m} | {len(gm):,} | {gm['code'].nunique():,} |")
    P("")
    n_lim = int(ev["is_limit_close"].sum())
    contrib = ev.loc[ev["is_limit_close"] == 1, "ret_stop2"].sum() / len(ev)
    P(f"- 상한가 잠김(진입 불가) 제외: **{c['limit_at_entry']:,}** 건")
    P(f"- `ret_at_trigger ≥ 20%`: **{int((ev['ret_at_trigger'] >= 0.20).sum()):,}** 건")
    P(f"- 당일 상한가 마감: **{n_lim:,}** 건 ({n_lim/len(ev):.1%}) · 주 규칙 수익 기여 "
      f"{contrib:+.2%}p (전체 평균 {ev['ret_stop2'].mean():+.2%} 중)")
    P(f"- 손절 도달 {ev['hit_stop'].mean():.1%} · +2% 도달 {ev['hit_target'].mean():.1%} · "
      f"+2% 가 먼저 {ev['label_tgt_first'].mean():.1%}")
    P("")
    P("피처 요약:\n")
    P("| 피처 | 평균 | 중앙 | min | max |\n|---|---:|---:|---:|---:|")
    for f in FEATURES:
        s = ev[f]
        P(f"| {f} | {s.mean():.4g} | {s.median():.4g} | {s.min():.4g} | {s.max():.4g} |")
    P("")

    # ---- 1단계
    months = sorted(m for m, n in months_all.items() if n >= FOLD_MIN_EVENTS)
    P("## 1단계 — 무조건 진입\n")
    P(f"월별 폴드 대상(≥{FOLD_MIN_EVENTS}건): {len(months)}개월 — {', '.join(months) or '없음'}\n")
    P("### 주 규칙 `손절2`\n")
    P(HDR)
    d_all = describe(ev["ret_stop2"])
    P(_row("전체", d_all))
    P(_row("독립 이벤트만", describe(ev.loc[ev["indep"] == 1, "ret_stop2"])))
    P(_row("상한가 마감 제외", describe(ev.loc[ev["is_limit_close"] == 0, "ret_stop2"])))
    P(_row("상한가 마감만", describe(ev.loc[ev["is_limit_close"] == 1, "ret_stop2"])))
    P(_row("상한가의심(ret_at_trigger≥20%) 제외", describe(ev.loc[ev["ret_at_trigger"] < 0.20, "ret_stop2"])))
    P("")
    P(f"비용 4단계 (전체 평균): 0bp {d_all['mean']:+.2%} · 21bp {d_all['net21']:+.2%} · "
      f"34bp {d_all['net34']:+.2%} · 55bp {d_all['net55']:+.2%} · min {d_all['mn']:+.1%} max {d_all['mx']:+.1%}\n")
    lines, neg = monthly_table(ev, "ret_stop2", months)
    P("월별 (주 규칙):\n")
    out.extend(lines)
    P("")
    cond, neg = five_conditions(ev, "ret_stop2", months)
    stage1_pass = all(cond.values())
    P("**채택 5조건 (주 규칙, 전부 진입):** " + _cond_line(cond, neg, len(months)) + "\n")
    P("### 비교 (각 1회, 채택 대상 아님)\n")
    P(HDR)
    P(_row("비교1 스승 +2/−2/종가", describe(ev["ret_gmh"])))
    P(_row("비교2 무관리 종가만", describe(ev["ret_none"])))
    P("")
    for tag, col in (("스승", "ret_gmh"), ("무관리", "ret_none")):
        cnd, ng = five_conditions(ev, col, months)
        P(f"- {tag} 5조건(참고): " + _cond_line(cnd, ng, len(months)))
    P("")

    # ---- AUC
    P("### 예측 가능성 — GBM(max_depth=2, n_estimators=100), 시간순 3폴드, 라벨 '+2% 먼저'\n")
    ev = ev.sort_values(["date", "code"]).reset_index(drop=True)
    x = ev.dropna(subset=FEATURES).reset_index(drop=True)
    P(f"피처 결측으로 제외 {len(ev)-len(x):,} 건 · 라벨 양성률 {x['label_tgt_first'].mean():.1%}\n")
    folds = np.array_split(np.arange(len(x)), 3)
    preds = np.full(len(x), np.nan)
    fold_id = np.zeros(len(x), dtype=int)
    aucs = []
    for i in (1, 2):
        tr, te = np.concatenate(folds[:i]), folds[i]
        fold_id[te] = i
        m = GradientBoostingClassifier(**GBM)
        m.fit(x.loc[tr, FEATURES], x.loc[tr, "label_tgt_first"])
        p = m.predict_proba(x.loc[te, FEATURES])[:, 1]
        preds[te] = p
        y = x.loc[te, "label_tgt_first"]
        a = roc_auc_score(y, p) if y.nunique() > 1 else np.nan
        aucs.append(a)
        imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
        P(f"- 폴드 {i}: train {len(tr):,} / test {len(te):,} "
          f"({x.loc[te,'date'].min().date()}~{x.loc[te,'date'].max().date()})  AUC **{a:.3f}**  "
          f"양성률 {y.mean():.1%}  중요도: " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    auc = float(np.nanmean(aucs))
    P(f"\n**평균 AUC {auc:.3f}** (기준 {AUC_GATE})\n")
    x["pred"], x["fold"] = preds, fold_id
    xo = x[x["fold"] > 0].copy()
    xo["q"] = pd.qcut(xo["pred"].rank(method="first"), 5, labels=False) + 1
    P("예측 5분위(OOS, 1=낮음): 양성률 / 주 규칙 평균 / net34 / n\n")
    for q, gq in xo.groupby("q"):
        P(f"- Q{q}: {gq['label_tgt_first'].mean():.1%} / {gq['ret_stop2'].mean():+.2%} / "
          f"{gq['ret_stop2'].mean()-COST:+.2%} / {len(gq)}")
    P("")

    # ---- 판정
    P("## 판정 (기계적 · 예비 표본이므로 확정 아님)\n")
    if stage1_pass:
        P("1단계 채택 5조건 **전부 통과** → 채택(전진 기록 착수) 후보. 2단계 생략.\n")
        return
    P(f"1단계 **미달** ({sum(cond.values())}/5). ")
    if auc < AUC_GATE:
        P(f"AUC {auc:.3f} < {AUC_GATE} → **2단계 없음.** 돌파를 접고 기록한다 (사전등록대로).\n")
        return
    P(f"AUC {auc:.3f} ≥ {AUC_GATE} → 2단계 진행.\n")

    # ---- 2단계
    P("## 2단계 — 선별 진입 (폴드 안에서 예측 상위 1/3)\n")
    sel = []
    for i in (1, 2):
        f = xo[xo["fold"] == i].sort_values("pred", ascending=False)
        sel.append(f.head(len(f) // 3))
    sel = pd.concat(sel)
    months2 = [m for m in months if (sel["month"] == m).any()]
    P(HDR)
    P(_row("상위 1/3 전체", describe(sel["ret_stop2"])))
    P(_row("상위 1/3 독립만", describe(sel.loc[sel["indep"] == 1, "ret_stop2"])))
    P(_row("상위 1/3 상한가 마감 제외", describe(sel.loc[sel["is_limit_close"] == 0, "ret_stop2"])))
    for i in (1, 2):
        P(_row(f"상위 1/3 폴드{i}", describe(sel.loc[sel["fold"] == i, "ret_stop2"])))
    top2 = xo.sort_values(["date", "pred"], ascending=[True, False]).groupby("date").head(2)
    P(_row("참고: 하루 상위 2건", describe(top2["ret_stop2"])))
    P("")
    lines, neg = monthly_table(sel, "ret_stop2", months2)
    out.extend(lines)
    P("")
    cond2, neg2 = five_conditions(sel, "ret_stop2", months2)
    P("**2단계 채택 5조건 (상위 1/3):** " + _cond_line(cond2, neg2, len(months2)) + "\n")
    P("2단계 " + ("**통과**" if all(cond2.values()) else "**미달**") + " (예비).\n")


# ====================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="이벤트 pickle. 있으면 읽고 없으면 만든다")
    ap.add_argument("--out", default=None, help="마크다운 보고 경로 (stdout 에도 찍는다)")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--max-codes", type=int, default=None, help="디버그용 종목 수 제한")
    args = ap.parse_args()
    t0 = time.time()

    if args.cache and os.path.exists(args.cache):
        blob = pd.read_pickle(args.cache)
        ev, meta = blob["ev"], blob["meta"]
        meta["cache"] = args.cache
        print(f"캐시 로드 {len(ev):,}건: {args.cache}", file=sys.stderr)
    else:
        c = connect()
        codes = available_codes(c)
        n_db = len(codes)
        if args.max_codes:
            codes = codes[:args.max_codes]
        print(f"분봉 종목 {n_db} (사용 {len(codes)})", file=sys.stderr)
        daily = load_ohlcv(c, DAILY_LEAD, END)
        cur = c.cursor()
        cur.execute("select code, date, close from daily_bars where date >= %s and date <= %s",
                    (START, END))
        raw = pd.DataFrame(cur.fetchall(), columns=["code", "date", "close"])
        cur.execute("select code from stocks where name like %s", ("%스팩%",))
        spac = {str(r[0]).strip() for r in cur.fetchall()}
        cur.close()
        raw["code"] = raw["code"].astype(str).str.strip()
        raw["date"] = pd.to_datetime(raw["date"])
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        daily["code"] = daily["code"].astype(str).str.strip()
        print(f"일봉 {len(daily):,} 행 · 원주가 {len(raw):,} 행 · 스팩 {len(spac)} ({time.time()-t0:.0f}s)",
              file=sys.stderr)
        w = prepare_daily(daily, raw, spac)
        key = _key_frame(w)
        dates = sorted(w.loc[(w["date"] >= START) & (w["date"] <= END), "date"].unique())
        acc = {"date_idx": {d: i for i, d in enumerate(dates)},
               "sum": np.zeros((len(dates), N_SLOTS)), "cnt": np.zeros((len(dates), N_SLOTS), dtype=np.int64)}
        cnt = {k: 0 for k in ("univ_days_with_bars", "daily_main_with_bars", "drop_no0900",
                              "drop_no1530", "drop_scale", "gap_break", "daily_main_after_dayfilters",
                              "intraday_break_any_minute", "daily_main_no_minute_break",
                              "trig_0900", "trig_1500plus", "limit_at_entry")}
        print(f"유니버스 (종목,일자) {len(key):,} · 일봉 주 이벤트 {int(key['ev_daily'].sum()):,} "
              f"({time.time()-t0:.0f}s)", file=sys.stderr)
        parts = []
        nb = (len(codes) + args.batch - 1) // args.batch
        for i in range(0, len(codes), args.batch):
            chunk = codes[i:i + args.batch]
            bars = load_minute_bars(c, START, END, codes=chunk)
            sub = events_from_batch(bars, key[key["code"].isin(chunk)], acc, cnt)
            if len(sub):
                parts.append(sub)
            print(f"  배치 {i//args.batch+1}/{nb}: {len(bars):,}행 → 이벤트 {len(sub):,} "
                  f"(누적 {sum(len(p) for p in parts):,}, {time.time()-t0:.0f}s)", file=sys.stderr)
            del bars
        c.close()
        ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        ev = finalize(ev, acc, key)
        meta = {"cnt": cnt, "n_daily_main": int(key["ev_daily"].sum()),
                "n_daily_codes": int(key.loc[key["ev_daily"], "code"].nunique()),
                "n_codes_db": n_db, "n_codes_loaded": len(codes),
                "run_at": time.strftime("%Y-%m-%d %H:%M"), "cache": args.cache}
        if args.cache:
            pd.to_pickle({"ev": ev, "meta": meta}, args.cache)

    out: list[str] = []
    analyze(ev, meta, out)
    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"→ {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
