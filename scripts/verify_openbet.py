"""시가 베팅 검증 — 11번 사전등록(`docs/research/manju/11-openbet-preregistration.md`) 그대로.

    uv run --extra pg python scripts/verify_openbet.py \\
        --env-file ../kr-quant/.env --cache <scratch>/openbet_verify.pkl --out <scratch>/openbet_prelim.md

이벤트 (문서와 동일, 값 변경 없음)
  전일(D)   tv_ratio >= 3.0 & close_pos >= 0.8 & 거래대금 >= 50억   (종가베팅 1차 규칙 — select_close_bet 재사용)
  당일(D+1) +3% <= 시가/전일종가 − 1 < +29%
  순위      전일 tv_ratio 내림차순, 하루 상위 2건

1단계 (일봉 11년, 연도별)  시가 → 당일 종가, 34bp 차감. 같은 날 유니버스(전일 거래대금 50억+)
  시가→종가 평균을 기준선으로 초과분. 중단: 초과분 평균 < −1.0%p AND 초과분 음수 해 >= 9/11.
2단계 (분봉 1년, 월별)     09:00 분봉 종가 진입 → 09:29 분봉 종가 청산, −2% 손절(분봉 저가 기준).
  비교 1회: + 익절 +2% (분봉 고가 기준, 같은 봉이면 손절 먼저). 채택 5조건.
  피처 8개 단변량 진단(스피어만·5분위)만.

파라미터·변형을 추가하지 않는다. DB 는 읽기만 한다.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from scalp_it.adapters.db import connect, load_ohlcv
from scalp_it.closebet import close_bet_features, close_bet_labels, select_close_bet

# ── 전일 조건 (종가베팅 1차 확정 규칙) ──
TV_RATIO_MIN = 3.0
CLOSE_POS_MIN = 0.8
TV_FLOOR = 5000.0       # 백만원 = 50억
TOP_N = 2
# ── 당일 갭 ──
GAP_MIN = 0.03
GAP_LIMIT = 0.29
LIMIT_UP_RET = 0.29     # 전일 상한가 마감 판정 (openbet_events 와 동일)
GAP_BINS = [0.03, 0.05, 0.10, 0.15, 0.29]
GAP_LABELS = ["3~5%", "5~10%", "10~15%", "15~29%"]
# ── 비용 ──
COST = 0.0034
COST_HI = 0.0055
COST_GRID = (0.0, 0.0021, 0.0034, 0.0055)
# ── 1단계 게이트 ──
GATE_EXCESS = -0.010
GATE_NEG_YEARS = 9
N_YEARS = 11
# ── 2단계 ──
STOP = -0.02
TAKE = 0.02
ENTRY_MIN = 0           # 09:00 분봉
EXIT_MIN = 29           # 09:29 분봉 종가 = 09:30 청산
MINUTE_TV_DIV = 1e6     # minute_bars.trade_value 원 → 백만원
ADJ_TOL = 0.002         # 분봉 09:00 시가 vs 일봉 D+1 시가 불일치 허용
NEG_MONTHS_MAX = 4
NEW_HIGH_WINDOW = 250
FEATURES = [
    "tv_ratio_prev", "ret_prev", "new_high_250", "gap_open",
    "open_vs_prev_high", "first_min_ret", "first_min_tv_share", "first_min_range",
]
MINUTE_START = "2025-08-01"


def _load_env(path: str | None) -> None:
    if not path or os.environ.get("KR_QUANT_DB"):
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


# ────────────────────────────── 일봉 이벤트 ──────────────────────────────

def build_daily_events(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """전일 조건(select_close_bet) → 달력상 다음 거래일 D+1 시가·종가 결합.

    Returns: (events[전일조건 충족·D+1 봉 있음], baseline[d1 → 유니버스 시가→종가 평균], meta)
    """
    ohlcv = ohlcv.sort_values(["code", "date"]).reset_index(drop=True)
    g = ohlcv.groupby("code", sort=False)
    ohlcv["max_close_250"] = g["close"].transform(
        lambda s: s.rolling(NEW_HIGH_WINDOW, min_periods=1).max())

    feats = close_bet_features(ohlcv)
    feats = feats.merge(
        ohlcv[["code", "date", "close", "high", "trade_value", "max_close_250"]],
        on=["code", "date"], how="left")
    labels = close_bet_labels(ohlcv)
    rules = {
        "trade_value": (TV_FLOOR, None),
        "tv_ratio": (TV_RATIO_MIN, None),
        "close_pos": (CLOSE_POS_MIN, None),
    }
    sel, n_excl_no_next = select_close_bet(feats, labels, rules=rules, top_n=None, exit_col="ret_open")
    prev = sel[["code", "date", "close", "high", "trade_value", "tv_ratio", "close_pos",
                "ret_day", "max_close_250"]].copy()
    prev = prev.rename(columns={"date": "d0", "close": "close_d0", "high": "high_d0",
                                "trade_value": "tv_d0"})
    prev["prev_limit_up"] = prev["ret_day"] >= LIMIT_UP_RET

    cal = pd.Series(sorted(ohlcv["date"].unique()))
    nxt = dict(pairwise(cal))
    prev["d1"] = prev["d0"].map(nxt)
    n_cal_end = int(prev["d1"].isna().sum())
    prev = prev[prev["d1"].notna()]

    d1bars = ohlcv[["code", "date", "open", "close"]].rename(
        columns={"date": "d1", "open": "open_d1", "close": "close_d1"})
    ev = prev.merge(d1bars, on=["code", "d1"], how="left")
    n_no_bar = int(ev["open_d1"].isna().sum())
    ev = ev[ev["open_d1"].notna() & (ev["open_d1"] > 0)].copy()
    ev["gap"] = ev["open_d1"] / ev["close_d0"] - 1.0
    ev["is_main"] = (ev["gap"] >= GAP_MIN) & (ev["gap"] < GAP_LIMIT)
    ev["ret_oc"] = ev["close_d1"] / ev["open_d1"] - 1.0
    ev["gap_bin"] = pd.cut(ev["gap"], bins=GAP_BINS, labels=GAP_LABELS, right=False)
    # 피처 1~5 (일봉)
    ev["tv_ratio_prev"] = ev["tv_ratio"]
    ev["ret_prev"] = ev["ret_day"]
    ev["new_high_250"] = (ev["open_d1"] >= ev["max_close_250"]).astype(int)
    ev["gap_open"] = ev["gap"]
    ev["open_vs_prev_high"] = ev["open_d1"] / ev["high_d0"] - 1.0

    # 기준선: D+1 에 전일(달력상 직전 거래일) 거래대금 >= 50억 인 종목의 시가→종가 평균
    prv = {b: a for a, b in pairwise(cal)}
    u = ohlcv[["code", "date", "open", "close", "trade_value"]].copy()
    u["tv_prev"] = g["trade_value"].shift(1)
    u["date_prev"] = g["date"].shift(1)
    u = u[(u["tv_prev"] >= TV_FLOOR) & (u["date_prev"] == u["date"].map(prv)) & (u["open"] > 0)]
    baseline = (u["close"] / u["open"] - 1.0).groupby(u["date"]).mean()

    meta = {
        "n_prev_cond": len(prev) + n_cal_end,
        "n_excl_no_next_any": n_excl_no_next,
        "n_cal_end": n_cal_end,
        "n_no_next_bar": n_no_bar,
        "n_with_open": len(ev),
        "universe_rows": len(u),
    }
    return ev, baseline, meta


def cap_top_n(ev: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    return (ev.sort_values(["d1", "tv_ratio"], ascending=[True, False])
              .groupby("d1", sort=False).head(n))


# ────────────────────────────── 1단계 ──────────────────────────────

def stage1(cap: pd.DataFrame, baseline: pd.Series) -> tuple[dict, list[str]]:
    cap = cap.copy()
    cap["base"] = cap["d1"].map(baseline)
    cap["net34"] = cap["ret_oc"] - COST
    cap["excess_net"] = cap["net34"] - cap["base"]        # 문서 문면: 이벤트 34bp 차감 − 유니버스 평균
    cap["excess_gross"] = cap["ret_oc"] - cap["base"]     # 참고: 양쪽 비용 없음
    cap["year"] = cap["d1"].dt.year
    L: list[str] = []
    L.append("| 연도 (D+1) | n | 시가→종가 평균 | net34 | 유니버스 기준선 | 초과(net34−기준) | 초과(gross) | 승률(gross) |")
    L.append("|---|---|---|---|---|---|---|---|")
    neg_net = neg_gross = 0
    years = sorted(cap["year"].unique())
    for y in years:
        s = cap[cap["year"] == y]
        en, eg = s["excess_net"].mean(), s["excess_gross"].mean()
        neg_net += en < 0
        neg_gross += eg < 0
        L.append(f"| {y} | {len(s)} | {s['ret_oc'].mean():+.2%} | {s['net34'].mean():+.2%} | "
                 f"{s['base'].mean():+.2%} | {en:+.2%} | {eg:+.2%} | {(s['ret_oc'] > 0).mean():.1%} |")
    en_all, eg_all = cap["excess_net"].mean(), cap["excess_gross"].mean()
    L.append(f"| **전체** | {len(cap)} | {cap['ret_oc'].mean():+.2%} | {cap['net34'].mean():+.2%} | "
             f"{cap['base'].mean():+.2%} | **{en_all:+.2%}** | {eg_all:+.2%} | {(cap['ret_oc'] > 0).mean():.1%} |")
    stop_net = (en_all < GATE_EXCESS) and (neg_net >= GATE_NEG_YEARS)
    stop_gross = (eg_all < GATE_EXCESS) and (neg_gross >= GATE_NEG_YEARS)
    L.append("")
    L.append("비용 4단계 (이벤트 시가→종가 평균 − 비용): " + ", ".join(
        f"{c*1e4:.0f}bp {cap['ret_oc'].mean() - c:+.2%}" for c in COST_GRID))
    L.append(f"중앙 {cap['ret_oc'].median():+.2%} · p5 {cap['ret_oc'].quantile(.05):+.1%} · "
             f"p25 {cap['ret_oc'].quantile(.25):+.1%} · p75 {cap['ret_oc'].quantile(.75):+.1%} · "
             f"p95 {cap['ret_oc'].quantile(.95):+.1%}")
    L.append("")
    L.append("| 분리 | n | 시가→종가 평균 | net34 | 기준선 | 초과(net) | 승률 |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, m in (("전일 상한가", cap["prev_limit_up"]), ("전일 비상한가", ~cap["prev_limit_up"])):
        s = cap[m]
        L.append(f"| {lab} | {len(s)} | {s['ret_oc'].mean():+.2%} | {s['net34'].mean():+.2%} | "
                 f"{s['base'].mean():+.2%} | {s['excess_net'].mean():+.2%} | {(s['ret_oc'] > 0).mean():.1%} |")
    for lab in GAP_LABELS:
        s = cap[cap["gap_bin"] == lab]
        if len(s):
            L.append(f"| 갭 {lab} | {len(s)} | {s['ret_oc'].mean():+.2%} | {s['net34'].mean():+.2%} | "
                     f"{s['base'].mean():+.2%} | {s['excess_net'].mean():+.2%} | {(s['ret_oc'] > 0).mean():.1%} |")
    L.append("")
    L.append(f"**중단 게이트** (초과 평균 < {GATE_EXCESS:+.1%} AND 음수 해 ≥ {GATE_NEG_YEARS}/{N_YEARS}): "
             f"net 기준 초과 {en_all:+.2%}, 음수 해 {neg_net}/{len(years)} → "
             f"{'**중단 (2단계 하지 않음)**' if stop_net else '통과 → 2단계'} ; "
             f"gross 기준 초과 {eg_all:+.2%}, 음수 해 {neg_gross}/{len(years)} → "
             f"{'중단' if stop_gross else '통과'}")
    res = {"excess_net": en_all, "excess_gross": eg_all, "neg_years_net": int(neg_net),
           "neg_years_gross": int(neg_gross), "n_years": len(years), "stop_net": bool(stop_net),
           "stop_gross": bool(stop_gross), "n": len(cap)}
    return res, L


# ────────────────────────────── 2단계: 분봉 ──────────────────────────────

def fetch_open_bars(conn, ev: pd.DataFrame) -> pd.DataFrame:
    """이벤트 (code, d1) 별로 09:00~09:29 분봉만 읽는다 (PK 조회, 건당 수 ms)."""
    cur = conn.cursor()
    parts = []
    rows_total = 0
    for i, (code, d1) in enumerate(zip(ev["code"], ev["d1"])):
        lo = d1.strftime("%Y-%m-%d 09:00:00")
        hi = d1.strftime("%Y-%m-%d 09:30:00")
        cur.execute(
            "select ts, open, high, low, close, trade_value from minute_bars "
            "where code = %s and ts >= %s and ts < %s order by ts", (code, lo, hi))
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "tv"])
            df.insert(0, "code", code)
            df.insert(1, "d1", d1)
            parts.append(df)
            rows_total += len(rows)
        if (i + 1) % 100 == 0:
            print(f"  분봉 조회 {i+1}/{len(ev)} (행 {rows_total:,})", file=sys.stderr)
    cur.close()
    if not parts:
        return pd.DataFrame(columns=["code", "d1", "ts", "open", "high", "low", "close", "tv"])
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    for c in ("open", "high", "low", "close", "tv"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def simulate(ev: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """이벤트별 주 규칙·비교 규칙 실행 + 피처 6~8."""
    bars = bars.copy()
    bars["m"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute - 540
    grouped = {k: g.sort_values("m") for k, g in bars.groupby(["code", "d1"], sort=False)}
    recs = []
    for r in ev.itertuples(index=False):
        key = (r.code, r.d1)
        rec = {"code": r.code, "d1": r.d1, "status": "분봉 없음"}
        g = grouped.get(key)
        if g is None:
            recs.append(rec)
            continue
        # 문서: "09:00 분봉(첫 1분봉) 종가". 시가 정적 VI(±10%) 가 걸리면 개장이 09:02 로 밀려
        # 그날 첫 봉이 09:02 다 — 그 경우 첫 봉을 쓰고 진입 분을 기록한다 (09:00 봉 없음 ≠ 제외).
        f = g.iloc[0]
        entry_m = int(f["m"])
        rec["entry_m"] = entry_m
        rec["open_min"] = float(f["open"])
        if not (f["open"] > 0) or abs(f["open"] / r.open_d1 - 1.0) > ADJ_TOL:
            rec["status"] = "일봉·분봉 시가 불일치"
            recs.append(rec)
            continue
        entry = float(f["close"])
        rec["entry"] = entry
        rec["first_min_ret"] = entry / f["open"] - 1.0
        rec["first_min_tv_share"] = (f["tv"] / MINUTE_TV_DIV) / r.tv_d0 if r.tv_d0 > 0 else np.nan
        rec["first_min_range"] = (f["high"] - f["low"]) / f["open"]
        after = g[(g["m"] > entry_m) & (g["m"] <= EXIT_MIN)]
        ms = set(after["m"].tolist())
        missing = [m for m in range(entry_m + 1, EXIT_MIN + 1) if m not in ms]
        rec["n_missing"] = len(missing)
        run = best = 0
        prevm = None
        for m in missing:
            run = run + 1 if prevm is not None and m == prevm + 1 else 1
            best = max(best, run)
            prevm = m
        rec["vi_like"] = best >= 2
        rec["n_bars_after"] = len(after)
        stop_px, take_px = entry * (1 + STOP), entry * (1 + TAKE)
        ret_main = ret_tp = None
        stop_hit = tp_hit = False
        exit_m_main = exit_m_tp = None
        for b in after.itertuples(index=False):
            if ret_main is None and b.low <= stop_px:
                ret_main, stop_hit, exit_m_main = STOP, True, b.m
            if ret_tp is None:
                if b.low <= stop_px:          # 같은 봉이면 손절 먼저 (보수)
                    ret_tp, exit_m_tp = STOP, b.m
                elif b.high >= take_px:
                    ret_tp, tp_hit, exit_m_tp = TAKE, True, b.m
            if ret_main is not None and ret_tp is not None:
                break
        if after.empty:
            last_close, last_m = entry, entry_m       # 진입 봉 이후 체결 없음 — 진입가 청산으로 둔다
        else:
            last_close, last_m = float(after.iloc[-1]["close"]), int(after.iloc[-1]["m"])
        if ret_main is None:
            ret_main, exit_m_main = last_close / entry - 1.0, last_m
        if ret_tp is None:
            ret_tp, exit_m_tp = last_close / entry - 1.0, last_m
        rec.update({"status": "ok", "ret": ret_main, "ret_tp": ret_tp, "stop_hit": stop_hit,
                    "tp_hit": tp_hit, "exit_m": exit_m_main, "exit_m_tp": exit_m_tp,
                    "last_m": last_m})
        recs.append(rec)
    sim = pd.DataFrame(recs)
    return ev.merge(sim, on=["code", "d1"], how="left")


def _desc(tag: str, r: pd.Series) -> str:
    if len(r) == 0:
        return f"| {tag} | 0 | — | — | — | — | — | — | — | — |"
    w, l = r[r > 0], r[r < 0]
    pl = (w.mean() / abs(l.mean())) if len(w) and len(l) and l.mean() != 0 else np.nan
    return (f"| {tag} | {len(r)} | {r.mean():+.2%} | {r.mean()-COST:+.2%} | {r.mean()-COST_HI:+.2%} | "
            f"{r.median():+.2%} | {(r > 0).mean():.1%} | {pl:.2f} | {r.quantile(.05):+.1%} / "
            f"{r.quantile(.25):+.1%} / {r.quantile(.75):+.1%} / {r.quantile(.95):+.1%} |")


_DESC_HDR = ["| 구분 | n | 평균 | net34 | net55 | 중앙 | 승률 | 손익비 | p5 / p25 / p75 / p95 |",
             "|---|---|---|---|---|---|---|---|---|"]


def _conditions(ok: pd.DataFrame, col: str, months: list) -> tuple[dict, int]:
    r = ok[col]
    neg = sum(1 for m in months if (ok.loc[ok["month"] == m, col].mean() - COST) < 0)
    ex_lu = ok.loc[~ok["prev_limit_up"], col]
    cond = {
        "1 net34>0": r.mean() - COST > 0,
        f"2 음수월<={NEG_MONTHS_MAX}": neg <= NEG_MONTHS_MAX,
        "3 상위1%제거 net34>0": r[r <= r.quantile(0.99)].mean() - COST > 0,
        "4 전일상한가제외 net34>0": (ex_lu.mean() - COST > 0) if len(ex_lu) else False,
        "5 승률>50%": (r > 0).mean() > 0.5,
    }
    return cond, neg


def stage2(ev2: pd.DataFrame) -> tuple[dict, list[str]]:
    L: list[str] = []
    n_all = len(ev2)
    st = ev2["status"].fillna("분봉 없음").value_counts()
    ok = ev2[ev2["status"] == "ok"].copy()
    L.append(f"분봉 구간 이벤트(상위 {TOP_N}건) {n_all}건 중 측정 {len(ok)}건 = **{len(ok)/n_all:.1%}** "
             f"(제외 사유: " + ", ".join(f"{k} {v}" for k, v in st.items() if k != "ok") + ")")
    res = {"n_events": n_all, "n_ok": len(ok), "coverage": len(ok) / n_all if n_all else np.nan}
    if len(ok) == 0:
        L.append("측정 표본 없음.")
        return res, L
    ok["month"] = ok["d1"].dt.strftime("%Y-%m")
    months = sorted(ok["month"].unique())
    per_day = ok.groupby("d1").size()
    L.append(f"거래일 {ok['d1'].nunique()}일 · 하루 건수 평균 {per_day.mean():.2f} / 최대 {per_day.max()} · "
             f"종목 {ok['code'].nunique()}개 · 같은 종목 재등장 최다 "
             + ", ".join(f"{k}×{v}" for k, v in ok["code"].value_counts().head(3).items()))
    em = ok["entry_m"].value_counts().sort_index()
    L.append(f"첫 봉이 09:00 이 아닌 건(시가 VI 로 개장 지연 근사) {int((ok['entry_m'] != ENTRY_MIN).sum())}건 "
             f"({(ok['entry_m'] != ENTRY_MIN).mean():.1%}) · 진입 분 분포: "
             + ", ".join(f"09:{int(m):02d} {int(n)}" for m, n in em.items()))
    L.append(f"손절 발동 {int(ok['stop_hit'].sum())}건 ({ok['stop_hit'].mean():.1%}) · "
             f"진입 봉 이후~09:29 분봉 공백 있음 {int((ok['n_missing'] > 0).sum())}건 · "
             f"연속 2분 이상 공백(VI 근사) {int(ok['vi_like'].sum())}건 · "
             f"진입 봉 이후 체결 없음 {int((ok['n_bars_after'] == 0).sum())}건 · "
             f"마지막 봉이 09:29 아닌 건 {int((ok['last_m'] != EXIT_MIN).sum())}건")
    fm = ok["first_min_ret"]
    L.append(f"운영 갭 (첫 분봉 종가 / 시가 − 1): 평균 {fm.mean():+.2%} · p5 {fm.quantile(.05):+.2%} · "
             f"p25 {fm.quantile(.25):+.2%} · 중앙 {fm.median():+.2%} · p75 {fm.quantile(.75):+.2%} · "
             f"p95 {fm.quantile(.95):+.2%}")
    L.append("")
    L.append("### 주 규칙 — 첫 분봉(09:00, VI 시 09:02) 종가 진입 → 09:29 분봉 종가 청산, −2% 손절")
    L.append("")
    L += _DESC_HDR
    r = ok["ret"]
    L.append(_desc("전체", r))
    L.append(_desc("상위 1% 제거", r[r <= r.quantile(0.99)]))
    L.append(_desc("전일 상한가", ok.loc[ok["prev_limit_up"], "ret"]))
    L.append(_desc("전일 비상한가", ok.loc[~ok["prev_limit_up"], "ret"]))
    for lab in GAP_LABELS:
        L.append(_desc(f"갭 {lab}", ok.loc[ok["gap_bin"] == lab, "ret"]))
    L.append("")
    L.append("비용 4단계 (전체 평균 − 비용): " + ", ".join(
        f"{c*1e4:.0f}bp {r.mean() - c:+.2%}" for c in COST_GRID))
    L.append("")
    L.append("| 월 | n | 평균 | net34 | 중앙 | 승률 | 손절 비율 |")
    L.append("|---|---|---|---|---|---|---|")
    for m in months:
        s = ok[ok["month"] == m]
        L.append(f"| {m} | {len(s)} | {s['ret'].mean():+.2%} | {s['ret'].mean()-COST:+.2%} | "
                 f"{s['ret'].median():+.2%} | {(s['ret'] > 0).mean():.1%} | {s['stop_hit'].mean():.0%} |")
    cond, neg = _conditions(ok, "ret", months)
    L.append("")
    L.append("**채택 5조건 (주 규칙)**: " + "  ".join(
        f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items())
        + f"  (음수 월 {neg}/{len(months)}) → **{'전부 통과 — 전진 기록 착수 후보' if all(cond.values()) else '미달 — 기록하고 접는다'}**")
    res.update({"main_pass": all(cond.values()), "main_cond": {k: bool(v) for k, v in cond.items()},
                "main_neg_months": neg, "n_months": len(months),
                "main_mean": float(r.mean()), "main_net34": float(r.mean() - COST),
                "main_win": float((r > 0).mean()), "main_median": float(r.median())})

    L.append("")
    L.append("### 비교 1회 — 주 규칙 + 익절 +2% (같은 봉이면 손절 먼저)")
    L.append("")
    L += _DESC_HDR
    rt = ok["ret_tp"]
    L.append(_desc("전체", rt))
    L.append(_desc("전일 비상한가", ok.loc[~ok["prev_limit_up"], "ret_tp"]))
    cond_tp, neg_tp = _conditions(ok, "ret_tp", months)
    L.append("")
    L.append(f"익절 발동 {int(ok['tp_hit'].sum())}건 ({ok['tp_hit'].mean():.1%}). 5조건: " + "  ".join(
        f"[{k}: {'O' if v else 'X'}]" for k, v in cond_tp.items()) + f"  (음수 월 {neg_tp}/{len(months)})")
    if res["main_pass"]:
        L.append(f"주 규칙 통과 → 개선 여부: net34 {r.mean()-COST:+.2%} → {rt.mean()-COST:+.2%} "
                 f"({'개선' if rt.mean() > r.mean() else '악화'})")
    else:
        L.append("주 규칙 미달 → 비교 결과는 **기록만** 한다 (채택 근거 아님).")
    res.update({"tp_pass": all(cond_tp.values()), "tp_net34": float(rt.mean() - COST)})

    L.append("")
    L.append("### 피처 8개 — 단변량 진단 (주 규칙 gross 수익률 기준, 모델 없음)")
    L.append("")
    L.append("| 피처 | 가설 | n | 스피어만 ρ | Q1 평균/중앙/승률 | Q2 | Q3 | Q4 | Q5 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    hyp = {"tv_ratio_prev": "+", "ret_prev": "±", "new_high_250": "+", "gap_open": "− (10%+)",
           "open_vs_prev_high": "+", "first_min_ret": "+", "first_min_tv_share": "+",
           "first_min_range": "−"}
    rows = []
    for f in FEATURES:
        x = ok[[f, "ret"]].dropna()
        rho = x[f].rank().corr(x["ret"].rank()) if len(x) > 2 else np.nan
        if x[f].nunique() <= 2:
            qs = x.groupby(x[f].astype(int))["ret"]
            cells = [f"{k}: {g.mean():+.2%}/{g.median():+.2%}/{(g>0).mean():.0%} (n{len(g)})"
                     for k, g in qs]
            cells += [""] * (5 - len(cells))
        else:
            q = pd.qcut(x[f].rank(method="first"), 5, labels=False)
            cells = [f"{g.mean():+.2%}/{g.median():+.2%}/{(g>0).mean():.0%}"
                     for _, g in x.groupby(q)["ret"]]
        rows.append((f, rho, len(x)))
        L.append(f"| {f} | {hyp[f]} | {len(x)} | {rho:+.3f} | " + " | ".join(cells) + " |")
    top3 = sorted(rows, key=lambda t: -abs(t[1]) if np.isfinite(t[1]) else 0)[:3]
    L.append("")
    L.append("|ρ| 상위 3: " + ", ".join(f"{f} {rho:+.3f}" for f, rho, _ in top3)
             + " — 후속 사전등록 후보로 기록만. 이 회차 규칙에 넣지 않는다.")
    res["feat_top3"] = [(f, float(rho)) for f, rho, _ in top3]
    return res, L


# ────────────────────────────── main ──────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2016-09-09")
    p.add_argument("--end", default="2026-08-21")
    p.add_argument("--minute-start", default=MINUTE_START)
    p.add_argument("--env-file", default="/home/young/Documents/git/kr-quant/.env")
    p.add_argument("--cache", default=None, help="일봉 이벤트·분봉 시뮬 결과 pickle (있으면 DB 를 읽지 않는다)")
    p.add_argument("--out", default=None, help="마크다운 보고 파일")
    p.add_argument("--skip-stage2", action="store_true", help="1단계 게이트와 무관하게 2단계를 돌리지 않는다")
    p.add_argument("--force-stage2", action="store_true",
                   help="1단계가 중단이어도 2단계를 **참고용**으로 계산한다 (판정에 쓰지 않는다)")
    p.add_argument("--refetch-minute", action="store_true", help="캐시의 분봉 시뮬 결과를 버리고 다시 조회")
    args = p.parse_args(argv)

    if args.cache and os.path.exists(args.cache):
        blob = pd.read_pickle(args.cache)
        ev, baseline, meta, ev2 = blob["ev"], blob["baseline"], blob["meta"], blob.get("ev2")
        if args.refetch_minute:
            ev2 = None
        print(f"캐시 로드: {args.cache}", file=sys.stderr)
        conn = None
    else:
        _load_env(args.env_file)
        conn = connect()
        print("일봉 로드…", file=sys.stderr)
        ohlcv = load_ohlcv(conn, args.start, args.end)
        print(f"  {len(ohlcv):,}행", file=sys.stderr)
        ev, baseline, meta = build_daily_events(ohlcv)
        del ohlcv
        ev2 = None

    main_ev = ev[ev["is_main"]]
    cap = cap_top_n(main_ev)
    L: list[str] = []
    L.append("# 시가 베팅 검증 — 11번 사전등록 실행 결과")
    L.append("")
    L.append(f"생성 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 일봉 {args.start}~{args.end} · "
             f"분봉 {args.minute_start}~{args.end} · 스크립트 `scripts/verify_openbet.py`")
    L.append("")
    L.append("## 표본 (일봉)")
    L.append("")
    L.append(f"전일 조건 충족 {meta['n_prev_cond']:,} (다음 봉 전무 {meta['n_excl_no_next_any']}, 달력 끝 {meta['n_cal_end']}, "
             f"D+1 봉 없음 {meta['n_no_next_bar']}) → D+1 시가 있음 {meta['n_with_open']:,} · "
             f"주 정의(갭 3~29%) {len(main_ev):,} · 전일 상한가 {int(main_ev['prev_limit_up'].sum()):,} · "
             f"상위 {TOP_N}건 {len(cap):,} (전일 상한가 {int(cap['prev_limit_up'].sum()):,}, "
             f"{cap['prev_limit_up'].mean():.0%})")
    L.append("")
    L.append("## 1단계 — 일봉 11년: 시가 → 당일 종가, 유니버스 기준선 대비 초과 (확정 표본)")
    L.append("")
    r1, L1 = stage1(cap, baseline)
    L += L1
    L.append("")

    do2 = not args.skip_stage2 and (not r1["stop_net"] or args.force_stage2)
    if r1["stop_net"]:
        L.append("1단계 중단 조건 충족 — 사전등록대로 **2단계를 하지 않는다.**"
                 + (" (--force-stage2: 아래 2단계 수치는 **참고용**이며 판정에 쓰지 않는다)"
                    if args.force_stage2 else ""))
        L.append("")
    cap2 = cap[cap["d1"] >= pd.Timestamp(args.minute_start)].copy()
    if do2:
        if ev2 is None:
            if conn is None:
                _load_env(args.env_file)
                conn = connect()
            print(f"분봉 조회: 이벤트 {len(cap2)}건", file=sys.stderr)
            bars = fetch_open_bars(conn, cap2)
            ev2 = simulate(cap2, bars)
        if args.cache:
            pd.to_pickle({"ev": ev, "baseline": baseline, "meta": meta, "ev2": ev2}, args.cache)
        tag = " · 1단계 중단 → 참고용" if r1["stop_net"] else ""
        L.append(f"## 2단계 — 분봉 {args.minute_start}~{args.end}: 월별 폴드 (예비 — 분봉 표본 일부{tag})")
        L.append("")
        _r2, L2 = stage2(ev2)
        L += L2
    elif args.cache:
        pd.to_pickle({"ev": ev, "baseline": baseline, "meta": meta, "ev2": ev2}, args.cache)
    if conn is not None:
        conn.close()
    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
