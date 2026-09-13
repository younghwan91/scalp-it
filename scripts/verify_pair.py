"""짝꿍매매 검증 — 사전등록 `docs/research/manju/12-pair-preregistration.md` 그대로 실행.

  트리거   대장 분봉 종가가 처음 상한가(전일 원주가 종가 × 1.3 호가단위 내림)와 같아진 분 T.
           T 까지 누적 거래대금 ≥ 300억, T ≤ 15:13, (대장, 일자)당 첫 잠김 1회.
  후속     T 시점 등락률 +10~15% · 누적 50억 · 대장의 60일 잔차 상관 상위 30 & 상관 > 0 ·
           T+1 봉 존재 · T+1 시가 < 상한가. 후보가 여럿이면 상관 최고 1건.
  진입     T+1 시가.   청산 = min(T+6 시가, 대장 이탈(T+1~T+5 중 대장 종가 < 상한가 인 첫 분 k) 다음 분 k+1 시가)
  비용     64bp 기본 (34 / 94 병기).
  1단계    5조건 전부 만족해야 채택: net64 평균·중앙 > 0 / 시간순 4폴드 음수 ≤ 1 / 상위 1% 제거 후 > 0 /
           진입 후 5분 내 후속 상한가 도달 건 제외 후 > 0 / 승률 > 50%.
  보조     `테마무시`(클러스터 조건 제거, 누적 거래대금 최고 1건) 1회, `브래킷`(+2/−2, 손절 우선) 1회.
  2단계    피처 8개 스피어만·5분위 + 얕은 GBM(max_depth=2, n_estimators=100) 시간순 3폴드 AUC. 채택 없음.

이벤트 생성부(트리거·후속 후보·잔차 상관)는 `scripts/pair_events.py` 를 그대로 쓴다. 문서에 없는
파라미터·변형은 넣지 않는다. DB 는 읽기만 한다. 실행:

    uv run --extra pg python scripts/verify_pair.py --start 2025-08-04 --end 2026-08-21 \\
        --cache /path/pair_trades.pkl --out /path/pair_prelim.md
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from limitup_timing import _load_env
from pair_events import (
    BUCKETS,
    TOP_K,
    _bucket,
    _q,
    follower_candidates,
    load_daily_adj_close,
    load_daily_raw,
    load_index_close,
    residual_corr_with,
    scan_date,
)

from scalp_it.intraday.loader import load_minute_bars
from scalp_it.intraday.store import connect

# ── 사전 고정 (문서 12번 5·7절) ────────────────────────────────────────────
HOLD_MIN = 5                 # 보유 분: 진입 T+1 시가 → T+6 시가
# 고정 가정값(값은 바꾸지 않는다). 일자별 거래세 스케줄은 krx_quant_core.costs 참고.
COST_BASE = 0.0064           # 왕복 34bp + 슬리피지 왕복 30bp
COST_LO = 0.0034             # 슬리피지 0 (낙관)
COST_HI = 0.0094             # 보수
COSTS = {"net34": COST_LO, "net64": COST_BASE, "net94": COST_HI}
BRACKET_TP, BRACKET_SL = 0.02, 0.02
N_FOLDS = 4                  # 1단계 시간순 폴드
N_FOLDS_GBM = 3              # 2단계 GBM 시간순 폴드
GBM_AUC_MIN = 0.60
ADV_WINDOW = 20              # fol_tv_ratio 분모: 직전 20일 일평균 거래대금
FEATURES = ["lead_lock_min", "lead_touch_to_lock", "lead_cum_tv", "fol_ret",
            "fol_tv_ratio", "fol_ret_5m", "pair_corr", "n_cluster_up"]
FEATURE_HYPO = {  # 유리한 방향 (문서 6절)
    "lead_lock_min": "낮을수록", "lead_touch_to_lock": "낮을수록", "lead_cum_tv": "높을수록",
    "fol_ret": "낮을수록", "fol_tv_ratio": "높을수록", "fol_ret_5m": "낮을수록",
    "pair_corr": "높을수록", "n_cluster_up": "높을수록",
}
_OPEN_MINUTE = 9 * 60


def _next_open(open_s: pd.Series, minute: int) -> tuple[float, int]:
    """``minute`` 이후(포함) 첫 봉의 시가와 그 분. 없으면 (nan, -1).

    청산 분에 봉이 없으면(무체결·VI) 그 다음 체결 봉 시가로 판다 — 보수적 해석(문서에 규정 없음).
    """
    s = open_s.dropna()
    s = s[s.index >= minute]
    if s.empty:
        return float("nan"), -1
    return float(s.iloc[0]), int(s.index[0])


def outcome(snap: dict, hl: dict, trig: dict, fol: str) -> dict:
    """후속주 한 건의 진입·청산·브래킷·상한가 도달·피처 보조값. 진입 이후 가격은 여기서만 읽는다."""
    T, lead = trig["T"], trig["lead"]
    close_w, open_w = snap["close"], snap["open"]
    high_w, low_w = hl["high"], hl["low"]
    fol_limit = float(snap["limit"][fol])
    entry = float(open_w.at[T + 1, fol])
    # 대장 이탈: T+1~T+5 중 대장 종가(ffill — 봉 없으면 직전 종가 유지) < 상한가 인 첫 분 k
    lead_close = close_w[lead]
    k_exit = None
    for k in range(T + 1, T + HOLD_MIN + 1):
        c = lead_close.get(k, np.nan)
        if np.isfinite(c) and c < trig["lead_limit"] - 1e-6:
            k_exit = k
            break
    exit_minute = T + HOLD_MIN + 1 if k_exit is None else min(T + HOLD_MIN + 1, k_exit + 1)
    exit_px, exit_actual = _next_open(open_w[fol], exit_minute)
    # 후속 상한가 도달 (진입 후 5분: T+1~T+5 봉 고가, 또는 청산 시가 자체가 상한가)
    win_hi = high_w[fol].reindex(range(T + 1, T + HOLD_MIN + 1)).dropna() if fol in high_w.columns else pd.Series(dtype=float)
    fol_hit_limit = bool((win_hi >= fol_limit - 1e-6).any() or (np.isfinite(exit_px) and exit_px >= fol_limit - 1e-6))
    # 브래킷: T+1~T+5 봉에서 고가 ≥ ×1.02 → +2%, 저가 ≤ ×0.98 → −2%, 같은 봉은 손절 우선, 아니면 T+6 시가
    br_ret = np.nan
    br_kind = "time"
    if fol in high_w.columns:
        for k in range(T + 1, T + HOLD_MIN + 1):
            h = high_w[fol].get(k, np.nan)
            lo = low_w[fol].get(k, np.nan)
            if not np.isfinite(h):
                continue
            if lo <= entry * (1 - BRACKET_SL):
                br_ret, br_kind = -BRACKET_SL, "stop"
                break
            if h >= entry * (1 + BRACKET_TP):
                br_ret, br_kind = BRACKET_TP, "tp"
                break
    if br_kind == "time":
        t6, _ = _next_open(open_w[fol], T + HOLD_MIN + 1)
        br_ret = t6 / entry - 1 if np.isfinite(t6) else np.nan
    # 피처 보조: fol_ret_5m (T−5 종가, 없으면 09:00 봉 시가, 없으면 전일 종가)
    fc = close_w[fol]
    ref = fc.get(T - 5, np.nan) if T - 5 >= _OPEN_MINUTE else np.nan
    ref_kind = "T-5"
    if not np.isfinite(ref):
        o0 = open_w[fol].get(_OPEN_MINUTE, np.nan)
        ref, ref_kind = (o0, "open0900") if np.isfinite(o0) else (float(snap["prev"][fol]), "prev_close")
    return {
        "entry": entry, "exit": exit_px, "exit_minute": exit_minute, "exit_actual": exit_actual,
        "lead_exit": k_exit is not None, "lead_exit_k": k_exit if k_exit is not None else -1,
        "gross": exit_px / entry - 1 if np.isfinite(exit_px) else np.nan,
        "fol_hit_limit": fol_hit_limit, "fol_limit": fol_limit,
        "br_ret": br_ret, "br_kind": br_kind,
        "fol_ret_5m": float(fc.get(T, np.nan)) / ref - 1 if np.isfinite(ref) and ref > 0 else np.nan,
        "fol_ret_5m_ref": ref_kind,
    }


def _trade_row(base: dict, snap: dict, hl: dict, trig: dict, fol: str, row: pd.Series,
               adv_day: pd.Series, name_of: dict) -> dict:
    out = outcome(snap, hl, trig, fol)
    adv = adv_day.get(fol, np.nan)
    return {**base, "fol": fol, "fol_name": name_of.get(fol, fol),
            "fol_ret": float(row["ret"]), "fol_cum_tv": float(row["cum_tv"]),
            "fol_tv_ratio": float(row["cum_tv"]) / adv if np.isfinite(adv) and adv > 0 else np.nan,
            "pair_corr": float(row["corr"]) if np.isfinite(row["corr"]) else np.nan,
            "pair_rank": float(row["rank"]) if np.isfinite(row["rank"]) else np.nan, **out}


def collect(conn, start: str, end: str) -> dict:
    """트리거 → 후속 선택 → 결과. 주 규칙(클러스터·상관 최고)과 테마무시(거래대금 최고) 두 묶음을 만든다."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    warm = (s - timedelta(days=130)).strftime("%Y-%m-%d")
    print("일봉 로드 …", file=sys.stderr)
    raw = load_daily_raw(conn, warm, end)
    raw = raw.sort_values(["code", "date"])
    raw["adv20"] = raw.groupby("code")["trade_value"].transform(
        lambda x: x.shift(1).rolling(ADV_WINDOW, min_periods=1).mean()) * 1e6   # 백만원 → 원
    px = load_daily_adj_close(conn, warm, end)
    idx = load_index_close(conn, warm, end)
    stocks = _q(conn, "select code, market, name from stocks")
    market_of = dict(zip(stocks["code"], stocks["market"]))
    name_of = dict(zip(stocks["code"], stocks["name"]))
    trading_days = [d for d in idx.index if s <= d <= e]

    trig_rows: list[dict] = []
    main_rows: list[dict] = []
    d_rows: list[dict] = []
    excl: dict[str, int] = {}

    def _add(k, v=1):
        excl[k] = excl.get(k, 0) + v

    for i, d in enumerate(trading_days):
        ds = d.strftime("%Y-%m-%d")
        raw_day = raw[raw["date"] == d]
        day_bars = load_minute_bars(conn, ds, ds)
        if day_bars.empty:
            continue
        trigs, ex1, snap = scan_date(day_bars, raw_day)
        for k, v in ex1.items():
            _add(k, v)
        if not trigs:
            continue
        bars = day_bars.copy()
        bars["minute"] = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute
        bars = bars[bars["code"].isin(snap["close"].columns)]
        hl = {"high": bars.pivot(index="minute", columns="code", values="high"),
              "low": bars.pivot(index="minute", columns="code", values="low")}
        adv_day = raw_day.set_index("code")["adv20"]
        corr_cache: dict[str, pd.Series] = {}
        for t in trigs:
            t["date"] = d
            lead = t["lead"]
            if lead not in corr_cache:
                corr_cache[lead] = residual_corr_with(px, idx, market_of, d, lead)
            corr = corr_cache[lead]
            cand, ex2 = follower_candidates(snap, t)
            for k, v in ex2.items():
                if not k.startswith("_"):
                    _add(k, v)
            if len(corr):
                rank_s = corr.rank(ascending=False, method="min")
                cand["corr"] = corr.reindex(cand.index)
                cand["rank"] = rank_s.reindex(cand.index)
                cluster_codes = corr[(rank_s <= TOP_K) & (corr > 0)].index
            else:
                cand["corr"] = np.nan
                cand["rank"] = np.nan
                cluster_codes = pd.Index([])
            cand["cluster"] = (cand["rank"] <= TOP_K) & (cand["corr"] > 0)
            # n_cluster_up: T 시점 클러스터 종목(분봉 있는 것) 중 등락률 ≥ +10%
            cl_in = [c for c in cluster_codes if c in snap["close"].columns]
            if cl_in and t["T"] in snap["close"].index:
                r_cl = snap["close"].loc[t["T"], cl_in] / snap["prev"].reindex(cl_in) - 1
                n_cluster_up = int((r_cl >= 0.10).sum())
            else:
                n_cluster_up = 0
            t["n_fol"] = len(cand)
            t["n_fol_cluster"] = int(cand["cluster"].sum())
            t["n_cluster_up"] = n_cluster_up
            t["n_cluster_codes_with_minute"] = len(cl_in)
            trig_rows.append(t)

            base = {"date": d, "lead": lead, "lead_name": name_of.get(lead, lead), "T": t["T"],
                    "lead_lock_min": t["T"] - _OPEN_MINUTE, "lead_touch_to_lock": t["touch_to_lock"],
                    "lead_cum_tv": t["lead_cum_tv"], "lead_limit": t["lead_limit"],
                    "n_cluster_up": n_cluster_up, "n_cand": len(cand), "n_cand_cluster": int(cand["cluster"].sum())}

            cl = cand[cand["cluster"]]
            if len(cl):
                fol = cl["corr"].idxmax()
                main_rows.append(_trade_row(base, snap, hl, t, fol, cl.loc[fol], adv_day, name_of))
            if len(cand):
                fol = cand["cum_tv"].idxmax()
                d_rows.append(_trade_row(base, snap, hl, t, fol, cand.loc[fol], adv_day, name_of))
        if i % 20 == 0:
            print(f"  {ds} 트리거 누적 {len(trig_rows)} / 주 규칙 거래 {len(main_rows)}", file=sys.stderr)
    return {"trig": pd.DataFrame(trig_rows), "main": pd.DataFrame(main_rows), "D": pd.DataFrame(d_rows),
            "excl": excl, "start": start, "end": end,
            "n_minute_codes": int(_q(conn, "select count(distinct code) from minute_bars").iloc[0, 0])}


# ── 분석 ────────────────────────────────────────────────────────────────────

def _stats(g: pd.Series) -> dict:
    """총수익 시리즈 → 기술통계(비용 3단계 포함). 손익비 = 평균 이익 / 평균 손실 (64bp 차감 기준)."""
    g = g.dropna()
    n = len(g)
    if n == 0:
        return {"n": 0}
    net = g - COST_BASE
    w, lo = net[net > 0], net[net <= 0]
    pl = abs(w.mean() / lo.mean()) if len(w) and len(lo) and lo.mean() != 0 else np.nan
    return {"n": n, "gross": g.mean(), "net34": g.mean() - COST_LO, "net64": g.mean() - COST_BASE,
            "net94": g.mean() - COST_HI, "median64": net.median(), "win64": (net > 0).mean(), "pl64": pl,
            "p5": net.quantile(.05), "p25": net.quantile(.25), "p75": net.quantile(.75), "p95": net.quantile(.95)}


def _fmt(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "| 0 | — | — | — | — | — | — | — | — |"
    return (f"| {s['n']} | {s['gross']:+.2%} | {s['net34']:+.2%} | **{s['net64']:+.2%}** | {s['net94']:+.2%} | "
            f"{s['median64']:+.2%} | {s['win64']:.1%} | {s['pl64']:.2f} | "
            f"{s['p5']:+.1%} / {s['p25']:+.1%} / {s['p75']:+.1%} / {s['p95']:+.1%} |")


_HDR = ("| 구분 | n | 평균(총) | net34 | net64 | net94 | 중앙(64) | 승률(64) | 손익비(64) | p5/p25/p75/p95 (64) |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def _folds(tr: pd.DataFrame, n_folds: int) -> list[np.ndarray]:
    order = tr.sort_values(["date", "T"]).index.to_numpy()
    return np.array_split(order, n_folds)


def stage1(tr: pd.DataFrame, L: list[str], label: str, judge: bool) -> dict | None:
    """1단계 표·5조건. ``judge`` 가 False 면(테마무시) 판정 문장은 쓰지 않는다."""
    L.append(f"## {label}\n")
    tr = tr.dropna(subset=["gross"]).copy()
    if tr.empty:
        L.append("거래 0건.\n")
        return None
    tr["net64"] = tr["gross"] - COST_BASE
    L.append(_HDR)
    L.append("| 전체 " + _fmt(_stats(tr["gross"])))
    q99 = tr["gross"].quantile(0.99)
    trim = tr[tr["gross"] <= q99]
    L.append("| 상위 1% 제거 " + _fmt(_stats(trim["gross"])))
    nolim = tr[~tr["fol_hit_limit"]]
    L.append("| 후속 5분내 상한가 도달 제외 " + _fmt(_stats(nolim["gross"])))
    L.append("| 후속 상한가 도달분만 " + _fmt(_stats(tr[tr["fol_hit_limit"]]["gross"])))
    L.append("| 대장 이탈 발생 " + _fmt(_stats(tr[tr["lead_exit"]]["gross"])))
    L.append("| 대장 이탈 없음(5분 보유) " + _fmt(_stats(tr[~tr["lead_exit"]]["gross"])))
    L.append("| touch_to_lock = 0 " + _fmt(_stats(tr[tr["lead_touch_to_lock"] == 0]["gross"])))
    L.append("| touch_to_lock > 0 " + _fmt(_stats(tr[tr["lead_touch_to_lock"] > 0]["gross"])))
    for b, _, _ in BUCKETS:
        sub = tr[tr["T"].map(_bucket) == b]
        if len(sub):
            L.append(f"| 잠김 {b} " + _fmt(_stats(sub["gross"])))
    if "br_ret" in tr.columns:
        L.append("| 브래킷(+2/−2, 손절 우선; 보조 1회) " + _fmt(_stats(tr["br_ret"])))
    L.append("")
    # 4폴드
    folds = _folds(tr, N_FOLDS)
    L.append(f"시간순 {N_FOLDS}폴드 (건수 균등):\n")
    L.append("| 폴드 | 기간 | n | net64 평균 | 중앙 | 승률 |")
    L.append("|---|---|---|---|---|---|")
    neg = 0
    for i, ix in enumerate(folds, 1):
        f = tr.loc[ix]
        m = f["net64"].mean()
        neg += int(m < 0)
        L.append(f"| {i} | {f['date'].min().date()}~{f['date'].max().date()} | {len(f)} | {m:+.2%} | "
                 f"{f['net64'].median():+.2%} | {(f['net64']>0).mean():.0%} |")
    L.append("")
    L.append("월별 (판정엔 쓰지 않는다):\n")
    L.append("| 월 | n | net64 평균 | 중앙 | 승률 |")
    L.append("|---|---|---|---|---|")
    for m, f in tr.groupby(tr["date"].dt.strftime("%Y-%m")):
        L.append(f"| {m} | {len(f)} | {f['net64'].mean():+.2%} | {f['net64'].median():+.2%} | {(f['net64']>0).mean():.0%} |")
    L.append("")
    st = _stats(tr["gross"])
    cond = {
        "1 net64 평균>0 & 중앙>0": st["net64"] > 0 and st["median64"] > 0,
        f"2 {N_FOLDS}폴드 음수≤1": neg <= 1,
        "3 상위1%제거 net64>0": _stats(trim["gross"])["net64"] > 0,
        "4 후속상한가제외 net64>0": (_stats(nolim["gross"]).get("net64", np.nan) > 0),
        "5 승률>50%": st["win64"] > 0.5,
    }
    if judge:
        L.append("**1단계 5조건 (64bp):** " + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items())
                 + f"  (음수 폴드 {neg}/{N_FOLDS}) → **{'전부 통과' if all(cond.values()) else '미달'}** (예비)\n")
    else:
        L.append("(비교용 — 채택 판정에 쓰지 않는다. 같은 5조건을 참고로 찍으면: "
                 + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items()) + ")\n")
    vc = tr["fol"].value_counts()
    L.append(f"같은 후속주 재등장: 최대 {vc.max()}회 — "
             + ", ".join(f"{tr.loc[tr['fol']==k, 'fol_name'].iloc[0]} {v}" for k, v in vc.head(6).items())
             + f"; 후속 종목 {tr['fol'].nunique()} / 대장 종목 {tr['lead'].nunique()}\n")
    return {"cond": cond, "neg": neg, "stats": st, "tr": tr}


def stage2(tr: pd.DataFrame, L: list[str]) -> dict:
    L.append("## 2단계 — 피처 진단 (채택 없음)\n")
    tr = tr.dropna(subset=["gross"]).copy()
    tr["net64"] = tr["gross"] - COST_BASE
    tr["y"] = (tr["net64"] > 0).astype(int)
    L.append("| 피처 | 가설 | 스피어만 ρ (net64) | p | Q1 평균/중앙/승률 | Q2 | Q3 | Q4 | Q5 | 결측 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    rho_tab = {}
    for f in FEATURES:
        x = tr[[f, "net64"]].dropna()
        if len(x) < 10 or x[f].nunique() < 2:
            L.append(f"| `{f}` | {FEATURE_HYPO[f]} | — | — | — | — | — | — | — | {tr[f].isna().sum()} |")
            continue
        rho, p = spearmanr(x[f], x["net64"])
        rho_tab[f] = rho
        try:
            q = pd.qcut(x[f].rank(method="first"), 5, labels=False) + 1
        except ValueError:
            q = pd.Series(1, index=x.index)
        cells = []
        for qi in range(1, 6):
            g = x.loc[q == qi, "net64"]
            cells.append(f"{g.mean():+.2%}/{g.median():+.2%}/{(g>0).mean():.0%} (n{len(g)})" if len(g) else "—")
        L.append(f"| `{f}` | {FEATURE_HYPO[f]} | {rho:+.3f} | {p:.2f} | " + " | ".join(cells) + f" | {tr[f].isna().sum()} |")
    L.append("")
    # GBM 시간순 3폴드
    tr = tr.sort_values(["date", "T"]).reset_index(drop=True)
    folds = np.array_split(np.arange(len(tr)), N_FOLDS_GBM)
    aucs, imps = [], []
    L.append(f"얕은 GBM(max_depth=2, n_estimators=100) — 'net64 > 0' 예측, 시간순 {N_FOLDS_GBM}폴드:\n")
    L.append("| 폴드 | n_train | n_test | 양성 비율(test) | AUC | 중요도 상위 3 |")
    L.append("|---|---|---|---|---|---|")
    for i in range(1, N_FOLDS_GBM):
        trn = np.concatenate(folds[:i])
        te = folds[i]
        m = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=0)
        m.fit(tr.loc[trn, FEATURES], tr.loc[trn, "y"])
        p = m.predict_proba(tr.loc[te, FEATURES])[:, 1]
        a = roc_auc_score(tr.loc[te, "y"], p) if tr.loc[te, "y"].nunique() > 1 else np.nan
        aucs.append(a)
        imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
        imps.append(imp)
        L.append(f"| {i} | {len(trn)} | {len(te)} | {tr.loc[te,'y'].mean():.0%} | {a:.3f} | "
                 + ", ".join(f"{k} {v:.2f}" for k, v in imp.head(3).items()) + " |")
    auc = float(np.nanmean(aucs)) if aucs else np.nan
    imp_mean = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False) if imps else pd.Series(dtype=float)
    L.append("")
    verdict = ("분봉으로는 선별 불가 (AUC < 0.60)" if not (auc >= GBM_AUC_MIN)
               else "AUC ≥ 0.60 — 후속 사전등록 대상으로만 기록 (채택 없음)")
    L.append(f"**평균 AUC {auc:.3f}** → {verdict}. 중요도 평균 상위 3: "
             + ", ".join(f"{k} {v:.2f}" for k, v in imp_mean.head(3).items()) + "\n")
    return {"auc": auc, "imp": imp_mean, "rho": rho_tab}


def build_report(data: dict) -> str:
    trig, main, D, excl = data["trig"], data["main"], data["D"], data["excl"]
    L: list[str] = []
    L.append(f"# 짝꿍매매 검증 — 예비 (분봉 표본, 후속 커버리지 ~63%) — {data['start']} ~ {data['end']}\n")
    L.append(f"사전등록 `docs/research/manju/12-pair-preregistration.md` 그대로. `minute_bars` {data['n_minute_codes']} 종목. "
             "비용 64bp 기본(34/94 병기). **판정은 기계적 예비 결과이며 확정이 아니다** "
             "(후속주 분봉 커버리지 ~63%, 분봉 유니버스가 일봉 급등 이벤트로 선별돼 낙관 편향 가능).\n")
    L.append("## 표본\n")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    if len(trig):
        L.append(f"| 대장 트리거 | {len(trig)} ({trig['lead'].nunique()} 종목, {trig['date'].nunique()} 거래일) |")
        L.append(f"| ≥1 클러스터 후보 트리거 | {int((trig['n_fol_cluster']>0).sum())} |")
        L.append(f"| 트리거당 클러스터 후보 수 평균 (전체 / ≥1 트리거만) | {trig['n_fol_cluster'].mean():.2f} / "
                 f"{trig.loc[trig['n_fol_cluster']>0, 'n_fol_cluster'].mean():.2f} |")
        L.append(f"| 트리거당 테마무시 후보 수 평균 | {trig['n_fol'].mean():.1f} |")
    L.append(f"| **주 규칙 거래 수** | **{len(main)}** (결과 계산 가능 {int(main['gross'].notna().sum()) if len(main) else 0}) |")
    L.append(f"| 테마무시 비교 거래 수 | {len(D)} |")
    if len(main):
        L.append(f"| 주 규칙 청산: 대장 이탈 발생 / 5분 보유 | {int(main['lead_exit'].sum())} / {int((~main['lead_exit']).sum())} |")
        L.append(f"| 주 규칙 청산 분에 봉 없어 다음 봉 시가로 판 건 | {int((main['exit_actual'] != main['exit_minute']).sum())} |")
        L.append(f"| 주 규칙 후속 5분내 상한가 도달 | {int(main['fol_hit_limit'].sum())} |")
        L.append("| fol_ret_5m 기준가 T−5 / 09:00 시가 / 전일 종가 | "
                 + " / ".join(str(int((main['fol_ret_5m_ref']==k).sum())) for k in ("T-5", "open0900", "prev_close")) + " |")
    L.append("")
    L.append("제외 사유:\n")
    L.append("| 사유 | 건수 |")
    L.append("|---|---|")
    for k, v in sorted(excl.items()):
        L.append(f"| {k} | {v} |")
    L.append("")
    res_main = stage1(main, L, "1단계 — 주 규칙 (클러스터 · 상관 최고 1건 · T+1 시가 → min(T+6, 대장 이탈+1) 시가)", judge=True)
    res_d = stage1(D, L, "보조 비교 — 테마무시 (클러스터 조건 제거 · 누적 거래대금 최고 1건) — 1회", judge=False)
    res2 = stage2(main, L) if len(main) >= 30 else {"auc": np.nan, "imp": pd.Series(dtype=float)}
    if len(main) < 30:
        L.append("## 2단계 — 거래 30건 미만이라 생략\n")
    # 선택 목록
    L.append("## 선택된 거래 목록 (주 규칙) — 눈으로 본다\n")
    L.append("| 날짜 | 대장 | 후속 | T | 후속 등락률 | 상관(순위) | 진입 | 청산 | 청산사유 | gross | 후속 상한가 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in main.sort_values(["date", "T"]).iterrows():
        why = f"대장이탈 k={r['lead_exit_k']-r['T']}" if r["lead_exit"] else "5분"
        L.append(f"| {r['date'].date()} | {r['lead_name']} | {r['fol_name']} | {int(r['T'])//60:02d}:{int(r['T'])%60:02d} | "
                 f"{r['fol_ret']:+.1%} | {r['pair_corr']:.2f} ({int(r['pair_rank'])}) | {r['entry']:,.0f} | {r['exit']:,.0f} | {why} | "
                 f"{r['gross']:+.2%} | {'Y' if r['fol_hit_limit'] else ''} |")
    L.append("")
    data["_res"] = {"main": res_main, "D": res_d, "s2": res2}
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="짝꿍매매 검증 (사전등록 12 그대로, 읽기 전용)")
    ap.add_argument("--start", default="2025-08-04")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--cache", default=None, help="거래 테이블 pickle. 있으면 읽고 없으면 만든 뒤 저장")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args(argv)

    if args.cache and os.path.exists(args.cache):
        data = pickle.loads(pathlib.Path(args.cache).read_bytes())
        print(f"캐시 로드: 주 규칙 {len(data['main'])}건 / 테마무시 {len(data['D'])}건", file=sys.stderr)
    else:
        _load_env()
        conn = connect(args.dsn or os.environ.get("KR_QUANT_DB"))
        data = collect(conn, args.start, args.end)
        if args.cache:
            pathlib.Path(args.cache).write_bytes(pickle.dumps(data))
    report = build_report(data)
    pathlib.Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
