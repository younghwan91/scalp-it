"""종가베팅 v2 검증 — 사전등록 14 (`docs/research/manju/14-closebet-v2-preregistration.md`) 그대로.

이벤트 생성(유니버스·수급·섹터·신고가·양봉·상한가 제외·금액 순위 하루 2건)은 `scripts/closebet2_events.py`
의 `load / compute / select_events` 를 **그대로 import** 한다 — 선택 로직을 여기서 다시 쓰지 않는다.
이 파일이 새로 읽는 것은 D+1 행(시가·종가)과 15:19 분봉(참고치)뿐이다.

  주 규칙    D 종가 매수 → D+1 시가 매도. 비교 1회: D+1 종가. 비용 34bp 기본, 0/21/55bp 병기
  표본       주 측정 2024-02-01 ~ 2026-08-21 (판정). 보조 2017~2023 (630종목·동일가중 섹터, 판정 불사용)
  1단계      채택 5조건 — net34>0 · 반기 6폴드 음수 ≤1 · 상위 1% 제거 net34>0 · 당일 +25% 이상 제외 net34>0 · 승률>50%
  2단계      피처 8개 스피어만·5분위, GBM(depth2·100) 연도 walk-forward AUC (채택 없음)

주 규칙은 일봉만으로 측정되므로 **이 실행이 확정 측정**이다. 파라미터·변형을 추가하지 않는다. DB 는 읽기만 한다.

    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/verify_closebet2.py [--out <scratch>/cbet2_result.md] [--no-ref]
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closebet2_events as c2  # 선택 로직은 전부 여기서 온다

# ── 비용 (06번과 동일) ──
COST = 0.0034
COST_HI = 0.0055
COST_GRID = (0.0, 0.0021, 0.0034, 0.0055)
# ── 채택 조건 파라미터 (문서 7절) ──
NEG_HALF_MAX = 1          # 반기 6폴드 중 음수 폴드 ≤ 1
TOP_TRIM = 0.01           # 상위 1% 제거
RET_DAY_EXCL = 0.25       # 당일 등락률 ≥ +25% 제외 (상따 트리거와 같은 기존 값)
WIN_MIN = 0.5
AUC_MIN = 0.60
# ── 분리 보고 ──
BIG2 = ("005930", "000660")   # 삼성전자·SK하이닉스
RESEL_DAYS = 7
ADJ_TOL = 0.002               # 분봉 15:30 종가 vs 일봉(수정) 종가 불일치 허용 — 15:19 참고치용
# ── 피처 8개 (문서 5절, 순서 고정) ──
FEATURES = ["fi_amt", "fi_share", "penfnd_pos", "sec_rel", "nh250", "box_width30", "ret_day", "opex"]
BINARY = {"penfnd_pos", "nh250", "opex"}
HYPOTHESIS = {"fi_amt": "+", "fi_share": "+", "penfnd_pos": "+", "sec_rel": "+", "nh250": "+",
              "box_width30": "−", "ret_day": "−", "opex": "+(약)"}
REF_COVERAGE = "커버리지 52~71%"


def _load_env(path: str | None) -> None:
    if not path or os.environ.get("KR_QUANT_DB"):
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


# ────────────────────────────── 청산 결합 ──────────────────────────────

def attach_exits(sel: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """선택 행에 달력상 다음 거래일(D+1) 시가·종가를 붙인다. D+1 봉 없음(거래정지·폐지)과 D+1 미도래는 따로 센다."""
    cal = pd.Series(sorted(bars["date"].unique()))
    nxt = dict(pairwise(cal))
    ev = sel.copy()
    ev["d1"] = ev["date"].map(nxt)
    n_cal_end = int(ev["d1"].isna().sum())
    ev = ev[ev["d1"].notna()]
    d1 = bars[["code", "date", "open", "close"]].rename(
        columns={"date": "d1", "open": "open_d1", "close": "close_d1"})
    ev = ev.merge(d1, on=["code", "d1"], how="left")
    no_bar = ev["open_d1"].isna() | ~(ev["open_d1"] > 0)
    n_no_bar = int(no_bar.sum())
    ev = ev[~no_bar].copy()
    ev["ret_open"] = ev["open_d1"] / ev["close"] - 1.0
    ev["ret_close"] = ev["close_d1"] / ev["close"] - 1.0
    ev["year"] = ev["date"].dt.year.astype(str)
    ev["half"] = ev["year"] + np.where(ev["date"].dt.month <= 6, "H1", "H2")
    meta = {"n_sel": len(sel), "n_cal_end": n_cal_end, "n_no_bar": n_no_bar, "n": len(ev)}
    return ev, meta


# ────────────────────────────── 통계 ──────────────────────────────

def _payoff(r: pd.Series) -> float:
    w, l = r[r > 0], r[r < 0]
    if w.empty or l.empty or l.mean() == 0:
        return float("nan")
    return float(w.mean() / abs(l.mean()))


def stats(r: pd.Series, dates: pd.Series | None = None) -> dict:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return {"n": 0}
    trimmed = r[r <= r.quantile(1 - TOP_TRIM)]
    return {
        "n": len(r), "days": int(dates.nunique()) if dates is not None else None,
        "mean": r.mean(), "net34": r.mean() - COST, "net55": r.mean() - COST_HI,
        "grid": {c: r.mean() - c for c in COST_GRID},
        "median": r.median(), "win": (r > 0).mean(), "win34": ((r - COST) > 0).mean(),
        "pl": _payoff(r), "p5": r.quantile(.05), "p25": r.quantile(.25), "p75": r.quantile(.75),
        "p95": r.quantile(.95), "min": r.min(), "max": r.max(),
        "trim_net34": trimmed.mean() - COST, "n_trim": len(r) - len(trimmed),
    }


def line(tag: str, r: pd.Series, dates: pd.Series | None = None) -> str:
    s = stats(r, dates)
    if s["n"] == 0:
        return f"  {tag}: n=0"
    per_day = f" 하루 {s['n']/s['days']:.2f}건" if s["days"] else ""
    return (f"  {tag}: n={s['n']:,}{per_day} 평균 {s['mean']:+.3%} net34 {s['net34']:+.3%} net55 {s['net55']:+.3%} "
            f"중앙 {s['median']:+.3%} 승률 {s['win']:.1%}(net34 {s['win34']:.1%}) 손익비 {s['pl']:.2f} "
            f"p5/p25/p75/p95 {s['p5']:+.1%}/{s['p25']:+.1%}/{s['p75']:+.1%}/{s['p95']:+.1%} "
            f"min/max {s['min']:+.1%}/{s['max']:+.1%} 상위1%제거 net34 {s['trim_net34']:+.3%}")


def fold_table(ev: pd.DataFrame, col: str, key: str, title: str) -> tuple[list[str], int, int]:
    L = [f"| {title} | n | 평균 | net34 | 중앙 | 승률 | 승률(net34) | 음수(net34) |", "|---|---|---|---|---|---|---|---|"]
    neg = 0
    folds = sorted(ev[key].unique())
    for f in folds:
        r = ev.loc[ev[key] == f, col]
        m = r.mean() - COST
        neg += m < 0
        L.append(f"| {f} | {len(r):,} | {r.mean():+.3%} | {m:+.3%} | {r.median():+.3%} | "
                 f"{(r > 0).mean():.1%} | {((r - COST) > 0).mean():.1%} | {'X' if m < 0 else ''} |")
    return L, int(neg), len(folds)


def neg_halves(ev: pd.DataFrame, col: str) -> tuple[int, int]:
    g = ev.groupby("half")[col].mean() - COST
    return int((g < 0).sum()), len(g)


def split_row(tag: str, ev: pd.DataFrame, col: str = "ret_open") -> str:
    if ev.empty:
        return f"| {tag} | 0 | | | | | | | |"
    s = stats(ev[col])
    nh, tot = neg_halves(ev, col)
    return (f"| {tag} | {s['n']:,} | {s['mean']:+.3%} | {s['net34']:+.3%} | {s['net55']:+.3%} | {s['median']:+.3%} | "
            f"{s['win']:.1%} ({s['win34']:.1%}) | {s['pl']:.2f} | {nh}/{tot} |")


def five_conditions(ev: pd.DataFrame, col: str = "ret_open") -> tuple[list[str], bool]:
    r = ev[col]
    s = stats(r)
    nh, tot = neg_halves(ev, col)
    excl = ev.loc[ev["ret_day"] < RET_DAY_EXCL, col]
    conds = {
        "1 net34 > 0": (s["net34"] > 0, f"{s['net34']:+.3%}"),
        f"2 반기 {tot}폴드 음수 ≤ {NEG_HALF_MAX}": (nh <= NEG_HALF_MAX, f"{nh}/{tot}"),
        "3 상위 1% 제거 net34 > 0": (s["trim_net34"] > 0, f"{s['trim_net34']:+.3%} (제거 {s['n_trim']}건)"),
        f"4 당일 +{RET_DAY_EXCL:.0%} 이상 제외 net34 > 0": (
            (excl.mean() - COST > 0) if len(excl) else False,
            f"{excl.mean() - COST:+.3%} (제외 {len(r) - len(excl)}건)"),
        f"5 승률 > {WIN_MIN:.0%} (net34 기준, 보수)": (s["win34"] > WIN_MIN, f"{s['win34']:.1%} (gross {s['win']:.1%})"),
    }
    L = ["| 조건 | 값 | 판정 |", "|---|---|---|"]
    for k, (ok, v) in conds.items():
        L.append(f"| {k} | {v} | {'통과' if ok else '**미달**'} |")
    n_ok = sum(ok for ok, _ in conds.values())
    L.append(f"| **합계** | {n_ok}/5 | {'**채택(전진 기록 착수)**' if n_ok == 5 else '**기각**'} |")
    return L, n_ok == 5


# ────────────────────────────── 15:19 참고치 ──────────────────────────────

def minute_ref(conn, ev: pd.DataFrame) -> pd.DataFrame:
    """2025-08~ 선택 건의 D 15:19 분봉 종가 진입 참고치. 15:30 동시호가 봉으로 분봉↔수정일봉 배율을 맞춘다."""
    sub = ev[ev["date"] >= pd.Timestamp(c2.MINUTE_START)]
    if sub.empty:
        return sub
    cur = conn.cursor()
    rows = []
    for code, d in zip(sub["code"], sub["date"]):
        cur.execute(
            "select close from minute_bars where code = %s and ts = %s",
            (code, d.strftime("%Y-%m-%d 15:19:00")))
        r19 = cur.fetchone()
        cur.execute(
            "select close from minute_bars where code = %s and ts = %s",
            (code, d.strftime("%Y-%m-%d 15:30:00")))
        r30 = cur.fetchone()
        rows.append({"code": code, "date": d, "px1519": r19[0] if r19 else np.nan,
                     "px1530": r30[0] if r30 else np.nan})
    cur.close()
    piv = pd.DataFrame(rows)
    out = sub.merge(piv, on=["code", "date"], how="left")
    ok = out["px1519"].notna() & out["px1530"].notna() & (out["px1519"] > 0)
    factor = out["close"] / out["px1530"]          # 수정일봉 / 원시분봉 배율 (무수정이면 1)
    match = (factor - 1).abs() <= ADJ_TOL
    out["status"] = np.select([~ok, ~match], ["분봉 없음", "15:30 불일치"], "OK")
    out["ret_1519"] = np.where(ok & match, out["open_d1"] / (out["px1519"] * factor) - 1.0, np.nan)
    out["slip_1519"] = np.where(ok & match, out["close"] / (out["px1519"] * factor) - 1.0, np.nan)  # 종가/15:19 − 1
    return out


# ────────────────────────────── 2단계 ──────────────────────────────

def stage2(ev: pd.DataFrame) -> list[str]:
    L: list[str] = []
    x = ev.copy()
    x["penfnd_pos"] = (x["penfnd_etc"] > 0).astype(float).where(x["penfnd_etc"].notna())
    x["box_width30"] = x["box_width"]
    x["nh250"] = x["nh250"].astype(float)
    x["opex"] = x["opex"].astype(float)
    x["y"] = (x["ret_open"] > 0).astype(int)
    L.append("### 단변량 — 스피어만 상관(vs D+1 시가 수익률) · 5분위 평균/승률 (Q1 = 낮음)\n")
    L.append("| 피처 | 가설 | n | ρ | p | Q1 | Q2 | Q3 | Q4 | Q5 | 단조 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in FEATURES:
        s = x[[f, "ret_open"]].dropna()
        if len(s) < 20:
            L.append(f"| {f} | {HYPOTHESIS[f]} | {len(s)} | | | | | | | | |")
            continue
        rho, p = spearmanr(s[f], s["ret_open"])
        if f in BINARY:
            cells = []
            for v in (0.0, 1.0):
                g = s.loc[s[f] == v, "ret_open"]
                cells.append(f"{int(v)}: {g.mean():+.2%}/{(g > 0).mean():.0%} (n{len(g)})" if len(g) else f"{int(v)}: —")
            L.append(f"| {f} | {HYPOTHESIS[f]} | {len(s)} | {rho:+.3f} | {p:.3f} | {cells[0]} | {cells[1]} | | | | 이진 |")
            continue
        q = pd.qcut(s[f].rank(method="first"), 5, labels=False) + 1
        means = s.groupby(q)["ret_open"].mean()
        wins = s.groupby(q)["ret_open"].apply(lambda v: (v > 0).mean())
        diffs = np.diff(means.values)
        mono = "↑" if (diffs > 0).all() else "↓" if (diffs < 0).all() else "아니오"
        cells = " | ".join(f"{means[i]:+.2%}/{wins[i]:.0%}" for i in range(1, 6))
        L.append(f"| {f} | {HYPOTHESIS[f]} | {len(s)} | {rho:+.3f} | {p:.3f} | {cells} | {mono} |")
    L.append("")

    L.append("### GBM(max_depth 2 · 100 trees) — `ret_open > 0`, 연도 walk-forward\n")
    z = x.dropna(subset=FEATURES)
    L.append(f"피처 결측 제외 {len(x) - len(z):,}건 → {len(z):,}건 · 양성률 {z['y'].mean():.1%}\n")
    L.append("| 폴드 | train | test | AUC | 중요도 상위 3 |")
    L.append("|---|---|---|---|---|")
    aucs, imps = [], []
    for train_years, test_year in ((["2024"], "2025"), (["2024", "2025"], "2026")):
        tr, te = z[z["year"].isin(train_years)], z[z["year"] == test_year]
        if len(tr) < 50 or te["y"].nunique() < 2:
            L.append(f"| {'+'.join(train_years)}→{test_year} | {len(tr)} | {len(te)} | 건너뜀 | |")
            continue
        mdl = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=0)
        mdl.fit(tr[FEATURES], tr["y"])
        p = mdl.predict_proba(te[FEATURES])[:, 1]
        a = roc_auc_score(te["y"], p)
        aucs.append(a)
        imp = pd.Series(mdl.feature_importances_, index=FEATURES).sort_values(ascending=False)
        imps.append(imp)
        top3 = ", ".join(f"{k} {v:.2f}" for k, v in imp.head(3).items())
        L.append(f"| {'+'.join(train_years)}→{test_year} | {len(tr):,} | {len(te):,} | {a:.3f} | {top3} |")
    if aucs:
        mean_auc = float(np.mean(aucs))
        imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
        L.append("")
        L.append(f"AUC 평균 {mean_auc:.3f} (폴드 {len(aucs)}) · 중요도(폴드 평균) " +
                 ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
        verdict = ("AUC < 0.60 → **선별 가능성 없음** (앞선 6기법 0.48~0.56 과 같은 자리)" if mean_auc < AUC_MIN
                   else "AUC ≥ 0.60 — 채택 아님, **후속 사전등록 대상**으로만 기록")
        L.append(f"판정: {verdict}")
        if imp.index[0] == "penfnd_pos":
            L.append("주의: 가장 강한 피처가 `penfnd_pos`(연기금) — 15:20 에 모르는 변수, 실전 변수 아님.")
    return L


# ────────────────────────────── main ──────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--out", default=None, help="결과 markdown 경로")
    ap.add_argument("--no-ref", action="store_true", help="보조 측정(2017~2023) 생략")
    args = ap.parse_args()
    _load_env(args.env_file)

    conn = c2._connect()
    start = c2.MAIN_START if args.no_ref else c2.REF_START
    daily, sd, stocks, si = c2.load(conn, start)
    # D+1 청산용 — END 이후 봉이 있으면 붙인다 (이벤트 생성에는 쓰지 않는다)
    extra = pd.read_sql(
        "select code, date, open, high, low, close, volume, trade_value from daily_bars_adjusted where date > %(end)s",
        conn, params={"end": c2.END})
    extra["date"] = pd.to_datetime(extra["date"])
    bars = pd.concat([daily, extra], ignore_index=True) if len(extra) else daily

    O: list[str] = []
    O.append("# 종가베팅 v2 — **확정 측정(일봉)** (사전등록 14 그대로)\n")
    O.append(f"주 규칙: 거래대금 ≥ {c2.TV_FLOOR_MN/100:.0f}억 · 외국인+기관계 순매수 > 0 · (업종지수 > 시장지수 OR 종가 > 직전 "
             f"{c2.LOOKBACK}일 고가) · 양봉 · 상한가 마감 선택 전 제외 · 순매수 금액 순 하루 {c2.TOP_N}건 · D 종가 → D+1 시가 · "
             f"비용 34bp(0/21/55 병기). 기간 {c2.MAIN_START} ~ {c2.END}. 일봉 최종일 {bars['date'].max().date()}.\n")

    # ---- 주 측정 ----
    w = c2.compute(daily, sd, stocks, si, c2.MAIN_START, "index")
    _base, pool, sel = c2.select_events(w, c2.TV_FLOOR_MN)
    pool_ok = pool[~pool["limit_up"]]
    ev, meta = attach_exits(sel, bars)
    univ_all = w[w["common"] & (w["trade_value"] >= c2.TV_FLOOR_MN)]
    n_delisted = int((~univ_all["listed_now"]).sum())
    O.append("## 1단계 — 규칙 성과 (주 측정)\n")
    O.append(f"선택 {meta['n_sel']:,}건 · D+1 미도래(달력 끝) {meta['n_cal_end']} · D+1 봉 없음(거래정지·폐지) {meta['n_no_bar']} → "
             f"**측정 n = {meta['n']:,}** · 거래일 {ev['date'].nunique():,} · 종목 {ev['code'].nunique():,}. "
             f"(참고: 거래대금 500억 이상 (종목,일자) 중 현재 비상장 {n_delisted:,}건 = "
             f"{n_delisted/max(len(univ_all),1):.1%} — 업종 없음으로 유니버스 밖, 생존편향 방향)\n")
    O.append("**주 규칙 D 종가 → D+1 시가**\n")
    O.append(line("전체", ev["ret_open"], ev["date"]))
    O.append("  비용 4단계: " + ", ".join(f"{c*1e4:.0f}bp {ev['ret_open'].mean() - c:+.3%}" for c in COST_GRID))
    O.append("")
    L, neg_y, ny = fold_table(ev, "ret_open", "year", "연도")
    O += L
    O.append(f"\n연도 음수 {neg_y}/{ny}\n")
    L, neg_h, nh = fold_table(ev, "ret_open", "half", "반기")
    O += L
    O.append(f"\n**반기 음수 {neg_h}/{nh}**\n")

    O.append("### 분리 보고 (D+1 시가)\n")
    O.append("| 분리 | n | 평균 | net34 | net55 | 중앙 | 승률 (net34) | 손익비 | 반기 음수 |")
    O.append("|---|---|---|---|---|---|---|---|---|")
    O.append(split_row("전체", ev))
    O.append(split_row(f"당일 +{RET_DAY_EXCL:.0%} 이상 제외", ev[ev["ret_day"] < RET_DAY_EXCL]))
    O.append(split_row(f"당일 +{RET_DAY_EXCL:.0%} 이상만", ev[ev["ret_day"] >= RET_DAY_EXCL]))
    O.append(split_row("삼성전자·SK하이닉스 제외", ev[~ev["code"].isin(BIG2)]))
    O.append(split_row("삼성전자·SK하이닉스만", ev[ev["code"].isin(BIG2)]))
    O.append(split_row("신규주(<250일)", ev[ev["is_new"]]))
    O.append(split_row("비신규주", ev[~ev["is_new"]]))
    O.append(split_row("섹터만", ev[ev["sec_strong"] & ~ev["nh250"]]))
    O.append(split_row("신고가만", ev[~ev["sec_strong"] & ev["nh250"]]))
    O.append(split_row("둘 다", ev[ev["sec_strong"] & ev["nh250"]]))
    ev_s = ev.sort_values(["code", "date"])
    gap = (ev_s["date"] - ev_s.groupby("code")["date"].shift(1)).dt.days
    resel = gap.reindex(ev.index) <= RESEL_DAYS
    O.append(split_row(f"같은 종목 {RESEL_DAYS}일 내 재선택", ev[resel]))
    O.append(split_row("재선택 아님", ev[~resel]))
    O.append(split_row("거래소", ev[ev["market"] == "거래소"]))
    O.append(split_row("코스닥", ev[ev["market"] == "코스닥"]))
    O.append(split_row("옵션만기일", ev[ev["opex"]]))
    O.append(split_row("하루 1위만", ev[ev["rank"] == 1]))
    O.append(split_row("하루 2위만", ev[ev["rank"] == 2]))
    O.append("")
    bins = [-1, 0, .05, .10, .20, .25, .295, 10]
    dist = ev.groupby(pd.cut(ev["ret_day"], bins), observed=False)["ret_open"].agg(["size", "mean"])
    O.append("당일 등락률 구간별 (n / D+1 시가 평균): " + " · ".join(
        f"{k} {int(v['size'])}/{v['mean']:+.2%}" if v["size"] else f"{k} 0" for k, v in dist.iterrows()) + "\n")

    # 비교 1회: D+1 종가
    O.append("**비교 1회 — D 종가 → D+1 종가**\n")
    O.append(line("전체", ev["ret_close"], ev["date"]))
    L, neg_hc, nhc = fold_table(ev, "ret_close", "half", "반기(D+1 종가)")
    O += L
    O.append(f"\n반기 음수 {neg_hc}/{nhc}\n")

    # 진단: 제외 후 풀 전체 동일가중
    pev, pmeta = attach_exits(pool_ok, bars)
    O.append("**(진단, 판정 불사용) 상한가 제외 후 풀 전체 동일가중 — 변수 자체의 부호**\n")
    O.append(line("풀 전체 D+1 시가", pev["ret_open"], pev["date"]))
    O.append(line("풀 전체 D+1 종가", pev["ret_close"], pev["date"]))
    pneg, pn = neg_halves(pev, "ret_open")
    O.append(f"  반기 음수(시가, net34) {pneg}/{pn} · D+1 봉 없음 {pmeta['n_no_bar']} · 미도래 {pmeta['n_cal_end']}\n")

    # 15:19 참고치
    mr = minute_ref(conn, ev)
    O.append(f"**15:19 분봉 종가 진입 참고치 ({c2.MINUTE_START}~, 규칙 변경 아님)**\n")
    if len(mr):
        okm = mr[mr["status"] == "OK"]
        O.append(f"  대상 {len(mr):,} · OK {len(okm):,} · 분봉 없음 {(mr['status'] == '분봉 없음').sum()} · "
                 f"15:30 불일치 {(mr['status'] == '15:30 불일치').sum()}")
        if len(okm):
            O.append(line("15:19 진입 → D+1 시가", okm["ret_1519"], okm["date"]))
            O.append(line("같은 건 종가 진입 → D+1 시가", okm["ret_open"], okm["date"]))
            O.append(f"  종가/15:19 − 1: 평균 {okm['slip_1519'].mean():+.3%} 중앙 {okm['slip_1519'].median():+.3%} "
                     f"|절대| 평균 {okm['slip_1519'].abs().mean():.3%}")
    O.append("")

    # 5조건
    O.append("### 채택 5조건 (34bp, D+1 시가) — 기계적 판정\n")
    L, passed = five_conditions(ev)
    O += L
    O.append("")
    O.append("**판정: " + ("채택 — 전진 기록 착수(잠정 통과: 확정 수급이 15:19 잠정치의 대리, 룩어헤드 낙관 방향 — "
                         "전진 기록에서 잠정 vs 확정 일치율 확인 전까지)" if passed else
                         "**기각** — 종가베팅 v2 를 접는다. 파라미터·요소를 바꿔 재시도하지 않는다") + "**\n")
    r_all = ev["ret_open"].sort_values()
    k = int(len(r_all) * TOP_TRIM)
    O.append("승률 조건은 문서에 gross/net 명시가 없어 **보수(net34 > 0) 기준**으로 판정하고 gross 를 병기했다. "
             f"상위 1% 제거는 앞선 스크립트와 같은 `r <= r.quantile(0.99)` (참고: 정확히 상위 {k}건 제거 시 net34 "
             f"{r_all.iloc[:len(r_all) - k].mean() - COST:+.3%}).\n")

    # ---- 2단계 ----
    O.append("## 2단계 — 피처 진단 (채택 없음)\n")
    O += stage2(ev)
    O.append("")

    # ---- 보조 측정 ----
    if not args.no_ref:
        w2 = c2.compute(daily, sd, stocks, si, c2.REF_START, "eqw")
        w2 = w2[w2["date"] < pd.Timestamp(c2.MAIN_START)]
        _b2, _p2, s2 = c2.select_events(w2, c2.TV_FLOOR_MN)
        ev2, m2 = attach_exits(s2, bars)
        ev2["year"] = ev2["year"].where(ev2["year"] != "2024", "2024(01월만)")
        O.append(f"## 보조 측정 {c2.REF_START} ~ {c2.MAIN_START} (630종목 수급 · 동일가중 섹터 근사 · {REF_COVERAGE} · **판정 불사용**)\n")
        O.append(f"선택 {m2['n_sel']:,} · D+1 봉 없음 {m2['n_no_bar']} → n {m2['n']:,}\n")
        O.append(line("전체 D+1 시가", ev2["ret_open"], ev2["date"]))
        O.append(line("전체 D+1 종가", ev2["ret_close"], ev2["date"]))
        O.append("")
        L, neg2, ny2 = fold_table(ev2, "ret_open", "year", f"연도 ({REF_COVERAGE})")
        O += L
        O.append(f"\n보조: 부호 {'양' if ev2['ret_open'].mean() - COST > 0 else '음'}(net34 {ev2['ret_open'].mean() - COST:+.3%}) · "
                 f"연도 음수 {neg2}/{ny2} · 삼전·하이닉스 제외 net34 "
                 f"{ev2.loc[~ev2['code'].isin(BIG2), 'ret_open'].mean() - COST:+.3%} (n {int((~ev2['code'].isin(BIG2)).sum()):,})\n")
    conn.close()

    text = "\n".join(O)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\n→ {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
