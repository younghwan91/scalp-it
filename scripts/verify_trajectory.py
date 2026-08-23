"""장중 궤적→익일 갭 검증 — 사전등록 16 (`docs/research/manju/16-intraday-trajectory-preregistration.md`) 그대로.

유니버스는 `scripts/traj_events.py` 의 `load_daily / minute_days / build_universe` 를 **그대로 import** 한다
(선택 로직을 여기서 다시 쓰지 않는다). 피처는 `scalp_it.intraday.features.intraday_features` 그대로 (8개, 15:19 계약).
이 파일이 새로 읽는 것은 D+1 행(수정 시가)과 유니버스 (종목,일자)의 분봉뿐.

  거래      D 종가(수정 일봉) 매수 → D+1 시가(수정 일봉) 매도. 비용 34bp 기본, 0/21/55bp 병기
  1단계     전부 매수 기준선. 월 13폴드. 채택 5조건 — net34>0 · 월 음수 ≤3 · 상위1% 제거 net34>0 · 당일 +25%↑ 제외 net34>0 · 승률(net34)>50%
  2단계     GBM(depth2·100) 피처 8개, 월별 확장 폴드(최소 학습 2개월 → 시험 11폴드), AUC 게이트 평균 ≥ 0.60
            (a) 폴드 안 예측 상위 1/3 = 주 판정 · (b) 하루 상위 3 = 실행 형태(참고, 착수 조건 net34>0)

파라미터·변형을 추가하지 않는다. DB 는 읽기만 한다. 분봉 로딩은 종목 묶음 병렬 + `--cache` 디렉터리 재사용.

    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/verify_trajectory.py --cache <scratch>/traj_cache --out <scratch>/traj_result.md
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import traj_events as te  # 유니버스는 전부 여기서 온다

from scalp_it.intraday.features import INTRADAY_FEATURE_COLUMNS, intraday_features

# ── 비용 (06번과 동일) ──
COST = 0.0034
COST_HI = 0.0055
COST_GRID = (0.0, 0.0021, 0.0034, 0.0055)
# ── 채택 조건 파라미터 (문서 "판정 절차") ──
NEG_MONTH_MAX = 3
TOP_TRIM = 0.01
RET_DAY_EXCL = 0.25
WIN_MIN = 0.5
AUC_MIN = 0.60
MIN_TRAIN_MONTHS = 2
TOP_FRAC = 1 / 3
TOP_N_DAY = 3
N_SHOW = 30
CLOSE_TOL = 0.002   # 분봉 15:30 종가 vs 수정 일봉 종가 불일치 (참고)
FEATURES = list(INTRADAY_FEATURE_COLUMNS)
HYPOTHESIS = {"late_ret": "+", "high_time_pct": "+", "tv_pm_share": "+", "n_shock": "−",
              "max_dd": "−", "recovery": "+", "trend_r2": "+", "close_vs_vwap": "+"}
RHO_SMALL = 0.05
CHUNK = 8   # 종목 묶음 크기 (분봉 질의 단위)


# ────────────────────────────── 분봉 → 피처 (병렬) ──────────────────────────────

def _feature_chunk(codes: list[str], wanted: set[tuple[str, pd.Timestamp]], cache_dir: str | None) -> pd.DataFrame:
    """종목 묶음의 분봉을 읽어 유니버스 (종목,일자)만 남기고 피처를 계산한다. 워커 프로세스에서 돈다."""
    key = f"{codes[0]}_{codes[-1]}_{len(codes)}"
    if cache_dir:
        p = Path(cache_dir) / f"feat_{key}.pkl"
        if p.exists():
            return pd.read_pickle(p)
    conn = te.connect()
    bars = pd.read_sql(
        "select code, ts, open, high, low, close, volume, trade_value from minute_bars "
        "where code = any(%(codes)s) and ts >= %(s)s and ts < %(e)s and ts::time <= '15:30' order by code, ts",
        conn, params={"codes": list(codes), "s": te.START,
                      "e": (pd.Timestamp(te.END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")})
    conn.close()
    if bars.empty:
        out = pd.DataFrame(columns=["code", "date", *FEATURES])
    else:
        bars["code"] = bars["code"].astype(str).str.strip()
        bars["ts"] = pd.to_datetime(bars["ts"])
        day = bars["ts"].dt.normalize()
        keep = pd.Series(list(zip(bars["code"], day)), index=bars.index).isin(wanted)
        bars = bars[keep.to_numpy()]
        out = intraday_features(bars) if len(bars) else pd.DataFrame(columns=["code", "date", *FEATURES])
    if cache_dir:
        out.to_pickle(Path(cache_dir) / f"feat_{key}.pkl")
    return out


def build_features(univ: pd.DataFrame, cache_dir: str | None, workers: int) -> pd.DataFrame:
    wanted = set(zip(univ["code"], univ["date"]))
    codes = sorted(univ["code"].unique())
    chunks = [codes[i:i + CHUNK] for i in range(0, len(codes), CHUNK)]
    parts: list[pd.DataFrame] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_feature_chunk, c, wanted, cache_dir) for c in chunks]
        for f in as_completed(futs):
            parts.append(f.result())
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f"  피처 {done}/{len(futs)} 묶음", file=sys.stderr)
    feat = pd.concat([p for p in parts if len(p)], ignore_index=True)
    feat["date"] = pd.to_datetime(feat["date"])
    return feat


# ────────────────────────────── 청산 결합 ──────────────────────────────

def load_adj_open(conn) -> pd.DataFrame:
    df = pd.read_sql(
        "select code, date, open, close from daily_bars_adjusted where date >= %(s)s order by code, date",
        conn, params={"s": te.START})
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_exits(univ: pd.DataFrame, adj: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cal = pd.Series(sorted(adj["date"].unique()))
    nxt = dict(pairwise(cal))
    ev = univ.copy()
    ev["d1"] = ev["date"].map(nxt)
    n_cal_end = int(ev["d1"].isna().sum())
    ev = ev[ev["d1"].notna()]
    d1 = adj.rename(columns={"date": "d1", "open": "open_d1", "close": "close_d1"})
    ev = ev.merge(d1, on=["code", "d1"], how="left")
    no_bar = ev["open_d1"].isna() | ~(ev["open_d1"] > 0)
    n_no_bar = int(no_bar.sum())
    ev = ev[~no_bar].copy()
    ev["ret_open"] = ev["open_d1"] / ev["close_adj"] - 1.0
    ev["y"] = (ev["ret_open"] > 0).astype(int)
    return ev, {"n_univ": len(univ), "n_cal_end": n_cal_end, "n_no_bar": n_no_bar, "n": len(ev)}


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
        "median": r.median(), "win": (r > 0).mean(), "win34": ((r - COST) > 0).mean(),
        "pl": _payoff(r), "p5": r.quantile(.05), "p25": r.quantile(.25), "p75": r.quantile(.75),
        "p95": r.quantile(.95), "min": r.min(), "max": r.max(),
        "trim_net34": trimmed.mean() - COST, "n_trim": len(r) - len(trimmed),
    }


def line(tag: str, r: pd.Series, dates: pd.Series | None = None) -> str:
    s = stats(r, dates)
    if s["n"] == 0:
        return f"  {tag}: n=0"
    per_day = f" 하루 {s['n']/s['days']:.1f}건" if s["days"] else ""
    return (f"  {tag}: n={s['n']:,}{per_day} 평균 {s['mean']:+.3%} net34 {s['net34']:+.3%} net55 {s['net55']:+.3%} "
            f"중앙 {s['median']:+.3%} 승률 {s['win']:.1%}(net34 {s['win34']:.1%}) 손익비 {s['pl']:.2f} "
            f"p5/p25/p75/p95 {s['p5']:+.1%}/{s['p25']:+.1%}/{s['p75']:+.1%}/{s['p95']:+.1%} "
            f"min/max {s['min']:+.1%}/{s['max']:+.1%} 상위1%제거 net34 {s['trim_net34']:+.3%}")


def fold_table(ev: pd.DataFrame, title: str, extra: dict[str, pd.Series] | None = None) -> tuple[list[str], int, int]:
    cols = ["n", "평균", "net34", "중앙", "승률", "승률(net34)", "음수(net34)"] + (list(extra) if extra else [])
    L = [f"| {title} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    neg = 0
    months = sorted(ev["month"].unique())
    for m in months:
        r = ev.loc[ev["month"] == m, "ret_open"]
        net = r.mean() - COST
        neg += net < 0
        row = (f"| {m} | {len(r):,} | {r.mean():+.3%} | {net:+.3%} | {r.median():+.3%} | "
               f"{(r > 0).mean():.1%} | {((r - COST) > 0).mean():.1%} | {'X' if net < 0 else ''} |")
        if extra:
            row += "".join(f" {extra[k].get(m, '')} |" for k in extra)
        L.append(row)
    return L, int(neg), len(months)


def neg_months(ev: pd.DataFrame) -> tuple[int, int]:
    g = ev.groupby("month")["ret_open"].mean() - COST
    return int((g < 0).sum()), len(g)


def split_row(tag: str, ev: pd.DataFrame) -> str:
    if ev.empty:
        return f"| {tag} | 0 | | | | | | | |"
    s = stats(ev["ret_open"], ev["date"])
    nm, tot = neg_months(ev)
    return (f"| {tag} | {s['n']:,} | {s['mean']:+.3%} | {s['net34']:+.3%} | {s['net55']:+.3%} | {s['median']:+.3%} | "
            f"{s['win']:.1%} ({s['win34']:.1%}) | {s['pl']:.2f} | {nm}/{tot} |")


SPLIT_HEADER = ["| 분리 | n | 평균 | net34 | net55 | 중앙 | 승률 (net34) | 손익비 | 월 음수 |",
                "|---|---|---|---|---|---|---|---|---|"]


def split_block(ev: pd.DataFrame, full: pd.DataFrame | None = None) -> list[str]:
    """공통 분리 보고. ``full`` 이 있으면 상·하한가 포함 행(유니버스 정의 확인)도 붙인다."""
    L = list(SPLIT_HEADER)
    L.append(split_row("전체", ev))
    if full is not None:
        L.append(split_row("상한가 마감 포함 (제외 전)", full[~full["limit_dn"]]))
        L.append(split_row("상한가 마감만", full[full["limit_up"]]))
        L.append(split_row("하한가 마감만", full[full["limit_dn"]]))
    L.append(split_row(f"당일 +{RET_DAY_EXCL:.0%} 이상 제외", ev[ev["ret_day"] < RET_DAY_EXCL]))
    L.append(split_row(f"당일 +{RET_DAY_EXCL:.0%} 이상만", ev[ev["ret_day"] >= RET_DAY_EXCL]))
    L.append(split_row("삼성전자·SK하이닉스 제외", ev[~ev["big2"]]))
    L.append(split_row("삼성전자·SK하이닉스만", ev[ev["big2"]]))
    L.append(split_row(f"거래대금 ≥ {te.TV_REF_MN/100:.0f}억", ev[ev["tv_ref"]]))
    L.append(split_row(f"거래대금 < {te.TV_REF_MN/100:.0f}억", ev[~ev["tv_ref"]]))
    L.append(split_row("신규주(첫 거래일 ≥ 2025-01-01)", ev[ev["is_new"]]))
    L.append(split_row("비신규주", ev[~ev["is_new"]]))
    L.append(split_row("거래소", ev[ev["market"] == "거래소"]))
    L.append(split_row("코스닥", ev[ev["market"] == "코스닥"]))
    return L


def five_conditions(ev: pd.DataFrame, n_folds_label: str) -> tuple[list[str], bool]:
    r = ev["ret_open"]
    s = stats(r)
    nm, tot = neg_months(ev)
    excl = ev.loc[ev["ret_day"] < RET_DAY_EXCL, "ret_open"]
    conds = {
        "1 net34 > 0": (s["net34"] > 0, f"{s['net34']:+.3%}"),
        f"2 월 {tot}폴드 음수 ≤ {NEG_MONTH_MAX}": (nm <= NEG_MONTH_MAX, f"{nm}/{tot}"),
        "3 상위 1% 제거 net34 > 0": (s["trim_net34"] > 0, f"{s['trim_net34']:+.3%} (제거 {s['n_trim']:,}건)"),
        f"4 당일 +{RET_DAY_EXCL:.0%} 이상 제외 net34 > 0": (
            (excl.mean() - COST > 0) if len(excl) else False,
            f"{excl.mean() - COST:+.3%} (제외 {len(r) - len(excl):,}건)"),
        f"5 승률 > {WIN_MIN:.0%} (net34 기준)": (s["win34"] > WIN_MIN, f"{s['win34']:.1%} (gross {s['win']:.1%})"),
    }
    L = [f"| 조건 ({n_folds_label}) | 값 | 판정 |", "|---|---|---|"]
    for k, (ok, v) in conds.items():
        L.append(f"| {k} | {v} | {'통과' if ok else '**미달**'} |")
    n_ok = sum(ok for ok, _ in conds.values())
    L.append(f"| **합계** | {n_ok}/5 | {'**전부 통과**' if n_ok == 5 else '**미달**'} |")
    return L, n_ok == 5


# ────────────────────────────── 단변량 ──────────────────────────────

def univariate(ev: pd.DataFrame) -> tuple[list[str], int]:
    L = ["| 피처 | 가설 | n | ρ | p | Q1 | Q2 | Q3 | Q4 | Q5 | 단조 | 가설 일치 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    n_match = 0
    for f in FEATURES:
        s = ev[[f, "ret_open"]].dropna()
        rho, p = spearmanr(s[f], s["ret_open"])
        sign = "+" if rho > 0 else "−"
        hyp = HYPOTHESIS[f]
        if sign == hyp and abs(rho) >= RHO_SMALL:
            verdict = "**일치**"
            n_match += 1
        elif sign == hyp:
            verdict = "방향만 (|ρ|<0.05)"
        else:
            verdict = "반대"
        q = pd.qcut(s[f].rank(method="first"), 5, labels=False) + 1
        means = s.groupby(q)["ret_open"].mean()
        wins = s.groupby(q)["ret_open"].apply(lambda v: (v > 0).mean())
        diffs = np.diff(means.values)
        mono = "↑" if (diffs > 0).all() else "↓" if (diffs < 0).all() else "아니오"
        cells = " | ".join(f"{means[i]:+.2%}/{wins[i]:.0%}" for i in range(1, 6))
        L.append(f"| {f} | {hyp} | {len(s):,} | {rho:+.3f} | {p:.1e} | {cells} | {mono} | {verdict} |")
    return L, n_match


# ────────────────────────────── 2단계 ──────────────────────────────

def walk_forward(ev: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """월별 확장 폴드. 시험 월 m 의 학습 = m 이전 전 월. 예측은 폴드 안에서만 쓴다(전역 풀링 금지)."""
    z = ev.dropna(subset=FEATURES).copy()
    months = sorted(z["month"].unique())
    preds, folds = [], []
    for i in range(MIN_TRAIN_MONTHS, len(months)):
        m = months[i]
        tr, tst = z[z["month"].isin(months[:i])], z[z["month"] == m]
        mdl = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=0)
        mdl.fit(tr[FEATURES], tr["y"])
        p = mdl.predict_proba(tst[FEATURES])[:, 1]
        auc = roc_auc_score(tst["y"], p) if tst["y"].nunique() == 2 else float("nan")
        imp = pd.Series(mdl.feature_importances_, index=FEATURES).sort_values(ascending=False)
        folds.append({"month": m, "n_train": len(tr), "n_test": len(tst), "auc": auc, "imp": imp,
                      "pos_rate": tst["y"].mean()})
        out = tst.copy()
        out["pred"] = p
        # 폴드 안 순위 (상위 1/3) 와 하루 순위 (상위 N). 동률은 거래대금 큰 쪽 우선. 전역 풀링 없음
        out = out.sort_values(["pred", "trade_value"], ascending=[False, False], kind="stable")
        out["rank_fold"] = np.arange(1, len(out) + 1)
        out["rank_day"] = out.groupby("date").cumcount() + 1
        out["top_third"] = out["rank_fold"] <= np.ceil(len(out) * TOP_FRAC)
        out["top_day"] = out["rank_day"] <= TOP_N_DAY
        preds.append(out)
    return pd.concat(preds, ignore_index=True), folds


def stage2(ev: pd.DataFrame) -> tuple[list[str], dict]:
    L: list[str] = []
    n_nan = int(ev[FEATURES].isna().any(axis=1).sum())
    L.append(f"피처 결측 제외 {n_nan:,}건 (late_ret NaN 등) → {len(ev) - n_nan:,}건 · 양성률 {ev['y'].mean():.1%}\n")
    pred, folds = walk_forward(ev)
    L.append("### GBM(max_depth 2 · 100 trees) — `ret_open > 0`, 월별 확장 폴드\n")
    L.append("| 시험 월 | train | test | 양성률 | AUC | 중요도 상위 3 |")
    L.append("|---|---|---|---|---|---|")
    for f in folds:
        top3 = ", ".join(f"{k} {v:.2f}" for k, v in f["imp"].head(3).items())
        L.append(f"| {f['month']} | {f['n_train']:,} | {f['n_test']:,} | {f['pos_rate']:.1%} | {f['auc']:.3f} | {top3} |")
    aucs = [f["auc"] for f in folds if np.isfinite(f["auc"])]
    mean_auc = float(np.mean(aucs))
    imp = pd.concat([f["imp"] for f in folds], axis=1).mean(axis=1).sort_values(ascending=False)
    L.append("")
    L.append(f"**AUC 평균 {mean_auc:.3f}** (폴드 {len(aucs)}, 최소 {min(aucs):.3f} 최대 {max(aucs):.3f}) · "
             f"AUC ≥ 0.60 인 폴드 {sum(a >= AUC_MIN for a in aucs)}/{len(aucs)} · 중요도(폴드 평균) " +
             ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    gate = mean_auc >= AUC_MIN
    L.append(f"**AUC 게이트 ({AUC_MIN:.2f}): {'통과' if gate else '미달 — 선별 가능성 없음 (6기법 0.48~0.56 과 같은 자리)'}**\n")

    res = {"mean_auc": mean_auc, "gate": gate}
    for tag, col, label in (("(a) 폴드 안 예측 상위 1/3 — 주 판정", "top_third", "11폴드"),
                            (f"(b) 하루 예측 상위 {TOP_N_DAY} — 실행 형태", "top_day", "11폴드")):
        sel = pred[pred[col]]
        rest = pred[~pred[col]]
        L.append(f"### {tag}\n")
        L.append(line("선택", sel["ret_open"], sel["date"]))
        L.append(line("비선택(같은 폴드)", rest["ret_open"], rest["date"]))
        L.append(line("폴드 전체(2단계 구간)", pred["ret_open"], pred["date"]))
        L.append("  비용 4단계(선택): " + ", ".join(f"{c*1e4:.0f}bp {sel['ret_open'].mean() - c:+.3%}" for c in COST_GRID))
        L.append("")
        extra = {"비선택 net34": (rest.groupby("month")["ret_open"].mean() - COST).map(lambda v: f"{v:+.3%}")}
        T, _neg, _tot = fold_table(sel, "시험 월", extra)
        L += T
        L.append("")
        L += split_block(sel)
        per_code = sel.groupby("code").size().sort_values(ascending=False)
        L.append("")
        L.append(f"선택 종목당 건수: 중앙 {per_code.median():.0f} · p95 {per_code.quantile(.95):.0f} · 최대 {per_code.max()} · "
                 f"종목 수 {sel['code'].nunique():,} · 상위 5: " +
                 ", ".join(f"{c}({sel.loc[sel['code'] == c, 'name'].iloc[0]}) {n}" for c, n in per_code.head(5).items()))
        L.append("")
        C, passed = five_conditions(sel, label)
        L += C
        L.append("")
        res[col] = {"passed": passed, "net34": sel["ret_open"].mean() - COST, "n": len(sel)}
        if col == "top_day":
            L.append(f"#### (b) 예측 확률 상위 {N_SHOW}건 — 눈으로 본다 (아카이브 실패 6)\n")
            show = sel.sort_values("pred", ascending=False).head(N_SHOW)
            L.append("| # | 종목 | 일자 | pred | 당일등락 | 거래대금 | late_ret | high_t | tv_pm | shock | max_dd | recov | r2 | vs_vwap | D+1 시가 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for i, (_, r) in enumerate(show.iterrows(), 1):
                L.append(f"| {i} | {r['name']} {r['code']} | {r['date'].date()} | {r['pred']:.3f} | {r['ret_day']:+.1%} | "
                         f"{r['trade_value']/100:,.0f}억 | {r['late_ret']:+.2%} | {r['high_time_pct']:.2f} | {r['tv_pm_share']:.2f} | "
                         f"{int(r['n_shock'])} | {r['max_dd']:.1%} | {r['recovery']:.1%} | {r['trend_r2']:+.2f} | "
                         f"{r['close_vs_vwap']:+.1%} | **{r['ret_open']:+.2%}** |")
            L.append("")
            L.append(f"상위 {N_SHOW}건 요약: 평균 D+1 시가 {show['ret_open'].mean():+.2%} · 승률 {(show['ret_open'] > 0).mean():.0%} · "
                     f"당일 등락 중앙 {show['ret_day'].median():+.1%} · +25% 이상 {int((show['ret_day'] >= RET_DAY_EXCL).sum())}건 · "
                     f"종목 {show['code'].nunique()}개 · 거래대금 중앙 {show['trade_value'].median()/100:,.0f}억\n")
            # 선택 전체의 성격
            L.append(f"(b) 선택 전체 {len(sel):,}건의 성격: 당일 등락 평균 {sel['ret_day'].mean():+.2%}(유니버스 {pred['ret_day'].mean():+.2%}) · "
                     f"+25% 이상 {int((sel['ret_day'] >= RET_DAY_EXCL).sum())}건({(sel['ret_day'] >= RET_DAY_EXCL).mean():.1%}) · "
                     f"거래대금 중앙 {sel['trade_value'].median()/100:,.0f}억(유니버스 {pred['trade_value'].median()/100:,.0f}억) · "
                     f"late_ret 평균 {sel['late_ret'].mean():+.2%}(유니버스 {pred['late_ret'].mean():+.2%}) · "
                     f"close_vs_vwap 평균 {sel['close_vs_vwap'].mean():+.2%}(유니버스 {pred['close_vs_vwap'].mean():+.2%})\n")
    # 예측 분위별 (폴드 안 5분위) — 예측이 단조한가
    pred["pq"] = pred.groupby("month")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    g = pred.groupby("pq")["ret_open"].agg(["size", "mean", "median", lambda v: (v > 0).mean()])
    L.append("예측 확률 5분위(폴드 안, Q5 = 높음): " + " · ".join(
        f"Q{int(k)} n{int(v['size']):,} 평균 {v['mean']:+.2%} 중앙 {v['median']:+.2%} 승률 {v.iloc[3]:.0%}" for k, v in g.iterrows()) + "\n")
    return L, res


# ────────────────────────────── main ──────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="캐시 디렉터리 (유니버스·피처 pickle)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cache = Path(args.cache) if args.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    conn = te.connect()
    upkl = cache / "univ.pkl" if cache else None
    if upkl and upkl.exists():
        d = pd.read_pickle(upkl)
    else:
        raw, adjc, master = te.load_daily(conn)
        md = te.minute_days(conn)
        d = te.build_universe(raw, adjc, master, md)
        if upkl:
            d.to_pickle(upkl)
    adj = load_adj_open(conn)
    conn.close()

    keep_cols = ["code", "date", "month", "name", "market", "close", "close_adj", "trade_value", "ret_day",
                 "limit_up", "limit_dn", "tv_ref", "is_new", "big2", "n_bars_1519"]
    full = d[d["daily_ok"] & d["has_min"]][keep_cols].copy()     # 상·하한가 포함 (정의 확인용)
    univ = d[d["in_univ"]][keep_cols].copy()

    fpkl = cache / "features.pkl" if cache else None
    if fpkl and fpkl.exists():
        feat = pd.read_pickle(fpkl)
    else:
        print(f"피처 계산: {len(full):,} (종목,일자) · 종목 {full['code'].nunique():,}", file=sys.stderr)
        feat = build_features(full, str(cache) if cache else None, args.workers)
        if fpkl:
            feat.to_pickle(fpkl)

    full = full.merge(feat, on=["code", "date"], how="left")
    n_nofeat = int(full["p1519"].isna().sum())
    full = full[full["p1519"].notna()]
    full_ev, _ = attach_exits(full, adj)
    ev = full_ev[~full_ev["limit_up"] & ~full_ev["limit_dn"]].copy()
    n_univ_feat = int((~full["limit_up"] & ~full["limit_dn"]).sum())
    mismatch = int(((full_ev["p1530"] / full_ev["close"] - 1).abs() > CLOSE_TOL).sum())
    n_p1530 = int(full_ev["p1530"].notna().sum())
    ev_m = ev[ev["p1530"].notna()]
    slip = ev_m["p1530"] / ev_m["p1519"] - 1

    O: list[str] = []
    O.append("# 장중 궤적 피처 8종 → 익일 시가 갭 — **1회 측정** (사전등록 16 그대로)\n")
    O.append(f"유니버스: 거래대금 ≥ {te.TV_FLOOR_MN/100:.0f}억 · 보통주 · 스팩/리츠 제외 · 15:19 분봉 있음 · 상·하한가 마감 제외. "
             f"기간 {te.START} ~ {te.END}. 거래 D 종가 → D+1 시가(수정 일봉). 비용 34bp(0/21/55 병기). "
             f"일봉 최종일 {adj['date'].max().date()}.\n")
    O.append(f"유니버스 {len(univ):,} · 피처 없음(분봉 결측) {n_nofeat:,} → {n_univ_feat:,} · D+1 미도래(달력 끝)·D+1 봉 없음 "
             f"{n_univ_feat - len(ev):,} 제외 → **측정 n = {len(ev):,}** · 거래일 {ev['date'].nunique()} · 종목 {ev['code'].nunique():,}. "
             f"참고: 15:30 분봉 종가 vs 수정 일봉 종가 0.2% 초과 불일치 {mismatch:,}/{n_p1530:,} · "
             f"15:19→15:30 괴리 |평균| {slip.abs().mean():.3%} 편향 {slip.mean():+.4%}\n")

    # ---- 1단계 ----
    O.append("## 1단계 — 전부 매수 기준선 (유니버스 평균)\n")
    O.append(line("전체 D 종가 → D+1 시가", ev["ret_open"], ev["date"]))
    O.append("  비용 4단계: " + ", ".join(f"{c*1e4:.0f}bp {ev['ret_open'].mean() - c:+.3%}" for c in COST_GRID))
    O.append("")
    T, neg1, tot1 = fold_table(ev, "월")
    O += T
    O.append(f"\n**월 음수(net34) {neg1}/{tot1}**\n")
    O.append("### 분리 보고 (D+1 시가)\n")
    O += split_block(ev, full_ev)
    O.append("")
    bins = [-1, -.1, -.05, 0, .05, .10, .20, .25, .295, 10]
    dist = ev.groupby(pd.cut(ev["ret_day"], bins), observed=False)["ret_open"].agg(["size", "mean"])
    O.append("당일 등락률 구간별 (n / D+1 시가 평균): " + " · ".join(
        f"{k} {int(v['size']):,}/{v['mean']:+.2%}" if v["size"] else f"{k} 0" for k, v in dist.iterrows()) + "\n")
    O.append("### 1단계 채택 5조건 (34bp) — 기계적 판정\n")
    C1, pass1 = five_conditions(ev, "13폴드")
    O += C1
    O.append("")
    O.append(f"**1단계 판정: {'통과 — 유니버스 전체 오버나이트가 비용을 이긴다 (전진 기록 대상)' if pass1 else '미달 (사전 기대대로: 기준선은 비용 후 남지 않는다)'}**\n")

    # ---- 단변량 ----
    O.append("## 단변량 — 스피어만 ρ (vs D+1 시가 수익률) · 5분위 평균/승률 (Q1 = 낮음) · 유니버스 전체\n")
    U, n_match = univariate(ev)
    O += U
    O.append(f"\n사전 가설 8개 중 방향·크기(|ρ|≥{RHO_SMALL}) 일치 **{n_match}/8**\n")

    # ---- 2단계 ----
    O.append("## 2단계 — 피처 8개로 고르기\n")
    S, res = stage2(ev)
    O += S

    # ---- 판정 ----
    O.append("## 판정 (기계적)\n")
    a, b = res["top_third"], res["top_day"]
    O.append(f"- AUC 게이트 (평균 ≥ {AUC_MIN:.2f}): {res['mean_auc']:.3f} → {'통과' if res['gate'] else '**미달**'}")
    O.append(f"- (a) 상위 1/3 5조건: {'전부 통과' if a['passed'] else '**미달**'} (net34 {a['net34']:+.3%}, n {a['n']:,})")
    O.append(f"- (b) 하루 상위 {TOP_N_DAY} net34 > 0: {'통과' if b['net34'] > 0 else '**미달**'} ({b['net34']:+.3%}, n {b['n']:,}) · 5조건 {'전부 통과' if b['passed'] else '미달'}")
    O.append(f"- 1단계 기준선 5조건: {'전부 통과' if pass1 else '미달'}")
    if res["gate"] and a["passed"] and b["net34"] > 0:
        verdict = "**채택 후보 — 전진 기록 착수** (15:19 피처 → 하루 상위 3 기록)"
    elif res["gate"] and a["passed"]:
        verdict = "**예측력은 있으나 실행 규모(하루 3건)에선 안 남음 — 채택 안 함**"
    elif res["gate"]:
        verdict = "**기각** — AUC 는 넘었으나 선별이 비용 후 남지 않는다"
    else:
        verdict = "**기각** — 분봉 궤적 8종으로 익일 갭을 고를 수 없다 (AUC 게이트 미달). 피처·모델·N 을 바꿔 재시도하지 않는다"
    O.append(f"\n{verdict}\n")

    text = "\n".join(O)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\n→ {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
