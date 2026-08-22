"""낙주 매매 검증 — 사전등록 10 (`docs/research/manju/10-dip-preregistration.md`) 그대로.

  이벤트   당일 누적고가 대비 -10% 를 처음 찍는 분 (종목·일자당 1건, 09:00~14:59)
           그 분까지 누적 거래대금 >= 500억 (§11 확정. 50억은 참고 건수만)
  진입     트리거 분 종가 전량          청산   +2% / -2% / 15:30 종가 (동시 → 손절 먼저)
  1단계    전부 진입 — 채택 5조건 (net34>0 · 음수월<=3/13 · net55>0 · 하한가 제외 net34>0 · 승률>50%)
  2단계    피처 8개 → GBM(depth2, 100) · 월별 확장 폴드 10개 · 폴드 AUC 중앙값 < 0.60 이면 접는다
           통과 시 폴드 안 예측 상위 1/3 에 1단계 5조건

문서 밖 파라미터·변형은 넣지 않는다. 모호한 곳은 보수적으로 해석하고 이 파일 머리와
보고서에 적는다:
  - 승률은 34bp 차감 후 > 0 으로 센다 (총수익 기준도 병기)
  - 하한가 = 전일종가 x 0.7 을 호가단위로 올림 (limitup_timing.tick_size). 당일 저가(15:30 포함) 가
    그 값 이하이면 "하한가 도달". 최악치 = 그 건 중 손절·종가 청산을 하한가 체결로 교체
  - 전일종가 = 분봉 전일(거래일 달력 기준 직전일) 15:30 봉 종가. 15:30 봉이 없으면 그날 마지막
    봉 종가. 전일 분봉 자체가 없으면 이벤트에서 제외하고 건수만 센다
  - 15:30 봉이 없는 날의 종가 청산은 마지막 정규장 봉 종가 (건수 보고)
  - `sell_climax` 는 트리거가 첫 봉이면, `lower_wick` 은 고가=저가이면 정의 불가(NaN) → 2단계에서만 제외
  - `n_shock_before` 는 트리거 분 **이전** 봉들만 센다 (트리거 분 자체 제외)
  - 2단계 폴드의 음수 월 조건은 검정 폴드 10개월에 같은 임계(<=3) 를 적용

DB 는 읽기만 한다. 실행:
  uv run --extra pg python scripts/verify_dip.py --cache /path/dip_events.pkl
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from scalp_it.adapters.db import load_ohlcv, load_stock_master
from scalp_it.intraday.loader import available_codes, load_minute_bars
from scalp_it.intraday.store import connect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dip_events import GAP_WINDOW, static_vi_sim
from limitup_timing import tick_size

FEATURES = ["trig_minute", "high_vs_prev", "minutes_from_high", "cum_tv_ratio",
            "sell_climax", "lower_wick", "vi_sim_near", "n_shock_before"]
DD_TRIGGER = 0.10
TV_FLOOR = 5e10          # 500억 원 — minute_bars.trade_value 는 원 (§11 확정)
TV_FLOOR_REF = 5e9       # 50억 — 참고 건수만
TV_FLOOR_DAILY = 50_000.0   # 500억 — daily_bars_adjusted.trade_value 는 백만원 (커버리지 추정용)
TAKE, STOP = 0.02, 0.02
COST, COST_HI = 0.0034, 0.0055
FIRST_MINUTE = 9 * 60
LAST_TRIGGER_MINUTE = 14 * 60 + 59
REGULAR_END_MINUTE = 15 * 60 + 19
CLOSE_MINUTE = 15 * 60 + 30
NEG_MONTHS_MAX = 3
AUC_MIN = 0.60
TRAIN_ONLY_MONTHS = ("2025-08", "2025-09", "2025-10")
TOP_FRACTION = 1 / 3
MINUTE_TV_DIV = 1e6      # 원 → 백만원
MFH_BINS = [-1, 30, 60, 120, 10_000]   # §9 의 구간 (<=30, <=60, <=120, >120)
MFH_LABELS = ["<=30분", "31~60분", "61~120분", ">120분"]


def _load_env(path: str | None) -> None:
    if not path or os.environ.get("KR_QUANT_DB"):
        return
    for line in pathlib.Path(path).read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def limit_down_price(prev_close: float) -> float:
    """전일종가 x 0.7 을 그 가격대의 호가단위로 **올림** (상한가 내림의 거울상)."""
    raw = prev_close * 0.7
    tick = tick_size(raw)
    return math.ceil(raw / tick - 1e-9) * tick


def _prev_close_map(bars: pd.DataFrame) -> tuple[dict, dict, dict]:
    """(code, date) → 전일 15:30 종가. 15:30 봉이 없으면 그날 마지막 봉 종가 (src='last')."""
    b = bars[bars["minute"] <= CLOSE_MINUTE]
    last = b.groupby(["code", "date"], sort=False)["close"].last()
    c1530 = b[b["minute"] == CLOSE_MINUTE].set_index(["code", "date"])["close"]
    close_of_day = {k: (float(v), "1530") for k, v in c1530.items()}
    for k, v in last.items():
        close_of_day.setdefault(k, (float(v), "last"))
    return close_of_day, last.to_dict(), c1530.to_dict()


def simulate_exit(o, h, l, c, i: int, entry: float, close_px: float) -> tuple[float, str]:
    """§4 청산. 트리거 분 다음 봉부터. 같은 분에 둘 다 닿으면 손절 먼저. 갭 관통은 시가 체결."""
    tp, sl = entry * (1 + TAKE), entry * (1 - STOP)
    for j in range(i + 1, len(c)):
        if o[j] <= sl:
            return float(o[j]), "stop_gap"
        if o[j] >= tp:
            return float(o[j]), "profit_gap"
        if l[j] <= sl:
            return sl, "stop"
        if h[j] >= tp:
            return tp, "profit"
    return close_px, "close"


def build_events(bars: pd.DataFrame, daily: pd.DataFrame, cal_prev: dict) -> tuple[pd.DataFrame, dict]:
    """한 배치(종목 묶음)의 분봉에서 이벤트·피처·수익을 만든다. 반환: (이벤트, 집계 카운터)."""
    cnt = {"days": 0, "no_touch": 0, "after_1459": 0, "below_500": 0, "ref_50_ok": 0,
           "no_prev_close": 0, "prev_close_last": 0, "close_src_last": 0}
    if bars.empty:
        return pd.DataFrame(), cnt
    b = bars.copy()
    b["date"] = b["ts"].dt.normalize()
    b["minute"] = b["ts"].dt.hour * 60 + b["ts"].dt.minute
    close_of_day, last_close, c1530 = _prev_close_map(b)
    day_low = b[b["minute"] <= CLOSE_MINUTE].groupby(["code", "date"], sort=False)["low"].min().to_dict()

    # adv20 — 직전 20 거래일 일평균 거래대금 (백만원), 5차와 같은 정의
    d = daily.sort_values(["code", "date"]).copy()
    d["adv20"] = d.groupby("code", sort=False)["trade_value"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    d.loc[d["adv20"] <= 0, "adv20"] = np.nan
    adv20 = d.set_index(["code", "date"])["adv20"].to_dict()

    reg = b[(b["minute"] >= FIRST_MINUTE) & (b["minute"] <= REGULAR_END_MINUTE)]
    reg = reg.sort_values(["code", "ts"])
    rows = []
    for (code, date), g in reg.groupby(["code", "date"], sort=False):
        cnt["days"] += 1
        o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        tv = g["trade_value"].to_numpy(float)
        minute = g["minute"].to_numpy(int)
        run_high = np.maximum.accumulate(h)
        hit = np.flatnonzero(l <= run_high * (1.0 - DD_TRIGGER))
        if len(hit) == 0:
            cnt["no_touch"] += 1
            continue
        i = int(hit[0])
        trig_minute = int(minute[i])
        if trig_minute > LAST_TRIGGER_MINUTE:
            cnt["after_1459"] += 1
            continue
        cum_tv = float(tv[: i + 1].sum())
        if cum_tv >= TV_FLOOR_REF:
            cnt["ref_50_ok"] += 1
        if cum_tv < TV_FLOOR:
            cnt["below_500"] += 1
            continue
        prev_day = cal_prev.get(date)
        pc = close_of_day.get((code, prev_day)) if prev_day is not None else None
        if pc is None:
            cnt["no_prev_close"] += 1
            continue
        prev_close, pc_src = pc
        if pc_src == "last":
            cnt["prev_close_last"] += 1

        entry = float(c[i])
        if (code, date) in c1530:
            close_px, close_src = float(c1530[(code, date)]), "1530"
        else:
            close_px, close_src = float(last_close[(code, date)]), "last"
            cnt["close_src_last"] += 1
        exit_px, reason = simulate_exit(o, h, l, c, i, entry, close_px)
        ret = exit_px / entry - 1.0

        ld = limit_down_price(prev_close)
        hit_ld = bool(day_low[(code, date)] <= ld + 1e-9)
        ret_worst = ret
        if hit_ld and reason in ("stop", "stop_gap", "close"):
            ret_worst = min(ret, ld / entry - 1.0)

        # --- 피처 8개: 트리거 시점까지의 정보만 ---
        high_idx = int(np.argmax(h[: i + 1]))
        a = adv20.get((code, date), np.nan)
        sell_climax = (tv[i] / tv[:i].mean()) if i > 0 and tv[:i].mean() > 0 else np.nan
        lower_wick = (c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else np.nan
        vi_minute, _ = static_vi_sim(g.reset_index(drop=True))
        r1 = np.abs(np.diff(c[: i + 1]) / c[:i]) if i > 0 else np.array([])   # 1..i 의 1분 수익
        n_shock_before = int((r1[:-1] > 0.03).sum()) if i > 1 else 0          # 트리거 분 제외

        rows.append({
            "code": code, "date": date, "trig_hhmm": f"{trig_minute // 60:02d}:{trig_minute % 60:02d}",
            "entry": entry, "exit": exit_px, "reason": reason, "ret": ret, "ret_worst": ret_worst,
            "label": int(reason in ("profit", "profit_gap")),
            "prev_close": prev_close, "limit_down": ld, "hit_limit_down": int(hit_ld),
            "day_low": float(day_low[(code, date)]), "close_src": close_src,
            "cum_tv_bn": cum_tv / 1e8, "run_high": float(run_high[i]),
            # 피처
            "trig_minute": trig_minute - FIRST_MINUTE,
            "high_vs_prev": float(run_high[i]) / prev_close - 1.0,
            "minutes_from_high": trig_minute - int(minute[high_idx]),
            "cum_tv_ratio": (cum_tv / MINUTE_TV_DIV / a) if a and not np.isnan(a) else np.nan,
            "sell_climax": sell_climax,
            "lower_wick": lower_wick,
            "vi_sim_near": int(vi_minute is not None and abs(vi_minute - trig_minute) <= GAP_WINDOW),
            "n_shock_before": n_shock_before,
        })
    return pd.DataFrame(rows), cnt


# ----------------------------------------------------------------------------- 보고

def _stats(r: pd.Series) -> dict:
    w, lo = r[r > 0], r[r <= 0]
    pl = abs(w.mean() / lo.mean()) if len(w) and len(lo) and lo.mean() != 0 else np.nan
    net = r - COST
    return {"n": len(r), "mean": r.mean(), "net34": r.mean() - COST, "net55": r.mean() - COST_HI,
            "median": r.median(), "win": (r > 0).mean(), "win34": (net > 0).mean(), "pl": pl,
            "p5": r.quantile(.05), "p25": r.quantile(.25), "p75": r.quantile(.75), "p95": r.quantile(.95)}


def _line(tag: str, r: pd.Series) -> str:
    if len(r) == 0:
        return f"  {tag}: n=0"
    s = _stats(r)
    return (f"  {tag}: n={s['n']:>5} 평균 {s['mean']:+.2%} net34 {s['net34']:+.2%} net55 {s['net55']:+.2%} "
            f"중앙 {s['median']:+.2%} 승률 {s['win']:.1%}(net34 {s['win34']:.1%}) 손익비 {s['pl']:.2f} "
            f"p5 {s['p5']:+.1%} p25 {s['p25']:+.1%} p75 {s['p75']:+.1%} p95 {s['p95']:+.1%}")


def _monthly(ev: pd.DataFrame) -> pd.DataFrame:
    m = ev.groupby("month")["ret"].agg(n="size", mean="mean")
    m["net34"] = m["mean"] - COST
    m["neg"] = m["net34"] < 0
    return m


def _five_conditions(ev: pd.DataFrame, tag: str, n_months_label: str) -> dict:
    r = ev["ret"]
    m = _monthly(ev)
    excl = ev.loc[ev["hit_limit_down"] == 0, "ret"]
    cond = {
        "1 net34>0": r.mean() - COST > 0,
        f"2 음수월<={NEG_MONTHS_MAX}": int(m["neg"].sum()) <= NEG_MONTHS_MAX,
        "3 net55>0": r.mean() - COST_HI > 0,
        "4 하한가제외 net34>0": (excl.mean() - COST > 0) if len(excl) else False,
        "5 승률(net34)>50%": ((r - COST) > 0).mean() > 0.5,
    }
    print(f"  {tag} 채택 5조건: " + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k, v in cond.items())
          + f"  (음수 월 {int(m['neg'].sum())}/{len(m)} {n_months_label}, 총수익 승률 {(r > 0).mean():.1%})")
    print(f"  → {'전부 만족' if all(cond.values()) else '미달'} (예비 표본 — 확정 아님)")
    return cond


def _analyze(ev: pd.DataFrame, meta: dict) -> int:
    print("\n=== 낙주 이벤트 (정의 A · 누적 500억 · 09:00~14:59) ===")
    print("  " + " · ".join(f"{k} {v:,}" for k, v in meta.get("count", {}).items()))
    print(f"  표본: 분봉 종목 {meta.get('n_codes', 0)} · 기간 {meta.get('start')}~{meta.get('end')} · "
          f"거래일(달력) {meta.get('n_cal_days', 0)}")
    cov = meta.get("coverage")
    if cov:
        print(f"  커버리지 추정: 일봉 이벤트(저가/고가 -10%, 당일 500억) {cov['daily']:,} 중 분봉 있는 (종목,일자) "
              f"{cov['with_minute']:,} = {cov['with_minute'] / max(1, cov['daily']):.1%}")
    print(f"  참고: 누적 50억 기준 이벤트 건수 {meta.get('count', {}).get('ref_50_ok', 0):,} (건수만, 수익 미계산)")
    if len(ev) < 100:
        print("표본 부족. 종료.")
        return 1
    ev = ev.sort_values(["date", "trig_minute", "code"]).reset_index(drop=True)
    ev["month"] = ev["date"].dt.strftime("%Y-%m")
    n_days = ev["date"].nunique()
    print(f"\n이벤트 {len(ev):,} · 종목 {ev['code'].nunique():,} · 이벤트 있는 날 {n_days} · "
          f"거래일당 {len(ev) / max(1, meta.get('n_cal_days', n_days)):.1f}건 (달력 기준) · "
          f"{len(ev) / n_days:.1f}건 (이벤트 있는 날 기준)")
    print(f"  15:30 봉 없어 마지막 봉 종가로 청산 {int((ev['close_src'] == 'last').sum())}건")

    print("\n=== 1단계: 전부 진입 (진입=트리거 분 종가, +2%/-2%/15:30 종가) ===")
    print(_line("전체", ev["ret"]))
    m = _monthly(ev)
    print("\n| 월 | n | 평균 | net34 | 음수 |\n|---|---|---|---|---|")
    for mo, r in m.iterrows():
        print(f"| {mo} | {int(r.n):,} | {r['mean']:+.2%} | {r.net34:+.2%} | {'X' if r.neg else ''} |")
    print("\n  종료 사유:")
    for k, g in ev.groupby("reason"):
        print(_line(f"  {k:<10}", g["ret"]))
    ld = ev[ev["hit_limit_down"] == 1]
    print(f"\n  하한가 도달 {len(ld):,}건 ({len(ld) / len(ev):.1%}) — 전일종가x0.7 호가단위 올림 이하 저가")
    print(_line("하한가 도달 건", ld["ret"]))
    print(_line("하한가 제외", ev.loc[ev["hit_limit_down"] == 0, "ret"]))
    print(_line("최악치(하한가 건 손절=하한가 체결)", ev["ret_worst"]))
    print("\n  시장별:")
    for k, g in ev.groupby(ev["market"].fillna("미상")):
        print(_line(f"  {k}", g["ret"]))
    print("\n  트리거 시각대:")
    for k, g in ev.groupby((ev["trig_minute"] + FIRST_MINUTE) // 60):
        print(_line(f"  {int(k):02d}시", g["ret"]))
    print("\n  고점→트리거 경과:")
    ev["mfh_bin"] = pd.cut(ev["minutes_from_high"], MFH_BINS, labels=MFH_LABELS)
    for k, g in ev.groupby("mfh_bin", observed=True):
        print(_line(f"  {k}", g["ret"]))
    top10 = ev["code"].value_counts().head(10)
    print(f"\n  종목 집중도: 상위 10종목 {int(top10.sum()):,}건 = {top10.sum() / len(ev):.1%} "
          f"({', '.join(f'{c}:{n}' for c, n in top10.items())})")
    r = ev["ret"]
    print(_line("상위 1% 제거", r[r <= r.quantile(0.99)]))
    first2 = ev.groupby("date", sort=False).head(2)
    print(_line("하루 선착 2건", first2["ret"]))
    print()
    _five_conditions(ev, "1단계(전부 진입)", "월")
    print("\n  피처 요약:\n" + ev[FEATURES].describe().T[["count", "mean", "50%", "min", "max"]].to_string())

    # ---------------- 2단계
    print("\n=== 2단계: 선별 가능성 — 라벨 '+2% 먼저', GBM(depth2,100), 월별 확장 폴드 ===")
    x = ev.dropna(subset=FEATURES).copy()
    print(f"  피처 결측 제외 {len(ev) - len(x):,}건 → {len(x):,}건 · 라벨 양성률 {x['label'].mean():.1%}")
    months = sorted(x["month"].unique())
    test_months = [mo for mo in months if mo not in TRAIN_ONLY_MONTHS]
    aucs, imps, parts = {}, [], []
    for mo in test_months:
        tr, te = x[x["month"] < mo], x[x["month"] == mo]
        if len(tr) < 50 or te["label"].nunique() < 2:
            print(f"  폴드 {mo}: 건너뜀 (train {len(tr)}, test {len(te)}, 라벨 단일={te['label'].nunique() < 2})")
            continue
        mdl = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=0)
        mdl.fit(tr[FEATURES], tr["label"])
        p = mdl.predict_proba(te[FEATURES])[:, 1]
        a = roc_auc_score(te["label"], p)
        aucs[mo] = a
        imps.append(pd.Series(mdl.feature_importances_, index=FEATURES))
        te = te.assign(pred=p)
        k = max(1, math.floor(len(te) * TOP_FRACTION))
        te["top"] = te["pred"].rank(method="first", ascending=False) <= k
        parts.append(te)
        print(f"  폴드 {mo}: train {len(tr):>5} test {len(te):>4}  AUC {a:.3f}  (양성률 {te['label'].mean():.1%})")
    if not aucs:
        print("  폴드 없음. 종료.")
        return 0
    med = float(np.median(list(aucs.values())))
    print(f"  폴드 AUC 중앙값 {med:.3f} (평균 {np.mean(list(aucs.values())):.3f}, 폴드 {len(aucs)}개, 기준 {AUC_MIN})")
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    print("  중요도(폴드 평균): " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    oos = pd.concat(parts, ignore_index=True)
    passed = med >= AUC_MIN
    if passed:
        print("\n  >>> AUC 통과 — 폴드 안 예측 상위 1/3 에 1단계 5조건 적용 (통과해도 전진 기록 착수이지 실전 아님)")
    else:
        print("\n  >>> 폴드 AUC 중앙값 < 0.60 → 사전등록대로 낙주를 접는다. "
              "(아래 상위 1/3 수치는 참고용 — 판정에 쓰지 않는다)")
    top = oos[oos["top"]]
    print(_line("상위 1/3", top["ret"]))
    print(_line("하위 2/3", oos.loc[~oos["top"], "ret"]))
    _five_conditions(top, "2단계(상위 1/3)", "검정월")
    oos["q"] = pd.qcut(oos["pred"].rank(method="first"), 3, labels=False) + 1
    print("  전역 3분위(1=낮음): 양성률 / 평균 / net34 / n")
    for q, gq in oos.groupby("q"):
        print(f"    Q{q}: {gq['label'].mean():.1%} / {gq['ret'].mean():+.2%} / {gq['ret'].mean() - COST:+.2%} / {len(gq)}")
    cols = ["code", "date", "trig_hhmm", "entry", "reason", "ret", "pred", "high_vs_prev",
            "minutes_from_high", "cum_tv_bn", "hit_limit_down"]
    t20 = oos.sort_values("pred", ascending=False).head(20)[cols]
    t20 = t20.assign(date=t20["date"].dt.date, ret=t20["ret"].map(lambda v: f"{v:+.2%}"),
                     pred=t20["pred"].round(3), high_vs_prev=t20["high_vs_prev"].round(3),
                     cum_tv_bn=t20["cum_tv_bn"].round(0))
    print("\n  OOS 예측 상위 20건 (눈으로 볼 것):\n" + t20.to_string(index=False))
    return 0


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--env-file", default="/home/young/Documents/git/kr-quant/.env")
    ap.add_argument("--cache", default=None, help="이벤트 pickle. 있으면 읽고 없으면 만든 뒤 저장")
    ap.add_argument("--batch", type=int, default=40, help="한 번에 읽을 종목 수")
    args = ap.parse_args()
    _load_env(args.env_file)

    if args.cache and os.path.exists(args.cache):
        blob = pd.read_pickle(args.cache)
        print(f"캐시 로드: {args.cache} ({len(blob['ev']):,}건)", file=sys.stderr)
        return _analyze(blob["ev"], blob["meta"])

    conn = connect()
    codes = available_codes(conn)
    print(f"분봉 보유 {len(codes)} 종목", file=sys.stderr)
    d_start = (pd.Timestamp(args.start) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    daily_all = load_ohlcv(conn, d_start, args.end)
    cal = list(pd.DatetimeIndex(daily_all["date"].unique()).sort_values())
    cal_prev = {d: p for p, d in itertools.pairwise(cal)}
    n_cal_days = sum(1 for d in cal if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end))
    master = load_stock_master(conn)
    market = master.set_index("code")["market"].to_dict()

    # 커버리지 추정용 일봉 이벤트 (전 종목): 저가/고가 -10%, 당일 500억
    de = daily_all[(daily_all["date"] >= pd.Timestamp(args.start)) & (daily_all["high"] > 0)]
    de = de[(de["low"] / de["high"] - 1 <= -DD_TRIGGER) & (de["trade_value"] >= TV_FLOOR_DAILY)]
    de_keys = set(zip(de["code"], de["date"]))

    parts, total = [], {}
    minute_days: set = set()
    n_batches = (len(codes) + args.batch - 1) // args.batch
    for bi in range(0, len(codes), args.batch):
        chunk = codes[bi:bi + args.batch]
        bars = load_minute_bars(conn, args.start, args.end, codes=chunk)
        if not bars.empty:
            minute_days |= set(zip(bars["code"], bars["ts"].dt.normalize()))
        sub, cnt = build_events(bars, daily_all[daily_all["code"].isin(chunk)], cal_prev)
        for k, v in cnt.items():
            total[k] = total.get(k, 0) + v
        if len(sub):
            parts.append(sub)
        print(f"  배치 {bi // args.batch + 1}/{n_batches}: {len(bars):,}행 → 이벤트 {len(sub):,} "
              f"(누적 {sum(len(p) for p in parts):,})", file=sys.stderr)
        del bars
    conn.close()
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(ev):
        ev["market"] = ev["code"].map(market)
    meta = {"count": total, "n_codes": len(codes), "start": args.start, "end": args.end,
            "n_cal_days": n_cal_days,
            "coverage": {"daily": len(de_keys), "with_minute": len(de_keys & minute_days)}}
    if args.cache:
        pd.to_pickle({"ev": ev, "meta": meta}, args.cache)
    return _analyze(ev, meta)


if __name__ == "__main__":
    raise SystemExit(main())
