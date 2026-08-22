"""상따 분봉 재검증 — 5차 사전등록(06번 문서) 그대로 실행.

  이벤트  분봉 고가가 전일종가 x 1.25 를 처음 넘는 그 분
  진입    그 분의 종가        청산  익일 시가
  1단계   상한가 마감 예측 AUC. 0.60 미만이면 2단계 없이 종료
  2단계   예측 상위분 매수 시 익일 시가 수익

피처 8개는 사전등록에 고정돼 있다. 추가·변경 금지.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from scalp_it.adapters.db import connect, load_ohlcv
from scalp_it.intraday.loader import load_minute_bars, available_codes

FEATURES = ["touch_minute","gap_open","cum_tv","cum_tv_ratio",
            "speed","max_dd_before","pos_in_range","n_vi_like"]
TRIGGER = 1.25
TV_FLOOR = 50000.0      # 500억 (백만원 단위)
COST = 0.0034
COST_HI = 0.0055        # 보수 시나리오 병기
LIMIT_AT_ENTRY = 1.295  # 진입 분 종가가 이미 상한가(≈+30%)면 체결 불가 의심
MINUTE_TV_DIV = 1e6     # minute_bars.trade_value 는 원, daily 는 백만원 → 단위 통일

def build_events(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["code","date"]).copy()
    d["prev_close"] = d.groupby("code", sort=False)["close"].shift(1)
    d["adv20"] = d.groupby("code", sort=False)["trade_value"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    d.loc[d["adv20"] <= 0, "adv20"] = np.nan      # 20일 무거래 종목 → 비율 정의 불가(inf 방지)
    d["ret_day"] = (d["close"] - d["prev_close"]) / d["prev_close"]
    d["n_open"] = d.groupby("code", sort=False)["open"].shift(-1)
    key = d.set_index(["code","date"])[["prev_close","adv20","ret_day","n_open","trade_value"]]

    b = bars.copy()
    b["date"] = b["ts"].dt.normalize()
    b = b.join(key, on=["code","date"])
    b = b.dropna(subset=["prev_close","adv20","n_open"])
    b = b[b["trade_value"] >= TV_FLOOR] if "trade_value" in key.columns else b
    b = b[b["ts"].dt.time < pd.Timestamp("15:20").time()]   # 정규장만
    b = b.sort_values(["code","date","ts"])

    g = b.groupby(["code","date"], sort=False)
    b["minute"] = (b["ts"] - b["ts"].dt.normalize() - pd.Timedelta("9h")).dt.total_seconds()/60
    b["cum_tv"] = g["trade_value_bar"].cumsum() / MINUTE_TV_DIV   # 백만원
    b["run_high"] = g["high"].cummax()
    b["run_low"] = g["low"].cummin()
    b["ret1"] = g["close"].pct_change()
    b["shock"] = (b["ret1"].abs() > 0.03).astype(int)
    b["n_shock_cum"] = g["shock"].cumsum()
    b["dd"] = b["close"]/b["run_high"] - 1
    b["max_dd_cum"] = g["dd"].cummin()
    b["open_day"] = g["open"].transform("first")

    b["hit"] = b["high"] >= b["prev_close"] * TRIGGER
    ev = b[b["hit"]].groupby(["code","date"], sort=False).head(1).copy()
    if ev.empty:
        return ev
    ev["entry"] = ev["close"]                       # 사전등록: 그 분의 종가에 체결
    ev["entry_trig"] = ev["prev_close"] * TRIGGER   # 참고: 트리거가 체결 가정(낙관)
    ev["touch_minute"] = ev["minute"]
    ev["gap_open"] = ev["open_day"]/ev["prev_close"] - 1
    ev["cum_tv_ratio"] = ev["cum_tv"]/ev["adv20"]
    ev["speed"] = (TRIGGER - 1 - ev["gap_open"]) / ev["touch_minute"].clip(lower=1)
    ev["max_dd_before"] = -ev["max_dd_cum"]
    rng = (ev["run_high"] - ev["run_low"]).replace(0, np.nan)
    ev["pos_in_range"] = ((ev["close"] - ev["run_low"])/rng).fillna(0.5)
    ev["n_vi_like"] = ev["n_shock_cum"]
    ev["is_limit"] = (ev["ret_day"] >= 0.295).astype(int)
    ev["ret"] = (ev["n_open"] - ev["entry"]) / ev["entry"]
    ev["ret_trig"] = (ev["n_open"] - ev["entry_trig"]) / ev["entry_trig"]
    ev["limit_at_entry"] = (ev["entry"] >= ev["prev_close"] * LIMIT_AT_ENTRY).astype(int)
    return ev.dropna(subset=FEATURES + ["ret","is_limit"])

def _full_codes(c, start: str, end: str) -> list[str]:
    """수집이 [start, end] 를 덮는 종목만 (표본 정의만 제한, 피처·임계 불변)."""
    cur = c.cursor()
    cur.execute("select code, min(ts)::date, max(ts)::date from minute_bars group by code")
    rows = cur.fetchall(); cur.close()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return sorted(str(r[0]).strip() for r in rows
                  if pd.Timestamp(r[1]) <= lo and pd.Timestamp(r[2]) >= hi)


def _describe(tag: str, r: pd.Series) -> None:
    w, l = r[r > 0], r[r <= 0]
    pl = abs(w.mean() / l.mean()) if len(w) and len(l) and l.mean() != 0 else np.nan
    print(f"  {tag}: n={len(r):>4} 평균 {r.mean():+.2%} net34 {r.mean()-COST:+.2%} "
          f"net55 {r.mean()-COST_HI:+.2%} 중앙 {r.median():+.2%} 승률 {(r>0).mean():.1%} "
          f"손익비 {pl:.2f}  p5 {r.quantile(.05):+.1%} p95 {r.quantile(.95):+.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-only", action="store_true",
                    help="분봉이 --full-start~--full-end 를 덮는 종목만 (예비 실행용 표본 제한)")
    ap.add_argument("--full-start", default="2025-08-01")
    ap.add_argument("--full-end", default="2026-08-13")
    ap.add_argument("--cache", default=None,
                    help="이벤트 테이블 pickle 경로. 있으면 읽고, 없으면 만든 뒤 저장 (재실행 시간 절약)")
    args = ap.parse_args()
    if args.cache and os.path.exists(args.cache):
        ev = pd.read_pickle(args.cache)
        print(f"캐시에서 이벤트 {len(ev):,}건 로드: {args.cache}", file=sys.stderr)
        return _analyze(ev)

    c = connect()
    codes = available_codes(c)
    print(f"분봉 보유 {len(codes)} 종목", file=sys.stderr)
    if args.full_only:
        codes = _full_codes(c, args.full_start, args.full_end)
        print(f"완비({args.full_start}~{args.full_end}) 종목만: {len(codes)}", file=sys.stderr)
    daily = load_ohlcv(c, "2025-06-01", "2026-08-21")
    daily = daily[daily["code"].isin(codes)]
    print(f"일봉 {len(daily):,} 행", file=sys.stderr)

    # 분봉 전체는 약 6천만 행이라 한 번에 못 올린다. 종목 배치로 나눈다.
    BATCH = 40
    parts = []
    for i in range(0, len(codes), BATCH):
        chunk = codes[i:i+BATCH]
        bars = load_minute_bars(c, "2025-08-01", "2026-08-21", codes=chunk)
        bars = bars.rename(columns={"trade_value":"trade_value_bar"})
        sub = build_events(bars, daily[daily["code"].isin(chunk)])
        if len(sub):
            parts.append(sub)
        print(f"  배치 {i//BATCH+1}/{(len(codes)+BATCH-1)//BATCH}: "
              f"{len(bars):,}행 → 이벤트 {len(sub):,}", file=sys.stderr)
        del bars
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if args.cache:
        ev.to_pickle(args.cache)
    return _analyze(ev)


def _analyze(ev: pd.DataFrame) -> int:
    print(f"\n=== 상따 트리거 이벤트 {len(ev):,}건 ===")
    if len(ev) < 200:
        print("표본 부족. 종료."); return 1
    print(f"  상한가 마감 {ev['is_limit'].sum():,}건 ({ev['is_limit'].mean():.1%})")
    print(f"  종목 {ev['code'].nunique()}개, 거래일 {ev['date'].nunique()}일, "
          f"진입 시 이미 상한가 의심 {ev['limit_at_entry'].sum():,}건")
    print("  전부 매수 (진입=트리거 분 종가, 사전등록):")
    _describe("전체", ev["ret"])
    _describe("참고 진입=전일종가×1.25", ev["ret_trig"])
    _describe("상한가의심 제외", ev.loc[ev["limit_at_entry"]==0, "ret"])
    print(f"  피처 요약:\n{ev[FEATURES].describe().T[['mean','50%','min','max']].to_string()}")

    ev = ev.sort_values("date").reset_index(drop=True)
    folds = np.array_split(np.arange(len(ev)), 3)
    print("\n=== 1단계: 상한가 마감 예측 (시간순 3폴드) ===")
    aucs, preds = [], np.full(len(ev), np.nan)
    for i in (1,2):
        tr = np.concatenate(folds[:i]); te = folds[i]
        m = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=0)
        m.fit(ev.loc[tr, FEATURES], ev.loc[tr,"is_limit"])
        p = m.predict_proba(ev.loc[te, FEATURES])[:,1]
        preds[te] = p
        a = roc_auc_score(ev.loc[te,"is_limit"], p) if ev.loc[te,"is_limit"].nunique()>1 else np.nan
        aucs.append(a)
        base = ev.loc[te, "is_limit"].mean()
        print(f"  폴드 {i}: n_train={len(tr):>5} n_test={len(te):>5}  AUC {a:.3f}  "
              f"(테스트 상한가 비율 {base:.1%}, 기간 {ev.loc[te,'date'].min().date()}~{ev.loc[te,'date'].max().date()})")
        imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("     중요도: " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    auc = float(np.nanmean(aucs))
    print(f"  평균 AUC {auc:.3f}   (사전등록 기준: 0.60)")
    passed = auc >= 0.60
    if not passed:
        print("\n>>> AUC 0.60 미만. 사전등록대로 2단계는 조건부 종료. "
              "(아래 2단계 수치는 예비 표본 참고용이며 판정에 쓰지 않는다)")
    print("\n=== 2단계: 예측 상위 매수 (OOS 예측이 있는 폴드 1·2만) ===")
    ev["pred"] = preds
    ev["fold"] = 0
    for i in (1,2): ev.loc[folds[i], "fold"] = i
    x = ev.dropna(subset=["pred"])
    for n in (1,2):
        top = x.sort_values(["date","pred"], ascending=[True,False]).groupby("date").head(n)
        r = top["ret"]
        _describe(f"하루 {n}건 전체", r)
        _describe(f"하루 {n}건 상위1%제거", r[r <= r.quantile(0.99)])
        _describe(f"하루 {n}건 상한가의심제외", top.loc[top["limit_at_entry"]==0, "ret"])
        _describe(f"하루 {n}건 참고 진입=1.25", top["ret_trig"])
        for f in (1,2):
            rf = top.loc[top["fold"]==f, "ret"]
            if len(rf): _describe(f"하루 {n}건 폴드{f}", rf)
        # 채택 5조건 (종가베팅과 동일, 음수 해 → 음수 폴드 ≤1)
        neg_folds = sum(1 for f in (1,2) if (top.loc[top["fold"]==f,"ret"].mean()-COST) < 0)
        cond = {
            "1 net34>0": r.mean()-COST > 0,
            "2 음수폴드<=1": neg_folds <= 1,
            "3 상위1%제거 net34>0": r[r <= r.quantile(0.99)].mean()-COST > 0,
            "4 상한가의심제외 net34>0": top.loc[top["limit_at_entry"]==0,"ret"].mean()-COST > 0,
            "5 승률>50%": (r>0).mean() > 0.5,
        }
        print(f"  하루 {n}건 채택조건: " + "  ".join(f"[{k}: {'O' if v else 'X'}]" for k,v in cond.items())
              + f"  (음수 폴드 {neg_folds}/2)")
    # 예측 상위/하위 5분위별 실현 상한가율·수익 (참고)
    x = x.assign(q=pd.qcut(x["pred"].rank(method="first"), 5, labels=False)+1)
    print("  예측 5분위(1=낮음) 별: 상한가율 / 평균ret / net34 / n")
    for q, gq in x.groupby("q"):
        print(f"    Q{q}: {gq['is_limit'].mean():.1%} / {gq['ret'].mean():+.2%} / {gq['ret'].mean()-COST:+.2%} / {len(gq)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
