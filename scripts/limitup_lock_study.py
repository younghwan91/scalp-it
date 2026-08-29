"""굳힐 대장 판별 — 사전등록 47번 측정.

상한가에 닿은 종목이 **종가까지 굳히는가**를 터치 시점에 알 수 있는 정보만으로
예측한다. 표본·피처·모델·판정선은 `docs/research/manju/47-limitup-lock-preregistration.md`
에 측정 전 고정돼 있다 — 이 스크립트는 그걸 그대로 집행할 뿐 문턱을 탐색하지 않는다.

    uv run --extra pg python scripts/limitup_lock_study.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

TOUCH = 0.28          # 상한가 터치·굳힘 판정선(사전등록 §1, 호가단위 반올림 흡수)
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TOP_Q = 0.20          # 상위 분위(사전등록 §5-1)
HIT_LINE = 0.60       # 상위 분위 굳힘률 판정선
BOOT = 1000

FEATURES = [
    "gap",            # 당일 시가 갭 — 터치 시점에 유일하게 아는 당일 정보
    "lv_turnover",    # log 전일 거래대금
    "lv_turnover20",  # log 20일 평균 거래대금
    "lv_mktcap",      # log 시총 대리
    "vol20",          # 20일 변동성
    "ret5",           # 5일 수익률
    "ret20",          # 20일 수익률
    "n_limitup20",    # 최근 20일 자기 상한가 횟수 — 재료 소진
    "foreign1",       # 전일 외인 순매수 / 20일 평균 거래대금
    "inst1",          # 전일 기관 순매수 / 〃
    "foreign5",       # 5일 누적 외인 / 〃
    "inst5",          # 5일 누적 기관 / 〃
]


def load(conn) -> pd.DataFrame:
    """일봉·수급·상장주식수를 종목별 시계열로 읽어 피처와 라벨을 만든다.

    **미래 참조 차단이 이 함수의 전부다.** 당일 정보는 `open` 하나만 쓰고, 나머지
    피처는 전부 `shift(1)` 로 전 거래일까지만 본다.
    """
    bars = pd.read_sql(
        "select code, date, open::numeric o, high::numeric h, close::numeric c,"
        " trade_value::numeric tv from daily_bars order by code, date", conn)
    sup = pd.read_sql(
        "select code, date, foreign_::numeric fg, institution::numeric inst"
        " from supply_demand", conn)
    sh = pd.read_sql(
        "select code, date, shares_outstanding::numeric so"
        " from shares_outstanding_history", conn)

    for frame in (bars, sup, sh):
        frame["date"] = pd.to_datetime(frame["date"])   # merge_asof 는 날짜형을 요구한다
    df = bars.merge(sup, on=["code", "date"], how="left")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", sort=False)

    df["pc"] = g["c"].shift(1)
    df = df[df["pc"] > 0].copy()

    # 라벨 — 터치한 건 중 종가까지 굳혔는가
    df["touch"] = df["h"] / df["pc"] - 1 >= TOUCH
    df["lock"] = (df["c"] / df["pc"] - 1 >= TOUCH).astype(int)

    # 전 거래일까지만 보는 피처들
    g = df.groupby("code", sort=False)
    df["ret1"] = df["c"] / df["pc"] - 1
    df["tv_prev"] = g["tv"].shift(1)
    df["tv20"] = g["tv"].shift(1).rolling(20).mean().reset_index(level=0, drop=True)
    df["vol20"] = g["ret1"].shift(1).rolling(20).std().reset_index(level=0, drop=True)
    df["ret5"] = g["c"].shift(1) / g["c"].shift(6) - 1
    df["ret20"] = g["c"].shift(1) / g["c"].shift(21) - 1
    lim = (df["c"] / df["pc"] - 1 >= TOUCH).astype(float)
    df["n_limitup20"] = (lim.groupby(df["code"]).shift(1)
                         .rolling(20).sum().reset_index(level=0, drop=True))
    df["fg1"] = g["fg"].shift(1)
    df["inst1_raw"] = g["inst"].shift(1)
    df["fg5"] = g["fg"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)
    df["inst5_raw"] = g["inst"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)

    # 시총 대리 — 상장주식수는 발표일 기준이라 as-of 로 뒤로 채운다
    sh = sh.sort_values(["code", "date"])
    df = pd.merge_asof(df.sort_values("date"), sh.sort_values("date"),
                       on="date", by="code", direction="backward")
    df = df.sort_values(["code", "date"])

    df["gap"] = df["o"] / df["pc"] - 1
    df["lv_turnover"] = np.log1p(df["tv_prev"].clip(lower=0))
    df["lv_turnover20"] = np.log1p(df["tv20"].clip(lower=0))
    df["lv_mktcap"] = np.log1p((df["so"] * df["pc"]).clip(lower=0))
    denom = df["tv20"].replace(0, np.nan)
    df["foreign1"] = df["fg1"] / denom
    df["inst1"] = df["inst1_raw"] / denom
    df["foreign5"] = df["fg5"] / denom
    df["inst5"] = df["inst5_raw"] / denom
    return df[df["touch"]].copy()


def boot_ci(y: np.ndarray, n: int = BOOT, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [y[rng.integers(0, len(y), len(y))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


SUPPLY = ["foreign1", "inst1", "foreign5", "inst5"]


def main() -> int:
    import psycopg2
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        print("KR_QUANT_DB 없음", file=sys.stderr)
        return 2
    with psycopg2.connect(dsn) as conn:
        s = load(conn)

    feats = list(FEATURES)
    if "--no-supply" in sys.argv:
        # **탐색적 실행이다(사전등록 판정 아님).** supply_demand 가 2023년까지
        # 650종목만 담고 있어(2024년부터 2,700종목) 등록 시험의 학습 표본이
        # 대형주 편향된 6% 로 쪼그라들었다. 수급을 빼면 터치의 96% 가 남는다.
        feats = [f for f in feats if f not in SUPPLY]
        print("[탐색] 수급 피처 제외 — 사전등록 판정이 아니라 조건 진단이다\n")
    s = s.dropna(subset=feats)
    tr = s[s["date"] <= pd.Timestamp(TRAIN_END)]
    te = s[s["date"] >= pd.Timestamp(TEST_START)]
    print(f"표본 {len(s):,}건 (터치·피처 완비) · 학습 {len(tr):,} · 검정 {len(te):,}")
    print(f"기저 굳힘률  학습 {tr['lock'].mean():.1%} · 검정 {te['lock'].mean():.1%}")

    sc = StandardScaler().fit(tr[feats])
    model = LogisticRegression(C=1.0, max_iter=1000)   # 사전등록 §3 — 탐색 없음
    model.fit(sc.transform(tr[feats]), tr["lock"])
    te = te.assign(p=model.predict_proba(sc.transform(te[feats]))[:, 1])

    print("\n계수(표준화, 큰 순):")
    for name, coef in sorted(zip(feats, model.coef_[0]),
                             key=lambda kv: -abs(kv[1])):
        print(f"  {name:14s} {coef:+.3f}")

    base = te["lock"].mean()
    cut = te["p"].quantile(1 - TOP_Q)
    top = te[te["p"] >= cut]
    lo, hi = boot_ci(top["lock"].to_numpy())
    print(f"\n[1] 상위 {TOP_Q:.0%} (n={len(top):,}) 굳힘률 {top['lock'].mean():.1%} "
          f"· 95%CI [{lo:.1%}, {hi:.1%}] · 기저 {base:.1%}")
    c1 = top["lock"].mean() >= HIT_LINE and lo > base

    te = te.assign(q=pd.qcut(te["p"], 5, labels=False, duplicates="drop"))
    byq = te.groupby("q")["lock"].agg(["mean", "size"])
    rho = spearmanr(byq.index, byq["mean"]).statistic
    print(f"[2] 5분위 굳힘률: " + " · ".join(f"{m:.1%}" for m in byq["mean"])
          + f"  Spearman {rho:+.2f}")
    c2 = rho >= 0.8

    print("[3] 연도별 (상위 20% vs 그 해 기저):")
    c3 = True
    for y, grp in te.groupby(te["date"].dt.year):
        t = grp[grp["p"] >= grp["p"].quantile(1 - TOP_Q)]
        ok = t["lock"].mean() > grp["lock"].mean()
        c3 &= ok
        print(f"  {y}  {t['lock'].mean():.1%} vs {grp['lock'].mean():.1%} "
              f"(n={len(t):,})  {'OK' if ok else 'X'}")

    print(f"\n판정: [1]{'OK' if c1 else 'X'} [2]{'OK' if c2 else 'X'} "
          f"[3]{'OK' if c3 else 'X'} → "
          f"{'신호 있음' if (c1 and c2 and c3) else '기각'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
