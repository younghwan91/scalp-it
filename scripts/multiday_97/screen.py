"""97번 급등주 다일 보유 — **유일하게 살아 있는 레버**를 우리 유니버스에서 잰다. 기술, 판정 아님.

    ~/of80/.rlvenv/bin/python scripts/multiday_97/screen.py --out data/multiday97

왜: [95번](../../docs/research/manju/95-cost-ceiling-measurement.md) 이 일중 레버 다섯 개를 전부 닫았다.
두 팀 측정을 합쳤을 때 남은 줄은 **보유 연장** 하나이고, daytrade-it 은 그걸 자기 뉴스 유니버스에서
+5d~+10d 순양수로 봤다. **우리 유니버스(급등주)에서도 그런지는 아무도 안 쟀다.**

고정 문턱을 두지 않는다(사용자 원칙). 유니버스는 **순위**로만 정의한다:
- 전일 거래대금 상위 `M` 안에서, 전일 수익률 상위 `N`.
- 진입은 다음 날 **시가**, 청산은 `h` 거래일 뒤 **종가**. 둘 다 사전확정이다.
- 비용 23bp 를 한 번 뺀다.

대조는 여기서 만들지 않는다 — daytrade-it 의 매치드 널(day × 전일 변동성 4분위 × 전일 거래대금 4분위,
10년 교정)에 **사건 목록을 넘겨** 돌린다. 우리가 대조를 새로 짜면 그쪽 도구와 중복이고 교정도 안 돼 있다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _envload  # noqa: E402,F401

COST = 0.0023
HOLDS = (1, 2, 3, 5, 10, 20)
MIN_PRICE = 1000


def load(conn, start: str) -> pd.DataFrame:
    df = pd.read_sql(
        "select code, date, open, high, low, close, volume, trade_value from daily_bars "
        "where date >= %s order by code, date", conn, params=(start,))
    df["date"] = pd.to_datetime(df.date)
    return df


def build_events(df: pd.DataFrame, *, top_value: int, top_ret: int) -> pd.DataFrame:
    """→ 사건 목록(진입일·종목·전일 지표)과 보유별 수익. 전부 사전확정이다."""
    df = df.sort_values(["code", "date"]).copy()
    g = df.groupby("code", sort=False)
    df["ret1"] = g.close.pct_change()
    df["prev_close"] = g.close.shift(1)
    # 전일 변동성(20일)·전일 거래대금 — 매칭용. 모두 신호일 **이전** 정보.
    df["vol20"] = g.ret1.transform(lambda s: s.rolling(20, min_periods=10).std())
    for h in HOLDS:
        df[f"fwd{h}"] = g.close.shift(-h) / df.open.shift(-1) - 1 - COST
    df = df[(df.close >= MIN_PRICE) & df.trade_value.notna()]
    day = df.groupby("date", sort=True)
    df["v_rank"] = day.trade_value.rank(ascending=False, method="first")
    sel = df[df.v_rank <= top_value].copy()
    sel["r_rank"] = sel.groupby("date").ret1.rank(ascending=False, method="first")
    ev = sel[sel.r_rank <= top_ret].copy()
    return ev.dropna(subset=["vol20"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-09-09")
    ap.add_argument("--top-value", type=int, default=300, help="전일 거래대금 상위 M")
    ap.add_argument("--top-ret", type=int, default=20, help="그 안에서 전일 수익률 상위 N")
    ap.add_argument("--out", default="data/multiday97")
    a = ap.parse_args()
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    ev = build_events(load(conn, a.start), top_value=a.top_value, top_ret=a.top_ret)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"사건 {len(ev):,}건 · {ev.date.min().date()}~{ev.date.max().date()} · "
          f"종목 {ev.code.nunique():,} · 거래일 {ev.date.nunique():,}")
    print(f"{'보유일':>6}{'건수':>10}{'평균bp':>10}{'중앙bp':>10}{'양(+)':>9}{'일평균bp':>11}")
    rows = []
    for h in HOLDS:
        r = ev[f"fwd{h}"].dropna().to_numpy() * 1e4
        if not len(r):
            continue
        rows.append({"hold": h, "n": int(len(r)), "mean": float(r.mean()),
                     "median": float(np.median(r)), "frac_pos": float((r > 0).mean())})
        print(f"{h:>6}{len(r):>10,}{r.mean():>+10.1f}{np.median(r):>+10.1f}"
              f"{(r > 0).mean():>9.1%}{r.mean() / h:>+11.2f}")
    # 사건 목록을 널 도구에 넘길 수 있는 모양으로 저장한다(daytrade-it 규격 + 매칭 열).
    cols = ["code", "date", "ret1", "vol20", "trade_value", *[f"fwd{h}" for h in HOLDS]]
    ev[cols].assign(date=ev.date.dt.strftime("%Y-%m-%d")).to_csv(out / "events.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"\n→ {out}/events.csv · {out}/summary.csv")
    print("대조는 daytrade-it 매치드 널에 넘긴다 — 여기서 새로 짜지 않는다(중복·미교정).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
