"""PEAD 실전 포트폴리오 생성 — 지금 매수/매도할 종목.

24번 강건성 통과(풀 롱숏, 실전 유니버스 50억) 후 배포 산출물.
최신 avail 분기 기준으로 분기내 데실을 매겨 롱(Q10)·숏(Q1) 리스트를 출력한다.
진입은 avail 익일 시가, 보유 +40거래일(≈2개월), 분기 리밸.
결과 → scratchpad pead_current_portfolio.md. DB 쓰기 없음.
"""
from __future__ import annotations

import os

import pandas as pd
import psycopg2
from pead_events import build_events, load_raw

OUT = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
       "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad/pead_current_portfolio.md")
DEPLOY_LIQ = 5000  # 배포 유동성 하한 50억(백만원). 24번: 50억서 net34 +4.47%, t=6.15, 47종목/side.


def main():
    L = []
    def w(s=""):
        L.append(s); print(s)

    earn, bars, raw, spac = load_raw()
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    meta = pd.read_sql("select code, name, sector, market from stocks", conn)
    conn.close()
    name_map = dict(zip(meta.code, meta.name))
    sec_map = dict(zip(meta.code, meta.sector))
    mkt_map = dict(zip(meta.code, meta.market))

    ev = build_events(earn, bars, raw, spac)
    uni = ev[~ev["spac"] & ev["sue_raw"].notna() & ~ev["entry_limit_up"]
             & (ev["liq"] >= DEPLOY_LIQ)].copy()

    # 최신 형성가능 분기 = 진입가(entry_open) 존재하고 종목수 >= 30 인 최대 period
    valid = (uni.groupby("period").size())
    cand = valid[valid >= 30].index
    period = max(cand)
    cur = uni[uni["period"] == period].copy()

    # 분기내 데실
    r = cur["sue_raw"].rank(method="first")
    cur["decile"] = pd.qcut(r, 10, labels=range(1, 11)).astype(int)
    cur["name"] = cur["code"].map(name_map)
    cur["sector"] = cur["code"].map(sec_map)
    cur["market"] = cur["code"].map(mkt_map)

    avail = cur["avail_date"].max()
    w("# PEAD 현재 포트폴리오 (배포 산출물, 24번 통과)")
    w()
    w(f"- 분기 **{period}** (avail {avail.date()}), 실전 유니버스 유동성 ≥ {DEPLOY_LIQ}백만(50억).")
    w(f"- 유니버스 {len(cur)}종목, 데실당 ≈ {len(cur)//10}종목.")
    w("- 진입: avail 익일 시가. 보유: +40거래일(≈2개월). 다음 분기 신호 도착 시 리밸.")
    w("- **롱 = Q10(서프라이즈 최상), 숏 = Q1(최하).** 동일가중. 왕복 68bp+숏차입 가정.")
    w()

    def dump(dd, title, asc):
        sub = dd.sort_values("sue_raw", ascending=asc)
        w(f"## {title} ({len(sub)}종목)")
        w()
        w("| # | 코드 | 이름 | 시장 | 섹터 | SUE(raw) | 20일거래대금(억) |")
        w("|---|---|---|---|---|---|---|")
        for i, x in enumerate(sub.itertuples(), 1):
            w(f"| {i} | {x.code} | {x.name} | {x.market} | {x.sector} | "
              f"{x.sue_raw:+.2f} | {x.liq/100:.0f} |")
        w()

    longs = cur[cur.decile == 10]
    shorts = cur[cur.decile == 1]
    dump(longs, "롱 (Q10) — 지금 매수", asc=False)
    dump(shorts, "숏 (Q1) — 지금 매도(대차)", asc=True)

    w("## 실전 메모")
    w("- 500억 유동성서도 롱숏 net>0(t=2.54)이나 종목 7개/side 로 과집중 → 배포는 50억(47종목/side).")
    w("- 500억선 롱온리 Q10 초과수익 무의(t=0.48) → **엣지는 롱숏(특히 숏 다리)에서 온다.** 롱온리 단독 비권장.")
    w("- 숏 실행가능성 99%(2024+ Q1 종목의 short_selling 등장률) → 대차 가능.")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
