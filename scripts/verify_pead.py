"""PEAD 측정 — 사전등록 23번 그대로 1회.

판정: 영업이익 YoY 서프라이즈, 분기내 데실, 롱숏 Q10-Q1, 주보유 +40거래일, 비용 후.
결과를 스크래치패드 pead_result.md 로 출력. DB 쓰기 없음.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg2
from pead_events import HORIZONS, LIQ_FLOOR_MWON, build_events, load_raw

OUT = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
       "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad/pead_result.md")
RT = 0.0034          # 왕복 비용 (주) 34bp
RT55 = 0.0055        # 병행 55bp
BORROW_40D = 0.005   # 숏 차입 연3% * ~2개월 ≈ 0.5%p (주보유 +40)
PRIMARY = 40


def decile_within(df, col="sue_raw"):
    """분기(period)내에서만 SUE 를 데실로. 전역 풀링 금지."""
    def _q(s):
        r = s.rank(method="first")
        return pd.qcut(r, 10, labels=range(1, 11)).astype(int)
    return df.groupby("period")[col].transform(_q)


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan, np.nan, len(x)
    m = x.mean()
    t = m / (x.std(ddof=1) / np.sqrt(len(x)))
    return m, t, len(x)


def main():
    L = []
    def w(s=""):
        L.append(s)
        print(s)

    earn, bars, raw, spac = load_raw()
    names = pd.read_sql("select code, name, market from stocks", psycopg2.connect(os.environ["KR_QUANT_DB"]))
    name_map = dict(zip(names.code, names.name))

    ev = build_events(earn, bars, raw, spac)

    w("# PEAD 측정 결과 (사전등록 23번, 1회)")
    w()
    w(f"- 유동성 하한 {LIQ_FLOOR_MWON}백만, 주 시그널 영업이익 YoY, 주보유 +{PRIMARY}거래일, 롱숏 Q10-Q1.")
    w(f"- 비용: 왕복 {RT*1e4:.0f}bp(주)/{RT55*1e4:.0f}bp, 숏 차입 +{PRIMARY}일 {BORROW_40D*100:.1f}%p.")
    w()

    # 판정 표본: 유니버스 & 진입 상한가 아님
    base = ev[ev["in_universe"] & ~ev["entry_limit_up"]].copy()
    base["decile"] = decile_within(base)

    # 상한가 오염 진단
    n_avail_lu = int(base["avail_day_limit_up"].sum())
    n_entry_lu_all = int((ev["in_universe"] & ev["entry_limit_up"]).sum())
    w("## 표본·상한가")
    w(f"- 판정 표본(유니버스, 진입 상한가 제외): {len(base):,}행, 분기 {base['period'].nunique()}개.")
    w(f"- 진입일 상한가(체결불가로 제외): {n_entry_lu_all}건.")
    w(f"- avail 당일 상한가 마감(오염 진단, 표본에 포함됨): {n_avail_lu}건 "
      f"({n_avail_lu/len(base)*100:.2f}%).")
    w()

    verdict = {}

    for H in HORIZONS:
        rc = f"ret{H}"
        d = base[base[rc].notna()].copy()
        tag = " (주)" if H == PRIMARY else ""
        w(f"## 보유 +{H}거래일{tag}")
        # 데실별 평균/중앙 (gross)
        g = d.groupby("decile")[rc]
        dec = pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size()})
        w("| 데실 | 평균% | 중앙% | N |")
        w("|---|---|---|---|")
        for dd in range(1, 11):
            if dd in dec.index:
                w(f"| Q{dd} | {dec.loc[dd,'mean']*100:+.2f} | {dec.loc[dd,'median']*100:+.2f} | {int(dec.loc[dd,'n'])} |")
        # 단조성: Spearman(데실, 평균수익)
        sp = pd.Series(dec["mean"].values, index=dec.index.astype(int)).sort_index()
        spear = np.corrcoef(sp.index, sp.rank())[0, 1]
        w(f"- 데실-평균수익 순위상관(Spearman) ≈ {spear:+.2f}; Q10−Q1 gross = "
          f"{(dec.loc[10,'mean']-dec.loc[1,'mean'])*100:+.2f}%")

        # 분기별 롱숏 스프레드
        q10 = d[d.decile == 10].groupby("period")[rc].mean()
        q1 = d[d.decile == 1].groupby("period")[rc].mean()
        spread = (q10 - q1).dropna()
        m, t, n = tstat(spread)
        # net: 양다리 비용 + 숏차입 (호라이즌 비례)
        borrow = BORROW_40D * (H / PRIMARY)
        net_ls = m - 2 * RT - borrow
        w(f"- 롱숏 분기 스프레드: 평균 {m*100:+.2f}%, t={t:+.2f}, N={n}분기; "
          f"net(양다리 68bp+차입{borrow*100:.1f}%p) = {net_ls*100:+.2f}%")
        neg_q = int((spread < 0).sum())
        w(f"- 음수 분기: {neg_q}/{n} ({neg_q/n*100:.0f}%)")

        # 롱온리 Q10 vs 유니버스 평균
        uni = d.groupby("period")[rc].mean()
        excess = (q10 - uni).dropna()
        me, te, _ne = tstat(excess)
        net_lo = q10.mean() - RT
        w(f"- 롱온리 Q10: gross {q10.mean()*100:+.2f}%, net34 {net_lo*100:+.2f}%; "
          f"vs 유니버스 초과 {me*100:+.2f}% (t={te:+.2f})")

        # 연도별 + 2024 전후
        d["year"] = d["avail_date"].dt.year
        yq10 = d[d.decile == 10].groupby("year")[rc].mean()
        yq1 = d[d.decile == 1].groupby("year")[rc].mean()
        yspread = (yq10 - yq1).dropna()
        neg_y = int((yspread < 0).sum())
        w(f"- 연도별 롱숏 스프레드(음수 {neg_y}/{len(yspread)}년):")
        w("  " + ", ".join(f"{int(y)}:{v*100:+.1f}" for y, v in yspread.items()))
        pre = spread[spread.index.str[:4].astype(int) < 2024]
        post = spread[spread.index.str[:4].astype(int) >= 2024]
        w(f"- 2024전 폴드 평균 {pre.mean()*100:+.2f}% (N={len(pre)}), "
          f"2024후 {post.mean()*100:+.2f}% (N={len(post)})")
        w()

        if H == PRIMARY:
            # 사전등록 6절-3: '음수 연도 없음(또는 극소)'. 극소 = |연도수익| < 1%p.
            mat_neg_y = int((yspread < -0.01).sum())
            verdict = {"spear": spear, "ls_mean": m, "ls_t": t, "net_ls": net_ls,
                           "neg_q": neg_q, "n_q": n, "neg_y": neg_y, "mat_neg_y": mat_neg_y,
                           "n_y": len(yspread), "pre": pre.mean(), "post": post.mean(),
                           "q10": dec.loc[10, "mean"], "q1": dec.loc[1, "mean"]}

    # 눈으로: 최근 분기 상/하위 데실 종목
    w("## 최근 분기(2026Q1) 상·하위 데실 종목 (눈 검사)")
    last = base[(base.period == "2026Q1") & base[f"ret{PRIMARY}"].notna()]
    for dd, lab in [(10, "상위 Q10"), (1, "하위 Q1")]:
        sub = last[last.decile == dd].sort_values("sue_raw", ascending=(dd == 1)).head(8)
        nm = ", ".join(f"{name_map.get(c, c)}({s:+.1f})" for c, s in zip(sub.code, sub.sue_raw))
        w(f"- {lab}: {nm}")
    w()

    # 기계적 판정
    v = verdict
    c1 = v["spear"] > 0 and v["q10"] > v["q1"]
    c2 = v["ls_mean"] > 0 and abs(v["ls_t"]) >= 2
    c3 = v["mat_neg_y"] == 0 and v["pre"] > 0 and v["post"] > 0 and (v["neg_q"] / v["n_q"] < 0.5)
    c4 = v["net_ls"] > 0
    w("## 판정 (사전등록 6절)")
    w(f"1. 단조성(Spearman>0 & Q10>Q1): {'OK' if c1 else 'FAIL'} (spear={v['spear']:+.2f})")
    w(f"2. 롱숏 양수 & |t|>=2: {'OK' if c2 else 'FAIL'} (mean={v['ls_mean']*100:+.2f}%, t={v['ls_t']:+.2f})")
    w(f"3. 폴드 일관성(유의미 음수연도0[극소 허용] & 2024전후 양수 & 음수분기<50%): {'OK' if c3 else 'FAIL'} "
      f"(음수연도 {v['neg_y']}/{v['n_y']} 중 <-1%p {v['mat_neg_y']}개, "
      f"전 {v['pre']*100:+.2f} 후 {v['post']*100:+.2f}, 음수분기 {v['neg_q']}/{v['n_q']})")
    w(f"4. 비용 후 롱숏 net>0: {'OK' if c4 else 'FAIL'} (net={v['net_ls']*100:+.2f}%)")
    allok = c1 and c2 and c3 and c4
    verdict_str = "채택" if allok else ("부분" if (c1 and v["ls_mean"] > 0) else "기각")
    w(f"\n### 최종: **{verdict_str}**")
    w("(강건성 5번 = 진입상한가 제외·중복0·오염 분리는 위에 반영됨.)")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()
