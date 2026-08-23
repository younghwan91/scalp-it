"""PEAD 결합 오버레이 측정 — 사전등록 25번 그대로 1회.

23·24 재사용(pead_events.build_events), 데실은 분기내 sue_raw, 유니버스 50억.
오버레이 3개: (1)수급 순매수(2024+), (2)12-1 모멘텀, (3)초기반응(진입 시프트).
판정: 필터 Q10 롱온리 초과 net34>0 & t>2 & 매칭베이스 대비 강화, 또는 롱숏 t 개선.
결과 → scratchpad pead_overlay.md. DB 쓰기 없음.
"""
from __future__ import annotations

import os
import warnings

import _envload  # noqa: F401
import numpy as np
import pandas as pd
import psycopg2
from pead_events import build_events, load_raw

warnings.filterwarnings("ignore")

OUT = ("/tmp/claude-1000/-home-young-Documents-git-scalp-it/"
       "92e8b58b-ceac-43df-bccd-3728ed037a08/scratchpad/pead_overlay.md")
RT = 0.0034          # 왕복 34bp/다리
BORROW_40D = 0.005   # 숏차입 +40일 0.5%p
LIQ50 = 5000         # 50억 배포 유니버스 (백만원)


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan, np.nan, len(x)
    m = x.mean()
    return m, m / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def decile_within(df, col="sue_raw"):
    def _q(s):
        r = s.rank(method="first")
        return pd.qcut(r, 10, labels=range(1, 11)).astype(int)
    return df.groupby("period")[col].transform(_q)


def longonly(d, retcol, longmask):
    """롱 다리: 분기별 롱 EW 평균 vs 유니버스 평균. gross Q10, 초과 mean/t, 초과 net."""
    dd = d[d[retcol].notna()]
    q = dd[longmask.loc[dd.index]].groupby("period")[retcol].mean()
    uni = dd.groupby("period")[retcol].mean()
    excess = (q - uni).dropna()
    me, te, ne = tstat(excess)
    return {"gross": q.mean(), "net34": q.mean() - RT,
            "exc": me, "exc_t": te, "exc_net": me - RT, "nq": ne,
            "n_long": int(longmask.loc[dd.index].sum())}


def longshort(d, retcol, longmask, shortmask):
    dd = d[d[retcol].notna()]
    ql = dd[longmask.loc[dd.index]].groupby("period")[retcol].mean()
    qs = dd[shortmask.loc[dd.index]].groupby("period")[retcol].mean()
    spread = (ql - qs).dropna()
    m, t, n = tstat(spread)
    return {"mean": m, "t": t, "n": n, "net": m - 2 * RT - BORROW_40D}


def main():
    L = []
    def w(s=""):
        L.append(s); print(s)

    earn, bars, raw, spac = load_raw()
    ev = build_events(earn, bars, raw, spac)

    # 종목별 수정주가 배열 (모멘텀·초기반응·시프트 진입)
    adj = {}
    for code, g in bars.groupby("code", sort=False):
        g = g.sort_values("date")
        adj[code] = (g["date"].values.astype("datetime64[ns]"),
                     g["open"].values.astype(float),
                     g["close"].values.astype(float))

    # supply_demand: code -> (dates, net_fi=foreign_+institution)
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    sd = pd.read_sql("select code, date, foreign_, institution from supply_demand "
                     "order by code, date", conn, parse_dates=["date"])
    conn.close()
    sd["net_fi"] = sd["foreign_"].fillna(0) + sd["institution"].fillna(0)
    sdd = {}
    for code, g in sd.groupby("code", sort=False):
        sdd[code] = (g["date"].values.astype("datetime64[ns]"),
                     g["net_fi"].values.astype(float))

    base = ev[ev["in_universe"] & ~ev["entry_limit_up"] & (ev["liq"] >= LIQ50)].copy()
    base["decile"] = decile_within(base)
    base["year"] = base["avail_date"].dt.year

    # per-event 부가지표: 모멘텀 M, 수급 net_fi, 초기반응 R0, 시프트 라벨 ret40s
    M = np.full(len(base), np.nan)
    NETFI = np.full(len(base), np.nan)
    R0 = np.full(len(base), np.nan)
    RET40S = np.full(len(base), np.nan)
    for i, r in enumerate(base.itertuples(index=False)):
        code = r.code
        avail = np.datetime64(pd.Timestamp(r.avail_date), "ns")
        if code in adj:
            dates, opens, closes = adj[code]
            ei = int(np.searchsorted(dates, avail, side="right"))  # 진입 idx (avail 초과 첫)
            # 12-1 모멘텀: close[ei-21]/close[ei-252]-1
            if ei - 252 >= 0 and ei - 21 >= 0 and closes[ei - 252] > 0:
                M[i] = closes[ei - 21] / closes[ei - 252] - 1.0
            # 초기반응 + 시프트: avail_last = ei-1
            al = ei - 1
            if al >= 0 and al + 46 < len(dates):
                c_last = closes[al]
                c5 = closes[al + 5]
                o6 = opens[al + 6]
                o46 = opens[al + 46]
                if c_last > 0:
                    R0[i] = c5 / c_last - 1.0
                if o6 > 0:
                    RET40S[i] = o46 / o6 - 1.0
        if code in sdd:
            sdates, snet = sdd[code]
            j = int(np.searchsorted(sdates, avail, side="right"))  # <=avail 은 j-1..
            lo = max(0, j - 20)
            if j - lo == 20:
                NETFI[i] = float(np.nansum(snet[lo:j]))
    base["mom"] = M
    base["net_fi"] = NETFI
    base["r0"] = R0
    base["ret40s"] = RET40S

    w("# PEAD 결합 오버레이 측정 (사전등록 25번, 1회)")
    w()
    w(f"- 유니버스 50억, 데실 분기내 sue_raw, 롱=Q10 숏=Q1, +40거래일. "
      f"비용 왕복 {RT*1e4:.0f}bp/다리, 숏차입 {BORROW_40D*100:.1f}%p.")
    w("- 판정: 필터 Q10 롱온리 초과 net34>0 & t>2 & 매칭베이스 대비 강화, 또는 롱숏 t 개선.")
    w("- **다중검정: 오버레이 3개 시험.**")
    w()

    results = {}

    # ---------- 오버레이 1: 수급 (2024+) ----------
    w("## 오버레이 1 — PEAD × 수급 (외인+기관 순매수, 2024+ 폴드)")
    d = base[(base["year"] >= 2024) & base["net_fi"].notna()].copy()
    rc = "ret40"
    is_q10 = d["decile"] == 10
    is_q1 = d["decile"] == 1
    filt_long = is_q10 & (d["net_fi"] > 0)
    w(f"- 부분표본: 2024+ 50억, net_fi 유효. Q10={int(is_q10.sum())}건, "
      f"필터통과(순매수>0) {int(filt_long.sum())}건 ({filt_long.sum()/max(is_q10.sum(),1)*100:.0f}%).")
    lo_f = longonly(d, rc, filt_long)
    lo_b = longonly(d, rc, is_q10)  # 매칭 베이스
    ls_f = longshort(d, rc, filt_long, is_q1)
    ls_b = longshort(d, rc, is_q10, is_q1)
    w("| | gross Q10% | 초과%(t) | 초과net34% | N분기 |")
    w("|---|---|---|---|---|")
    w(f"| 필터(순매수) | {lo_f['gross']*100:+.2f} | {lo_f['exc']*100:+.2f}({lo_f['exc_t']:+.2f}) | {lo_f['exc_net']*100:+.2f} | {lo_f['nq']} |")
    w(f"| 매칭베이스 Q10 | {lo_b['gross']*100:+.2f} | {lo_b['exc']*100:+.2f}({lo_b['exc_t']:+.2f}) | {lo_b['exc_net']*100:+.2f} | {lo_b['nq']} |")
    w(f"- 롱숏: 필터 {ls_f['mean']*100:+.2f}% t={ls_f['t']:+.2f} net={ls_f['net']*100:+.2f}% (N={ls_f['n']}); "
      f"베이스 {ls_b['mean']*100:+.2f}% t={ls_b['t']:+.2f} net={ls_b['net']*100:+.2f}%")
    results["1_수급"] = (lo_f, lo_b, ls_f, ls_b)
    w()

    # ---------- 오버레이 2: 모멘텀 ----------
    w("## 오버레이 2 — PEAD × 12-1 모멘텀 (상위 절반)")
    d = base[base["mom"].notna()].copy()
    is_q10 = d["decile"] == 10
    is_q1 = d["decile"] == 1
    # 분기 Q10 내 모멘텀 중앙값 이상
    med = d[is_q10].groupby("period")["mom"].transform("median")
    filt_long = pd.Series(False, index=d.index)
    filt_long.loc[is_q10[is_q10].index] = (d.loc[is_q10, "mom"] >= med)
    w(f"- 표본: 모멘텀 유효 50억. Q10={int(is_q10.sum())}건, 상위절반 {int(filt_long.sum())}건.")
    lo_f = longonly(d, rc, filt_long)
    lo_b = longonly(d, rc, is_q10)
    ls_f = longshort(d, rc, filt_long, is_q1)
    ls_b = longshort(d, rc, is_q10, is_q1)
    w("| | gross Q10% | 초과%(t) | 초과net34% | N분기 |")
    w("|---|---|---|---|---|")
    w(f"| 필터(모멘텀상위) | {lo_f['gross']*100:+.2f} | {lo_f['exc']*100:+.2f}({lo_f['exc_t']:+.2f}) | {lo_f['exc_net']*100:+.2f} | {lo_f['nq']} |")
    w(f"| 매칭베이스 Q10 | {lo_b['gross']*100:+.2f} | {lo_b['exc']*100:+.2f}({lo_b['exc_t']:+.2f}) | {lo_b['exc_net']*100:+.2f} | {lo_b['nq']} |")
    w(f"- 롱숏: 필터 {ls_f['mean']*100:+.2f}% t={ls_f['t']:+.2f} net={ls_f['net']*100:+.2f}% (N={ls_f['n']}); "
      f"베이스 {ls_b['mean']*100:+.2f}% t={ls_b['t']:+.2f} net={ls_b['net']*100:+.2f}%")
    results["2_모멘텀"] = (lo_f, lo_b, ls_f, ls_b)
    w()

    # ---------- 오버레이 3: 초기반응 (시프트 진입) ----------
    w("## 오버레이 3 — PEAD 드리프트 지속조건 (초기반응>0, 진입 avail+6 시프트)")
    d = base[base["ret40s"].notna() & base["r0"].notna()].copy()
    rc3 = "ret40s"
    is_q10 = d["decile"] == 10
    is_q1 = d["decile"] == 1
    filt_long = is_q10 & (d["r0"] > 0)
    w(f"- 표본: 시프트 라벨 유효 50억. Q10={int(is_q10.sum())}건, 초기반응>0 {int(filt_long.sum())}건 "
      f"({filt_long.sum()/max(is_q10.sum(),1)*100:.0f}%).")
    lo_f = longonly(d, rc3, filt_long)
    lo_b = longonly(d, rc3, is_q10)
    ls_f = longshort(d, rc3, filt_long, is_q1)
    ls_b = longshort(d, rc3, is_q10, is_q1)
    w("| | gross Q10% | 초과%(t) | 초과net34% | N분기 |")
    w("|---|---|---|---|---|")
    w(f"| 필터(초기반응>0) | {lo_f['gross']*100:+.2f} | {lo_f['exc']*100:+.2f}({lo_f['exc_t']:+.2f}) | {lo_f['exc_net']*100:+.2f} | {lo_f['nq']} |")
    w(f"| 매칭베이스 Q10 | {lo_b['gross']*100:+.2f} | {lo_b['exc']*100:+.2f}({lo_b['exc_t']:+.2f}) | {lo_b['exc_net']*100:+.2f} | {lo_b['nq']} |")
    w(f"- 롱숏(시프트): 필터 {ls_f['mean']*100:+.2f}% t={ls_f['t']:+.2f} net={ls_f['net']*100:+.2f}% (N={ls_f['n']}); "
      f"베이스 {ls_b['mean']*100:+.2f}% t={ls_b['t']:+.2f} net={ls_b['net']*100:+.2f}%")
    results["3_초기반응"] = (lo_f, lo_b, ls_f, ls_b)
    w()

    # ---------- 판정 ----------
    w("## 판정 (사전등록 3절, 측정 전 고정)")
    w()
    any_adopt = False
    for name, (lo_f, lo_b, ls_f, ls_b) in results.items():
        long_ok = (lo_f["exc_net"] > 0 and lo_f["exc_t"] > 2
                   and lo_f["exc_net"] > lo_b["exc_net"] and lo_f["exc_t"] >= lo_b["exc_t"])
        ls_ok = (ls_f["net"] > 0 and ls_f["t"] > ls_b["t"])
        adopt = long_ok or ls_ok
        any_adopt = any_adopt or adopt
        w(f"- **{name}**: 롱다리 살림={'OK' if long_ok else 'X'} "
          f"(초과net {lo_f['exc_net']*100:+.2f}% t={lo_f['exc_t']:+.2f} vs 베이스 "
          f"{lo_b['exc_net']*100:+.2f}%/{lo_b['exc_t']:+.2f}); "
          f"롱숏개선={'OK' if ls_ok else 'X'} (t {ls_f['t']:+.2f} vs {ls_b['t']:+.2f}) "
          f"→ **{'채택' if adopt else '기각'}**")
    w()
    if any_adopt:
        w("### 전체: 오버레이 일부 채택 → 실전 스펙 갱신 검토 (단, 다중검정 3개 감안).")
    else:
        w("### 전체: **3개 전부 기각 → PEAD 순수 롱숏이 최선.**")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
