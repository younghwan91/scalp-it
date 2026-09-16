"""80-3: 체결강도 가속 × 1틱 스프레드 대형주. IS 만.

1) 체결강도 수준·10/30/60초 변화 십분위 → 고정 지평 순수익
2) 경로 시뮬: 진입 = 1틱 스프레드 + 체결강도 가속(dstr30 상위) [+매수비중]
   청산(이벤트) = 체결강도 꺾임(진입 이후 최고치 대비 −k) · 매수벽 붕괴 · 익절 +n틱 · 최대 보유
"""
import glob
import itertools
import pickle
import sys
from datetime import date as _date

import numpy as np
import pandas as pd
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.market.codes import Market
from krx_quant_core.market.ticks import tick_size_int

COST = round_trip_cost(_date(2026, 1, 2), Market.KOSDAQ)  # 2026 세율 — 날짜 문맥 없음
IS_END = "2026-09-07"
HS = (5, 15, 30, 60, 120, 300)
CACHE = "/home/young/of80/cache2"
nets = [f"net{h}" for h in HS]
pd.set_option("display.width", 250)


def tick(p):
    # 호가단위 정본은 krx-quant-core(주식 표). NaN 은 옛 표 구현처럼 최상위 틱(1000).
    return tick_size_int(p) if p == p else 1000


def show(name, g):
    t = g[nets].mean() * 1e4
    t["n"] = len(g); t["days"] = g.day.nunique(); t["codes"] = g.code.nunique()
    t["gross60"] = (g.net60.mean() + COST) * 1e4
    return t.rename(name)


files = [f for f in sorted(glob.glob(f"{CACHE}/*.pkl")) if not f.endswith(".path.pkl")
         and f.split("/")[-1][:10] <= IS_END]
df = pd.concat([pickle.load(open(f, "rb")) for f in files], ignore_index=True)
df = df[df.strength.notna()]
one = df[(df.spread_ticks <= 1.01)]
print("IS rows", len(df), "1tick rows", len(one), "1tick codes", one.code.nunique())
print(pd.DataFrame([show("all", df), show("1tick", one)]).round(1).to_string())

for name, d in (("all", df), ("1tick", one)):
    for feat in ("strength", "dstr10", "dstr30", "dstr60"):
        q = pd.qcut(d[feat], 10, duplicates="drop")
        print(f"\n== {name} {feat} deciles (bp)")
        print(pd.DataFrame([show(str(k), g) for k, g in d.groupby(q, observed=True)]).round(1).to_string())

# ---- 경로 시뮬 ----
SIG = {
    "dstr30_top5": lambda d: d.dstr30 >= d.dstr30.quantile(0.95),
    "dstr30_top1": lambda d: d.dstr30 >= d.dstr30.quantile(0.99),
    "dstr10_top1": lambda d: d.dstr10 >= d.dstr10.quantile(0.99),
    "dstr30_top5_bs": lambda d: (d.dstr30 >= d.dstr30.quantile(0.95)) & (d.buyshare10 >= 0.7) & (d.ret30 > 0),
    "str>150_dstr30_top5": lambda d: (d.strength >= 150) & (d.dstr30 >= d.dstr30.quantile(0.95)),
}
EXITS = list(itertools.product((None, 2, 4), (None, 3.0, 10.0), (None, 0.3), (60, 300)))


def sim(p, e, entry, tp_k, sdrop, wall, T):
    n = len(p["bid"])
    end = min(e + T, n - 1)
    bid = p["bid"]; hi = p["hi"]; stg = p["strength"]; bv = p["bidval1"]
    tk = tick(entry)
    peak = stg[e]; w0 = bv[e]; b0 = bid[e]
    for s in range(e + 1, end + 1):
        if tp_k is not None and hi[s] == hi[s] and hi[s] > entry + tp_k * tk:
            return entry + tp_k * tk, s - e
        if stg[s] == stg[s]:
            peak = max(peak, stg[s])
            if sdrop is not None and stg[s] <= peak - sdrop:
                return float(bid[s]), s - e
        if wall is not None and w0 > 0 and bv[s] < wall * w0 and bid[s] <= b0:
            return float(bid[s]), s - e
    return float(bid[end]), end - e


rows = []
for f in files:
    day = f.split("/")[-1][:10]
    d0 = pickle.load(open(f, "rb"))
    d0 = d0[d0.strength.notna() & (d0.spread_ticks <= 1.01)]
    paths = pickle.load(open(f.replace(".pkl", ".path.pkl"), "rb"))
    for sname, fn in SIG.items():
        # 분위수는 IS 전체가 아니라 그날 자기 자신으로(하루 안에서 상대적) — 미래 날짜 안 씀
        m = fn(d0)
        for code, g in d0[m].groupby("code"):
            p = paths[code]
            free = -1
            for t in g.sec.to_numpy():
                if t < free:
                    continue
                e = t + 1
                a = p["ask"][e]
                if not (a == a and a > 0):
                    continue
                for k, (tp_k, sdrop, wall, T) in enumerate(EXITS):
                    px, held = sim(p, e, float(a), tp_k, sdrop, wall, T)
                    rows.append((day, code, sname, k, px / a - 1 - COST, held))
                free = e + 60
r = pd.DataFrame(rows, columns=["day", "code", "sig", "k", "net", "held"])
out = []
for (sname, k), g in r.groupby(["sig", "k"]):
    dm = g.groupby("day").net.mean()
    tp_k, sdrop, wall, T = EXITS[k]
    out.append(dict(sig=sname, tp=tp_k, sdrop=sdrop, wall=wall, T=T, n=len(g), codes=g.code.nunique(),
                    mean=g.net.mean() * 1e4, med=g.net.median() * 1e4, gross=(g.net.mean() + COST) * 1e4,
                    win=(g.net > 0).mean(), daymean=dm.mean() * 1e4, posdays=f"{(dm > 0).sum()}/{len(dm)}",
                    held=g.held.median()))
o = pd.DataFrame(out)
print("\n== path sim cells", len(o), "daymean>0", (o.daymean > 0).sum())
print(o.sort_values("daymean", ascending=False).head(20).round(1).to_string())
o.to_csv("/home/young/of80/strength_is.csv", index=False)
