import sys
from datetime import date as _date

import numpy as np
import pandas as pd
from krx_quant_core.costs import round_trip_cost
from krx_quant_core.market.codes import Market

COST = round_trip_cost(_date(2026, 1, 2), Market.KOSDAQ)  # 2026 세율 — 날짜 문맥 없음

t = pd.read_csv(sys.argv[1], dtype={"code": str})
pd.set_option("display.width", 250)
rows = []
for topn in (3, 5, 10):
    u = t[t["rank"] < topn]
    for key, g in u.groupby(["sig", "tp", "scr", "hold"]):
        dm = g.groupby("day").net.mean()
        rows.append(dict(topn=topn, sig=key[0], tp=key[1], scr=key[2], hold=key[3], n=len(g),
                         per_day=len(g) / g.day.nunique(), mean=g.net.mean() * 1e4, med=g.net.median() * 1e4,
                         win=(g.net > 0).mean(), daymean=dm.mean() * 1e4, posdays=f"{(dm > 0).sum()}/{len(dm)}",
                         held=g.held.median(), tick_bp=g.tick_bp.median()))
r = pd.DataFrame(rows)
print("cells", len(r), "daymean>0:", (r.daymean > 0).sum())
print(r.sort_values("daymean", ascending=False).head(20).round(2).to_string())
print("\nsignal avg (topn=5)")
print(r[r.topn == 5].groupby("sig")[["mean", "win", "per_day"]].mean().round(2).to_string())
print("\ngross decomposition best cell")
b = r.sort_values("daymean", ascending=False).iloc[0]
g = t[(t["rank"] < b.topn) & (t.sig == b.sig) & (t.tp == b.tp) & (t.scr == b.scr) & (t.hold == b.hold)]
print("gross mean bp", round((g.net.mean() + COST) * 1e4, 1), "win gross>0", round(((g.net + COST) > 0).mean(), 3))
print(g.groupby("code").net.agg(["size", "mean"]).assign(mean=lambda x: x["mean"] * 1e4).round(1).to_string())
