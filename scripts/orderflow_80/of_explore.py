import glob
import pickle
import sys

import numpy as np
import pandas as pd

IS_END = "2026-09-07"
HS = (5, 15, 30, 60, 120, 300)
files = sorted(glob.glob("/home/young/of80/cache/*.pkl"))
df = pd.concat([pickle.load(open(f, "rb")) for f in files if f.split("/")[-1][:10] <= IS_END],
               ignore_index=True)
print("IS rows", len(df), "days", df.day.nunique(), "codes", df.code.nunique())
nets = [f"net{h}" for h in HS]
pd.set_option("display.width", 250)


def show(name, g):
    t = g[nets].mean() * 1e4
    t["n"] = len(g)
    t["days"] = g.day.nunique()
    t["codes"] = g.code.nunique()
    t["mfe60"] = g.mfe60.median() * 1e4
    t["spr"] = g.spread_bp.median()
    return t.rename(name)


print("\n== baseline")
print(pd.DataFrame([show("all", df)]).round(1).to_string())

for feat in ("acc", "buyshare30", "buyshare10", "ret30", "ret300", "ofi10", "imb3", "spread_bp", "v600"):
    q = pd.qcut(df[feat], 10, duplicates="drop")
    rows = [show(str(k), g) for k, g in df.groupby(q, observed=True)]
    print(f"\n== {feat} deciles (bp)")
    print(pd.DataFrame(rows).round(1).to_string())

print("\n== combos (bp)")
rows = []
for acc in (3, 5, 10, 20):
    for bs in (0.6, 0.75, 0.9):
        for r in (0.0, 0.005, 0.01):
            for spr in (1e9, 15):
                m = (df.acc >= acc) & (df.buyshare30 >= bs) & (df.ret30 > r) & (df.spread_bp <= spr)
                g = df[m]
                if len(g) < 200:
                    continue
                rows.append(show(f"acc{acc} bs{bs} r{r} spr{spr:g}", g))
print(pd.DataFrame(rows).round(1).sort_values("net60", ascending=False).head(30).to_string())
