import glob
import pickle
import sys

import numpy as np
import pandas as pd

COST = 0.0023
start, end = sys.argv[1:3]
HS = (300, 600, 1200, 1800, 3600)
out = []
for f in sorted(glob.glob("/home/young/of80/cache/*.pkl")):
    if f.endswith(".path.pkl"):
        continue
    day = f.split("/")[-1][:10]
    if not (start <= day <= end):
        continue
    df = pickle.load(open(f, "rb"))
    paths = pickle.load(open(f.replace(".pkl", ".path.pkl"), "rb"))
    for liq, lo_v in (("v600>=3e9", 3e9), ("v600>=7e9", 7e9), ("all", 0)):
        for name, m in {
            "trend": (df.v600 >= lo_v) & (df.acc >= 2) & (df.buyshare30 >= 0.7) & (df.ret300 > 0.005) & (df.spread_bp <= 15),
            "trend_strong": (df.v600 >= lo_v) & (df.acc >= 3) & (df.buyshare30 >= 0.75) & (df.ret300 > 0.01) & (df.spread_bp <= 15),
            "random": (df.v600 >= lo_v) & (df.spread_bp <= 15),
        }.items():
            g = df.loc[m, ["code", "sec"]]
            if name == "random":
                g = g.sample(min(len(g), 300), random_state=80)
            for code, gg in g.groupby("code"):
                last = -10**9
                for t in sorted(gg.sec):
                    if t - last < 600:
                        continue
                    last = t
                    p = paths[code]
                    e = t + 1
                    ask = p["ask"][e]
                    if np.isnan(ask):
                        continue
                    rec = dict(day=day, liq=liq, sig=name, code=code)
                    bid = p["bid"].astype(float)
                    n = len(bid)
                    for h in HS:
                        rec[f"net{h}"] = bid[min(e + h, n - 1)] / ask - 1 - COST
                    rec["netclose"] = bid[n - 1] / ask - 1 - COST
                    # 트레일 1% (최고 매수호가 대비), 최대 60분
                    w = bid[e:min(e + 3600, n)]
                    peak = np.fmax.accumulate(np.nan_to_num(w, nan=0))
                    hit = w <= peak * 0.99
                    i = int(np.argmax(hit)) if hit.any() else len(w) - 1
                    rec["trail1"] = w[i] / ask - 1 - COST
                    out.append(rec)
t = pd.DataFrame(out)
cols = [c for c in t.columns if c.startswith("net") or c == "trail1"]
pd.set_option("display.width", 250)
agg = t.groupby(["liq", "sig"]).agg(n=("code", "size"), days=("day", "nunique"),
                                    **{c: (c, lambda x: x.mean() * 1e4) for c in cols})
print(agg.round(1).to_string())
med = t.groupby(["liq", "sig"])[cols].median() * 1e4
print("\nmedian\n", med.round(1).to_string())
dm = t.groupby(["liq", "sig", "day"])[cols].mean().groupby(["liq", "sig"]).agg(lambda x: f"{(x > 0).sum()}/{len(x)}")
print("\npos days\n", dm.to_string())
