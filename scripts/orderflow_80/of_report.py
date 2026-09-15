import pickle
import sys

import numpy as np
import pandas as pd

d = pickle.load(open(sys.argv[1], "rb"))
EXITS = d["exits"]
out = []
for sig in d["signals"]:
    parts = []
    for day, members, df in d["res"]:
        if df.empty:
            continue
        mem = pd.DataFrame(members[sig], columns=["code", "sec"])
        if mem.empty:
            continue
        parts.append(df.merge(mem, on=["code", "sec"]))
    if not parts:
        continue
    a = pd.concat(parts, ignore_index=True)
    for mode in ("taker", "maker"):
        b = a[a["mode"] == mode]
        fill = b.filled.mean() if len(b) else np.nan
        b = b[b.filled]
        for k, (tp, sl, wall, T) in enumerate(EXITS):
            r = b[k].dropna()
            if len(r) < 30:
                continue
            daym = b.assign(r=b[k]).groupby("day").r.mean()
            out.append(dict(sig=sig, mode=mode, tp=tp, sl=sl, wall=wall, T=T, n=len(r),
                            fill=round(fill, 2), mean=r.mean() * 1e4, med=r.median() * 1e4,
                            win=(r > 0).mean(), daymean=daym.mean() * 1e4,
                            posdays=f"{(daym > 0).sum()}/{len(daym)}",
                            trim1=np.sort(r.values)[:-max(1, int(np.ceil(0.01 * len(r))))].mean() * 1e4))
t = pd.DataFrame(out)
pd.set_option("display.width", 250)
print("cells", len(t), "positive daymean", (t.daymean > 0).sum(), "positive mean&med", ((t["mean"] > 0) & (t.med > 0)).sum())
print(t.sort_values("daymean", ascending=False).head(25).round(1).to_string())
print("\nby mode/tp/wall avg daymean")
print(t.groupby(["mode", "tp", "wall"], dropna=False).daymean.mean().round(1).to_string())
t.to_csv(sys.argv[1].replace(".pkl", ".csv"), index=False)
