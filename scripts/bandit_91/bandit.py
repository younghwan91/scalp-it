"""91번 — 89번 2단 지정가 밴딧을 지정가용 30피처로. 사전등록 docs/research/manju/91.

    ~/of80/.rlvenv/bin/python scripts/bandit_91/bandit.py        # 데이터 준비 + 17일 선행 평가(기술)
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home() / "of80"
DATA = HOME / "bandit91"
FEATS = json.loads((Path(__file__).parent / "features.json").read_text())
FORWARD_START = "2026-09-16"

_spec = importlib.util.spec_from_file_location("bandit89_mod", ROOT / "scripts" / "bandit_89" / "bandit.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["bandit89_mod"] = B
_spec.loader.exec_module(B)


def build(day: str, *, forward_ok: bool = False) -> None:
    if not forward_ok:
        assert day < FORWARD_START
    out = DATA / f"{day}.pkl"
    if out.exists():
        return
    rew = pickle.loads((HOME / "bandit88" / f"{day}.pkl").read_bytes())[["day", "code", "row", "y1", "free1", "y2",
                                                                          "free2", "filled2"]]
    rows = pickle.loads((HOME / "flow84" / f"{day}.pkl").read_bytes())[["code", "row", *FEATS]]
    DATA.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(rew.merge(rows, on=["code", "row"]), protocol=5))


def configure(out: Path) -> None:
    B.DATA, B.OUT, B.SEED = DATA, out, 91


def run(days, *, forward_ok=False, state_in=None, out=HOME / "bandit91_run"):
    for d in days:
        build(d, forward_ok=forward_ok)
    configure(out)
    return B.run(days, forward_ok=forward_ok, state_in=state_in, feats=FEATS)


if __name__ == "__main__":
    from krx_quant_core.runtime import DataSpec, start_run

    days = sorted(p.name[:10] for p in (HOME / "bandit88").glob("*.pkl") if p.name[:10] < FORWARD_START)
    for d in days:
        build(d)
    configure(HOME / "bandit91_run")
    with start_run("bandit91", {"prereg": "docs/research/manju/91", "feats": FEATS}, repo_root=ROOT,
                   data=DataSpec("2026-08-24", "2026-09-15", "prequential"), seed=91, allow_dirty=True) as r_:
        t = B.run(days, feats=FEATS)
        # 89번 main 의 집계를 그대로 쓴다
        B.OUT.mkdir(parents=True, exist_ok=True)
        t.to_csv(B.OUT / "trades.csv", index=False)
        ev = t[~t.day.isin(days[:B.WARMUP_DAYS])]
        f = ev[ev.filled]
        dm = f.groupby("day").ret_bp.mean()
        rep = {"orders": len(ev), "filled": len(f), "fill_rate": round(len(f) / max(len(ev), 1), 3),
               "mean_bp": float(f.ret_bp.mean()) if len(f) else None,
               "median_bp": float(f.ret_bp.median()) if len(f) else None,
               "daymean_bp": float(dm.mean()) if len(f) else None, "pos_days": f"{int((dm > 0).sum())}/{len(dm)}",
               "per_day": f.groupby("day").ret_bp.agg(["size", "mean"]).round(1).to_dict(orient="index")}
        (B.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        r_.log_result(rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "per_day"}, ensure_ascii=False, indent=1))
