"""82번 최종 OOS 평가 — A·B 각각 **한 번**. 사전등록 §5·§6.

    ~/of80/.rlvenv/bin/python scripts/rl_82/evaluate.py --final
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import dqn
from common import fmt, gates, random_control
from dataset import load
from lgbm import policy_trades

OUT = Path.home() / "of80" / "rl82"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", required=True)
    ap.parse_args()
    lock = OUT / "FINAL_DONE"
    if lock.exists():
        raise SystemExit("OOS 평가는 이미 한 번 돌았다 — 사전등록 §1. 다시 돌리지 않는다.")
    oos = load("oos", allow_oos=True)
    by_key = {(e.day, e.code): e for e in oos}
    lines = ["## 82번 OOS 최종 평가 (2026-09-08 ~ 09-15)", ""]

    # --- A
    ma = pickle.load(open(OUT / "model_a.pkl", "rb"))
    booster = lgb.Booster(model_str=ma["booster"])
    sc = ma["scaler"]
    preds = [booster.predict(sc(e.X)) for e in oos]
    ta = policy_trades(oos, preds, ma["selected"]["thr_bp"])
    ta["hold"] = [by_key[(r.day, r.code)].hold[r.i] for r in ta.itertuples()]
    ta = random_control(ta, by_key)
    ga = gates(ta)
    lines += [fmt(f"A LightGBM (VAL 선택 q={ma['selected']['q']})", ga), ""]
    lines += ["종목별 상위 거래 수: " + ", ".join(f"{c} {n}" for c, n in ta.code.value_counts().head(5).items()), ""]

    # --- B
    sel = json.loads((OUT / "dqn_selected.json").read_text())
    q = dqn.Q()
    q.load_state_dict(torch.load(OUT / f"dqn_{sel['steps'] // 1_000_000:02d}M.pt"))
    scb = pickle.load(open(OUT / "scaler_b.pkl", "rb"))
    tb = dqn.greedy_trades(q, dqn.Book(oos, scb))
    tb = random_control(tb, by_key)
    gb = gates(tb)
    lines += [fmt(f"B Double DQN (VAL 선택 {sel['steps'] // 1_000_000}M 스텝)", gb), ""]
    lines += [f"B 보유초 중앙 {tb.hold.median() if len(tb) else '-'} · 하루 거래 {len(tb) / max(1, tb.day.nunique()):.0f}건", ""]

    ta.to_csv(OUT / "oos_trades_a.csv", index=False)
    tb.to_csv(OUT / "oos_trades_b.csv", index=False)
    text = "\n".join(lines)
    (OUT / "final_report.md").write_text(text)
    lock.write_text("done")
    print(text)


if __name__ == "__main__":
    main()
