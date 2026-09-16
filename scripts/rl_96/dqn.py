"""96번 분 단위 지정가 RL — 사전등록 docs/research/manju/96 §2.

    ~/of80/.rlvenv/bin/python scripts/rl_96/dqn.py --seed 196
    ~/of80/.rlvenv/bin/python scripts/rl_96/dqn.py --pick

93번과 다른 점은 셋뿐이다(나머지는 그대로 재사용한다 — 같은 코드여야 비교가 된다):
1. 상태에 **분 단위 피처**(scripts/minute_96/features.py)를 더한다. 93 의 창은 1~30초뿐이라 분 맥락이 없었다.
2. 행동에서 **시장가 매수를 뺀다** — 95번 천장 측정이 오라클 기준으로도 불가능함을 보였다(−29~−36bp).
3. γ = 0.999 — 95번이 가리키는 5~30분 구간을 시야에 담는다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R85 = _load("rl85_dqn", "scripts/rl_85/dqn.py")
R93 = _load("rl93_dqn", "scripts/rl_93/dqn.py")
M96 = _load("minute96_features", "scripts/minute_96/features.py")

HOME = Path.home() / "of80"
FILLS = HOME / "fills87"
OUT = Path(os.environ.get("RL96_OUT", HOME / "rl96"))
TOTAL = int(os.environ.get("RL96_TOTAL", 40_000_000))
EVAL_EVERY = int(os.environ.get("RL96_EVAL", 2_000_000))
GAMMA = 0.999
N_ACT = 3                      # 0 유지 · 1 지정가 매수 · 2 시장가 매도
ACT4 = np.array([0, 2, 3])     # 93번 행동 공간으로의 사상(1=시장가매수 제외)
KEEP4 = [0, 2, 3]
EXTRA = R93.EXTRA
SEEDS = (196, 197, 198)
FEATS = list(json.loads((ROOT / "scripts" / "rl_93" / "features.json").read_text())) + list(M96.FEATURES)


class Q(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_ACT))

    def forward(self, x):
        return self.net(x)


class Env(R93.Env):
    """93번 환경을 그대로 쓰고 **행동만 3개로 좁힌다** — 체결·보상 코드가 같아야 비교가 성립한다."""

    def mask(self):
        return super().mask()[:, KEEP4]

    def step(self, act):
        return super().step(ACT4[np.asarray(act)])


def load_days(days, *, forward_ok=False):
    """93번 로더 + 분 단위 피처를 (code,row) 로 붙인다. 행 순서·필터는 93 과 동일하게 유지한다."""
    parts = []
    for d in days:
        if not forward_ok:
            assert d < R85.FORWARD_START
        rows = pickle.loads((R85.C84 / f"{d}.pkl").read_bytes())
        m = pickle.loads((M96.CACHE / f"{d}.pkl").read_bytes())
        merged = rows.merge(m.drop(columns=["day"]), on=["code", "row"], how="left", validate="one_to_one")
        assert len(merged) == len(rows), "분 단위 피처가 결정 행과 1:1 이 아니다"
        merged[list(M96.FEATURES)] = merged[list(M96.FEATURES)].fillna(0.0)
        parts += _one_day(d, merged)
    return parts


def _one_day(day, merged):
    paths = pickle.loads((R85.C83 / f"{day}.pkl").read_bytes())["paths"]
    fills = pickle.loads((FILLS / f"{day}.pkl").read_bytes())
    out = []
    for code, g in merged.groupby("code"):
        if code not in fills:
            continue
        g = g.sort_values("row")
        frows, fsec, fP, _tick = fills[code]
        if not np.array_equal(frows, g.row.to_numpy(np.int64)):
            continue
        p = paths[code]
        n = len(p["bid"])
        nx = np.minimum(frows + 1, n - 1)
        buy, sell = p["ask"][nx].astype(np.float64), p["bid"][nx].astype(np.float64)
        ok = (buy > 0) & (sell > 0) & np.isfinite(buy) & np.isfinite(sell)
        if ok.sum() < 30:
            continue
        out.append(((day, code), g[FEATS].to_numpy(np.float32)[ok], buy[ok], sell[ok],
                    p["bid"][frows[ok]].astype(np.float64), p["tick"][nx[ok]].astype(np.float64), frows[ok],
                    fsec[ok], fP[ok]))
    return out


def greedy(q, bk) -> pd.DataFrame:
    n = len(bk.keys)
    env = Env(bk, bk.start.copy(), bk.end.copy())
    alive = np.ones(n, bool)
    rows = []
    with torch.no_grad():
        while alive.any():
            a_idx = np.nonzero(alive)[0]
            sub = Env(bk, env.idx[a_idx], env.end[a_idx])
            for name in R93.STATE:
                setattr(sub, name, getattr(env, name)[a_idx].copy())
            qv = q(torch.from_numpy(sub.obs())).numpy()
            qv[~sub.mask()] = -np.inf
            _, done, closed = sub.step(qv.argmax(1))
            for k, ei, xi, P in closed:
                day, code = bk.keys[a_idx[k]]
                rows.append((day, code, int(bk.sec[ei]), int(bk.sec[xi] - bk.sec[ei]),
                             float(bk.sell[xi] / P - 1 - R85.COST)))
            env.idx[a_idx] = sub.idx
            for name in R93.STATE:
                getattr(env, name)[a_idx] = getattr(sub, name)
            alive[a_idx[done]] = False
    return pd.DataFrame(rows, columns=["day", "code", "sec", "hold", "ret"])


def train(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(5)
    rng = np.random.default_rng(seed)
    out = OUT / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name[:10] for p in R85.C84.glob("*.pkl"))
    tr = load_days([d for d in days if d <= R85.TRAIN_END])
    va = load_days([d for d in days if R85.VAL[0] <= d <= R85.VAL[1]])
    sc = R85.Scaler(np.concatenate([p[1] for p in tr]))
    btr, bva = R93.make_book(tr, sc), R93.make_book(va, sc)
    d = len(FEATS) + EXTRA
    q, tgt = Q(d), Q(d)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    B, NE = R85.BUF, R85.N_ENV
    bo, bn = np.zeros((B, d), np.float32), np.zeros((B, d), np.float32)
    ba, br, bd = np.zeros(B, np.int64), np.zeros(B, np.float32), np.zeros(B, np.float32)
    bm = np.zeros((B, N_ACT), bool)
    ptr, full = 0, False

    def reset(k):
        ep = rng.integers(0, len(btr.keys), len(k))
        span = btr.end[ep] - btr.start[ep] - 1
        return btr.start[ep] + (rng.random(len(k)) * span).astype(np.int64), btr.end[ep]

    idx, end = reset(np.arange(NE))
    env = Env(btr, idx, end)
    steps, updates, log, t0, nxt = 0, 0, [], time.time(), EVAL_EVERY
    loss = torch.tensor(0.0)
    while steps < TOTAL:
        o, m = env.obs(), env.mask()
        eps = max(0.05, 1 - 0.95 * steps / R85.EPS_END)
        with torch.no_grad():
            qv = q(torch.from_numpy(o)).numpy()
        qv[~m] = -np.inf
        act = qv.argmax(1)
        rnd = rng.random(NE) < eps
        if rnd.any():
            k = np.nonzero(rnd)[0]
            pick_a = rng.integers(0, N_ACT, len(k))
            act[k] = np.where(m[k, pick_a], pick_a, 0)
        r, done, _ = env.step(act)
        if done.any():
            k = np.nonzero(done)[0]
            env.idx[k], env.end[k] = reset(k)
            env.pos[k] = 0
            env.pend[k] = 0
        o2 = env.obs()
        sl = (np.arange(NE) + ptr) % B
        bo[sl], bn[sl], ba[sl], br[sl], bd[sl], bm[sl] = o, o2, act, r, done, env.mask()
        ptr = (ptr + NE) % B
        full = full or ptr < NE
        steps += NE
        hi = B if full else ptr
        if hi >= R85.BATCH * 4:
            j = rng.integers(0, hi, R85.BATCH)
            to, tn = torch.from_numpy(bo[j]), torch.from_numpy(bn[j])
            with torch.no_grad():
                qn = q(tn)
                qn[~torch.from_numpy(bm[j])] = -1e9
                y = torch.from_numpy(br[j]) + GAMMA * (1 - torch.from_numpy(bd[j])) * \
                    tgt(tn).gather(1, qn.argmax(1, keepdim=True)).squeeze(1)
            qs = q(to).gather(1, torch.from_numpy(ba[j])[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(qs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            if updates % R85.SYNC == 0:
                tgt.load_state_dict(q.state_dict())
        if steps >= nxt:
            nxt += EVAL_EVERY
            t = greedy(q, bva)
            dm = float(t.groupby("day").ret.mean().mean()) if len(t) else float("nan")
            rec = {"seed": seed, "steps": steps, "val_n": len(t), "val_daymean_bp": dm * 1e4,
                   "val_mean_bp": float(t.ret.mean()) * 1e4 if len(t) else None,
                   "val_hold_med": float(t.hold.median()) if len(t) else None,
                   "loss": float(loss), "min": round((time.time() - t0) / 60, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save(q.state_dict(), out / f"ckpt_{steps // 1_000_000:02d}M.pt")
            (out / "val_log.json").write_text(json.dumps(log, indent=1))
    pickle.dump({"scaler": sc, "features": FEATS}, open(out / "prep.pkl", "wb"))


def pick() -> None:
    best = None
    for seed in SEEDS:
        f = OUT / f"seed{seed}" / "val_log.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text()):
            if r["val_n"] >= 30 and r["val_daymean_bp"] == r["val_daymean_bp"] and \
                    (best is None or r["val_daymean_bp"] > best["val_daymean_bp"]):
                best = r
    (OUT / "selected.json").write_text(json.dumps(best, indent=1))
    print("selected", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--pick", action="store_true")
    a = ap.parse_args()
    from krx_quant_core.runtime import DataSpec, start_run

    with start_run("rl96", {"prereg": "docs/research/manju/96", "seed": a.seed, "n_feats": len(FEATS),
                            "gamma": GAMMA, "n_act": N_ACT},
                   repo_root=ROOT, data=DataSpec("2026-08-24", R85.VAL[1], "train+val"), seed=a.seed,
                   allow_dirty=True) as run:
        pick() if a.pick else train(a.seed)
        sel = OUT / "selected.json"
        if a.pick and sel.exists():
            run.log_result({"selected": json.loads(sel.read_text())})
