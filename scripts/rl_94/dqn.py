"""94번 RL 전권 2단계 — 93번과 같은 환경에, **종목·가격·시간 필터를 뺀 캐시**(flow94)를 쓴다.
사전등록 docs/research/manju/94 §1~§2.

    ~/of80/.rlvenv/bin/python scripts/rl_94/dqn.py --seed 194
    ~/of80/.rlvenv/bin/python scripts/rl_94/dqn.py --pick
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
_spec = importlib.util.spec_from_file_location("rl85_dqn", ROOT / "scripts" / "rl_85" / "dqn.py")
R85 = importlib.util.module_from_spec(_spec)
sys.modules["rl85_dqn"] = R85
_spec.loader.exec_module(R85)

HOME = Path.home() / "of80"
C94 = HOME / "flow94"
FILLS = HOME / "fills94"
OUT = Path(os.environ.get("RL94_OUT", HOME / "rl94"))
TOTAL = int(os.environ.get("RL94_TOTAL", 40_000_000))
EVAL_EVERY = int(os.environ.get("RL94_EVAL", 2_000_000))
FEATS = json.loads((ROOT / "scripts" / "rl_93" / "features.json").read_text())   # 93번과 같은 37피처
EXTRA = 4                      # 보유 · 미실현% · 보유초/60 · 주문중
N_ACT = 4                      # 0 유지 · 1 시장가매수 · 2 지정가매수 · 3 매도
SEEDS = (194, 195, 196)


class Q(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_ACT))

    def forward(self, x):
        return self.net(x)


class Book(R85.Book):
    pass


def load_days(days, feats=FEATS, *, forward_ok=False):
    parts = []
    for d in days:
        if not forward_ok:
            assert d < R85.FORWARD_START
        obj = pickle.loads((C94 / f"{d}.pkl").read_bytes())
        rows, paths = obj["rows"], obj["paths"]      # 94번 캐시가 자기 경로를 들고 있다
        fills = pickle.loads((FILLS / f"{d}.pkl").read_bytes())
        for code, g in rows.groupby("code"):
            if code not in fills:
                continue
            g = g.sort_values("row")
            frows, fsec, fP, _ftick = fills[code]
            if not np.array_equal(frows, g.row.to_numpy(np.int64)):
                continue
            p = paths[code]
            n = len(p["bid"])
            nx = np.minimum(frows + 1, n - 1)
            buy, sell = p["ask"][nx].astype(np.float64), p["bid"][nx].astype(np.float64)
            ok = (buy > 0) & (sell > 0) & np.isfinite(buy) & np.isfinite(sell)
            if ok.sum() < 30:
                continue
            parts.append(((d, code), g[feats].to_numpy(np.float32)[ok], buy[ok], sell[ok],
                          p["bid"][frows[ok]].astype(np.float64), p["tick"][nx[ok]].astype(np.float64), frows[ok],
                          fsec[ok], fP[ok]))
    return parts


def make_book(parts, sc) -> Book:
    base = R85.make_book([p[:7] for p in parts], sc)
    bk = Book(**base.__dict__)
    bk.fill_sec = np.concatenate([p[7] for p in parts])
    bk.P = np.concatenate([p[8] for p in parts])
    return bk


class Env(R85.Env):
    def __init__(self, bk, idx, end) -> None:
        super().__init__(bk, idx, end)
        n = len(idx)
        self.pend = np.zeros(n, np.int64)
        self.pend_sec = np.zeros(n, np.int64)
        self.pend_P = np.zeros(n)

    def obs(self):
        b, i = self.bk, self.idx
        unreal = np.where(self.pos == 1, (b.sell[i] / np.where(self.entry > 0, self.entry, 1) - 1) * 100, 0.0)
        hold = np.where(self.pos == 1, np.clip((b.sec[i] - self.entry_sec) / 60.0, 0, 10), 0.0)
        return np.concatenate([b.X[i], np.stack([self.pos, unreal, hold, self.pend], 1)], 1).astype(np.float32)

    def mask(self):
        n = len(self.idx)
        m = np.ones((n, N_ACT), bool)
        busy = (self.pos == 1) | (self.pend == 1)
        m[busy, 1] = False
        m[busy, 2] = False
        m[self.pos == 0, 3] = False
        m[self.pend == 1] = [True, False, False, False]
        return m

    def step(self, act):
        b, i = self.bk, self.idx
        r = np.zeros(len(i))
        closed = []
        act = np.where(self.pend == 1, 0, act)
        # 지정가 체결 → 보유 시작
        on = (self.pend == 1) & (b.sec[i] >= self.pend_sec)
        if on.any():
            k = np.nonzero(on)[0]
            self.pos[k] = 1
            self.entry[k] = self.pend_P[k]
            self.entry_sec[k] = self.pend_sec[k]
            self.entry_i[k] = i[k]
            self.mark[k] = b.sell[i[k]]
            r[k] = (-R85.COST + (b.sell[i[k]] / self.pend_P[k] - 1)) * 100
            self.pend[k] = 0
        just = on.copy()
        hold = (self.pos == 1) & ~just
        sell = (act == 3) & hold
        keep = hold & ~sell
        r[keep] += (b.sell[i[keep]] / self.mark[keep] - 1) * 100
        self.mark[keep] = b.sell[i[keep]]
        r[sell] += (b.sell[i[sell]] / self.mark[sell] - 1) * 100
        for k in np.nonzero(sell)[0]:
            closed.append((k, self.entry_i[k], i[k], self.entry[k]))
        self.pos[sell] = 0
        free = (self.pos == 0) & (self.pend == 0) & ~just
        mk = (act == 1) & free                                  # 시장가 매수
        self.entry[mk] = b.buy[i[mk]]
        self.entry_sec[mk] = b.sec[i[mk]]
        self.entry_i[mk] = i[mk]
        self.mark[mk] = b.sell[i[mk]]
        r[mk] = (-R85.COST + (b.sell[i[mk]] / b.buy[i[mk]] - 1)) * 100
        self.pos[mk] = 1
        lm = (act == 2) & free & (b.fill_sec[i] >= 0)            # 지정가 매수(체결되는 경우만 대기)
        self.pend[lm] = 1
        self.pend_sec[lm] = b.fill_sec[i[lm]]
        self.pend_P[lm] = b.P[i[lm]]
        self.idx = i + 1
        done = self.idx >= self.end
        last = self.idx - 1
        force = done & (self.pos == 1)
        r[force] += (b.sell[last[force]] / self.mark[force] - 1) * 100
        for k in np.nonzero(force)[0]:
            closed.append((k, self.entry_i[k], last[k], self.entry[k]))
        self.pos[force] = 0
        self.pend[done] = 0
        return r, done, closed


STATE = ("pos", "entry", "entry_bid", "entry_tick", "entry_sec", "entry_i", "mark", "pend", "pend_sec", "pend_P")


def greedy(q, bk: Book) -> pd.DataFrame:
    n = len(bk.keys)
    env = Env(bk, bk.start.copy(), bk.end.copy())
    alive = np.ones(n, bool)
    rows = []
    with torch.no_grad():
        while alive.any():
            a_idx = np.nonzero(alive)[0]
            sub = Env(bk, env.idx[a_idx], env.end[a_idx])
            for name in STATE:
                setattr(sub, name, getattr(env, name)[a_idx].copy())
            qv = q(torch.from_numpy(sub.obs())).numpy()
            qv[~sub.mask()] = -np.inf
            _, done, closed = sub.step(qv.argmax(1))
            for k, ei, xi, P in closed:
                day, code = bk.keys[a_idx[k]]
                rows.append((day, code, int(bk.sec[ei]), int(bk.sec[xi] - bk.sec[ei]),
                             float(bk.sell[xi] / P - 1 - R85.COST)))
            env.idx[a_idx] = sub.idx
            for name in STATE:
                getattr(env, name)[a_idx] = getattr(sub, name)
            alive[a_idx[done]] = False
    return pd.DataFrame(rows, columns=["day", "code", "sec", "hold", "ret"])


def train(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(5)
    rng = np.random.default_rng(seed)
    out = OUT / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name[:10] for p in C94.glob("*.pkl"))
    tr = load_days([d for d in days if d <= R85.TRAIN_END])
    va = load_days([d for d in days if R85.VAL[0] <= d <= R85.VAL[1]])
    sc = R85.Scaler(np.concatenate([p[1] for p in tr]))
    btr, bva = make_book(tr, sc), make_book(va, sc)
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
            pick = rng.integers(0, N_ACT, len(k))
            act[k] = np.where(m[k, pick], pick, 0)
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
                y = torch.from_numpy(br[j]) + R85.GAMMA * (1 - torch.from_numpy(bd[j])) * \
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

    with start_run("rl94", {"prereg": "docs/research/manju/94", "seed": a.seed, "n_feats": len(FEATS)},
                   repo_root=ROOT, data=DataSpec("2026-08-24", R85.VAL[1], "train+val"), seed=a.seed,
                   allow_dirty=True) as run:
        pick() if a.pick else train(a.seed)
        sel = OUT / "selected.json"
        if a.pick and sel.exists():
            run.log_result({"selected": json.loads(sel.read_text())})
