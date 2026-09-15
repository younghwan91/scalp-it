"""87번 RL 지정가 파도타기 — 85번 환경에 대기열 지정가 진입. 사전등록 docs/research/manju/87 §2~§3.

    ~/of80/.rlvenv/bin/python scripts/rl_87/dqn.py --seed 187
    ~/of80/.rlvenv/bin/python scripts/rl_87/dqn.py --pick
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import importlib.util  # noqa: E402

# 85번 모듈 이름도 dqn 이라 파일 경로로 따로 불러온다(자기 자신을 import 하지 않게).
_spec = importlib.util.spec_from_file_location("rl85_dqn", Path(__file__).resolve().parents[1] / "rl_85" / "dqn.py")
R85 = importlib.util.module_from_spec(_spec)
sys.modules["rl85_dqn"] = R85          # dataclass 가 모듈을 sys.modules 에서 찾는다
_spec.loader.exec_module(R85)

HOME = Path.home() / "of80"
FILLS = HOME / "fills87"
OUT = Path(os.environ.get("RL87_OUT", HOME / "rl87"))
TOTAL = int(os.environ.get("RL87_TOTAL", 40_000_000))
EVAL_EVERY = int(os.environ.get("RL87_EVAL", 2_000_000))
EXTRA = 5
SEEDS = (187, 188, 189)


@dataclass
class Book(R85.Book):
    fill_sec: np.ndarray = None     # 지정가 체결 초(-1 미체결)
    P: np.ndarray = None            # 지정가
    ptick: np.ndarray = None


def load_days(days, feats, *, forward_ok=False):
    parts = []
    for d in days:
        if not forward_ok:
            assert d < R85.FORWARD_START
        rows = pickle.loads((R85.C84 / f"{d}.pkl").read_bytes())
        paths = pickle.loads((R85.C83 / f"{d}.pkl").read_bytes())["paths"]
        fills = pickle.loads((FILLS / f"{d}.pkl").read_bytes())
        for code, g in rows.groupby("code"):
            if code not in fills:
                continue
            g = g.sort_values("row")
            frows, fsec, fP, ftick = fills[code]
            assert np.array_equal(frows, g.row.to_numpy(np.int64))
            p = paths[code]
            r = frows
            n = len(p["bid"])
            nx = np.minimum(r + 1, n - 1)
            buy, sell = p["ask"][nx].astype(np.float64), p["bid"][nx].astype(np.float64)
            ok = (buy > 0) & (sell > 0) & np.isfinite(buy) & np.isfinite(sell)
            if ok.sum() < 30:
                continue
            parts.append(((d, code), g[feats].to_numpy(np.float32)[ok], buy[ok], sell[ok],
                          p["bid"][r[ok]].astype(np.float64), p["tick"][nx[ok]].astype(np.float64), r[ok],
                          fsec[ok], fP[ok], ftick[ok]))
    return parts


def make_book(parts, sc) -> Book:
    base = R85.make_book([p[:7] for p in parts], sc)
    return Book(**base.__dict__, fill_sec=np.concatenate([p[7] for p in parts]),
                P=np.concatenate([p[8] for p in parts]), ptick=np.concatenate([p[9] for p in parts]))


class Env(R85.Env):
    def __init__(self, bk: Book, idx, end) -> None:
        super().__init__(bk, idx, end)
        n = len(idx)
        self.pend = np.zeros(n, np.int64)
        self.pend_sec = np.zeros(n, np.int64)
        self.pend_P = np.zeros(n)
        self.pend_tick = np.ones(n)

    def obs(self):
        return np.concatenate([super().obs(), self.pend[:, None].astype(np.float32)], 1)

    def mask(self):
        m = super().mask()
        m[self.pend == 1] = [True, False, False]
        return m

    def step(self, act):
        b, i = self.bk, self.idx
        act = np.where(self.pend == 1, 0, act)
        r = np.zeros(len(i))
        closed = []
        # 주문 중 → 체결 초가 지난 첫 행에서 보유 시작
        act_on = (self.pend == 1) & (b.sec[i] >= self.pend_sec)
        if act_on.any():
            k = np.nonzero(act_on)[0]
            self.pos[k] = 1
            self.entry[k] = self.pend_P[k]
            self.entry_bid[k] = self.pend_P[k]
            self.entry_tick[k] = self.pend_tick[k]
            self.entry_sec[k] = self.pend_sec[k]
            self.entry_i[k] = i[k]
            self.mark[k] = b.sell[i[k]]
            r[k] = (-R85.COST + (b.sell[i[k]] / self.pend_P[k] - 1)) * 100
            self.pend[k] = 0
        just = act_on.copy()
        hold = (self.pos == 1) & ~just
        stop = hold & (b.bidnow[i] <= self.entry_bid - R85.STOP_TICKS * self.entry_tick)
        a2 = np.where(stop, 2, act)
        sell = (a2 == 2) & hold
        keep = hold & ~sell
        r[keep] += (b.sell[i[keep]] / self.mark[keep] - 1) * 100
        self.mark[keep] = b.sell[i[keep]]
        r[sell] += (b.sell[i[sell]] / self.mark[sell] - 1) * 100
        for k in np.nonzero(sell)[0]:
            closed.append((k, self.entry_i[k], i[k]))
        self.pos[sell] = 0
        # 지정가 매수
        order = (a2 == 1) & (self.pos == 0) & (self.pend == 0) & ~just
        filled = order & (b.fill_sec[i] >= 0)
        self.pend[filled] = 1
        self.pend_sec[filled] = b.fill_sec[i[filled]]
        self.pend_P[filled] = b.P[i[filled]]
        self.pend_tick[filled] = b.ptick[i[filled]]
        self.idx = i + 1
        done = self.idx >= self.end
        last = self.idx - 1
        force = done & (self.pos == 1)
        r[force] += (b.sell[last[force]] / self.mark[force] - 1) * 100
        for k in np.nonzero(force)[0]:
            closed.append((k, self.entry_i[k], last[k]))
        self.pos[force] = 0
        self.pend[done] = 0
        return r, done, closed


STATE = ("pos", "entry", "entry_bid", "entry_tick", "entry_sec", "entry_i", "mark", "pend", "pend_sec",
         "pend_P", "pend_tick")


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
            entry_before = sub.entry.copy()
            qv = q(torch.from_numpy(sub.obs())).numpy()
            qv[~sub.mask()] = -np.inf
            _, done, closed = sub.step(qv.argmax(1))
            for k, ei, xi in closed:
                day, code = bk.keys[a_idx[k]]
                P = entry_before[k] if entry_before[k] > 0 else sub.entry[k]
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
    feats = R85.features_list()
    days = sorted(p.name[:10] for p in R85.C84.glob("*.pkl"))
    tr = load_days([d for d in days if d <= R85.TRAIN_END], feats)
    va = load_days([d for d in days if R85.VAL[0] <= d <= R85.VAL[1]], feats)
    sc = R85.Scaler(np.concatenate([p[1] for p in tr]))
    btr, bva = make_book(tr, sc), make_book(va, sc)
    d = len(feats) + EXTRA
    q, tgt = R85.Q(d), R85.Q(d)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    B = R85.BUF
    bo, bn = np.zeros((B, d), np.float32), np.zeros((B, d), np.float32)
    ba, br, bd = np.zeros(B, np.int64), np.zeros(B, np.float32), np.zeros(B, np.float32)
    bm = np.zeros((B, 3), bool)
    ptr, full = 0, False
    NE = R85.N_ENV

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
        rnd = (rng.random(NE) < eps) & (env.pend == 0)
        act[rnd] = np.where(env.pos[rnd] == 0, rng.integers(0, 2, rnd.sum()), rng.integers(0, 2, rnd.sum()) * 2)
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
    pickle.dump({"scaler": sc, "features": feats}, open(out / "prep.pkl", "wb"))


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

    cfg = {"mode": "pick" if a.pick else "train", "seed": a.seed, "TOTAL": TOTAL, "prereg": "docs/research/manju/87"}
    with start_run("rl87", cfg, repo_root=Path(__file__).resolve().parents[2],
                   data=DataSpec("2026-08-24", R85.VAL[1], "train+val"), seed=a.seed, allow_dirty=True) as run:
        pick() if a.pick else train(a.seed)
        sel = OUT / "selected.json"
        if a.pick and sel.exists():
            run.log_result({"selected": json.loads(sel.read_text())})
