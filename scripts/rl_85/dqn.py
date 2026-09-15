"""85번 RL 파도타기 — 평가손익 보상 · 하드 손절 · 84번 피처. 사전등록 §2~§5.

    ~/of80/.rlvenv/bin/python scripts/rl_85/dqn.py --seed 85      # 학습 + VAL 체크포인트
    ~/of80/.rlvenv/bin/python scripts/rl_85/dqn.py --pick           # 시드 3개 중 VAL 최고 하나 고정
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HOME = Path.home() / "of80"
C84 = HOME / "flow84"
C83 = HOME / "flow83"
OUT = Path(os.environ.get("RL85_OUT", HOME / "rl85"))
TRAIN_END = "2026-09-10"
VAL = ("2026-09-11", "2026-09-15")
FORWARD_START = "2026-09-16"
COST = 0.0023
GAMMA = 0.99
N_ENV = 512
BATCH = 8192
BUF = 4_000_000
SYNC = 2_000
EPS_END = 4_000_000
TOTAL = int(os.environ.get("RL85_TOTAL", 40_000_000))       # 스모크 테스트만 줄인다
EVAL_EVERY = int(os.environ.get("RL85_EVAL", 2_000_000))
STOP_TICKS = 2
EXTRA = 4                       # pos · unreal% · hold/60 · 손절선까지 호가 수


@dataclass
class Book:
    keys: list
    X: np.ndarray
    buy: np.ndarray             # 다음 초 매도1
    sell: np.ndarray            # 다음 초 매수1
    bidnow: np.ndarray          # 결정 초 매수1
    tick: np.ndarray
    sec: np.ndarray
    start: np.ndarray
    end: np.ndarray


def features_list() -> list[str]:
    if os.environ.get("RL85_FEATS"):                     # 스모크 테스트 전용
        return os.environ["RL85_FEATS"].split(",")
    return pickle.loads((HOME / "flow84_model" / "model.pkl").read_bytes())["features"]


def load_days(days: list[str], feats: list[str], *, forward_ok: bool = False):
    parts = []
    for d in days:
        if not forward_ok:
            assert d < FORWARD_START, "학습이 전진 날짜를 읽으려 했다"
        rows = pickle.loads((C84 / f"{d}.pkl").read_bytes())
        paths = pickle.loads((C83 / f"{d}.pkl").read_bytes())["paths"]
        for code, g in rows.groupby("code"):
            p = paths[code]
            r = g.row.to_numpy(np.int64)
            n = len(p["bid"])
            nx = np.minimum(r + 1, n - 1)
            buy, sell = p["ask"][nx].astype(np.float64), p["bid"][nx].astype(np.float64)
            ok = (buy > 0) & (sell > 0) & np.isfinite(buy) & np.isfinite(sell)
            if ok.sum() < 30:
                continue
            parts.append(((d, code), g[feats].to_numpy(np.float32)[ok], buy[ok], sell[ok],
                          p["bid"][r[ok]].astype(np.float64), p["tick"][nx[ok]].astype(np.float64), r[ok]))
    return parts


class Scaler:
    def __init__(self, X: np.ndarray) -> None:
        Xf = X[np.isfinite(X).all(axis=1)]
        self.lo, self.hi = np.nanpercentile(Xf, 1, axis=0), np.nanpercentile(Xf, 99, axis=0)
        Xc = np.clip(Xf, self.lo, self.hi)
        self.mu, self.sd = Xc.mean(axis=0), Xc.std(axis=0) + 1e-8

    def __call__(self, X):
        return np.nan_to_num((np.clip(X, self.lo, self.hi) - self.mu) / self.sd).astype(np.float32)


def make_book(parts, sc: Scaler) -> Book:
    lens = np.array([len(p[2]) for p in parts])
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return Book([p[0] for p in parts], sc(np.concatenate([p[1] for p in parts])),
                np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]),
                np.concatenate([p[4] for p in parts]), np.concatenate([p[5] for p in parts]),
                np.concatenate([p[6] for p in parts]), start, start + lens)


class Q(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, x):
        return self.net(x)


class Env:
    """벡터 환경. 상태 배열은 전부 길이 n 의 numpy."""

    def __init__(self, bk: Book, idx, end) -> None:
        self.bk, self.idx, self.end = bk, idx, end
        n = len(idx)
        self.pos = np.zeros(n, np.int64)
        self.entry = np.zeros(n)
        self.entry_bid = np.zeros(n)
        self.entry_tick = np.ones(n)
        self.entry_sec = np.zeros(n, np.int64)
        self.entry_i = np.zeros(n, np.int64)
        self.mark = np.zeros(n)

    def obs(self) -> np.ndarray:
        b, i = self.bk, self.idx
        unreal = np.where(self.pos == 1, (b.sell[i] / np.where(self.entry > 0, self.entry, 1) - 1) * 100, 0.0)
        hold = np.where(self.pos == 1, np.clip((b.sec[i] - self.entry_sec) / 60.0, 0, 10), 0.0)
        to_stop = np.where(self.pos == 1, (b.bidnow[i] - (self.entry_bid - STOP_TICKS * self.entry_tick))
                           / self.entry_tick, 0.0)
        return np.concatenate([b.X[i], np.stack([self.pos, unreal, hold, np.clip(to_stop, -5, 20)], 1)],
                              1).astype(np.float32)

    def mask(self) -> np.ndarray:
        m = np.ones((len(self.idx), 3), bool)
        m[self.pos == 0, 2] = False
        m[self.pos == 1, 1] = False
        return m

    def step(self, act):
        """→ 보상(%), done, 닫힌 거래 [(k, entry_i, exit_i)]."""
        b, i = self.bk, self.idx
        r = np.zeros(len(i))
        closed = []
        hold = self.pos == 1
        stop = hold & (b.bidnow[i] <= self.entry_bid - STOP_TICKS * self.entry_tick)
        act = np.where(stop, 2, act)
        buy = (act == 1) & (self.pos == 0)
        sell = (act == 2) & hold
        keep = hold & ~sell
        r[keep] = (b.sell[i[keep]] / self.mark[keep] - 1) * 100
        self.mark[keep] = b.sell[i[keep]]
        r[sell] = (b.sell[i[sell]] / self.mark[sell] - 1) * 100
        for k in np.nonzero(sell)[0]:
            closed.append((k, self.entry_i[k], i[k]))
        self.pos[sell] = 0
        self.entry[buy] = b.buy[i[buy]]
        self.entry_bid[buy] = b.sell[i[buy]]
        self.entry_tick[buy] = b.tick[i[buy]]
        self.entry_sec[buy] = b.sec[i[buy]]
        self.entry_i[buy] = i[buy]
        self.mark[buy] = b.sell[i[buy]]
        r[buy] = (-COST + (b.sell[i[buy]] / b.buy[i[buy]] - 1)) * 100
        self.pos[buy] = 1
        self.idx = i + 1
        done = self.idx >= self.end
        last = self.idx - 1
        force = done & (self.pos == 1)
        r[force] += (b.sell[last[force]] / self.mark[force] - 1) * 100
        for k in np.nonzero(force)[0]:
            closed.append((k, self.entry_i[k], last[k]))
        self.pos[force] = 0
        return r, done, closed


def trade_rows(bk: Book, closed, owner) -> list[tuple]:
    out = []
    for k, ei, xi in closed:
        day, code = bk.keys[owner[k]]
        out.append((day, code, int(bk.sec[ei]), int(bk.sec[xi] - bk.sec[ei]),
                    float(bk.sell[xi] / bk.buy[ei] - 1 - COST)))
    return out


def greedy(q: Q, bk: Book) -> pd.DataFrame:
    n = len(bk.keys)
    env = Env(bk, bk.start.copy(), bk.end.copy())
    alive = np.ones(n, bool)
    rows = []
    with torch.no_grad():
        while alive.any():
            a_idx = np.nonzero(alive)[0]
            sub = Env(bk, env.idx[a_idx], env.end[a_idx])
            for name in ("pos", "entry", "entry_bid", "entry_tick", "entry_sec", "entry_i", "mark"):
                setattr(sub, name, getattr(env, name)[a_idx].copy())
            qv = q(torch.from_numpy(sub.obs())).numpy()
            qv[~sub.mask()] = -np.inf
            _, done, closed = sub.step(qv.argmax(1))
            rows += trade_rows(bk, closed, a_idx)
            env.idx[a_idx] = sub.idx
            for name in ("pos", "entry", "entry_bid", "entry_tick", "entry_sec", "entry_i", "mark"):
                getattr(env, name)[a_idx] = getattr(sub, name)
            alive[a_idx[done]] = False
    return pd.DataFrame(rows, columns=["day", "code", "sec", "hold", "ret"])


def train(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(5)
    rng = np.random.default_rng(seed)
    out = OUT / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    feats = features_list()
    days = sorted(p.name[:10] for p in C84.glob("*.pkl"))
    tr_days = [d for d in days if d <= TRAIN_END]
    va_days = [d for d in days if VAL[0] <= d <= VAL[1]]
    tr, va = load_days(tr_days, feats), load_days(va_days, feats)
    sc = Scaler(np.concatenate([p[1] for p in tr]))
    btr, bva = make_book(tr, sc), make_book(va, sc)
    d = len(feats) + EXTRA
    q, tgt = Q(d), Q(d)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    bo, bn = np.zeros((BUF, d), np.float32), np.zeros((BUF, d), np.float32)
    ba, br, bd = np.zeros(BUF, np.int64), np.zeros(BUF, np.float32), np.zeros(BUF, np.float32)
    bm = np.zeros((BUF, 3), bool)
    ptr, full = 0, False

    def reset(k):
        ep = rng.integers(0, len(btr.keys), len(k))
        span = btr.end[ep] - btr.start[ep] - 1
        return btr.start[ep] + (rng.random(len(k)) * span).astype(np.int64), btr.end[ep]

    idx, end = reset(np.arange(N_ENV))
    env = Env(btr, idx, end)
    steps, updates, log, t0, nxt = 0, 0, [], time.time(), EVAL_EVERY
    loss = torch.tensor(0.0)
    while steps < TOTAL:
        o, m = env.obs(), env.mask()
        eps = max(0.05, 1 - 0.95 * steps / EPS_END)
        with torch.no_grad():
            qv = q(torch.from_numpy(o)).numpy()
        qv[~m] = -np.inf
        act = qv.argmax(1)
        rnd = rng.random(N_ENV) < eps
        act[rnd] = np.where(env.pos[rnd] == 0, rng.integers(0, 2, rnd.sum()), rng.integers(0, 2, rnd.sum()) * 2)
        r, done, _ = env.step(act)
        if done.any():
            k = np.nonzero(done)[0]
            env.idx[k], env.end[k] = reset(k)
            env.pos[k] = 0
        o2 = env.obs()
        sl = (np.arange(N_ENV) + ptr) % BUF
        bo[sl], bn[sl], ba[sl], br[sl], bd[sl], bm[sl] = o, o2, act, r, done, env.mask()
        ptr = (ptr + N_ENV) % BUF
        full = full or ptr < N_ENV
        steps += N_ENV
        hi = BUF if full else ptr
        if hi >= BATCH * 4:
            j = rng.integers(0, hi, BATCH)
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
            if updates % SYNC == 0:
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
    for seed in (85, 86, 87):
        for r in json.loads((OUT / f"seed{seed}" / "val_log.json").read_text()):
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

    # 학습·VAL 은 전진 이전 날짜만 — research/runs/rl85 OOS(2026-09-16~) 잠금이 막는다.
    cfg = {"mode": "pick" if a.pick else "train", "seed": a.seed, "COST": COST, "GAMMA": GAMMA,
           "N_ENV": N_ENV, "TRAIN_END": TRAIN_END, "VAL": VAL}
    with start_run("rl85", cfg, repo_root=Path(__file__).resolve().parents[2],
                   data=DataSpec("2026-08-24", VAL[1], "train+val"), seed=a.seed,
                   allow_dirty=True) as run:
        pick() if a.pick else train(a.seed)
        sel = OUT / "selected.json"
        if a.pick and sel.exists():
            run.log_result({"selected": json.loads(sel.read_text())})
