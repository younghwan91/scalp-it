"""92번 RL 빠른 청산 — 호랑이 신호에서만 시장가 진입, 강제 손절 없음, 보유 10초 상한.
사전등록 docs/research/manju/92 §2~§3. 85번 환경을 상속하고 (1) 진입 가능 행 마스크 (2) 손절 제거 (3) 10초 상한만 바꾼다.

    ~/of80/.rlvenv/bin/python scripts/rl_92/dqn.py --seed 192
    ~/of80/.rlvenv/bin/python scripts/rl_92/dqn.py --pick
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

sys.path.insert(0, str(ROOT / "scripts" / "tiger_86"))
sys.path.insert(0, str(ROOT / "scripts" / "flow_83"))
import engine as TIGER  # noqa: E402

HOME = Path.home() / "of80"
OUT = Path(os.environ.get("RL92_OUT", HOME / "rl92"))
TOTAL = int(os.environ.get("RL92_TOTAL", 40_000_000))
EVAL_EVERY = int(os.environ.get("RL92_EVAL", 2_000_000))
MAX_HOLD = 10                      # 사용자 지시: 10초 이상 버티지 않는다
EXTRA = 3                          # 보유 여부 · 미실현 % · 보유 초/10
SEEDS = (192, 193, 194)


def signal_rows(days: list[str], *, forward_ok: bool = False) -> dict[tuple[str, str], set[int]]:
    """86번 §2 호랑이 신호가 선 (날짜, 종목) → 행 집합. 문턱은 전일 분위를 이어서 쓴다."""
    allday = sorted(p.name[:10] for p in (HOME / "flow83").glob("*.pkl"))
    prev = None
    out: dict[tuple[str, str], set[int]] = {}
    for d in allday:
        if d > max(days):
            break
        rows = pickle.loads((HOME / "flow83" / f"{d}.pkl").read_bytes())["rows"]
        thr = TIGER.day_thresholds(rows, prev)
        sig = TIGER.signals(rows, thr)
        if d in days:
            for code, g in rows[sig].groupby("code"):
                out[(d, str(code))] = set(g.row.to_numpy().tolist())
        v = rows.dstr30.to_numpy(np.float64)
        v = v[np.isfinite(v)]
        prev = float(np.quantile(v, 0.99)) if len(v) else prev
    return out


class Book(R85.Book):
    pass


def load_days(days, feats, *, forward_ok=False):
    sig = signal_rows(days, forward_ok=forward_ok)
    parts = []
    for d in days:
        if not forward_ok:
            assert d < R85.FORWARD_START
        rows = pickle.loads((R85.C84 / f"{d}.pkl").read_bytes())
        paths = pickle.loads((R85.C83 / f"{d}.pkl").read_bytes())["paths"]
        for code, g in rows.groupby("code"):
            s = sig.get((d, str(code)))
            if not s:
                continue
            g = g.sort_values("row")
            p = paths[code]
            r = g.row.to_numpy(np.int64)
            n = len(p["bid"])
            nx = np.minimum(r + 1, n - 1)
            buy, sell = p["ask"][nx].astype(np.float64), p["bid"][nx].astype(np.float64)
            ok = (buy > 0) & (sell > 0) & np.isfinite(buy) & np.isfinite(sell)
            if ok.sum() < 30:
                continue
            can = np.array([x in s for x in r], bool)[ok]
            if not can.any():
                continue
            parts.append(((d, code), g[feats].to_numpy(np.float32)[ok], buy[ok], sell[ok],
                          p["bid"][r[ok]].astype(np.float64), p["tick"][nx[ok]].astype(np.float64), r[ok], can))
    return parts


def make_book(parts, sc) -> Book:
    base = R85.make_book([p[:7] for p in parts], sc)
    bk = Book(**base.__dict__)
    bk.can_enter = np.concatenate([p[7] for p in parts])
    return bk


class Env(R85.Env):
    def obs(self):
        b, i = self.bk, self.idx
        unreal = np.where(self.pos == 1, (b.sell[i] / np.where(self.entry > 0, self.entry, 1) - 1) * 100, 0.0)
        hold = np.where(self.pos == 1, np.clip((b.sec[i] - self.entry_sec) / MAX_HOLD, 0, 2), 0.0)
        return np.concatenate([b.X[i], np.stack([self.pos, unreal, hold], 1)], 1).astype(np.float32)

    def mask(self):
        m = super().mask()
        m[(self.pos == 0) & ~self.bk.can_enter[self.idx], 1] = False   # 신호 행에서만 매수
        return m

    def step(self, act):
        b, i = self.bk, self.idx
        r = np.zeros(len(i))
        closed = []
        hold = self.pos == 1
        cap = hold & (b.sec[i] - self.entry_sec >= MAX_HOLD)            # 10초 상한 강제 매도
        act = np.where(cap, 2, act)
        buy = (act == 1) & (self.pos == 0) & b.can_enter[i]
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
        self.entry_sec[buy] = b.sec[i[buy]]
        self.entry_i[buy] = i[buy]
        self.mark[buy] = b.sell[i[buy]]
        r[buy] = (-R85.COST + (b.sell[i[buy]] / b.buy[i[buy]] - 1)) * 100
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


STATE = ("pos", "entry", "entry_bid", "entry_tick", "entry_sec", "entry_i", "mark")


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
            for k, ei, xi in closed:
                day, code = bk.keys[a_idx[k]]
                rows.append((day, code, int(bk.sec[ei]), int(bk.sec[xi] - bk.sec[ei]),
                             float(bk.sell[xi] / bk.buy[ei] - 1 - R85.COST)))
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
    B, NE = R85.BUF, R85.N_ENV
    bo, bn = np.zeros((B, d), np.float32), np.zeros((B, d), np.float32)
    ba, br, bd = np.zeros(B, np.int64), np.zeros(B, np.float32), np.zeros(B, np.float32)
    bm = np.zeros((B, 3), bool)
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
            choice = np.where(env.pos[k] == 0, rng.integers(0, 2, len(k)), rng.integers(0, 2, len(k)) * 2)
            act[k] = np.where(m[k, choice], choice, 0)
        r, done, _ = env.step(act)
        if done.any():
            k = np.nonzero(done)[0]
            env.idx[k], env.end[k] = reset(k)
            env.pos[k] = 0
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

    with start_run("rl92", {"prereg": "docs/research/manju/92", "seed": a.seed, "MAX_HOLD": MAX_HOLD},
                   repo_root=ROOT, data=DataSpec("2026-08-24", R85.VAL[1], "train+val"), seed=a.seed,
                   allow_dirty=True) as run:
        pick() if a.pick else train(a.seed)
        sel = OUT / "selected.json"
        if a.pick and sel.exists():
            run.log_result({"selected": json.loads(sel.read_text())})
