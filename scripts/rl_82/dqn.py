"""82번 모델 B — Double DQN(CPU). 사전등록 §4-B. 벡터화 환경 = 시뮬레이터(§2).

    ~/of80/.rlvenv/bin/python scripts/rl_82/dqn.py      # TRAIN 학습, 1M 스텝마다 VAL 평가·체크포인트
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset import COST, Scaler, load

OUT = Path.home() / "of80" / "rl82"
SEED = 82
N_ENV = 256
GAMMA = 0.999
LR = 1e-4
BATCH = 4096
BUF = 2_000_000
SYNC = 5_000
EPS_END_STEP = 2_000_000
import os
TOTAL = int(os.environ.get("RL82_TOTAL", 20_000_000))  # 스모크 테스트만 줄인다
EVAL_EVERY = int(os.environ.get("RL82_EVAL", 1_000_000))
OBS = 19


class Q(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Book:
    """에피소드들을 한 배열로 이어 붙인다 — 인덱스 하나로 벡터화 스텝."""

    def __init__(self, eps, sc: Scaler) -> None:
        self.eps = eps
        self.X = np.concatenate([sc(e.X) for e in eps])
        self.buy = np.concatenate([e.buy for e in eps])
        self.sell = np.concatenate([e.sell for e in eps])
        self.sec = np.concatenate([e.secs for e in eps]).astype(np.int64)
        lens = np.array([len(e.secs) for e in eps])
        self.start = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.end = self.start + lens          # 배타


def obs(book: Book, idx, pos, entry, entry_sec) -> np.ndarray:
    unreal = np.where(pos == 1, (book.sell[idx] / np.where(entry > 0, entry, 1.0) - 1.0) * 100.0, 0.0)
    hold = np.where(pos == 1, np.clip((book.sec[idx] - entry_sec) / 300.0, 0, 10), 0.0)
    return np.concatenate([book.X[idx], pos[:, None], unreal[:, None], hold[:, None]], axis=1).astype(np.float32)


def masks(pos) -> np.ndarray:
    m = np.ones((len(pos), 3), bool)
    m[pos == 0, 2] = False
    m[pos == 1, 1] = False
    return m


def env_step(book: Book, idx, ep_end, pos, entry, entry_sec, act):
    """행동 적용 → (보상 bp/100, done). 매수·매도는 그 결정 행의 다음 초 호가(§2)."""
    r = np.zeros(len(idx), np.float64)
    buy = (act == 1) & (pos == 0)
    sell = (act == 2) & (pos == 1)
    entry[buy] = book.buy[idx[buy]]
    entry_sec[buy] = book.sec[idx[buy]]
    pos[buy] = 1
    r[sell] = (book.sell[idx[sell]] / entry[sell] - 1.0 - COST) * 100.0
    pos[sell] = 0
    idx += 1
    done = idx >= ep_end
    last = idx - 1
    force = done & (pos == 1)                 # 15:10 강제 매도
    r[force] += (book.sell[last[force]] / entry[force] - 1.0 - COST) * 100.0
    pos[force] = 0
    return r, done, buy, sell, force


def greedy_trades(q: Q, book: Book) -> pd.DataFrame:
    """모든 에피소드를 처음부터 동시에 탐욕 정책으로 돌려 거래 목록을 만든다."""
    n_ep = len(book.eps)
    idx = book.start.copy()
    end = book.end.copy()
    pos = np.zeros(n_ep, np.int64)
    entry = np.zeros(n_ep)
    entry_sec = np.zeros(n_ep, np.int64)
    entry_i = np.zeros(n_ep, np.int64)
    alive = np.ones(n_ep, bool)
    rows = []
    with torch.no_grad():
        while alive.any():
            a_idx = np.nonzero(alive)[0]
            o = obs(book, idx[a_idx], pos[a_idx], entry[a_idx], entry_sec[a_idx])
            qv = q(torch.from_numpy(o)).numpy()
            qv[~masks(pos[a_idx])] = -np.inf
            act = qv.argmax(1)
            sub_idx = idx[a_idx].copy()
            buy = (act == 1)
            entry_i[a_idx[buy]] = sub_idx[buy]
            p, e, es, ix = pos[a_idx], entry[a_idx], entry_sec[a_idx], idx[a_idx]
            _, done, _b, sell, force = env_step(book, ix, end[a_idx], p, e, es, act)
            closed = sell | force
            for k in np.nonzero(closed)[0]:
                gi = a_idx[k]
                ep = book.eps[gi]
                i_out = sub_idx[k] if sell[k] else ix[k] - 1
                i_in = entry_i[gi]
                ret = book.sell[i_out] / book.buy[i_in] - 1.0 - COST
                rows.append((ep.day, ep.code, int(book.sec[i_in]), int(book.sec[i_out] - book.sec[i_in]), float(ret)))
            pos[a_idx], entry[a_idx], entry_sec[a_idx], idx[a_idx] = p, e, es, ix
            alive[a_idx[done]] = False
    return pd.DataFrame(rows, columns=["day", "code", "sec", "hold", "ret"])


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(12)
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = load("train"), load("val")
    sc = Scaler(tr)
    btr, bva = Book(tr, sc), Book(va, sc)
    q, tgt = Q(), Q()
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=LR)

    bo = np.zeros((BUF, OBS), np.float32)
    bn = np.zeros((BUF, OBS), np.float32)
    ba = np.zeros(BUF, np.int64)
    br = np.zeros(BUF, np.float32)
    bd = np.zeros(BUF, np.float32)
    bm = np.zeros((BUF, 3), bool)
    ptr = 0
    full = False

    def reset(k):
        ep = rng.integers(0, len(tr), len(k))
        st = btr.start[ep] + (rng.random(len(k)) * (btr.end[ep] - btr.start[ep] - 1)).astype(np.int64)
        return st, btr.end[ep]

    idx, end = reset(np.arange(N_ENV))
    pos = np.zeros(N_ENV, np.int64)
    entry = np.zeros(N_ENV)
    entry_sec = np.zeros(N_ENV, np.int64)
    steps = 0
    it = 0
    log = []
    t0 = time.time()
    next_eval = EVAL_EVERY
    while steps < TOTAL:
        o = obs(btr, idx, pos, entry, entry_sec)
        m = masks(pos)
        eps = max(0.05, 1.0 - 0.95 * steps / EPS_END_STEP)
        with torch.no_grad():
            qv = q(torch.from_numpy(o)).numpy()
        qv[~m] = -np.inf
        act = qv.argmax(1)
        rnd = rng.random(N_ENV) < eps
        if rnd.any():
            choice = np.where(pos[rnd] == 0, rng.integers(0, 2, rnd.sum()), rng.integers(0, 2, rnd.sum()) * 2)
            act[rnd] = choice
        r, done, *_ = env_step(btr, idx, end, pos, entry, entry_sec, act)
        if done.any():
            k = np.nonzero(done)[0]
            idx[k], end[k] = reset(k)
            pos[k] = 0
        o2 = obs(btr, np.minimum(idx, btr.end.max() - 1), pos, entry, entry_sec)
        sl = (np.arange(N_ENV) + ptr) % BUF
        bo[sl], bn[sl], ba[sl], br[sl], bd[sl], bm[sl] = o, o2, act, r, done, masks(pos)
        ptr = (ptr + N_ENV) % BUF
        full = full or ptr < N_ENV
        steps += N_ENV
        it += 1
        hi = BUF if full else ptr
        if hi >= BATCH * 4:
            j = rng.integers(0, hi, BATCH)
            to, tn = torch.from_numpy(bo[j]), torch.from_numpy(bn[j])
            with torch.no_grad():
                qn = q(tn)
                qn[~torch.from_numpy(bm[j])] = -1e9
                an = qn.argmax(1, keepdim=True)
                y = torch.from_numpy(br[j]) + GAMMA * (1 - torch.from_numpy(bd[j])) * tgt(tn).gather(1, an).squeeze(1)
            qs = q(to).gather(1, torch.from_numpy(ba[j])[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(qs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % max(1, SYNC // 1) == 0:
                tgt.load_state_dict(q.state_dict())
        if steps >= next_eval:
            next_eval += EVAL_EVERY
            t = greedy_trades(q, bva)
            dm = float(t.groupby("day").ret.mean().mean()) if len(t) else float("nan")
            rec = {"steps": steps, "val_n": len(t), "val_daymean_bp": dm * 1e4,
                   "val_gross_bp": (t.ret.mean() + COST) * 1e4 if len(t) else None,
                   "val_hold_med": float(t.hold.median()) if len(t) else None,
                   "loss": float(loss), "min": round((time.time() - t0) / 60, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save(q.state_dict(), OUT / f"dqn_{steps // 1_000_000:02d}M.pt")
            (OUT / "dqn_val_log.json").write_text(json.dumps(log, indent=1))
    pickle.dump(sc, open(OUT / "scaler_b.pkl", "wb"))
    ok = [r for r in log if r["val_n"] >= 20 and r["val_daymean_bp"] == r["val_daymean_bp"]]
    best = max(ok, key=lambda r: r["val_daymean_bp"]) if ok else None
    (OUT / "dqn_selected.json").write_text(json.dumps(best, indent=1))
    print("selected", best)


if __name__ == "__main__":
    main()
