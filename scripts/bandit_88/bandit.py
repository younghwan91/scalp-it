"""88번 얕은 뉴럴-리니어 톰슨 밴딧 — 선행(prequential) 평가. 사전등록 docs/research/manju/88 §1~§4.

    ~/of80/.rlvenv/bin/python scripts/bandit_88/bandit.py            # 17일 선행 평가(기술)
결과: ~/of80/bandit88_run/{trades.csv, state.pt, report.json}
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HOME = Path.home() / "of80"
DATA = HOME / "bandit88"
OUT = Path(os.environ.get("BANDIT88_OUT", HOME / "bandit88_run"))
SEED = 88
HID = 64
LAMBDA = 1.0
MIN_VAR = 100.0
EPOCHS, BATCH, LR = 5, 4096, 1e-3
WARMUP_DAYS = 3
FORWARD_START = "2026-09-16"


class Body(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, HID), nn.ReLU())
        self.heads = nn.Linear(HID, 2)

    def forward(self, x):
        return self.heads(self.phi(x))


class Head:
    """팔 하나의 베이지안 선형회귀(φ 위)."""

    def __init__(self) -> None:
        self.A = np.eye(HID) * LAMBDA
        self.b = np.zeros(HID)

    def add(self, phi: np.ndarray, r: np.ndarray, var: float) -> None:
        if len(r):
            self.A += phi.T @ phi / var
            self.b += phi.T @ r / var

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        cov = np.linalg.inv(self.A)
        cov = (cov + cov.T) / 2
        return rng.multivariate_normal(cov @ self.b, cov, method="cholesky")


def load_day(day: str, feats: list[str]) -> pd.DataFrame:
    return pickle.loads((DATA / f"{day}.pkl").read_bytes()).sort_values(["row", "code"], kind="stable")


def run(days: list[str], *, forward_ok: bool = False, state_in: Path | None = None) -> pd.DataFrame:
    feats = pickle.loads((HOME / "flow84_model" / "model.pkl").read_bytes())["features"]
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    body = Body(len(feats))
    obs_X: list[np.ndarray] = []
    obs_a: list[np.ndarray] = []
    obs_r: list[np.ndarray] = []
    mu = sd = None
    if state_in is not None and state_in.exists():
        st = torch.load(state_in, weights_only=False)
        body.load_state_dict(st["body"])
        mu, sd = st["mu"], st["sd"]
        obs_X, obs_a, obs_r = [st["X"]], [st["a"]], [st["r"]]
    trades = []
    for di, day in enumerate(days):
        if not forward_ok:
            assert day < FORWARD_START
        df = load_day(day, feats)
        X = df[feats].to_numpy(np.float64)
        if mu is None:                      # 첫 3거래일로 표준화 고정(§1)
            first = np.concatenate([load_day(d, feats)[feats].to_numpy(np.float64) for d in days[:WARMUP_DAYS]])
            first = first[np.isfinite(first).all(axis=1)]
            mu, sd = first.mean(axis=0), first.std(axis=0) + 1e-8
        Z = np.nan_to_num(np.clip((X - mu) / sd, -10, 10)).astype(np.float32)
        with torch.no_grad():
            PHI = body.phi(torch.from_numpy(Z)).numpy().astype(np.float64)
        # 머리: 지금까지 관찰 전부를 현재 몸통 φ 로 다시 쌓는다
        heads = [Head(), Head()]
        allr = np.concatenate(obs_r) if obs_r else np.zeros(0)
        var = max(float(allr.var()) if len(allr) > 1 else MIN_VAR, MIN_VAR)
        if obs_r:
            Xo = np.concatenate(obs_X)
            ao = np.concatenate(obs_a)
            ro = allr
            with torch.no_grad():
                Po = body.phi(torch.from_numpy(Xo)).numpy().astype(np.float64)
            for a in (1, 2):
                m = ao == a
                heads[a - 1].add(Po[m], ro[m], var)
        rows = df.row.to_numpy(np.int64)
        codes = df.code.to_numpy()
        y1, y2 = df.y1.to_numpy(), df.y2.to_numpy()
        f1, f2 = df.free1.to_numpy(np.int64), df.free2.to_numpy(np.int64)
        free: dict[str, int] = {}
        pending: list[tuple[int, int, int]] = []     # (청산 초, 행 인덱스, 팔)
        minutes = rows // 60
        day_X, day_a, day_r = [], [], []
        bounds = np.flatnonzero(np.diff(minutes)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(rows)]])
        for s0, e0 in zip(starts, ends, strict=True):
            t_now = rows[s0]
            matured = [p for p in pending if p[0] <= t_now]
            if matured:
                pending = [p for p in pending if p[0] > t_now]
                for a in (1, 2):
                    idx = [i for _, i, aa in matured if aa == a]
                    if idx:
                        r = (y1 if a == 1 else y2)[idx]
                        heads[a - 1].add(PHI[idx], r, var)
            w1, w2 = heads[0].sample(rng), heads[1].sample(rng)
            s1 = PHI[s0:e0] @ w1
            s2 = PHI[s0:e0] @ w2
            best = np.maximum(s1, s2)
            for k in np.flatnonzero(best > 0):
                i = s0 + k
                c = codes[i]
                if rows[i] < free.get(c, -1):
                    continue
                a = 1 if s1[k] >= s2[k] else 2
                r = y1[i] if a == 1 else y2[i]
                fr = f1[i] if a == 1 else f2[i]
                free[c] = fr
                pending.append((fr, i, a))
                day_X.append(Z[i])
                day_a.append(a)
                day_r.append(r)
                filled = True if a == 1 else bool(df.filled2.iat[i])
                trades.append((day, c, int(rows[i]), a, float(r), filled))
        # 장 마감 뒤 몸통 재학습(관찰된 보상만)
        if day_r:
            obs_X.append(np.stack(day_X))
            obs_a.append(np.array(day_a))
            obs_r.append(np.array(day_r))
        Xo = torch.from_numpy(np.concatenate(obs_X)) if obs_X else None
        if Xo is not None and len(Xo) >= 64:
            ao = torch.from_numpy(np.concatenate(obs_a) - 1)
            ro = torch.from_numpy(np.concatenate(obs_r).astype(np.float32))
            opt = torch.optim.Adam(body.parameters(), lr=LR)
            for _ in range(EPOCHS):
                perm = torch.randperm(len(Xo))
                for j in range(0, len(Xo), BATCH):
                    b = perm[j:j + BATCH]
                    pred = body(Xo[b]).gather(1, ao[b][:, None]).squeeze(1)
                    loss = nn.functional.mse_loss(pred, ro[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        n_day = len([t for t in trades if t[0] == day])
        print(f"{day} 결정행 {len(df):,} · 선택 {n_day}건 · 누적 관찰 {sum(len(r) for r in obs_r)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"body": body.state_dict(), "mu": mu, "sd": sd,
                "X": np.concatenate(obs_X) if obs_X else np.zeros((0, len(feats)), np.float32),
                "a": np.concatenate(obs_a) if obs_a else np.zeros(0, int),
                "r": np.concatenate(obs_r) if obs_r else np.zeros(0)}, OUT / "state.pt")
    return pd.DataFrame(trades, columns=["day", "code", "row", "arm", "ret_bp", "filled"])


def main() -> None:
    days = sorted(p.name[:10] for p in DATA.glob("*.pkl") if p.name[:10] < FORWARD_START)
    t = run(days)
    t.to_csv(OUT / "trades.csv", index=False)
    ev = t[~t.day.isin(days[:WARMUP_DAYS])]
    real = ev[ev.filled]
    rep = {"days_eval": len(days) - WARMUP_DAYS, "choices": len(ev), "trades_filled": len(real),
           "arm_share": ev.arm.value_counts(normalize=True).round(3).to_dict(),
           "mean_bp": float(real.ret_bp.mean()) if len(real) else None,
           "median_bp": float(real.ret_bp.median()) if len(real) else None,
           "daymean_bp": float(real.groupby("day").ret_bp.mean().mean()) if len(real) else None,
           "pos_days": f"{int((real.groupby('day').ret_bp.mean() > 0).sum())}/{real.day.nunique()}",
           "per_arm": real.groupby("arm").ret_bp.agg(["size", "mean", "median"]).round(1).to_dict(orient="index"),
           "per_day": real.groupby("day").ret_bp.agg(["size", "mean"]).round(1).to_dict(orient="index")}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    if "--forward" in sys.argv:
        raise SystemExit("전진 실행은 forward_gate.py --study bandit88 로만")
    from krx_quant_core.runtime import DataSpec, start_run

    with start_run("bandit88", {"prereg": "docs/research/manju/88", "HID": HID, "LAMBDA": LAMBDA},
                   repo_root=Path(__file__).resolve().parents[2],
                   data=DataSpec("2026-08-24", "2026-09-15", "prequential"), seed=SEED, allow_dirty=True) as run_:
        main()
        run_.log_result(json.loads((OUT / "report.json").read_text()))
