"""89번 지정가 전용 2단 뉴럴 밴딧 — 체결 확률 p̂ × 체결 시 수익 톰슨 r̃. 사전등록 docs/research/manju/89.

    ~/of80/.rlvenv/bin/python scripts/bandit_89/bandit.py         # 17일 선행 평가(기술)
보상 사전계산은 88번 `~/of80/bandit88/<day>.pkl`(y2·free2·filled2) 을 그대로 쓴다.
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
OUT = Path(os.environ.get("BANDIT89_OUT", HOME / "bandit89_run"))
SEED, HID, LAMBDA, MIN_VAR = 89, 64, 1.0, 100.0
EPOCHS, BATCH, LR = 5, 4096, 1e-3
WARMUP_DAYS, FILL_DELAY = 3, 11
FORWARD_START = "2026-09-16"


class Body(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, HID), nn.ReLU())
        self.fill = nn.Linear(HID, 1)
        self.ret = nn.Linear(HID, 1)


class Head:
    def __init__(self) -> None:
        self.A = np.eye(HID) * LAMBDA
        self.b = np.zeros(HID)

    def add(self, phi, r, var):
        if len(r):
            self.A += phi.T @ phi / var
            self.b += phi.T @ r / var

    def sample(self, rng):
        cov = np.linalg.inv(self.A)
        cov = (cov + cov.T) / 2
        return rng.multivariate_normal(cov @ self.b, cov, method="cholesky")


def run(days: list[str], *, forward_ok: bool = False, state_in: Path | None = None,
        feats: list[str] | None = None) -> pd.DataFrame:
    """feats 를 안 주면 84번 13피처(89번 사전등록). 91번은 30피처를 넘기고 DATA·OUT·SEED 모듈 값을 바꿔 호출한다."""
    if feats is None:
        feats = pickle.loads((HOME / "flow84_model" / "model.pkl").read_bytes())["features"]
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    body = Body(len(feats))
    oX, oF, oR = [], [], []            # 관찰: 문맥 · 체결여부 · 체결 시 수익(미체결 nan)
    mu = sd = None
    trained = False
    if state_in is not None and state_in.exists():
        st = torch.load(state_in, weights_only=False)
        body.load_state_dict(st["body"])
        mu, sd, trained = st["mu"], st["sd"], True
        oX, oF, oR = [st["X"]], [st["F"]], [st["R"]]
    trades = []
    for day in days:
        if not forward_ok:
            assert day < FORWARD_START
        df = pickle.loads((DATA / f"{day}.pkl").read_bytes()).sort_values(["row", "code"], kind="stable")
        X = df[feats].to_numpy(np.float64)
        if mu is None:
            first = np.concatenate([pickle.loads((DATA / f"{d}.pkl").read_bytes())[feats].to_numpy(np.float64)
                                    for d in days[:WARMUP_DAYS]])
            first = first[np.isfinite(first).all(axis=1)]
            mu, sd = first.mean(axis=0), first.std(axis=0) + 1e-8
        Z = np.nan_to_num(np.clip((X - mu) / sd, -10, 10)).astype(np.float32)
        with torch.no_grad():
            h = body.phi(torch.from_numpy(Z))
            PHI = h.numpy().astype(np.float64)
            pfill = torch.sigmoid(body.fill(h)).squeeze(1).numpy() if trained else np.full(len(Z), 0.5)
        head = Head()
        R_all = np.concatenate(oR) if oR else np.zeros(0)
        filled_obs = R_all[np.isfinite(R_all)]
        var = max(float(filled_obs.var()) if len(filled_obs) > 1 else MIN_VAR, MIN_VAR)
        if len(filled_obs):
            Xo = np.concatenate(oX)
            m = np.isfinite(R_all)
            with torch.no_grad():
                Po = body.phi(torch.from_numpy(Xo[m])).numpy().astype(np.float64)
            head.add(Po, R_all[m], var)
        rows = df.row.to_numpy(np.int64)
        codes = df.code.to_numpy()
        y2, free2, filled2 = df.y2.to_numpy(), df.free2.to_numpy(np.int64), df.filled2.to_numpy()
        free: dict[str, int] = {}
        pending: list[tuple[int, int]] = []
        dX, dF, dR = [], [], []
        minutes = rows // 60
        bounds = np.flatnonzero(np.diff(minutes)) + 1
        for s0, e0 in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(rows)]]), strict=True):
            t_now = rows[s0]
            mat = [p for p in pending if p[0] <= t_now]
            if mat:
                pending = [p for p in pending if p[0] > t_now]
                idx = [i for _, i in mat]
                head.add(PHI[idx], y2[idx], var)
            w = head.sample(rng)
            score = pfill[s0:e0] * (PHI[s0:e0] @ w)
            for k in np.flatnonzero(score > 0):
                i = s0 + k
                c = codes[i]
                if rows[i] < free.get(c, -1):
                    continue
                free[c] = free2[i]
                f = bool(filled2[i])
                if f:
                    pending.append((free2[i], i))
                dX.append(Z[i])
                dF.append(float(f))
                dR.append(float(y2[i]) if f else np.nan)
                trades.append((day, c, int(rows[i]), f, float(y2[i]) if f else np.nan, float(pfill[i])))
        if dX:
            oX.append(np.stack(dX))
            oF.append(np.array(dF, np.float32))
            oR.append(np.array(dR))
        if oX and sum(len(x) for x in oX) >= 64:
            Xo = torch.from_numpy(np.concatenate(oX))
            Fo = torch.from_numpy(np.concatenate(oF))
            Ro = torch.from_numpy(np.nan_to_num(np.concatenate(oR)).astype(np.float32))
            Mo = torch.from_numpy(np.isfinite(np.concatenate(oR)).astype(np.float32))
            opt = torch.optim.Adam(body.parameters(), lr=LR)
            for _ in range(EPOCHS):
                perm = torch.randperm(len(Xo))
                for j in range(0, len(Xo), BATCH):
                    b = perm[j:j + BATCH]
                    hh = body.phi(Xo[b])
                    l_fill = nn.functional.binary_cross_entropy_with_logits(body.fill(hh).squeeze(1), Fo[b])
                    err = (body.ret(hh).squeeze(1) - Ro[b]) ** 2 * Mo[b]
                    l_ret = err.sum() / Mo[b].sum().clamp(min=1) / 100.0
                    loss = l_fill + l_ret
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            trained = True
        n = sum(1 for t in trades if t[0] == day)
        nf = sum(1 for t in trades if t[0] == day and t[3])
        print(f"{day} 주문 {n} · 체결 {nf}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"body": body.state_dict(), "mu": mu, "sd": sd,
                "X": np.concatenate(oX) if oX else np.zeros((0, len(feats)), np.float32),
                "F": np.concatenate(oF) if oF else np.zeros(0, np.float32),
                "R": np.concatenate(oR) if oR else np.zeros(0)}, OUT / "state.pt")
    return pd.DataFrame(trades, columns=["day", "code", "row", "filled", "ret_bp", "p_fill"])


def main() -> None:
    days = sorted(p.name[:10] for p in DATA.glob("*.pkl") if p.name[:10] < FORWARD_START)
    t = run(days)
    t.to_csv(OUT / "trades.csv", index=False)
    ev = t[~t.day.isin(days[:WARMUP_DAYS])]
    f = ev[ev.filled]
    dm = f.groupby("day").ret_bp.mean()
    rep = {"orders": len(ev), "filled": len(f), "fill_rate": round(len(f) / max(len(ev), 1), 3),
           "mean_bp": float(f.ret_bp.mean()) if len(f) else None, "median_bp": float(f.ret_bp.median()) if len(f) else None,
           "daymean_bp": float(dm.mean()) if len(f) else None, "pos_days": f"{int((dm > 0).sum())}/{len(dm)}",
           "p_fill_mean_ordered": float(ev.p_fill.mean()) if len(ev) else None,
           "per_day": f.groupby("day").ret_bp.agg(["size", "mean"]).round(1).to_dict(orient="index")}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    from krx_quant_core.runtime import DataSpec, start_run

    with start_run("bandit89", {"prereg": "docs/research/manju/89"}, repo_root=Path(__file__).resolve().parents[2],
                   data=DataSpec("2026-08-24", "2026-09-15", "prequential"), seed=SEED, allow_dirty=True) as r_:
        main()
        r_.log_result(json.loads((OUT / "report.json").read_text()))
