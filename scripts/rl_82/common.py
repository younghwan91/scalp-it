"""82번 공통 — 시뮬레이터 체결로 거래 목록 만들기 · 무작위 대조 · 게이트(사전등록 §2·§6)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dataset import COST, Episode

N_TRIALS = 1283
SEED = 82
N_BOOT = 10_000


def trade_ret(ep: Episode, i_in: int, i_out: int) -> float:
    return ep.sell[i_out] / ep.buy[i_in] - 1.0 - COST


def random_control(trades: pd.DataFrame, eps: dict[tuple[str, str], Episode]) -> pd.DataFrame:
    """같은 종목·날, 같은 보유 시간, 진입 초만 무작위 — 사전등록 §6-6."""
    rng = np.random.default_rng(SEED)
    rets = []
    for r in trades.itertuples():
        ep = eps[(r.day, r.code)]
        hold = int(r.hold)
        cand = np.nonzero(ep.secs + hold <= ep.secs[-1])[0]
        if len(cand) == 0:
            rets.append(np.nan)
            continue
        i = int(rng.choice(cand))
        j = int(np.searchsorted(ep.secs, ep.secs[i] + hold))
        j = min(j, len(ep.secs) - 1)
        rets.append(trade_ret(ep, i, j))
    return trades.assign(ret_ctrl=rets)


def gates(trades: pd.DataFrame) -> dict:
    from krx_quant_core.stats.metrics import paired_date_bootstrap
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    R = trades.ret.to_numpy()
    n = len(R)
    res: dict = {"n": n}
    if n < 2:
        res["verdict"] = "표본 부족 — 판정 불가"
        return res
    days = sorted(trades.day.unique())
    res["mean"] = float(R.mean())
    res["median"] = float(np.median(R))
    res["gross_mean"] = float(R.mean() + COST)
    rng = np.random.default_rng(SEED)
    by = {d: trades.loc[trades.day == d, "ret"].to_numpy() for d in days}
    means = [np.concatenate([by[days[k]] for k in rng.integers(0, len(days), len(days))]).mean()
             for _ in range(N_BOOT)]
    res["ci_low"] = float(np.percentile(means, 2.5))
    k = max(1, math.ceil(0.01 * n))
    res["trim1_mean"] = float(np.sort(R)[:-k].mean())
    dm = trades.groupby("day").ret.mean()
    res["neg_days"] = f"{int((dm < 0).sum())}/{len(dm)}"
    res["dsr_prob"] = float(deflated_sharpe_from_sample(R, N_TRIALS)["prob_deflated_sharpe"])
    c = trades.dropna(subset=["ret_ctrl"])
    a = c.groupby("day").ret.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
    b = c.groupby("day").ret_ctrl.mean().rename("mean_excess").reset_index().rename(columns={"day": "date"})
    pb = paired_date_bootstrap(a, b, n_boot=N_BOOT, seed=SEED)
    res["ctrl_mean"] = float(c.ret_ctrl.mean())
    res["ctrl_p"] = 1 - pb["p_greater"]
    checks = {
        "1 평균>0": res["mean"] > 0,
        "2 CI하한>0": res["ci_low"] > 0,
        "3 상위1%제거>0": res["trim1_mean"] > 0,
        "4 음수일≤2": int((dm < 0).sum()) <= 2,
        "5 DSR>0.95": res["dsr_prob"] > 0.95,
        "6 대조 p<0.05": res["ctrl_p"] < 0.05,
        "7 n≥30": n >= 30,
    }
    res["checks"] = checks
    res["verdict"] = ("표본 부족 — 판정 불가" if not checks["7 n≥30"]
                      else "채택" if all(checks.values()) else "기각")
    return res


def fmt(name: str, g: dict) -> str:
    bp = ("mean", "median", "gross_mean", "ci_low", "trim1_mean", "ctrl_mean")
    lines = [f"### {name}", ""]
    for k, v in g.items():
        if k == "checks":
            lines.append("- 게이트: " + " · ".join(f"[{c}: {'✓' if ok else '✗'}]" for c, ok in v.items()))
        elif k in bp:
            lines.append(f"- {k}: {v * 1e4:+.1f}bp")
        else:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)
