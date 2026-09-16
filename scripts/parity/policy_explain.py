"""학습된 RL 정책이 **무엇을 보고 사고 파는지** 숫자로 설명한다 — 승격 체크리스트 10번(기준 문서화) 재료.

    ~/of80/.rlvenv/bin/python scripts/parity/policy_explain.py --study rl93 --out data/rl93_criteria.md

내용: (1) 매매 동기 — 진입 순간 피처가 모집단 대비 어디에 있나(백분위), (2) 매수 시점 — 시간대·직전 움직임,
(3) 매도 시점 — 보유 시간 분포, 이익 청산 vs 손실 청산의 직전 변화, (4) 손절·익절이 실제로 언제 일어나는가.
**정책을 해석할 뿐 바꾸지 않는다.**
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home() / "of80"
MODULE = {"rl85": "rl_85", "rl87": "rl_87", "rl92": "rl_92", "rl93": "rl_93", "rl94": "rl_94"}


def _load(study: str):
    spec = importlib.util.spec_from_file_location(f"{study}_mod", ROOT / "scripts" / MODULE[study] / "dqn.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{study}_mod"] = mod
    spec.loader.exec_module(mod)
    sel = json.loads((HOME / study / "selected.json").read_text())
    base = HOME / study / f"seed{sel['seed']}"

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "Scaler":
                return getattr(mod, "Scaler", None) or mod.R85.Scaler
            return super().find_class(module, name)

    with open(base / "prep.pkl", "rb") as fh:
        prep = _U(fh).load()
    return mod, sel, prep, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=sorted(MODULE))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch

    mod, sel, prep, base = _load(a.study)
    feats = prep["features"]
    days = [d for d in sorted(p.name[:10] for p in (mod.R85.C84 if a.study not in ("rl94",) else mod.C94).glob("*.pkl"))
            if mod.R85.VAL[0] <= d <= mod.R85.VAL[1]]
    parts = mod.load_days(days, feats) if a.study != "rl94" else mod.load_days(days)
    bk = mod.make_book(parts, prep["scaler"])
    q = (mod.Q if hasattr(mod, "Q") else mod.R85.Q)(len(feats) + mod.EXTRA)
    q.load_state_dict(torch.load(base / f"ckpt_{sel['steps'] // 1_000_000:02d}M.pt"))
    trades = mod.greedy(q, bk)
    if trades.empty:
        print("정책이 거래하지 않는다 — 설명할 것이 없다")
        return 2

    # 진입 순간 피처가 모집단 대비 어디인가
    X = np.concatenate([p[1] for p in parts])
    key = {k: i for i, k in enumerate(bk.keys)}
    idx = []
    for r in trades.itertuples():
        s, e = bk.start[key[(r.day, r.code)]], bk.end[key[(r.day, r.code)]]
        secs = bk.sec[s:e]
        j = int(np.searchsorted(secs, r.sec))
        if j < len(secs):
            idx.append(s + j)
    idx = np.array(idx)
    pct = {}
    for i, f in enumerate(feats):
        col = X[:, i].astype(float)
        col = col[np.isfinite(col)]
        at = X[idx, i].astype(float)
        at = at[np.isfinite(at)]
        if len(col) and len(at):
            pct[f] = float((col[:, None] < at[None, :]).mean() * 100)
    top = sorted(pct.items(), key=lambda kv: -abs(kv[1] - 50))[:8]

    win, lose = trades[trades.ret > 0], trades[trades.ret <= 0]
    L = [f"## 매매 기준 확정판({a.study}) — 정책이 실제로 하는 것", "",
         f"- 선택 체크포인트: seed {sel['seed']} · {sel['steps'] // 1_000_000}M · "
         f"검증 {sel['val_n']}거래 · 일평균 {sel['val_daymean_bp']:+.1f}bp · 거래당 {sel.get('val_mean_bp', float('nan')):+.1f}bp",
         f"- 검증 기간 거래 {len(trades)}건 · 보유 중앙 {trades.hold.median():.0f}초 "
         f"(25%={trades.hold.quantile(.25):.0f}s · 75%={trades.hold.quantile(.75):.0f}s)", "",
         "### 1. 매매 동기 — 진입 순간 무엇이 특이한가 (모집단 대비 백분위)", "",
         "| 피처 | 진입 시 평균 백분위 | 읽는 법 |", "|---|--:|---|"]
    for f, p in top:
        L.append(f"| `{f}` | {p:.0f}% | {'상위' if p > 50 else '하위'} 구간에서 들어간다 |")
    L += ["", "### 2. 매수 시점", "",
          f"- 하루 중 분포: 09시대 {(trades.sec < 3600).mean():.0%} · 점심 전 {((trades.sec >= 3600) & (trades.sec < 10800)).mean():.0%}"
          f" · 오후 {(trades.sec >= 10800).mean():.0%}",
          f"- 종목당 하루 평균 {len(trades) / max(trades.groupby(['day', 'code']).ngroups, 1):.1f}건", "",
          "### 3. 매도 시점 — 손절·익절이 실제로 언제 일어나나", "",
          f"- 이익 청산 {len(win)}건: 보유 중앙 {win.hold.median() if len(win) else float('nan'):.0f}초 · "
          f"평균 {win.ret.mean() * 1e4 if len(win) else float('nan'):+.1f}bp",
          f"- 손실 청산 {len(lose)}건: 보유 중앙 {lose.hold.median() if len(lose) else float('nan'):.0f}초 · "
          f"평균 {lose.ret.mean() * 1e4 if len(lose) else float('nan'):+.1f}bp",
          f"- 손실 청산이 이익 청산보다 {'빠르다' if len(win) and len(lose) and lose.hold.median() < win.hold.median() else '느리다'}"
          " — 빠를수록 '아니다 싶으면 바로 손절'에 가깝다", "",
          "### 4. 한계", "",
          "- 위 숫자는 **검증 3일**의 행동이다. 전진 기간에 같은 행동을 하는지는 의사결정 패리티(9번)가 확인한다.",
          "- 백분위는 상관이지 인과가 아니다 — 정책이 그 피처를 '봤다'는 증거가 아니라 그런 순간에 들어갔다는 기술이다."]
    text = "\n".join(L)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
