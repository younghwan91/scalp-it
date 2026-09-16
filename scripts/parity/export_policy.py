"""학습된 RL 정책을 실시간 그림자가 읽는 형식으로 내보낸다(승격 체크리스트 9번 준비).

    ~/of80/.rlvenv/bin/python scripts/parity/export_policy.py --study rl93 --out ~/of80/policy_rl93

`<out>/policy.json`(피처 이름·표준화 통계·연구 이름) + `<out>/ckpt.pt`(선택된 체크포인트 가중치).
감지기는 `--rl-shadow <out>` 로 읽는다. **여기서 값을 새로 만들지 않는다** — 학습이 저장한 것을 그대로 옮긴다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home() / "of80"
MODULE = {"rl85": "rl_85", "rl87": "rl_87", "rl92": "rl_92", "rl93": "rl_93", "rl94": "rl_94"}


ACTIONS_4 = ("hold", "market_buy", "limit_buy", "sell")
ACTIONS_3 = ("hold", "limit_buy", "sell")          # 96번 — 시장가 매수는 95번 측정으로 배제


def _actions(prep: dict, *, out_dim: int) -> tuple[str, ...]:
    """학습 산출물이 적어 뒀으면 그대로, 아니면 출력 차원으로 고른다."""
    saved = prep.get("actions")
    if saved:
        return tuple(saved)
    if out_dim == 3:
        return ACTIONS_3
    if out_dim == 4:
        return ACTIONS_4
    raise SystemExit(f"행동 {out_dim}개짜리 정책은 아직 모른다 — 행동 이름을 학습 쪽에서 저장하라")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=sorted(MODULE))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = HOME / a.study
    sel = json.loads((base / "selected.json").read_text())
    if not sel:
        print("선택된 체크포인트가 없다(VAL 거래 30건 이상 없음) — 내보낼 정책이 없다")
        return 2
    seed, steps = sel["seed"], sel["steps"] // 1_000_000
    # 학습이 rl_85 를 "rl85_dqn" 으로 불러 썼다 — 피클이 그 이름을 참조하므로 먼저 등록한다.
    r85_spec = importlib.util.spec_from_file_location("rl85_dqn", ROOT / "scripts" / "rl_85" / "dqn.py")
    r85 = importlib.util.module_from_spec(r85_spec)
    sys.modules["rl85_dqn"] = r85
    r85_spec.loader.exec_module(r85)
    spec = importlib.util.spec_from_file_location("mod", ROOT / "scripts" / MODULE[a.study] / "dqn.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod"] = mod

    class _U(pickle.Unpickler):                       # 학습이 __main__ 소속으로 피클한 Scaler 를 이어 준다
        def find_class(self, module, name):
            if name == "Scaler" and module in ("__main__", "rl85_dqn", "mod"):
                return r85.Scaler
            return super().find_class(module, name)

    with open(base / f"seed{seed}" / "prep.pkl", "rb") as fh:
        prep = _U(fh).load()
    sc = prep["scaler"]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = base / f"seed{seed}" / f"ckpt_{steps:02d}M.pt"
    shutil.copy(ckpt, out / "ckpt.pt")
    # 운용 환경에 torch 를 넣지 않으려고 가중치를 numpy 로도 저장한다(그림자는 이걸 읽는다).
    import numpy as np
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    keys = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith(".weight")},
                  key=lambda k: int(k.rsplit(".", 1)[-1]))
    np.savez(out / "weights.npz",
             **{f"w{i}": sd[f"{k}.weight"].numpy() for i, k in enumerate(keys)},
             **{f"b{i}": sd[f"{k}.bias"].numpy() for i, k in enumerate(keys)})
    (out / "policy.json").write_text(json.dumps({
        "study": a.study, "seed": seed, "steps_m": steps, "features": list(prep["features"]),
        # 행동 공간을 산출물에 박는다 — 그림자가 인덱스로 추측하면 한 칸 밀려도 조용하다.
        "actions": list(_actions(prep, out_dim=len(sd[f"{keys[-1]}.bias"]))),
        "mu": list(map(float, sc.mu)), "sd": list(map(float, sc.sd)),
        "lo": list(map(float, sc.lo)), "hi": list(map(float, sc.hi)),
        "val": {k: sel[k] for k in ("val_n", "val_daymean_bp", "val_mean_bp", "val_hold_med") if k in sel},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"내보냄: {out} · 피처 {len(prep['features'])}개 · seed {seed} · {steps}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
