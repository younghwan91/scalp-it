"""98번 — 저장된 체크포인트를 **검증 구간에서 다시 돌려 거래 단위 결과를 꺼낸다**.

    ~/of80/.rlvenv/bin/python scripts/selection_98/replay_val.py rl96 rl93 rl92

왜: `val_log.json` 에는 `val_n`·`val_daymean_bp`·`val_mean_bp` 셋뿐이라 **하루에 몇 건이 들어갔고
그날 성과가 얼마였는지**를 복원할 수 없다. 검증 3 일에 미지수 6 개라 산술적으로 불가능하다.
그런데 98 이 물어야 할 것(하루 거래 k 건일 때 일평균의 표준오차, 거래 가중 규칙, 하루 하한 규칙,
신호 없음 귀무분포)은 전부 거래 단위 자료를 요구한다. 그래서 다시 돌린다.

학습 코드는 **한 줄도 바꾸지 않는다** — 각 연구의 `dqn.py` 를 그대로 import 해서 같은 로더·같은
스케일러·같은 `greedy()` 를 쓴다. 되돌린 값이 `val_log.json` 과 다르면 재현이 깨진 것이므로
그 자리에서 알린다(조용히 넘어가면 98 의 모든 숫자가 무의미해진다).

산출: `~/of80/sel98/<연구>/trades.pkl` — 체크포인트마다 (day, code, sec, hold, ret) 거래 표.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path.home() / "of80" / "sel98"
STUDY_SRC = {"rl85": "scripts/rl_85/dqn.py", "rl87": "scripts/rl_87/dqn.py",
             "rl92": "scripts/rl_92/dqn.py", "rl93": "scripts/rl_93/dqn.py",
             "rl94": "scripts/rl_94/dqn.py", "rl96": "scripts/rl_96/dqn.py"}


def load_module(study: str):
    spec = importlib.util.spec_from_file_location(f"{study}_dqn_replay", ROOT / STUDY_SRC[study])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def ckpt_dim(path: Path) -> tuple[dict, int]:
    """체크포인트에서 입력 차원을 읽는다 — 연구마다 EXTRA 가 달라 코드로 세면 어긋난다."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    first = next(v for k, v in sd.items() if k.endswith("0.weight"))
    return sd, int(first.shape[1])


def replay(study: str) -> dict:
    mod = load_module(study)
    R85 = mod.R85
    days = sorted(p.name[:10] for p in R85.C84.glob("*.pkl"))
    tr_days = [d for d in days if d <= R85.TRAIN_END]
    va_days = [d for d in days if R85.VAL[0] <= d <= R85.VAL[1]]
    print(f"[{study}] 학습 {len(tr_days)}일로 스케일러 복원 · 검증 {va_days}")
    tr = mod.load_days(tr_days, mod.FEATS)
    sc = R85.Scaler(np.concatenate([p[1] for p in tr]))
    del tr                                            # 스케일러만 있으면 된다 — 메모리를 놔 준다
    va = mod.load_days(va_days, mod.FEATS)
    bva = mod.make_book(va, sc)
    torch.set_num_threads(5)
    out: dict[tuple[int, int], object] = {}
    mismatch = []
    for f in sorted(Path.home().glob(f"of80/{study}/seed*/val_log.json")):
        seed_dir = f.parent
        # 체크포인트 파일명은 steps//1e6 으로 찍힌다(ckpt_08M.pt) — 그 키로 로그 행을 맞춘다
        log = {r["steps"] // 1_000_000: r for r in json.loads(f.read_text())}
        for ck in sorted(seed_dir.glob("ckpt_*.pt")):
            m = int(ck.stem.split("_")[1].rstrip("M"))
            if m not in log:
                print(f"  {ck.name}: 로그에 없는 체크포인트 — 건너뛴다")
                continue
            rec = log[m]
            steps = rec["steps"]
            sd, d = ckpt_dim(ck)
            q = mod.Q(d)
            q.load_state_dict(sd)
            q.eval()
            t = mod.greedy(q, bva)
            n_ok = len(t) == rec["val_n"]
            dm = float(t.groupby("day").ret.mean().mean()) * 1e4 if len(t) else float("nan")
            d_ok = (dm != dm and rec["val_daymean_bp"] != rec["val_daymean_bp"]) or \
                abs(dm - rec["val_daymean_bp"]) < 1e-6
            if not (n_ok and d_ok):
                mismatch.append({"ckpt": str(ck), "replay_n": len(t), "log_n": rec["val_n"],
                                 "replay_daymean": dm, "log_daymean": rec["val_daymean_bp"]})
            out[(rec["seed"], steps)] = t
            print(f"  {ck.name} seed{rec['seed']} {steps:>10,}  n={len(t):>6,} "
                  f"일평균 {dm:+8.1f}bp  {'✓' if n_ok and d_ok else '✗ 재현 불일치'}", flush=True)
    dst = OUT_ROOT / study
    dst.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(dst / "trades.pkl", "wb"))
    (dst / "mismatch.json").write_text(json.dumps(mismatch, indent=1))
    print(f"[{study}] 체크포인트 {len(out)}개 · 재현 불일치 {len(mismatch)}개 → {dst}")
    return {"n_ckpt": len(out), "n_mismatch": len(mismatch)}


def main(argv: list[str]) -> int:
    for study in argv or ["rl96"]:
        replay(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
