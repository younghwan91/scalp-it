"""승격 체크리스트 9번 — 실시간 RL 추론 그림자와 백테스트 재현의 **판단이 같은지** 대조한다.

    ~/of80/.rlvenv/bin/python scripts/parity/decision_parity.py --study rl93 --days 5

- 실시간: trader `data/rl_shadow_<study>.jsonl`(감지기가 소켓에서 낸 매수·매도)
- 배치:   `scripts/forward_gate.py` 가 같은 날 저장 틱으로 재현한 거래
판정선(체크리스트 9번): 신호 초 ±2초 일치 **≥ 90%**, 일평균 순수익 차이 **≤ 5bp**. 둘 다 만족해야 통과.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MATCH_SEC, MIN_RATE, MAX_BP_DIFF = 2, 0.90, 5.0


def live_trades(study: str) -> pd.DataFrame:
    p = ROOT / "data" / f"rl_shadow_{study}.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["day", "code", "sec", "ret"])
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    buys = {(r["day"], r["code"]): r["sec"] for r in rows if r["event"] == "buy"}
    out = [{"day": r["day"], "code": r["code"], "sec": buys.get((r["day"], r["code"]), r["sec"]),
            "ret": r["ret"]} for r in rows if r["event"] == "sell"]
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--days", type=int, default=5, help="최근 N 거래일만 본다")
    ap.add_argument("--batch-csv", default=None, help="배치 재현 거래 CSV(기본: 매일 저장본 → 최종본 순)")
    ap.add_argument("--agreement-only", action="store_true",
                    help="일치율만 본다 — 수익 수치는 전진 최종 평가 전까지 계산하지 않는다")
    a = ap.parse_args()

    live = live_trades(a.study)
    if live.empty:
        print(f"실시간 그림자 기록이 없다 — 감지기에 --rl-shadow {a.study} 를 켜고 거래일을 쌓아야 한다")
        return 2
    daily = ROOT / "data" / f"{a.study}_forward.daily_trades.csv"
    bp = Path(a.batch_csv) if a.batch_csv else (daily if daily.exists()
                                                else ROOT / "data" / f"{a.study}_forward.final_trades.csv")
    if not bp.exists():
        print(f"배치 재현 결과가 없다: {bp}")
        return 2
    batch = pd.read_csv(bp, dtype={"code": str})
    days = sorted(set(live.day) & set(batch.day))[-a.days:]
    live, batch = live[live.day.isin(days)], batch[batch.day.isin(days)]
    matched = 0
    for r in live.itertuples():
        same = batch[(batch.day == r.day) & (batch.code == r.code)]
        if len(same) and (np.abs(same.sec.to_numpy() - r.sec) <= MATCH_SEC).any():
            matched += 1
    rate = matched / max(len(live), 1)
    if a.agreement_only:
        ok = rate >= MIN_RATE and len(days) >= 5
        print(f"수익 수치는 계산하지 않음(--agreement-only) · 거래일 {len(days)}/5")
        print("\n판정:", "일치율 기준 충족" if ok else "미달 — 일치율 또는 거래일 부족")
        return 0 if ok else 1
    lb = live.groupby("day").ret.mean().mean() * 1e4
    bb = batch.groupby("day").ret.mean().mean() * 1e4
    print(f"거래일 {len(days)} · 실시간 {len(live)}건 · 배치 {len(batch)}건")
    print(f"신호 초 ±{MATCH_SEC}초 일치: {matched}/{len(live)} = {rate:.0%} (기준 ≥{MIN_RATE:.0%})")
    print(f"일평균 순수익: 실시간 {lb:+.1f}bp · 배치 {bb:+.1f}bp · 차이 {abs(lb - bb):.1f}bp (기준 ≤{MAX_BP_DIFF}bp)")
    ok = rate >= MIN_RATE and abs(lb - bb) <= MAX_BP_DIFF
    print("\n판정:", "통과 — 운용과 백테스트가 같은 판단을 한다" if ok else "미달 — 실매매 승격 불가")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
