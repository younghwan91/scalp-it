"""전진 최종 판정 모아보기 — research/runs/*/FINAL.lock 이 있는 연구의 한 번뿐인 판정을 표로.

    uv run python scripts/forward_verdicts.py

왜: 전진 크론은 조건(≥10일·≥30거래)이 차는 날 **한 번** 판정하고 FINAL 로 잠근다. 그 결과는 각 연구의
RUNS.jsonl 에만 남아 사람이 모아 보지 않으면 묻힌다(2026-10-01~05 사이 9개가 판정됐는데 10-05 에야 알았다).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verdicts() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "research" / "runs" / "*" / "FINAL.lock"))):
        d = Path(f).parent
        rec = None
        runs = d / "RUNS.jsonl"
        if runs.exists():
            for line in runs.read_text(encoding="utf-8").splitlines():
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if j.get("event") == "end" and isinstance(j.get("result"), dict) and "verdict" in j["result"]:
                    rec = j
        row = {"study": d.name, "locked": json.loads(Path(f).read_text()).get("ts", "")[:10]}
        if rec:
            r = rec["result"]
            g = r.get("gates", r)
            ck = r.get("checks") or g.get("checks") or {}
            row.update(verdict=r.get("verdict"), n=g.get("n"), days=g.get("days"), mean_bp=g.get("mean_bp"),
                       ci_low_bp=g.get("ci_low_bp"), neg_days=g.get("neg_days"), dsr=g.get("dsr_prob"),
                       passed=f"{sum(1 for v in ck.values() if v)}/{len(ck)}" if ck else "-")
        else:
            row.update(verdict="(결과 없음)")
        out.append(row)
    return out


def main() -> int:
    rows = verdicts()
    print(f"{'연구':<9}{'판정':<8}{'n':>6}{'일':>4}{'평균bp':>9}{'CI하한':>9}{'음수일':>8}{'DSR':>9}{'통과':>6}  잠금일")
    for r in rows:
        f = lambda k, w: (f"{r[k]:>{w}.1f}" if isinstance(r.get(k), float) else f"{str(r.get(k, '-')):>{w}}")  # noqa: E731
        dsr = r.get("dsr")
        print(f"{r['study']:<9}{str(r.get('verdict')):<8}{str(r.get('n', '-')):>6}{str(r.get('days', '-')):>4}"
              f"{f('mean_bp', 9)}{f('ci_low_bp', 9)}{str(r.get('neg_days', '-')):>8}"
              f"{(f'{dsr:.1e}' if isinstance(dsr, float) else '-'):>9}{str(r.get('passed', '-')):>6}  {r['locked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
