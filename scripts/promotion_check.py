"""실매매 승격 자동 점검 — docs/PROMOTION_CHECKLIST.md 의 11개 항목을 기계가 확인한다.

    ~/of80/.rlvenv/bin/python scripts/promotion_check.py --study rl93
    ~/of80/.rlvenv/bin/python scripts/promotion_check.py --study rl93 --parity-day 2026-09-24

사람 판단이 끼어들 자리를 없애는 게 목적이다. 하나라도 미달이면 승격하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research" / "runs"
GATE_NAMES = ("1 평균>0", "2 CI하한>0", "3 상위1%제거>0", "4 음수일≤1/3", "5 DSR>0.95",
              "6 대조 p<0.05", "7 일≥10·n≥30")


def _final_result(study: str) -> dict | None:
    """전진 최종 평가 기록(start_run final=True 가 남긴 것)에서 게이트 결과를 찾는다."""
    d = RUNS / study
    if not d.exists():
        return None
    best = None
    for p in sorted(d.rglob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        blob = json.dumps(obj, ensure_ascii=False)
        if "checks" in blob and "verdict" in blob:
            best = obj
    return best


def _checks_of(obj: dict) -> dict:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "checks" in cur and isinstance(cur["checks"], dict):
                return cur["checks"]
            stack += list(cur.values())
        elif isinstance(cur, list):
            stack += cur
    return {}


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return 99, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--parity-day", default=None, help="피처 패리티를 돌릴 날짜(생략 시 건너뛰고 미달 처리)")
    ap.add_argument("--python", default=str(Path.home() / "of80" / ".rlvenv" / "bin" / "python"))
    a = ap.parse_args()
    rows: list[tuple[str, bool, str]] = []

    # A. 통계 게이트 1~7 — 전진 최종 평가가 남긴 기록에서 읽는다(여기서 다시 계산하지 않는다)
    final = _final_result(a.study)
    checks = _checks_of(final) if final else {}
    for name in GATE_NAMES:
        got = checks.get(name)
        rows.append((f"A{name}", bool(got), "기록 없음(전진 최종 평가 전)" if got is None else str(got)))

    # 8. 피처 패리티
    if a.parity_day:
        code, out = _run([a.python, str(ROOT / "scripts" / "parity" / "feature_parity.py"), a.parity_day,
                          "--codes", "5", "--v2"])
        rows.append(("B8 피처 패리티", code == 0, out.splitlines()[-1] if out else ""))
    else:
        rows.append(("B8 피처 패리티", False, "--parity-day 미지정"))

    # 9. 의사결정 패리티
    code, out = _run([a.python, str(ROOT / "scripts" / "parity" / "decision_parity.py"), "--study", a.study])
    rows.append(("B9 의사결정 패리티", code == 0, out.splitlines()[-1] if out else ""))

    # 10. 매매 기준 문서화 — 확정판 표시가 있어야 한다
    crit = (ROOT / "docs" / "TRADING_CRITERIA.md").read_text(encoding="utf-8")
    has_doc = f"확정판({a.study})" in crit
    rows.append(("B10 기준 문서화", has_doc, "docs/TRADING_CRITERIA.md 에 '확정판(<study>)' 절 필요"))

    # 11. 안전장치 — 실행줄에 1주·가격밴드·킬스위치가 붙어 있나
    sh = (ROOT / "scripts" / "pair_detect_daily.sh").read_text(encoding="utf-8")
    need = {"--qty 1": r"--qty\s+1\b", "가격밴드": r"--price-band|price_band|±3", "킬스위치": r"--max-consecutive-losses"}
    missing = [k for k, pat in need.items() if not re.search(pat, sh)]
    rows.append(("B11 안전장치", not missing, "누락: " + ", ".join(missing) if missing else "1주·가격밴드·킬스위치 확인"))

    width = max(len(r[0]) for r in rows)
    print(f"# 승격 점검 — {a.study}\n")
    for name, ok, note in rows:
        print(f"{'✅' if ok else '❌'} {name.ljust(width)}  {note}")
    passed = all(ok for _, ok, _ in rows)
    token = RUNS / a.study / "PROMOTED"
    if passed:
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"checked_at": __import__("datetime").datetime.now().isoformat(),
                                     "study": a.study}, ensure_ascii=False), encoding="utf-8")
        print("\n판정: 승격 가능 — 11개 항목 전부 통과")
        print(f"승격 토큰 생성: {token} (이 파일이 있어야 --rl-live 가 주문을 낸다)")
    else:
        if token.exists():
            token.unlink()
            print(f"\n승격 토큰 회수: {token}")
        print("판정: 승격 불가 — 미달 항목이 있다")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
