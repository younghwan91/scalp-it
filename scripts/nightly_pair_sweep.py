#!/usr/bin/env python3
"""nightly job ``pair-sweep-yesterday`` — 최근 거래일 틱으로 짝꿍 파라미터 스윕.

simnode 크론(17:45)이 ``kqc nightly .`` 로 ``research/nightly.toml`` 을 돌리면 이 래퍼가 불린다.

* 날짜: ``daily_bars`` 기준 **오늘 이하 가장 최근 거래일**(:func:`market_regime.previous_session`
  재사용). 17:45 에 당일 일봉이 아직 없으면 직전 거래일이 된다.
* 쌍: 만쥬 방식은 고정 쌍이 없다(워치리스트 비어 있음). 그래서 **그날 감지기가 실제로
  알림을 낸 (대장, 후속) 쌍**을 ``data/pair_alerts.log`` 에서 모은다(감지기가 같은
  호스트의 data/ 에 쓴다). 그날 알림이 없으면 스윕할 대상이 없으니 건너뛰고 0 으로 끝낸다.
* 그리드: ``scripts/pair_sweep.py`` docstring 사용 예 그대로
  (``surge_window=10,15,30`` × ``cosurge_min=0.005,0.002,0.001,0`` = 12조합).
* 실행: ``pair_sweep.py`` 를 자식 프로세스로 — krx-quant-core ``run_sweep``(병렬·시행 원장).

    uv run --extra pg python scripts/nightly_pair_sweep.py [--day YYYY-MM-DD] [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALERTS_LOG = ROOT / "data" / "pair_alerts.log"
OUT = "data/pair_sweep_nightly.jsonl"
GRID = ("surge_window=10,15,30", "cosurge_min=0.005,0.002,0.001,0")
DEFAULT_JOBS = 4  # 틱 수백만 건을 fork 로 공유 — 자식이 많으면 CoW 복사로 메모리가 는다


def latest_trading_day() -> date:
    """오늘 이하 가장 최근 거래일(일봉 기준). 연결은 짧게 열고 닫는다."""
    sys.path.insert(0, str(ROOT / "src"))
    from scalp_it.intraday.store import connect
    from scalp_it.market_regime import previous_session

    conn = connect(os.environ.get("KR_QUANT_DB"))
    try:
        today = date.today()  # noqa: DTZ011 — KST naive, 프로젝트 표준
        day = previous_session(conn, today + timedelta(days=1))
    finally:
        conn.close()
    if day is None:
        raise SystemExit("일봉에서 최근 거래일을 못 찾았다")
    return day if isinstance(day, date) else date.fromisoformat(str(day)[:10])


def pairs_from_alerts(path: Path, day: date) -> list[str]:
    """``day`` 에 알림이 난 ``대장:후속`` 쌍(처음 나온 순서, 중복 제거)."""
    if not path.exists():
        return []
    prefix = day.isoformat()
    seen: dict[str, None] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            lead, fol, ts = row.get("leader"), row.get("follower"), str(row.get("ts") or "")
            if lead and fol and ts.startswith(prefix):
                seen.setdefault(f"{lead}:{fol}", None)
    return list(seen)


def sweep_argv(day: date, pairs: list[str], *, jobs: int | None) -> list[str]:
    argv = ["--replay", day.isoformat(), "--pairs", ",".join(pairs)]
    for g in GRID:
        argv += ["--grid", g]
    argv += ["--yes", "--out", OUT]
    if jobs is not None:
        argv += ["--jobs", str(jobs)]
    return argv


def _run_pair_sweep(argv: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "pair_sweep.py"), *argv]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main(argv: list[str] | None = None, *,
         latest_day: Callable[[], date] = latest_trading_day,
         alerts_log: Path = ALERTS_LOG,
         run: Callable[[list[str]], int] = _run_pair_sweep) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--day", help="YYYY-MM-DD (기본: 최근 거래일)")
    p.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    args = p.parse_args(argv)
    day = date.fromisoformat(args.day) if args.day else latest_day()
    pairs = pairs_from_alerts(alerts_log, day)
    if not pairs:
        print(f"[nightly] {day} 에 감지기 알림 쌍이 없다({alerts_log}) — 스윕을 건너뛴다.",
              flush=True)
        return 0
    print(f"[nightly] {day} 알림 쌍 {len(pairs)}개: {', '.join(pairs)}", flush=True)
    return int(run(sweep_argv(day, pairs, jobs=args.jobs)) or 0)


if __name__ == "__main__":
    sys.exit(main())
