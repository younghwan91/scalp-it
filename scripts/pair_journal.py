#!/usr/bin/env python3
"""만쥬식 매매일지 생성 — 그날 로그를 사람이 복기할 글로 바꾼다(문서 44 §10-2).

만쥬는 일지를 실력 유지의 필수 요건이라 못박았다. 우리는 로그가 다 있었지만 그건
기계가 읽는 형식이라 복기가 안 됐다.

    export $(grep -v '^#' ../kr-quant/.env | xargs)
    uv run --extra pg python scripts/pair_journal.py            # 오늘
    uv run --extra pg python scripts/pair_journal.py --date 2026-08-25
    uv run --extra pg python scripts/pair_journal.py --out data/journal/2026-08-26.md

시황(국면)은 DB 에서 읽어 머리말에 박는다 — **쉬라고 한 날의 성적을 활황장 성적과
섞어 읽으면 판단이 무너진다**. DB 를 못 읽으면 국면 없이 그냥 만든다(일지가 안 나오는
것보다 낫다).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scalp_it.journal import build_entries, render_journal

ALERTS = "data/pair_alerts.log"
POSITIONS = "data/pair_positions.log"


def _load(path: str, day: str) -> tuple[list[dict], list[dict]]:
    """그날 레코드와 통지를 갈라 읽는다. 파일이 없으면 빈 리스트."""
    recs: list[dict] = []
    notices: list[dict] = []
    p = Path(path)
    if not p.exists():
        return recs, notices
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "notice" in d:
            # 통지에는 날짜가 없다(시각만) — 그날 파일 안에 있으면 그날 것으로 본다.
            notices.append(d)
        elif str(d.get("ts", "")).startswith(day):
            recs.append(d)
    return recs, notices


def _regime(day: str) -> str:
    """그날 국면 한 줄. DB 를 못 읽으면 빈 문자열(일지는 그래도 나온다)."""
    try:
        import psycopg2

        from scalp_it.market_regime import evaluate_gate, previous_session

        dsn = os.environ.get("KR_QUANT_DB")
        if not dsn:
            return ""
        with psycopg2.connect(dsn) as conn:
            prev = previous_session(conn, day)
            if prev is None:
                return ""
            g = evaluate_gate(conn, prev, target=date.fromisoformat(day))
            tag = "매매 권고" if g.tradable else "**만쥬 기준 쉬는 날**"
            extra = ""
            try:
                from scalp_it.market_regime import count_limitups, is_mania

                n_lim = count_limitups(conn, prev)
                extra = f" · 상한가 {n_lim}건"
                if is_mania(turnover_jo=g.turnover_jo, limitups=n_lim):
                    # 광기 구간의 성적을 평상시와 섞어 읽으면 판단이 무너진다(§4-3).
                    extra += " · **광기 구간**"
            except Exception:  # noqa: BLE001, S110 — 부가 정보라 없어도 된다
                pass
            return (f"{g.asof} 코스닥 {g.turnover_jo:.2f}조 · {g.regime} · {tag}"
                    f" (비중배수 {g.size_multiplier:g}){extra}")
    except Exception as exc:  # noqa: BLE001 — 국면을 못 읽어도 일지는 나와야 한다
        print(f"[경고] 국면 조회 실패(일지는 계속 생성): {exc}", file=sys.stderr)
        return ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="만쥬식 매매일지 생성")
    p.add_argument("--date", default=None, help="YYYY-MM-DD. 생략 시 오늘")
    p.add_argument("--alerts", default=ALERTS)
    p.add_argument("--positions", default=POSITIONS)
    p.add_argument("--out", default=None, help="파일로 저장. 생략 시 표준출력")
    p.add_argument("--no-regime", action="store_true", help="국면 조회 생략(DB 불필요)")
    a = p.parse_args(argv)

    day = a.date or date.today().isoformat()  # noqa: DTZ011 — KST naive, 프로젝트 표준
    alerts, notices = _load(a.alerts, day)
    closes, _ = _load(a.positions, day)
    entries = build_entries(alerts, closes, notices=notices)
    text = render_journal(entries, day=day,
                          regime="" if a.no_regime else _regime(day))

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"저장 {out} ({len(entries)}건)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
