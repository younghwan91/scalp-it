"""수집 건강 점검 — 하루 커버리지를 재고 불량일을 **자동으로** 목록에 올린다.

    ~/of80/.rlvenv/bin/python scripts/collect_health.py              # 어제까지 전부
    ~/of80/.rlvenv/bin/python scripts/collect_health.py 2026-10-02   # 특정일

왜: 2026-09-28·09-29·10-02 의 분커버가 37~52% 로 떨어졌는데 **12일 뒤에야 알았다**. 사람이 안 보면
아무도 안 본다. 매일 재서 기록하고, 문턱 아래면 `data/bad_days_auto.json` 에 **자동으로** 올려 캐시
생성에서 빠지게 한다(`scripts/_bad_days.load` 가 추적 파일과 합쳐 읽는다).

추적 파일 `research/bad_days.json` 에는 크론이 쓰지 않는다 — 운영 체크아웃이 더러워지면 07:50
pull-all 이 동기화를 건너뛴다. 자동 등록분을 연구 원본에 남기려면:

    ~/of80/.rlvenv/bin/python scripts/collect_health.py --promote    # 옮긴 뒤 research/bad_days.json 커밋

분커버 = 390 거래분(09:00~15:19) 중 틱이 한 건이라도 있는 분의 비율.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bad_days
import _envload  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
BAD = _bad_days.TRACKED   # 사람·승격 항목(git 추적) — 크론은 읽기만
AUTO = _bad_days.AUTO     # 자동 등록분(추적 안 함)
THRESHOLD = 0.90
TOTAL_MIN = 380            # 09:00~15:19 (마지막 분은 집계에서 흔히 비어 380 으로 센다)


def coverage(conn, day: str) -> dict | None:
    """→ {cover, minutes, codes, ticks, last_tick} 또는 거래가 아예 없으면 None."""
    import pandas as pd

    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    r = pd.read_sql(
        "select count(distinct date_trunc('minute', ts)) m, count(distinct code) c, "
        "count(*) n, max(ts)::time t from ticks where ts >= %s and ts < %s",
        conn, params=(day, nxt)).iloc[0]
    if not int(r.n):
        return None
    return {"cover": round(min(1.0, int(r.m) / TOTAL_MIN), 3), "minutes": int(r.m),
            "codes": int(r.c), "ticks": int(r.n), "last_tick": str(r.t)}


def record(day: str, info: dict, *, why: str) -> bool:
    """문턱 아래면 자동 목록(AUTO)에 올린다. 이미 어느 목록에든 있으면 건드리지 않는다. → 새로 올렸는가."""
    if info["cover"] >= THRESHOLD:
        return False
    entry = {"cover": info["cover"], "last_tick": info["last_tick"], "why": why}
    return _bad_days.record_auto(day, entry, tracked=BAD, auto=AUTO)


def main(days: list[str]) -> int:
    import pandas as pd
    import psycopg2

    conn = psycopg2.connect(os.environ["KR_QUANT_DB"])
    if not days:
        days = pd.read_sql("select distinct ts::date d from ticks order by d",
                           conn).d.astype(str).tolist()
    added = []
    print(f"{'날짜':<12}{'종목':>5}{'틱':>11}{'분':>5}{'분커버':>8}  판정")
    for day in days:
        info = coverage(conn, day)
        if info is None:
            print(f"{day:<12}{'-':>5}{'-':>11}{'-':>5}{'-':>8}  거래 없음(휴장)")
            continue
        ok = info["cover"] >= THRESHOLD
        if not ok and record(day, info, why=f"분커버 {info['cover']:.0%} — 자동 등록(collect_health)"):
            added.append(day)
        print(f"{day:<12}{info['codes']:>5}{info['ticks']:>11,}{info['minutes']:>5}"
              f"{info['cover']:>8.0%}  {'정상' if ok else '불량'}")
    if added:
        print(f"\n불량일 {len(added)}개를 data/bad_days_auto.json 에 올렸다: {', '.join(added)}")
        print("→ 캐시 생성에서 자동으로 빠진다. 연구 원본에 남기려면 --promote 후 research/bad_days.json 커밋.")
    else:
        print("\n새로 등록할 불량일 없음")
    return 0


def promote() -> int:
    moved = _bad_days.promote(tracked=BAD, auto=AUTO)
    if moved:
        print(f"{len(moved)}개를 research/bad_days.json 으로 옮겼다: {', '.join(moved)} — 커밋할 것")
    else:
        print("옮길 자동 등록분 없음")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--promote"]:
        raise SystemExit(promote())
    raise SystemExit(main(sys.argv[1:]))
