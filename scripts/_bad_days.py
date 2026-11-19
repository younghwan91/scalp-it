"""불량 거래일 목록의 유일한 정문 — 읽기·자동 등록·승격.

두 파일로 나눈다.

* ``research/bad_days.json`` (git 추적): 사람이 적었거나 승격해 커밋한 항목. 연구의 원본이다.
* ``data/bad_days_auto.json`` (추적 안 함): 16:10 크론(``collect_health.py``)이 자동으로 올린 항목.

예전엔 크론이 추적 파일에 바로 썼다. 그러면 운영 체크아웃(``~/git/scalp-it``)이 더러워져
07:50 ``pull-all`` 이 "uncommitted changes" 로 그 레포 동기화를 건너뛴다 — main 에 올린
수정이 라이브에 안 들어간다(2026-11-19 발견). 크론은 이제 추적 파일을 건드리지 않는다.

읽을 때는 둘을 합친다. 같은 날짜가 둘 다 있으면 추적 파일(사람)이 이긴다.
표준 라이브러리만 쓴다 — 연구 스크립트 일부가 ``scalp_it`` 없는 전용 venv 로 돈다.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKED = ROOT / "research" / "bad_days.json"
AUTO = ROOT / "data" / "bad_days_auto.json"


def _days(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("days", {}) or {}
    except (OSError, ValueError, AttributeError):
        return {}


def load(*, tracked: Path = TRACKED, auto: Path = AUTO) -> dict:
    """날짜 → {cover, last_tick, why}. 자동 항목 위에 추적 항목을 덮어 합친다."""
    merged = dict(_days(auto))
    merged.update(_days(tracked))
    return dict(sorted(merged.items()))


def record_auto(day: str, entry: dict, *, tracked: Path = TRACKED, auto: Path = AUTO) -> bool:
    """자동 항목을 ``data/bad_days_auto.json`` 에 올린다. 이미 어느 쪽에든 있으면 건드리지 않는다."""
    if day in _days(tracked) or day in _days(auto):
        return False
    doc = {"_why": f"collect_health 자동 등록분. 연구 원본({TRACKED.relative_to(ROOT)})으로 옮기려면 "
                   "`scripts/collect_health.py --promote` 후 커밋한다.",
           "days": dict(sorted({**_days(auto), day: entry}.items()))}
    auto.parent.mkdir(parents=True, exist_ok=True)
    tmp = auto.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(auto)
    return True


def promote(*, tracked: Path = TRACKED, auto: Path = AUTO) -> list[str]:
    """자동 항목을 추적 파일로 옮긴다(사람이 커밋하기 전 단계). 추적 쪽에 있던 날짜는 건드리지 않는다."""
    pending = _days(auto)
    doc = json.loads(tracked.read_text(encoding="utf-8"))
    days = doc.setdefault("days", {})
    moved = [d for d in sorted(pending) if d not in days]
    for d in moved:
        days[d] = pending[d]
    doc["days"] = dict(sorted(days.items()))
    tracked.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if auto.exists():
        auto.unlink()
    return moved
