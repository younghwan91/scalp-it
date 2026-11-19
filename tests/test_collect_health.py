"""수집 건강 점검 — **자동 등록이 사람이 적은 이유를 지우면 안 된다.**

2026-09-28~10-02 의 분커버가 37~52% 로 떨어졌는데 12일 뒤에 알았다. 사람이 안 보면 아무도 안 본다.
자동화하면서 기존 항목을 덮어쓰면 "왜 제외했는지"가 사라지고, 그러면 제외가 체리피킹과 구별되지 않는다.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod(tmp_bad: Path, tmp_auto: Path | None = None):
    spec = importlib.util.spec_from_file_location("collect_health", ROOT / "scripts" / "collect_health.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["collect_health"] = m
    spec.loader.exec_module(m)
    m.BAD = tmp_bad
    m.AUTO = tmp_auto if tmp_auto is not None else tmp_bad.with_name("auto.json")
    return m


def _auto_days(m) -> dict:
    return json.loads(m.AUTO.read_text())["days"] if m.AUTO.exists() else {}


def test_good_day_is_not_recorded(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"days": {}}), encoding="utf-8")
    m = _mod(bad)
    assert m.record("2026-10-01", {"cover": 0.97, "last_tick": "15:19:59"}, why="x") is False
    assert json.loads(bad.read_text())["days"] == {}
    assert _auto_days(m) == {}


def test_bad_day_is_recorded_with_reason(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"days": {}}), encoding="utf-8")
    m = _mod(bad)
    before = bad.read_bytes()
    assert m.record("2026-10-02", {"cover": 0.52, "last_tick": "15:18:58"}, why="자동") is True
    rec = _auto_days(m)["2026-10-02"]
    assert rec["cover"] == 0.52 and rec["why"] == "자동"
    # 크론은 추적 파일을 건드리지 않는다 — 운영 체크아웃이 더러워지면 07:50 동기화가 건너뛴다
    assert bad.read_bytes() == before


def test_existing_manual_entry_is_never_overwritten(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"days": {"2026-09-16": {"cover": 0.5, "last_tick": "15:19:28",
                                                       "why": "사람이 적은 이유"}}}), encoding="utf-8")
    m = _mod(bad)
    assert m.record("2026-09-16", {"cover": 0.5, "last_tick": "15:19:28"}, why="자동") is False
    assert json.loads(bad.read_text())["days"]["2026-09-16"]["why"] == "사람이 적은 이유"
    assert _auto_days(m) == {}


def test_threshold_matches_the_bad_days_document():
    doc = json.loads((ROOT / "research" / "bad_days.json").read_text(encoding="utf-8"))
    m = _mod(ROOT / "research" / "bad_days.json")
    assert m.THRESHOLD == 0.90
    assert "90" in doc["_measure"] or "0.9" in doc["_measure"]


@pytest.mark.parametrize("cover", [0.899, 0.0])
def test_boundary_is_inclusive_at_the_threshold(tmp_path, cover):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"days": {}}), encoding="utf-8")
    m = _mod(bad)
    assert m.record("2026-10-05", {"cover": cover, "last_tick": "x"}, why="자동") is True


def test_auto_entry_is_recorded_once(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"days": {}}), encoding="utf-8")
    m = _mod(bad)
    assert m.record("2026-11-19", {"cover": 0.868, "last_tick": "15:19:57"}, why="자동") is True
    assert m.record("2026-11-19", {"cover": 0.1, "last_tick": "x"}, why="다시") is False
    assert _auto_days(m)["2026-11-19"]["why"] == "자동"


def test_promote_moves_auto_entries_into_the_tracked_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"_measure": "90%", "days": {
        "2026-09-16": {"cover": 0.5, "last_tick": "x", "why": "사람"}}}), encoding="utf-8")
    m = _mod(bad)
    m.record("2026-11-19", {"cover": 0.868, "last_tick": "15:19:57"}, why="자동")
    assert m.promote() == 0
    doc = json.loads(bad.read_text())
    assert doc["_measure"] == "90%"  # 머리말 보존
    assert list(doc["days"]) == ["2026-09-16", "2026-11-19"]
    assert not m.AUTO.exists()
