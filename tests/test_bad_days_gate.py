"""불량일 목록은 ``scripts/_bad_days.py`` 한 곳으로만 읽고 쓴다.

예전엔 다섯 스크립트가 ``research/bad_days.json`` 을 각자 읽었고, 16:10 크론이 그 추적 파일에
바로 썼다. 운영 체크아웃이 더러워지면 07:50 pull-all 이 "uncommitted changes" 로 동기화를
건너뛴다. 자동 등록분은 ``data/bad_days_auto.json``(추적 안 함)에 쓰고, 읽기는 둘을 합친다.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("_bad_days_t", ROOT / "scripts" / "_bad_days.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_only_the_gate_names_the_tracked_file():
    """새 스크립트가 추적 파일을 직접 읽으면 자동 등록분을 놓친다(그리고 쓰면 체크아웃이 더러워진다)."""
    out = subprocess.run(["git", "grep", "-n", "-e", '"bad_days.json"', "-e", "bad_days.json\"",
                          "--", "scripts", "src"], cwd=ROOT, capture_output=True, text=True, check=False)
    hits = [ln for ln in out.stdout.splitlines() if not ln.startswith("scripts/_bad_days.py:")]
    assert not hits, "research/bad_days.json 은 scripts/_bad_days.py 로만 읽는다:\n" + "\n".join(hits)


def test_auto_file_is_not_tracked():
    """자동 등록분이 git 에 추적되면 같은 사고(더러운 운영 체크아웃)가 되풀이된다."""
    m = _mod()
    rel = m.AUTO.relative_to(ROOT).as_posix()
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT, check=False)
    assert r.returncode == 0, f"{rel} 가 .gitignore 에 걸리지 않는다"


def test_load_merges_and_tracked_wins(tmp_path):
    m = _mod()
    tracked = tmp_path / "t.json"
    auto = tmp_path / "a.json"
    tracked.write_text(json.dumps({"days": {"2026-09-16": {"cover": 0.5, "why": "사람"}}}), encoding="utf-8")
    auto.write_text(json.dumps({"days": {"2026-09-16": {"cover": 0.4, "why": "자동"},
                                         "2026-11-19": {"cover": 0.868, "why": "자동"}}}), encoding="utf-8")
    got = m.load(tracked=tracked, auto=auto)
    assert list(got) == ["2026-09-16", "2026-11-19"]
    assert got["2026-09-16"]["why"] == "사람"


def test_load_without_auto_file_is_the_tracked_list(tmp_path):
    m = _mod()
    got = m.load(auto=tmp_path / "missing.json")
    assert got == json.loads(m.TRACKED.read_text(encoding="utf-8"))["days"]
