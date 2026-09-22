"""불량 거래일 목록 — **부분 수집된 날이 연구 표본에 섞이면 조용히 편향된다.**

2026-09-22 발견: 08-31 은 11:40 에 수집이 죽었고(분커버 41%), 09-16·09-17 은 trader 에서
연구 부하(창 커널 벤치마크·RL 그림자 8종목)를 돌려 50% 대로 떨어졌다. **원본은 지우지 않는다** —
부분 데이터도 관측이고 지우면 되돌릴 수 없다. 캐시 생성에서 빼는 것으로 처리한다.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD = ROOT / "research" / "bad_days.json"


def test_file_exists_and_parses():
    d = json.loads(BAD.read_text(encoding="utf-8"))
    assert d["days"], "불량일 목록이 비었다"


def test_every_entry_states_cover_and_reason():
    d = json.loads(BAD.read_text(encoding="utf-8"))["days"]
    for day, rec in d.items():
        assert len(day) == 10 and day[4] == "-", f"날짜 형식: {day}"
        assert 0 < rec["cover"] < 1, f"{day} 분커버가 비율이 아니다"
        assert rec["why"], f"{day} 이유가 없다 — 이유 없는 제외는 체리피킹이다"


def test_threshold_is_documented_and_consistent():
    d = json.loads(BAD.read_text(encoding="utf-8"))
    assert "0.9" in d["_measure"] or "90" in d["_measure"], "문턱이 문서에 없다"
    for day, rec in d["days"].items():
        assert rec["cover"] < 0.9, f"{day} 는 문턱 위인데 목록에 있다"


def test_flow83_skips_bad_days():
    """캐시 생성기가 실제로 이 목록을 본다 — 목록만 있고 안 쓰면 아무 소용 없다."""
    src = (ROOT / "scripts" / "flow_83" / "features.py").read_text(encoding="utf-8")
    assert "bad_days" in src, "flow_83 이 불량일 목록을 읽지 않는다"
