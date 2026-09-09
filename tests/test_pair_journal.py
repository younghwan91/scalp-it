"""만쥬식 매매일지 생성기 테스트(§10-2).

만쥬는 일지를 실력 유지의 **필수 요건**으로 말한다 — "매매일지와 복기가 중단되면
실력 유지가 어렵고 언제든 무너질 수 있다". 그리고 형식이 구체적이다:

> **진입 이유** (예: "대장주 상한가 도달 시 2·3등주 짝꿍매매 진입")
> **손절/청산 사유** (예: "프로그램 매수를 보고 반등을 기대했으나 예상과 다른 하락으로
> 비중 축소 후 전량 손절")
> **주요 매매 종목 중심** — 모든 사소한 매매를 다 적기보다 큰 손익이 났거나 복기
> 가치가 높은 핵심 종목 위주로

우리는 로그가 전부 있는데 일지가 없었다. 로그는 기계가 읽는 형식이라 사람이 복기할
수 없다 — 그 간극을 메우는 게 이 모듈이다.
"""

import json

from scalp_it.journal import (
    JournalEntry,
    build_entries,
    render_journal,
    why_entered,
    why_exited,
)


def _alert(**kw):
    base = {
        "ts": "2026-08-26 10:15:00", "leader": "100000", "follower": "200000",
        "leader_name": "대장전자", "follower_name": "후속전자",
        "entry": 11000.0, "take_profit": None, "stop_loss": 10945.0,
        "leader_surge": 0.021, "cosurge": 0.008, "premove": 0.001,
        "buy_ratio": 0.62, "turnover": 5.2e9,
    }
    base.update(kw)
    return base


def _close(**kw):
    base = {
        "ts": "2026-08-26 10:17:30", "event": "close", "alert_ts": "2026-08-26 10:15:00",
        "leader": "100000", "follower": "200000", "follower_name": "후속전자",
        "entry": 11000.0, "exit_price": 11132.0, "exit_reason": "대장이탈",
        "status": "closed", "pnl": 0.0056, "pnl_krw": 560.0,
        "mae": -0.0021, "mfe": 0.0135, "filled": True,
    }
    base.update(kw)
    return base


# ----- 진입 이유 --------------------------------------------------------------

def test_why_entered_reads_like_manju_wrote_it():
    """진입 이유가 만쥬 일지 형식이어야 한다 — 사람이 읽고 복기할 수 있게."""
    # 감지기가 실제로 넣는 값은 영문 코드다(`Alert.trigger`). 한국어 이름으로
    # 검증하면 사전이 miss 나도 테스트가 초록이라 버그를 놓친다.
    text = why_entered(_alert(trigger="near"))
    assert "대장전자" in text and "후속전자" in text
    assert "상한가근접" in text, "코드가 사람 이름으로 풀려야 한다"
    # 숫자만 나열하면 복기가 안 된다 — 무슨 상황이었는지 문장이어야 한다.
    assert len(text) > 20


def test_why_entered_names_the_trigger_kind():
    """방아쇠 종류가 드러나야 한다 — 터치인지 근접인지가 복기의 핵심이다."""
    assert "상한가터치" in why_entered(_alert(trigger="touch"))
    assert "VI근접" in why_entered(_alert(trigger="vi"))
    assert "서지" in why_entered(_alert(trigger=None))     # 방아쇠 미기록이면 기존 서지


# ----- 청산 사유 --------------------------------------------------------------

def test_why_exited_explains_leader_exit_in_manju_terms():
    """대장이탈은 만쥬의 핵심 규칙이라 그 말로 적혀야 한다."""
    text = why_exited(_close(exit_reason="대장이탈"))
    assert "대장" in text and "매수 이유" in text


def test_why_exited_covers_each_reason():
    for reason in ("익절", "손절", "시간청산", "대장이탈", "미체결"):
        text = why_exited(_close(exit_reason=reason))
        assert text and len(text) > 5, f"{reason} 사유 문장이 없다"


# ----- 조립 ------------------------------------------------------------------

def test_build_entries_pairs_alert_with_close():
    entries = build_entries([_alert()], [_close()])
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e, JournalEntry)
    assert e.follower_name == "후속전자"
    assert e.pnl_krw == 560.0
    assert "대장" in e.why_exit


def test_build_entries_keeps_alert_without_close():
    """진입만 하고 아직 안 닫힌 것도 일지에 남아야 한다 — 장중에도 읽는다."""
    entries = build_entries([_alert()], [])
    assert len(entries) == 1
    assert entries[0].exit_reason == "미청산"


def test_build_entries_marks_filtered_signals():
    """배제된 신호도 복기 대상이다 — 왜 안 샀는지가 왜 샀는지만큼 중요하다."""
    notice = {"notice": "[10:20:00] 짝꿍 필터배제(min_sl_ticks)  대장(100000) → 후속(200000)"}
    entries = build_entries([_alert()], [], notices=[notice])
    assert any(e.exit_reason == "배제" for e in entries)


# ----- 렌더 ------------------------------------------------------------------

def test_render_journal_is_readable_and_leads_with_summary():
    entries = build_entries([_alert()], [_close()])
    out = render_journal(entries, day="2026-08-26", regime="극침체(4.69조)")
    assert "2026-08-26" in out
    assert "극침체" in out, "국면이 없으면 나중에 결과 해석이 안 된다"
    assert "후속전자" in out
    assert "대장" in out


def test_render_journal_handles_empty_day():
    """신호 0건인 날도 일지가 나와야 한다 — 만쥬 방아쇠는 하루 2~3건이라 흔하다."""
    out = render_journal([], day="2026-08-26", regime="극침체(4.69조)")
    assert "2026-08-26" in out
    assert "0" in out or "없" in out


def test_entries_serialize_to_jsonl():
    """며칠치를 기계적으로 합산할 수 있어야 한다 — 사람용 한 줄 텍스트만으론 부족하다."""
    entries = build_entries([_alert()], [_close()])
    line = json.loads(entries[0].to_json())
    assert line["follower"] == "200000"
    assert line["pnl_krw"] == 560.0
    assert "why_entry" in line and "why_exit" in line


def test_load_keeps_only_that_days_notices(tmp_path):
    """날짜 있는 통지는 그날 것만, 날짜 없는 옛 통지는 버린다 — 09-09 일지에 09-01~08 배제가 섞였다."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "pair_journal", Path(__file__).resolve().parents[1] / "scripts" / "pair_journal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    log = tmp_path / "alerts.log"
    log.write_text("\n".join([
        json.dumps({"ts": "2026-09-09 09:00:00", "follower": "1"}),
        json.dumps({"notice": "[09:01:00] 짝꿍 필터배제(roundtrip_bp) a(1) → b(2)", "day": "2026-09-09"}),
        json.dumps({"notice": "[09:02:00] 짝꿍 필터배제(roundtrip_bp) a(1) → c(3)", "day": "2026-09-08"}),
        json.dumps({"notice": "[09:03:00] 짝꿍 필터배제(roundtrip_bp) a(1) → d(4)"}),
    ]) + "\n", encoding="utf-8")
    recs, notices = mod._load(str(log), "2026-09-09")
    assert len(recs) == 1
    assert [n["notice"][:10] for n in notices] == ["[09:01:00]"]
