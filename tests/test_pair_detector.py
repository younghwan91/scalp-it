# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""짝꿍 감지 로직 단위테스트 — mock 틱으로 서지·리드랙(프리무브) 필터·쿨다운·모드.

감지기는 DB·시계를 모르므로 오름차순 틱만 먹여 순수 판정을 고정한다.
"""

import json
from datetime import datetime, timedelta

import pytest

from scalp_it.realtime.alerts import AlertBus, LogFileSink
from scalp_it.realtime.pair_detector import (
    DetectorConfig,
    Pair,
    PairDetector,
    load_watchlist,
    parse_pairs_arg,
)

BASE = datetime(2026, 8, 24, 9, 10, 0)
PAIR = Pair(leader="000001", follower="000002", leader_name="대장", follower_name="후속")


def _det(**cfg_kw) -> PairDetector:
    return PairDetector([PAIR], DetectorConfig(**cfg_kw))


def _feed(det, code, secs, price, *, bid1=None, volume=None):
    return det.on_tick(code, BASE + timedelta(seconds=secs), float(price),
                       bid1=bid1, volume=volume)


def _base_timeline(det, *, fol60=100.0, fol30=100.0, fol_now=100.6,
                   lead30=100.0, lead_now=102.0, fol_bid=100.0):
    """서지 순간(t=60) 직전 히스토리를 채우고, 마지막 대장 틱의 알림을 돌려준다."""
    _feed(det, PAIR.follower, 0, fol60, bid1=fol_bid, volume=1000)
    _feed(det, PAIR.follower, 30, fol30, bid1=fol_bid, volume=1000)
    _feed(det, PAIR.leader, 30, lead30, volume=1000)
    _feed(det, PAIR.follower, 60, fol_now, bid1=fol_bid, volume=1000)
    return _feed(det, PAIR.leader, 60, lead_now, volume=1000)


# --------------------------------------------------------------- 서지 + 매칭

def test_surge_with_leadlag_fires_alert():
    det = _det()
    alerts = _base_timeline(det)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.leader == "000001" and a.follower == "000002"
    assert a.leader_surge == pytest.approx(0.02)
    assert a.cosurge == pytest.approx(0.006)
    assert a.premove == pytest.approx(0.0)
    assert a.entry == pytest.approx(100.0)          # bid1 지정가
    assert a.take_profit == pytest.approx(100.5)    # +0.5% 안정형 기본
    assert a.stop_loss == pytest.approx(99.6)       # −0.4%
    assert a.disaster_line is None
    assert a.exit_ts == BASE + timedelta(seconds=150)  # +90s


def test_no_alert_when_leader_below_surge():
    det = _det()
    alerts = _base_timeline(det, lead_now=100.5)  # +0.5% < 1.5%
    assert alerts == []


def test_premove_filter_rejects_already_moved_follower():
    # 후속이 직전 30초에 이미 +1% 뛰었으면(추격) 기각.
    det = _det()
    alerts = _base_timeline(det, fol30=101.0, fol_now=101.5)
    # premove = 101/100 = +1% > 0.2% → 기각
    assert alerts == []


def test_cosurge_required():
    # 후속이 대장과 같이 뛰지 않으면(동반급등 < 0.5%) 기각.
    det = _det()
    alerts = _base_timeline(det, fol_now=100.1)  # cosurge +0.1% < 0.5%
    assert alerts == []


def test_unregistered_leader_ignored():
    det = _det()
    _feed(det, "999999", 30, 100.0)
    alerts = _feed(det, "999999", 60, 101.5)  # 서지지만 등록 안 된 대장
    assert alerts == []


# --------------------------------------------------------------- 쿨다운

def test_cooldown_suppresses_repeat():
    det = _det(cooldown=90.0)
    a1 = _base_timeline(det)
    assert len(a1) == 1
    # 60초 뒤 또 서지 — 쿨다운(90s) 안이라 억제
    _feed(det, PAIR.follower, 90, 100.0, bid1=100.0)
    _feed(det, PAIR.follower, 120, 101.0, bid1=100.0)   # cosurge +1%
    _feed(det, PAIR.leader, 90, 100.0)
    a2 = _feed(det, PAIR.leader, 120, 102.0)            # 서지, 하지만 t=120 < 60+90
    assert a2 == []
    # 쿨다운 지난 뒤(t=160)엔 다시 알림
    _feed(det, PAIR.follower, 130, 100.0, bid1=100.0)
    _feed(det, PAIR.follower, 160, 101.0, bid1=100.0)
    _feed(det, PAIR.leader, 130, 100.0)
    a3 = _feed(det, PAIR.leader, 160, 102.0)
    assert len(a3) == 1


# --------------------------------------------------------------- 거래대금 하한

def test_min_turnover_floor():
    det = _det(min_turnover=1e9)  # 후속 누적 체결대금 10억 하한
    alerts = _base_timeline(det)  # 후속 누적 = 100*1000*3 = 30만 < 10억
    assert alerts == []


# --------------------------------------------------------------- 워밍업 부족

def test_insufficient_history_skips():
    det = _det()
    # 후속 프리무브 창(t-60) 가격이 없다 → 스킵
    _feed(det, PAIR.follower, 40, 100.0, bid1=100.0)
    _feed(det, PAIR.leader, 30, 100.0)
    _feed(det, PAIR.follower, 60, 100.5, bid1=100.0)
    alerts = _feed(det, PAIR.leader, 60, 101.5)
    assert alerts == []


# --------------------------------------------------------------- 모드

def test_stable_vs_aggressive_config():
    stable = DetectorConfig.stable()
    assert stable.take_profit == pytest.approx(0.005)
    assert stable.stop_loss == pytest.approx(-0.004)
    assert stable.disaster_line is None

    agg = DetectorConfig.aggressive()
    assert agg.take_profit == pytest.approx(0.008)
    assert agg.stop_loss is None
    assert agg.disaster_line == pytest.approx(-0.007)


def test_aggressive_alert_has_disaster_line():
    det = PairDetector([PAIR], DetectorConfig.aggressive())
    a = _base_timeline(det)[0]
    assert a.stop_loss is None
    assert a.disaster_line == pytest.approx(99.3)   # −0.7% of 100
    assert a.take_profit == pytest.approx(100.8)    # +0.8%
    assert "재난선" in a.to_line()


# --------------------------------------------------------------- 커버리지 경고

def test_missing_followers():
    det = _det()
    assert det.codes == {"000001", "000002"}
    missing = det.missing_followers({"000001"})   # 후속 구독 안 됨
    assert [p.follower for p in missing] == ["000002"]
    assert det.missing_followers({"000001", "000002"}) == []


# --------------------------------------------------------------- 로딩

def test_parse_pairs_arg():
    pairs = parse_pairs_arg("014990:033340, 462860:094480")
    assert pairs[0] == Pair(leader="014990", follower="033340")
    assert pairs[1].follower == "094480"
    with pytest.raises(ValueError):
        parse_pairs_arg("badformat")


def test_load_watchlist(tmp_path):
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"pairs": [
        {"leader": "014990", "follower": "033340", "leader_name": "인디에프"},
    ]}), encoding="utf-8")
    pairs = load_watchlist(p)
    assert pairs == [Pair(leader="014990", follower="033340", leader_name="인디에프")]


# --------------------------------------------------------------- 알림 채널

def test_logfile_sink_writes(tmp_path):
    det = _det()
    alert = _base_timeline(det)[0]
    log = tmp_path / "alerts.log"
    bus = AlertBus([LogFileSink(log, echo=False)])
    bus.emit(alert)
    line = log.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["leader"] == "000001"
    assert record["entry"] == 100.0
    assert bus.count == 1


def test_dry_run_bus_writes_nothing(tmp_path):
    det = _det()
    alert = _base_timeline(det)[0]
    log = tmp_path / "alerts.log"
    bus = AlertBus([LogFileSink(log, echo=False)], dry_run=True)
    bus.emit(alert)
    assert not log.exists()
    assert bus.count == 1
