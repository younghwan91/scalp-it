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
    """기존 시나리오용 감지기. 100원짜리 가짜 가격이라 호가단위 필터는 꺼 둔다.

    (100원이면 1틱 1원 = 1%, 손절폭 −0.4% 는 0.4틱이라 실전이면 당연히 배제된다.
    이 헬퍼의 관심사는 서지·프리무브·쿨다운이므로 필터를 빼고 본다. 필터 자체는
    아래 '호가단위(틱) 손절폭 필터' 절에서 실제 가격대로 검증한다.)
    """
    cfg_kw.setdefault("min_sl_ticks", 0.0)
    return PairDetector([PAIR], DetectorConfig(**cfg_kw))


def _feed(det, code, secs, price, *, bid1=None, volume=None, side=1):
    return det.on_tick(code, BASE + timedelta(seconds=secs), float(price),
                       bid1=bid1, volume=volume, side=side)


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
    _feed(det, PAIR.follower, 130, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 160, 101.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.leader, 130, 100.0)
    a3 = _feed(det, PAIR.leader, 160, 102.0)
    assert len(a3) == 1


# --------------------------------------------------------------- 거래대금 하한

def test_min_turnover_floor():
    det = _det(min_turnover=1e9)  # 후속 누적 체결대금 10억 하한
    alerts = _base_timeline(det)  # 후속 누적 = 100*1000*3 = 30만 < 10억
    assert alerts == []


# --------------------------------------------------------- 매수체결비율(역선택 방어)

def _buyratio_timeline(det, *, s30, s60, v30=1000, v60=1000):
    """서지+리드랙은 통과시키되 후속의 t=30·t=60 체결 side/vol 만 바꿔 매수비율을 조절.

    창 [30,60] 안 체결로 매수비율 = (매수량)/(전체량) 이 결정된다.
    """
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1000, side=1)
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=v30, side=s30)
    _feed(det, PAIR.leader, 30, 100.0, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=v60, side=s60)
    return _feed(det, PAIR.leader, 60, 102.0, volume=1000)


def test_buy_ratio_passes_when_buys_dominate():
    det = _det(min_buy_ratio=0.30)
    alerts = _buyratio_timeline(det, s30=1, s60=1)   # 매수 100%
    assert len(alerts) == 1
    assert alerts[0].buy_ratio == pytest.approx(1.0)


def test_buy_ratio_blocks_when_sells_dominate():
    det = _det(min_buy_ratio=0.30)
    alerts = _buyratio_timeline(det, s30=-1, s60=-1)  # 매수 0% < 0.30
    assert alerts == []


def test_buy_ratio_threshold_boundary():
    # 창 안 매수 1000 / 전체 4000 = 0.25 < 0.30 → 차단
    det = _det(min_buy_ratio=0.30)
    blocked = _buyratio_timeline(det, s30=-1, s60=1, v30=3000, v60=1000)
    assert blocked == []
    # 0.50 ≥ 0.30 → 통과
    det2 = _det(min_buy_ratio=0.30)
    ok = _buyratio_timeline(det2, s30=-1, s60=1, v30=1000, v60=1000)
    assert len(ok) == 1
    assert ok[0].buy_ratio == pytest.approx(0.5)


def test_buy_ratio_warmup_skips_when_no_volume():
    # 후속 체결에 volume 이 없어 매수비율을 못 재면(워밍업) 안전 스킵(진입 안 함).
    det = _det(min_buy_ratio=0.30)
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0)          # volume 없음
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0)
    _feed(det, PAIR.leader, 30, 100.0, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0)
    alerts = _feed(det, PAIR.leader, 60, 102.0, volume=1000)
    assert alerts == []


def test_buy_ratio_toggle_off_lets_sells_through():
    # min_buy_ratio=0 → 필터 끔: 매도 우위여도 알림(기존 동작).
    det = _det(min_buy_ratio=0.0)
    alerts = _buyratio_timeline(det, s30=-1, s60=-1)
    assert len(alerts) == 1


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
    det = PairDetector([PAIR], DetectorConfig.aggressive(min_sl_ticks=0.0))  # 100원 가짜가라 필터 끔
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


# ------------------------------------------------- 호가단위(틱) 손절폭 필터

def test_tick_size_boundaries():
    """KRX 호가단위 — 경계값. scripts/limitup_timing.py 의 기존 정의와 값이 같아야 한다."""
    from scalp_it.ticksize import tick_size
    assert tick_size(1) == 1
    assert tick_size(1_999) == 1
    assert tick_size(2_000) == 5
    assert tick_size(4_999) == 5
    assert tick_size(5_000) == 10
    assert tick_size(19_999) == 10
    assert tick_size(20_000) == 50
    assert tick_size(49_999) == 50
    assert tick_size(50_000) == 100
    assert tick_size(199_999) == 100
    assert tick_size(200_000) == 500
    assert tick_size(499_999) == 500
    assert tick_size(500_000) == 1_000
    assert tick_size(1_000_000) == 1_000


def test_tick_size_matches_limitup_timing():
    """위임 후에도 반환값이 완전히 동일해야 한다(verify_* 재현성)."""
    import importlib.util

    from scalp_it.ticksize import tick_size
    spec = importlib.util.spec_from_file_location(
        "_limitup_timing", "scripts/limitup_timing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for px in [1, 500, 539, 1_999, 2_000, 4_999, 5_000, 6_230, 15_760, 19_999,
               20_000, 49_999, 50_000, 199_999, 200_000, 499_999, 500_000, 1_234_567]:
        assert mod.tick_size(px) == tick_size(px), px


def _fires_with_price(px, **cfg_kw) -> bool:
    """후속주 가격대만 바꿔 알림이 나오는지. cosurge/premove 는 비율이라 동일."""
    det = PairDetector([PAIR], DetectorConfig(**cfg_kw))   # 기본 min_sl_ticks=3.0 그대로
    alerts = _base_timeline(det, fol60=px, fol30=px, fol_now=px * 1.006, fol_bid=px)
    return bool(alerts)


def test_low_price_stock_filtered_by_tick_width():
    """539원: 1틱 1원 = 0.186%, 손절폭 −0.4% = 2.16틱 < 3틱 → 배제(노이즈 청산)."""
    assert not _fires_with_price(539.0)


def test_mid_price_stock_filtered_by_tick_width():
    """6230원: 1틱 10원, 손절폭 24.9원 = 2.49틱 < 3틱 → 배제."""
    assert not _fires_with_price(6230.0)


def test_high_tick_ratio_stock_passes():
    """15760원: 1틱 10원, 손절폭 63.0원 = 6.3틱 ≥ 3틱 → 통과."""
    assert _fires_with_price(15760.0)


def test_min_sl_ticks_zero_disables_filter():
    """0 이면 과거 동작 그대로 — 539원도 알림."""
    assert _fires_with_price(539.0, min_sl_ticks=0.0)


def test_aggressive_mode_uses_disaster_line_width():
    """일상 손절이 없으면 재난선(−0.7%) 폭으로 판정한다."""
    # 539원 × 0.7% = 3.77원 = 3.77틱 ≥ 3 → 통과
    assert _fires_with_price(539.0, take_profit=0.008, stop_loss=None, disaster_line=-0.007)
    # 3000원 × 0.7% = 21원, 1틱 5원 → 4.2틱 통과 / min 5틱이면 배제
    assert not _fires_with_price(539.0, take_profit=0.008, stop_loss=None,
                                 disaster_line=-0.007, min_sl_ticks=5.0)


def test_no_stop_falls_back_to_take_profit_width():
    """손절·재난선 둘 다 없으면 익절폭으로 판정."""
    # 539원 × 0.5% = 2.7원 = 2.7틱 < 3 → 배제
    assert not _fires_with_price(539.0, stop_loss=None, disaster_line=None)
    # 익절폭도 0 이면 잴 게 없으니 막지 않는다.
    assert _fires_with_price(539.0, stop_loss=None, disaster_line=None, take_profit=0.0)


def test_min_sl_ticks_default_is_three():
    assert DetectorConfig().min_sl_ticks == 3.0


def test_filtered_signal_is_recorded_not_silently_dropped():
    """배제 건은 세고, 왜 걸렀는지(가격·틱·손절폭 틱수) 남긴다."""
    det = PairDetector([PAIR], DetectorConfig())
    assert not _base_timeline(det, fol60=539.0, fol30=539.0, fol_now=539.0 * 1.006,
                              fol_bid=539.0)
    assert det.filtered_count == 1
    recs = det.drain_filtered()
    assert len(recs) == 1
    r = recs[0]
    assert r.follower == "000002"
    assert r.reason == "min_sl_ticks"
    assert r.tick == 1
    assert r.sl_ticks == pytest.approx(539.0 * 0.004, abs=0.01)
    assert r.min_sl_ticks == 3.0
    d = r.to_dict()
    assert d["filtered"] == "min_sl_ticks"
    assert "2.2틱" in r.to_line() and "000002" in r.to_line()
    assert det.drain_filtered() == []      # 비운 뒤엔 다시 안 나온다
    assert det.filtered_count == 1         # 누계는 유지(며칠 뒤 집계용)


# ----- 가속도 필터 (프로그램매매성 급등만 잡기) --------------------------------
#
# 감지기는 워밍업을 요구한다 — 대장은 surge_window(30s), 짝꿍은 cosurge+premove(60s)
# 만큼의 과거 가격이 있어야 판정이 돈다. 그래서 모든 시나리오는 BASE-60 부터 먹인다.

def _accel_det(**kw):
    """가속도 필터 검증용 — 다른 관문은 다 열어두고 가속만 본다."""
    cfg = DetectorConfig.stable(
        cosurge_min=0.0, premove_max=1.0, min_buy_ratio=0.0, min_sl_ticks=0.0,
        surge_window=30.0, surge_ret=0.015, **kw)
    return PairDetector([Pair(leader="000001", follower="000002")], cfg)


def _feed_series(det, code, series, *, vol=100.0):
    """``series`` = [(초, 가격)] 을 순서대로 흘리고 발생한 알림을 모아 돌려준다."""
    out = []
    for sec, px in series:
        out += det.on_tick(code, BASE + timedelta(seconds=sec), px,
                           bid1=px, volume=vol, side=1)
    return out


def _flat(lo, hi, px=100.0):
    return [(s, px) for s in range(lo, hi + 1)]


def _linear(lo, hi, *, start=100.0, per_sec=0.0533):
    """전 구간 같은 기울기 — 가속이 없는 상승."""
    return [(s, start + (s - lo) * per_sec) for s in range(lo, hi + 1)]


def test_accel_filter_rejects_steady_climb():
    """고르게 오른 대장은 가속이 없어 기각된다(30초에 걸친 야금야금 상승)."""
    det = _accel_det(accel_min_ratio=1.5)
    _feed_series(det, "000002", _flat(-60, 30))
    alerts = _feed_series(det, "000001", _linear(-60, 30))
    assert alerts == []


def test_accel_filter_accepts_late_burst():
    """잠잠하다가 마지막에 확 튀는 대장은 통과한다(프로그램매매 지문)."""
    det = _accel_det(accel_min_ratio=1.5)
    _feed_series(det, "000002", _flat(-60, 30))
    burst = [(25 + i, 100.0 + i * 0.6) for i in range(1, 6)]   # +3% 를 5초에
    alerts = _feed_series(det, "000001", _flat(-60, 25) + burst)
    assert alerts and alerts[-1].follower == "000002"


def test_accel_filter_off_by_default_keeps_old_behaviour():
    """기본값 0 이면 필터가 없다 — 고른 상승도 그대로 알림(기존 동작 보존)."""
    det = _accel_det()
    assert det.config.accel_min_ratio == 0.0 and det.config.volume_burst_ratio == 0.0
    _feed_series(det, "000002", _flat(-60, 30))
    alerts = _feed_series(det, "000001", _linear(-60, 30))
    assert alerts


def test_volume_burst_filter_rejects_thin_surge():
    """가격은 튀었지만 체결량이 안 늘면 기각(얇은 급등)."""
    det = _accel_det(volume_burst_ratio=3.0)
    _feed_series(det, "000002", _flat(-60, 30))
    burst = [(25 + i, 100.0 + i * 0.6) for i in range(1, 6)]
    alerts = _feed_series(det, "000001", _flat(-60, 25) + burst, vol=100.0)
    assert alerts == []


def test_volume_burst_filter_accepts_volume_explosion():
    """같은 급등이어도 체결량이 쏟아지면 통과."""
    det = _accel_det(volume_burst_ratio=3.0)
    _feed_series(det, "000002", _flat(-60, 30))
    _feed_series(det, "000001", _flat(-60, 25), vol=100.0)      # 평소 체결량
    burst = [(25 + i, 100.0 + i * 0.6) for i in range(1, 6)]
    alerts = _feed_series(det, "000001", burst, vol=1000.0)     # 10배 폭발
    assert alerts and alerts[-1].follower == "000002"
