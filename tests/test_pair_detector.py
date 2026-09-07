# ruff: noqa: DTZ001 — 테스트 datetime 은 KST naive(프로젝트 표준)
"""짝꿍 감지 로직 단위테스트 — mock 틱으로 서지·리드랙(프리무브) 필터·쿨다운·모드.

감지기는 DB·시계를 모르므로 오름차순 틱만 먹여 순수 판정을 고정한다.
"""

import json
from datetime import datetime, timedelta

import pytest

from scalp_it.realtime.alerts import AlertBus, LogFileSink
from scalp_it.realtime.pair_detector import (
    Alert,
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


# ----- 만쥬 손익 구조 (익절 상한 없음) ----------------------------------------

def test_take_profit_can_be_disabled():
    """익절 상한을 끌 수 있어야 한다 — 만쥬 손익 구조.

    원본은 "짧은 손절(−0.5%)로 방어하고 수익은 슈팅·상한가까지 열어둔다". 상한을
    걸면 전부를 먹여살릴 우측 꼬리를 잘라낸다. 1년 240조합 스윕에서도 익절을
    풀수록 net 이 단조 증가했다(TP 0.3% → 익절없음, hold 30 기준 +5.6bp).
    """
    cfg = DetectorConfig.stable(take_profit=None, stop_loss=-0.005)
    assert cfg.take_profit is None and cfg.stop_loss == -0.005


def test_alert_without_take_profit_renders_and_serializes():
    """익절 없는 알림도 사람이 읽는 줄과 JSON 으로 나가야 한다."""
    det = _accel_det(take_profit=None, stop_loss=-0.005)
    _feed_series(det, "000002", _flat(-60, 30))
    burst = [(25 + i, 100.0 + i * 0.6) for i in range(1, 6)]
    alerts = _feed_series(det, "000001", _flat(-60, 25) + burst)
    assert alerts, "익절 없음 설정에서도 알림이 나가야 한다"
    a = alerts[-1]
    assert a.take_profit is None
    assert "익절 없음" in a.to_line()
    assert a.to_dict()["take_profit"] is None


# ----- 상한가 방아쇠 (만쥬 문서 44 §5-1) --------------------------------------
#
# 원자료가 터치인지 잠김인지 확정되지 않아(§11 #4) 셋을 각각 켜고 끌 수 있어야 한다.

def test_limit_up_price_matches_limitup_timing():
    """상한가 계산은 scripts/limitup_timing.py 와 **값이 한 치도** 달라선 안 된다.

    그 스크립트로 뽑은 상한가 검증 결과와 실시간 판정이 어긋나면 백테스트가 무의미하다.
    """
    import importlib.util

    from scalp_it.realtime.pair_detector import limit_up_price
    spec = importlib.util.spec_from_file_location(
        "_limitup_timing", "scripts/limitup_timing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for prev in [100, 539, 1_000, 1_538, 1_539, 2_000, 3_846, 6_230, 15_760, 19_999,
                 20_000, 38_461, 49_999, 50_000, 153_846, 199_999, 384_615, 1_234_567]:
        assert limit_up_price(prev) == mod.limit_up_price(prev), prev


def _limit_det(prev=10_000.0, **cfg_kw):
    """대장 전일종가 1만원(상한가 12,990원) 짜리 감지기. 다른 관문은 다 열어둔다."""
    cfg_kw.setdefault("min_sl_ticks", 0.0)
    cfg_kw.setdefault("cosurge_min", 0.0)
    cfg_kw.setdefault("premove_max", 1.0)
    cfg_kw.setdefault("min_buy_ratio", 0.0)
    return PairDetector([PAIR], DetectorConfig(**cfg_kw),
                        prev_close={PAIR.leader: prev, PAIR.follower: 100.0})


def _limit_timeline(det, lead_now, *, lead30=12_000.0):
    """후속 히스토리를 채우고 대장의 마지막 틱(가격 ``lead_now``)이 낸 알림을 준다."""
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.leader, 30, lead30, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=1000)
    return _feed(det, PAIR.leader, 60, lead_now, volume=1000)


def test_limit_up_price_value():
    from scalp_it.realtime.pair_detector import limit_up_price
    assert limit_up_price(10_000) == 13_000        # 13,000 은 1틱 10원의 배수
    assert limit_up_price(539) == 700              # 700.7 → 1틱 1원 내림
    assert limit_up_price(1_539) == 2_000          # 2000.7 → 1틱 5원 내림(경계)


def test_touch_trigger_fires_without_surge():
    """상한가에 닿으면 서지(+1.5%/30s) 없이도 방아쇠 — 서지를 '대체'한다."""
    det = _limit_det(limitup_touch=True)
    # 12,990 → 12,990: 30초 수익률 0% 라 기존 서지 방아쇠로는 절대 안 터진다.
    alerts = _limit_timeline(det, 13_000.0, lead30=13_000.0)
    assert len(alerts) == 1
    assert alerts[0].trigger == "touch"
    assert alerts[0].leader_limit == pytest.approx(13_000.0)
    assert alerts[0].leader_ret == pytest.approx(0.30, abs=1e-4)


def test_touch_trigger_not_fired_below_limit():
    det = _limit_det(limitup_touch=True)
    assert _limit_timeline(det, 12_990.0, lead30=12_990.0) == []


def test_near_trigger_fires_in_run_up_zone():
    """'20%대 중후반에서 치고 들어가는' 구간 — 상한가까지 4% 이내."""
    det = _limit_det(limitup_near_pct=0.04)
    # 12,500 / 12,990 = 상한가의 96.2% → 4% 이내 → 발동
    a = _limit_timeline(det, 12_500.0, lead30=12_500.0)
    assert len(a) == 1 and a[0].trigger == "near"
    # 12,400 = 상한가의 95.5% → 4% 밖 → 미발동
    assert _limit_timeline(_limit_det(limitup_near_pct=0.04), 12_400.0,
                           lead30=12_400.0) == []


def test_locked_trigger_needs_continuous_hold():
    """상한가 M초 연속 유지(잠김 근사) — 중간에 풀리면 시계가 되감긴다."""
    det = _limit_det(limitup_lock_secs=20.0)
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=1000)
    _feed(det, PAIR.leader, 30, 13_000.0, volume=1000)     # 잠김 시작
    assert _feed(det, PAIR.leader, 45, 13_000.0, volume=1000) == []   # 15초 < 20초
    assert _feed(det, PAIR.leader, 46, 12_990.0, volume=1000) == []   # 풀림 — 되감기
    assert _feed(det, PAIR.leader, 47, 13_000.0, volume=1000) == []   # 다시 시작
    assert _feed(det, PAIR.leader, 60, 13_000.0, volume=1000) == []   # 13초 < 20초
    a = _feed(det, PAIR.leader, 68, 13_000.0, volume=1000)            # 21초 ≥ 20초
    assert len(a) == 1 and a[0].trigger == "locked"


def test_limitup_skips_stock_without_prev_close():
    """전일종가를 못 받은 종목은 상한가를 모른다 — 억지 추정 없이 안전 스킵."""
    det = PairDetector([PAIR], DetectorConfig(
        limitup_touch=True, min_sl_ticks=0.0, cosurge_min=0.0,
        premove_max=1.0, min_buy_ratio=0.0))            # prev_close 미주입
    assert _limit_timeline(det, 13_000.0, lead30=13_000.0) == []


def test_limitup_require_surge_adds_surge_condition():
    """옵션을 켜면 상한가 방아쇠에 서지까지 AND 로 요구한다."""
    flat = _limit_det(limitup_touch=True, limitup_require_surge=True)
    assert _limit_timeline(flat, 13_000.0, lead30=13_000.0) == []      # 30초 수익 0%
    rush = _limit_det(limitup_touch=True, limitup_require_surge=True)
    assert len(_limit_timeline(rush, 13_000.0, lead30=12_000.0)) == 1  # +8.25%


def test_limitup_off_by_default_keeps_surge_trigger():
    """기본값은 셋 다 꺼짐 = 기존 서지 방아쇠 그대로(하위호환)."""
    cfg = DetectorConfig()
    assert cfg.limitup_touch is False
    assert cfg.limitup_near_pct == 0.0 and cfg.limitup_lock_secs == 0.0
    det = _det()
    assert len(_base_timeline(det)) == 1
    assert _base_timeline(_det(), lead_now=100.5) == []


# ----- 호가 두께 필터 (만쥬 최우선) --------------------------------------------

def _book(px, qty, n=10, step=0):
    """``n`` 단계 호가 — 각 단계 ``qty`` 주. ``step`` 만큼 가격이 멀어진다."""
    return [(px + i * step, qty) for i in range(n)]


def test_roundtrip_bp_thick_book_is_cheap():
    """두꺼운 호가: 1000만원이 1호가에서 다 체결 → 충격은 스프레드뿐."""
    from scalp_it.realtime.pair_detector import roundtrip_bp
    bids = _book(9_990.0, 10_000, step=-10)
    asks = _book(10_000.0, 10_000, step=10)
    bp = roundtrip_bp(bids, asks, 10_000_000)
    # 스프레드 10원 / mid 9995 = 10.0bp
    assert bp == pytest.approx(10.0, abs=0.1)


def test_roundtrip_bp_thin_book_is_expensive():
    """얇은 호가: 상단까지 긁어 올라가며 VWAP 이 밀린다."""
    from scalp_it.realtime.pair_detector import roundtrip_bp
    bids = _book(9_990.0, 200, step=-10)
    asks = _book(10_000.0, 200, step=10)
    thin = roundtrip_bp(bids, asks, 10_000_000)
    assert thin > 50.0


def test_roundtrip_bp_insufficient_depth_is_inf():
    """10호가를 다 긁어도 주문금액을 못 채우면 무한대 = 즉시 배제 대상."""
    import math

    from scalp_it.realtime.pair_detector import roundtrip_bp
    bids = _book(1_000.0, 10, step=-1)      # 10단계 × 1만원 = 10만원뿐
    asks = _book(1_001.0, 10, step=1)
    assert math.isinf(roundtrip_bp(bids, asks, 10_000_000))


def _depth_det(**cfg_kw):
    cfg_kw.setdefault("min_sl_ticks", 0.0)
    return PairDetector([PAIR], DetectorConfig(**cfg_kw))


def _depth_timeline(det, bids, asks):
    """후속주에 10호가를 넣고 평소 시나리오를 돌린다(후속가 1만원대)."""
    px = 10_000.0
    for sec in (0, 30):
        det.on_quote(PAIR.follower, BASE + timedelta(seconds=sec), bids=bids, asks=asks)
        _feed(det, PAIR.follower, sec, px, volume=1000)
    _feed(det, PAIR.leader, 30, 100.0, volume=1000)
    det.on_quote(PAIR.follower, BASE + timedelta(seconds=60), bids=bids, asks=asks)
    _feed(det, PAIR.follower, 60, px * 1.006, volume=1000)
    return _feed(det, PAIR.leader, 60, 102.0, volume=1000)


def test_thick_book_passes_depth_filter():
    det = _depth_det()
    alerts = _depth_timeline(det, _book(9_990.0, 10_000, step=-10),
                             _book(10_000.0, 10_000, step=10))
    assert len(alerts) == 1
    assert alerts[0].rt_bp == pytest.approx(10.0, abs=0.1)


def test_thin_book_rejected_and_recorded():
    det = _depth_det()
    alerts = _depth_timeline(det, _book(9_990.0, 200, step=-10),
                             _book(10_000.0, 200, step=10))
    assert alerts == []
    recs = det.drain_filtered()
    assert len(recs) == 1 and recs[0].reason == "roundtrip_bp"
    assert recs[0].rt_bp > 20.0
    assert recs[0].order_krw == 10_000_000
    assert recs[0].to_dict()["filtered"] == "roundtrip_bp"
    assert "왕복충격" in recs[0].to_line()


def test_insufficient_depth_rejected_as_depth_short():
    """10호가로 못 채우는 종목은 별도 사유로 기록한다(만쥬: 즉시 제외)."""
    det = _depth_det()
    alerts = _depth_timeline(det, _book(9_990.0, 1, step=-10), _book(10_000.0, 1, step=10))
    assert alerts == []
    recs = det.drain_filtered()
    assert len(recs) == 1 and recs[0].reason == "depth_short"
    assert "못 채움" in recs[0].to_line()


def test_depth_filter_without_quote_data_passes():
    """10호가를 안 넣어주던 기존 호출부는 그대로 동작해야 한다(잴 수 없으면 통과)."""
    det = _det()                       # 호가 스냅샷 없음, max_roundtrip_bp 기본 20
    assert det.config.max_roundtrip_bp == 20.0
    assert len(_base_timeline(det)) == 1


def test_depth_filter_off_by_zero():
    det = _depth_det(max_roundtrip_bp=0.0)
    alerts = _depth_timeline(det, _book(9_990.0, 200, step=-10),
                             _book(10_000.0, 200, step=10))
    assert len(alerts) == 1


def test_depth_filter_runs_before_tick_width():
    """두 필터가 동시에 걸리면 만쥬 최우선인 두께 쪽 사유로 기록된다."""
    det = PairDetector([PAIR], DetectorConfig())     # min_sl_ticks 3.0 도 켜짐
    alerts = _depth_timeline(det, _book(9_990.0, 200, step=-10),
                             _book(10_000.0, 200, step=10))
    assert alerts == []
    recs = det.drain_filtered()
    assert [r.reason for r in recs] == ["roundtrip_bp"]


def test_on_quote_keeps_positional_bid1():
    """기존 호출부 시그니처 on_quote(code, ts, bid1) 가 그대로 살아 있어야 한다."""
    det = _det()
    det.on_quote(PAIR.follower, BASE, 99.0)
    _feed(det, PAIR.follower, 0, 100.0, volume=1000)
    _feed(det, PAIR.follower, 30, 100.0, volume=1000)
    _feed(det, PAIR.leader, 30, 100.0, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, volume=1000)
    a = _feed(det, PAIR.leader, 60, 102.0, volume=1000)
    assert a and a[0].entry == pytest.approx(99.0)


# ----- 동적 대장/후속 선정 -----------------------------------------------------

UNIV = ["100001", "100002", "100003", "100004", "100005"]
SECTORS = dict.fromkeys(UNIV[:4], "로봇") | {"100005": "방산"}
PREV = dict.fromkeys(UNIV, 10_000.0)


def _dyn_det(**cfg_kw):
    cfg_kw.setdefault("min_sl_ticks", 0.0)
    cfg_kw.setdefault("cosurge_min", 0.0)
    cfg_kw.setdefault("premove_max", 1.0)
    cfg_kw.setdefault("min_buy_ratio", 0.0)
    cfg_kw.setdefault("limitup_touch", True)
    return PairDetector([], DetectorConfig(**cfg_kw),
                        prev_close=PREV, sectors=SECTORS, universe=UNIV,
                        names={"100001": "대장", "100002": "2등"})


def _dyn_timeline(det, prices):
    """``prices`` = {코드: (t0가격, t60가격)}. 마지막에 대장(100001) 틱을 흘린다."""
    for code, (p0, p60) in prices.items():
        _feed(det, code, 0, p0, bid1=p0, volume=1000)
        _feed(det, code, 30, p0, bid1=p0, volume=1000)
        _feed(det, code, 60, p60, bid1=p60, volume=1000)
    return _feed(det, "100001", 61, 13_000.0, volume=1000)


def test_dynamic_mode_picks_ranks_two_and_three():
    """대장은 방아쇠가 터진 종목, 후속은 같은 섹터 등락률 2~3위."""
    det = _dyn_det()
    alerts = _dyn_timeline(det, {
        "100001": (12_000.0, 13_000.0),   # +29.9% 1위 = 대장
        "100002": (11_000.0, 11_100.0),   # +11.0% 2위
        "100003": (10_500.0, 10_600.0),   # +6.0%  3위
        "100004": (10_100.0, 10_150.0),   # +1.5%  4위 — 제외
        "100005": (12_500.0, 12_800.0),   # 다른 섹터 — 제외
    })
    assert sorted(a.follower for a in alerts) == ["100002", "100003"]
    assert all(a.leader == "100001" and a.trigger == "touch" for a in alerts)
    assert [a.follower_name for a in alerts if a.follower == "100002"] == ["2등"]


def test_dynamic_rank_range_is_configurable():
    det = _dyn_det(follower_rank_min=2, follower_rank_max=2)
    alerts = _dyn_timeline(det, {
        "100001": (12_000.0, 13_000.0),
        "100002": (11_000.0, 11_100.0),
        "100003": (10_500.0, 10_600.0),
        "100004": (10_100.0, 10_150.0),
    })
    assert [a.follower for a in alerts] == ["100002"]


def test_dynamic_excludes_other_sectors():
    det = _dyn_det()
    alerts = _dyn_timeline(det, {
        "100001": (12_000.0, 13_000.0),
        "100005": (10_100.0, 10_200.0),   # 유일한 다른 섹터 종목
    })
    assert alerts == []                    # 같은 섹터에 2~3위가 없다


def test_fixed_pairs_still_work_alongside_dynamic():
    """고정 쌍 모드는 하위호환 — 동적 모드를 켜도 함께 돈다."""
    cfg = DetectorConfig(min_sl_ticks=0.0, cosurge_min=0.0, premove_max=1.0,
                         min_buy_ratio=0.0, limitup_touch=True)
    det = PairDetector([Pair(leader="100001", follower="100004")], cfg,
                       prev_close=PREV, sectors=SECTORS, universe=UNIV)
    alerts = _dyn_timeline(det, {
        "100001": (12_000.0, 13_000.0),
        "100002": (11_000.0, 11_100.0),
        "100003": (10_500.0, 10_600.0),
        "100004": (10_100.0, 10_150.0),
    })
    # 고정 쌍의 100004(4위) + 동적 2~3위
    assert sorted(a.follower for a in alerts) == ["100002", "100003", "100004"]


def test_dynamic_mode_off_without_sectors():
    """유니버스만 주면 동적 모드는 안 켜진다(섹터 없이 후속을 못 고른다)."""
    det = PairDetector([], DetectorConfig(limitup_touch=True, min_sl_ticks=0.0),
                       prev_close=PREV, universe=UNIV)
    assert _dyn_timeline(det, {"100001": (12_000.0, 13_000.0)}) == []


# ----- 순간 체결량 필터 --------------------------------------------------------

def test_big_trade_filter_blocks_when_no_big_prints():
    """대장에 5천만원 이상 체결이 없으면 상한가를 굳히는 자금이 아니다 — 기각."""
    det = _limit_det(limitup_touch=True, min_big_trades=1)
    # 대장 체결 1000주 × 12,990원 = 1,299만원 < 5천만원
    assert _limit_timeline(det, 13_000.0, lead30=13_000.0) == []


def test_big_trade_filter_passes_with_big_prints():
    det = _limit_det(limitup_touch=True, min_big_trades=1)
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.leader, 30, 13_000.0, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=1000)
    a = _feed(det, PAIR.leader, 60, 13_000.0, volume=5000)   # 6,495만원 ≥ 5천만원
    assert len(a) == 1 and a[0].big_trades == 1


def test_big_trade_window_expires():
    """관측창(기본 30초)을 벗어난 대량체결은 세지 않는다."""
    det = _limit_det(limitup_touch=True, min_big_trades=1, big_trade_window=10.0)
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.leader, 30, 13_000.0, volume=5000)       # 대량체결이지만 30초 전
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=1000)
    assert _feed(det, PAIR.leader, 60, 13_000.0, volume=1000) == []


def test_big_trade_filter_off_by_default():
    det = _det()
    assert det.config.min_big_trades == 0
    assert det.config.big_trade_krw == 50_000_000
    assert len(_base_timeline(det)) == 1


# ----- 대장 상태 통보 (만쥬 §5-3 — 짝꿍의 진짜 청산 규칙) ----------------------
#
# "대장이 상한가를 못 말거나 풀리는 순간 = 손익 무관 전량 매도". 상한가를 계산할 수
# 있는 건 전일종가를 주입받은 감지기뿐이라 판정은 여기서 내고 밖으로 알린다.

def test_leader_state_constants_match_risk_guard():
    """가드가 비교하는 문자열과 글자까지 같아야 한다 — 다르면 청산이 조용히 안 돈다."""
    from scalp_it.realtime import risk_guard
    from scalp_it.realtime.pair_detector import LEADER_HOLDING, LEADER_LOST
    assert LEADER_HOLDING == risk_guard.LEADER_HOLDING == "holding"
    assert LEADER_LOST == risk_guard.LEADER_LOST == "lost"


def _leader_det(**cfg_kw):
    """전일종가 1만원(상한가 13,000원) 대장 하나를 보는 감지기."""
    cfg_kw.setdefault("min_sl_ticks", 0.0)
    return PairDetector([PAIR], DetectorConfig(**cfg_kw),
                        prev_close={PAIR.leader: 10_000.0})


def test_leader_holding_then_lost_emitted_once_each():
    det = _leader_det()
    _feed(det, PAIR.leader, 0, 12_000.0, volume=1000)
    assert det.drain_leader_events() == []            # 상한가 근처도 아님
    _feed(det, PAIR.leader, 10, 13_000.0, volume=1000)
    _feed(det, PAIR.leader, 11, 13_000.0, volume=1000)   # 계속 잠김 — 재통보 없음
    ev = det.drain_leader_events()
    assert ev == [(PAIR.leader, BASE + timedelta(seconds=10), "holding")]
    _feed(det, PAIR.leader, 20, 12_990.0, volume=1000)   # 풀림
    _feed(det, PAIR.leader, 21, 12_900.0, volume=1000)   # 계속 아래 — 재통보 없음
    ev = det.drain_leader_events()
    assert ev == [(PAIR.leader, BASE + timedelta(seconds=20), "lost")]
    assert det.drain_leader_events() == []             # 비운 뒤엔 다시 안 나온다


def test_leader_lost_not_emitted_without_prior_touch():
    """상한가에 닿은 적 없는 대장은 이탈도 없다(그냥 오르내리는 종목)."""
    det = _leader_det()
    for sec, px in [(0, 12_000.0), (10, 12_500.0), (20, 11_000.0)]:
        _feed(det, PAIR.leader, sec, px, volume=1000)
    assert det.drain_leader_events() == []


def test_leader_relock_emits_holding_again():
    """풀렸다 다시 붙으면 holding 을 다시 알린다 — 가드가 재진입 판단을 할 수 있게."""
    det = _leader_det()
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    _feed(det, PAIR.leader, 10, 12_900.0, volume=1000)
    _feed(det, PAIR.leader, 20, 13_000.0, volume=1000)
    assert [(s, st) for _c, s, st in det.drain_leader_events()] == [
        (BASE, "holding"),
        (BASE + timedelta(seconds=10), "lost"),
        (BASE + timedelta(seconds=20), "holding"),
    ]


def test_leader_lost_confirm_window():
    """확인 구간을 주면 그 안의 한 틱 이탈은 통보하지 않는다(노이즈 방어)."""
    det = _leader_det(leader_lost_confirm_secs=2.0)
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    assert [st for _c, _t, st in det.drain_leader_events()] == ["holding"]
    _feed(det, PAIR.leader, 10, 12_990.0, volume=1000)   # 스침 — 아직 확인 안 됨
    assert det.drain_leader_events() == []
    _feed(det, PAIR.leader, 11, 13_000.0, volume=1000)   # 곧바로 복귀 — 이탈 아님
    assert det.drain_leader_events() == []
    _feed(det, PAIR.leader, 20, 12_990.0, volume=1000)   # 다시 아래로
    assert det.drain_leader_events() == []
    _feed(det, PAIR.leader, 23, 12_950.0, volume=1000)   # 3초 뒤에도 아래 → 이탈 확정
    assert [st for _c, _t, st in det.drain_leader_events()] == ["lost"]


def test_leader_state_skipped_without_prev_close():
    """전일종가를 모르면 상한가를 모른다 — 억지 추정 없이 이벤트를 내지 않는다."""
    det = PairDetector([PAIR], DetectorConfig(min_sl_ticks=0.0))   # prev_close 미주입
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    _feed(det, PAIR.leader, 10, 12_000.0, volume=1000)
    assert det.drain_leader_events() == []


def test_leader_events_not_mixed_into_alerts():
    """이벤트는 on_tick 반환값에 섞이면 안 된다 — 호출부가 진입 신호로 오해한다."""
    det = _leader_det(limitup_touch=True, cosurge_min=0.0, premove_max=1.0,
                      min_buy_ratio=0.0)
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=1000)
    alerts = _feed(det, PAIR.leader, 60, 13_000.0, volume=1000)
    assert all(isinstance(a, Alert) for a in alerts)
    assert len(det.drain_leader_events()) == 1


def test_leader_state_tracked_for_dynamic_universe():
    """동적 모드에선 유니버스 종목이 다 대장 후보라 전부 추적한다."""
    det = _dyn_det()
    _feed(det, "100003", 0, 13_000.0, volume=1000)
    _feed(det, "100003", 5, 12_900.0, volume=1000)
    assert [(c, st) for c, _t, st in det.drain_leader_events()] == [
        ("100003", "holding"), ("100003", "lost")]


def test_follower_only_code_emits_no_leader_events():
    """고정 쌍의 후속주는 대장이 아니다 — 가드가 안 쓸 이벤트를 만들지 않는다."""
    det = PairDetector([PAIR], DetectorConfig(min_sl_ticks=0.0),
                       prev_close={PAIR.follower: 10_000.0})
    _feed(det, PAIR.follower, 0, 13_000.0, bid1=13_000.0, volume=1000)
    _feed(det, PAIR.follower, 10, 12_000.0, bid1=12_000.0, volume=1000)
    assert det.drain_leader_events() == []


# ----- 상한가 매수벽 두께 (만쥬 §5-2 "수십만~수백만 주를 안정적으로 굳히면") -----
#
# 지금까지 '굳힘' 판정은 가격만 봤다(현재가 ≥ 상한가). 만쥬는 매수 잔량의 두께로
# 판정한다 — 상한가에 수십만~수백만 주가 쌓여야 굳힌 것이고, 그 벽이 얇으면 곧 풀린다.

def _wall(det, secs, qty, px=13_000.0, code=None):
    """대장 호가 스냅샷 — 상한가 ``px`` 에 ``qty`` 주가 쌓인 상태."""
    det.on_quote(code or PAIR.leader, BASE + timedelta(seconds=secs), px,
                 bids=[(px, qty)], asks=[(px + 10, qty)])


def _wall_timeline(det, *, qty, lead_now=13_000.0, wall_px=13_000.0):
    """후속 히스토리를 채우고, 대장 상한가 벽을 세운 뒤 마지막 대장 틱의 알림을 준다."""
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=1000)
    _feed(det, PAIR.leader, 30, 13_000.0, volume=1000)
    if qty is not None:
        _wall(det, 45, qty, wall_px)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=1000)
    return _feed(det, PAIR.leader, 60, lead_now, volume=1000)


def test_lock_wall_qty_blocks_thin_wall():
    """상한가에 1만 주밖에 없으면 '굳혔다'가 아니다 — 방아쇠 기각."""
    det = _limit_det(limitup_touch=True, lock_min_qty=100_000)
    assert _wall_timeline(det, qty=10_000) == []


def test_lock_wall_qty_passes_thick_wall():
    det = _limit_det(limitup_touch=True, lock_min_qty=100_000)
    a = _wall_timeline(det, qty=300_000)
    assert len(a) == 1 and a[0].lock_qty == pytest.approx(300_000)
    assert a[0].lock_krw == pytest.approx(300_000 * 13_000.0)


def test_lock_wall_krw_threshold_is_price_aware():
    """주 수만으로는 가격대별 의미가 달라 원화 환산 문턱도 따로 건다."""
    det = _limit_det(limitup_touch=True, lock_min_krw=10_000_000_000)   # 100억
    assert _wall_timeline(det, qty=100_000) == []                        # 13억
    ok = _limit_det(limitup_touch=True, lock_min_krw=10_000_000_000)
    assert len(_wall_timeline(ok, qty=1_000_000)) == 1                   # 130억


def test_lock_wall_both_thresholds_are_and():
    det = _limit_det(limitup_touch=True, lock_min_qty=100_000,
                     lock_min_krw=10_000_000_000)
    assert _wall_timeline(det, qty=200_000) == []      # 주 수는 통과, 금액은 미달


def test_lock_wall_requires_book_when_enabled():
    """벽을 재라고 했는데 호가를 못 받으면 굳힘을 확인할 수 없다 — 안전 기각."""
    det = _limit_det(limitup_touch=True, lock_min_qty=100_000)
    assert _wall_timeline(det, qty=None) == []


def test_lock_wall_off_by_default_keeps_old_behaviour():
    cfg = DetectorConfig()
    assert cfg.lock_min_qty == 0.0 and cfg.lock_min_krw == 0.0
    det = _limit_det(limitup_touch=True)
    assert len(_wall_timeline(det, qty=None)) == 1


def test_lock_wall_not_required_for_near_trigger():
    """'상한가 직전' 방아쇠는 아직 상한가 호가가 없다 — 벽을 요구하지 않는다."""
    det = _limit_det(limitup_near_pct=0.04, lock_min_qty=100_000)
    assert len(_wall_timeline(det, qty=None, lead_now=12_600.0)) == 1


# ----- 매수벽 깎임 = 이탈 예고 ------------------------------------------------

def test_wall_decay_emits_leader_lost():
    """상한가는 아직 안 풀렸어도 매수벽이 반토막 나면 이탈로 본다."""
    det = _leader_det(lock_wall_decay=0.5, lock_wall_window=30.0)
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    _wall(det, 0, 1_000_000)
    assert [st for _c, _t, st in det.drain_leader_events()] == ["holding"]
    _wall(det, 10, 700_000)
    assert det.drain_leader_events() == []             # 70% — 아직 아니다
    _wall(det, 20, 400_000)                            # 40% < 50%
    ev = det.drain_leader_events()
    assert [st for _c, _t, st in ev] == ["lost"]
    assert ev[0].reason == "wall"
    _wall(det, 21, 300_000)
    assert det.drain_leader_events() == []             # 이탈은 한 번만


def test_wall_decay_off_by_default():
    cfg = DetectorConfig()
    assert cfg.lock_wall_decay == 0.0
    det = _leader_det()
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    _wall(det, 0, 1_000_000)
    _wall(det, 10, 10_000)
    assert [st for _c, _t, st in det.drain_leader_events()] == ["holding"]


# ----- 매도 폭탄 (만쥬 §5-4 "프로그램 매도 폭탄이 쏟아져 막히면 지체 없이 손절") ----

def test_sell_dump_emits_lost_while_price_still_at_limit():
    """가격이 아직 상한가여도 매도 폭탄이 쏟아지면 이탈이다 — 만쥬는 or 로 말한다."""
    det = _leader_det(dump_krw=1_000_000_000, dump_window=10.0)
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    assert [st for _c, _t, st in det.drain_leader_events()] == ["holding"]
    _feed(det, PAIR.leader, 5, 13_000.0, volume=50_000, side=-1)    # 6.5억
    assert det.drain_leader_events() == []
    _feed(det, PAIR.leader, 6, 13_000.0, volume=50_000, side=-1)    # 누적 13억 ≥ 10억
    ev = det.drain_leader_events()
    assert [st for _c, _t, st in ev] == ["lost"]
    assert ev[0].reason == "dump"


def test_sell_dump_ignores_buy_side_volume():
    det = _leader_det(dump_krw=1_000_000_000, dump_window=10.0)
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    det.drain_leader_events()
    _feed(det, PAIR.leader, 5, 13_000.0, volume=200_000, side=1)    # 26억이지만 매수
    assert det.drain_leader_events() == []


def test_sell_dump_window_expires():
    det = _leader_det(dump_krw=1_000_000_000, dump_window=10.0)
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    det.drain_leader_events()
    _feed(det, PAIR.leader, 5, 13_000.0, volume=50_000, side=-1)
    _feed(det, PAIR.leader, 30, 13_000.0, volume=50_000, side=-1)   # 창 밖 — 누적 아님
    assert det.drain_leader_events() == []


def test_sell_dump_does_not_immediately_relock():
    """폭탄으로 이탈시킨 뒤 가격이 상한가라고 곧바로 holding 으로 되돌리면 안 된다."""
    det = _leader_det(dump_krw=1_000_000_000, dump_window=10.0)
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    _feed(det, PAIR.leader, 5, 13_000.0, volume=100_000, side=-1)
    assert [st for _c, _t, st in det.drain_leader_events()] == ["holding", "lost"]
    _feed(det, PAIR.leader, 30, 13_000.0, volume=1000)
    assert det.drain_leader_events() == []
    _feed(det, PAIR.leader, 40, 12_900.0, volume=1000)   # 실제로 풀렸다가
    _feed(det, PAIR.leader, 50, 13_000.0, volume=1000)   # 다시 붙으면 재잠김
    assert [st for _c, _t, st in det.drain_leader_events()] == ["holding"]


def test_sell_dump_off_by_default():
    cfg = DetectorConfig()
    assert cfg.dump_krw == 0.0
    det = _leader_det()
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    _feed(det, PAIR.leader, 5, 13_000.0, volume=10_000_000, side=-1)
    assert [st for _c, _t, st in det.drain_leader_events()] == ["holding"]


def test_leader_event_is_plain_three_tuple_with_reason():
    """소비자는 3-튜플로 언패킹한다 — reason 은 곁다리 속성이라 배선을 안 깬다."""
    det = _leader_det()
    _feed(det, PAIR.leader, 0, 13_000.0, volume=1000)
    _feed(det, PAIR.leader, 10, 12_900.0, volume=1000)
    ev = det.drain_leader_events()
    code, ts, state = ev[1]
    assert (code, ts, state) == (PAIR.leader, BASE + timedelta(seconds=10), "lost")
    assert ev[1] == (PAIR.leader, BASE + timedelta(seconds=10), "lost")
    assert ev[1].reason == "unlock"


# ----- 테마 신선도 대용치 (만쥬 §5-5) -----------------------------------------

def _fresh_det(counts, **cfg_kw):
    cfg_kw.setdefault("min_sl_ticks", 0.0)
    cfg_kw.setdefault("cosurge_min", 0.0)
    cfg_kw.setdefault("premove_max", 1.0)
    cfg_kw.setdefault("min_buy_ratio", 0.0)
    return PairDetector(
        [PAIR], DetectorConfig(**cfg_kw),
        prev_close={PAIR.leader: 10_000.0, PAIR.follower: 100.0},
        sectors={PAIR.leader: "정치", PAIR.follower: "정치"},
        sector_limitup_counts=counts,
    )


def test_stale_sector_is_excluded_and_recorded():
    """최근 N일 상한가를 이미 여러 번 낸 테마는 후발주가 안 따라온다 — 배제하되 기록."""
    det = _fresh_det({"정치": 5}, limitup_touch=True, max_sector_limitups=3)
    assert _limit_timeline(det, 13_000.0, lead30=13_000.0) == []
    rec = det.drain_filtered()
    assert len(rec) == 1 and rec[0].reason == "sector_stale"
    assert rec[0].sector == "정치" and rec[0].sector_limitups == 5
    assert "정치" in rec[0].to_line()
    assert det.filtered_count == 1


def test_fresh_sector_passes():
    det = _fresh_det({"정치": 1}, limitup_touch=True, max_sector_limitups=3)
    assert len(_limit_timeline(det, 13_000.0, lead30=13_000.0)) == 1
    assert det.drain_filtered() == []


def test_unknown_sector_count_passes():
    """섹터를 모르거나 집계가 없으면 판단 근거가 없다 — 막지 않는다."""
    det = _fresh_det({}, limitup_touch=True, max_sector_limitups=3)
    assert len(_limit_timeline(det, 13_000.0, lead30=13_000.0)) == 1


def test_sector_freshness_off_by_default():
    cfg = DetectorConfig()
    assert cfg.max_sector_limitups == 0
    det = _fresh_det({"정치": 99}, limitup_touch=True)
    assert len(_limit_timeline(det, 13_000.0, lead30=13_000.0)) == 1


def test_sector_limitup_counts_can_be_updated_intraday():
    det = _fresh_det({"정치": 1}, limitup_touch=True, max_sector_limitups=3)
    det.set_sector_limitup_counts({"정치": 9})
    assert _limit_timeline(det, 13_000.0, lead30=13_000.0) == []


# ----- 순간 체결량 필터 재검증 (만쥬 §5-1) -------------------------------------

def test_big_trade_counts_leader_only():
    """수급 확인은 **대장** 기준이다 — 후속주에 대량체결이 있어도 소용없다."""
    det = _limit_det(limitup_touch=True, min_big_trades=1)
    _feed(det, PAIR.follower, 0, 100.0, bid1=100.0, volume=1_000_000)   # 1억
    _feed(det, PAIR.follower, 30, 100.0, bid1=100.0, volume=1_000_000)
    _feed(det, PAIR.leader, 30, 13_000.0, volume=1000)
    _feed(det, PAIR.follower, 60, 100.6, bid1=100.0, volume=1_000_000)
    assert _feed(det, PAIR.leader, 60, 13_000.0, volume=1000) == []


def test_big_trade_needs_positive_threshold():
    """big_trade_krw 를 0 으로 두면 모든 체결이 '대량'이 된다 — 그건 필터가 아니다."""
    det = _limit_det(limitup_touch=True, min_big_trades=1, big_trade_krw=0.0)
    assert _limit_timeline(det, 13_000.0, lead30=13_000.0) == []


# ----- 만쥬 §5-1 VI 방아쇠 ----------------------------------------------------

def test_vi_trigger_fires_near_static_vi_band():
    """정적 VI(당일 기준가 ±10%) 근처를 방아쇠로 — 만쥬 §5-1.

    원문: "대장주가 상한가에 안착하거나 상한가 직전 급등·**VI에 들어가는 순간**".
    VI 는 상한가(+30%)보다 훨씬 이른 +10% 에서 걸리므로, 상한가만 보면 놓치는
    구간이 있다.

    실시간 VI 피드(`1h`)를 구독하면 종목당 등록 단위가 2→3 이 되어 구독 가능 종목이
    50→33 으로 준다. 정적 VI 는 **가격만으로 계산**되므로 구독 비용 없이 근사한다.
    """
    det = _accel_det(vi_near_pct=0.01, limitup_near_pct=0.0)   # VI 까지 1%p 이내
    _feed_series(det, "000002", _flat(-90, 40, 10_000))
    # 그날 첫 체결가가 기준가가 된다 → 10,000. 정적 VI 상단 11,000, 1%p 이내는 10,890.
    alerts = _feed_series(det, "000001", _flat(-90, 20, 10_000)
                          + [(25, 10_500), (30, 10_900)])
    assert alerts, "VI 근접 방아쇠가 안 터졌다"


def test_vi_trigger_off_by_default():
    det = _accel_det()
    assert det.config.vi_near_pct == 0.0


def test_vi_trigger_works_without_prev_close_but_ranking_does_not():
    """VI 는 전일종가 없이도 되지만, 후속주 순위는 전일종가가 있어야 매겨진다.

    상한가 방아쇠는 전일종가가 있어야 계산되는데(×1.3) **VI 는 당일 기준가만 쓴다**.
    그래서 전일종가를 못 받은 종목도 VI 로는 대장이 될 수 있다 — 다만 후속주를
    당일등락률로 고르는 단계에서 막힌다. 두 제약이 다른 자리에 있다는 뜻이다.
    """
    cfg = _manju_cfg_for_vi()
    det = PairDetector([], cfg, universe=["000001", "000002"],
                       sectors={"000001": "A", "000002": "A"}, prev_close={})
    _feed_series(det, "000002", _flat(-90, 40, 10_000))
    _feed_series(det, "000001", _flat(-90, 20, 10_000) + [(30, 10_900)])
    # 대장 방아쇠 자체는 VI 로 성립한다(전일종가 불필요).
    assert det._vi_kind("000001", 10_900) == "vi"
    assert det.limit_up("000001") is None       # 상한가는 못 구한다
    # 그러나 후속주를 등락률로 고를 수 없어 알림은 안 나간다.
    assert det._dynamic_pairs("000001", BASE + timedelta(seconds=30)) == []


def _manju_cfg_for_vi():
    return DetectorConfig.stable(
        vi_near_pct=0.01, limitup_near_pct=0.0, cosurge_min=0.0, premove_max=1.0,
        min_buy_ratio=0.0, min_sl_ticks=0.0, max_roundtrip_bp=0.0, cooldown=0.0,
        surge_ret=0.015, surge_window=30.0)


# ----- 만쥬 §6 장중 종목 추가 · §5-2 매수벽 거부 우회 -------------------------

def test_add_codes_makes_detector_accept_their_ticks():
    """장중에 추가한 종목의 틱을 감지기가 받아야 한다 — 만쥬 §6.

    2026-08-26: 수집기 구독만 늘리고 감지기에는 안 넣어서, 추가한 종목의 틱이
    `if code not in self._codes: return []` 로 전부 버려졌다. 로그에는 "장중 구독
    추가"가 찍히는데 알림은 영원히 안 나오는 **완전한 무동작**이었다.
    """
    det = PairDetector([], DetectorConfig.stable(min_sl_ticks=0.0),
                       universe=["000001", "000002"],
                       sectors={"000001": "A", "000002": "A"},
                       prev_close={"000001": 10_000, "000002": 10_000})
    assert "000009" not in det.codes
    det.on_tick("000009", BASE, 5_000)          # 모르는 종목 → 무시돼야 정상
    assert "000009" not in det._prices

    added = det.add_codes(["000009"], sectors={"000009": "A"},
                          prev_close={"000009": 5_000})
    assert added == ["000009"]
    assert "000009" in det.codes
    det.on_tick("000009", BASE + timedelta(seconds=1), 5_100)
    assert "000009" in det._prices, "추가한 종목의 틱이 여전히 버려진다"
    assert det.limit_up("000009") == 6_500      # 전일종가 5,000 → 상한가


def test_add_codes_is_idempotent_and_reports_only_new():
    det = PairDetector([], DetectorConfig.stable(),
                       universe=["000001"], sectors={"000001": "A"},
                       prev_close={"000001": 10_000})
    assert det.add_codes(["000001"]) == []      # 이미 있는 건 안 센다
    assert det.add_codes(["000007"], sectors={"000007": "B"}) == ["000007"]


def test_wall_rejection_is_not_bypassed_by_vi():
    """매수벽이 얇아 상한가 방아쇠가 거부되면 VI 로 되살아나면 안 된다 — §5-2.

    상한가에 붙은 종목은 정적 VI 상단(기준가 +10%)을 당연히 넘으므로, 벽 거부를
    `or self._vi_kind(...)` 로 덮으면 **얇은 종목이 그대로 통과**한다. 게다가 실제로는
    상한가 터치인 건이 "VI근접"으로 기록돼 복기까지 틀어진다.
    """
    cfg = DetectorConfig.stable(
        limitup_touch=True, vi_near_pct=0.01, lock_min_krw=300_000_000,
        cosurge_min=0.0, premove_max=1.0, min_buy_ratio=0.0, min_sl_ticks=0.0,
        max_roundtrip_bp=0.0, cooldown=0.0, surge_ret=0.0)
    det = PairDetector([], cfg, universe=["000001", "000002"],
                       sectors={"000001": "A", "000002": "A"},
                       prev_close={"000001": 10_000, "000002": 10_000})
    # 상한가 13,000. 매수벽이 얇다(1호가 1,000원어치) → 굳힘 실패.
    # 기준가(그날 첫 체결가) 10,000 → 정적 VI 상단 11,000. 상한가 13,000 은 그보다
    # 한참 위라 VI 조건을 **자동으로** 만족한다 — 벽 거부를 덮을 수 있는 상황이다.
    det.on_quote("000001", BASE, 13_000, bids=[(13_000.0, 1.0)], asks=[])
    _feed_series(det, "000002", _flat(-90, 30, 10_000))
    _feed_series(det, "000001", _flat(-90, 20, 10_000) + [(30, 13_000)])
    assert det._vi_kind("000001", 13_000) == "vi", "VI 는 성립하는 상황이어야 한다"
    assert det._limitup_kind("000001", BASE + timedelta(seconds=30), 13_000) is None
    assert det._leader_trigger("000001", BASE + timedelta(seconds=30), 13_000) is None, (
        "벽 거부가 VI 로 우회됐다")


# ==================================================================== 가격 이상 감지 (2026-09-07)


def test_price_jump_recorded_when_neighboring_ticks_diverge():
    """직전 틱 대비 10%+ 급변(짧은 간격)이면 큐에 남는다."""
    det = _det()
    _feed(det, PAIR.follower, 0, 6960.0)
    _feed(det, PAIR.follower, 1, 5570.0)   # 1초 만에 -20% — 2026-09-07 실제 사례
    recs = det.drain_price_anomalies()
    assert len(recs) == 1
    r = recs[0]
    assert r["code"] == PAIR.follower
    assert r["prev_price"] == 6960.0
    assert r["price"] == 5570.0
    assert r["pct"] == pytest.approx(5570.0 / 6960.0 - 1.0)
    assert r["gap_sec"] == pytest.approx(1.0)


def test_price_jump_not_recorded_for_small_moves():
    """정상적인 틱 단위 변동(10% 미만)은 안 남는다."""
    det = _det()
    _feed(det, PAIR.follower, 0, 1000.0)
    _feed(det, PAIR.follower, 1, 1050.0)   # +5%
    assert det.drain_price_anomalies() == []


def test_price_jump_not_recorded_across_long_gap():
    """틱 간격이 PRICE_JUMP_MAX_GAP 을 넘으면 정상적인 급등(랠리)일 수 있어 안 잰다."""
    det = _det()
    _feed(det, PAIR.follower, 0, 1000.0)
    _feed(det, PAIR.follower, 60, 1300.0)  # 60초 뒤 +30% — 이 시스템이 원래 잡는 급등
    assert det.drain_price_anomalies() == []


def test_price_jump_drain_empties_queue_but_keeps_going():
    det = _det()
    _feed(det, PAIR.follower, 0, 1000.0)
    _feed(det, PAIR.follower, 1, 1200.0)
    assert len(det.drain_price_anomalies()) == 1
    assert det.drain_price_anomalies() == []      # 비운 뒤엔 다시 안 나온다
    _feed(det, PAIR.follower, 2, 900.0)           # 다시 급변하면 또 잡힌다
    assert len(det.drain_price_anomalies()) == 1
