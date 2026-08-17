import pandas as pd
import pytest

from scalp_it.dart.events import EVENT_COLUMNS, build_disclosure_events


def _prices(codes, n, adv=9000.0, spike_at=None):
    rows = []
    for code in codes:
        for i in range(n):
            tv = adv * (5.0 if i == spike_at else 1.0)
            rows.append({
                "code": code,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "open": 100.0, "close": 100.0, "trade_value": tv,
            })
    return pd.DataFrame(rows)


def _disc(rows):
    """(rcept_dt, code, material_type)."""
    return pd.DataFrame({
        "rcept_no": [f"r{i}" for i in range(len(rows))],
        "rcept_dt": pd.to_datetime([r[0] for r in rows]),
        "code": [r[1] for r in rows],
        "corp_name": ["회사"] * len(rows),
        "corp_cls": ["K"] * len(rows),
        "report_nm": [r[2] for r in rows],
        "material_type": [r[2] for r in rows],
        "is_correction": [False] * len(rows),
    })


def test_event_columns_contract():
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert list(out.columns) == EVENT_COLUMNS


def test_disclosure_becomes_an_event():
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert len(out) == 1
    assert out.iloc[0]["code"] == "A"
    assert out.iloc[0]["date"] == pd.Timestamp("2026-01-30")


def test_bad_news_is_excluded():
    """롱온리라 악재는 후보가 될 수 없다."""
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "악재")])
    assert build_disclosure_events(disc, prices).empty


def test_same_code_same_day_collapses_to_one_event():
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "A", "공급계약"), ("2026-01-30", "A", "시설투자")])
    assert len(build_disclosure_events(disc, prices)) == 1


def test_illiquid_stock_is_dropped():
    """PEAD 검증에서 유동성 하한 없이는 소형주 꼬리가 시그널을 음수로 뒤집었다."""
    prices = _prices(["A"], 40, adv=100.0)
    disc = _disc([("2026-01-30", "A", "공급계약")])
    assert build_disclosure_events(disc, prices, adv_floor=5000.0).empty


def test_date_without_price_row_is_dropped():
    """거래정지·상장 전이면 진입할 수 없다."""
    prices = _prices(["A"], 40)
    disc = _disc([("2026-01-30", "B", "공급계약")])  # B 는 가격이 없다
    assert build_disclosure_events(disc, prices).empty


def test_tv_ratio_is_carried_as_a_feature_not_a_filter():
    """거래대금 급증은 rev2 에서 이벤트가 아니라 피처다. 급증이 없어도 이벤트다."""
    prices = _prices(["A"], 40)  # 급증 없음
    disc = _disc([("2026-01-30", "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert len(out) == 1
    assert out.iloc[0]["tv_ratio"] == pytest.approx(1.0)


def test_spike_is_reflected_in_tv_ratio():
    prices = _prices(["A"], 40, spike_at=35)
    spike_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=35)
    disc = _disc([(str(spike_date.date()), "A", "공급계약")])
    out = build_disclosure_events(disc, prices)
    assert out.iloc[0]["tv_ratio"] > 4.0


def test_holiday_disclosure_moves_to_next_trading_day():
    """가격 달력에 없는 날 접수된 공시가 사라지면 안 된다."""
    prices = pd.DataFrame({
        "code": ["A"] * 3,
        "date": pd.to_datetime(["2026-08-13", "2026-08-14", "2026-08-17"]),
        "open": [100.0] * 3, "close": [100.0] * 3, "trade_value": [9000.0] * 3,
    })
    disc = _disc([("2026-08-15", "A", "공급계약")])   # 토요일
    out = build_disclosure_events(disc, prices, adv_window=1, adv_floor=0.0)
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2026-08-17")


def test_empty_disclosures_returns_empty_with_columns():
    out = build_disclosure_events(_disc([]), _prices(["A"], 40))
    assert out.empty
    assert list(out.columns) == EVENT_COLUMNS
