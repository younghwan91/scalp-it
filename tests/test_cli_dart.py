import itertools

import httpx

from scalp_it.cli_dart import main, quarter_ranges


def test_quarter_ranges_spans_are_positive_and_bounded():
    """구간은 뒤집히지 않고 달력 3개월을 넘지 않는다.

    처음에는 '90일 이하'로 고정했는데 그것이 버그였다 — DART 는 달력 3개월로
    센다. 90일이 3개월보다 짧은 달 조합에서는 통과하지만, 긴 달 조합에서는
    3개월을 넘겨 죽는다. 아래 달력 기준 테스트가 진짜 계약이다.
    """
    import pandas as pd

    for bgn, end in quarter_ranges("20240101", "20261231"):
        span = (pd.Timestamp(end) - pd.Timestamp(bgn)).days
        assert 0 <= span <= 92


def test_quarter_ranges_cover_the_whole_span_without_gaps():
    import pandas as pd

    ranges = quarter_ranges("20240101", "20241231")
    assert ranges[0][0] == "20240101"
    assert ranges[-1][1] == "20241231"
    for (_, prev_end), (next_bgn, _) in itertools.pairwise(ranges):
        assert pd.Timestamp(next_bgn) == pd.Timestamp(prev_end) + pd.Timedelta(days=1)


def test_quarter_ranges_single_short_span():
    assert quarter_ranges("20260801", "20260814") == [("20260801", "20260814")]


def test_cli_collects_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "k1")
    item = {
        "corp_cls": "K", "corp_name": "에프알텍", "corp_code": "00123456",
        "stock_code": "073540", "report_nm": "단일판매ㆍ공급계약체결",
        "rcept_no": "20260814000123", "flr_nm": "에프알텍",
        "rcept_dt": "20260814", "rm": "",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "status": "000", "message": "정상", "page_no": 1, "page_count": 100,
            "total_count": 1, "total_page": 1, "list": [item],
        })

    db = tmp_path / "dart.db"
    rc = main([
        "--start", "20260801", "--end", "20260814", "--db", str(db),
        "--corp-cls", "K", "--sleep", "0",
    ], client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert rc == 0

    from scalp_it.dart.store import init_dart_db, load_disclosures

    got = load_disclosures(init_dart_db(db), "20260801", "20260831")
    assert len(got) == 1
    assert got.iloc[0]["material_type"] == "공급계약"


def test_cli_dry_run_makes_no_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "k1")
    db = tmp_path / "dart.db"
    rc = main(["--start", "20260801", "--end", "20260814", "--db", str(db), "--dry-run"])
    assert rc == 0
    assert not db.exists()


def test_cli_without_key_fails_clearly(tmp_path, monkeypatch):
    for name in ("DART_API_KEY", "DART_API_KEY_2", "DART_API_KEY_3"):
        monkeypatch.delenv(name, raising=False)
    rc = main(["--start", "20260801", "--end", "20260814", "--db", str(tmp_path / "d.db")])
    assert rc == 1


def test_quarter_ranges_never_exceed_three_calendar_months():
    """DART 제한은 '3개월'이지 90일이 아니다.

    회귀 방지: 90일 고정 분할은 짧은 달(2월)을 낀 구간에서 달력 3개월을 넘어
    `100 corp_code가 없는 경우 검색기간은 3개월만 가능합니다` 로 죽는다.
    10년 수집이 21/80 구간에서 이 오류로 중단됐다.
    """
    import pandas as pd

    for bgn, end in quarter_ranges("20160901", "20260814"):
        start = pd.Timestamp(bgn)
        stop = pd.Timestamp(end)
        assert stop <= start + pd.DateOffset(months=3) - pd.Timedelta(days=1)


def test_quarter_ranges_handles_february_boundaries():
    """2월을 낀 구간이 특히 위험하다."""
    import pandas as pd

    for bgn, end in quarter_ranges("20231201", "20240601"):
        start, stop = pd.Timestamp(bgn), pd.Timestamp(end)
        assert stop <= start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
