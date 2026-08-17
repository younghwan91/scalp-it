import itertools

import httpx

from swing_it.cli_dart import main, quarter_ranges


def test_quarter_ranges_never_exceed_ninety_days():
    ranges = quarter_ranges("20240101", "20261231")
    import pandas as pd

    for bgn, end in ranges:
        span = (pd.Timestamp(end) - pd.Timestamp(bgn)).days
        assert 0 <= span <= 90


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

    from swing_it.dart.store import init_dart_db, load_disclosures

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
