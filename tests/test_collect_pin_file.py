"""고정 수집 목록 — 다른 전략이 떨군 파일 때문에 **매매가 멈추면 안 된다**."""
from scalp_it.cli_pair_detect import _pin_codes


def test_inline_only():
    assert _pin_codes("069500, 122630", "") == ["069500", "122630"]


def test_file_adds_and_dedupes(tmp_path):
    f = tmp_path / "pins.txt"
    f.write_text("# daytrade-it 09-17\n005930\n069500\n000660,035720\n")
    assert _pin_codes("069500", str(f)) == ["069500", "005930", "000660", "035720"]


def test_missing_file_does_not_break(capsys):
    assert _pin_codes("069500", "/nonexistent/pins.txt") == ["069500"]
    assert "경고" in capsys.readouterr().err


def test_empty_inputs():
    assert _pin_codes("", "") == []


def test_rejects_things_that_are_not_krx_codes(capsys):
    """남의 파일이다 — 헤더·설명·경로가 들어와도 구독 시도로 새면 안 된다."""
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as d:
        f = _P(d) / "pins.txt"
        f.write_text("005930\nAAPL\n12345\n0059301\n/tmp/foo\n종목코드\n069500\n")
        assert _pin_codes("", str(f)) == ["005930", "069500"]
        assert "건너뛴다" in capsys.readouterr().err
