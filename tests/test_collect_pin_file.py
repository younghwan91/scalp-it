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


def test_cron_uses_the_incremental_kernel_not_the_window_one(cron_args):
    """창 재계산 커널은 초당 55~129ms 라 매매 프로세스 안에서 못 돈다(09-16~18 에 수집까지 갉아먹었다).

    core 0.5.1 의 증분 커널(v2)은 실측 **초당 75µs**(1,000배 빠름)이고 배치와 불일치 0열이다.
    그래서 종목 수를 다시 올릴 수 있다 — 20종목이면 초당 1.5ms.
    """
    assert cron_args.rl_kernel == "auto", "window 로 두면 다시 초당 100ms 가 된다"
    assert 1 <= cron_args.rl_codes <= 40
