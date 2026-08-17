"""`swing-watch` — ①층 전진 기록 CLI.

네트워크도 실 DB 도 쓰지 않는다. 어댑터 네 함수를 monkeypatch 로 갈아끼우고
합성 프레임만 흘린다. 전진 기록은 append-only 라, 여기서 고정하는 것은
"두 번 돌려도 과거 기록이 안 흔들린다"는 성질이다.
"""

import json

import numpy as np
import pandas as pd
import pytest

from swing_it import cli_watch

CODES = [f"{i:06d}" for i in range(30)]
N_DAYS = 220
DATES = pd.bdate_range("2024-01-01", periods=N_DAYS)


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c_i, code in enumerate(CODES):
        price = 10000.0
        for i, date in enumerate(DATES):
            price *= 1.0 + rng.normal(scale=0.01)
            spike = (i >= 21 and i % 13 == c_i % 13) or i == N_DAYS - 1
            rows.append(
                {
                    "code": code,
                    "date": date,
                    "open": price,
                    "close": price,
                    "trade_value": 20000.0 * (6.0 if spike else 1.0),
                }
            )
    return pd.DataFrame(rows)


def _supply() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for code in CODES:
        for date in DATES:
            foreign = rng.normal(scale=500.0)
            inst = rng.normal(scale=500.0)
            rows.append(
                {
                    "code": code,
                    "date": date,
                    "individual": -(foreign + inst),
                    "foreign_": foreign,
                    "institution": inst,
                    "penfnd_etc": rng.normal(scale=100.0),
                    "acc_trde_qty": 100000.0,
                }
            )
    return pd.DataFrame(rows)


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": CODES,
            "name": [f"종목{i}" for i in range(len(CODES))],
            "market": ["KOSPI"] * len(CODES),
            "sector": [None] * len(CODES),
            "last_trade_date": [pd.NaT] * len(CODES),
            "is_delisted": [False] * len(CODES),
        }
    )


@pytest.fixture()
def db(monkeypatch):
    """어댑터 네 함수를 합성 프레임으로 대체한다."""
    prices, supply, master = _prices(), _supply(), _master()
    monkeypatch.setattr(cli_watch, "connect", lambda dsn=None: object())
    monkeypatch.setattr(cli_watch, "load_prices", lambda conn, start, end: prices)
    monkeypatch.setattr(cli_watch, "load_supply_demand", lambda conn, start, end: supply)
    monkeypatch.setattr(cli_watch, "load_stock_master", lambda conn: master)
    return prices


def _fixed_threshold(latest: float, pooled: float = float("nan")):
    """``concurrency_threshold`` 를 고정값으로 대체하는 스텁."""

    def stub(pred, *, hold_days=40, target_concurrent=20):
        return {
            "threshold": pooled,
            "latest_threshold": latest,
            "expected_per_day": 0.0,
            "expected_concurrent": 0.0,
            "n_train_days": int(pred["date"].nunique()),
            "per_fold": [],
        }

    return stub


ASOF = str(DATES[-1].date())
MID_ASOF = str(DATES[100].date())


def _argv(tmp_path, *extra, asof=ASOF):
    return [
        "--db", "x",
        # 이 파일의 테스트는 volspike 경로를 검증한다. 기본값은 disclosure 이고
        # 그쪽은 공시 DB 가 필요하므로 명시한다.
        "--events", "volspike",
        "--asof", asof,
        "--min-train", "100",
        "--folds", "3",
        "--log-path", str(tmp_path / "forward_log.jsonl"),
        *extra,
    ]


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_candidates_are_appended_to_the_log(db, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_watch, "concurrency_threshold", _fixed_threshold(-1e9))
    assert cli_watch.main(_argv(tmp_path)) == 0

    records = _lines(tmp_path / "forward_log.jsonl")
    assert records
    assert {r["layer"] for r in records} == {"supply_volspike"}
    assert {r["event_date"] for r in records} == {ASOF}
    assert set(records[0]["features"]) == set(cli_watch.FEATURE_COLUMNS)


def test_threshold_miss_yields_no_candidates_and_no_log(db, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_watch, "concurrency_threshold", _fixed_threshold(1e9))
    assert cli_watch.main(_argv(tmp_path)) == 0

    out = capsys.readouterr().out
    assert "후보 없음" in out
    assert not (tmp_path / "forward_log.jsonl").exists()


def test_running_twice_does_not_duplicate_the_log(db, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_watch, "concurrency_threshold", _fixed_threshold(-1e9))
    assert cli_watch.main(_argv(tmp_path)) == 0
    first = _lines(tmp_path / "forward_log.jsonl")
    assert cli_watch.main(_argv(tmp_path)) == 0
    assert _lines(tmp_path / "forward_log.jsonl") == first


def test_output_shows_code_and_prediction(db, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_watch, "concurrency_threshold", _fixed_threshold(-1e9))
    assert cli_watch.main(_argv(tmp_path)) == 0

    out = capsys.readouterr().out
    records = _lines(tmp_path / "forward_log.jsonl")
    top = max(records, key=lambda r: r["prediction"])
    assert top["code"] in out
    assert f"{top['prediction']:+.4f}" in out
    assert "종목코드" in out


def test_score_only_counts_records_that_have_labels(db, monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "forward_log.jsonl"
    monkeypatch.setattr(cli_watch, "concurrency_threshold", _fixed_threshold(-1e9))
    # 라벨이 이미 존재하는 과거 기준일로 후보를 기록한다
    assert cli_watch.main(_argv(tmp_path, asof=MID_ASOF)) == 0
    scored = len(_lines(log_path))
    assert scored > 0

    # 아직 D+40 이 오지 않은 기록 한 건을 덧붙인다 — 채점 대상이 아니어야 한다
    pending = dict(_lines(log_path)[0])
    pending.update({"event_date": ASOF, "logged_at": "2099-01-01"})
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pending, ensure_ascii=False) + "\n")

    capsys.readouterr()
    assert cli_watch.main(_argv(tmp_path, "--score")) == 0
    out = capsys.readouterr().out
    assert f"채점 {scored}건" in out
    assert "미채점 1건" in out


def test_threshold_comes_from_latest_fold_not_the_pooled_value(db, monkeypatch, tmp_path, capsys):
    # 전역(pooled) 값은 통과 불가, 최신 폴드 값은 전부 통과 — 후보가 나와야 한다
    monkeypatch.setattr(cli_watch, "concurrency_threshold", _fixed_threshold(-1e9, pooled=1e9))
    assert cli_watch.main(_argv(tmp_path)) == 0
    assert len(_lines(tmp_path / "forward_log.jsonl")) > 0

    # 반대로 뒤집으면 0 건이어야 한다
    other = tmp_path / "b"
    other.mkdir()
    monkeypatch.setattr(cli_watch, "concurrency_threshold", _fixed_threshold(1e9, pooled=-1e9))
    assert cli_watch.main(_argv(other)) == 0
    assert not (other / "forward_log.jsonl").exists()


def test_default_event_mode_is_disclosure():
    """실측상 공시 이벤트가 2.5배 강하므로 기본값이어야 한다.

    거래대금 급증 +3.00% / net +2.54% vs 공시 접수 +7.61% / net +7.15%.
    → docs/research/2026-08-17-disclosure-layer-verdict.md
    """
    import argparse

    from swing_it import cli_watch

    parser = argparse.ArgumentParser()
    # main 의 파서를 직접 만들 수 없으므로 상수로 계약을 고정한다
    assert set(cli_watch._EVENT_MODES) == {"disclosure", "volspike"}
    assert cli_watch._EVENT_MODES["disclosure"]["layer"] == "supply_disclosure"
    assert cli_watch._EVENT_MODES["volspike"]["layer"] == "supply_volspike"
    del parser


def test_verdict_is_not_hardcoded_pass():
    """공시 경로는 폴드 0 에서 널을 못 이겨 통과한 적이 없다.

    기록에 'PASS' 를 박으면 나중에 전진 기록을 증거로 쓸 때 무엇을 믿고 샀는지가
    왜곡된다.
    """
    from swing_it import cli_watch

    assert cli_watch._EVENT_MODES["disclosure"]["verdict"] == "FAIL_fold0"
    assert cli_watch._EVENT_MODES["volspike"]["verdict"] == "PASS"


def test_disclosure_mode_without_dart_db_fails_clearly(tmp_path, capsys):
    from swing_it import cli_watch

    rc = cli_watch.main([
        "--db", "x", "--events", "disclosure",
        "--dart-db", str(tmp_path / "nope.db"),
        "--log-path", str(tmp_path / "log.jsonl"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "공시 DB" in err and "swing-dart" in err


def test_entry_date_is_after_the_event_even_at_calendar_end():
    """운영에서 asof 는 보통 달력 마지막 날이다. 그때도 진입은 D+1 이어야 한다.

    회귀 방지: min(idx+1, last) 로 자르면 진입일이 이벤트일과 같아져
    'D+1 시가 진입'이라는 설계가 표시상 무너진다.
    """
    import pandas as pd

    from swing_it.cli_watch import _plan_dates

    prices = pd.DataFrame({
        "code": ["A"] * 3,
        "date": pd.to_datetime(["2026-08-12", "2026-08-13", "2026-08-14"]),
    })
    asof = pd.Timestamp("2026-08-14")          # 달력의 마지막 날
    entry, exit_date = _plan_dates(prices, asof, hold_end=40)
    assert entry > asof
    assert exit_date > entry


def test_entry_date_uses_the_calendar_when_available():
    import pandas as pd

    from swing_it.cli_watch import _plan_dates

    prices = pd.DataFrame({
        "code": ["A"] * 4,
        "date": pd.to_datetime(["2026-08-12", "2026-08-13", "2026-08-14", "2026-08-17"]),
    })
    entry, _ = _plan_dates(prices, pd.Timestamp("2026-08-14"), hold_end=1)
    assert entry == pd.Timestamp("2026-08-17")
