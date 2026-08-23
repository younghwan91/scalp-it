"""`scalp-pead-watch` — PEAD 전진 기록 CLI.

네트워크도 실 DB 도 쓰지 않는다. 어댑터 로더를 monkeypatch 로 갈아끼우고 합성
earnings·수정주가만 흘린다. 여기서 고정하는 것은 배포 스펙(유동성 50억, 분기내
데실, 롱 Q10·숏 Q1)이 그대로 적용된다는 것과, 분기 리밸 로그가 append-only 라
두 번 돌려도 흔들리지 않는다는 성질이다.
"""

import json

import pandas as pd
import pytest

from scalp_it import cli_pead_watch
from scalp_it.report.forward_log import append_forward_log, read_forward_log

DATES = pd.bdate_range("2026-01-02", "2026-05-01")
ASOF = str(DATES[-1].date())
AVAIL = DATES[-15]  # 진입 = 익일 거래일, asof 보다 앞선다
N = 40
CODES = [f"{i:06d}" for i in range(1, N + 1)]


def _bars(codes=CODES, dates=DATES, tv=6000.0):
    """모든 종목·모든 날짜 평평한 봉. 유동성 tv(백만원) 는 배포 하한을 넘긴다."""
    rows = [
        {"code": c, "date": d, "open": 10000.0, "close": 10000.0, "trade_value": tv}
        for c in codes
        for d in dates
    ]
    return pd.DataFrame(rows)


def _raw(codes=CODES, dates=DATES):
    rows = [
        {"code": c, "date": d, "open": 10000.0, "close": 10000.0}
        for c in codes
        for d in dates
    ]
    return pd.DataFrame(rows)


def _earnings(codes=CODES, avail=AVAIL, period="2026Q1"):
    """종목마다 서로 다른 SUE. op_income_prior=100, op_income=100+rank*10 → 데실 분리."""
    rows = [
        {
            "code": c,
            "period": period,
            "avail_date": avail,
            "op_income": 100.0 + i * 10.0,
            "op_income_prior": 100.0,
        }
        for i, c in enumerate(codes)
    ]
    return pd.DataFrame(rows)


def _meta(codes=CODES):
    return pd.DataFrame(
        {
            "code": codes,
            "name": [f"종목{c}" for c in codes],
            "market": ["KOSPI"] * len(codes),
            "sector": ["섹터"] * len(codes),
        }
    )


class _FakeConn:
    def close(self):
        pass


@pytest.fixture()
def db(monkeypatch):
    state = {
        "earn": _earnings(),
        "adj": _bars(),
        "raw": _raw(),
        "meta": _meta(),
    }
    monkeypatch.setattr(cli_pead_watch, "connect", lambda dsn=None: _FakeConn())
    monkeypatch.setattr(cli_pead_watch, "load_earnings", lambda conn: state["earn"])
    monkeypatch.setattr(
        cli_pead_watch, "load_prices", lambda conn, start, end: state["adj"]
    )
    monkeypatch.setattr(
        cli_pead_watch, "load_daily_bars_raw", lambda conn, start, end: state["raw"]
    )
    monkeypatch.setattr(cli_pead_watch, "load_stock_master", lambda conn: state["meta"])
    return state


def _argv(tmp_path, *extra):
    return ["--asof", ASOF, "--log-path", str(tmp_path / "pead.jsonl"), *extra]


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --- 순수 함수: 포트폴리오 생성 -------------------------------------------------


def test_select_portfolio_splits_top_and_bottom_deciles(db):
    pf = cli_pead_watch.select_portfolio(
        db["earn"], db["adj"], db["raw"], db["meta"], set(), pd.Timestamp(ASOF)
    )
    assert pf is not None
    assert pf.period == "2026Q1"
    # 데실 40/10 = 4 종목/side
    assert len(pf.longs) == 4
    assert len(pf.shorts) == 4
    # 롱은 서프라이즈 최상(마지막 코드들), 숏은 최하(첫 코드들)
    assert set(pf.longs["code"]) == set(CODES[-4:])
    assert set(pf.shorts["code"]) == set(CODES[:4])
    # 진입 기준가·진입일·목표청산일이 붙는다
    assert (pf.longs["entry_open"] == 10000.0).all()
    assert pf.longs["entry_date"].notna().all()
    assert pf.longs["exit_date_planned"].notna().all()


def test_latest_avail_quarter_wins(db):
    """옛 분기·새 분기가 다 유효하면 최신 분기로 포트폴리오를 만든다."""
    old = _earnings(period="2025Q4", avail=DATES[-40])
    both = pd.concat([old, db["earn"]], ignore_index=True)
    pf = cli_pead_watch.select_portfolio(
        both, db["adj"], db["raw"], db["meta"], set(), pd.Timestamp(ASOF)
    )
    assert pf.period == "2026Q1"


def test_thin_universe_forms_no_portfolio(db):
    """MIN_NAMES 미만이면 분기를 형성하지 않는다."""
    few = _earnings(codes=CODES[:10])
    pf = cli_pead_watch.select_portfolio(
        few, _bars(CODES[:10]), _raw(CODES[:10]), _meta(CODES[:10]),
        set(), pd.Timestamp(ASOF),
    )
    assert pf is None


def test_illiquid_names_are_excluded(db):
    """거래대금 50억 미만이면 유니버스 밖 → 형성 불가."""
    pf = cli_pead_watch.select_portfolio(
        db["earn"], _bars(tv=1000.0), db["raw"], db["meta"], set(), pd.Timestamp(ASOF)
    )
    assert pf is None


# --- CLI: 기록·중복·평가 -------------------------------------------------------


def test_portfolio_is_written_to_the_log(db, tmp_path):
    assert cli_pead_watch.main(_argv(tmp_path)) == 0
    records = _lines(tmp_path / "pead.jsonl")
    assert len(records) == 8  # 롱 4 + 숏 4
    sides = {r["side"] for r in records}
    assert sides == {"long", "short"}
    rec = records[0]
    assert rec["strategy"] == "pead"
    assert rec["period"] == "2026Q1"
    assert rec["rule_version"] == cli_pead_watch.RULE_VERSION
    assert rec["entry_open"] == 10000.0
    assert "exit_date_planned" in rec


def test_rerunning_the_same_quarter_writes_nothing(db, tmp_path, capsys):
    assert cli_pead_watch.main(_argv(tmp_path)) == 0
    first = _lines(tmp_path / "pead.jsonl")
    assert len(first) == 8

    assert cli_pead_watch.main(_argv(tmp_path)) == 0
    assert _lines(tmp_path / "pead.jsonl") == first
    assert "이미 기록된 분기" in capsys.readouterr().out


def test_dry_run_does_not_write(db, tmp_path, capsys):
    assert cli_pead_watch.main(_argv(tmp_path, "--dry-run")) == 0
    assert _lines(tmp_path / "pead.jsonl") == []
    assert "dry-run" in capsys.readouterr().out


def test_output_shows_long_and_short_tables(db, tmp_path, capsys):
    assert cli_pead_watch.main(_argv(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "롱 (Q10)" in out
    assert "숏 (Q1)" in out
    assert "2026Q1" in out


def test_holdings_evaluation_is_printed(db, tmp_path, capsys):
    assert cli_pead_watch.main(_argv(tmp_path)) == 0
    out = capsys.readouterr().out
    # 목표청산 미도래라 8건 모두 보유 중, 진입가와 현재가가 같아 0%
    assert "보유 중 포지션 평가" in out
    assert "보유 8건" in out


def test_no_portfolio_still_evaluates_holdings(db, tmp_path, capsys):
    """형성 가능한 분기가 없어도 이전 보유 평가는 출력한다."""
    thin = _earnings(codes=CODES[:5])
    db["earn"] = thin
    assert cli_pead_watch.main(_argv(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "형성 가능한 avail 분기가 없다" in out
    assert "보유 중 포지션" in out


# --- append_forward_log: 분기 키로 멱등 ----------------------------------------


def test_append_is_idempotent_with_quarter_key(tmp_path):
    """logged_at 이 달라도 같은 (strategy, period, code, side) 는 재기록 안 된다."""
    path = tmp_path / "pead.jsonl"
    day1 = [
        {"logged_at": "2026-05-01", "strategy": "pead", "period": "2026Q1",
         "code": "000001", "side": "long"},
    ]
    day2 = [  # 다른 날 같은 분기·종목·side — 중복이어야 한다
        {"logged_at": "2026-05-08", "strategy": "pead", "period": "2026Q1",
         "code": "000001", "side": "long"},
    ]
    assert append_forward_log(path, day1, key=cli_pead_watch.DEDUP_KEY) == 1
    assert append_forward_log(path, day2, key=cli_pead_watch.DEDUP_KEY) == 0
    assert len(read_forward_log(path)) == 1
