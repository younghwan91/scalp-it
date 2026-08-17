"""`swing-cbet-watch` — 종가베팅 전진 기록 CLI.

네트워크도 실 DB 도 쓰지 않는다. 어댑터 세 함수를 monkeypatch 로 갈아끼우고
합성 OHLCV 만 흘린다. 여기서 고정하는 것은 사전등록 확정 규칙(거래대금 50억,
tv_ratio 3.0, close_pos 0.8, 하루 2건)이 그대로 적용된다는 것과, 전진 기록이
append-only 라 두 번 돌려도 흔들리지 않는다는 성질이다.
"""

import json

import pandas as pd
import pytest

from swing_it import cli_cbet_watch

DATES = pd.bdate_range("2026-01-05", periods=40)
ASOF = str(DATES[-1].date())
BASE_TV = 4000.0  # 백만원. 평시 거래대금 — tv_ratio 분모가 된다


def _bar(code, date, *, close, tv, high=None, low=None, open_=None, prev_close=10000.0):
    high = close if high is None else high
    low = prev_close * 0.98 if low is None else low
    return {
        "code": code,
        "date": date,
        "open": prev_close if open_ is None else open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": tv * 10.0,
        "trade_value": tv,
    }


def _ohlcv(spikes: dict[str, dict], base_tv: float = BASE_TV) -> pd.DataFrame:
    """평시 40일 + 마지막 날에 종목별 이벤트를 심은 합성 OHLCV.

    ``spikes`` 는 ``{code: {"tv": …, "close_pos": …}}``. 언급되지 않은 종목은
    마지막 날도 평시 그대로다. ``base_tv`` 가 tv_ratio 의 분모다.
    """
    rows = []
    for code in ("000001", "000002", "000003", "000004"):
        for date in DATES[:-1]:
            rows.append(_bar(code, date, close=10000.0, tv=base_tv, high=10100.0, low=9900.0))
        spec = spikes.get(code)
        if spec is None:
            rows.append(_bar(code, DATES[-1], close=10000.0, tv=base_tv, high=10100.0, low=9900.0))
            continue
        pos = spec.get("close_pos", 1.0)
        low, high = 10000.0, 11000.0
        rows.append(
            _bar(
                code,
                DATES[-1],
                close=low + pos * (high - low),
                tv=spec["tv"],
                high=high,
                low=low,
            )
        )
    return pd.DataFrame(rows)


def _master() -> pd.DataFrame:
    codes = ["000001", "000002", "000003", "000004"]
    return pd.DataFrame(
        {
            "code": codes,
            "name": [f"종목{c[-1]}" for c in codes],
            "market": ["KOSPI"] * len(codes),
            "sector": [None] * len(codes),
            "last_trade_date": [pd.NaT] * len(codes),
            "is_delisted": [False] * len(codes),
        }
    )


class _FakeConn:
    """CLI 가 ``finally: conn.close()`` 를 부르므로 최소한 그것만 받아준다."""

    def close(self):
        pass


@pytest.fixture()
def db(monkeypatch):
    """``load_ohlcv`` 가 돌려줄 프레임을 갈아끼우는 설정 함수를 준다."""
    state = {"ohlcv": _ohlcv({})}
    monkeypatch.setattr(cli_cbet_watch, "connect", lambda dsn=None: _FakeConn())
    monkeypatch.setattr(
        cli_cbet_watch, "load_ohlcv", lambda conn, start, end: state["ohlcv"]
    )
    monkeypatch.setattr(cli_cbet_watch, "load_stock_master", lambda conn: _master())

    def setup(spikes, base_tv=BASE_TV):
        state["ohlcv"] = _ohlcv(spikes, base_tv=base_tv)
        return state["ohlcv"]

    setup.state = state
    return setup


def _argv(tmp_path, *extra, asof=ASOF):
    return ["--asof", asof, "--log-path", str(tmp_path / "cbet.jsonl"), *extra]


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_rule_hit_is_written_to_the_log(db, tmp_path, capsys):
    db({"000001": {"tv": BASE_TV * 5, "close_pos": 1.0}})
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0

    records = _lines(tmp_path / "cbet.jsonl")
    assert [r["code"] for r in records] == ["000001"]
    rec = records[0]
    assert rec["strategy"] == "closebet"
    assert rec["event_date"] == ASOF
    assert rec["rule_version"] == cli_cbet_watch.RULE_VERSION
    assert rec["tv_ratio"] == pytest.approx(5.0)
    assert rec["close_pos"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("spike", "base_tv"),
    [
        pytest.param({"tv": BASE_TV * 2, "close_pos": 1.0}, BASE_TV, id="tv_ratio_2.0_미달"),
        pytest.param({"tv": BASE_TV * 5, "close_pos": 0.5}, BASE_TV, id="close_pos_0.5_미달"),
        # 배수 4.0·종가위치 1.0 이라 스캔은 통과하지만 거래대금 40억이라 유니버스 밖이다
        pytest.param({"tv": 4000.0, "close_pos": 1.0}, 1000.0, id="거래대금_50억_미달"),
    ],
)
def test_rule_miss_yields_no_candidate_and_exit_zero(db, tmp_path, capsys, spike, base_tv):
    db({"000001": spike}, base_tv=base_tv)
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0

    assert "오늘은 후보 없음" in capsys.readouterr().out
    assert _lines(tmp_path / "cbet.jsonl") == []


def test_running_twice_does_not_duplicate_the_log(db, tmp_path, capsys):
    db({"000001": {"tv": BASE_TV * 5}, "000002": {"tv": BASE_TV * 4}})
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0
    first = _lines(tmp_path / "cbet.jsonl")
    assert len(first) == 2

    assert cli_cbet_watch.main(_argv(tmp_path)) == 0
    assert _lines(tmp_path / "cbet.jsonl") == first
    assert "중복 2건 건너뜀" in capsys.readouterr().out


def test_at_most_two_candidates_per_day(db, tmp_path, capsys):
    db({
        "000001": {"tv": BASE_TV * 5},
        "000002": {"tv": BASE_TV * 4},
        "000003": {"tv": BASE_TV * 6},
    })
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0
    assert len(_lines(tmp_path / "cbet.jsonl")) == cli_cbet_watch.TOP_N == 2


def test_ranking_is_tv_ratio_descending(db, tmp_path, capsys):
    db({
        "000001": {"tv": BASE_TV * 4},
        "000002": {"tv": BASE_TV * 7},
        "000003": {"tv": BASE_TV * 5},
    })
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0

    records = _lines(tmp_path / "cbet.jsonl")
    # 상위 2건은 배수 7·5 이고, 그 순서로 기록돼야 한다
    assert [r["code"] for r in records] == ["000002", "000003"]
    ratios = [r["tv_ratio"] for r in records]
    assert ratios == sorted(ratios, reverse=True)


def test_output_shows_code_and_tv_ratio(db, tmp_path, capsys):
    db({"000001": {"tv": BASE_TV * 5}})
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0

    out = capsys.readouterr().out
    assert "000001" in out
    assert "tv_ratio" in out
    assert "5.00" in out
    assert "종목1" in out
    assert "15:20" in out  # 운영 갭 한 줄이 출력에 있어야 한다


def test_score_uses_next_day_open(db, tmp_path, capsys):
    """기록 뒤 익일 봉이 생기면 (익일시가 − 종가)/종가 로 채점된다."""
    db({"000001": {"tv": BASE_TV * 5, "close_pos": 1.0}})
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0
    entry = _lines(tmp_path / "cbet.jsonl")[0]["close"]

    # 익일 봉을 붙인다 — 시가는 종가의 +10%
    next_day = pd.Timestamp(DATES[-1]) + pd.Timedelta(days=3)
    extra = _bar("000001", next_day, close=entry, tv=BASE_TV, open_=entry * 1.10)
    db.state["ohlcv"] = pd.concat(
        [db.state["ohlcv"], pd.DataFrame([extra])], ignore_index=True
    )

    capsys.readouterr()
    assert cli_cbet_watch.main(_argv(tmp_path, "--score")) == 0
    out = capsys.readouterr().out
    assert "채점 1건 · 미채점 0건" in out
    assert "+10.00%" in out          # 총수익
    assert "+9.66%" in out           # 34bp 차감 후
    assert "61.0%" in out            # 백테스트 기대 승률이 나란히 찍힌다


def test_score_separates_records_without_a_next_day(db, tmp_path, capsys):
    db({"000001": {"tv": BASE_TV * 5}, "000002": {"tv": BASE_TV * 4}})
    assert cli_cbet_watch.main(_argv(tmp_path)) == 0
    assert len(_lines(tmp_path / "cbet.jsonl")) == 2

    # 익일 봉을 한 종목에만 붙인다 — 나머지 한 건은 미채점이어야 한다
    next_day = pd.Timestamp(DATES[-1]) + pd.Timedelta(days=3)
    entry = _lines(tmp_path / "cbet.jsonl")[0]["close"]
    extra = _bar("000001", next_day, close=entry, tv=BASE_TV, open_=entry)
    db.state["ohlcv"] = pd.concat(
        [db.state["ohlcv"], pd.DataFrame([extra])], ignore_index=True
    )

    capsys.readouterr()
    assert cli_cbet_watch.main(_argv(tmp_path, "--score")) == 0
    out = capsys.readouterr().out
    assert "기록 2건 · 채점 1건 · 미채점 1건" in out


def test_score_on_empty_log_is_not_a_failure(tmp_path, capsys):
    assert cli_cbet_watch.main(_argv(tmp_path, "--score")) == 0
    assert "전진 기록이 비어 있다" in capsys.readouterr().out


def test_rule_constants_match_the_preregistration():
    """확정 규칙은 문서(06-closebet-preregistration.md)와 일치해야 한다."""
    assert cli_cbet_watch.ADV_FLOOR == 5000.0
    assert cli_cbet_watch.TV_RATIO_MIN == 3.0
    assert cli_cbet_watch.CLOSE_POS_MIN == 0.8
    assert cli_cbet_watch.TOP_N == 2
    assert cli_cbet_watch.COST_ROUND_TRIP == 0.0034
    assert cli_cbet_watch.EXPECTED == {"mean": 0.0307, "net34": 0.0273, "win_rate": 0.610}
