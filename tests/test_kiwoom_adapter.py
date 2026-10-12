"""키움 어댑터 테스트. 네트워크는 절대 타지 않는다 — 전부 MockTransport/가짜 객체.

2026-09-12: ka10080(분봉)·ka90001/ka90002(테마)는 배치 전용이라 kiwoom_client로
옮겼고, ka10027(등락률상위)·ka90005(프로그램매매, 라이브 루프가 씀)는 여전히
아래 ``_fetch_rows``/``_post`` 저수준 전송을 쓴다. 그래서 토큰 재사용·재시도·
레이트리밋처럼 그 저수준 전송 자체를 검증하는 테스트는 ``fetch_rank_rows``를
호출 대상으로 쓴다 — ``minute_chart``를 쓰면 이제 그 전송을 안 타서 검증이 안
된다. 분봉의 페이지네이션·중단 로직(``minute_chart`` 자체 코드)은 ``fetch_page``
를 몽키패치해서 HTTP 없이 검증한다.
"""

from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from kiwoom_client import KiwoomAPIError

from scalp_it.adapters.kiwoom import (
    CHART_ROWS_KEY,
    THEME_GROUP_ROWS_KEY,
    THEME_MEMBER_ROWS_KEY,
    KiwoomClient,
    KiwoomError,
    api_credentials,
    normalize_row,
    parse_cntr_tm,
    strip_sign,
)

_TOKEN_BODY = {
    "expires_dt": "20260818120000",
    "return_msg": "정상적으로 처리되었습니다",
    "token_type": "bearer",
    "return_code": 0,
    "token": "tok-abc",
}

#: 실제 응답에서 그대로 가져온 분봉 한 행 (normalize_row 테스트용, 여전히 유효).
_ROW = {
    "cur_prc": "+274500",
    "trde_qty": "1366914",
    "cntr_tm": "20260814153000",
    "open_pric": "+274500",
    "high_pric": "+274500",
    "low_pric": "-273000",
    "acc_trde_qty": "21668199",
    "pred_pre": "+6500",
    "pred_pre_sig": "2",
}

#: 등락률상위(ka10027) 한 행 — 저수준 전송(_fetch_rows/_post) 검증용 껍데기.
_RANK_ROWS_KEY = "pred_pre_flu_rt_upper"
_RANK_ROW = {"stk_cd": "005930", "flu_rt": "+2.50"}


def _rank(rows, *, return_code=0, return_msg="정상적으로 처리되었습니다"):
    return httpx.Response(200, json={
        _RANK_ROWS_KEY: rows, "return_code": return_code, "return_msg": return_msg,
    })


def _make(handler, **kwargs):
    return KiwoomClient(
        "key", "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _s: None,
        **kwargs,
    )


def _router(rank_handler):
    """토큰 요청은 공통 처리하고 그 외(등락률상위 등)만 넘긴다."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/token":
            return httpx.Response(200, json=_TOKEN_BODY)
        return rank_handler(request)

    return handler


# ------------------------------------------------------------------ 인증


def test_api_credentials_reads_both_env_vars():
    env = {"KIWOOM_APP_KEY": "k", "KIWOOM_APP_SECRET": "s"}
    assert api_credentials(env) == ("k", "s")


def test_api_credentials_raises_when_missing():
    with pytest.raises(KiwoomError):
        api_credentials({"KIWOOM_APP_KEY": "k"})


def test_token_response_uses_token_key_not_access_token():
    """응답 키는 ``token`` 이다. ``access_token`` 을 찾으면 영원히 실패한다."""
    client = _make(_router(lambda r: _rank([])))
    assert client.issue_token() == "tok-abc"


def test_token_missing_key_raises():
    body = {"return_code": 0, "access_token": "wrong-key"}
    client = _make(lambda r: httpx.Response(200, json=body))
    with pytest.raises(KiwoomError):
        client.issue_token()


def test_token_is_reused_across_calls():
    """``.token``은 ``OrderExecutor``·WS 로그인이 직접 가져다 쓰는 그 프로퍼티다
    (2026-09-12 조회 5종을 kiwoom_client로 옮긴 뒤로도 이건 안 바뀜) — 두 번
    불러도 토큰 발급은 한 번만 나가야 한다."""
    calls = {"token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["token"] += 1
        return httpx.Response(200, json=_TOKEN_BODY)

    client = _make(handler)
    assert client.token == "tok-abc"
    assert client.token == "tok-abc"
    assert calls["token"] == 1


def test_refresh_token_discards_cache_and_reissues():
    calls = {"token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["token"] += 1
        return httpx.Response(200, json=dict(_TOKEN_BODY, token=f"tok-{calls['token']}"))

    client = _make(handler)
    assert client.token == "tok-1"
    assert client.refresh_token() == "tok-2"
    assert client.token == "tok-2"
    assert calls["token"] == 2


# ------------------------------------------------------------------ 파싱 (분봉 전용, 그대로 유효)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("+274500", 274500), ("-258000", 258000), ("258000", 258000), ("", 0), (None, 0)],
)
def test_strip_sign_converts_signed_values_to_int(raw, expected):
    """부호는 전일 대비 방향 표시다. 값의 음양이 아니라서 떼고 읽는다."""
    assert strip_sign(raw) == expected


def test_strip_sign_rejects_garbage():
    with pytest.raises(KiwoomError):
        strip_sign("abc")


def test_parse_cntr_tm_reads_full_timestamp():
    assert parse_cntr_tm("20260814153000") == pd.Timestamp(2026, 8, 14, 15, 30, 0)


def test_parse_cntr_tm_rejects_bad_value():
    with pytest.raises(KiwoomError):
        parse_cntr_tm("2026081415")


def test_normalize_row_maps_every_ohlcv_field():
    got = normalize_row("005930", _ROW)
    assert got["code"] == "005930"
    assert got["ts"] == pd.Timestamp(2026, 8, 14, 15, 30)
    assert (got["open"], got["high"], got["low"], got["close"]) == (
        274500, 274500, 273000, 274500,
    )
    assert got["volume"] == 1366914
    # ka10080 에 거래대금 필드가 없어서 close × volume 근사다.
    assert got["trade_value"] == pytest.approx(274500 * 1366914)


# ------------------------------------------------------------------ 분봉 페이지네이션 (minute_chart 자체 로직)
#
# fetch_page 를 몽키패치해서 minute_chart 의 집계·중단 로직만 검증한다 — HTTP는
# 안 탄다(그 경계는 아래 "fetch_page → kiwoom_client 위임" 섹션에서 따로 본다).


def _row_at(hhmmss: str) -> dict:
    return dict(_ROW, cntr_tm=f"202608{hhmmss}")


def test_cont_yn_y_requests_next_page_with_next_key(monkeypatch):
    pages_seen = []

    def fake_fetch_page(self, code, *, tic_scope="1", cont_yn=None, next_key=None):
        pages_seen.append(next_key)
        if len(pages_seen) == 1:
            return [_row_at("14153000")], "Y", "NK1"
        return [_row_at("14152900")], "N", ""

    monkeypatch.setattr(KiwoomClient, "fetch_page", fake_fetch_page)
    client = _make(_router(lambda r: _rank([])))
    rows = client.minute_chart("005930", pages=5)
    assert pages_seen == [None, "NK1"]
    assert len(rows) == 2


def test_cont_yn_not_y_stops_pagination(monkeypatch):
    calls = []

    def fake_fetch_page(self, code, *, tic_scope="1", cont_yn=None, next_key=None):
        calls.append(1)
        return [_row_at("14153000")], "N", "NK1"

    monkeypatch.setattr(KiwoomClient, "fetch_page", fake_fetch_page)
    client = _make(_router(lambda r: _rank([])))
    client.minute_chart("005930", pages=10)
    assert len(calls) == 1


def test_pages_limit_caps_call_count(monkeypatch):
    calls = []

    def fake_fetch_page(self, code, *, tic_scope="1", cont_yn=None, next_key=None):
        calls.append(1)
        return [_row_at(f"14{15 - len(calls):02d}0000")], "Y", f"NK{len(calls)}"

    monkeypatch.setattr(KiwoomClient, "fetch_page", fake_fetch_page)
    client = _make(_router(lambda r: _rank([])))
    client.minute_chart("005930", pages=3)
    assert len(calls) == 3


def test_since_stops_the_stock_when_older_data_arrives(monkeypatch):
    def fake_fetch_page(self, code, *, tic_scope="1", cont_yn=None, next_key=None):
        rows = [
            _row_at("14153000"),
            dict(_ROW, cntr_tm="20260813153000"),
            dict(_ROW, cntr_tm="20260810153000"),  # since 이전
        ]
        return rows, "Y", "NK1"

    monkeypatch.setattr(KiwoomClient, "fetch_page", fake_fetch_page)
    client = _make(_router(lambda r: _rank([])))
    rows = client.minute_chart("005930", pages=10, since="20260812")
    assert [r["ts"].strftime("%Y%m%d") for r in rows] == ["20260814", "20260813"]


def test_stop_at_resumes_from_stored_latest_ts(monkeypatch):
    def fake_fetch_page(self, code, *, tic_scope="1", cont_yn=None, next_key=None):
        rows = [
            dict(_ROW, cntr_tm="20260814153000"),
            dict(_ROW, cntr_tm="20260814152900"),
            dict(_ROW, cntr_tm="20260814152800"),
        ]
        return rows, "Y", "NK1"

    monkeypatch.setattr(KiwoomClient, "fetch_page", fake_fetch_page)
    client = _make(_router(lambda r: _rank([])))
    rows = client.minute_chart(
        "005930", pages=10, stop_at=pd.Timestamp(2026, 8, 14, 15, 29)
    )
    assert [r["ts"].minute for r in rows] == [30]


def test_blank_cntr_tm_rows_are_skipped_not_fatal(monkeypatch, capsys):
    """빈 ``cntr_tm`` 행은 건너뛰고 세기만 한다 — 그 종목 전체를 날리지 않는다.

    2026-10-12 감사: 008290 등 6종목이 빈 체결시각 한 줄 때문에 몇 주째 통째 실패했다.
    """
    def fake_fetch_page(self, code, *, tic_scope="1", cont_yn=None, next_key=None):
        rows = [
            _row_at("14153000"),
            dict(_ROW, cntr_tm=""),
            dict(_ROW, cntr_tm=None),
            _row_at("14152800"),
        ]
        return rows, "N", ""

    monkeypatch.setattr(KiwoomClient, "fetch_page", fake_fetch_page)
    client = _make(_router(lambda r: _rank([])))
    rows = client.minute_chart("008290", pages=5)
    assert [r["ts"].minute for r in rows] == [30, 28]
    err = capsys.readouterr().err
    assert err.count("008290: cntr_tm 빈 행 2개 건너뜀") == 1


def test_malformed_nonblank_cntr_tm_still_raises(monkeypatch):
    """비어 있지 않은데 길이가 틀린 값은 엉뚱한 시각이 될 수 있어 여전히 던진다."""
    monkeypatch.setattr(
        KiwoomClient, "fetch_page",
        lambda self, code, *, tic_scope="1", cont_yn=None, next_key=None:
            ([dict(_ROW, cntr_tm="2026081415")], "N", ""),
    )
    client = _make(_router(lambda r: _rank([])))
    with pytest.raises(KiwoomError, match="체결시각"):
        client.minute_chart("005930", pages=1)


def test_empty_payload_is_not_an_error(monkeypatch):
    """신규 상장·거래정지 종목은 실제로 빈 배열이 온다."""
    monkeypatch.setattr(
        KiwoomClient, "fetch_page",
        lambda self, code, *, tic_scope="1", cont_yn=None, next_key=None: ([], "N", ""),
    )
    client = _make(_router(lambda r: _rank([])))
    assert client.minute_chart("999999", pages=5) == []


def test_null_rows_key_is_not_an_error():
    """``fetch_page`` → ``_via_kiwoom_api`` 는 rows_key 가 ``None`` 이어도 빈 리스트를 돌려준다."""
    fake_api = MagicMock()
    fake_api.chart.stock_minute_chart.return_value = {
        "stk_cd": "005930", CHART_ROWS_KEY: None, "return_code": 0, "return_msg": "정상",
        "cont-yn": "N", "next-key": "",
    }
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    assert client.minute_chart("005930", pages=2) == []


# ------------------------------------------------------------------ fetch_page → kiwoom_client 위임
#
# 분봉·테마는 이제 kiwoom_client.KiwoomAPI 를 거친다 — 그 경계에서 바로 목을
# 걸어 (1) 파라미터가 맞게 전달되는지, (2) cont-yn/next-key 를 응답에서 제대로
# 뽑는지, (3) KiwoomAPIError 가 이 레포의 KiwoomError 로 바뀌는지를 본다.


def test_fetch_page_maps_kiwoom_client_response_and_pagination():
    fake_api = MagicMock()
    fake_api.chart.stock_minute_chart.return_value = {
        CHART_ROWS_KEY: [_ROW], "cont-yn": "Y", "next-key": "NK1",
    }
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    rows, cont, key = client.fetch_page("005930", tic_scope="3", next_key="prev")
    assert rows == [_ROW]
    assert (cont, key) == ("Y", "NK1")
    fake_api.chart.stock_minute_chart.assert_called_once_with(
        cont_yn="N", next_key="prev", stk_cd="005930", tic_scope="3", upd_stkpc_tp="1",
    )


def test_fetch_page_wraps_kiwoom_api_error():
    fake_api = MagicMock()
    fake_api.chart.stock_minute_chart.side_effect = KiwoomAPIError(3, "종목코드 오류")
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    with pytest.raises(KiwoomError, match="종목코드 오류"):
        client.fetch_page("BAD")


def test_fetch_theme_groups_page_maps_response():
    fake_api = MagicMock()
    fake_api.theme.theme_group_list.return_value = {
        THEME_GROUP_ROWS_KEY: [{"thema_grp_cd": "001"}], "cont-yn": "N", "next-key": "",
    }
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    rows, cont, key = client.fetch_theme_groups_page()
    assert rows == [{"thema_grp_cd": "001"}]
    assert (cont, key) == ("N", "")
    fake_api.theme.theme_group_list.assert_called_once_with(
        cont_yn="N", next_key="", qry_tp="0", stk_cd="", date_tp="1", thema_nm="",
        flu_pl_amt_tp="1", stex_tp="1",
    )


def test_fetch_theme_members_page_maps_response():
    fake_api = MagicMock()
    fake_api.theme.theme_component_stocks.return_value = {
        THEME_MEMBER_ROWS_KEY: [{"stk_cd": "005930"}], "cont-yn": "N", "next-key": "",
    }
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    rows, _cont, _key = client.fetch_theme_members_page("001")
    assert rows == [{"stk_cd": "005930"}]
    fake_api.theme.theme_component_stocks.assert_called_once_with(
        cont_yn="N", next_key="", thema_grp_cd="001", date_tp="2", stex_tp="1",
    )


# ------------------------------------------------------------------ 토큰 발급 오류
#
# 429/5xx 재시도·401 갱신재시도·레이트리밋 sleep 은 2026-09-12 이전엔
# ``_fetch_rows``(조회 5종 공용)가 직접 구현했지만, 다섯 다 kiwoom_client로
# 옮기면서 그 구현 자체를 지웠다 — 이제 그 재시도는 kiwoom-client 쪽 책임이고
# (kiwoom-client 저장소 tests/test_base.py 에 이미 있다), 여기서 다시 재는 건
# 남의 코드를 목(mock)으로 재검증하는 것뿐이라 값이 없다. 토큰 발급
# (``issue_token``, ``OrderExecutor``·WS 로그인이 직접 쓰는 그 메서드) 자체의
# 오류 처리만 여기 남긴다.


def test_issue_token_nonzero_return_code_raises_with_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"return_code": 3, "return_msg": "인증키 오류"})

    with pytest.raises(KiwoomError, match="인증키 오류"):
        _make(handler).issue_token()


# ------------------------------------------------------------------ fetch_rank_rows/fetch_program_rows → kiwoom_client 위임


def test_fetch_rank_rows_maps_request_and_response():
    fake_api = MagicMock()
    fake_api.ranking.top_change_rate.return_value = {_RANK_ROWS_KEY: [_RANK_ROW]}
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    rows = client.fetch_rank_rows(market="001")
    assert rows == [_RANK_ROW]
    fake_api.ranking.top_change_rate.assert_called_once_with(
        mrkt_tp="001", sort_tp="1", trde_qty_cnd="0000", stk_cnd="0",
        crd_cnd="0", updown_incls="1", pric_cnd="0", trde_prica_cnd="0", stex_tp="3",
    )


def test_fetch_rank_rows_wraps_kiwoom_api_error():
    fake_api = MagicMock()
    fake_api.ranking.top_change_rate.side_effect = KiwoomAPIError(10, "요청 실패")
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    with pytest.raises(KiwoomError, match="요청 실패"):
        client.fetch_rank_rows()


def test_fetch_program_rows_maps_request_and_response():
    fake_api = MagicMock()
    fake_api.market.program_trading_by_time.return_value = {
        "prm_trde_trnsn": [{"stk_cd": "005930"}],
    }
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    rows = client.fetch_program_rows(day="20260912", market="P001")
    assert rows == [{"stk_cd": "005930"}]
    fake_api.market.program_trading_by_time.assert_called_once_with(
        date="20260912", amt_qty_tp="1", mrkt_tp="P001", min_tic_tp="1", stex_tp="3",
    )
