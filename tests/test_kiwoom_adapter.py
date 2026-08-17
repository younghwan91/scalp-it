"""키움 어댑터 테스트. 네트워크는 절대 타지 않는다 — 전부 MockTransport."""

import httpx
import pandas as pd
import pytest

from scalp_it.adapters.kiwoom import (
    CHART_ROWS_KEY,
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

#: 실제 응답에서 그대로 가져온 한 행.
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


def _chart(rows, *, cont_yn="N", next_key="", return_code=0):
    body = {
        "stk_cd": "005930",
        CHART_ROWS_KEY: rows,
        "return_code": return_code,
        "return_msg": "정상적으로 처리되었습니다",
    }
    return httpx.Response(200, json=body, headers={"cont-yn": cont_yn, "next-key": next_key})


def _make(handler, **kwargs):
    return KiwoomClient(
        "key", "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _s: None,
        **kwargs,
    )


def _router(chart_handler):
    """토큰 요청은 공통 처리하고 차트 요청만 넘긴다."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/token":
            return httpx.Response(200, json=_TOKEN_BODY)
        return chart_handler(request)

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
    client = _make(_router(lambda r: _chart([])))
    assert client.issue_token() == "tok-abc"


def test_token_missing_key_raises():
    body = {"return_code": 0, "access_token": "wrong-key"}
    client = _make(lambda r: httpx.Response(200, json=body))
    with pytest.raises(KiwoomError):
        client.issue_token()


def test_token_is_reused_across_calls():
    calls = {"token": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/token":
            calls["token"] += 1
            return httpx.Response(200, json=_TOKEN_BODY)
        return _chart([_ROW])

    client = _make(handler)
    client.minute_chart("005930", pages=1)
    client.minute_chart("000660", pages=1)
    assert calls["token"] == 1


def test_bearer_token_and_api_id_go_in_headers():
    seen = {}

    def chart(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return _chart([_ROW])

    _make(_router(chart)).minute_chart("005930", pages=1)
    assert seen["authorization"] == "Bearer tok-abc"
    assert seen["api-id"] == "ka10080"


# ------------------------------------------------------------------ 파싱


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


# ------------------------------------------------------------------ 페이지네이션


def test_cont_yn_y_requests_next_page_with_next_key():
    seen_keys = []

    def chart(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers.get("next-key"))
        page = len(seen_keys)
        row = dict(_ROW, cntr_tm=f"2026081415{30 - page:02d}00")
        if page == 1:
            return _chart([row], cont_yn="Y", next_key="NK1")
        return _chart([row], cont_yn="N", next_key="")

    rows = _make(_router(chart)).minute_chart("005930", pages=5)
    assert seen_keys == [None, "NK1"]
    assert len(rows) == 2


def test_cont_yn_not_y_stops_pagination():
    calls = []

    def chart(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _chart([_ROW], cont_yn="N", next_key="NK1")

    _make(_router(chart)).minute_chart("005930", pages=10)
    assert len(calls) == 1


def test_pages_limit_caps_call_count():
    calls = []

    def chart(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        row = dict(_ROW, cntr_tm=f"202608141{len(calls):02d}000"[:14])
        return _chart([row], cont_yn="Y", next_key=f"NK{len(calls)}")

    _make(_router(chart)).minute_chart("005930", pages=3)
    assert len(calls) == 3


def test_since_stops_the_stock_when_older_data_arrives():
    def chart(request: httpx.Request) -> httpx.Response:
        rows = [
            dict(_ROW, cntr_tm="20260814153000"),
            dict(_ROW, cntr_tm="20260813153000"),
            dict(_ROW, cntr_tm="20260810153000"),  # since 이전
        ]
        return _chart(rows, cont_yn="Y", next_key="NK1")

    rows = _make(_router(chart)).minute_chart("005930", pages=10, since="20260812")
    assert [r["ts"].strftime("%Y%m%d") for r in rows] == ["20260814", "20260813"]


def test_stop_at_resumes_from_stored_latest_ts():
    def chart(request: httpx.Request) -> httpx.Response:
        rows = [
            dict(_ROW, cntr_tm="20260814153000"),
            dict(_ROW, cntr_tm="20260814152900"),
            dict(_ROW, cntr_tm="20260814152800"),
        ]
        return _chart(rows, cont_yn="Y", next_key="NK1")

    rows = _make(_router(chart)).minute_chart(
        "005930", pages=10, stop_at=pd.Timestamp(2026, 8, 14, 15, 29)
    )
    assert [r["ts"].minute for r in rows] == [30]


def test_empty_payload_is_not_an_error():
    """신규 상장·거래정지 종목은 실제로 빈 배열이 온다."""
    rows = _make(_router(lambda r: _chart([]))).minute_chart("999999", pages=5)
    assert rows == []


def test_null_rows_key_is_not_an_error():
    body = {"stk_cd": "005930", CHART_ROWS_KEY: None, "return_code": 0, "return_msg": "정상"}
    chart = httpx.Response(200, json=body, headers={"cont-yn": "N"})
    assert _make(_router(lambda r: chart)).minute_chart("005930", pages=2) == []


# ------------------------------------------------------------------ 오류·레이트리밋


def test_nonzero_return_code_raises_with_message():
    def chart(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"return_code": 3, "return_msg": "종목코드 오류"})

    with pytest.raises(KiwoomError, match="종목코드 오류"):
        _make(_router(chart)).minute_chart("BAD", pages=1)


def test_429_is_retried_then_succeeds():
    calls = []

    def chart(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="too many")
        return _chart([_ROW])

    rows = _make(_router(chart), max_retries=3).minute_chart("005930", pages=1)
    assert len(calls) == 2
    assert len(rows) == 1


def test_retries_exhausted_raises():
    with pytest.raises(KiwoomError):
        _make(
            _router(lambda r: httpx.Response(503, text="down")), max_retries=2
        ).minute_chart("005930", pages=1)


def test_401_triggers_token_refresh_and_retry():
    state = {"tokens": 0, "charts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/token":
            state["tokens"] += 1
            return httpx.Response(200, json=_TOKEN_BODY)
        state["charts"] += 1
        if state["charts"] == 1:
            return httpx.Response(401, text="expired")
        return _chart([_ROW])

    rows = _make(handler, max_retries=3).minute_chart("005930", pages=1)
    assert state["tokens"] == 2
    assert len(rows) == 1


def test_rate_limit_sleep_happens_between_calls():
    naps: list[float] = []
    calls = []

    def chart(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        cont = "Y" if len(calls) < 3 else "N"
        row = dict(_ROW, cntr_tm=f"2026081415{30 - len(calls):02d}00")
        return _chart([row], cont_yn=cont, next_key=f"NK{len(calls)}")

    client = KiwoomClient(
        "key", "secret",
        client=httpx.Client(transport=httpx.MockTransport(_router(chart))),
        sleep=0.12,
        sleeper=naps.append,
    )
    client.minute_chart("005930", pages=5)
    assert len(calls) == 3
    assert naps == [0.12, 0.12, 0.12]


def test_zero_sleep_skips_the_nap():
    naps: list[float] = []
    client = KiwoomClient(
        "key", "secret",
        client=httpx.Client(transport=httpx.MockTransport(_router(lambda r: _chart([_ROW])))),
        sleep=0.0,
        sleeper=naps.append,
    )
    client.minute_chart("005930", pages=1)
    assert naps == []
