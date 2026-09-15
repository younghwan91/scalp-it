import pytest

from scalp_it.adapters.dart import DartClient, DartError, api_keys

_ITEM = {
    "corp_cls": "K", "corp_name": "에프알텍", "corp_code": "00123456",
    "stock_code": "073540", "report_nm": "단일판매ㆍ공급계약체결",
    "rcept_no": "20260814000123", "flr_nm": "에프알텍", "rcept_dt": "20260814", "rm": "",
}


def _page(items, total_page=1, status="000"):
    return {
        "status": status,
        "message": "정상" if status == "000" else "오류",
        "page_no": 1,
        "page_count": 100,
        "total_count": len(items),
        "total_page": total_page,
        "list": items,
    }


def test_api_keys_reads_all_three_in_order():
    env = {"DART_API_KEY": "a", "DART_API_KEY_2": "b", "DART_API_KEY_3": "c"}
    assert api_keys(env) == ["a", "b", "c"]


def test_api_keys_skips_missing_and_blank():
    env = {"DART_API_KEY": "a", "DART_API_KEY_2": "", "DART_API_KEY_3": "c"}
    assert api_keys(env) == ["a", "c"]


def test_api_keys_raises_when_none_set():
    with pytest.raises(DartError):
        api_keys({})


def test_search_returns_payload(httpx_mock):
    httpx_mock.add_response(json=_page([_ITEM]))
    client = DartClient(["k1"])
    got = client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert got["total_count"] == 1
    assert got["list"][0]["stock_code"] == "073540"


def test_search_sends_expected_params(httpx_mock):
    httpx_mock.add_response(json=_page([]))
    client = DartClient(["k1"])
    client.search(bgn_de="20260101", end_de="20260331", corp_cls="Y", page_no=3)
    request = httpx_mock.get_requests()[0]
    seen = dict(request.url.params)
    assert seen["bgn_de"] == "20260101"
    assert seen["end_de"] == "20260331"
    assert seen["corp_cls"] == "Y"
    assert seen["page_no"] == "3"
    assert seen["page_count"] == "100"
    assert seen["crtfc_key"] == "k1"


def test_status_013_is_empty_not_error(httpx_mock):
    """조회 데이터 없음은 오류가 아니다. 공시 없는 날이 실제로 있다."""
    httpx_mock.add_response(json={"status": "013", "message": "조회된 데이터가 없습니다."})
    client = DartClient(["k1"])
    assert client.search(bgn_de="20260101", end_de="20260101", corp_cls="Y")["list"] == []


def test_other_error_status_raises(httpx_mock):
    httpx_mock.add_response(json={"status": "010", "message": "등록되지 않은 키입니다."})
    client = DartClient(["k1"])
    with pytest.raises(DartError):
        client.search(bgn_de="20260101", end_de="20260101", corp_cls="Y")


def test_rate_limited_key_falls_through_to_next(httpx_mock):
    httpx_mock.add_response(json={"status": "020", "message": "요청 제한을 초과하였습니다."})
    httpx_mock.add_response(json=_page([_ITEM]))
    client = DartClient(["k1", "k2"])
    got = client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert got["total_count"] == 1
    used = [dict(r.url.params)["crtfc_key"] for r in httpx_mock.get_requests()]
    assert used == ["k1", "k2"]


def test_all_keys_rate_limited_raises(httpx_mock):
    httpx_mock.add_response(json={"status": "020", "message": "초과"})
    httpx_mock.add_response(json={"status": "020", "message": "초과"})
    client = DartClient(["k1", "k2"])
    with pytest.raises(DartError):
        client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")


def test_search_all_walks_every_page(httpx_mock):
    for page in (1, 2, 3):
        item = dict(_ITEM, rcept_no=f"2026081400{page:04d}")
        httpx_mock.add_response(json=_page([item], total_page=3))
    client = DartClient(["k1"])
    rows = client.search_all(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert len(rows) == 3
    assert [r["rcept_no"][-4:] for r in rows] == ["0001", "0002", "0003"]


def test_consecutive_calls_on_one_client_do_not_reuse_a_dead_event_loop(httpx_mock):
    """cli_dart 는 한 DartClient 로 법인구분 Y 다음 K 를 부른다.

    호출마다 ``asyncio.run`` 이 새 루프를 여는데 스크레이퍼의 httpx.AsyncClient 가 첫
    루프에 묶여 남아 있으면 두 번째 호출이 ``Event loop is closed`` 로 죽는다 —
    2026-09-14 재현: 코스피(Y)만 적재되고 코스닥(K) 공시가 통째로 빠졌다.
    """
    httpx_mock.add_response(json=_page([_ITEM]))
    httpx_mock.add_response(json=_page([_ITEM]))
    httpx_mock.add_response(json=_page([_ITEM]))
    client = DartClient(["k1"])
    assert len(client.search_all(bgn_de="20260907", end_de="20260914", corp_cls="Y")) == 1
    assert len(client.search_all(bgn_de="20260907", end_de="20260914", corp_cls="K")) == 1
    assert client.search(bgn_de="20260907", end_de="20260914", corp_cls="K")["total_count"] == 1
    # mock 전송은 루프에 안 묶여 위 호출만으로는 재현이 안 된다 — 원인 자체를 막는다:
    # 호출이 끝나면 루프에 묶인 AsyncClient 가 남아 있으면 안 된다.
    assert client._scraper._client is None


def test_library_rebuilds_client_when_event_loop_changes(httpx_mock):
    """krx-news-client 0.4.0 이 루프가 바뀌면 httpx 클라이언트를 새로 만든다.

    파사드의 ``async with`` 닫기가 빠져도 두 번째 ``asyncio.run`` 이 죽은 루프의
    클라이언트를 재사용하지 않는다는 것 — 0.4.0 미만으로 내려가면 여기서 깨진다.
    """
    import asyncio

    from krx_news_client import DartScraper

    httpx_mock.add_response(json=_page([_ITEM]))
    httpx_mock.add_response(json=_page([_ITEM]))
    scraper = DartScraper(api_key="k1")

    async def call():
        await scraper.search_disclosures(bgn_de="20260907", end_de="20260914", corp_cls="Y")
        return scraper._client

    first = asyncio.run(call())  # 일부러 close 하지 않는다
    second = asyncio.run(call())
    assert first is not None and second is not None
    assert second is not first


def test_api_keys_error_message_is_unchanged():
    with pytest.raises(DartError, match="DART 인증키가 없다"):
        api_keys({"DART_API_KEY": "  ", "DART_API_KEY_2": ""})
