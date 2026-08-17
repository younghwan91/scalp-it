import httpx
import pytest

from scalp_it.adapters.dart import DartClient, DartError, api_keys


def _resp(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


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


_ITEM = {
    "corp_cls": "K", "corp_name": "에프알텍", "corp_code": "00123456",
    "stock_code": "073540", "report_nm": "단일판매ㆍ공급계약체결",
    "rcept_no": "20260814000123", "flr_nm": "에프알텍", "rcept_dt": "20260814", "rm": "",
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


def test_search_returns_payload():
    transport = httpx.MockTransport(lambda request: _resp(_page([_ITEM])))
    client = DartClient(["k1"], client=httpx.Client(transport=transport))
    got = client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert got["total_count"] == 1
    assert got["list"][0]["stock_code"] == "073540"


def test_search_sends_expected_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return _resp(_page([]))

    client = DartClient(["k1"], client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.search(bgn_de="20260101", end_de="20260331", corp_cls="Y", page_no=3)
    assert seen["bgn_de"] == "20260101"
    assert seen["end_de"] == "20260331"
    assert seen["corp_cls"] == "Y"
    assert seen["page_no"] == "3"
    assert seen["page_count"] == "100"
    assert seen["crtfc_key"] == "k1"


def test_status_013_is_empty_not_error():
    """조회 데이터 없음은 오류가 아니다. 공시 없는 날이 실제로 있다."""
    payload = {"status": "013", "message": "조회된 데이터가 없습니다."}
    client = DartClient(["k1"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: _resp(payload))))
    assert client.search(bgn_de="20260101", end_de="20260101", corp_cls="Y")["list"] == []


def test_other_error_status_raises():
    payload = {"status": "010", "message": "등록되지 않은 키입니다."}
    client = DartClient(["k1"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: _resp(payload))))
    with pytest.raises(DartError):
        client.search(bgn_de="20260101", end_de="20260101", corp_cls="Y")


def test_rate_limited_key_falls_through_to_next():
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.params["crtfc_key"]
        used.append(key)
        if key == "k1":
            return _resp({"status": "020", "message": "요청 제한을 초과하였습니다."})
        return _resp(_page([_ITEM]))

    client = DartClient(["k1", "k2"], client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    got = client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert got["total_count"] == 1
    assert used == ["k1", "k2"]


def test_all_keys_rate_limited_raises():
    client = DartClient(["k1", "k2"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: _resp({"status": "020", "message": "초과"}))))
    with pytest.raises(DartError):
        client.search(bgn_de="20260810", end_de="20260814", corp_cls="K")


def test_search_all_walks_every_page():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page_no"])
        item = dict(_ITEM, rcept_no=f"2026081400{page:04d}")
        return _resp(_page([item], total_page=3))

    client = DartClient(["k1"], client=httpx.Client(transport=httpx.MockTransport(handler)))
    rows = client.search_all(bgn_de="20260810", end_de="20260814", corp_cls="K")
    assert len(rows) == 3
    assert [r["rcept_no"][-4:] for r in rows] == ["0001", "0002", "0003"]
