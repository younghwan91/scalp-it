"""테마 스냅샷 — 어댑터(ka90001/ka90002) 파싱·페이지네이션, 저장소 upsert, CLI.

네트워크는 타지 않는다(MockTransport). 응답 행은 2026-08-23 실측 원문 그대로다.
"""

import json
import sqlite3
from unittest.mock import MagicMock

import httpx
import pytest
from kiwoom_client import KiwoomAPI

from scalp_it import cli_theme
from scalp_it.adapters.kiwoom import (
    THEME_GROUP_ROWS_KEY,
    THEME_MEMBER_ROWS_KEY,
    KiwoomClient,
    KiwoomError,
    normalize_theme_group,
    normalize_theme_member,
    parse_rate,
)
from scalp_it.theme.store import (
    _CREATE_GROUPS,
    _CREATE_MEMBERS,
    count_groups,
    count_members,
    init_theme_tables,
    normalize_snapshot_date,
    upsert_theme_groups,
    upsert_theme_members,
)

_TOKEN_BODY = {"return_code": 0, "return_msg": "정상", "token_type": "bearer", "token": "tok-abc"}

#: ka90001 실측 행.
_GROUP_ROW = {
    "thema_grp_cd": "141", "thema_nm": "2차전지_소재(양극화물질등)", "stk_num": "14",
    "flu_sig": "5", "flu_rt": "-3.74", "rising_stk_num": "2", "fall_stk_num": "12",
    "dt_prft_rt": "+25.22", "main_stk": "보유B, 티엔엔터테인먼트",
}
_GROUP_ROW2 = dict(_GROUP_ROW, thema_grp_cd="420", thema_nm="보험_생명보험", stk_num="3",
                   flu_sig="2", flu_rt="+3.29", rising_stk_num="1", fall_stk_num="1",
                   dt_prft_rt="+3.29", main_stk="삼성생명, 동양생명")
#: ka90002 실측 행.
_MEMBER_ROW = {
    "stk_cd": "000020", "stk_nm": "보유B", "cur_prc": "+116200", "flu_sig": "2",
    "pred_pre": "+2600", "flu_rt": "+2.29", "acc_trde_qty": "575218", "sel_bid": "+116200",
    "sel_req": "153", "buy_bid": "+116100", "buy_req": "627", "dt_prft_rt_n": "-2.92",
}
_MEMBER_ROW2 = dict(_MEMBER_ROW, stk_cd="131100", stk_nm="티엔엔터테인먼트", cur_prc="+2280",
                    flu_rt="+1.33", acc_trde_qty="15491", dt_prft_rt_n="+406.67")


def _groups_resp(rows, *, cont_yn="N", next_key=""):
    body = {THEME_GROUP_ROWS_KEY: rows, "return_code": 0, "return_msg": "정상"}
    return httpx.Response(200, json=body, headers={"cont-yn": cont_yn, "next-key": next_key})


def _members_resp(rows, *, cont_yn="N", next_key=""):
    body = {"flu_rt": "-3.74", "dt_prft_rt": "+26.32", THEME_MEMBER_ROWS_KEY: rows,
            "return_code": 0, "return_msg": "정상"}
    return httpx.Response(200, json=body, headers={"cont-yn": cont_yn, "next-key": next_key})


def _groups_page(rows, *, cont_yn="N", next_key=""):
    """``kiwoom_client`` 경계 목(mock)용 — ka90001 한 페이지."""
    return {THEME_GROUP_ROWS_KEY: rows, "cont-yn": cont_yn, "next-key": next_key}


def _members_page(rows, *, cont_yn="N", next_key=""):
    """``kiwoom_client`` 경계 목(mock)용 — ka90002 한 페이지."""
    return {THEME_MEMBER_ROWS_KEY: rows, "cont-yn": cont_yn, "next-key": next_key}


def _fake_kiwoom_api(*, group_pages=None, member_pages=None):
    api = MagicMock()
    if group_pages is not None:
        api.theme.theme_group_list.side_effect = group_pages
    if member_pages is not None:
        api.theme.theme_component_stocks.side_effect = member_pages
    return api


def _router(theme_handler):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth2/token":
            return httpx.Response(200, json=_TOKEN_BODY)
        assert request.url.path == "/api/dostk/thme"
        return theme_handler(request)

    return handler


def _make(handler, **kwargs):
    return KiwoomClient(
        "key", "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _s: None,
        **kwargs,
    )


def _real_api_on_mock_wire(handler):
    """**진짜** ``KiwoomAPI`` 를 가짜 전선 위에 올린다.

    위 ``_fake_kiwoom_api`` 는 kiwoom_client 경계에서 목을 걸기 때문에 cont-yn 이
    payload 에 있다고 가정한다. 그런데 키움 실제 응답은 cont-yn/next-key 를
    **헤더로만** 주고, 그걸 바디로 옮겨 담는 건 kiwoom_client ``request()`` 의
    책임이다. 즉 경계 목만 있으면 그 폴백이 사라져도(핀이 낡아도) 테스트는
    초록불이고 연속조회는 조용히 1페이지에서 끝난다 — 2026-09-13 에 실제로
    그랬다(핀 1debc40 이 폴백 커밋 492aa60 의 한 커밋 앞이었다).

    그래서 여기서는 라이브러리를 목하지 않고 **HTTP 전선만** 갈아끼운다.
    ``KiwoomAPI`` 와 ``KiwoomAuth`` 가 각자 httpx 클라이언트를 들고 있어 둘 다
    바꿔야 한다(사설 속성 접근은 이 계약 테스트의 대가로 감수한다 —
    kiwoom_client 는 transport 주입구를 아직 안 열어놨다).
    """
    api = KiwoomAPI("key", "secret", base_url="https://mockapi.kiwoom.com", rate_limit=None)
    wire = httpx.MockTransport(handler)
    for owner in (api._base_client, api._auth):
        owner._client = httpx.Client(base_url=owner.base_url, transport=wire, timeout=30.0)
    return api


# ------------------------------------------------------------------ 어댑터 파싱


@pytest.mark.parametrize(
    ("raw", "expected"), [("+25.22", 25.22), ("-3.74", -3.74), ("0.00", 0.0), ("", None)]
)
def test_parse_rate_keeps_real_sign(raw, expected):
    """등락률 부호는 가격과 달리 실제 음양이다."""
    assert parse_rate(raw) == expected


def test_normalize_theme_group_maps_fields_and_keeps_raw():
    got = normalize_theme_group(_GROUP_ROW)
    assert got["theme_code"] == "141"
    assert got["theme_name"] == "2차전지_소재(양극화물질등)"
    assert got["stk_num"] == 14
    assert got["flu_rt"] == -3.74
    assert got["dt_prft_rt"] == 25.22
    assert (got["rising_stk_num"], got["fall_stk_num"]) == (2, 12)
    assert got["main_stk"] == "보유B, 티엔엔터테인먼트"
    assert got["raw"] == _GROUP_ROW


def test_normalize_theme_member_strips_price_sign_but_keeps_rate_sign():
    got = normalize_theme_member("141", _MEMBER_ROW)
    assert got["theme_code"] == "141"
    assert got["code"] == "000020"
    assert got["name"] == "보유B"
    assert got["cur_prc"] == 116200
    assert got["flu_rt"] == 2.29
    assert got["acc_trde_qty"] == 575218
    assert got["dt_prft_rt_n"] == -2.92
    assert got["raw"]["sel_bid"] == "+116200"


def test_normalize_theme_group_without_code_raises():
    with pytest.raises(KiwoomError):
        normalize_theme_group({"thema_nm": "코드 없음"})


# ------------------------------------------------------------------ 어댑터 호출


def test_theme_groups_sends_ka90001_body_and_follows_next_key():
    fake_api = _fake_kiwoom_api(group_pages=[
        _groups_page([_GROUP_ROW], cont_yn="Y", next_key="00100"),
        _groups_page([_GROUP_ROW2, _GROUP_ROW], cont_yn="N"),  # 경계 중복 포함
    ])
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    groups = client.theme_groups()

    calls = fake_api.theme.theme_group_list.call_args_list
    assert [c.kwargs["next_key"] for c in calls] == ["", "00100"]
    assert calls[0].kwargs == {
        "cont_yn": "N", "next_key": "", "qry_tp": "0", "stk_cd": "", "date_tp": "1",
        "thema_nm": "", "flu_pl_amt_tp": "1", "stex_tp": "1",
    }
    assert [g["theme_code"] for g in groups] == ["141", "420"]


def test_theme_members_sends_ka90002_body():
    fake_api = _fake_kiwoom_api(member_pages=[_members_page([_MEMBER_ROW, _MEMBER_ROW2])])
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    members = client.theme_members("141")

    fake_api.theme.theme_component_stocks.assert_called_once_with(
        cont_yn="N", next_key="", thema_grp_cd="141", date_tp="2", stex_tp="1",
    )
    assert [m["code"] for m in members] == ["000020", "131100"]
    assert all(m["theme_code"] == "141" for m in members)


def test_theme_members_empty_is_not_an_error():
    fake_api = _fake_kiwoom_api(member_pages=[_members_page([])])
    client = KiwoomClient("key", "secret", kiwoom_api=fake_api)
    assert client.theme_members("999") == []


# --------------------------------------------------- kiwoom-client 의존성 계약
# 핀이 너무 낡으면 여기서 잡힌다. 위의 경계 목 테스트들은 못 잡는다.


def test_theme_groups_follow_next_key_when_cont_yn_only_in_headers():
    """헤더에만 cont-yn 이 온 실제 응답에서도 2페이지를 따라가야 한다.

    ka90001 은 한 페이지 100행이고 `cont-yn: Y` 로 이어진다. 이걸 못 따라가면
    테마 그룹이 첫 100개로 잘린 채 짝꿍 선정 입력이 된다 — 에러도 경고도 없다.
    핀을 되돌려 보면(1debc40) 이 테스트만 빨간불이 된다.
    """
    pages = [
        _groups_resp([_GROUP_ROW], cont_yn="Y", next_key="00100"),
        _groups_resp([_GROUP_ROW2], cont_yn="N"),
    ]
    seen_next_keys = []

    def theme_handler(request: httpx.Request) -> httpx.Response:
        seen_next_keys.append(request.headers.get("next-key", ""))
        return pages[len(seen_next_keys) - 1]

    client = KiwoomClient("key", "secret", kiwoom_api=_real_api_on_mock_wire(_router(theme_handler)))
    groups = client.theme_groups()

    assert seen_next_keys == ["", "00100"], "2페이지를 안 따라갔다 — 핀이 폴백 이전이다"
    assert [g["theme_code"] for g in groups] == ["141", "420"]


# ------------------------------------------------------------------ 저장소


@pytest.fixture()
def conn(tmp_path):
    connection = sqlite3.connect(tmp_path / "theme.db")
    init_theme_tables(connection)
    yield connection
    connection.close()


def test_schema_uses_only_portable_types():
    for ddl in (_CREATE_GROUPS, _CREATE_MEMBERS):
        lowered = ddl.lower()
        assert "autoincrement" not in lowered
        assert "serial" not in lowered
        assert "snapshot_date" in lowered


@pytest.mark.parametrize("value", ["20260823", "2026-08-23"])
def test_normalize_snapshot_date_accepts_both_forms(value):
    assert normalize_snapshot_date(value) == "2026-08-23"


def test_normalize_snapshot_date_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_snapshot_date("2026-8-23")


def test_upsert_groups_is_idempotent_and_keyed_by_date(conn):
    rows = [normalize_theme_group(_GROUP_ROW), normalize_theme_group(_GROUP_ROW2)]
    assert upsert_theme_groups(conn, "20260823", rows) == 2
    upsert_theme_groups(conn, "20260823", rows)
    upsert_theme_groups(conn, "20260824", rows)
    assert count_groups(conn) == 4
    assert count_groups(conn, "2026-08-23") == 2


def test_upsert_groups_overwrites_and_keeps_raw_json(conn):
    upsert_theme_groups(conn, "20260823", [normalize_theme_group(_GROUP_ROW)])
    changed = normalize_theme_group(dict(_GROUP_ROW, thema_nm="이름변경", flu_rt="+1.00"))
    upsert_theme_groups(conn, "20260823", [changed])
    name, flu, raw = conn.execute(
        "select theme_name, flu_rt, raw from theme_groups").fetchone()
    assert (name, flu) == ("이름변경", 1.0)
    assert json.loads(raw)["thema_nm"] == "이름변경"
    assert count_groups(conn) == 1


def test_upsert_members_pk_is_date_theme_code(conn):
    m = [normalize_theme_member("141", _MEMBER_ROW), normalize_theme_member("141", _MEMBER_ROW2)]
    assert upsert_theme_members(conn, "20260823", m) == 2
    upsert_theme_members(conn, "20260823", m)
    # 같은 종목이 다른 테마에도 있을 수 있다.
    upsert_theme_members(conn, "20260823", [normalize_theme_member("999", _MEMBER_ROW)])
    assert count_members(conn, "20260823") == 3
    got = conn.execute(
        "select code, name, cur_prc, flu_rt, acc_trde_qty, dt_prft_rt_n from theme_members "
        "where theme_code='141' order by code").fetchall()
    assert got[0] == ("000020", "보유B", 116200, 2.29, 575218, -2.92)


def test_upsert_empty_is_zero(conn):
    assert upsert_theme_groups(conn, "20260823", []) == 0
    assert upsert_theme_members(conn, "20260823", []) == 0


# ------------------------------------------------------------------ CLI


def _cli_kiwoom_api():
    """``cli_theme.main`` 이 통으로 거치는 kiwoom_api 목 — ka90001 한 번, 테마마다 ka90002."""
    calls: list[str] = []
    fake_api = MagicMock()

    def group_list(**_kwargs):
        calls.append("ka90001")
        return _groups_page([_GROUP_ROW, _GROUP_ROW2])

    def member_stocks(*, thema_grp_cd, **_kwargs):
        calls.append("ka90002")
        if thema_grp_cd == "141":
            return _members_page([_MEMBER_ROW, _MEMBER_ROW2])
        return _members_page([_MEMBER_ROW])

    fake_api.theme.theme_group_list.side_effect = group_list
    fake_api.theme.theme_component_stocks.side_effect = member_stocks
    return fake_api, calls


def test_cli_saves_groups_and_members_for_the_date(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KIWOOM_APP_KEY", "k")
    monkeypatch.setenv("KIWOOM_APP_SECRET", "s")
    fake_api, calls = _cli_kiwoom_api()
    db = str(tmp_path / "theme.db")

    rc = cli_theme.main(["--dsn", db, "--date", "20260823", "--sleep", "0"], kiwoom_api=fake_api)
    assert rc == 0
    assert calls == ["ka90001", "ka90002", "ka90002"]
    out = capsys.readouterr()
    assert out.out.strip() == "3"
    assert "테마 2개" in out.err

    conn = sqlite3.connect(db)
    assert count_groups(conn, "2026-08-23") == 2
    assert count_members(conn, "2026-08-23") == 3
    codes = {r[0] for r in conn.execute(
        "select code from theme_members where theme_code='141'").fetchall()}
    assert codes == {"000020", "131100"}
    conn.close()

    # 같은 날 다시 돌려도 행이 늘지 않는다.
    fake_api2, _ = _cli_kiwoom_api()
    rc = cli_theme.main(["--dsn", db, "--date", "20260823", "--sleep", "0"], kiwoom_api=fake_api2)
    assert rc == 0
    conn = sqlite3.connect(db)
    assert count_members(conn) == 3
    conn.close()


def test_cli_dry_run_lists_themes_without_members_or_db(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KIWOOM_APP_KEY", "k")
    monkeypatch.setenv("KIWOOM_APP_SECRET", "s")
    fake_api, calls = _cli_kiwoom_api()
    db = tmp_path / "theme.db"
    rc = cli_theme.main(["--dsn", str(db), "--dry-run", "--sleep", "0"], kiwoom_api=fake_api)
    assert rc == 0
    assert calls == ["ka90001"]
    assert not db.exists()
    out = capsys.readouterr()
    assert out.out.strip() == "2"
    assert "보험_생명보험" in out.err


def test_cli_rejects_bad_date(monkeypatch):
    monkeypatch.setenv("KIWOOM_APP_KEY", "k")
    monkeypatch.setenv("KIWOOM_APP_SECRET", "s")
    assert cli_theme.main(["--date", "2026-8-1", "--dry-run"]) == 2
