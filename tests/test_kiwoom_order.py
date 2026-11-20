"""주문 어댑터 테스트. **실 api.kiwoom.com 으로는 절대 안 나간다** — 전부 MockTransport.

안전가드(보유종목 차단·수량상한·화이트리스트·가격범위·횟수상한)·dry-run·계좌조회·
fail-closed 를 검증한다. 실주문(live=True) 경로도 mock POST 로만 확인한다.
"""

import json

import httpx
import pytest

from scalp_it.adapters.kiwoom import KiwoomError
from scalp_it.adapters.kiwoom_order import (
    HELD_CODES_ENV,
    HeldStockConfigError,
    KiwoomOrderClient,
    OrderGuardConfig,
    OrderIntent,
    normalize_code,
)

FAKE_HELD = ("000010", "000020", "222222")  # 가짜 코드 — 실보유는 레포에 두지 않는다


@pytest.fixture(autouse=True)
def _held_env(monkeypatch):
    monkeypatch.setenv(HELD_CODES_ENV, ",".join(FAKE_HELD))


_TOKEN_BODY = {"return_code": 0, "return_msg": "ok", "token": "tok-xyz", "token_type": "bearer"}


class Recorder:
    """주문 POST 를 기록하는 mock 라우터. 조회는 기본값으로 응답."""

    def __init__(self, holdings=None, holdings_fail=False, order_rc=0):
        self.order_posts = []          # (api_id, body) — 실제 나간 주문
        self.holdings = holdings or []  # [{stk_cd, stk_nm, rmnd_qty, trde_able_qty, cur_prc}]
        self.holdings_fail = holdings_fail
        self.order_rc = order_rc

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        api = request.headers.get("api-id", "")
        if path == "/oauth2/token":
            return httpx.Response(200, json=_TOKEN_BODY)
        if path == "/api/dostk/acnt" and api == "ka00001":
            return httpx.Response(200, json={"return_code": 0, "acctNo": "1234567890"})
        if path == "/api/dostk/acnt" and api == "kt00018":
            if self.holdings_fail:
                return httpx.Response(200, json={"return_code": 3, "return_msg": "조회 실패"})
            return httpx.Response(200, json={"return_code": 0,
                                             "acnt_evlt_remn_indv_tot": self.holdings})
        if path == "/api/dostk/stkinfo" and api == "ka10001":
            return httpx.Response(200, json={"return_code": 0, "cur_prc": "-10000"})
        if path == "/api/dostk/acnt" and api == "ka10075":
            return httpx.Response(200, json={"return_code": 0, "oso": []})
        if path == "/api/dostk/ordr":
            self.order_posts.append((api, json.loads(request.content)))
            return httpx.Response(200, json={"return_code": self.order_rc,
                                             "return_msg": "완료", "ord_no": "0000001"})
        return httpx.Response(404, json={"return_code": 9, "return_msg": f"unrouted {path} {api}"})


def _client(rec: Recorder, guard: OrderGuardConfig | None = None, *, verify=False):
    return KiwoomOrderClient(
        "key", "secret", guard=guard,
        client=httpx.Client(transport=httpx.MockTransport(rec)),
        sleeper=lambda _s: None, verify_holdings_before_order=verify,
    )


# ------------------------------------------------------------------ 코드 정규화

def test_normalize_code_strips_a_prefix_and_pads():
    assert normalize_code("A000010") == "000010"
    assert normalize_code("20") == "000020"
    assert normalize_code(" 222222 ") == "222222"
    assert normalize_code("A000020") == "000020"


# ------------------------------------------------------------------ 보유 종목 하드 차단

@pytest.mark.parametrize("code", [*FAKE_HELD, "A000010"])
def test_held_stocks_hard_blocked_dry_run(code):
    rec = Recorder()
    c = _client(rec)
    r = c.buy_limit(code, 1, 10000, live=False, ref_price=10000)
    assert r.blocked is True
    assert "블록리스트" in r.blocked_reason or "보유" in r.blocked_reason
    assert rec.order_posts == []          # 발주 안 함


def test_held_stocks_hard_blocked_even_live():
    rec = Recorder()
    c = _client(rec)
    r = c.buy_limit("000010", 1, 10000, live=True, ref_price=10000)
    assert r.blocked is True
    assert rec.order_posts == []          # live 라도 절대 POST 안 함


def test_held_codes_loaded_from_env():
    from scalp_it.adapters.kiwoom_order import held_stock_codes

    assert held_stock_codes() == frozenset(FAKE_HELD)
    assert OrderGuardConfig().blocklist >= frozenset(FAKE_HELD)


@pytest.mark.parametrize("value", [None, "", "  ", ",", ", ,", "1234567", "abcdef"])
def test_live_client_refuses_without_valid_held_env(monkeypatch, value):
    """fail-closed — 차단 목록 환경변수가 없거나 비었거나 깨졌으면 클라이언트를 만들지 않는다."""
    if value is None:
        monkeypatch.delenv(HELD_CODES_ENV, raising=False)
    else:
        monkeypatch.setenv(HELD_CODES_ENV, value)
    rec = Recorder()
    with pytest.raises(HeldStockConfigError):
        _client(rec)
    with pytest.raises(HeldStockConfigError):
        OrderGuardConfig()
    assert rec.order_posts == []


def test_held_env_is_read_on_every_guard_build(monkeypatch):
    monkeypatch.setenv(HELD_CODES_ENV, "333333")
    assert OrderGuardConfig().blocklist == frozenset({"333333"})


# ------------------------------------------------------------------ 화이트리스트

def test_whitelist_blocks_unlisted():
    rec = Recorder()
    guard = OrderGuardConfig(whitelist=frozenset({"294570"}))
    c = _client(rec, guard)
    assert c.buy_limit("999999", 1, 10000, ref_price=10000).blocked is True
    ok = c.buy_limit("294570", 1, 10000, live=False, ref_price=10000)
    assert ok.blocked is False and ok.dry_run is True
    assert rec.order_posts == []


# ------------------------------------------------------------------ 수량 상한

def test_qty_over_max_blocked():
    rec = Recorder()
    guard = OrderGuardConfig(max_qty=1, whitelist=frozenset({"294570"}))
    c = _client(rec, guard)
    assert c.buy_limit("294570", 2, 10000, ref_price=10000).blocked is True
    assert c.buy_limit("294570", 1, 10000, ref_price=10000).blocked is False


def test_qty_hard_cap_cannot_be_exceeded_by_config():
    # max_qty 를 100 으로 줘도 하드캡 10 으로 강제.
    guard = OrderGuardConfig(max_qty=100, max_qty_hard_cap=10)
    assert guard.max_qty == 10
    rec = Recorder()
    c = _client(rec, OrderGuardConfig(max_qty=10, whitelist=frozenset({"294570"})))
    assert c.buy_limit("294570", 11, 10000, ref_price=10000).blocked is True


# ------------------------------------------------------------------ 가격 범위

def test_price_band_rejects_outside():
    rec = Recorder()
    guard = OrderGuardConfig(price_band_pct=0.03, whitelist=frozenset({"294570"}))
    c = _client(rec, guard)
    # 참조가 10000, ±3% → 9700~10300. 10500 거부, 10200 통과.
    assert c.buy_limit("294570", 1, 10500, ref_price=10000).blocked is True
    assert c.buy_limit("294570", 1, 10200, ref_price=10000).blocked is False
    assert c.buy_limit("294570", 1, 0, ref_price=10000).blocked is True


# ------------------------------------------------------------------ 횟수 상한

def test_order_count_limits():
    rec = Recorder()
    guard = OrderGuardConfig(max_orders_per_code=2, max_orders_total=3,
                             whitelist=frozenset({"294570", "033340"}))
    c = _client(rec, guard)
    assert c.buy_limit("294570", 1, 10000, ref_price=10000).blocked is False
    assert c.buy_limit("294570", 1, 10000, ref_price=10000).blocked is False
    # 종목당 2회 초과
    assert c.buy_limit("294570", 1, 10000, ref_price=10000).blocked is True
    assert c.buy_limit("033340", 1, 10000, ref_price=10000).blocked is False
    # 일일 총 3회 초과
    assert c.buy_limit("033340", 1, 10000, ref_price=10000).blocked is True


# ------------------------------------------------------------------ dry-run vs live

def test_dry_run_does_not_post():
    rec = Recorder()
    c = _client(rec, OrderGuardConfig(whitelist=frozenset({"294570"})))
    r = c.buy_limit("294570", 1, 10000, live=False, ref_price=10000)
    assert r.dry_run is True and r.submitted is False and r.ok is True
    assert r.intent.body() == {"dmst_stex_tp": "KRX", "stk_cd": "294570",
                               "ord_qty": "1", "ord_uv": "10000", "trde_tp": "0"}
    assert rec.order_posts == []          # 네트워크 POST 없음


def test_live_posts_once_and_parses_ord_no():
    rec = Recorder()
    c = _client(rec, OrderGuardConfig(whitelist=frozenset({"294570"})))
    r = c.buy_limit("294570", 1, 10000, live=True, ref_price=10000)
    assert r.submitted is True and r.dry_run is False and r.ord_no == "0000001"
    assert r.return_code == 0 and r.ok is True
    assert len(rec.order_posts) == 1
    api, body = rec.order_posts[0]
    assert api == "kt10000" and body["stk_cd"] == "294570" and body["trde_tp"] == "0"


def test_sell_uses_kt10001():
    rec = Recorder()
    c = _client(rec, OrderGuardConfig(whitelist=frozenset({"294570"})))
    r = c.sell_limit("294570", 1, 10000, live=True, ref_price=10000)
    assert rec.order_posts[0][0] == "kt10001" and r.submitted is True


def test_cancel_dry_run_no_post():
    rec = Recorder()
    c = _client(rec)
    r = c.cancel("0000001", "294570", 1, live=False)
    assert r.dry_run is True and rec.order_posts == []


# ------------------------------------------------------------------ 계좌/보유 조회

def test_account_number_and_masking():
    rec = Recorder()
    c = _client(rec)
    assert c.account_number() == "1234567890"
    assert c.masked_account() == "1234******"


def test_holdings_parses_a_prefix_and_pads():
    rec = Recorder(holdings=[
        {"stk_cd": "A000010", "stk_nm": "가짜A", "rmnd_qty": "000000000000007",
         "trde_able_qty": "000000000000007", "cur_prc": "000000012300"},
    ])
    c = _client(rec)
    h = c.holdings()
    assert h[0]["code"] == "000010" and h[0]["qty"] == 7 and h[0]["cur_prc"] == 12300


def test_refresh_blocklist_merges_current_holdings():
    rec = Recorder(holdings=[
        {"stk_cd": "A294570", "stk_nm": "쿠콘", "rmnd_qty": "10",
         "trde_able_qty": "10", "cur_prc": "10000"},
    ])
    c = _client(rec)
    added = c.refresh_blocklist_from_account()
    assert "294570" in added
    # 이제 294570 은 매수 차단(방금 보유로 확인됨).
    r = c.buy_limit("294570", 1, 10000, ref_price=10000)
    assert r.blocked is True


def test_current_price_strips_sign():
    rec = Recorder()
    c = _client(rec)
    assert c.current_price("005930") == 10000   # 응답 "-10000"


# ------------------------------------------------------------------ fail-closed

def test_preorder_holdings_check_blocks_held():
    # verify 켜짐: 주문 직전 실시간 잔고에 이 종목이 있으면 차단.
    rec = Recorder(holdings=[
        {"stk_cd": "A294570", "stk_nm": "쿠콘", "rmnd_qty": "10",
         "trde_able_qty": "10", "cur_prc": "10000"},
    ])
    c = _client(rec, OrderGuardConfig(whitelist=frozenset({"294570"})), verify=True)
    r = c.buy_limit("294570", 1, 10000, live=True, ref_price=10000)
    assert r.blocked is True and "보유" in r.blocked_reason
    assert rec.order_posts == []


def test_preorder_holdings_fail_closed():
    # verify 켜짐 + 잔고조회 실패 → 주문 거부(fail-closed), POST 안 함.
    rec = Recorder(holdings_fail=True)
    c = _client(rec, OrderGuardConfig(whitelist=frozenset({"294570"})), verify=True)
    r = c.buy_limit("294570", 1, 10000, live=True, ref_price=10000)
    assert r.blocked is True and "fail-closed" in r.blocked_reason
    assert rec.order_posts == []


# ------------------------------------------------------------------ N-1: 동적 보유 병합은 매수만

def test_dynamic_holdings_block_buy_but_not_sell():
    """N-1: refresh 로 병합된 실시간 보유분은 **매수만** 차단하고 청산 매도는 막지 않는다."""
    rec = Recorder(holdings=[
        {"stk_cd": "A294570", "stk_nm": "쿠콘", "rmnd_qty": "1",
         "trde_able_qty": "1", "cur_prc": "10000"},
    ])
    c = _client(rec, OrderGuardConfig(whitelist=frozenset({"294570"})))
    added = c.refresh_blocklist_from_account()
    assert "294570" in added
    # 매수는 차단(방금 보유로 확인).
    assert c.buy_limit("294570", 1, 10000, ref_price=10000).blocked is True
    # 그러나 그 종목의 **청산 매도**는 차단되지 않는다 — 무감시 방치 방지.
    s = c.sell_limit("294570", 1, 10000, live=False, ref_price=10000)
    assert s.blocked is False
    assert rec.order_posts == []          # dry-run 이라 POST 는 없음


def test_preorder_verify_blocks_buy_but_sell_unaffected():
    """N-1: verify=True 여도 매도는 잔고병합 차단을 받지 않는다(side!=buy)."""
    rec = Recorder(holdings=[
        {"stk_cd": "A294570", "stk_nm": "쿠콘", "rmnd_qty": "1",
         "trde_able_qty": "1", "cur_prc": "10000"},
    ])
    c = _client(rec, OrderGuardConfig(whitelist=frozenset({"294570"})), verify=True)
    assert c.buy_limit("294570", 1, 10000, live=False, ref_price=10000).blocked is True
    assert c.sell_limit("294570", 1, 10000, live=False, ref_price=10000).blocked is False


def test_hardcoded_held_blocks_sell_too():
    """환경변수 보유 종목은 매도까지 양방향 차단(우리가 안 파는 종목이라 무해)."""
    rec = Recorder()
    c = _client(rec)
    assert c.sell_limit("000010", 1, 10000, live=False, ref_price=10000).blocked is True


# ------------------------------------------------------------------ N-3: 체결 오판 방지

def test_is_filled_false_when_not_in_balance():
    """N-3: 미체결 목록에 없어도(취소·거부) 잔고에 없으면 체결로 오판하지 않는다."""
    rec = Recorder(holdings=[])           # 잔고 비었음 · 미체결 목록도 빈 oso
    c = _client(rec)
    assert c.is_filled("0000001", "294570") is False


def test_is_filled_true_when_in_balance():
    """N-3: 잔고에 종목·수량이 잡히면 체결로 본다."""
    rec = Recorder(holdings=[
        {"stk_cd": "A294570", "stk_nm": "쿠콘", "rmnd_qty": "1",
         "trde_able_qty": "1", "cur_prc": "10000"},
    ])
    c = _client(rec)
    assert c.is_filled("0000001", "294570") is True


def test_is_filled_false_on_balance_query_failure():
    """N-3: 잔고조회 실패 시 체결로 단정하지 않는다(fail-safe)."""
    rec = Recorder(holdings_fail=True)
    c = _client(rec)
    assert c.is_filled("0000001", "294570") is False


def test_intent_api_id_and_body():
    intent = OrderIntent(side="buy", code="294570", qty=1, price=1928)
    assert intent.api_id() == "kt10000"
    assert intent.body()["ord_uv"] == "1928"
    with pytest.raises(KiwoomError):
        OrderIntent(side="weird", code="294570", qty=1, price=1).api_id()


# ------------------------------------------------------------------ 주문 지연 실측


def test_live_order_records_post_timing_in_log(tmp_path):
    """실발주는 보낸 시각(µs)·왕복 ms 를 남긴다 — 대기열 모델 지연을 실측 분포로 넣는다."""
    from scalp_it.adapters.kiwoom_order import OrderLogSink

    rec = Recorder()
    sink = OrderLogSink(tmp_path / "orders.log", echo=False)
    holder: dict = {}
    c = KiwoomOrderClient(
        "key", "secret", guard=OrderGuardConfig(whitelist=frozenset({"294570"})),
        client=httpx.Client(transport=httpx.MockTransport(rec)),
        sleeper=lambda _s: None, verify_holdings_before_order=False,
        on_order=lambda r: sink.emit(r, timing=holder["c"].last_timing),
    )
    holder["c"] = c
    c.buy_limit("294570", 1, 10000, live=True, ref_price=10000)
    line = json.loads((tmp_path / "orders.log").read_text().splitlines()[-1])
    assert line["post_ms"] >= 0
    assert "." in line["sent_at"]

    c.buy_limit("294570", 1, 10000, live=False, ref_price=10000)  # dry-run: POST 없음
    assert c.last_timing is None
    line = json.loads((tmp_path / "orders.log").read_text().splitlines()[-1])
    assert "post_ms" not in line
