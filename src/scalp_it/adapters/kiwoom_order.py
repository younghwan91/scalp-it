"""키움 실계좌 **주문** 어댑터 — 주문 HTTP 호출은 이 파일에만 존재한다.

시세 조회(``kiwoom.py``)와 분리한다. 이 모듈은 **실계좌에 진짜 주문을 낼 수 있는**
유일한 코드라, 안전 가드를 코드 레벨 불변으로 박아 둔다.

확인한 사실 (2026-08-25 실호출 — 조회 API 만, 주문은 스키마만)
==========================================================

* **토큰**: ``kiwoom.py`` 와 동일 (``/oauth2/token``, 응답 키 ``token``).
* **계좌번호 자동조회**: ``ka00001`` (계좌번호조회), 경로 ``/api/dostk/acnt``,
  빈 바디. 응답 ``acctNo`` (10자리). 사용자가 손으로 안 넣게 이걸로 얻는다.
  (주문 바디에는 계좌번호가 **안 들어간다** — 토큰이 계좌에 묶여 있다. 조회는
  로깅·확인용.)
* **잔고/보유**: ``kt00018`` (계좌평가잔고내역요청), ``/api/dostk/acnt``, 바디
  ``{qry_tp:"1", dmst_stex_tp:"KRX"}``. 보유는 ``acnt_evlt_remn_indv_tot`` 리스트,
  종목코드 필드 ``stk_cd`` 는 **'A' 접두**가 붙어 온다("A000010"). 6자리로 정규화.
* **현재가**: ``ka10001`` (주식기본정보요청), ``/api/dostk/stkinfo``, 바디
  ``{stk_cd}``. 응답 ``cur_prc`` 는 부호 붙은 값이라 ``strip_sign`` 으로 뗀다.
* **미체결**: ``ka10075`` (미체결요청), ``/api/dostk/acnt``, 바디
  ``{all_stk_tp:"0", trde_tp:"0", stex_tp:"0"}``. 리스트 키 ``oso``.

주문 API (키움 문서 + younghwan91/kiwoom-client 래퍼로 확인, 실호출 안 함)
====================================================================

* 경로는 셋 다 ``/api/dostk/ordr``. api-id 로 구분한다.
* **현금 매수 지정가**: ``kt10000``. **현금 매도 지정가**: ``kt10001``.
  **취소**: ``kt10003``.
* 매수/매도 바디: ``{dmst_stex_tp, stk_cd, ord_qty, ord_uv, trde_tp}``.
  - ``dmst_stex_tp`` 거래소구분 = ``"KRX"`` (기본). "NXT"/"SOR" 가능.
  - ``stk_cd`` 종목코드(6자리). ``ord_qty`` 주문수량(문자열). ``ord_uv`` 주문단가.
  - ``trde_tp`` 매매구분 = ``"0"`` 보통(=지정가). "3"=시장가. 우리는 **지정가만**.
* 취소 바디: ``{dmst_stex_tp, stk_cd, org_ord_no, ord_qty}`` — 원주문번호 필수.
* 성공 응답: ``return_code=0`` + ``ord_no`` (주문번호).

안전 가드 (코드 레벨 불변)
========================

1. **보유 3종목 하드 차단** — 보유A(000010)·보유B(000020)·보유C(222222)엔 어떤 주문도
   못 나간다. 계좌 조회로 확인된 실보유. :data:`HELD_STOCK_CODES`. 계좌를 조회하면
   현재 보유 전체를 블록리스트에 **동적 병합**한다(:meth:`refresh_blocklist_from_account`).
2. **화이트리스트** — 등록 쌍의 후속주 코드만 허용. 비어 있지 않으면 그 외 전부 차단.
3. **수량 상한** — 주문당 ``max_qty``(기본 1주), 하드캡 ``max_qty_hard_cap``(10)는
   설정으로도 못 넘는다.
4. **가격 범위** — 참조가(현재가) 대비 ``price_band_pct``(±3%) 밖 지정가 거부.
5. **횟수 상한** — 종목당(``max_orders_per_code``)·일일(``max_orders_total``).

**dry-run 이 기본.** :meth:`submit` 은 ``live=True`` 를 명시해야만 진짜 POST 한다.
아니면 "이런 주문을 낼 것"만 만들어 반환·로깅하고 네트워크를 안 탄다.

주문 POST 는 **자동 재시도하지 않는다** — 통과했는지 모르는 주문을 다시 보내면 이중
발주 위험. 조회(read) 만 401 갱신 1회 재시도한다.

순수 부분은 krx-quant-core 에 있다
==================================

주문 값 객체(``OrderIntent``·``OrderResult``)·가드 판정(``OrderGuardConfig``·
``OrderGuard``)·취소 바디·프로토콜 상수·``normalize_code`` 는 daytrade-it 과 함께 쓰는
``krx_quant_core.execution``/``market.codes`` 로 옮겼다(판정 순서·사유 문자열까지 동일).
이 파일에는 **계좌에 묶인 것**(보유 3종목 표)과 **HTTP 전송**만 남는다. 기존 이름은
전부 여기서 그대로 import 된다.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
from krx_quant_core.execution import OrderBlocked as _CoreOrderBlocked
from krx_quant_core.execution import OrderGuard, OrderResult, cancel_body
from krx_quant_core.execution import OrderGuardConfig as _CoreOrderGuardConfig
from krx_quant_core.execution import OrderIntent as _CoreOrderIntent

# 경로·api-id·바디 값은 kiwoom_spec 이 정본이다(실주문 확인 여부가 값 옆에 적혀 있다).
# 여기서 안 쓰는 이름(BUY/SELL_API_ID·TRDE_TP_LIMIT·KIWOOM_MOCK_BASE)도 기존 import 경로를
# 살리려고 재노출한다.
from krx_quant_core.execution.kiwoom_spec import (  # noqa: F401
    ACCOUNT_NO_API_ID,
    ACCOUNT_PATH,
    BALANCE_API_ID,
    BALANCE_ROWS_KEY,
    BUY_API_ID,
    CANCEL_API_ID,
    DEFAULT_EXCHANGE,
    KIWOOM_MOCK_BASE,
    ORDER_PATH,
    PRICE_API_ID,
    SELL_API_ID,
    STKINFO_PATH,
    TRDE_TP_LIMIT,
    UNFILLED_API_ID,
    UNFILLED_ROWS_KEY,
)
from krx_quant_core.market.codes import normalize_code

from .kiwoom import (
    KIWOOM_BASE,
    KiwoomClient,
    KiwoomError,
    strip_sign,
)

__all__ = [
    "HELD_STOCK_CODES",
    "KiwoomOrderClient",
    "OrderBlocked",
    "OrderGuardConfig",
    "OrderIntent",
    "OrderLogSink",
    "OrderResult",
    "normalize_code",
]

_JSON_CT = "application/json;charset=UTF-8"

#: **하드 차단** — 사용자 실보유 3종목. 계좌 조회(2026-08-25)로 확인:
#: 보유A A000010 · 보유B A000020 · 보유C A222222. 절대 건들지 않는다.
HELD_STOCK_CODES: frozenset[str] = frozenset({"000010", "000020", "222222"})
HELD_STOCK_NAMES: dict[str, str] = {
    "000010": "보유A",
    "000020": "보유B",
    "222222": "보유C",
}


class OrderBlocked(_CoreOrderBlocked, KiwoomError):
    """안전 가드가 주문을 막았다. 실계좌 사고 방지의 마지막 벽.

    코어 예외이면서 기존 ``except KiwoomError`` 에도 계속 잡힌다(다중상속).
    """


# --------------------------------------------------------------------- 설정

class OrderGuardConfig(_CoreOrderGuardConfig):
    """주문 안전 가드 파라미터. 보수적 기본값(코어와 같음) + **보유 3종목 상시 차단**.

    코어는 블록리스트 기본이 비어 있고 ``always_blocked`` 를 무조건 병합한다. 여기서
    그걸 :data:`HELD_STOCK_CODES` 로 채우므로 기본 생성·``with_allowed``·``with_blocked``
    결과가 코어 이전 구현과 정확히 같다(보유 3종목은 무슨 설정을 줘도 빠지지 않는다).
    """

    always_blocked: ClassVar[frozenset[str]] = HELD_STOCK_CODES


# --------------------------------------------------------------------- 주문

class OrderIntent(_CoreOrderIntent):
    """낼(또는 낼 뻔한) 주문 하나. 종목명 표시는 보유 3종목 표를 쓴다."""

    code_names: ClassVar[Mapping[str, str]] = HELD_STOCK_NAMES

    def api_id(self) -> str:
        # 코어는 키움 예외를 몰라 ValueError 를 던진다 — 기존 호출부 계약대로 KiwoomError 로.
        try:
            return super().api_id()
        except ValueError as exc:
            raise KiwoomError(str(exc)) from None


# --------------------------------------------------------------------- 클라이언트

class KiwoomOrderClient:
    """실계좌 주문 클라이언트. 안전 가드 통과분만, 그것도 ``live=True`` 라야 발주.

    토큰·httpx 는 :class:`~scalp_it.adapters.kiwoom.KiwoomClient` 를 그대로 재사용한다
    (인증 로직 중복 방지). 주문 POST 는 여기서만, 재시도 없이 한 번.
    """

    def __init__(
        self,
        appkey: str,
        secretkey: str,
        *,
        guard: OrderGuardConfig | None = None,
        client: httpx.Client | None = None,
        base_url: str = KIWOOM_BASE,
        sleep: float = 0.1,
        sleeper: Callable[[float], None] | None = None,
        on_order: Callable[[OrderResult], None] | None = None,
        clock: Callable[[], datetime] | None = None,
        verify_holdings_before_order: bool = True,
        holdings_ttl: float = 5.0,
    ) -> None:
        self._auth = KiwoomClient(
            appkey, secretkey, client=client, base_url=base_url, sleep=0.0, sleeper=sleeper
        )
        self._client = self._auth._client
        self._base = base_url.rstrip("/")
        self._sleep = sleep
        self._sleeper = sleeper or time.sleep
        self._on_order = on_order
        self._clock = clock or (lambda: datetime.now())  # noqa: DTZ005 — KST naive
        # 가드 판정 상태(설정 + 매수 전용 동적 차단 + 일일 횟수 카운터)는 코어 OrderGuard 가
        # 들고 있다. 시계는 이 클라이언트의 naive KST 시계를 그대로 넘긴다.
        #
        # N-1: 실시간 보유 동적 병합은 **매수(buy)에만** 적용하는 별도 집합
        # (OrderGuard.dynamic_buy_block). guard.blocklist(하드 3종목+설정)는 양방향 차단이지만,
        # 우리가 방금 매수해 보유가 된 후속주까지 거기에 넣으면 그 종목의 **청산 매도**까지
        # 막혀 실포지션이 무감시로 방치된다.
        self._order_guard = OrderGuard(
            guard or OrderGuardConfig(), clock=self._clock, code_names=HELD_STOCK_NAMES
        )
        self._account_no: str | None = None
        # 주문 직전 실시간 잔고조회로 보유종목 매수를 이중차단(fail-closed).
        self._verify_holdings = bool(verify_holdings_before_order)
        self._holdings_ttl = float(holdings_ttl)
        self._holdings_cache: list[dict[str, Any]] | None = None
        self._holdings_at: float = 0.0

    @property
    def guard(self) -> OrderGuardConfig:
        """현재 가드 설정(장중 화이트리스트 병합분 포함)."""
        return self._order_guard.config

    @guard.setter
    def guard(self, config: OrderGuardConfig) -> None:
        self._order_guard.config = config

    # ----- 인증/HTTP ---------------------------------------------------------

    @property
    def token(self) -> str:
        return self._auth.token

    def _headers(self, api_id: str) -> dict[str, str]:
        return {
            "Content-Type": _JSON_CT,
            "authorization": f"Bearer {self.token}",
            "api-id": api_id,
        }

    def _read(self, path: str, api_id: str, body: dict, what: str) -> dict:
        """조회(read) 호출. 401 이면 토큰 1회 갱신 후 재시도. **주문엔 안 씀.**"""
        for attempt in range(2):
            if self._sleep:
                self._sleeper(self._sleep)
            resp = self._client.post(self._base + path, headers=self._headers(api_id), json=body)
            if resp.status_code == 401 and attempt == 0:
                self._auth.refresh_token()
                continue
            if resp.status_code >= 400:
                raise KiwoomError(f"{what} HTTP {resp.status_code}: {resp.text[:200]}")
            payload = resp.json()
            code = payload.get("return_code")
            if code is not None and int(code) != 0:
                raise KiwoomError(f"{what} 실패: return_code={code} {payload.get('return_msg')}")
            return payload
        raise KiwoomError(f"{what} 재시도 소진")

    # ----- 계좌/보유/현재가 --------------------------------------------------

    def account_number(self) -> str:
        """``ka00001`` 로 계좌번호 자동조회(캐시). 사용자가 손으로 안 넣게."""
        if self._account_no is None:
            payload = self._read(ACCOUNT_PATH, ACCOUNT_NO_API_ID, {}, "계좌번호 조회")
            self._account_no = str(payload.get("acctNo") or "").strip()
            if not self._account_no:
                raise KiwoomError("계좌번호 조회 응답에 acctNo 가 없다.")
        return self._account_no

    def masked_account(self) -> str:
        """로그·출력용 마스킹 계좌번호(비밀 노출 금지)."""
        no = self._account_no or ""
        if len(no) <= 4:
            return "****"
        return no[:4] + "*" * (len(no) - 4)

    def holdings(self, *, force: bool = False) -> list[dict[str, Any]]:
        """``kt00018`` 보유 종목 리스트. 코드는 6자리로 정규화. TTL 캐시."""
        now = time.monotonic()
        if (not force and self._holdings_cache is not None
                and (now - self._holdings_at) < self._holdings_ttl):
            return self._holdings_cache
        payload = self._read(
            ACCOUNT_PATH, BALANCE_API_ID, {"qry_tp": "1", "dmst_stex_tp": "KRX"}, "잔고 조회"
        )
        rows = payload.get(BALANCE_ROWS_KEY) or []
        out: list[dict[str, Any]] = []
        for row in rows:
            code = normalize_code(row.get("stk_cd"))
            if not code:
                continue
            out.append({
                "code": code,
                "name": str(row.get("stk_nm") or "").strip(),
                "qty": strip_sign(row.get("rmnd_qty")),
                "sellable": strip_sign(row.get("trde_able_qty")),
                "cur_prc": strip_sign(row.get("cur_prc")),
            })
        self._holdings_cache = out
        self._holdings_at = now
        return out

    def _preorder_holdings_guard(self, intent: OrderIntent) -> str | None:
        """매수 직전 실시간 잔고조회로 보유종목을 이중차단. **fail-closed**.

        조회가 실패(타임아웃·에러)하면 주문을 거부한다 — fail-open(조회 실패 →
        주문 강행)이면 이중방어가 무의미. 조회 성공분은 블록리스트에 병합한다.
        """
        if not self._verify_holdings or intent.side != "buy":
            return None
        try:
            held = {h["code"] for h in self.holdings()}
        except (KiwoomError, httpx.HTTPError, ValueError, KeyError) as exc:
            return f"주문 직전 잔고조회 실패 → fail-closed 로 주문 거부: {exc}"
        # N-1: 동적 보유분은 **매수 전용** 집합에 병합(guard.blocklist 를 건드리지 않는다).
        return self._order_guard.holdings_reason(intent, held)

    def allow_codes(self, codes: Sequence[str]) -> None:
        """장중에 붙은 종목을 주문 화이트리스트에 더한다.

        대장 스캔·돌발재료로 감시에 들어온 종목은 장전 화이트리스트에 없다. 이걸
        안 하면 신호는 나는데 주문이 전부 차단된다.
        """
        self._order_guard.allow_codes(codes)

    def refresh_blocklist_from_account(self) -> set[str]:
        """계좌 보유 전체를 **매수 차단** 집합에 동적 병합. 보유 종목 매수 사고를 이중 차단.

        N-1: 청산 매도까지 막지 않도록 guard.blocklist(양방향) 가 아니라 매수 전용
        집합(``OrderGuard.dynamic_buy_block``)에 병합한다. 하드코딩 3종목만 양방향 차단.

        반환: 새로 추가된 보유 코드 집합.
        """
        return self._order_guard.block_buys([h["code"] for h in self.holdings()])

    def current_price(self, code: str) -> int:
        """``ka10001`` 현재가(부호 제거). 가격 범위 가드·참조가로 쓴다."""
        payload = self._read(STKINFO_PATH, PRICE_API_ID, {"stk_cd": normalize_code(code)}, "현재가 조회")
        return strip_sign(payload.get("cur_prc"))

    def unfilled_orders(self, code: str | None = None) -> list[dict[str, Any]]:
        """``ka10075`` 미체결 리스트. 코드 주면 그 종목만."""
        payload = self._read(
            ACCOUNT_PATH, UNFILLED_API_ID,
            {"all_stk_tp": "0", "trde_tp": "0", "stex_tp": "0"}, "미체결 조회",
        )
        rows = payload.get(UNFILLED_ROWS_KEY) or []
        out: list[dict[str, Any]] = []
        want = normalize_code(code) if code else None
        for row in rows:
            c = normalize_code(row.get("stk_cd"))
            if want and c != want:
                continue
            out.append({
                "ord_no": str(row.get("ord_no") or "").strip(),
                "code": c,
                "name": str(row.get("stk_nm") or "").strip(),
                "unfilled_qty": strip_sign(row.get("oso_qty") or row.get("ord_remnq")),
                "raw": dict(row),
            })
        return out

    def is_filled(self, ord_no: str, code: str | None = None) -> bool:
        """매수 체결 확인 — **잔고에 해당 종목이 잡히면** 체결로 본다(N-3).

        '미체결 목록에 없음' 만으로 체결을 단정하면 **취소·거부 주문도 체결로 오판**해
        무보유 매도(→ 거부 → N-2 크래시)로 이어진다. 그래서 잔고조회를 1차 근거로
        삼는다. 우리는 보유 종목을 하드 차단하므로, 대기 매수 중 잔고에 그 종목이
        새로 잡혔다면 그것은 우리 매수의 체결이다.

        * ``code`` 주어짐: 잔고에 그 종목 수량>0 이면 체결. 없으면 미체결(보수).
          조회 실패 시 체결로 단정하지 않고 ``False``.
        * ``code`` 없음(옛 경로): 잔고를 못 좁히므로 미체결 목록 부재로 근사하되,
          이 경로는 취소·거부 오판 위험이 있으니 호출부는 되도록 code 를 넘길 것.
        """
        norm = normalize_code(code) if code else None
        if norm:
            try:
                for h in self.holdings():
                    if h["code"] == norm and int(h.get("qty") or 0) > 0:
                        return True
            except (KiwoomError, httpx.HTTPError, ValueError, KeyError):
                return False   # 조회 실패 → 체결 단정 안 함(fail-safe)
            return False       # 잔고에 없음 → 아직 미체결(취소·거부 오판 방지)
        return not any(o["ord_no"] == str(ord_no) for o in self.unfilled_orders(code))

    # ----- 가드 판정 ---------------------------------------------------------

    def guard_reason(self, intent: OrderIntent, ref_price: float | None) -> str | None:
        """주문을 막을 사유. None 이면 통과. **dry-run 에서도 항상 돈다.**

        판정 순서(코드 형식 → 블록리스트 → 동적 매수차단 → 화이트리스트 → 수량 → 가격 →
        횟수)와 사유 문자열은 코어 :class:`~krx_quant_core.execution.OrderGuard` 에 있다.
        """
        return self._order_guard.reason(intent, ref_price)

    # ----- 발주 --------------------------------------------------------------

    def submit(
        self,
        intent: OrderIntent,
        *,
        live: bool = False,
        ref_price: float | None = None,
    ) -> OrderResult:
        """주문을 낸다. **기본은 dry-run** — ``live=True`` 라야 진짜 POST.

        어떤 경우든 안전 가드를 먼저 통과해야 한다(dry-run 에서도). 막히면
        ``blocked`` 결과를 돌려주고 절대 POST 하지 않는다.

        매수는 **주문 직전 실시간 잔고조회(fail-closed)** 로 보유종목을 한 번 더
        차단한다 — 하드코딩 3종목 + 현재 보유 전종목. dry-run 도 동일 경로.
        """
        reason = self._preorder_holdings_guard(intent) or self.guard_reason(intent, ref_price)
        if reason is not None:
            result = OrderResult(intent=intent, dry_run=not live, submitted=False,
                                 blocked=True, blocked_reason=reason)
            self._notify(result)
            return result

        # 가드 통과 → 횟수 소진(dry-run 도 리허설로 카운트).
        self._order_guard.record_order(intent.code)

        if not live:
            # dry-run: POST 안 함. 낼 주문만 만들어 반환·로깅.
            result = OrderResult(intent=intent, dry_run=True, submitted=False)
            self._notify(result)
            return result

        # ---- 실발주 (live=True) : 재시도 없이 한 번만 POST ----
        if self._sleep:
            self._sleeper(self._sleep)
        resp = self._client.post(
            self._base + ORDER_PATH, headers=self._headers(intent.api_id()), json=intent.body()
        )
        if resp.status_code >= 400:
            raise KiwoomError(f"주문 HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        rc = payload.get("return_code")
        rc_int = int(rc) if rc is not None else None
        result = OrderResult(
            intent=intent, dry_run=False, submitted=True,
            ord_no=(str(payload.get("ord_no")).strip() if payload.get("ord_no") else None),
            return_code=rc_int, return_msg=str(payload.get("return_msg") or ""),
        )
        self._notify(result)
        if rc_int is not None and rc_int != 0:
            raise KiwoomError(f"주문 실패: return_code={rc} {result.return_msg}")
        return result

    def cancel(
        self,
        ord_no: str,
        code: str,
        qty: int,
        *,
        exchange: str = DEFAULT_EXCHANGE,
        live: bool = False,
    ) -> OrderResult:
        """미체결 주문 취소(``kt10003``). dry-run 기본."""
        code = normalize_code(code)
        intent = OrderIntent(side="cancel", code=code, qty=int(qty), price=0, exchange=exchange)
        if not live:
            result = OrderResult(intent=intent, dry_run=True, submitted=False)
            self._notify(result)
            return result
        body = cancel_body(ord_no, code, qty, exchange)
        if self._sleep:
            self._sleeper(self._sleep)
        resp = self._client.post(self._base + ORDER_PATH, headers=self._headers(CANCEL_API_ID), json=body)
        if resp.status_code >= 400:
            raise KiwoomError(f"취소 HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        rc = payload.get("return_code")
        rc_int = int(rc) if rc is not None else None
        result = OrderResult(
            intent=intent, dry_run=False, submitted=True,
            ord_no=(str(payload.get("ord_no")).strip() if payload.get("ord_no") else str(ord_no)),
            return_code=rc_int, return_msg=str(payload.get("return_msg") or ""),
        )
        self._notify(result)
        return result

    # ----- 편의 --------------------------------------------------------------

    def buy_limit(self, code: str, qty: int, price: int, *, live: bool = False,
                  ref_price: float | None = None, exchange: str = DEFAULT_EXCHANGE) -> OrderResult:
        intent = OrderIntent(side="buy", code=normalize_code(code), qty=int(qty),
                             price=int(price), exchange=exchange)
        return self.submit(intent, live=live, ref_price=ref_price)

    def sell_limit(self, code: str, qty: int, price: int, *, live: bool = False,
                   ref_price: float | None = None, exchange: str = DEFAULT_EXCHANGE) -> OrderResult:
        intent = OrderIntent(side="sell", code=normalize_code(code), qty=int(qty),
                             price=int(price), exchange=exchange)
        return self.submit(intent, live=live, ref_price=ref_price)

    def _notify(self, result: OrderResult) -> None:
        if self._on_order is not None:
            self._on_order(result)


# --------------------------------------------------------------------- 로그 sink

class OrderLogSink:
    """주문(및 dry-run·차단) 이벤트를 파일에 JSON 한 줄씩 append + stdout.

    **비밀(토큰·계좌번호)은 절대 남기지 않는다.** 종목·수량·가격·매수매도만.
    """

    def __init__(self, path: str | Path = "data/pair_orders.log", *, echo: bool = True) -> None:
        self.path = Path(path)
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, result: OrderResult) -> None:
        rec = result.to_record()
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print("[주문] " + self._human(result), flush=True)

    @staticmethod
    def _human(result: OrderResult) -> str:
        if result.blocked:
            return f"🚫 차단 — {result.intent.describe()}  ({result.blocked_reason})"
        if result.dry_run:
            return f"🧪 DRY-RUN(발주 안 함) — {result.intent.describe()}"
        tag = "✅ 발주" if result.ok else f"❌ 실패(rc={result.return_code})"
        no = f"  주문#{result.ord_no}" if result.ord_no else ""
        return f"{tag} — {result.intent.describe()}{no}"
