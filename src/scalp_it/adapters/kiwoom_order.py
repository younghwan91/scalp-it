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

주문 API (키움 문서 + younghwan91/kiwoom-rest-api 래퍼로 확인, 실호출 안 함)
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
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

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

# 모의투자 REST 베이스 — ``--mock`` 일 때 주문도 이 엔드포인트로 강제.
KIWOOM_MOCK_BASE = "https://mockapi.kiwoom.com"

# 경로·api-id
ORDER_PATH = "/api/dostk/ordr"
ACCOUNT_PATH = "/api/dostk/acnt"
STKINFO_PATH = "/api/dostk/stkinfo"

BUY_API_ID = "kt10000"
SELL_API_ID = "kt10001"
CANCEL_API_ID = "kt10003"
ACCOUNT_NO_API_ID = "ka00001"
BALANCE_API_ID = "kt00018"
PRICE_API_ID = "ka10001"
UNFILLED_API_ID = "ka10075"

BALANCE_ROWS_KEY = "acnt_evlt_remn_indv_tot"
UNFILLED_ROWS_KEY = "oso"

#: 매매구분 — 보통(지정가). 시장가("3")는 이 모듈에서 절대 안 쓴다.
TRDE_TP_LIMIT = "0"
#: 거래소구분 기본.
DEFAULT_EXCHANGE = "KRX"

_JSON_CT = "application/json;charset=UTF-8"

#: **하드 차단** — 사용자 실보유 3종목. 계좌 조회(2026-08-25)로 확인:
#: 보유A A000010 · 보유B A000020 · 보유C A222222. 절대 건들지 않는다.
HELD_STOCK_CODES: frozenset[str] = frozenset({"000010", "000020", "222222"})
HELD_STOCK_NAMES: dict[str, str] = {
    "000010": "보유A",
    "000020": "보유B",
    "222222": "보유C",
}


class OrderBlocked(KiwoomError):
    """안전 가드가 주문을 막았다. 실계좌 사고 방지의 마지막 벽."""


def normalize_code(code: Any) -> str:
    """종목코드를 6자리로 정규화한다.

    kt00018 응답은 ``"A000010"`` 처럼 알파벳 접두가 붙는다. 접두·공백을 떼고
    숫자 6자리만 남긴다. 우리 블록/화이트리스트·주문은 전부 이 6자리 기준.
    """
    text = str(code if code is not None else "").strip().upper()
    # 앞의 알파벳 접두(A 등)를 제거
    digits = text.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(ch for ch in digits if ch.isdigit())
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6) if digits else ""


# --------------------------------------------------------------------- 설정

@dataclass(frozen=True)
class OrderGuardConfig:
    """주문 안전 가드 파라미터. 보수적 기본값."""

    max_qty: int = 1                    # 주문당 수량 상한(기본 1주)
    max_qty_hard_cap: int = 10          # 절대 하드캡 — 설정으로도 못 넘음
    price_band_pct: float = 0.03        # 참조가 대비 ±3% 밖 지정가 거부. 0 = 끔
    max_orders_per_code: int = 5        # 종목당 일일 주문 횟수 상한. 0 = 끔
    max_orders_total: int = 20          # 일일 총 주문 횟수 상한. 0 = 끔
    whitelist: frozenset[str] = frozenset()   # 허용 종목(후속주). 비면 화이트리스트 미적용
    blocklist: frozenset[str] = HELD_STOCK_CODES  # 항상 차단(보유 3종목 + 병합분)

    def __post_init__(self) -> None:
        # 하드캡을 넘는 max_qty 는 하드캡으로 강제(불변).
        cap = max(1, int(self.max_qty_hard_cap))
        object.__setattr__(self, "max_qty_hard_cap", cap)
        object.__setattr__(self, "max_qty", max(1, min(int(self.max_qty), cap)))
        # 보유 3종목은 무슨 일이 있어도 블록리스트에 포함(정규화).
        merged = {normalize_code(c) for c in self.blocklist} | set(HELD_STOCK_CODES)
        object.__setattr__(self, "blocklist", frozenset(c for c in merged if c))
        object.__setattr__(
            self, "whitelist", frozenset(normalize_code(c) for c in self.whitelist if normalize_code(c))
        )

    def with_blocked(self, codes: set[str]) -> OrderGuardConfig:
        """계좌 보유분을 블록리스트에 병합한 새 설정."""
        merged = set(self.blocklist) | {normalize_code(c) for c in codes if normalize_code(c)}
        return replace(self, blocklist=frozenset(merged))


# --------------------------------------------------------------------- 주문

@dataclass(frozen=True)
class OrderIntent:
    """낼(또는 낼 뻔한) 주문 하나. 종목·수량·가격·매수/매도."""

    side: str                           # "buy" | "sell"
    code: str                           # 6자리
    qty: int
    price: int                          # 지정가(원)
    exchange: str = DEFAULT_EXCHANGE

    def api_id(self) -> str:
        if self.side == "buy":
            return BUY_API_ID
        if self.side == "sell":
            return SELL_API_ID
        raise KiwoomError(f"알 수 없는 매매구분: {self.side!r}")

    def body(self) -> dict[str, str]:
        """키움 주문 바디. 수량·가격은 문자열."""
        return {
            "dmst_stex_tp": self.exchange,
            "stk_cd": self.code,
            "ord_qty": str(int(self.qty)),
            "ord_uv": str(int(self.price)),
            "trde_tp": TRDE_TP_LIMIT,
        }

    def describe(self) -> str:
        kind = "매수" if self.side == "buy" else "매도"
        name = HELD_STOCK_NAMES.get(self.code, "")
        tag = f"{name}({self.code})" if name else self.code
        return f"{kind} 지정가  {tag}  {self.qty}주 @ {self.price:,}원 [{self.exchange}]"


@dataclass
class OrderResult:
    """:meth:`submit` 결과. dry-run 이면 ``submitted=False`` 이고 POST 안 함."""

    intent: OrderIntent
    dry_run: bool
    submitted: bool                     # 실제 POST 했는가
    blocked: bool = False
    blocked_reason: str = ""
    ord_no: str | None = None
    return_code: int | None = None
    return_msg: str = ""

    @property
    def ok(self) -> bool:
        """실주문이 성공했거나(dry 아님·rc0) dry-run 으로 통과했는가."""
        if self.blocked:
            return False
        if self.dry_run:
            return True
        return self.submitted and self.return_code == 0

    def to_record(self, ts: datetime | None = None) -> dict[str, Any]:
        return {
            "ts": (ts or datetime.now()).isoformat(sep=" ", timespec="seconds"),  # noqa: DTZ005
            "side": self.intent.side,
            "code": self.intent.code,
            "name": HELD_STOCK_NAMES.get(self.intent.code, ""),
            "qty": self.intent.qty,
            "price": self.intent.price,
            "exchange": self.intent.exchange,
            "dry_run": self.dry_run,
            "submitted": self.submitted,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "ord_no": self.ord_no,
            "return_code": self.return_code,
        }


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
        self.guard = guard or OrderGuardConfig()
        self._auth = KiwoomClient(
            appkey, secretkey, client=client, base_url=base_url, sleep=0.0, sleeper=sleeper
        )
        self._client = self._auth._client
        self._base = base_url.rstrip("/")
        self._sleep = sleep
        self._sleeper = sleeper or time.sleep
        self._on_order = on_order
        self._clock = clock or (lambda: datetime.now())  # noqa: DTZ005 — KST naive
        self._account_no: str | None = None
        # 주문 직전 실시간 잔고조회로 보유종목 매수를 이중차단(fail-closed).
        self._verify_holdings = bool(verify_holdings_before_order)
        self._holdings_ttl = float(holdings_ttl)
        self._holdings_cache: list[dict[str, Any]] | None = None
        self._holdings_at: float = 0.0
        # N-1: 실시간 보유 동적 병합은 **매수(buy)에만** 적용하는 별도 집합.
        # guard.blocklist(하드 3종목+설정)는 양방향 차단이지만, 우리가 방금 매수해
        # 보유가 된 후속주까지 여기에 넣으면 그 종목의 **청산 매도**까지 막혀 실포지션이
        # 무감시로 방치된다. 그래서 동적 보유분은 이 집합에 담아 매수만 차단한다.
        self._dynamic_buy_block: frozenset[str] = frozenset()
        # 일일 주문 횟수 카운터(메모리). 날짜 바뀌면 리셋.
        self._count_day: date | None = None
        self._count_total = 0
        self._count_by_code: dict[str, int] = {}

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
        if held:
            self._dynamic_buy_block = self._dynamic_buy_block | {
                normalize_code(c) for c in held if normalize_code(c)
            }
        if intent.code in self.guard.blocklist or intent.code in self._dynamic_buy_block:
            name = HELD_STOCK_NAMES.get(intent.code, "")
            return f"실시간 보유종목 차단{': ' + name if name else ''} — {intent.code}"
        return None

    def refresh_blocklist_from_account(self) -> set[str]:
        """계좌 보유 전체를 **매수 차단** 집합에 동적 병합. 보유 종목 매수 사고를 이중 차단.

        N-1: 청산 매도까지 막지 않도록 guard.blocklist(양방향) 가 아니라 매수 전용
        집합(:attr:`_dynamic_buy_block`)에 병합한다. 하드코딩 3종목만 양방향 차단.

        반환: 새로 추가된 보유 코드 집합.
        """
        held = {normalize_code(h["code"]) for h in self.holdings() if normalize_code(h["code"])}
        self._dynamic_buy_block = self._dynamic_buy_block | held
        return held

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

    # ----- 카운터 ------------------------------------------------------------

    def _observe_day(self) -> None:
        d = self._clock().date()
        if self._count_day != d:
            self._count_day = d
            self._count_total = 0
            self._count_by_code = {}

    def _count_ok(self, code: str) -> str | None:
        """횟수 상한 위반이면 사유 문자열, 아니면 None."""
        self._observe_day()
        g = self.guard
        if g.max_orders_total and self._count_total >= g.max_orders_total:
            return f"일일 주문 횟수 상한({g.max_orders_total}) 초과"
        if g.max_orders_per_code and self._count_by_code.get(code, 0) >= g.max_orders_per_code:
            return f"종목당 주문 횟수 상한({g.max_orders_per_code}) 초과 — {code}"
        return None

    def _bump_count(self, code: str) -> None:
        self._observe_day()
        self._count_total += 1
        self._count_by_code[code] = self._count_by_code.get(code, 0) + 1

    # ----- 가드 판정 ---------------------------------------------------------

    def guard_reason(self, intent: OrderIntent, ref_price: float | None) -> str | None:
        """주문을 막을 사유. None 이면 통과. **dry-run 에서도 항상 돈다.**"""
        g = self.guard
        code = intent.code
        if not code or len(code) != 6:
            return f"잘못된 종목코드: {code!r}"
        # (1) 보유 3종목 하드 차단 — 최우선. **매수·매도 양방향**(우리가 안 파는 종목).
        if code in g.blocklist:
            name = HELD_STOCK_NAMES.get(code, "")
            return f"블록리스트 차단(보유/금지 종목{': ' + name if name else ''}) — {code}"
        # (1b) 동적 보유 병합 — **매수에만** 적용(N-1). 청산 매도는 막지 않는다.
        if intent.side == "buy" and code in self._dynamic_buy_block:
            name = HELD_STOCK_NAMES.get(code, "")
            return f"실시간 보유종목 차단(매수){': ' + name if name else ''} — {code}"
        # (2) 화이트리스트 — 비어있지 않으면 그 외 전부 차단.
        if g.whitelist and code not in g.whitelist:
            return f"화이트리스트 외 종목 — {code} (등록 쌍 후속주만 허용)"
        # (3) 수량 상한.
        if intent.qty < 1:
            return f"수량이 1주 미만: {intent.qty}"
        if intent.qty > g.max_qty_hard_cap:
            return f"수량 하드캡({g.max_qty_hard_cap}) 초과: {intent.qty}"
        if intent.qty > g.max_qty:
            return f"수량 상한({g.max_qty}) 초과: {intent.qty}"
        # (4) 가격 범위.
        if intent.price <= 0:
            return f"지정가가 0 이하: {intent.price}"
        if g.price_band_pct and ref_price and ref_price > 0:
            dev = abs(intent.price / float(ref_price) - 1.0)
            if dev > g.price_band_pct + 1e-9:
                return (f"지정가 {intent.price:,} 가 참조가 {ref_price:,.0f} 대비 "
                        f"{dev:.2%} (±{g.price_band_pct:.1%} 밖)")
        # (5) 횟수 상한.
        return self._count_ok(code)

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
        self._bump_count(intent.code)

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
        body = {"dmst_stex_tp": exchange, "stk_cd": code,
                "org_ord_no": str(ord_no), "ord_qty": str(int(qty))}
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
