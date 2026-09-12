"""키움 REST 어댑터 — 키움 HTTP 호출은 이 파일에만 존재한다.

일중 단타로 넘어가면서 분봉이 모든 후속 작업의 전제가 됐다. 분봉은
``ka10080`` (주식분봉차트조회) 하나로 받는다.

실제 응답에서 확인한 사실만 코드에 반영한다.

* 토큰 응답의 키는 ``token`` 이다. ``access_token`` 이 아니다.
* 한 콜에 900행(≈2.5 거래일)이 오고, 최신 봉이 맨 앞이다(내림차순).
* 다음 페이지 여부는 **응답 헤더** ``cont-yn`` / ``next-key`` 로 온다.
* 가격 필드에는 ``+`` / ``-`` 부호가 붙어 있다. 부호를 떼고 정수로 읽는다.
  이 부호는 전일 대비 방향 표시일 뿐 값의 음양이 아니다 — ``-258000`` 은
  마이너스 25만 원이 아니라 '전일 대비 하락한 258,000원'이다.
* ``ka10080`` 응답에 거래대금 필드는 **없다**. 실측 필드는
  ``cntr_tm, cur_prc, open_pric, high_pric, low_pric, trde_qty,
  acc_trde_qty, pred_pre, pred_pre_sig`` 뿐이다.

테마는 ``ka90001`` (테마그룹별요청) 과 ``ka90002`` (테마구성종목요청) 두 개다.
키움 테마 API 는 **현재 스냅샷**만 주므로 매일 저장해야 과거 소속 히스토리가 생긴다.
2026-08-23 실측:

* 경로는 ``/api/dostk/thme``. 헤더·토큰·연속조회(``cont-yn``/``next-key``)는
  분봉과 같다. ``ka90001`` 은 한 페이지 100행이고 ``cont-yn: Y`` 로 이어진다.
* ``ka90001`` 요청 바디 ``qry_tp=0``(전체) · ``stk_cd=""`` · ``date_tp``(기간수익률
  일수) · ``thema_nm=""`` · ``flu_pl_amt_tp``(정렬) · ``stex_tp``(1=KRX). 응답의
  테마 행은 ``thema_grp`` 배열이고 필드는 ``thema_grp_cd, thema_nm, stk_num, flu_sig,
  flu_rt, rising_stk_num, fall_stk_num, dt_prft_rt, main_stk`` 다.
* ``ka90002`` 요청 바디 ``thema_grp_cd, date_tp, stex_tp``. 응답의 구성종목 행은
  ``thema_comp_stk`` 배열이고 필드는 ``stk_cd, stk_nm, cur_prc, flu_sig, pred_pre,
  flu_rt, acc_trde_qty, sel_bid, sel_req, buy_bid, buy_req, dt_prft_rt_n`` 다.
  최상위에 테마 등락률 ``flu_rt`` · 기간수익률 ``dt_prft_rt`` 가 따로 온다.
* 등락률·수익률(``flu_rt, dt_prft_rt, dt_prft_rt_n``)의 부호는 가격과 달리
  **실제 음양**이다 (``-3.74`` 는 마이너스 3.74%). 가격(``cur_prc`` 등)은 분봉과
  같은 방향 표시 부호라 ``strip_sign`` 으로 뗀다.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
from kiwoom_client import KiwoomAPI, KiwoomAPIError

KIWOOM_BASE = "https://api.kiwoom.com"
TOKEN_PATH = "/oauth2/token"
CHART_PATH = "/api/dostk/chart"
THEME_PATH = "/api/dostk/thme"
MINUTE_API_ID = "ka10080"
THEME_GROUP_API_ID = "ka90001"
THEME_MEMBER_API_ID = "ka90002"

#: 분봉 행이 담겨 오는 키.
CHART_ROWS_KEY = "stk_min_pole_chart_qry"
#: 테마 목록 행 · 테마 구성종목 행이 담겨 오는 키.
THEME_GROUP_ROWS_KEY = "thema_grp"
THEME_MEMBER_ROWS_KEY = "thema_comp_stk"

_JSON_CT = "application/json;charset=UTF-8"
_KEY_ENV = "KIWOOM_APP_KEY"
_SECRET_ENV = "KIWOOM_APP_SECRET"
_RETRY_STATUS = (429, 500, 502, 503, 504)


class KiwoomError(Exception):
    """키움이 오류를 돌려주었거나 인증 정보가 없다."""


def api_credentials(env: dict | None = None) -> tuple[str, str]:
    """``KIWOOM_APP_KEY`` / ``KIWOOM_APP_SECRET`` 를 읽는다."""
    source = os.environ if env is None else env
    appkey = str(source.get(_KEY_ENV) or "").strip()
    secret = str(source.get(_SECRET_ENV) or "").strip()
    if not appkey or not secret:
        raise KiwoomError(
            "키움 인증 정보가 없다. KIWOOM_APP_KEY 와 KIWOOM_APP_SECRET 을 설정하라 "
            "(quant-airflow/.env 에 이미 있다면 export 해서 쓰면 된다)."
        )
    return appkey, secret


def strip_sign(value: Any) -> int:
    """``+274500`` / ``-258000`` 같은 부호 붙은 값을 정수로 만든다.

    부호는 전일 대비 방향 표시라 값의 음양이 아니다. 그래서 떼고 읽는다.
    빈 값은 0 이다 — 거래 없는 분봉이 실제로 온다.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return 0
    text = text.lstrip("+-")
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError as exc:
        raise KiwoomError(f"숫자로 읽을 수 없는 값이다: {value!r}") from exc


def parse_cntr_tm(value: Any) -> datetime:
    """``YYYYMMDDHHMMSS`` 체결시각을 ``datetime`` 으로 읽는다.

    한국 거래소 시각이라 tz 없는 naive 로 둔다. DB 의 ``timestamp`` 컬럼도
    tz 없는 KST 로 통일돼 있다.
    """
    text = str(value if value is not None else "").strip()
    # 길이를 먼저 본다. 잘린 값은 예외가 아니라 **엉뚱한 시각**이 되기 때문이다 —
    # ``2026081415`` 는 %Y%m%d%H%M%S 로 2026-08-01 04:01:05 로 조용히 읽힌다.
    if len(text) != 14 or not text.isdigit():
        raise KiwoomError(f"체결시각을 읽을 수 없다: {value!r}")
    try:
        return pd.to_datetime(text, format="%Y%m%d%H%M%S").to_pydatetime()
    except ValueError as exc:
        raise KiwoomError(f"체결시각을 읽을 수 없다: {value!r}") from exc


def normalize_row(code: str, row: dict) -> dict[str, Any]:
    """응답 한 행을 저장 계약(``minute_bars``)에 맞춘다.

    ``trade_value`` 는 응답에 없어서 ``close × volume`` 으로 **근사**한다.
    분 안의 체결가 분포를 모르므로 정확한 거래대금이 아니다.
    """
    close = strip_sign(row.get("cur_prc"))
    volume = strip_sign(row.get("trde_qty"))
    return {
        "code": code,
        "ts": parse_cntr_tm(row.get("cntr_tm")),
        "open": strip_sign(row.get("open_pric")),
        "high": strip_sign(row.get("high_pric")),
        "low": strip_sign(row.get("low_pric")),
        "close": close,
        "volume": volume,
        "trade_value": float(close) * float(volume),
    }


def parse_rate(value: Any) -> float | None:
    """``+25.22`` / ``-3.74`` 같은 등락률·수익률을 부호 **그대로** 실수로 읽는다.

    가격과 달리 여기 부호는 실제 음양이다. 빈 값은 ``None`` 이다.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise KiwoomError(f"비율로 읽을 수 없는 값이다: {value!r}") from exc


def _int_or_none(value: Any) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise KiwoomError(f"정수로 읽을 수 없는 값이다: {value!r}") from exc


def normalize_theme_group(row: dict) -> dict[str, Any]:
    """``ka90001`` 한 행을 ``theme_groups`` 저장 계약에 맞춘다.

    실측 필드만 컬럼으로 뽑고, 응답 원문은 ``raw`` 에 그대로 남긴다 — 키움이
    필드를 바꾸거나 추가해도 원문은 잃지 않는다.
    """
    code = str(row.get("thema_grp_cd") or "").strip()
    if not code:
        raise KiwoomError(f"테마 행에 thema_grp_cd 가 없다: {sorted(row)}")
    return {
        "theme_code": code,
        "theme_name": str(row.get("thema_nm") or "").strip(),
        "stk_num": _int_or_none(row.get("stk_num")),
        "flu_rt": parse_rate(row.get("flu_rt")),
        "dt_prft_rt": parse_rate(row.get("dt_prft_rt")),
        "rising_stk_num": _int_or_none(row.get("rising_stk_num")),
        "fall_stk_num": _int_or_none(row.get("fall_stk_num")),
        "main_stk": str(row.get("main_stk") or "").strip(),
        "raw": dict(row),
    }


def normalize_theme_member(theme_code: str, row: dict) -> dict[str, Any]:
    """``ka90002`` 한 행을 ``theme_members`` 저장 계약에 맞춘다."""
    code = str(row.get("stk_cd") or "").strip()
    if not code:
        raise KiwoomError(f"구성종목 행에 stk_cd 가 없다: {sorted(row)}")
    return {
        "theme_code": str(theme_code),
        "code": code,
        "name": str(row.get("stk_nm") or "").strip(),
        "cur_prc": strip_sign(row.get("cur_prc")),
        "flu_rt": parse_rate(row.get("flu_rt")),
        "acc_trde_qty": strip_sign(row.get("acc_trde_qty")),
        "dt_prft_rt_n": parse_rate(row.get("dt_prft_rt_n")),
        "raw": dict(row),
    }


def since_to_datetime(since: str | None) -> datetime | None:
    """``--since`` 인자(YYYYMMDD)를 하한 시각으로 바꾼다."""
    if not since:
        return None
    text = str(since).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise KiwoomError(f"--since 는 YYYYMMDD 여야 한다: {since!r}")
    try:
        return pd.to_datetime(text, format="%Y%m%d").to_pydatetime()
    except ValueError as exc:
        raise KiwoomError(f"--since 는 YYYYMMDD 여야 한다: {since!r}") from exc


def _raise_on_return_code(payload: Any, what: str) -> None:
    """``return_code`` 가 0 이 아니면 메시지를 붙여 던진다."""
    if not isinstance(payload, dict):
        raise KiwoomError(f"{what}: 응답이 JSON 객체가 아니다.")
    code = payload.get("return_code")
    if code is None:
        return
    try:
        ok = int(code) == 0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise KiwoomError(f"{what} 실패: return_code={code} {payload.get('return_msg')}")


class KiwoomClient:
    """토큰 발급·갱신과 ``ka10080`` · ``ka90001`` · ``ka90002`` · ``ka10027`` ·
    ``ka90005`` 페이지네이션을 담당한다.

    ``ka10080``(분봉)·``ka90001``/``ka90002``(테마)는 배치 전용(장 마감 후
    크론)이라 2026-09-12에 ``kiwoom_client.KiwoomAPI``로 옮겼다 — 실시간
    루프가 안 쓰는 걸 확인하고서다. ``ka10027``(등락률상위)·``ka90005``
    (프로그램매매, ``cli_pair_detect.py`` 라이브 루프가 장중 10초마다
    호출)는 그대로 아래 ``_fetch_rows``/``_post``를 쓴다 — 장이 열려 있어야
    실제 페이지네이션·에러 경로를 실측으로 검증할 수 있어서, 검증 전엔
    안 건드린다.
    """

    def __init__(
        self,
        appkey: str,
        secretkey: str,
        *,
        client: httpx.Client | None = None,
        sleep: float = 0.12,
        max_retries: int = 3,
        sleeper: Callable[[float], None] | None = None,
        base_url: str = KIWOOM_BASE,
        kiwoom_api: Any = None,
    ) -> None:
        if not appkey or not secretkey:
            raise KiwoomError("인증 정보가 비어 있다.")
        self._appkey = appkey
        self._secretkey = secretkey
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        self._max_retries = max(1, int(max_retries))
        self._sleeper = sleeper or time.sleep
        self._base = base_url.rstrip("/")
        self._token: str | None = None
        # 배치 전용 엔드포인트(분봉·테마)만 여기로 간다 — 주입 가능하게 해서
        # 테스트가 진짜 HTTP를 안 타고 kiwoom_client 경계에서 바로 목을 건다.
        self._kiwoom_api = kiwoom_api or KiwoomAPI(appkey, secretkey, base_url=base_url)

    # ------------------------------------------------------------------ 토큰

    def issue_token(self) -> str:
        """접근 토큰을 새로 받는다. 응답 키는 ``token`` 이다."""
        response = self._client.post(
            f"{self._base}{TOKEN_PATH}",
            headers={"Content-Type": _JSON_CT},
            json={
                "grant_type": "client_credentials",
                "appkey": self._appkey,
                "secretkey": self._secretkey,
            },
        )
        payload = response.json()
        _raise_on_return_code(payload, "토큰 발급")
        token = str(payload.get("token") or "").strip()
        if not token:
            raise KiwoomError(f"토큰 응답에 token 이 없다: {sorted(payload)}")
        self._token = token
        return token

    @property
    def token(self) -> str:
        """없으면 발급하고 있으면 재사용한다."""
        if not self._token:
            return self.issue_token()
        return self._token

    def refresh_token(self) -> str:
        """토큰을 버리고 다시 받는다 (401 대응)."""
        self._token = None
        return self.issue_token()

    # ------------------------------------------------------------------ 조회

    def _post(
        self,
        path: str,
        api_id: str,
        body: dict,
        cont_yn: str | None,
        next_key: str | None,
    ) -> httpx.Response:
        headers = {
            "Content-Type": _JSON_CT,
            "authorization": f"Bearer {self.token}",
            "api-id": api_id,
        }
        if cont_yn:
            headers["cont-yn"] = cont_yn
        if next_key:
            headers["next-key"] = next_key
        return self._client.post(f"{self._base}{path}", headers=headers, json=body)

    def _fetch_rows(
        self,
        path: str,
        api_id: str,
        body: dict,
        rows_key: str,
        what: str,
        *,
        cont_yn: str | None = None,
        next_key: str | None = None,
    ) -> tuple[list[dict], str | None, str | None, dict]:
        """한 페이지를 받아 ``(행, cont-yn, next-key, 응답 전체)`` 를 돌려준다.

        레이트리밋·401 토큰 갱신·5xx 재시도는 모든 API 가 같으므로 여기 한 곳에 있다.
        """
        last_error = ""
        for attempt in range(self._max_retries):
            # 레이트리밋. 호출 **직전**에 재우므로 재시도에도 간격이 보장된다.
            if self._sleep:
                self._sleeper(self._sleep)
            response = self._post(path, api_id, body, cont_yn, next_key)
            if response.status_code == 401 and attempt + 1 < self._max_retries:
                # 토큰 만료. 한 번 다시 받고 재시도한다.
                self.refresh_token()
                last_error = "401 토큰 만료"
                continue
            if response.status_code in _RETRY_STATUS and attempt + 1 < self._max_retries:
                last_error = f"HTTP {response.status_code}"
                self._sleeper(self._sleep * (attempt + 1) + 1.0)
                continue
            if response.status_code >= 400:
                raise KiwoomError(f"키움 HTTP {response.status_code}: {response.text[:200]}")
            payload = response.json()
            _raise_on_return_code(payload, what)
            rows = payload.get(rows_key) or []
            # 빈 응답은 오류가 아니다. 신규 상장·거래정지 종목이 실제로 그렇다.
            return (
                list(rows),
                response.headers.get("cont-yn"),
                response.headers.get("next-key"),
                payload,
            )
        raise KiwoomError(f"{what} 재시도 소진: {last_error}")

    def _via_kiwoom_api(self, what: str, fn: Callable[[], dict], rows_key: str) -> tuple[list[dict], str | None, str | None]:
        """``kiwoom_client`` 경계를 거치는 배치 호출 공통 처리.

        ``KiwoomAPIError``를 이 레포의 ``KiwoomError``로 바꿔 던진다 — 호출부는
        전송 계층이 바뀐 걸 몰라도 된다. cont-yn/next-key는 kiwoom-client의
        ``request()``가 응답 헤더에서 바디로 옮겨준다(2026-09-12 kiwoom-client
        자체 버그로 발견·수정 — 안 그러면 페이지가 있어도 1페이지 만에 조용히
        멈춘다).
        """
        try:
            payload = fn()
        except KiwoomAPIError as exc:
            raise KiwoomError(f"{what} 실패: {exc}") from exc
        rows = payload.get(rows_key) or []
        return list(rows), payload.get("cont-yn"), payload.get("next-key")

    def fetch_page(
        self,
        code: str,
        *,
        tic_scope: str = "1",
        cont_yn: str | None = None,
        next_key: str | None = None,
    ) -> tuple[list[dict], str | None, str | None]:
        """분봉 한 페이지. ``(행, cont-yn, next-key)`` 를 돌려준다."""
        return self._via_kiwoom_api(
            f"{code} 분봉",
            lambda: self._kiwoom_api.chart.stock_minute_chart(
                cont_yn=cont_yn or "N", next_key=next_key or "",
                stk_cd=code, tic_scope=str(tic_scope), upd_stkpc_tp="1",
            ),
            CHART_ROWS_KEY,
        )

    def fetch_rank_rows(self, *, market: str = "101") -> list[dict]:
        """``ka10027`` 전일대비등락률상위 — **시장 전체**를 한 번에 훑는다.

        만쥬가 HTS 등락률 상위·순간체결량 창으로 대장을 낙점하는 자리다. 응답에
        등락률·매도잔량(0 이면 상한가 잠김)·매수잔량·체결강도가 같이 온다.
        페이지네이션은 안 한다 — 첫 페이지(최대 200행)면 대장 후보는 다 들어온다.
        """
        from ..realtime.leader_scan import RANK_API_ID, RANK_PATH, RANK_ROWS_KEY, rank_body

        rows, _cont, _key, _payload = self._fetch_rows(
            RANK_PATH, RANK_API_ID, rank_body(market), RANK_ROWS_KEY, "등락률 상위",
        )
        return rows

    def fetch_program_rows(self, *, day: str, market: str = "P101") -> list[dict]:
        """``ka90005`` 프로그램매매추이 — 분 단위 시장 전체 프로그램 수급.

        만쥬가 "외인/기관 프로그램이 쏟아내는 장세면 진입을 거른다"고 한 자리다.
        장중에는 첫 행이 직전 분이라 첫 페이지만 받으면 된다.
        """
        from ..realtime.program_flow import (
            PROGRAM_API_ID,
            PROGRAM_PATH,
            PROGRAM_ROWS_KEY,
            program_body,
        )

        rows, _cont, _key, _payload = self._fetch_rows(
            PROGRAM_PATH, PROGRAM_API_ID, program_body(day, market),
            PROGRAM_ROWS_KEY, "프로그램 매매추이",
        )
        return rows

    def minute_chart(
        self,
        code: str,
        *,
        tic_scope: str = "1",
        pages: int = 40,
        since: str | None = None,
        stop_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """분봉을 페이지네이션으로 모아 정규화된 행 리스트로 돌려준다.

        최신 봉이 먼저 오므로 아래로 내려갈수록 과거다. ``since`` (YYYYMMDD)
        이전 봉이 나오면 그 종목을 거기서 멈춘다. ``stop_at`` 은 이어받기용으로,
        이미 저장된 최신 ``ts`` 이하가 나오면 멈춘다 — 같은 명령을 다시 돌려도
        이미 받은 구간을 다시 받지 않는다.
        """
        floor = since_to_datetime(since)
        collected: list[dict[str, Any]] = []
        cont_yn: str | None = None
        next_key: str | None = None

        for _ in range(max(0, int(pages))):
            rows, cont_yn, next_key = self.fetch_page(
                code, tic_scope=tic_scope, cont_yn=cont_yn, next_key=next_key
            )
            if not rows:
                break
            stop = False
            for row in rows:
                item = normalize_row(code, row)
                if floor is not None and item["ts"] < floor:
                    stop = True
                    break
                if stop_at is not None and item["ts"] <= stop_at:
                    stop = True
                    break
                collected.append(item)
            if stop or str(cont_yn or "").upper() != "Y" or not next_key:
                break
        return collected

    # ------------------------------------------------------------------ 테마

    def fetch_theme_groups_page(
        self,
        *,
        date_tp: str = "1",
        flu_pl_amt_tp: str = "1",
        stex_tp: str = "1",
        cont_yn: str | None = None,
        next_key: str | None = None,
    ) -> tuple[list[dict], str | None, str | None]:
        """``ka90001`` 한 페이지(100행). ``(행, cont-yn, next-key)`` 를 돌려준다.

        ``qry_tp=0`` 으로 전체 테마를 받는다. ``date_tp`` 는 기간수익률 일수,
        ``flu_pl_amt_tp`` 는 정렬 기준(1=기간수익률 상위), ``stex_tp`` 는 거래소
        (1=KRX) 다. 행 자체는 원문 그대로다 — 정규화는 ``normalize_theme_group``.
        """
        return self._via_kiwoom_api(
            "테마 목록",
            lambda: self._kiwoom_api.theme.theme_group_list(
                cont_yn=cont_yn or "N", next_key=next_key or "",
                qry_tp="0", stk_cd="", date_tp=str(date_tp), thema_nm="",
                flu_pl_amt_tp=str(flu_pl_amt_tp), stex_tp=str(stex_tp),
            ),
            THEME_GROUP_ROWS_KEY,
        )

    def fetch_theme_members_page(
        self,
        theme_code: str,
        *,
        date_tp: str = "2",
        stex_tp: str = "1",
        cont_yn: str | None = None,
        next_key: str | None = None,
    ) -> tuple[list[dict], str | None, str | None]:
        """``ka90002`` 한 페이지. ``(행, cont-yn, next-key)`` 를 돌려준다."""
        return self._via_kiwoom_api(
            f"테마 {theme_code} 구성종목",
            lambda: self._kiwoom_api.theme.theme_component_stocks(
                cont_yn=cont_yn or "N", next_key=next_key or "",
                thema_grp_cd=str(theme_code), date_tp=str(date_tp), stex_tp=str(stex_tp),
            ),
            THEME_MEMBER_ROWS_KEY,
        )

    def theme_groups(
        self,
        *,
        pages: int = 50,
        date_tp: str = "1",
        flu_pl_amt_tp: str = "1",
        stex_tp: str = "1",
    ) -> list[dict[str, Any]]:
        """전 테마 목록을 페이지네이션으로 모아 정규화된 행 리스트로 돌려준다.

        같은 테마코드가 두 번 오면 처음 것만 남긴다 — 페이지 경계에서 정렬이
        흔들려 중복이 올 수 있다.
        """
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        cont_yn: str | None = None
        next_key: str | None = None
        for _ in range(max(0, int(pages))):
            rows, cont_yn, next_key = self.fetch_theme_groups_page(
                date_tp=date_tp, flu_pl_amt_tp=flu_pl_amt_tp, stex_tp=stex_tp,
                cont_yn=cont_yn, next_key=next_key,
            )
            if not rows:
                break
            for row in rows:
                item = normalize_theme_group(row)
                if item["theme_code"] in seen:
                    continue
                seen.add(item["theme_code"])
                collected.append(item)
            if str(cont_yn or "").upper() != "Y" or not next_key:
                break
        return collected

    def theme_members(
        self,
        theme_code: str,
        *,
        date_tp: str = "2",
        stex_tp: str = "1",
        pages: int = 10,
    ) -> list[dict[str, Any]]:
        """``ka90002`` 로 한 테마의 구성종목을 정규화된 행 리스트로 돌려준다.

        실측으로는 한 페이지에 다 오지만(``cont-yn: N``), 혹시 이어지면 따라간다.
        """
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        cont_yn: str | None = None
        next_key: str | None = None
        for _ in range(max(0, int(pages))):
            rows, cont_yn, next_key = self.fetch_theme_members_page(
                theme_code, date_tp=date_tp, stex_tp=stex_tp,
                cont_yn=cont_yn, next_key=next_key,
            )
            if not rows:
                break
            for row in rows:
                item = normalize_theme_member(theme_code, row)
                if item["code"] in seen:
                    continue
                seen.add(item["code"])
                collected.append(item)
            if str(cont_yn or "").upper() != "Y" or not next_key:
                break
        return collected
