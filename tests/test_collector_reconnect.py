"""소켓 재연결 비용 — **시세 피드에서 백오프 1초는 데이터 1초다.**

2026-10-02 실측: 한 세션에 끊김 185회, 그중 27회는 opening handshake 타임아웃. 백오프가 2배씩
늘어 최대 30초, 핸드셰이크 기본 타임아웃 10초라 연속 실패 한 번에 40초를 잃었다. 결과로 2~7분
공백이 85~90개 생겨 분커버가 37~52% 로 떨어졌다(정상일은 96~97%).
"""
import inspect

from scalp_it.realtime.collector import KiwoomRealtimeCollector as C


def test_backoff_cap_is_short_enough_for_a_market_feed():
    cap = inspect.signature(C.__init__).parameters["max_backoff"].default
    assert cap <= 5.0, f"백오프 상한 {cap}s — 시세 피드에 너무 길다(그만큼이 결측이다)"


def test_handshake_timeout_is_explicit_and_short():
    """기본 10초를 그대로 쓰면 죽은 엔드포인트에 매번 10초를 버린다."""
    src = inspect.getsource(C._connect)
    assert "open_timeout" in src, "핸드셰이크 타임아웃을 명시하지 않았다"
    assert "ping_interval=None" in src, "키움 WS 는 자체 PING 을 쓴다 — 라이브러리 핑을 켜면 안 된다"


def test_backoff_still_grows_so_a_dead_endpoint_is_not_hammered():
    src = inspect.getsource(C._connection_loop)
    assert "backoff * 2" in src and "_max_backoff" in src


# ── 핸드셰이크 중에 오는 PING ─────────────────────────────────────────────────
# 2026-10-02 로그: `REG 실패: {'trnm': 'PING'}`. LOGIN·REG 응답을 recv() 한 번으로 받는데
# 그 사이 키움 PING 이 오면 응답으로 오인해 실패 처리하고, 끊고 재연결하고 다시 REG 한다.
# 재연결마다 REG 를 하니 이 오인이 끊김을 증폭시킨다(그날 끊김 185회, 분커버 52%).
import asyncio
import json

import pytest


class FakeWS:
    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []

    async def send(self, raw):
        self.sent.append(raw)

    async def recv(self):
        if not self._frames:
            raise AssertionError("프레임을 다 썼는데 또 받으려 한다")
        return self._frames.pop(0)


def _collector():
    c = C.__new__(C)
    c._token = "tok"
    c._types = ["0B", "0D"]
    c._log = lambda m: None
    return c


def test_ping_during_handshake_is_answered_and_skipped():
    c = _collector()
    ws = FakeWS([
        json.dumps({"trnm": "PING"}),
        json.dumps({"trnm": "LOGIN", "return_code": 0}),
        json.dumps({"trnm": "PING"}),
        json.dumps({"trnm": "REG", "return_code": 0}),
    ])
    asyncio.run(c._login_and_reg(ws, "1", ["005930"]))
    pings = [s for s in ws.sent if json.loads(s).get("trnm") == "PING"]
    assert len(pings) == 2, f"PING 을 그대로 되돌려주지 않았다: {ws.sent}"


def test_real_failure_is_still_a_failure():
    c = _collector()
    ws = FakeWS([
        json.dumps({"trnm": "LOGIN", "return_code": 0}),
        json.dumps({"trnm": "REG", "return_code": 1, "return_msg": "거절"}),
    ])
    with pytest.raises(RuntimeError, match="REG 실패"):
        asyncio.run(c._login_and_reg(ws, "1", ["005930"]))


def test_login_failure_is_still_a_failure():
    c = _collector()
    ws = FakeWS([json.dumps({"trnm": "LOGIN", "return_code": 1})])
    with pytest.raises(RuntimeError, match="LOGIN 실패"):
        asyncio.run(c._login_and_reg(ws, "1", ["005930"]))
