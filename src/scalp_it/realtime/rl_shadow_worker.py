"""그림자를 매매 루프에서 떼어내는 작업 스레드.

창 재계산 커널은 종목당 1초 **81ms** 다(2026-09-16 실측). 8종목이면 초당 650ms —
이걸 소켓 이벤트 루프 안에서 돌리면 주문 판단이 그만큼 밀린다. 장중 지연은 곧 손실이라
프레임 전달은 **큐에 넣는 것으로 끝**내고, 계산은 별도 스레드가 한다. 큐가 넘치면 **버린다**
(그림자는 늦어도 되지만 매매는 늦으면 안 된다).

**버리는 중에 알린다.** 2026-09-18·21 에 8종목으로 돌려 프레임 115,547 / 76,005 건을 버리고도
종료 때까지 알리지 않아 이틀을 날렸다. 마감 때 아는 것은 늦다.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

__all__ = ["ShadowWorker"]


class ShadowWorker:
    def __init__(self, shadow: Any, *, maxsize: int = 20000,
                 warn: Callable[[str], None] | None = None, warn_every: int = 1000) -> None:
        """warn: 버릴 때 부를 알림(기본 없음). `warn_every` 건마다 한 번만 부른다 — 매번 떠들면 묻힌다."""
        self.shadow = shadow
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._warn = warn
        self._warn_every = max(1, int(warn_every))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="rl-shadow", daemon=True)
            self._thread.start()

    def feed_tick(self, code, ts, price, **kw) -> None:
        self._put(("tick", code, ts, price, kw))

    def feed_quote(self, code, ts, **kw) -> None:
        self._put(("quote", code, ts, None, kw))

    def _put(self, item: tuple) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            if self._warn is not None and self.dropped % self._warn_every == 0:
                self._warn(f"[RL그림자] 프레임을 버리고 있다 — 누적 {self.dropped:,}건. "
                           "창 커널이 종목 수를 못 따라간다(--rl-codes 를 줄여라)")

    def _run(self) -> None:
        while not self._stop.is_set() or not self.q.empty():
            try:
                kind, code, ts, price, kw = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "tick":
                    self.shadow.on_tick(code, ts, price, **kw)
                else:
                    self.shadow.on_quote(code, ts, **kw)
            except Exception:  # noqa: BLE001 — 그림자 오류가 매매를 건드리면 안 된다
                pass

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        try:
            self.shadow.flush()
        except Exception:  # noqa: BLE001
            pass
