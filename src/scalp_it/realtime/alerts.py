"""짝꿍 알림 채널 — 사람에게 알린다. **주문 자동화는 하지 않는다(알림만).**

채널(옵션):

* :class:`LogFileSink` — 항상. ``data/pair_alerts.log`` 에 JSON 한 줄 + 사람이 읽는
  한 줄. stdout 로도 찍고, ``bell`` 이면 터미널 벨(\\a)을 울린다.
* :class:`DiscordSink` — **훅 자리**. 환경변수 ``PAIR_ALERT_DISCORD_WEBHOOK`` 에 웹훅
  URL 이 있으면 stdlib ``urllib`` 로 POST 한다(외부 의존성 없음). 없으면 조용히 끈다.
  이 레포의 discord 플러그인(MCP)은 대화형 에이전트용이라 데몬 프로세스에서 직접 못
  부른다 — 데몬은 웹훅으로, 감지 결과를 사람이 대화로 받고 싶으면 로그를 태우면 된다.

:class:`AlertBus` 가 여러 sink 를 묶는다. ``dry_run`` 이면 어떤 sink 도 실제로 쓰지
않고 stdout 로만 미리보기한다(리플레이·검증용).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .pair_detector import Alert

__all__ = ["AlertBus", "AlertSink", "DiscordSink", "LogFileSink", "build_default_bus"]


class AlertSink(Protocol):
    def emit(self, alert: Alert) -> None: ...


class LogFileSink:
    """``data/pair_alerts.log`` 에 append + stdout. 항상 켠다."""

    def __init__(self, path: str | Path = "data/pair_alerts.log", *,
                 echo: bool = True, bell: bool = False) -> None:
        self.path = Path(path)
        self.echo = echo
        self.bell = bell
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: Alert) -> None:
        line = alert.to_line()
        record = json.dumps(alert.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record + "\n")
        if self.echo:
            prefix = "\a" if self.bell else ""
            print(prefix + line, flush=True)

    def notice(self, text: str) -> None:
        """알림이 아닌 상태 메시지(킬스위치 배너 등)를 로그·stdout 에 남긴다."""
        # 통지 본문엔 시각만 있어 날짜를 같이 남긴다 — 일지가 날짜로 걸러 읽는다.
        day = date.today().isoformat()  # noqa: DTZ011 — KST naive, 프로젝트 표준
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"notice": text, "day": day}, ensure_ascii=False) + "\n")
        if self.echo:
            print(text, flush=True)


class DiscordSink:
    """Discord 웹훅 훅. URL 없으면 비활성(``enabled=False``)."""

    def __init__(self, webhook_url: str | None = None, *, timeout: float = 5.0) -> None:
        self.webhook_url = webhook_url or os.environ.get("PAIR_ALERT_DISCORD_WEBHOOK")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def emit(self, alert: Alert) -> None:
        if not self.enabled:
            return
        payload = json.dumps({"content": alert.to_line()}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout)
        except (urllib.error.URLError, OSError) as exc:
            # 알림 채널 실패가 감지 루프를 죽이면 안 된다 — 경고만.
            print(f"[알림] Discord 웹훅 실패: {exc}", file=sys.stderr, flush=True)

    def notice(self, text: str) -> None:
        """상태 메시지(킬스위치 배너 등)를 웹훅으로."""
        if not self.enabled:
            return
        payload = json.dumps({"content": text}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[알림] Discord 웹훅 실패: {exc}", file=sys.stderr, flush=True)


class AlertBus:
    """여러 sink 를 묶어 한 번에 emit. ``dry_run`` 이면 stdout 미리보기만."""

    def __init__(self, sinks: list[AlertSink] | None = None, *, dry_run: bool = False) -> None:
        self.sinks = sinks or []
        self.dry_run = dry_run
        self.count = 0

    def emit(self, alert: Alert) -> None:
        self.count += 1
        if self.dry_run:
            print("[dry-run] " + alert.to_line(), flush=True)
            return
        for sink in self.sinks:
            sink.emit(alert)

    def notice(self, text: str) -> None:
        """알림이 아닌 상태 메시지를 sink 로 전달(킬스위치 배너 등)."""
        if self.dry_run:
            print("[dry-run] " + text, flush=True)
            return
        for sink in self.sinks:
            fn = getattr(sink, "notice", None)
            if fn is not None:
                fn(text)


def build_default_bus(*, log_path: str | Path = "data/pair_alerts.log",
                      bell: bool = False, discord: bool = True,
                      dry_run: bool = False) -> AlertBus:
    """로그(항상) + Discord(웹훅 있으면) 로 기본 버스를 만든다."""
    sinks: list[AlertSink] = [LogFileSink(log_path, bell=bell)]
    if discord:
        ds = DiscordSink()
        if ds.enabled:
            sinks.append(ds)
    return AlertBus(sinks, dry_run=dry_run)
