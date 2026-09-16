"""RL 정책 → 실주문 다리. **승격 토큰 + --rl-live 둘 다 있을 때만** 주문이 나간다.

승격 체크리스트(docs/PROMOTION_CHECKLIST.md)의 11개 항목을 `scripts/promotion_check.py` 가 전부 통과시키면
`research/runs/<study>/PROMOTED` 를 만든다. 이 다리는 그 파일이 있을 때만 열린다 —
검증 안 된 정책이 실수로 주문을 내는 경로를 코드 수준에서 막는 게 목적이다.

주문 실패는 삼켜서 기록만 남긴다. 그림자 기록이 끊기면 의사결정 패리티(9번)를 못 재기 때문이다.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["make_sink", "is_promoted"]


def is_promoted(study: str, runs_dir: str | Path) -> bool:
    return (Path(runs_dir) / study / "PROMOTED").exists()


def make_sink(base_sink: Callable[[dict[str, Any]], None], *, executor: Any, study: str,
              runs_dir: str | Path, live: bool) -> Callable[[dict[str, Any]], None]:
    """RLShadow 의 sink 를 감싸 주문까지 내는 sink 를 만든다."""
    promoted = is_promoted(study, runs_dir)

    def sink(rec: dict[str, Any]) -> None:
        ev = rec.get("event")
        if ev in ("buy", "sell"):
            if not promoted:
                rec = {**rec, "order": "차단(미승격)"}
            elif not live:
                rec = {**rec, "order": "그림자"}
            elif executor is None:
                rec = {**rec, "order": "차단(실행기 없음)"}
            else:
                try:
                    fn = executor.submit_buy if ev == "buy" else executor.submit_sell
                    fn(rec["code"], rec["price"], qty=1)
                    rec = {**rec, "order": "실주문"}
                except Exception as exc:  # noqa: BLE001 — 주문 실패가 기록을 끊으면 패리티를 못 잰다
                    rec = {**rec, "order": f"실패: {exc}"}
        base_sink(rec)

    return sink
