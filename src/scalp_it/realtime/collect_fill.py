"""수집 슬롯 채우기 — 매매가 안 쓰는 등록 자리로 틱을 쌓는다.

감지기는 그날 대장·짝꿍 후보만 구독한다(만쥬 §2). 그래서 후보가 적은 날은 정원 50 중
10~20자리만 쓰고 나머지가 하루 종일 빈다 — 2026-08-28 실측 구독 12종목·저장 틱 6만행,
같은 달 8/24 는 80종목·180만행이었다.

사전등록 47번의 결론은 "굳힘을 가르는 정보는 일봉에 없고 장중 미시구조에 있다"였다.
그 데이터를 모으는 게 지금 병목인데, 자리를 비워두는 건 그냥 손해다.

**여기서 붙이는 종목은 감지기에 등록하지 않는다.** 구독은 저장만 만들고, 신호와 주문은
감지기가 아는 종목에서만 나온다(`PairDetector.add_codes` 를 안 부른다).
"""
from __future__ import annotations

from collections.abc import Sequence


def plan_fill(candidates: Sequence[str], subscribed: Sequence[str],
              *, cap: int, reserve: int) -> list[str]:
    """수집용으로 새로 붙일 종목 → 후보 순서대로, 남는 자리만큼.

    ``reserve`` 는 감지기가 장중에 대장을 붙일 자리다. 이걸 안 남기면 수집이 정원을
    다 먹고 그날 대장을 못 붙인다 — 매매가 데이터에 밀리는 건 본말전도다.
    """
    have = set(subscribed)
    room = cap - len(have) - reserve
    if room <= 0:
        return []
    fresh = [c for c in dict.fromkeys(candidates) if c not in have]
    return fresh[:room]
