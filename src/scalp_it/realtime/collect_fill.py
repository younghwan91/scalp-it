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


def fill_pool(movers: Sequence[str], turnover: Sequence[str]) -> list[str]:
    """수집 후보 순서 — **오늘 등락률 상위가 먼저**, 전일 거래대금이 뒤를 받친다.

    만쥬 방식이 성립하려면 대장이 상한가 3호가에 닿는 순간 이미 구독돼 있어야 한다.
    전일 거래대금으로 고르면 그 순간을 못 본다 — 실측 커버리지 2%(그 주 상한가 52건
    중 1건)가 그 얘기였다. 스캐너가 10초마다 받아오는 등락률 순위를 **낙점 문턱
    아래까지** 후보로 써서, 12% 를 못 넘은 종목이 이따가 올라올 때 이미 붙어 있게 한다.

    전일 거래대금은 스캔이 아직 한 바퀴도 안 돌았을 때(장 시작 직후)의 폴백이다.
    """
    return list(dict.fromkeys([*movers, *turnover]))


def plan_evictions(last_seen: dict[str, float], *, now: float, ttl: float) -> list[str]:
    """자리를 내줄 수집 종목 — 순위에서 사라진 지 ``ttl`` 초가 지난 것, 오래된 순.

    자리를 한 번 잡고 안 놓으면 오전에 붙은 종목이 하루를 다 먹는다. 만쥬의 대장은
    오후에도 나오고, 그때 자리가 없으면 그날 그 종목은 못 본다. 반대로 순위를 잠깐
    들락거린다고 바로 떼면 등록만 낭비하므로 유예를 둔다.
    """
    stale = [(t, c) for c, t in last_seen.items() if now - t >= ttl]
    return [c for _, c in sorted(stale)]
