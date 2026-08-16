"""휴장일 기사를 다음 거래일로 이월한다.

이 단계가 없으면 **주말·공휴일 기사가 통째로 사라진다.** 기사 날짜에 가격 행이
없으면 ``adv`` 가 ``NaN`` 이 되고, 랭킹의 유동성 하한 비교가 ``False`` 라
조용히 탈락하기 때문이다. 실패로 보이지 않아서 더 위험하다.

한국 시장에서 주말·연휴는 재료가 쌓이는 시간이다 — 해외 뉴스, 금요일 장 마감 후
공시, 주말 기획기사. 이걸 버리면 표본이 평일 기사 쪽으로 기운다.

**이월은 앞으로만 한다.** 토요일 기사를 금요일에 알 수는 없으므로 뒤로 당기면
lookahead 다. 원래 게재일은 ``published_date`` 로 보존한다 — 근거 카드에
"토요일 보도"라고 적어야 사람이 판단할 수 있다.
"""

from __future__ import annotations

import pandas as pd

PUBLISHED_COLUMN = "published_date"


def align_to_trading_days(articles: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """기사의 ``date`` 를 그날 이후 첫 거래일로 옮긴다.

    Args:
        articles: 최소한 ``date`` 컬럼을 가진 프레임. 나머지 컬럼은 그대로 보존된다.
        calendar: 거래일 목록. 보통 가격 데이터의 ``date`` 를 넘긴다.

    Returns:
        ``date`` 가 거래일로 바뀌고 ``published_date`` 가 추가된 프레임.
        달력 마지막 거래일을 넘어선 기사는 **버린다** — 이월할 곳이 없고,
        남겨두면 라벨이 생기지 않아 뒤에서 조용히 떨어진다.
    """
    out = articles.copy()
    if PUBLISHED_COLUMN not in out.columns:
        out[PUBLISHED_COLUMN] = out["date"]

    if out.empty:
        return out

    trading_days = pd.Index(pd.to_datetime(pd.Series(calendar)).unique()).sort_values()
    if trading_days.empty:
        return out.iloc[0:0]

    published = pd.to_datetime(out[PUBLISHED_COLUMN])
    # searchsorted "left" 는 거래일 당일이면 자기 자신을, 휴장일이면 다음 거래일을 준다.
    position = trading_days.searchsorted(published, side="left")
    within = position < len(trading_days)

    out = out.loc[within.tolist()].copy()
    out["date"] = trading_days[position[within]]
    return out.reset_index(drop=True)
