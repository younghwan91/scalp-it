"""휴장일 기사를 다음 거래일로 이월한다.

이 단계가 없으면 **주말·공휴일 기사가 통째로 사라진다.** 기사 날짜에 가격 행이
없으면 ``adv`` 가 ``NaN`` 이 되고, 랭킹의 유동성 하한 비교가 ``False`` 라
조용히 탈락하기 때문이다. 실패로 보이지 않아서 더 위험하다.

한국 시장에서 주말·연휴는 재료가 쌓이는 시간이다 — 해외 뉴스, 금요일 장 마감 후
공시, 주말 기획기사. 이걸 버리면 표본이 평일 기사 쪽으로 기운다.

**이월은 앞으로만 한다.** 토요일 기사를 금요일에 알 수는 없으므로 뒤로 당기면
lookahead 다. 원래 게재일은 ``published_date`` 로 보존한다 — 근거 카드에
"토요일 보도"라고 적어야 사람이 판단할 수 있다.

구현은 krx-quant-core ``market.calendar.align_to_trading_days`` 한 곳에만 둔다
(이 레포 원본의 정확한 이식). 이 모듈은 기존 import 경로를 살리는 재노출 shim 이다.
"""

from __future__ import annotations

from krx_quant_core.market.calendar import PUBLISHED_COLUMN, align_to_trading_days

__all__ = ["PUBLISHED_COLUMN", "align_to_trading_days"]
