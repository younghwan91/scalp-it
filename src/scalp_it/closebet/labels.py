"""종가베팅 라벨 — D 종가 매수 후 청산 방식별 수익률.

기본 청산은 **다음날 시가** (``ret_open``) 다. ``ret_high`` 는 다음날 장중
최고가로 판 상한이므로 달성 불가능한 낙관치이고, ``ret_close`` 는 하루 더
쥔 경우다. 셋 다 비용 차감 전 총수익이다.

거래정지·상장폐지로 다음 거래일 봉이 없으면 값은 NaN 이며, 그 행을
버리는 책임은 :mod:`.backtest` 에 있다 (몇 건을 버렸는지 보고해야 한다).
"""

from __future__ import annotations

import pandas as pd

#: 반환 프레임의 라벨 컬럼.
CB_LABEL_COLUMNS = ["ret_open", "ret_high", "ret_close", "gap_next"]


def close_bet_labels(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """(종목, 날짜) 마다 D+1 청산 수익률을 계산한다.

    Args:
        ohlcv: 최소 ``code, date, open, high, close``.

    Returns:
        ``[code, date, ret_open, ret_high, ret_close, gap_next]``.
    """
    missing = [c for c in ("code", "date", "open", "high", "close") if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv 에 컬럼이 없다: {missing}")
    if ohlcv.empty:
        return pd.DataFrame(columns=["code", "date", *CB_LABEL_COLUMNS])

    out = ohlcv.sort_values(["code", "date"]).reset_index(drop=True).copy()
    for col in ("open", "high", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    grouped = out.groupby("code", sort=False)
    close = out["close"].where(out["close"] != 0)
    next_open = grouped["open"].shift(-1)
    next_high = grouped["high"].shift(-1)
    next_close = grouped["close"].shift(-1)

    out["ret_open"] = (next_open - close) / close
    out["ret_high"] = (next_high - close) / close
    out["ret_close"] = (next_close - close) / close
    # 갭 분석용 별칭 — 정의상 ret_open 과 같은 값이다.
    out["gap_next"] = out["ret_open"]

    return out[["code", "date", *CB_LABEL_COLUMNS]].reset_index(drop=True)
