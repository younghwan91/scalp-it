"""종가베팅 백테스트 엔진 — D 종가 매수, D+1 청산.

일봉만으로 정확히 계산되는 전략이라 분봉을 기다릴 필요가 없다.
피처는 전부 D 종가 시점에 알 수 있는 값만 쓴다 (:mod:`.features` 참고).
"""

from __future__ import annotations

from .backtest import run_close_bet
from .features import CB_FEATURE_COLUMNS, close_bet_features
from .labels import CB_LABEL_COLUMNS, close_bet_labels

__all__ = [
    "CB_FEATURE_COLUMNS",
    "CB_LABEL_COLUMNS",
    "close_bet_features",
    "close_bet_labels",
    "run_close_bet",
]
