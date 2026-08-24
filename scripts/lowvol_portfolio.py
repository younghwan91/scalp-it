"""저변동성 실전 포트폴리오 생성 — 지금 매수/매도할 종목 (스크립트 진입점).

31번 채택 후 배포 산출물. CLI ``scalp-lowvol-portfolio`` 와 동일 로직(공용 모듈
:mod:`scalp_it.lowvol`)을 쓰며, PEAD 의 ``scripts/pead_portfolio.py`` 와 대칭이다.
리밸일(오늘 이하 최신 거래일) 기준 폴드내 데실로 롱(Q10 저변동)·숏(Q1 고변동)을 낸다.
결과 → scratchpad lowvol_current_portfolio.md 저장 + data/ CSV. DB 쓰기 없음.
"""
from __future__ import annotations

import sys

from scalp_it.cli_lowvol_portfolio import main as cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main(sys.argv[1:]))
