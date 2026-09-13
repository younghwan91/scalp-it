# scripts/ 색인

39개 파일 전부 살아있는 코드다 — 각각 사전등록 문서의 "산출물"이거나 지금
도는 운영 스크립트라 **삭제하지 않는다**(재현성이 이 레포 방법론의 핵심).
길을 잃지 않기 위한 색인만 여기 둔다. 각 파일 자체의 docstring이 원 출처다.

## A. 운영 — 지금 cron 으로 매일 돈다

| 파일 | 역할 | cron |
|---|---|---|
| `pair_detect_daily.sh` | 짝꿍 감지기 실행(**실주문**, `--confirm-real`) | 평일 08:55 |
| `db_guard.sh` | 장중 DB 헬스가드 | 평일 08-15시 5분마다 |
| `pair_daily_summary.py` | dry-run 일일 성과 집계 | 평일 15:25 |
| `pair_journal.py` | 매매일지(문서44 §10-2) | 평일 15:26 |
| `tick_sanity.py` | 틱 sanity(28번) — 겸용, 아래 E 참고 | 평일 16:00 |
| `news_judge_shadow.py` | 뉴스판단 반응매매 그림자(55번) | 평일 16:05 |

## B. 사전등록 이벤트 수집 (`*_events.py`) — "이벤트 규모" 절 산출

측정 전에 표본 크기·분포를 먼저 보는 용도. 문서 번호 순:

| 파일 | 문서 |
|---|---|
| `pair_events.py` | [12](../docs/research/manju/12-pair-preregistration.md) |
| `dip_events.py` | [10](../docs/research/manju/10-dip-preregistration.md) |
| `breakout_events.py` | [09](../docs/research/manju/09-breakout-preregistration.md) |
| `newlisting_events.py` | [15](../docs/research/manju/15-newlisting-preregistration.md) |
| `traj_events.py` | [16](../docs/research/manju/16-intraday-trajectory-preregistration.md) |
| `closebet2_events.py` | [14](../docs/research/manju/14-closebet-v2-preregistration.md) |
| `dart_catalyst_events.py` | [21](../docs/research/manju/21-dart-catalyst-preregistration.md) |
| `openbet_events.py` | (openbet 확정 검증 부속) |
| `manju_events.py` | 테마별 후속주 수집(실전 테마, 08-27) |

## C. 확정 검증 (`verify_*.py`) — 각 문서의 최종 판정

이벤트 생성부를 **재선별 없이 import**해 판정만 낸다(재현성 핵심 원칙).

| 파일 | 대상 |
|---|---|
| `verify_pair.py` | 짝꿍(12번) |
| `verify_sangtta.py` | 상따(06번 5차) — **기각·재시도 금지** |
| `verify_dip.py` | 낙주(10번) — **기각·재시도 금지** |
| `verify_breakout.py` | 돌파(09번) |
| `verify_newlisting.py` | 신규상장(15번) |
| `verify_closebet2.py` | 종가베팅 v2(14번) |
| `verify_dart_catalyst.py` | DART 재료 결합(21번) |
| `verify_trajectory.py` | 장중 궤적(16번) |
| `verify_openbet.py` | 오픈베팅 |
| `verify_shortsell.py` | 공매도 추종 숏 |
| `verify_exits.py` | 여러 기법 공용 청산 방식 비교 |

## D. 진단·최적화 — 일회성 연구 도구

| 파일 | 용도 |
|---|---|
| `limitup_lock_study.py` | 굳힐 대장 판별(47번) — **기각** |
| `limitup_timing.py` | 상한가 진입 타이밍 진단(06번 부속) |
| `manju_optimize.py` | 분봉 1년 재최적화(문서42 선결조건) |
| `manju_week_review.py` | 주간 복기(돈 기준) |
| `pair_sweep.py` | 감지기 파라미터 스윕(틱 1회 로드) |

## E. 틱·호가 sanity — 장중 미시구조, 반복 가능한 프레임

`tick_sanity.py`가 원형(28번 사전등록: 체결강도·매수비율). 나머지는 신호
정의만 바꿔 같은 절차(트리거 vs 비트리거, 부트CI, 분위 단조성, 비용후 net,
최소표본)를 반복한다.

| 파일 | 신호 | 문서 |
|---|---|---|
| `tick_sanity.py` | 체결강도+매수비율(하루치) | [28](../docs/research/manju/28-tick-sanity-preregistration.md) |
| `tick_sanity_confirm.py` | 〃(여러 날 풀링 확정) | 28 |
| `quote_sanity.py` | 호가 임밸런스(탐색적) | — |
| `trail_stop_sanity.py` | 트레일링 스탑 청산(탐색적) | — |
| `accel_sanity.py` | 체결 가속도(대장 자신) | [56](../docs/research/manju/56-accel-filter-preregistration.md) — **방향 반대, 확정** |
| `accel_sanity_minute.py` | 〃(분봉·국면별 재검증) | [57](../docs/research/manju/57-accel-filter-minute-generalization.md) — **확정** |
| `runway_sanity.py` | 라운드피겨·전고점 거리 | [58](../docs/research/manju/58-runway-filter-preregistration.md) — **무신호/방향 반대, 확정** |
| `round_exhaustion_sanity.py` | 라운드피겨 매도 소진(호가) | [70](../docs/research/manju/70-round-figure-exhaustion-preregistration.md) — **방향 맞음·비용 미달** |

## F. 헬퍼

`_envload.py` — `../quant-airflow/.env` 환경변수 주입(`pair_sweep.py`가 import).
