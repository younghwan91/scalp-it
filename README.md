<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="scalp-it" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

# scalp-it

**한국 주식(KOSPI·KOSDAQ) 단타 전략을 사전등록으로 검증하고 장중 틱·호가를 수집하는 도구**

현재는 2026-09-16부로 실주문을 중단하고 dry-run 감지기와 장중 수집 크론만 운영합니다.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/younghwan91/scalp-it/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/scalp-it/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[가이드](docs/guide.md) · [판정 기록](docs/TRADING_CRITERIA.md) · [런북](docs/PAIR_TRADING_RUNBOOK.md) · [English](README.en.md)

</div>

> [!NOTE]
> 레포 이름과 달리 현재는 스윙이 아니라 **공개된 개인 트레이더의 일중 단타 시스템 복제**를 다룹니다.
> 2026년 10월 사전등록 연구 프로그램은 **전진 판정 전부 기각으로 종결**했습니다(소매 왕복비용 미달성).
> 현재 판정과 결정은 [`docs/TRADING_CRITERIA.md`](docs/TRADING_CRITERIA.md)에서 확인합니다.

## 바로 해 보기

키움 계좌·DB·API 키 없이 돌아갑니다. 커밋된 사전등록 판정 잠금 파일(`research/runs/*/FINAL.lock`)에서 전진 판정 10건을 읽어 표로 보여 줍니다(읽기 전용).

```bash
git clone https://github.com/younghwan91/scalp-it && cd scalp-it
uv sync --extra dev
uv run python scripts/forward_verdicts.py   # 전진 최종 판정 모아보기
uv run pytest -q                            # 단위 테스트
```

## 핵심 결과

부정적 결과를 그대로 공개합니다. 수치는 모두 레포의 문서에 근거가 있습니다.

- **사전등록 전진 판정 10건 전부 기각**: 한 번만 측정한 평균이 −5.9 ~ −44.7bp, CI 하한은 전부 음수입니다. `uv run python scripts/forward_verdicts.py`, [판정 기록](docs/TRADING_CRITERIA.md).
- **비용 벽**: 주식 왕복 실비용 약 41.5bp(거래세·농특세 20 + 수수료 3 + 스프레드 14~31). 매도 시점을 사후에 완벽히 고른 오라클로도 시장가 10초 스캘핑은 −35.9bp입니다. [한 장 요약](docs/research/manju/99-where-we-stand.md).
- **유니버스 진단**: 일봉 10년 550만 관측에서 거래대금 상위 300종목의 일중 수익은 −28.0bp로, 이익은 야간에만 있고 일중은 잃는 구간입니다. 같은 문서.
- **야간 선택(autoresearch) 기각**: 거래당 총 +45.3bp, 순 −18.0bp. [VERDICT](research/autoresearch/VERDICT.md).

## 주요 기능

- **단일 세션 수집 + 실시간 감지**: 키움 WebSocket(`0B` 체결·`0D` 호가) 프레임을 TimescaleDB에 저장하면서 동시에 `PairDetector`로 대장-후속 짝꿍 신호를 지연 없이 판정합니다.
- **엄격한 가드레일과 dry-run 기본값**: `RiskGuard`(일일 손실한도·연속 손절·킬스위치)와 `krx-quant-core` 공용 주문가드를 통과해야 실행 경로에 도달합니다. `--confirm-real`이 없으면 주문 0건(dry-run)으로 동작합니다.
- **사전등록 & 단일 검증 규율**: 기각된 가설을 살리려고 파라미터를 사후 조정하지 않습니다. 모든 판정과 결정은 [`docs/TRADING_CRITERIA.md`](docs/TRADING_CRITERIA.md)에 누적 기록합니다.

## 실행과 환경 설정

실시간 수집과 DB 스크립트를 돌리려면 아래 환경이 필요합니다.

- **키움증권 계좌 및 API 키**: App key/secret을 발급받고 형제 레포 [kiwoom-client](https://github.com/younghwan91/kiwoom-client)로 접근합니다.
- **형제 레포 `quant-airflow`**: 같은 부모 디렉터리에 클론합니다. `.env`에서 `KR_QUANT_DB`, `KIWOOM_APP_KEY`, `KIWOOM_APP_SECRET`, `DART_API_KEY`를 읽어 옵니다(이 레포에는 값을 적지 않습니다).
- **실주문 차단 목록**: `--confirm-real` 경로는 계좌 보유 종목 코드(`SCALP_HELD_STOCK_CODES`, 쉼표 구분)가 비어 있으면 주문을 거부합니다(fail-closed).

```bash
export $(grep -v '^#' ../quant-airflow/.env | xargs)                   # KR_QUANT_DB + KIWOOM_APP_*
```

**장중 실시간** — 아래 명령은 실 API를 호출하고 라이브 DB에 씁니다. cron이 이미 실행하므로 수동 실행은 App key 세션 충돌이 없을 때만 사용합니다.

```bash
uv run --extra pg scalp-pair-detect --live --live-order --duration 09:00-15:20   # 짝꿍 감지기 (--confirm-real 없으면 dry-run)
uv run scalp-tick-collect --auto 45 --duration 08:50-15:40                        # 틱·호가 수집 데몬 (0B/0D)
uv run scalp-tick-health --minutes 20                                             # 수집 헬스체크 (읽기 전용)
```

**배치 · 검증**

```bash
uv run scalp-theme-snapshot --sleep 0.3                      # 키움 테마 소속 오늘 스냅샷
uv run scalp-intraday --codes 005930 --pages 5               # 분봉 수집
uv run --extra pg python scripts/tick_sanity.py --date today # 틱 sanity 확인
python3 scripts/pair_daily_summary.py                        # 짝꿍 dry-run 일일 성과 집계
```

## 주의사항

- **실주문 중단 상태**: 2026-09-16 밤부터 실주문을 중단했습니다. 2026-08-27~09-16 기간에만 `--confirm-real`로 1주 실주문을 냈으며, 현재는 주문 0건입니다.
- **App key 1세션 제약**: 키움 App key 1개당 WebSocket 세션 1개만 연결할 수 있습니다. 짝꿍 감지기가 수집을 겸하므로 `scalp-tick-collect`를 동시에 띄우면 서로 연결이 끊깁니다.
- **DB 의존성과 데이터 보존**: 틱·호가는 소급 수집이 불가능합니다. 장중 TimescaleDB PRIMARY(`KR_QUANT_DB`) 장애 시 당일 데이터는 영구 손실됩니다([상세 내용](docs/guide.md#db-의존성-주의)).
- **규율과 비용**: 사전등록 후 한 번만 측정하고 결과를 그대로 기록합니다. 평균만 보지 않고 중앙값·승률·손익비·분위·연도별 분포를 함께 확인하며, 2026년 왕복 비용(34bp)을 먼저 뺍니다. 데이터 파일은 커밋하지 않습니다.

## 문서

| 문서 | 설명 |
|---|---|
| [`docs/guide.md`](docs/guide.md) | 아키텍처, 평일 cron 운영, DB 의존성, `krx-quant-core` 공통 코어 매핑 |
| [`docs/TRADING_CRITERIA.md`](docs/TRADING_CRITERIA.md) | **여기부터** — 날짜별 결정과 현재 판정 상태 |
| [`docs/PROMOTION_CHECKLIST.md`](docs/PROMOTION_CHECKLIST.md) | 실주문 재개 전 통과해야 하는 11개 게이트 |
| [`docs/PAIR_TRADING_RUNBOOK.md`](docs/PAIR_TRADING_RUNBOOK.md) | 짝꿍 실시간 매매 런북 — dry-run → 실발주 순서, 안전장치 |
| [`docs/process/daily-ops.html`](docs/process/daily-ops.html) | 평일 크론 전체 시간대·역할 프로세스 그래프 |
| `docs/research/manju/` | 원본 시스템 리서치, 사전등록, 판정 결과 |
| `docs/archive/` | v1·v2 시기 기록 및 실패 사례 5가지 |

## 같은 저자의 도구

| 레포 | 역할 |
|---|---|
| [kiwoom-client](https://github.com/younghwan91/kiwoom-client) | 시세·수급·분봉·테마·주문 (182 REST + WebSocket) |
| [krx-quant-core](https://github.com/younghwan91/krx-quant-core) | 국장 공통 코어 — 호가단위·상한가·세션·주문 가드·체결 판정·검증 통계 ([매핑 표](docs/guide.md#공통-코어-의존성-krx-quant-core)) |
| [fin-checkup](https://github.com/younghwan91/fin-checkup) | 상장사 재무제표를 신호등으로 읽어 주는 무료 도구 — 지표 17개의 계산식·경계값 공개, 측정값만 |
| [daytrade-it](https://github.com/younghwan91/daytrade-it) | 뉴스 데이트레이딩 전략에 실제 우위가 있는지 측정한 독립 리서치 기록 |
| [swing-it](https://github.com/younghwan91/swing-it) | 검증 프레임워크 (walk-forward·음성대조) 및 이관된 저빈도 팩터 |
| [quant-airflow](https://github.com/younghwan91/quant-airflow) | 일일 데이터 적재 및 TimescaleDB PRIMARY/리플리카 구성 ([DB 주의사항](docs/guide.md#db-의존성-주의)) |

## 라이선스

[MIT](LICENSE)
