# scalp-it

> ⚠️ **레포 이름은 역사의 잔재다.** 지금은 스윙이 아니라
> **트레이더 만쥬의 일중 단타 시스템 복제**를 만든다.
> **먼저 읽을 것 → [`docs/STATUS.md`](docs/STATUS.md)**

한국 주식(KOSPI·KOSDAQ) 단타 전략의 **검증 프레임워크이자 실행 도구**.

핵심 규율은 하나다 — **사전등록하고, 한 번만 재고, 결과를 그대로 기록한다.**
기각된 것을 살리려고 파라미터를 바꾸지 않는다.

## 현재 판정: 8개 시험, 1개 채택

**종가베팅** — net34 **+2.73%** · 승률 61% · 손익비 2.62 · 11년 음수 해 **0개**

```
당일 거래대금 50억+ · 거래대금 20일평균 대비 3배+ · 종가위치 0.8+
→ 거래대금 배수 상위 2건 → D 종가 매수 → D+1 시가 매도
```

기각: 상따(일봉) · 시장레짐 필터 · 수급 조건 · 52주 신고가 · 장중 ±2% 관리 ·
D+1 종가 청산 · 공시 피처층

## 실행

```bash
uv sync --extra dev
uv run pytest                       # 500+ passed

export $(grep -v '^#' ../kr-quant/.env | xargs)          # KR_QUANT_DB
export $(grep -v '^#' ../quant-airflow/.env | grep KIWOOM | xargs)

uv run scalp-closebet --start 2016-09-09 --end 2026-08-14 \
  --rule tv_ratio:3.0: --rule close_pos:0.8: --exit open   # 종가베팅 백테스트
uv run scalp-cbet-watch                                    # 오늘 후보 기록
uv run scalp-cbet-watch --score                            # 전진 기록 채점
uv run scalp-intraday --codes 005930 --pages 5             # 분봉 수집
```

## 문서

| | |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | **여기부터** — 현재 상태·판정 현황·다음 할 일 |
| `docs/research/manju/` | 만쥬·스승 리서치, 사전등록, 판정 결과 |
| `docs/research/2026-08-17-*` | v1·v2 스윙 시기 기록 (역사) |
| `docs/superpowers/` | v1·v2 시기 스펙·계획 (역사) |

## 관련 레포

| 레포 | 역할 |
|---|---|
| [kiwoom-rest-api](https://github.com/younghwan91/kiwoom-rest-api) | 시세·수급·분봉·테마·주문 (182 REST + WebSocket) |
| [kr-quant](https://github.com/younghwan91/kr-quant) | 검증 프레임워크 (walk-forward·음성대조) |
| quant-airflow | 일일 데이터 적재 → TimescaleDB |

## 원칙

- **사전등록하고 한 번만 잰다.** 기각된 것을 살리려 파라미터를 바꾸지 않는다
- **평균만 보지 않는다.** 중앙값·승률·손익비·분위·연도별을 함께 본다
- **비용을 먼저 뺀다.** 2026년 왕복 34bp. 단타는 비용이 지배한다
- **데이터는 커밋하지 않는다**
