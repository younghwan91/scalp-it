# swing-it

한국 주식(KOSPI·KOSDAQ) **2~8주 스윙 후보 리서치 툴**.

매일 소수의 핫한 기사를 재료로 삼아, 그 재료가 몇 주 갈 추세인지 판정하고 근거와 함께
후보 종목을 제시한다. 최종 매매 판단은 사람이 한다.

> 기사에서 재료를 찾고 → LLM 이 구조화 피처로 변환하고 → 수급이 독립 확증하고 →
> ML 이 D+20~40 초과수익으로 학습하고 → kr-quant 게이트가 심판하고 →
> 통과한 것만 후보 리스트로 나온다.

**상태:** 설계 단계. 구현 전.

- 설계 스펙: [`docs/superpowers/specs/2026-08-17-swing-it-design.md`](docs/superpowers/specs/2026-08-17-swing-it-design.md)

## 관련 레포

| 레포 | 역할 |
|---|---|
| [kr-quant](https://github.com/younghwan91/kr-quant) | 검증 프레임워크 (walk-forward · DSR · 음성대조) |
| [kiwoom-rest-api](https://github.com/younghwan91/kiwoom-rest-api) | 시세·수급·주문 |
| [krx-news-rest-api](https://github.com/younghwan91/krx-news-rest-api) | 뉴스·공시 수집 |
| quant-airflow | 일일 데이터 적재 |

## 원칙

- 검증은 kr-quant 게이트에서만 한다. 이 레포는 자기 검증을 하지 않는다.
- 음성대조를 못 이긴 시그널은 운영에 올리지 않는다.
- **데이터는 커밋하지 않는다** (기사 저작권 · 빅카인즈 재배포 금지).
