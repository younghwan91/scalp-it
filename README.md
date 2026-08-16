# swing-it

한국 주식(KOSPI·KOSDAQ) **2~8주 스윙 후보 리서치 툴**.

매일 소수의 핫한 기사를 재료로 삼아, 그 재료가 몇 주 갈 추세인지 판정하고 근거와 함께
후보 종목을 제시한다. 최종 매매 판단은 사람이 한다.

> 기사에서 재료를 찾고 → LLM 이 구조화 피처로 변환하고 → 수급이 독립 확증하고 →
> ML 이 D+20~40 초과수익으로 학습하고 → kr-quant 게이트가 심판하고 →
> 통과한 것만 후보 리스트로 나온다.

## 상태

| 단계 | 상태 |
|---|---|
| 설계 스펙 | ✅ [스펙](docs/superpowers/specs/2026-08-17-swing-it-design.md) |
| 계획 1 — 수급 베이스라인 (①층) | ✅ 구현·실행 완료 · **판정 FAIL** |
| 계획 2 — 뉴스 인제스트 | ◐ [계획](docs/superpowers/plans/2026-08-17-news-ingest.md) · 빅카인즈 파일 대기 |
| 계획 3 — LLM 피처 추출 | ◐ [계획](docs/superpowers/plans/2026-08-17-llm-features.md) · API 키 대기 |
| 계획 4 — 층별 비교 + 근거 카드 | ✅ [계획](docs/superpowers/plans/2026-08-17-layers-and-report.md) |

미구현으로 남은 것은 외부 입력이 있어야만 가능한 부분뿐이다.

**베이스라인 판정: FAIL** — 종합 판정은 FAIL 이지만 "수급이 무력하다"는 뜻이 아니다.
4개 폴드 중 2개에서만 랜덤 널을 이겨 재현되지 않는다. **레짐 의존적인 신호가 있고
아직 재현되지 않는다**가 정확한 요약이다.
자세히 → [`docs/research/2026-08-17-baseline-verdict.md`](docs/research/2026-08-17-baseline-verdict.md)

첫 측정은 잘못 쟀다 — `top_n=15` 인데 하루 후보가 중앙값 8건이라 선택이 아예
일어나지 않았다. 사전등록 후 `top_n=2` 로 다시 쟀고, 그 경위를 문서에 남겼다.
→ [`topn-preregistration.md`](docs/research/2026-08-17-topn-preregistration.md)

다음에 할 일 → [`docs/HANDOFF.md`](docs/HANDOFF.md)

## 실행

```bash
uv sync --extra dev
uv run pytest                      # 294 passed

export $(grep -v '^#' ../kr-quant/.env | xargs)   # KR_QUANT_DB
uv run swing-baseline              # ①층 판정 → results/baseline.json

# LLM 배선 확인 (API 키 불필요)
uv run swing-score --provider fake --news-db <경로> --cache-db <경로> \
  --start 2026-08-01 --end 2026-08-14

uv run swing-layers                # ①②③ 층 비교 (계획 2·3 산출물 필요)
uv run swing-today                 # 오늘의 후보 + 근거 카드 (동일)
```

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
