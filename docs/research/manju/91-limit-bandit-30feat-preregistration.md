# 2단 지정가 밴딧 · 지정가용 30피처 — 사전등록 (91번)

> 2026-09-16 새벽 작성, **실행 전 커밋**. 커밋 뒤 수정하지 않는다(결과는 `91-limit-bandit-30feat-results.md`).

## 0. 왜 재나

[89번](89-two-head-limit-bandit-results.md) 2단 지정가 밴딧은 84번 13피처(시장가 기준으로 뽑힌)만 썼다.
[90번 §4](90-big-wave-exit-exploration.md)에서 체결된 지정가 거래의 단일 피처 10분위는 비용을 못 넘었지만, **조합**은 다를 수 있다.

## 1. 피처 — 고정

90번 §4 선별(체결 거래 50만 표본, 상·하위 10분위 5블록 평균이 좋은 순, 스피어만 |ρ| ≤ 0.9 중복 제거) 상위 30:

`["n_tr5", "buyval3", "n_tr1", "burst_secs10", "buyval1", "sv20", "ntr60", "bv30", "buyval10", "sv60", "sweep_ticks3", "sellval5", "sellval3", "maxbuy5", "spread_bp", "dayret_rank", "sweep10", "sellval1", "sweep_ticks1", "dstr3", "imb3", "dstr5", "dS2", "ofiw20", "ofiw30", "dist_lo300", "ofiw60", "log_cumval", "ofiw1", "ofiw2"]`

**주의**: 이 30개는 17일 체결 결과를 보고 골랐다 — 17일 선행 평가는 낙관 편향이 크고 판정에 쓰지 않는다.

## 2. 나머지 — 89번과 동일

행동·관찰·체결 모형·청산·몸통(입력 30 → 64 ReLU)·두 머리·톰슨·학습 일정·표준화(첫 3거래일)·종목당 1주문·워밍업 3일. seed 91.

## 3. 채택 게이트 — 전진, AND

83번 §6 과 같다(89번 대조군). N ≈ 2,093(90 까지) + 1. 판정은 2026-09-16 부터 전진, 온라인 학습 계속. research/runs/bandit91 잠금.
