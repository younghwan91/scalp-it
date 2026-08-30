# 만쥬 실행 지시서 (50번) — 시스템 프롬프트 형식

> 2026-08-30, 사용자가 그대로 옮겨준 것. **출처 정리 문서다** — 원문을 (비공개 회신 — 공개본에서 뺐다) 와 같은 규칙).
>
> [`49`](49-manju-manual-integrated.md) 의 통합 매뉴얼·A1~A3 을 **실행 가능한 규칙으로
> 조여 놓은 판**이다. 수치가 여러 곳에서 더 엄격해졌다(아래 "49 와 달라진 수치").

## 원문

```
[1. GATE 0: MARKET REGIME & LIQUIDITY FILTER]
- KOSDAQ Daily Projected Volume Check:
  * If KOSDAQ Total Turnover < 7 Trillion KRW:
    -> TRIGGER: DEFENSIVE MODE. Max capital per trade capped at 10% of normal size.
  * If KOSDAQ Total Turnover >= 8 Trillion KRW AND Index is consolidating:
    -> TRIGGER: ACTIVE SYMPATHY MODE. Full operational execution enabled.

[2. DUAL ENTRY PIPELINE (STOCK A: LEADER / STOCK B: SYMPATHY FOLLOWER)]
* Universal Filter for Target Pair:
  - Stock A must be within Top 3 of Real-Time Search Ranking (실시간 검색순위 1~3위).
  - Clear upward runway: Target resistance (e.g., 2nd VI or Upper Limit) must be open.
  - Orderbook Dynamics: Transaction velocity & acceleration (체결 가속도) spike confirmed.

* PIPELINE 1: UPPER LIMIT PROXIMITY (High-Conviction Default)
  - Condition: Stock A enters within 3 ticks of the Upper Limit (+30%) with massive buying delta.
  - Action: Execute immediate MARKET/LIMIT BUY on Stock B.

* PIPELINE 2: STATIC VI SURGE (Spread Arbitrage)
  - Condition: Stock A triggers Static VI (+10% / +20%) with explosive volume acceleration.
  - Action: Execute immediate BUY on Stock B within the 2-minute VI auction window of Stock A.

[3. STATE-DEPENDENT LIQUIDATION & EXIT ENGINE (STOCK B)]
* PHASE 1: STOCK A APPROACHING CEILING (Pre-Lock Phase)
  - Trigger 1 (Micro Stalling): Stock A's buy volume velocity decelerates, or ticks down by 1 tick.
    -> Action: IMMEDIATELY SELL 100% of Stock B at MARKET.
  - Trigger 2 (Execution Failure): Stock B fails to move upward within 1-2 ticks post-entry.
    -> Action: EXECUTE SCRATCH STOP (매수가 즉시 손절/본절 탈출).

* PHASE 2: STOCK A LOCKED AT UPPER LIMIT (Ceiling-Locked Phase)
  - Trigger: Stock A upper-limit bid volume decreases rapidly, locks break (풀림), or VI gap-down occurs.
    -> Action: IMMEDIATELY DUMP 100% of Stock B at MARKET. No limit orders permitted.

* PROFIT HARVESTING:
  - If target momentum exhausts or market slows: Sell 40% in tranches, or 100% on reversal.

[4. POSITION SIZING & PYRAMIDING RULES]
- Martingale / Averaging Down (물타기): ABSOLUTELY FORBIDDEN.
- Pyramiding (불타기):
  * Only permitted when position is in confirmed profit >= +3.0% ~ +4.0%.
  * Sizing limit: Maximum 2x initial size.
- Liquidity-Based Sizing:
  * Thick Orderbook (e.g., Large Group Stocks): Normal allocation.
  * Thin Orderbook (e.g., Small Theme Followers): Cap size to prevent slippage (Max 10-20% of normal size).

[5. HARDWARE RISK CONTROLS & TRADING HALT (CIRCUIT BREAKERS)]
* TRAILING PROFIT DRAWDOWN (50% Rule):
  - Track intraday peak net profit (P_peak).
  - If (P_current <= P_peak * 0.50) after achieving significant P_peak:
    -> ACTION: HARD STOP. Terminate all open positions and shutdown trading for the day.

* MORNING TIME-WINDOW CAP:
  - If market shows low clarity or high chop: Halt all new entries post 11:00 AM KST.

* EMERGENCY ISOLATION (Drawdown Lockdown):
  - On 2 consecutive losses or severe tilt: Freeze Main Account HTS.
  - Divert subsequent operations strictly to 3M KRW Buffer Sub-Account.
  - Zero Deposit Rule (추가 입금 금지): Account deposits strictly disabled.
```

## 49 와 달라진 수치 — 전부 **더 엄격한 쪽**이다

| 항목 | 49(A1~A3) | 50(지시서) |
|---|---|---|
| Phase 1 하락 문턱 | "1~2호가라도 밀리면" | **1호가** |
| 연속 손절 킬 | 46 §10 "2~3회" | **2회** |
| 정점 이익 반납 | 46 §10 "1/3~1/5 축소"(킬 아님) | **50% 반납 시 하드스톱** |
| Gate 0 | "7~8조 이하" | **7조 미만** |
| 오후 차단 | "오후 추격 배제"(§1-5) | **11:00 이후**(단, "low clarity" 조건부) |

## 새로 나온 규칙 — 49 에 없던 것

- **실시간 검색순위 1~3위** 를 대장 필터로 요구(§2 Universal Filter)
- **상단 저항 공백**(2차 VI 또는 상한가까지 길이 열려 있을 것)
- **체결 가속도 spike** 확인
- **Phase 1 Trigger 2 — 진입 후 B 가 1~2호가 안에 위로 안 가면 본절 탈출**
- 익절 수확: 모멘텀 소진 시 **40% 분할**, 반전 시 100%
- 유동성 기반 사이징: 얇은 호가 종목은 **평소의 10~20%**
- 비상 격리: 메인 계좌 동결 → **300만원 버퍼 서브계좌**로만, **추가 입금 금지**
