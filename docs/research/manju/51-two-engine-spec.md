# 2엔진 시스템 명세 (51번) — 모듈 A(만쥬) · 모듈 B(타 트레이더 B)

> 2026-08-30, 사용자가 그대로 옮겨준 것. **출처 정리 문서다.**
>
> **이 문서가 49·50 의 출처 혼선을 푼다.** 49(통합 매뉴얼)·50(실행 지시서)은 두 사람의
> 기법이 섞인 판이었고, 어디까지가 만쥬인지 우리가 44·46 대조로 추정만 했다. 이 명세는
> **모듈로 갈라서** 알려준다.

## 요약 — 누구 것이 무엇인가

| 모듈 | 사람 | 영역 |
|---|---|---|
| **A** | 만쥬 | 시황·국면, 짝꿍매매, NXT 오버나이트, 국면별 비중 |
| **B** | **타 트레이더 B** | 미시구조, 스캘핑, 트레일링 스톱, 계좌 규율 |

**우리가 2026-08-30 에 배선한 것 중 둘은 모듈 B 였다** — `본절 탈출`(B-2)과
`정점 이익 50% 반납 하드스톱`(B-3). 종전 커밋·주석에 "만쥬 지시서"로 적었던 것을
이 문서로 바로잡는다.

---

## 원문

```
MODULE A: MANJU ENGINE (Macro, Pair Trading, NXT Overnight & Regime Sizing)

[A-1. MARKET REGIME & CAPITAL ALLOCATION GATE]
- Regime Classification:
  * Trending Bull (Large Cap Bias): Focus capital on KOSPI large caps (e.g., Semi).
  * Consolidating / Range-bound (박스권/수급분산): Shift execution to Pair/Theme plays.
- Liquidity Filter (Gate 0):
  * IF KOSDAQ Projected Volume < 7~8T KRW: Cap position sizing to <= 10% (Defensive).
  * IF KOSDAQ Volume >= 8T KRW AND Index Ranging: MAX Sympathy Mode Enabled.

[A-2. PAIR TRADING & GLOBAL SYMPATHY PIPELINE]
- Domestic Leader-Follower Pair (대장주 ↔ 2등주):
  * Trigger: Leader (Stock A) breaks into VI or hits within 3 ticks of Upper Limit (+30%).
  * Execution: Market BUY Follower (Stock B) utilizing the execution delay gap.
  * Thin Orderbook Restriction: If Stock B has thin depth, force max size <= 20% normal.
- Global Sympathy (NXT / Pre-market):
  * Trigger: US tech catalyst/earnings beat (e.g., AMD, Lumentum) + US Futures breakout.
  * Execution: Accumulate domestic corresponding sector in NXT (15:30~20:00) or pre-market.

[A-3. NXT OVERNIGHT / CLOSING BET ENGINE (16:00~20:00)]
- 3 Core Pillars: [Market/Index Structure] + [Event/News Continuity] + [Position Sizing].
- Regular Close Passing: Skip 15:30 regular market close due to algorithm volatility amplification.
- Alternative Exchange Entry:
  * Wait with 100% Cash (무포 관망) until 19:00 KST if themes are fragmented.
  * Scale-in on pullbacks while monitoring US Nasdaq Futures real-time direction.
  * Overweight confirm: Add final sizing only when US futures break upward.
- Morning Exit: Sell in 5~10 tranches during pre-market/regular open on gap-ups.

[A-4. DRAWDOWN & PYRAMIDING EXECUTION]
- Averaging Down (물타기): STRICTLY BANNED.
- Pyramiding (불타기): Add up to 2x position size ONLY when unrealized PnL >= +3.0% ~ +4.0%.
- Severe Drawdown Rule:
  * On consecutive heavy loss / tilt: Instantly cut default sizing by 90% (1/10 룰).
  * Operate in pure cash (No Margin / No Loan) to stabilize emotional equilibrium.

MODULE B: 타 트레이더 B ENGINE (Microstructure, Scalping & Trailing Stop)

[B-1. MICROSTRUCTURE & INTRADAY ENTRY FILTER]
- Real-Time Search Ranking Filter: Target must rank within Top 3 (실시간 검색 Top 3).
- Order Flow Dynamics (Velocity & Acceleration):
  * Ignore static bid/ask layout ratios.
  * Trigger entry ONLY when execution velocity (체결 가속도) + chunk buy orders accelerate.
- Target Runway Verification:
  * Enter ONLY if there is clear upside runway (e.g., 2nd VI, Round-Number, Pre-high).
- Round-Figure Inflection:
  * Exploit order flow at round prices (e.g., 500, 1000, 10000); enter when sell bursts exhaust.

[B-2. MICRO EXIT & SCRATCH STOP RULES]
- Immediate Acceleration Rule (즉각 손절):
  * IF stock does not surge within 1~2 ticks immediately post-entry:
    -> ACTION: Trigger SCRATCH STOP (Cut at entry price or -1 tick immediately).
- Wave Exhaustion Exit:
  * Sell 40% when upward momentum stalls; dump remaining 60% on first lower tick.

[B-3. INTRADAY PROFIT TRAILING STOP (50% RULE)]
- Dynamic Intraday Peak PnL Tracking:
  * System continuously records Day Peak Net Profit (P_peak).
  * HARD SHUTDOWN TRIGGER:
    IF (P_current <= P_peak * 0.50) after establishing substantial profit:
      1. Liquidate 100% of all open positions at MARKET immediately.
      2. Lock HTS software and terminate all trading processes for the day.

[B-4. HARDWARE ISOLATION & CAPITAL DISCIPLINE]
- Sub-Account Routing (3M KRW Buffer Account):
  * IF trading tilt occurs or loss limit hit on Main Account:
    -> Lock Main Account. Reroute all further trading exclusively to 3M KRW Sub-Account.
- Asymmetric Capital Policy:
  * Additional Inbound Deposits: PERMANENTLY DISABLED (입금 0원 원칙).
  * Mandatory Outbound Withdrawals: Regularly sweep profits above baseline equity.
- Morning Window Termination:
  * If market lacks definitive themes, halt all new scans and shut down by 11:00 AM KST.
```

## 배선 현황 (2026-08-31 기준)

| 항목 | 모듈 | 상태 |
|---|---|---|
| Gate 0 → 비중 1/10 | A-1 | ✅ `--scout-gate-jo 8` |
| 짝꿍 진입 — 상한가 3호가 | A-2 | ✅ `--limitup-near-ticks 3` |
| 짝꿍 진입 — **VI 경로** | A-2 | ❌ 별도 파이프라인(49 A3) |
| 얇은 호가 ≤20% | A-2 | ✅ `--thin-book-bp 10 --thin-book-size 0.2` |
| 글로벌 동조화 · NXT | A-2·A-3 | ❌ NXT 데이터 없음 |
| 물타기 금지 | A-4 | ✅ 분할매수 자체가 없다 |
| 불타기 +3~4% 2배 | A-4 | ⏸ 함수만(`pyramid_multiplier`). 1주 매매라 회계·실주문이 갈린다 |
| 1/10 룰 | A-4 | ✅ Gate 0 와 같은 경로 |
| 실시간 검색 Top 3 | B-1 | ❌ 우리가 안 받는 API |
| 체결 가속도 | B-1 | ⏸ `accel_min_ratio` 노브 있음, 수치 없어 꺼둠 |
| 상단 런웨이 검증 | B-1 | ❌ |
| 라운드 피겨 변곡 | B-1 | ❌ |
| **본절 탈출** | **B-2** | ✅ `--scratch-ticks 1 --scratch-secs 30` |
| 파동 소진 청산 40/60 | B-2 | ❌ 1주는 못 쪼갠다 |
| **정점 이익 50% 반납 하드스톱** | **B-3** | ✅ `--profit-giveback 0.5` |
| 서브계좌 격리 · 입금 0원 | B-4 | ❌ 계좌 운용이라 코드 밖 |
| 11:00 종료 | B-4 | ✅ 더 엄격하게 10:00 (`--no-entry-after 1000`) |
