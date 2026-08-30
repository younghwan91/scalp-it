# 3인 엔진 명세 v9 (52번) — 만쥬 · 침착해 · 타 트레이더 B

> 2026-08-30, 사용자가 그대로 옮겨준 것. **출처 정리 문서다.**
> [`51`](51-two-engine-spec.md) 의 후속판이고 **모듈 A 가 통째로 바뀌었다.**

## 51 → 52 에서 달라진 것

| | 51 (2엔진) | 52 (v9) |
|---|---|---|
| 모듈 A | 만쥬 — 시황·**짝꿍매매**·NXT·국면별 비중 | 만쥬 **+ 침착해** — 섹터 대장 선정·**수급 턴어라운드**·과매도 종배·**레짐 스위칭** |
| 모듈 B | 타 트레이더 B — 미시구조·스캘핑 | 같음(런웨이 항목이 B-1 로 이동) |

**모듈 A 에서 짝꿍 진입 트리거(상한가 3호가·VI)가 사라졌다.** 51 의 A-2 였던 그 규칙이
v9 에는 없다. 대신 A-1 이 **섹터 대장 vs 2등주의 상대강도**를 말한다 — 같은 "짝꿍"이라는
말을 쓰지만 **시간축이 다르다**(장중 초단타 → 일봉 스윙).

⚠️ **우리 시스템 전체는 51 의 A-2(장중 짝꿍)로 만들어져 있다.** v9 를 그대로 따르면
지금 배선의 상당 부분이 대상 없는 코드가 된다. 어느 쪽으로 갈지는 사용자 결정 사항이다.

---

## 원문

```
MODULE A: MANJU & CHIMCHAK-HAE ENGINE (Macro, Sector/Pair Leader, Flow & Swing)

[A-1. SECTOR LEADER SELECTION & PAIR ABSORPTION (조선/주도 섹터 짝꿍 분석)]
- Dominant Sector Identification:
  * Rank sectors by Macro Tailwind (e.g., Trump tariff exemptions, interest rate drops)
    + Multi-Stock 52-week Highs [00:00:00], [00:20:54].
- Leader vs. Sub-Leader Relative Strength (예: 한화오션 vs 삼성중공업):
  * On market crash/pullback (지수 급락일):
    - Target the stock that resists downflow and breaks the box ceiling
      (한화오션: 박스권 상단 돌파) rather than the heavy pullback stock
      (삼성중공업: 매물 누적) [00:25:27], [00:26:04].
    - Heavy pullback stock contains overhead trapped volume; leader with ceiling break
      clears supply [00:26:13].
- Institutional/Foreign Order Flow Reversal (수급 턴어라운드 타점):
  * When foreign/institutional sellers dump 2 continuous big negative candles (2음봉)
    after a massive inflow, do NOT buy blindly.
  * Trigger: Enter on the day when Foreigners/Institutions re-enter with high volume,
    re-absorbing 50%+ of the 1st negative candle [00:00:35], [00:21:45].
  * Intraday Confirmation: Wait until 10:00~11:00 AM KST; if Program/Institutional
    provisional net buys expand and absorb previous upper shadows, execute Pyramiding
    [00:22:16], [00:22:43].

[A-2. MARKET EXCESSIVE PRICING & CLOSING BET GATE (선반영 과매도 역발상 종배)]
- Over-Correction Gate (선반영 과매도 필터):
  * IF US Market drops 2.5% BUT KOSPI/KOSDAQ plunges 3.0%+ due to foreign futures
    dumping (>1.5T KRW):
    -> Market is deemed "Over-Corrected" (과도한 선반영) [00:25:37], [00:28:07].
  * Action: Buy leading sector stocks (조선, 바이오 등 주도주) at close with high
    conviction [00:26:21], [00:28:26].
- Gap-Up Morning Profit Harvesting:
  * Morning open: Trim 30~50% immediately if stock opens above prior box resistance
    [00:26:58].
  * Remainder: Trail exit at major round-number resistance (e.g., 80,000 KRW) [00:27:06].

[A-3. REGIME SWITCHING & HORIZON TUNING]
- Large-Cap Flow Market (조 단위 거래대금 장세):
  * BANNED: Low-cap junk stocks (잡주/테마주) [00:02:21], [00:19:03].
  * Target: Mega-cap trend leaders with solid volume (>1T KRW). Carry overnight /
    1-2 day short swing with zero intraday panic [00:02:38], [00:09:52].
- Sudden News Disruption & Stop-Loss (돌발 악재 칼손절):
  * When unexpected negative research reports / short-selling articles hit leading stocks:
    -> SELL 100% IMMEDIATELY at market open (시가 전량 손절) [00:04:37], [00:06:41].
    -> Re-enter ONLY after institutional supply re-establishes positive flow [00:05:40].

MODULE B: 타 트레이더 B ENGINE (Microstructure, Scalping & Trailing Stop)

[B-1. MICROSTRUCTURE & INTRADAY ENTRY FILTER]
- Real-Time Search Ranking Filter: Target must rank within Top 3 [00:07:32].
- Order Flow Velocity & Acceleration:
  * Enter ONLY when execution velocity (체결 가속도) spikes with clustered market
    orders [00:28:47].
  * Check upside runway: 2nd VI or Round-Number targets must be open [00:25:20].

[B-2. MICRO EXIT & SCRATCH STOP RULES]
- Immediate Acceleration Rule (즉각 손절):
  * IF stock does not surge within 1~2 ticks immediately post-entry:
    -> Trigger SCRATCH STOP (Cut at entry price or -1 tick immediately) [00:02:28].
- Wave Exhaustion Exit: Sell 40% when upward momentum stalls; dump remaining 60% on
  first lower tick [00:10:41], [00:11:41].

[B-3. INTRADAY PROFIT TRAILING STOP (50% RULE)]
- Hard Trailing Profit Cut:
  * Track intraday peak net profit (P_peak).
  * IF (P_current <= P_peak * 0.50):
    -> Terminate all open positions and shutdown trading for the day [00:01:21].

[B-4. HARDWARE ISOLATION & CAPITAL DISCIPLINE]
- Sub-Account Buffer: Divert to 3M KRW Sub-Account upon tilt/loss limit hit [00:33:55].
- Zero Inbound Deposits: No additional deposit allowed under any circumstances [00:17:18].
```

## 우리 배선과의 충돌 — **정면으로 부딪히는 것 하나**

**A-3: "조 단위 거래대금 장세에서는 잡주/테마주 금지(BANNED)."**

우리 짝꿍 엔진은 **테마 소형주만** 매매한다. 대장이 상한가로 가는 종목은 거의 전부
저가 테마주다. 즉 v9 A-3 은 우리 시스템을 **대형주 장세에서 통째로 끄라**는 말이다.

51 의 A-1 은 "박스권이면 짝꿍으로 옮겨라"였고 이건 **켜는 조건**이었다. v9 A-3 은
반대편에서 **끄는 조건**을 준다. 둘을 합치면:

    거래대금 8조↑ + 지수 횡보  → 짝꿍(테마 소형주) 최대
    조 단위 대형주 장세         → 짝꿍 금지, 대형주 스윙으로 전환
    코스닥 7~8조↓              → 비중 1/10 방어

우리는 지금 셋 중 마지막 구간에 있고(코스닥 5~6조) 그건 이미 배선돼 있다
(`--scout-gate-jo 8`). **대형주 전환 갈래는 배선이 없다.**

## v9 에서 새로 생긴 것 — 전부 **일봉으로 검증 가능하다**

지금까지 만쥬 규칙은 호가·틱이라 데이터가 없어 검증을 못 했다. v9 모듈 A 는 다르다.

| 규칙 | 필요한 데이터 | 우리 보유 |
|---|---|---|
| 수급 턴어라운드(2음봉 후 50%+ 되흡수일 진입) | 일봉 + 외인·기관 순매수 | ✅ 10년 |
| 섹터 대장 선정(52주 신고가 다수) | 일봉 | ✅ 10년 |
| 대장 vs 2등주 상대강도(지수 급락일 박스 상단 돌파) | 일봉 + 지수 | ✅ |
| 과매도 종배 게이트(미국 −2.5% vs 한국 −3.0%↑) | 미국 지수 + 국내 지수 + 선물 수급 | ⚠️ 미국·선물 미보유 |
| 돌발 악재 시가 전량 손절 | 뉴스 | ❌ 크롤러 금지 |

**수급 턴어라운드와 섹터 대장은 지금 당장 사전등록·측정이 가능하다.**
