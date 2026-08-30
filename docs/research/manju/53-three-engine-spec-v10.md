# 3인 엔진 명세 v10 (53번) — 매크로 크로스에셋 전환

> 2026-08-30, 사용자가 그대로 옮겨준 것. **출처 정리 문서다.**
> [`52`](52-three-engine-spec-v9.md)(v9)의 후속. **모듈 A 가 또 통째로 바뀌었다.**

## 버전별 모듈 A 변천 — 3판 연속 다른 것을 말한다

| 판 | 모듈 A 의 핵심 | 우리 시스템과의 관계 |
|---|---|---|
| 50·51 | 만쥬 — **장중 짝꿍**(상한가 3호가·VI 진입) | **우리 시스템 전부가 여기서 나왔다** |
| 52(v9) | 만쥬+침착해 — 섹터 대장·수급 턴어라운드·과매도 종배 | 짝꿍 트리거가 사라짐 |
| 53(v10) | 만쥬+침착해 — **매크로 크로스에셋 상태기계**·자회사 짝꿍·NXT | 짝꿍 트리거 여전히 없음 |

**상한가 3호가 짝꿍 진입이 두 판 연속 빠졌다.** 우리가 만든 것 전부가 그 규칙 위에
서 있다. 이건 기록해 둘 사실이지 우리가 판단할 문제가 아니다 — 사용자 결정 사항이다.

**모듈 B(타 트레이더 B)는 네 판 내내 거의 그대로다.** 흔들리는 건 A 뿐이다.

---

## 원문

```
MODULE A: MANJU & CHIMCHAK-HAE ENGINE (Macro, Sector/Pair Leader, Flow & Swing)

[A-1. MACRO CROSS-ASSET SIGNAL MATRIX (다중 자산 매크로 연동 신호망)]
- Real-Time Cross-Asset State Machine:
  * STATE 1 (Bullish Over-Correction / 지수주 베팅):
    - Conditions: WTI Crude plunges/stabilizes + US Nasdaq Futures rise
      + Domestic Index drops excessively (선반영 과매도) [00:12:44], [00:15:00].
    - Action: BUY / ACCUMULATE Index Heavyweights (SK Hynix, Samsung Electronics).
  * STATE 2 (Geopolitical Risk Off / 헷지 테마 가동):
    - Conditions: WTI Crude spikes (+3%~+29%) + US Futures drop + War headlines
      escalate [00:11:08], [00:12:44].
    - Action: BUY Sector Sympathy Hedge Plays (Defense: Hanwha Systems/LIG Nex1,
      Energy, Shipping).
  * STATE 3 (Extreme Ambiguity / 무포지션 셧다운):
    - Conditions: Conflicting macro news + Extreme volatility (5% swings on news)
      + Foreign selloff > 2T KRW [00:16:42], [00:20:46], [00:25:28].
    - Action: CASH 100% (무포 관망) & Terminate execution. Never bottom-fish blindly.

[A-2. SECTOR SYMPATHY & SUBSIDIARY PAIR DYNAMICS (섹터 짝꿍 & 자회사 동조화)]
- Subsidiary/Affiliate Sympathy Pair (지분/자회사 짝꿍):
  * Trigger: Parent Stock hits Limit Up on AI/M&A news (e.g., Polaris Office locked
    at +30%) [00:18:40].
  * Action: Buy Subsidiary/Affiliate (Polaris AI) before morning gap-up; sell on
    morning opening gap [00:19:04].
- Multi-Asset Sympathy Rotation:
  * Strike/Labor Issue Sympathy: Samsung Electronics strike news -> Buy Robot/Automation
    sector [00:40:56], [00:44:25].
  * Heavy vs. Agile Sympathy Sizing:
    - 1st Leader with massive cap/heavy chart resistance -> Shift to 2nd Sympathy Leader
      near 52-week breakout (e.g., LIG Nex1 over Hanwha Systems) [00:11:46].

[A-3. NXT TIMING & SQUEEZE LOGIC (대체거래소 호가 및 숏커버링)]
- NXT Dip Sizing Law:
  * BANNED: Buying into intraday/after-hours vertical spikes (추격 매수 절대 금지)
    [00:29:33].
  * Rule: Accumulate strictly on pullbacks towards the end of NXT session
    (18:00~19:50) [00:29:45].
- Institutional/Foreign Short-Squeeze Setup:
  * Consecutive foreign/institutional net buy accumulation at 52-week highs ->
    Bet on next-day Short Squeeze gap-up [00:02:10], [00:32:18].
- Global Lead-Lag Decoupling:
  * IF US stock catalyst spikes +10% AH but closes -2% by US morning:
    -> Scratch exit immediately at KR market open (수수료 손절 탈출) [00:33:00].

[A-4. MANDATORY EXECUTION HARD STOPS & CAPITAL DISCIPLINE]
- Strict 1~2% Stop-Loss Gate:
  * Cut positions ruthlessly within 1~2% loss upon thesis invalidation
    (수수료/약손절은 손실이 아님) [00:22:47], [00:34:03], [00:44:47].
- Consecutive Loss / Multi-Day Regime Drawdown:
  * Consecutive losses / chopping: Reduce active size to 1/10th (10% limit) or
    switch to Cash 100% [00:53:36], [00:56:24].
- FOMO Lockdown:
  * After early exit or missing a massive continuation spike: DO NOT CHASE.
    Terminate terminal/HTS for the day [00:04:41], [00:50:17].

MODULE B: 타 트레이더 B ENGINE — v9 과 동일
[B-1] 실시간 검색 Top 3 · 체결 가속도 · 런웨이 검증
[B-2] 본절 탈출 · 파동 소진 40/60
[B-3] 정점 이익 50% 반납 → 전량 청산·셧다운
[B-4] 300만원 서브계좌 격리 · 입금 0원
```

## 우리가 지금 쓸 수 있는가 — 데이터 기준

| 규칙 | 필요한 것 | 보유 |
|---|---|---|
| A-1 상태기계(WTI·나스닥 선물·외인 선물) | 유가·미 선물·선물 수급 | ❌ 셋 다 없다 |
| A-2 **자회사 짝꿍**(모회사 상한가 → 자회사 익일 갭 매도) | 지분·계열 관계 | ❌ 관계 데이터 없음 |
| A-2 2등주 전환(52주 돌파 근접) | 일봉 | ✅ |
| A-3 숏스퀴즈(52주 신고가 + 외인·기관 연속 순매수 → 익일 갭) | 일봉 + 수급 | ✅ **바로 측정 가능** |
| A-3 NXT 눌림 매수 | NXT 시세 | ❌ |
| A-4 1~2% 손절 · 1/10 축소 · FOMO 잠금 | — | ⏸ 우리 손절은 −0.5%(초단타 기준) |

**A-3 숏스퀴즈 셋업이 지금 데이터로 검증 가능한 유일한 신규 규칙이다.**
52주 신고가 + 외인·기관 연속 순매수 → 익일 갭업. 일봉 10년 + supply_demand 로 잰다.
