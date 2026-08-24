# 짝꿍 실시간 감지·주문 런북 (2026-08-25)

> **현재 단계: dry-run 검증까지 완료. 실발주는 미검증 — 반드시 dry-run 부터.**
> 엣지 자체는 미확정(감사 doc38: 사후선택 착시 60~70%). 리허설 목적 = 돈벌이 아니라 실체결·역선택 실측.

## 파이프라인 (전부 구현됨)
소켓 수집 → 감지(대장 서지+후속 리드랙+매수비율≥30%) → 리스크가드(5만원·킬스위치) → 알림 → 주문(1주 지정가, dry-run 기본) → 청산.

## 내일 장중 순서 (실발주까지)
1. `cd /home/young/Documents/git/scalp-it`
2. `export $(grep -E '^KIWOOM_APP_(KEY|SECRET)=' ../quant-airflow/.env | xargs)`  (전용키 쓰면 PAIR_DETECT_APP_KEY/SECRET)
3. `export $(grep -v '^#' ../kr-quant/.env | xargs)`
4. **① 먼저 dry-run (실주문 0):**
   `uv run --extra pg scalp-pair-detect --live --live-order`
   → 화이트리스트·계좌·보유차단·매수비율·주문로그를 눈으로 확인. --confirm-real 없으면 절대 실발주 안 함.
5. **② dry-run 정상 확인 후, 첫 1주 실발주 (사용자 동석):**
   `uv run --extra pg scalp-pair-detect --live --live-order --confirm-real --qty 1`
   → 첫 알림에서 1주 매수→체결→청산→미체결취소 동작 관찰. 미체결(oso) 필드명 첫 실주문으로 확정.

## 안전장치 (코드 레벨)
- 보유 3종목 하드차단: 보유A 000010 · 보유B 000020 · 보유C 222222 + **주문 직전 실시간 잔고조회 전종목 차단(fail-closed)**
- 이중 게이트: `--live-order` + `--confirm-real` 둘 다 있어야 실발주
- 수량 1주 하드캡(최대 10), 종목 dedup, 무재시도, 가격밴드 ±3%, 종목당 5회/일 20회
- 일일 손실한도 5만원 + 연속손절 3회 + 정지파일 `data/pair_STOP` → 킬스위치
- **자동 청산 주문 없음(알림·리허설), 기존 보유 절대 미접근**

## 등록 쌍 (data/pair_watchlist.json)
인디에프→좋은사람들 · 더즌→갤럭시아머니트리 · 아이티센글로벌→쿠콘 · 금호전기→금호건설

## 주의
- App key 1세션: `--live` 감지기와 tick-collect 데몬이 같은 키면 충돌 → 전용키 쓰거나 tick-collect 잠깐 내림
- 어제 dry-run 리플레이: 감지→가드→주문dry-run→킬스위치(연속손절3) 완주 확인, 실POST 0.
