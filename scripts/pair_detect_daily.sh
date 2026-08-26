#!/usr/bin/env bash
# 장중 짝꿍 감지기 dry-run 자동 실행 (소켓 수집 겸함, 실주문 절대 없음).
# App key 1세션 제약: 이 프로세스가 tick-collect 를 대체한다(감지+저장).
# 크래시 시 15:20 까지 자동 재시작해 하루를 통째로 날리지 않는다.
set -u
cd /home/young/Documents/git/scalp-it
export $(grep -E '^KIWOOM_APP_(KEY|SECRET)=' ../quant-airflow/.env | xargs)
export $(grep -v '^#' ../kr-quant/.env | xargs)
pkill -f "scalp-tick-collect" 2>/dev/null; sleep 2
END=$(date -d "15:20" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  [ "$REMAIN" -lt 1 ] && break
  echo "=== [$(date '+%F %T')] 짝꿍 감지기 시작 (남은 ${REMAIN}분, dry-run·수집겸함)"
  # --live-order 있으나 --confirm-real 없음 → 실주문 0. --duration 이 15:20 에 자체 종료.
  # ── 만쥬 방식 (docs/research/manju/44-manju-method-from-videos.md) ──────────
  # 손익 구조 — 짧게 잃고 크게 먹는다.
  #   --take-profit 0    익절 상한 없음. 상한을 걸면 전부를 먹여살릴 우측 꼬리를 자른다.
  #                      1년 240조합 스윕에서 익절을 풀수록 net 이 단조 증가했다.
  #   --stop-loss -0.005 약손절 −0.5%. **대용치다** — 원본은 "−0.5%에서 자른다"가
  #                      아니라 "대장이 상한가를 놓칠 때 자르는데 그게 대체로 −0.5%".
  #   --hold 240         타임아웃 1~2분, 최대 3~4분.
  # 진입 방아쇠(§5-1) — 1년 3,021건 실측으로 **잠김직전**을 골랐다. 후속주 전방수익:
  #   잠김직전 +10분 +0.178%(30분까지 유일하게 플러스) > 잠김 +0.064% > 터치 +0.009%
  #   > 3분잠김 −0.007%. 대장이 굳는 걸 확인하고 들어가면(3분잠김 굳힘 80%) 이미 먹을
  #   게 없다 — 굳힘 비율과 수익이 정확히 반대 순서다. 그래서 터치는 끈다.
  #   --limitup-near 0.05 = 상한가까지 5%p 이내(≈ 등락률 +25%), 원본 presurge 정의.
  #   --limitup-require-surge 로 모멘텀까지 AND — 없으면 상한가 근처에서 정체된
  #   종목에도 발동한다(원본은 +25% 교차 **그 분에 +1%** 올랐을 때만).
  # 종목 선정(§3·§4-2) — 고정 쌍이 아니라 그날 유니버스에서 섹터별로 새로 고른다.
  #   유니버스 선정이 실패하면 기존 워치리스트로 자동 폴백한다(하루를 날리지 않는다).
  #   45종목 = 단일 앱키 한도 50 안. 넘기면 커넥션이 쪼개져 서로 kick 한다(08-24 사고).
  # 호가 두께(§4-1, 만쥬 최우선 필터) — 1000만원 왕복 충격 20bp 초과 종목은 배제.
  # 돌발 재료(§6) — 장중 새 공시로 강한 재료가 뜬 종목을 구독에 추가한다.
  #   "예상치 못한 돌발 뉴스가 나오면 즉각 관련 종목 매수". 90초마다 확인하고
  #   한 번에 3종목까지. 구독 상한(50)이 있어 자리가 없으면 거절을 로그에 남긴다.
  # 대장 조건부 청산(§5-3)은 기본 켬 — 대장이 상한가를 놓치면 손익 무관 즉시 청산.
  # 대장 굳힘 판정(§5-2 "수십만~수백만 주를 안정적으로 굳히면") — 가격만 보던 걸
  #   매수벽 잔량까지 본다. 2026-08-24 실측 분위: 매수1호가 금액 중앙 2,203만 /
  #   75% 1억 / 95% 8.4억. 만쥬가 말한 두께는 상위 5% 수준이라 3억을 문턱으로 둔다.
  #   --lock-wall-decay 0.5 = 벽이 최근 최대의 반토막이면 이탈(풀리기 전에 먼저 보인다).
  # 매도 폭탄(§5-4 "프로그램 매도 폭탄이 쏟아져 막히면") — 분당 매도 체결대금 실측
  #   중앙 3,887만 / 95% 9.7억 / 99% 112억. 10초 창에서 3억이면 상위권 투매다.
  # 순간 체결량(§5-1) — 5천만원 이상 체결이 대장에 최소 1건. 대량 자금 유입 확인.
  # 국면별 비중(§8-1) — 거래대금으로 자동 결정. 극침체면 배수 0 = 신규 진입 없음.
  #   **오늘은 4.69조라 배수 0이다.** dry-run 이므로 --size-by-regime 을 켜지 않고
  #   기록만 한다 — 만쥬가 쉬라고 한 국면의 표본도 검증에 필요하기 때문. 실발주로
  #   전환할 때 이 플래그를 켜야 한다.
  /home/young/.local/bin/uv run --extra pg scalp-pair-detect \
    --live --live-order --duration "$(date '+%H:%M')-15:20" \
    --take-profit 0 --stop-loss -0.005 --hold 240 \
    --limitup-near 0.05 --limitup-require-surge \
    --universe-size 45 --max-roundtrip-bp 20 --order-krw 10000000 \
    --lock-min-krw 300000000 --lock-wall-decay 0.5 \
    --dump-krw 300000000 --big-trades 1 \
    --new-listing-days 5 \
    --breaking-interval 90 --breaking-min-weight 0.7 --breaking-max 3 \
    >> data/pair_detect.log 2>&1
  echo "=== [$(date '+%F %T')] 감지기 종료(exit $?), 재시작 검토"
  sleep 5
done
echo "=== [$(date '+%F %T')] 장 마감, 짝꿍 감지기 일과 종료"
