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
  # 진입 방아쇠(§5-1) — 원자료가 터치/잠김/잠김직전 중 무엇인지 확정 못 한다(§11 #4).
  #   그래서 터치와 근접(4%p)을 **둘 다 켜서** 각각 기록하고 하루치 실측으로 가린다.
  # 종목 선정(§3·§4-2) — 고정 쌍이 아니라 그날 유니버스에서 섹터별로 새로 고른다.
  #   유니버스 선정이 실패하면 기존 워치리스트로 자동 폴백한다(하루를 날리지 않는다).
  #   45종목 = 단일 앱키 한도 50 안. 넘기면 커넥션이 쪼개져 서로 kick 한다(08-24 사고).
  # 호가 두께(§4-1, 만쥬 최우선 필터) — 1000만원 왕복 충격 20bp 초과 종목은 배제.
  # 대장 조건부 청산(§5-3)은 기본 켬 — 대장이 상한가를 놓치면 손익 무관 즉시 청산.
  /home/young/.local/bin/uv run --extra pg scalp-pair-detect \
    --live --live-order --duration "$(date '+%H:%M')-15:20" \
    --take-profit 0 --stop-loss -0.005 --hold 240 \
    --limitup-touch --limitup-near 0.04 \
    --universe-size 45 --max-roundtrip-bp 20 --order-krw 10000000 \
    >> data/pair_detect.log 2>&1
  echo "=== [$(date '+%F %T')] 감지기 종료(exit $?), 재시작 검토"
  sleep 5
done
echo "=== [$(date '+%F %T')] 장 마감, 짝꿍 감지기 일과 종료"
