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
  # 만쥬 손익 구조 — 짧게 잃고 크게 먹는다(docs/research/manju).
  #   --take-profit 0 : 익절 상한 없음. 상한을 걸면 전부를 먹여살릴 우측 꼬리를 자른다.
  #                     1년 240조합 스윕에서 익절을 풀수록 net 이 단조 증가했다.
  #   --stop-loss -0.005 : 원본이 말하는 약손절 −0.5%.
  #   --hold 240 : 원본 타임아웃 1~2분, 최대 3~4분.
  # 진입 방아쇠(대장 상한가)와 대장 조건부 청산은 아직 미구현 — 검증 중이다.
  /home/young/.local/bin/uv run --extra pg scalp-pair-detect \
    --live --live-order --duration "$(date '+%H:%M')-15:20" \
    --take-profit 0 --stop-loss -0.005 --hold 240 \
    >> data/pair_detect.log 2>&1
  echo "=== [$(date '+%F %T')] 감지기 종료(exit $?), 재시작 검토"
  sleep 5
done
echo "=== [$(date '+%F %T')] 장 마감, 짝꿍 감지기 일과 종료"
