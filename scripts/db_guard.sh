#!/usr/bin/env bash
# 장중 DB 컨테이너 헬스가드 — 틱 수집(scalp-tick-collect, 평일 08:50~15:40)이 쓰는
# KR_QUANT_DB(quant-airflow-timescaledb-1, localhost:5432)가 장중에 죽지 않게 지킨다.
#
# **2026-09-11 크론 비활성화됨(crontab 참고) — 이 스크립트를 그대로 두지 마라.**
# DB 원본이 trader 에서 simnode(192.168.45.9:5433)로 옮겨갔다(pg_promote). 이
# 컨테이너(quant-airflow-timescaledb-1)는 이제 **의도적으로 stop 상태**다 —
# 옛 primary 의 잔해라 살리면 안 된다. 이 가드가 여전히 크론에 걸려 있었다면
# "컨테이너가 죽어 있다"고 오인해 `docker start` 로 되살렸을 것이고, 그러면
# 최신 데이터가 없는 옛 DB 가 다시 쓰기 가능 상태로 돌아와 **split-brain**
# 위험을 만든다 — 예전에 적었던 "멱등하게 아무 해가 없다"는 지금은 틀린
# 말이다. 파일은 남겨둔다(trader 가 다시 primary 를 맡게 되면 그때 되살릴 수
# 있게) — 하지만 크론에 다시 걸기 전에 CONTAINER 가 가리키는 게 지금 진짜
# primary 가 맞는지 반드시 확인할 것.
set -uo pipefail

CONTAINER="${DB_GUARD_CONTAINER:-quant-airflow-timescaledb-1}"
DB_USER="${TIMESCALE_USER:-kr_quant}"
LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
LOG="$LOG_DIR/db_guard.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" >>"$LOG"; }

ready() { docker exec "$CONTAINER" pg_isready -U "$DB_USER" -q >/dev/null 2>&1; }
running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]
}

# 컨테이너가 아예 없으면(예: quant-airflow 가 `docker compose down` 으로 제거) 우리가
# 만들 수는 없다 — compose up 은 남의 레포 소관. 로그만 남기고 빠진다.
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  log "컨테이너 $CONTAINER 없음 — quant-airflow 스택 미기동(10:00 compose up 전?). 조치 불가."
  exit 0
fi

# 정상이면 조용히 종료(로그 스팸 방지).
if running && ready; then
  exit 0
fi

if ! running; then
  log "DB 컨테이너 DOWN 감지 — docker start $CONTAINER"
  docker start "$CONTAINER" >/dev/null 2>&1 || log "경고: docker start 실패"
fi

# pg_isready 될 때까지 최대 ~30s 폴링.
for i in $(seq 1 15); do
  if ready; then
    log "DB 복구 OK ($CONTAINER, ${i}회 폴링)"
    exit 0
  fi
  sleep 2
done

log "경고: $CONTAINER 재기동했으나 pg_isready 30s 실패 — 수동 점검 필요"
exit 0
