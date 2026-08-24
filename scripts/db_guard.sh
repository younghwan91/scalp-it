#!/usr/bin/env bash
# 장중 DB 컨테이너 헬스가드 — 틱 수집(scalp-tick-collect, 평일 08:50~15:40)이 쓰는
# KR_QUANT_DB(quant-airflow-timescaledb-1, localhost:5432)가 장중에 죽지 않게 지킨다.
#
# 왜 필요한가:
#   1) 형제 레포 quant-airflow 의 cron 이 매일 10:00 `docker compose up -d`,
#      10:10 `wait_and_stop.sh` 를 돌린다. wait_and_stop 은 그날 예정 DAG 가 모두
#      끝나면 `docker compose stop` 으로 스택 전체를 내린다 — 그 스택에는 우리 틱 DB 인
#      timescaledb 가 포함된다(같은 컨테이너 공유: quant-airflow 의 TIMESCALE_DB=kr_quant
#      == 우리 KR_QUANT_DB 의 localhost:5432/kr_quant). DAG 가 장중에 일찍 끝나면
#      장중에 우리 DB 가 꺼져 그날 틱 수집이 죽는다.
#   2) 전날 저녁 `docker compose stop` 뒤 compose 의 `restart: unless-stopped` 는
#      명시적으로 stop 된 컨테이너를 되살리지 않는다. 그래서 다음날 10:00 `compose up`
#      전까지 — 즉 08:50 데몬 시작 시점엔 — DB 가 내려가 있을 수 있다.
#
# 이 스크립트는 우리 쪽에서만 도는 비침습 방어다. quant-airflow 는 절대 건드리지
# 않는다(읽지도 쓰지도 않음, docker start 대상 컨테이너 하나만 만진다).
# 멱등: 이미 정상이면 아무것도 안 하고 조용히 끝난다. 죽어 있으면 `docker start`
# 후 pg_isready 로 복구를 확인한다. cron 메일 스팸을 피하려고 정상일 땐 로그도 안 남긴다.
#
# 잔여 위험(주의): 데몬(collector.py `_flush`)은 DB 순단 시 재연결하지 못하고 flush
# 태스크가 죽는다 — 이 가드는 DB 를 다시 살리지만, 순단이 이미 났다면 데몬 자체 복구는
# 못 한다. 데몬 복원력 개선(제안 별도)이 함께 필요하다.
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
