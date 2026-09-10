#!/usr/bin/env bash
# 장중 DB 컨테이너 헬스가드 — 틱 수집(scalp-tick-collect, 평일 08:50~15:40)이 쓰는
# KR_QUANT_DB(quant-airflow-timescaledb-1, localhost:5432)가 장중에 죽지 않게 지킨다.
#
# 2026-09-11 갱신 — 이 가드가 막던 근본 원인은 이제 없다. quant-airflow 의
# timescaledb 는 docker-compose.timescale.yml 로 분리돼 24/7 상시고, Airflow(예전엔
# 같은 스택으로 하루 4번 up/down 하던 그것)는 simnode 로 옮겨 그쪽에서 자체
# 컴포즈로 돈다 — 더는 이 컨테이너를 내렸다 올리는 주체가 없다. 그리고 어제
# (2026-09-10) collector.py 에 `_reconnect_db`/`_flush` 재연결이 들어가 DB 순단
# 중에도 데몬이 안 죽고 버퍼를 보존한다(2026-09-11 spool_dir 로 프로세스 재시작
# durability 까지 보강, tests/test_realtime_collector.py 의 spool 테스트 참고).
#
# 그래도 이 스크립트는 지운다. 방금 막은 위험이 사라졌다고 그 자리에서 바로
# 안전망까지 걷어내는 건 검증 기간 없는 성급한 정리다 — 컨테이너가 죽어 있어도
# 멱등하게 아무 해가 없으니(정상이면 조용히 종료) 당분간 무해한 이중 방어로 둔다.
# 며칠 지켜본 뒤에도 발동 이력이 없으면(`data/db_guard.log` 확인) 지워도 된다.
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
