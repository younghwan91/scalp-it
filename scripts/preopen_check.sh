#!/usr/bin/env bash
# 개장 전 점검 — 08:55 런처(`pair_detect_daily.sh`)가 필요로 하는 전제를 전부 읽어본다.
#
# 왜 있나: 2026-09-11 레포 이동(`~/Documents/git` → `~/git`) 이후 9건이 **조용히**
# 깨졌다. 런처의 `cd` 실패, 자격증명 빈 export, 낡은 의존성 핀 같은 것들이다.
# 공통점은 에러가 안 나거나 로그 한 줄만 남고, 장이 열린 뒤에야 드러난다는 것이다.
# 그래서 "장 시작 전에, 읽기만 해서, 시끄럽게" 확인하는 자리를 따로 만든다.
#
# 읽기 전용이다 — 프로세스를 죽이지도, 파일을 쓰지도, 주문을 내지도 않는다.
# 한 항목이라도 FAIL 이면 종료코드 1. WARN 은 0(사람이 판단할 것).
#
#   사용: scripts/preopen_check.sh            ← 손으로
#         crontab 08:30 에 걸어 로그로 받아도 된다(런처보다 먼저 돌게)
set -uo pipefail

FAIL=0
WARN=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; WARN=$((WARN + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL + 1)); }
sec()  { printf '\n%s\n' "$*"; }

# 런처와 **같은 방식**으로 경로를 유도한다 — 여기서 다른 방식을 쓰면 점검이
# 런처의 실제 동작을 안 재는 꼴이 된다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || { echo "레포 경로 유도 실패"; exit 1; }
cd "$REPO" || { echo "cd 실패: $REPO"; exit 1; }

echo "개장 전 점검 — $(date '+%F %T %A') · $(hostname) · $REPO"

# ─────────────────────────────────────────────────────────── 1. 경로·자격증명
sec "1) 경로와 자격증명"

ok "레포 경로 유도: $REPO"

KW_ENV="$REPO/../quant-airflow/.env"
KQ_ENV="$REPO/../kr-quant/.env"

if [ -r "$KW_ENV" ]; then
  # 값은 절대 찍지 않는다. 개수와 길이만 본다.
  n_key=$(grep -cE '^KIWOOM_APP_(KEY|SECRET)=' "$KW_ENV" || true)
  n_key2=$(grep -cE '^KIWOOM_APP_(KEY_2|SECRET_2)=' "$KW_ENV" || true)
  [ "$n_key" -eq 2 ] && ok "키움 1세트 키(KEY/SECRET) 있음" \
                     || bad "키움 KEY/SECRET 이 $n_key 개다(2 여야 함) — 자격증명 없이 감지기가 뜬다"
  # 2세트는 필수가 아니다. 없으면 감지기와 tick-collect 가 같은 키를 물고 충돌한다.
  [ "$n_key2" -eq 2 ] && ok "키움 2세트 키(KEY_2/SECRET_2) 있음 — 수집·감지 분리 가능" \
                      || warn "KIWOOM_APP_KEY_2/SECRET_2 없음 — 감지기와 tick-collect 가 같은 키로 충돌할 수 있다"
else
  bad "읽을 수 없음: $KW_ENV — 런처의 export 가 조용히 빈 값이 된다"
fi

if [ -r "$KQ_ENV" ]; then
  grep -q '^KR_QUANT_DB=' "$KQ_ENV" && ok "KR_QUANT_DB 선언됨" \
                                    || bad "KR_QUANT_DB 가 $KQ_ENV 에 없다"
else
  bad "읽을 수 없음: $KQ_ENV"
fi

# 런처가 **절대경로로 박아 쓰는** uv. PATH 와 무관하게 이 파일이 있어야 한다.
UV=/home/young/.local/bin/uv
[ -x "$UV" ] && ok "uv 실행 가능: $UV" || bad "uv 없음/실행 불가: $UV — 런처가 못 뜬다"

# ─────────────────────────────────────────────────────────── 2. 실행 환경
sec "2) 실행 환경(가상환경·콘솔 스크립트·의존성)"

# `--help` 가 뜨면 venv·콘솔 스크립트·import 사슬이 전부 살아있다는 뜻이다.
# 런처와 같은 `uv run --extra pg` 를 쓴다(activate 는 쓰지 않는다 — 런북 참고).
if "$UV" run --extra pg scalp-pair-detect --help >/dev/null 2>&1; then
  ok "scalp-pair-detect --help 통과 — venv·진입점·import 정상"
else
  bad "scalp-pair-detect 가 뜨지 않는다 — 'uv run --extra pg scalp-pair-detect --help' 를 직접 돌려 원인을 보라"
fi

# 의존성 핀 계약. 키움은 cont-yn/next-key 를 **헤더로만** 주고, 그걸 바디로 옮겨
# 담는 건 kiwoom_client `request()` 다. 이게 없는(낡은) 핀이면 연속조회가 1페이지에서
# 조용히 끝나고 테마 그룹이 첫 100개로 잘린다 — 2026-09-13 에 실제로 그랬다.
if "$UV" run --extra pg python - <<'PY' 2>/dev/null
import inspect, sys
import kiwoom_client
from kiwoom_client import tick_size
src = inspect.getsource(kiwoom_client.base)
assert 'resp.headers.get("cont-yn"' in src, "헤더→바디 폴백 없음(핀이 낡았다)"
assert tick_size(12345) == 10 and tick_size(999) == 1, "호가단위 표가 틀렸다"
PY
then
  ok "kiwoom-client 계약 통과 — 헤더 페이지네이션 폴백·호가단위 정상"
else
  bad "kiwoom-client 핀이 낡았거나 호가단위가 틀렸다 — 'uv sync --extra pg' 후 재확인 (테마 조회가 조용히 잘린다)"
fi

# ─────────────────────────────────────────────────────────── 3. DB
sec "3) DB(틱이 실제로 들어갈 곳)"

# 감지기는 틱을 KR_QUANT_DB 에 **쓴다.** 그게 스탠바이를 가리키면 모든 insert 가
# 실패한다 — 2026-09-11 에 프라이머리가 simnode 로 승격된 뒤 특히 조심할 것.
DB_OUT="DB_ENV_MISSING"
if [ -r "$KQ_ENV" ]; then
  # 런처와 같은 방식으로 읽는다(`set -a` + source). 이 서브셸 안에서만 유효하다.
  DB_OUT="$(set -a; . "$KQ_ENV"; set +a; "$UV" run --extra pg python -c '
import os, psycopg2
con = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
cur = con.cursor()
cur.execute("SELECT pg_is_in_recovery()"); rec = cur.fetchone()[0]
cur.execute("SELECT max(date) FROM supply_demand")
print(f"RECOVERY={rec} 기준일={cur.fetchone()[0]}")
con.close()' 2>&1 | tail -1)"
fi

case "$DB_OUT" in
  RECOVERY=False*)
    ok "DB 접속 OK · 쓰기 가능(프라이머리) · ${DB_OUT#* }" ;;
  RECOVERY=True*)
    bad "DB 가 스탠바이다(pg_is_in_recovery=true) — 틱 insert 가 전부 실패한다. KR_QUANT_DB 가 프라이머리를 가리키는지 보라" ;;
  *)
    bad "DB 점검 실패 — $DB_OUT" ;;
esac

# ─────────────────────────────────────────────────────────── 4. 상태 파일·프로세스
sec "4) 킬스위치와 잔류 프로세스"

# 정지파일이 남아 있으면 감지기는 뜨지만 그날 아무것도 안 한다(조용한 무거래).
if [ -e "$REPO/data/pair_STOP" ]; then
  bad "킬스위치가 켜져 있다: data/pair_STOP — 지우지 않으면 오늘 거래가 없다"
else
  ok "킬스위치 없음(data/pair_STOP)"
fi

# 어제 것이 남아 있으면 08:55 런처와 겹쳐 중복 주문이 날 수 있다.
LEFT="$(pgrep -fa 'scalp-pair-detect|pair_detect_daily' 2>/dev/null | grep -v preopen_check || true)"
if [ -n "$LEFT" ]; then
  bad "감지기가 이미 돌고 있다 — 08:55 런처와 겹치면 중복 주문 위험"
  echo "$LEFT" | sed 's/^/       /' | cut -c1-120
else
  ok "잔류 감지기 없음"
fi

# ─────────────────────────────────────────────────────────── 5. 데이터 신선도
sec "5) 입력 데이터 신선도"

# 테마 스냅샷은 짝꿍 선정의 입력이다(16:10 크론, DB 의 theme_groups 에 쌓인다 —
# 파일이 아니다). 며칠 묵으면 편입·편출을 놓친다. 그룹 수도 같이 본다: 100 이면
# 연속조회가 첫 페이지에서 끊긴 흔적이다(ka90001 은 한 페이지 100행).
SNAP_OUT="THEME_ENV_MISSING"
if [ -r "$KQ_ENV" ]; then
  SNAP_OUT="$(set -a; . "$KQ_ENV"; set +a; "$UV" run --extra pg python -c '
import os, psycopg2
con = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
cur = con.cursor()
cur.execute("SELECT max(snapshot_date) FROM theme_groups")
latest = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM theme_groups WHERE snapshot_date = %s", (latest,))
print(f"{latest} {cur.fetchone()[0]}")
con.close()' 2>&1 | tail -1)"
fi
SNAP_DATE="${SNAP_OUT%% *}"
SNAP_ROWS="${SNAP_OUT##* }"
if [ "$SNAP_DATE" = "$SNAP_OUT" ]; then
  warn "테마 스냅샷 조회 실패 — $SNAP_OUT"
else
  AGE_D=$(( ( $(date +%s) - $(date -d "$SNAP_DATE" +%s) ) / 86400 ))
  if [ "$AGE_D" -le 3 ]; then
    ok "테마 스냅샷 $SNAP_DATE (${AGE_D}일 전) · 그룹 ${SNAP_ROWS}개"
  else
    warn "테마 스냅샷이 $SNAP_DATE (${AGE_D}일 전)로 묵었다 · 그룹 ${SNAP_ROWS}개 — 16:10 scalp-theme-snapshot 확인"
  fi
  [ "$SNAP_ROWS" = "100" ] && warn "그룹이 정확히 100개다 — 연속조회가 첫 페이지에서 끊겼을 수 있다(kiwoom-client 핀 확인)"
fi

# ─────────────────────────────────────────────────────────── 6. 기록할 자리
sec "6) 로그·디스크"

[ -w "$REPO/data" ] && ok "data/ 쓰기 가능" || bad "data/ 에 쓸 수 없다 — 로그·일지가 안 남는다"

AVAIL_MB=$(df -Pm "$REPO" | awk 'NR==2{print $4}')
if [ "$AVAIL_MB" -ge 5000 ]; then
  ok "디스크 여유 $((AVAIL_MB / 1024))GB"
elif [ "$AVAIL_MB" -ge 1000 ]; then
  warn "디스크 여유 $((AVAIL_MB / 1024))GB — 틱 수집이 하루 수 GB 를 쓴다"
else
  bad "디스크 여유 ${AVAIL_MB}MB — 틱 수집이 곧 멈춘다"
fi

# ─────────────────────────────────────────────────────────── 7. 크론
sec "7) 크론 등록 상태"

CRON="$(crontab -l 2>/dev/null || true)"
printf '%s\n' "$CRON" | grep -q 'pair_detect_daily.sh' \
  && ok "08:55 런처가 크론에 있다" \
  || bad "크론에 pair_detect_daily.sh 가 없다 — 아무것도 안 뜬다"
# 옛 경로는 09-11 이후 전부 죽는다. 크론에 남아 있으면 그 줄은 조용히 실패한다.
if printf '%s\n' "$CRON" | grep -q 'Documents/git'; then
  bad "크론에 옛 경로(~/Documents/git)가 남아 있다:"
  printf '%s\n' "$CRON" | grep -n 'Documents/git' | sed 's/^/       /' | cut -c1-120
else
  ok "크론에 옛 경로 없음"
fi

# ─────────────────────────────────────────────────────────── 요약
printf '\n──────────────────────────────────────────\n'
if [ "$FAIL" -gt 0 ]; then
  printf 'FAIL %d · WARN %d — **개장 전에 FAIL 을 먼저 고칠 것**\n' "$FAIL" "$WARN"
  exit 1
fi
printf 'FAIL 0 · WARN %d — 런처 전제는 충족됐다\n' "$WARN"
