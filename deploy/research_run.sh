#!/usr/bin/env bash
# 무거운 연구·학습은 항상 이걸로 실행한다 — ssh·매매가 CPU 를 못 잡는 일이 없게.
#   deploy/research_run.sh python train.py ...
# 1순위 보호는 research.slice(CPUQuota: 코어-1, CPUWeight 10) — deploy/protect_ssh.sh 가 만든다.
# slice 를 못 쓰는 환경이면 nice+ionice 로 떨어진다(그래도 ssh 가 앞선다).
set -euo pipefail
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
CORES=$(nproc); QUOTA=$(( (CORES - 1) * 100 ))
# 소켓 파일이 있어도 user bus 가 연결을 거부할 수 있다("Failed to connect to user scope bus").
# 그때 바로 exec systemd-run 하면 명령이 아예 안 돌고 끝난다 — data/nightly.cron.log 에 그 오류만
# 남고 그날 kqc nightly 가 돌지 않았다. 빈 scope 를 먼저 띄워 보고, 되는 경우에만 slice 로 간다.
if [ -S "$XDG_RUNTIME_DIR/bus" ] && command -v systemd-run >/dev/null \
   && systemd-run --user --scope --quiet -- true >/dev/null 2>&1; then
  exec systemd-run --user --scope --quiet \
    -p CPUWeight=10 -p IOWeight=10 -p CPUQuota=${QUOTA}% \
    --nice=15 -- "$@"
fi
echo "[research_run] systemd scope 를 못 써서 nice+ionice 로 돈다(research.slice 보호 없음)" >&2
exec nice -n 15 ionice -c3 "$@"
