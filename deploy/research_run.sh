#!/usr/bin/env bash
# 무거운 연구·학습은 항상 이걸로 실행한다 — ssh·매매가 CPU 를 못 잡는 일이 없게.
#   deploy/research_run.sh python train.py ...
# 1순위 보호는 research.slice(CPUQuota: 코어-1, CPUWeight 10) — deploy/protect_ssh.sh 가 만든다.
# slice 를 못 쓰는 환경이면 nice+ionice 로 떨어진다(그래도 ssh 가 앞선다).
set -euo pipefail
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
CORES=$(nproc); QUOTA=$(( (CORES - 1) * 100 ))
if [ -S "$XDG_RUNTIME_DIR/bus" ] && command -v systemd-run >/dev/null; then
  exec systemd-run --user --scope --quiet \
    -p CPUWeight=10 -p IOWeight=10 -p CPUQuota=${QUOTA}% \
    --nice=15 -- "$@"
fi
exec nice -n 15 ionice -c3 "$@"
