#!/usr/bin/env bash
# CPU 100% 에서도 ssh 가 밀리지 않게 — 커널 cgroup(systemd) 수준 보호. root 로 한 번만 실행한다.
#   sudo bash deploy/protect_ssh.sh
# 1) sshd 에 CPU·IO 가중치와 음수 nice 를 준다(다른 프로세스보다 먼저 스케줄링).
# 2) 연구·학습 전용 slice(research.slice)를 만들어 코어 하나를 남기고 가중치를 최하로 둔다.
# 3) 매매 감지기(scalp-it)는 손대지 않는다 — 장중 지연이 곧 손실이라 우선순위를 낮추면 안 된다.
set -euo pipefail
CORES=$(nproc)
QUOTA=$(( (CORES - 1) * 100 ))          # 코어 1개는 항상 비워 둔다

install -d /etc/systemd/system/ssh.service.d
cat > /etc/systemd/system/ssh.service.d/10-priority.conf <<CONF
[Service]
CPUWeight=10000
IOWeight=1000
Nice=-10
OOMScoreAdjust=-900
CONF

install -d /etc/systemd/system/user@.service.d
cat > /etc/systemd/system/user@.service.d/10-login-priority.conf <<CONF
[Service]
CPUWeight=5000
IOWeight=500
CONF

cat > /etc/systemd/system/research.slice <<CONF
[Unit]
Description=연구·학습(백테스트·RL) 전용 — ssh·매매보다 항상 뒤로
Before=slices.target

[Slice]
CPUWeight=10
CPUQuota=${QUOTA}%
IOWeight=10
MemoryHigh=70%
CONF

# polkit: 이 사용자가 research.slice 안에서 systemd-run 을 쓸 수 있게(비번 없이)
install -d /etc/polkit-1/rules.d
cat > /etc/polkit-1/rules.d/50-research-slice.rules <<CONF
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.systemd1.manage-units" &&
        subject.user == "young" &&
        action.lookup("unit") && action.lookup("unit").indexOf("research-") === 0) {
        return polkit.Result.YES;
    }
});
CONF

systemctl daemon-reload
systemctl restart ssh
systemctl start research.slice 2>/dev/null || true
echo "설치 완료 — 코어 ${CORES}개 중 연구용 상한 ${QUOTA}%"
systemctl show ssh -p CPUWeight -p Nice | sed 's/^/  ssh: /'
systemctl show research.slice -p CPUQuotaPerSecUSec -p CPUWeight | sed 's/^/  research.slice: /'
